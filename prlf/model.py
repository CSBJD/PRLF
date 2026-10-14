"""PRLF: AMRE (Eqs. 1–12), ProgInteract (13–22), classification (23–25)."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .text_encoder import BertTextEncoder


class PRLF(nn.Module):
    modality_order = ('language', 'audio', 'vision')

    def __init__(self, args):
        super().__init__()
        self.use_bert = args.use_bert
        self.text_model = BertTextEncoder(args.use_finetune, args.transformers, args.pretrained) if self.use_bert else None
        if self.text_model is not None and not args.use_finetune:
            self.text_model.requires_grad_(False)
        dim, heads = args.dst_feature_dim_nheads
        self.dim = dim
        self.tokens = args.get('interaction_tokens', 8)
        self.steps = args.get('steps', 4)
        self.gamma = args.get('gamma', 0.8)
        if self.steps < 1:
            raise ValueError('steps must be positive')
        dropout = args.output_dropout
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(size, dim), nn.ReLU(),
                          nn.TransformerEncoder(nn.TransformerEncoderLayer(
                              dim, heads, dim * 2, dropout=dropout, batch_first=True),
                              num_layers=args.get('encoder_layers', 1)))
            for size in args.feature_dims
        ])
        self.unimodal_heads = nn.ModuleList([nn.Linear(dim, 2) for _ in range(3)])
        self.refiners = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                                    nn.Linear(dim, dim), nn.Dropout(dropout)) for _ in range(3)])
        self.cross_projections = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(3)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(),
                                                nn.Linear(dim, dim), nn.Sigmoid()) for _ in range(3)])
        self.denoisers = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(3)])
        self.classifier = nn.Linear(3 * dim, 2)
        self.register_buffer('previous_fisher', torch.empty(0, 3))
        self.register_buffer('current_fisher', torch.empty(0, 3))
        self.register_buffer('reference_fisher', torch.zeros(3))
        self.register_buffer('history_ready', torch.tensor(False))

    def start_epoch(self, sample_count):
        device = self.reference_fisher.device
        if self.current_fisher.shape == (sample_count, 3):
            self.previous_fisher = self.current_fisher.clone()
            self.history_ready.fill_(True)
        else:
            self.previous_fisher = torch.zeros(sample_count, 3, device=device)
            self.history_ready.fill_(False)
        self.current_fisher = torch.zeros(sample_count, 3, device=device)

    def _fisher(self, logits, modality):
        """Exact categorical expectation of per-sample parameter-gradient energy (3–4).

        Includes the dedicated encoder and classification head, and trainable BERT
        parameters for language. No second derivatives or optimizer-gradient writes.
        """
        parameters = list(self.encoders[modality].parameters()) + list(self.unimodal_heads[modality].parameters())
        if modality == 0 and self.text_model is not None:
            parameters += list(self.text_model.parameters())
        parameters = [p for p in parameters if p.requires_grad]
        log_prob = logits.log_softmax(-1)
        prob = log_prob.detach().exp()
        traces = []
        for i in range(logits.shape[0]):
            energy = logits.new_zeros(())
            for c in range(logits.shape[-1]):
                gradients = torch.autograd.grad(log_prob[i, c], parameters, retain_graph=True, allow_unused=True)
                energy = energy + prob[i, c] * sum(g.detach().square().sum() for g in gradients if g is not None)
            traces.append(energy)
        return torch.stack(traces)

    @staticmethod
    def normalize(values):
        total = values.sum(-1, keepdim=True)
        return torch.where(total > 1e-12, values / total.clamp_min(1e-12), torch.full_like(values, 1 / 3))

    def forward(self, text, audio, vision, labels=None, indices=None, masks=None):
        # Fisher is needed at inference too, even when the caller uses no_grad.
        outer_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            if self.use_bert:
                text_mask = text[:, 1].bool()
                text = self.text_model(text)
            else:
                text_mask = text.abs().sum(-1) != 0
            inputs = [text, audio, vision]
            valid = [text_mask, audio.abs().sum(-1) != 0, vision.abs().sum(-1) != 0]
            if masks is not None:
                valid = [v & mask.bool() for v, mask in zip(valid, masks)]
            features = []
            for encoder, value, mask in zip(self.encoders, inputs, valid):
                safe_mask = mask.clone()
                safe_mask[:, 0] |= ~mask.any(-1)
                value = value * mask.unsqueeze(-1)
                projected = encoder[1](encoder[0](value))
                encoded = encoder[2](projected, src_key_padding_mask=~safe_mask)
                encoded = encoded * mask.unsqueeze(-1)
                pooled = F.adaptive_avg_pool1d(encoded.transpose(1, 2), self.tokens).transpose(1, 2)
                features.append(pooled)
            unimodal = torch.stack([head(f.mean(1)) for head, f in zip(self.unimodal_heads, features)], 1)
            probabilities = unimodal.softmax(-1)
            confidence = probabilities.max(-1).values if labels is None else probabilities.gather(
                -1, labels[:, None, None].expand(-1, 3, 1)).squeeze(-1)
            fisher = torch.stack([self._fisher(unimodal[:, m], m) for m in range(3)], 1)
            available = torch.stack([v.any(-1) for v in valid], 1)
            confidence = confidence * available
            fisher = fisher * available
            if self.training and indices is not None:
                previous = self.previous_fisher[indices]
                ready = bool(self.history_ready)
                self.current_fisher[indices] = fisher.detach()
                self.reference_fisher.copy_(self.current_fisher.mean(0))
            else:
                previous = self.reference_fisher.expand_as(fisher)
                ready = bool(self.history_ready)
            growth = (fisher - previous) / fisher.clamp_min(1e-12)
            weight = growth.sigmoid() if ready else torch.zeros_like(fisher)
            importance = self.normalize(((1 - weight) * self.normalize(confidence.detach()) +
                                         weight * self.normalize(fisher)) * available)
            order = importance.argsort(dim=-1, descending=True, stable=True)
            batch_index = torch.arange(text.shape[0], device=text.device)
            phase_losses = []
            for t in range(self.steps):
                lam = 1 - t / max(1, self.steps - 1)
                fused = []
                for m in range(3):
                    self_feature = features[m] + self.refiners[m](features[m])
                    query = importance[:, m, None, None] * features[m]
                    interactions = []
                    for n in range(3):
                        if n != m:
                            key = importance[:, n, None, None] * features[n]
                            attention = (query @ key.transpose(-1, -2) / math.sqrt(self.dim)).softmax(-1)
                            interactions.append(attention @ query)
                    cross = torch.cat(interactions, -1)
                    cross = (cross @ cross.transpose(-1, -2) / math.sqrt(self.dim)).softmax(-1) @ cross
                    cross = self.cross_projections[m](cross)
                    fused.append(lam * self_feature + (1 - lam) * cross)
                stacked = torch.stack(fused, 1)
                dominant = stacked[batch_index, order[:, 0]]
                updated = []
                step_phase = stacked.new_zeros(())
                for m in range(3):
                    auxiliary = fused[m]
                    projection = self.gates[m](torch.cat([dominant, auxiliary], -1)) * dominant
                    residual = auxiliary - projection
                    is_auxiliary = order[:, 0] != m
                    dot = (projection * residual).sum(-1).square().mean(-1)
                    step_phase = step_phase + (dot * is_auxiliary).mean()
                    refined = projection + self.gamma * (residual - self.denoisers[m](residual))
                    updated.append(torch.where(is_auxiliary[:, None, None], refined, dominant))
                features = updated
                phase_losses.append(step_phase)
            stacked = torch.stack([f.mean(1) for f in features], 1)
            ordered = stacked.gather(1, order[..., None].expand(-1, -1, self.dim))
            fusion = ordered.flatten(1)
            result = {'logits': self.classifier(fusion), 'unimodal_logits': unimodal,
                      'phase_loss': torch.stack(phase_losses).sum(), 'importance': importance,
                      'fisher': fisher, 'dominant': order[:, 0], 'features': stacked,
                      'fusion': fusion}
        return result if outer_grad else {k: v.detach() for k, v in result.items()}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for key in ('previous_fisher', 'current_fisher'):
            if key in state_dict:
                setattr(self, key, torch.empty_like(state_dict[key], device=self.reference_fisher.device))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def prlf_loss(output, labels, eta1=0.5, eta2=0.1):
    task = F.cross_entropy(output['logits'], labels)
    uni = sum(F.cross_entropy(output['unimodal_logits'][:, m], labels) for m in range(3))
    phase = output['phase_loss']
    return task + eta1 * uni + eta2 * phase, {'task': task.detach(), 'uni': uni.detach(), 'phase': phase.detach()}
