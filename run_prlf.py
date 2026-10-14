"""Train/evaluate PRLF on local MOSI/MOSEI data; python run_prlf.py --smoke."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from prlf.config import get_config
from prlf.data import MMDataset
from prlf.model import PRLF, prlf_loss
from prlf.utils import setup_seed


def corrupt_batch(batch, rate, seed, device, available='lav'):
    """Zero frame-level features, with an epoch-specific RNG and no label access."""
    generator = torch.Generator().manual_seed(seed)
    values = [batch[key].to(device).clone() for key in ('text', 'audio', 'vision')]
    masks = []
    for m, value in enumerate(values):
        valid = value[:, 1].bool() if m == 0 else value.abs().sum(-1) != 0
        keep = torch.rand(valid.shape, generator=generator).to(device) >= rate
        if 'lav'[m] not in available:
            keep.zero_()
        mask = valid & keep
        if m == 0:
            value[:, 0] *= mask
            value[:, 1] = mask.to(value.dtype)
            value[:, 2] *= mask
        else:
            value *= mask.unsqueeze(-1)
        masks.append(mask)
    return values, masks


def run_epoch(model, loader, args, epoch=0, optimizer=None, available='lav'):
    training = optimizer is not None
    model.train(training)
    if training:
        model.start_epoch(len(loader.dataset))
        optimizer.zero_grad(set_to_none=True)
    total, count, predictions, targets = 0., 0, [], []
    parts = {'task': 0., 'uni': 0., 'phase': 0.}
    for step, batch in enumerate(loader):
        values, masks = corrupt_batch(batch, args.missing_rate,
                                     args.seed + (epoch if training else 100000) * 10000 + step,
                                     args.device, available)
        labels = (batch['labels']['M'].view(-1).to(args.device) >= 0).long()
        indices = batch['index'].to(args.device)
        with torch.set_grad_enabled(training):
            output = model(*values, labels=labels if training else None,
                           indices=indices if training else None, masks=masks)
            loss, components = prlf_loss(output, labels, args.eta1, args.eta2)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {step}')
            if training:
                # Average each accumulation window, including the final partial one.
                window_start = (step // args.accumulate) * args.accumulate
                window_size = min(args.accumulate, len(loader) - window_start)
                (loss / window_size).backward()
                if (step + 1) % args.accumulate == 0 or step + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        n = labels.numel()
        total += loss.item() * n
        count += n
        for key in parts:
            parts[key] += components[key].item() * n
        predictions.extend(output['logits'].argmax(-1).detach().cpu().tolist())
        targets.extend(labels.cpu().tolist())
    from sklearn.metrics import accuracy_score, f1_score
    return {'loss': total / count, 'accuracy': accuracy_score(targets, predictions),
            'f1': f1_score(targets, predictions, average='weighted', zero_division=0),
            **{k: v / count for k, v in parts.items()}, 'samples': count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    parser.add_argument('--data-path', type=Path, help='Path to the aligned feature pickle')
    parser.add_argument('--pretrained', default=str(ROOT / 'bert-base-uncased'), help='Local BERT directory or Hugging Face model ID')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--missing-rate', type=float, default=0.3)
    parser.add_argument('--available', choices=['l', 'a', 'v', 'la', 'lv', 'av', 'lav'], default='lav')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--freeze-bert', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Two tiny epochs on real data, then checkpoint reload and evaluation')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'pt' / 'prlf.pth')
    parser.add_argument('--eta1', type=float, default=0.5)
    parser.add_argument('--eta2', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=1)
    cli = parser.parse_args()
    if not 0 <= cli.missing_rate <= 1 or cli.accumulate < 1 or cli.epochs < 1:
        parser.error('missing-rate must be in [0,1]; accumulate and epochs must be positive')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    setup_seed(cli.seed)
    args = get_config(cli.dataset, ROOT / 'configs' / 'default.json')
    args.update(vars(cli))
    args.device = torch.device(cli.device)
    args.featurePath = str(cli.data_path or ROOT / 'dataset' / cli.dataset.upper() / 'Processed' / 'aligned_50.pkl')
    if not Path(args.featurePath).is_file():
        parser.error(f'Dataset not found: {args.featurePath}; provide --data-path')
    args.pretrained = cli.pretrained
    args.feature_T = args.feature_A = args.feature_V = ''
    args.use_finetune = not cli.freeze_bert
    if cli.smoke:
        args.epochs, args.batch_size, args.use_finetune = 2, 2, False
        args.checkpoint = ROOT / 'pt' / 'prlf_smoke.pth'
    datasets = {split: MMDataset(args, split) for split in ('train', 'valid', 'test')}
    # Read dimensions from the actual features rather than dataset-name assumptions.
    args.feature_dims = [768, datasets['train'].audio.shape[-1], datasets['train'].vision.shape[-1]]
    loaders = {split: DataLoader(Subset(ds, range(4)) if cli.smoke else ds,
                                batch_size=args.batch_size, shuffle=split == 'train', num_workers=0)
               for split, ds in datasets.items()}
    checkpoint = None
    if cli.mode == 'test':
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        args.update(checkpoint.get('settings', {}))
    model = PRLF(args).to(args.device)
    if cli.mode == 'test':
        model.load_state_dict(checkpoint['model'])
        print(json.dumps(run_epoch(model, loaders['test'], args, available=cli.available)))
        return
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, stale = float('inf'), 0
    history = []
    for epoch in range(args.epochs):
        train = run_epoch(model, loaders['train'], args, epoch, optimizer)
        valid = run_epoch(model, loaders['valid'], args)
        row = {'epoch': epoch + 1, 'train': train, 'valid': valid}
        history.append(row)
        print(json.dumps(row), flush=True)
        if valid['loss'] < best:
            best, stale = valid['loss'], 0
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1,
                        'settings': {k: args[k] for k in ('gamma', 'steps', 'eta1', 'eta2', 'feature_dims')}}, args.checkpoint)
        else:
            stale += 1
        if stale >= args.early_stop:
            break
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    test = run_epoch(model, loaders['test'], args, available=cli.available)
    print(json.dumps({'checkpoint': str(args.checkpoint), 'test': test}), flush=True)
    report = {'history': history, 'test': test, 'smoke': cli.smoke}
    args.checkpoint.with_suffix('.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
