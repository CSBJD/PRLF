import unittest
import torch
from prlf.model import PRLF, prlf_loss


class Args(dict):
    __getattr__ = dict.__getitem__


class PRLFTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        self.model = PRLF(Args(use_bert=False, feature_dims=[5, 3, 4],
                               dst_feature_dim_nheads=[8, 2], output_dropout=0.,
                               steps=4, interaction_tokens=4))
        self.values = [torch.randn(2, 6, size) for size in (5, 3, 4)]
        self.labels = torch.tensor([0, 1])

    def test_fisher_matches_parameter_gradient_expectation(self):
        features = self.model.encoders[1](self.values[1]).mean(1)
        logits = self.model.unimodal_heads[1](features)
        trace = self.model._fisher(logits, 1)
        params = list(self.model.encoders[1].parameters()) + list(self.model.unimodal_heads[1].parameters())
        expected = torch.zeros(2)
        for i in range(2):
            for c in range(2):
                gradients = torch.autograd.grad(logits.log_softmax(-1)[i, c], params, retain_graph=True)
                expected[i] += logits.softmax(-1)[i, c].detach() * sum(g.square().sum().detach() for g in gradients)
        torch.testing.assert_close(trace, expected)
        self.assertTrue(all(p.grad is None for p in params))

    def test_loss_and_gradients_reach_all_modules(self):
        self.model.start_epoch(2)
        out = self.model(*self.values, labels=self.labels, indices=torch.arange(2))
        loss, parts = prlf_loss(out, self.labels)
        torch.testing.assert_close(loss.detach(), parts['task'] + .5 * parts['uni'] + .1 * parts['phase'])
        loss.backward()
        for module in (self.model.encoders, self.model.unimodal_heads, self.model.gates,
                       self.model.denoisers, self.model.cross_projections, self.model.classifier):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        history = self.model.current_fisher.clone()
        self.model.start_epoch(2)
        torch.testing.assert_close(self.model.previous_fisher, history)
        self.assertTrue(self.model.history_ready)

    def test_missing_modalities_no_grad_and_batch_independence(self):
        self.model.eval()
        values = [v.clone() for v in self.values]
        values[0].zero_()
        values[2].zero_()
        with torch.no_grad():
            out = self.model(*values)
            single = self.model(*[v[:1] for v in values])
        self.assertFalse(out['logits'].requires_grad)
        self.assertTrue(torch.equal(out['dominant'], torch.ones(2, dtype=torch.long)))
        torch.testing.assert_close(out['importance'], torch.tensor([[0., 1., 0.], [0., 1., 0.]]))
        torch.testing.assert_close(out['logits'][:1], single['logits'], atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            empty = self.model(*[torch.zeros_like(v) for v in values])
        self.assertTrue(torch.isfinite(empty['logits']).all())
        torch.testing.assert_close(empty['importance'].sum(-1), torch.ones(2))

    def test_history_checkpoint_reload(self):
        self.model.start_epoch(2)
        self.model(*self.values, labels=self.labels, indices=torch.arange(2))
        state = self.model.state_dict()
        self.model.previous_fisher = torch.empty(0, 3)
        self.model.current_fisher = torch.empty(0, 3)
        self.model.load_state_dict(state)
        self.assertEqual(self.model.current_fisher.shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
