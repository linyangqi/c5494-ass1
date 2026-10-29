"""Small mathematical checks, without training or dataset downloads.

Run: python -m unittest test_autoregressive -v
"""
import itertools
import unittest

import torch

from autoregressive import FVSBN, MADE, AutoregressiveModel, check_causality


class AutoregressiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(14)

    def models(self):
        return [FVSBN(5), MADE(5, (11,)), MADE(5, (11, 9)), MADE(5, (11, 9), residual=True)]

    def test_joint_distribution_normalizes_exactly(self):
        # Enumerate every possible binary image: catches invalid factorization.
        x = torch.tensor(list(itertools.product([0., 1.], repeat=5)))
        for m in self.models():
            with self.subTest(model=type(m).__name__):
                self.assertAlmostEqual(m.nll(x).neg().exp().sum().item(), 1.0, places=5)

    def test_all_target_and_future_jacobian_entries_are_zero(self):
        for m in self.models():
            x = torch.rand(5, requires_grad=True)
            jac = torch.autograd.functional.jacobian(lambda z: m(z[None])[0], x)
            self.assertEqual(torch.triu(jac).abs().max().item(), 0.0)
            check_causality(m)

    def test_vectorized_logits_match_prefix_conditioning(self):
        x = torch.rand(4, 5)
        for m in self.models():
            parallel = m(x)
            for j in range(5):
                prefix = x.clone(); prefix[:, j:] = 0
                torch.testing.assert_close(parallel[:, j], m(prefix)[:, j])

    def test_fast_fvsbn_sampler_matches_generic_ancestral_sampler(self):
        m = FVSBN(5)
        with torch.no_grad():
            m.linear.weight.normal_()
        torch.testing.assert_close(m.sample(100, 13), AutoregressiveModel.sample(m, 100, 13))

    def test_samples_binary_reproducible_and_do_not_change_global_rng(self):
        for m in self.models():
            before = torch.random.get_rng_state().clone()
            a, b = m.sample(20, 17), m.sample(20, 17)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(((a == 0) | (a == 1)).all())


if __name__ == "__main__":
    unittest.main()
