from __future__ import annotations

import unittest

import torch

from experiments.qbc_loss import qbc_hinge_loss, signed_boundary_margin


class QBCLossTest(unittest.TestCase):
    def test_signed_margin_changes_sign_at_boundary(self) -> None:
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        inside_angle = torch.tensor(0.2)
        outside_angle = torch.tensor(1.0)
        view = torch.stack(
            [
                torch.stack([torch.cos(inside_angle), torch.sin(inside_angle)]),
                torch.stack([torch.cos(outside_angle), torch.sin(outside_angle)]),
            ]
        )
        result = signed_boundary_margin(view, q, torch.tensor([0, 0]))
        self.assertGreater(result.margin[0].item(), 0.0)
        self.assertLess(result.margin[1].item(), 0.0)
        expected = torch.sin(torch.tensor(torch.pi / 4) - inside_angle)
        self.assertTrue(torch.allclose(result.margin[0], expected, atol=1e-6))

    def test_qbc_detaches_codebook_but_updates_view(self) -> None:
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        view = torch.tensor([[0.72, 0.69]], requires_grad=True)
        loss, _, _ = qbc_hinge_loss(
            view,
            q,
            torch.tensor([0]),
            target_margin=0.1,
        )
        loss.backward()
        self.assertIsNotNone(view.grad)
        self.assertGreater(view.grad.abs().sum().item(), 0.0)
        self.assertIsNone(q.grad)


if __name__ == "__main__":
    unittest.main()
