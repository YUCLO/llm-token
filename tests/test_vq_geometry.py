from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from experiments.vq_geometry import (
    exact_boundary_geometry,
    nearest_code,
    spherical_displacement,
)


class VQGeometryTest(unittest.TestCase):
    def test_one_dimensional_boundary_example(self) -> None:
        # Unit-circle analogue: codes at 0 and 90 degrees, boundary at 45.
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        theta = torch.tensor(0.2)
        u = torch.stack([torch.cos(theta), torch.sin(theta)])[None, :]
        geom = exact_boundary_geometry(u, q)
        expected_angle = torch.tensor(torch.pi / 4) - theta
        self.assertEqual(geom.token_ids.item(), 0)
        self.assertTrue(torch.allclose(geom.angular_radius[0], expected_angle, atol=1e-6))

    def test_certificate_keeps_random_points_in_same_cell(self) -> None:
        generator = torch.Generator().manual_seed(7)
        q = F.normalize(torch.randn(64, 8, generator=generator), dim=-1)
        u = F.normalize(torch.randn(128, 8, generator=generator), dim=-1)
        geom = exact_boundary_geometry(u, q)

        tangent = torch.randn(u.shape, generator=generator)
        tangent = tangent - (tangent * u).sum(dim=-1, keepdim=True) * u
        tangent = F.normalize(tangent, dim=-1)
        step = 0.5 * geom.angular_radius
        moved = torch.cos(step)[:, None] * u + torch.sin(step)[:, None] * tangent

        self.assertTrue(torch.equal(nearest_code(moved, q), geom.token_ids))
        angle, chord = spherical_displacement(u, moved)
        self.assertTrue(torch.all(angle < geom.angular_radius + 1e-6))
        self.assertTrue(torch.all(chord < geom.chord_radius + 1e-6))


if __name__ == "__main__":
    unittest.main()
