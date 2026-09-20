"""Geometry helpers for normalized vector quantizers.

The AUV quantizer selects a code after L2-normalizing both the projected
encoder representation and the codebook.  Its cells are therefore spherical
Voronoi cells.  This module computes the exact distance from a point on the
unit sphere to the closest pairwise decision boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class BoundaryGeometry:
    """Per-frame geometry relative to the selected VQ code."""

    token_ids: torch.Tensor
    competitor_ids: torch.Tensor
    signed_euclidean_margin: torch.Tensor
    angular_radius: torch.Tensor
    chord_radius: torch.Tensor
    selected_distance: torch.Tensor


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize the final dimension in float32 for stable diagnostics."""

    return F.normalize(x.float(), dim=-1)


def nearest_code(
    u: torch.Tensor,
    q: torch.Tensor,
    *,
    assume_normalized: bool = False,
) -> torch.Tensor:
    """Return nearest code IDs using AUV's complete squared-distance formula."""

    if not assume_normalized:
        u = normalize_rows(u)
        q = normalize_rows(q)
    else:
        u = u.float()
        q = q.float()
    dist = (
        u.pow(2).sum(1, keepdim=True)
        - 2.0 * (u @ q.transpose(0, 1))
        + q.pow(2).sum(1, keepdim=True).transpose(0, 1)
    )
    return torch.argmin(dist, dim=-1)


def exact_boundary_geometry(
    u: torch.Tensor,
    q: torch.Tensor,
    token_ids: torch.Tensor | None = None,
    *,
    frame_chunk: int = 512,
    duplicate_eps: float = 1e-7,
    assume_normalized: bool = False,
) -> BoundaryGeometry:
    """Compute exact nearest-boundary geometry over the complete codebook.

    Args:
        u: Frame representations with shape ``(T, D)``.
        q: Codebook with shape ``(V, D)``.
        token_ids: Optional deployed VQ IDs. If omitted, nearest IDs are used.
        frame_chunk: Number of frames processed per matrix multiply.
        duplicate_eps: Separation below which two different codewords count as
            duplicates and imply zero certified radius.
        assume_normalized: If true, use the supplied float32 normalized values
            without applying a second normalization operation.

    For selected code ``k`` and competitor ``j``, the signed ambient margin is

        (d_j(u) - d_k(u)) / (2 ||q_j - q_k||).

    With exact unit codewords the pairwise boundary is a great hyperplane and
    angle is ``asin(margin)``.  The implementation also handles the tiny affine
    offset caused by finite-precision normalization.
    """

    if u.ndim != 2 or q.ndim != 2:
        raise ValueError(f"Expected u=(T,D), q=(V,D); got {u.shape}, {q.shape}")
    if u.shape[1] != q.shape[1]:
        raise ValueError(f"Dimension mismatch: u={u.shape}, q={q.shape}")
    if frame_chunk <= 0:
        raise ValueError("frame_chunk must be positive")

    if not assume_normalized:
        u = normalize_rows(u)
        q = normalize_rows(q).to(u.device)
    else:
        u = u.float()
        q = q.to(device=u.device, dtype=torch.float32)
    if token_ids is None:
        token_ids = nearest_code(u, q, assume_normalized=True)
    token_ids = token_ids.to(device=u.device, dtype=torch.long).reshape(-1)
    if token_ids.numel() != u.shape[0]:
        raise ValueError("token_ids must contain one ID per frame")

    all_competitors: list[torch.Tensor] = []
    all_signed: list[torch.Tensor] = []
    all_angular: list[torch.Tensor] = []
    all_selected_dist: list[torch.Tensor] = []
    vocab = q.shape[0]

    for start in range(0, u.shape[0], frame_chunk):
        stop = min(start + frame_chunk, u.shape[0])
        uc = u[start:stop]
        kc = token_ids[start:stop]
        qk = q[kc]

        dot_uq = uc @ q.transpose(0, 1)
        u_norm_sq = uc.pow(2).sum(1, keepdim=True)
        q_norm_sq = q.pow(2).sum(1, keepdim=True).transpose(0, 1)
        # Match AUV's deployed float32 squared-distance operations exactly.
        dist = u_norm_sq - 2.0 * dot_uq + q_norm_sq
        selected_dist = dist.gather(1, kc[:, None]).squeeze(1)

        qk_norm_sq = qk.pow(2).sum(1, keepdim=True)
        code_sep_sq = (
            qk_norm_sq
            - 2.0 * (qk @ q.transpose(0, 1))
            + q_norm_sq
        ).clamp_min(0.0)
        code_sep = torch.sqrt(code_sep_sq)
        denominator = 2.0 * code_sep.clamp_min(duplicate_eps)
        signed = (dist - selected_dist[:, None]) / denominator

        # With exact real-valued normalization the pairwise boundary passes
        # through the origin and angle=asin(signed). In float32 the code norms
        # differ slightly, producing an affine plane x.n=b. Its exact closest
        # spherical boundary distance is asin(a)-asin(b), where a=u.n.
        dot_uqk = dot_uq.gather(1, kc[:, None])
        sep_safe = code_sep.clamp_min(duplicate_eps)
        a = (dot_uqk - dot_uq) / sep_safe
        b = (qk_norm_sq - q_norm_sq) / (2.0 * sep_safe)
        angular = torch.asin(a.clamp(-1.0, 1.0)) - torch.asin(b.clamp(-1.0, 1.0))

        row_ids = torch.arange(stop - start, device=u.device)
        signed[row_ids, kc] = torch.inf
        angular[row_ids, kc] = torch.inf

        # A duplicate non-selected code makes the hard decision tie-sensitive.
        duplicate = code_sep < duplicate_eps
        duplicate[row_ids, kc] = False
        signed = torch.where(duplicate, torch.zeros_like(signed), signed)
        angular = torch.where(duplicate, torch.zeros_like(angular), angular)

        angular_min, competitor = torch.min(angular, dim=1)
        signed_min = signed.gather(1, competitor[:, None]).squeeze(1)
        if torch.any((competitor < 0) | (competitor >= vocab)):
            raise RuntimeError("Invalid nearest-boundary competitor ID")

        all_competitors.append(competitor)
        all_signed.append(signed_min)
        all_angular.append(angular_min)
        all_selected_dist.append(selected_dist)

    competitor_ids = torch.cat(all_competitors)
    signed_margin = torch.cat(all_signed)
    angular_signed = torch.cat(all_angular)
    selected_distance = torch.cat(all_selected_dist)

    # For a deployed nearest-neighbour decision this should be non-negative;
    # tiny negatives can occur from float32 roundoff.
    angular_radius = angular_signed.clamp(min=0.0, max=torch.pi)
    chord_radius = 2.0 * torch.sin(angular_radius / 2.0)

    return BoundaryGeometry(
        token_ids=token_ids,
        competitor_ids=competitor_ids,
        signed_euclidean_margin=signed_margin,
        angular_radius=angular_radius,
        chord_radius=chord_radius,
        selected_distance=selected_distance,
    )


def spherical_displacement(
    u_clean: torch.Tensor,
    u_view: torch.Tensor,
    *,
    assume_normalized: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return angular and chord displacement between paired unit vectors."""

    if u_clean.shape != u_view.shape:
        raise ValueError(f"Paired shapes differ: {u_clean.shape} vs {u_view.shape}")
    if not assume_normalized:
        u_clean = normalize_rows(u_clean)
        u_view = normalize_rows(u_view)
    else:
        u_clean = u_clean.float()
        u_view = u_view.float()
    cosine = (u_clean * u_view).sum(dim=-1).clamp(-1.0, 1.0)
    angular = torch.acos(cosine)
    chord = torch.linalg.vector_norm(u_clean - u_view, dim=-1)
    return angular, chord
