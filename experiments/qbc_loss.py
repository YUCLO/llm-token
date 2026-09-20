"""Differentiable quantization-boundary consistency objectives.

The deployed AUV quantizer L2-normalizes both projected encoder vectors and
codewords.  Given a teacher code ``k``, the signed ambient distance from a
view vector ``u`` to the closest ``k``/competitor Voronoi boundary is

    min_j (d_j(u) - d_k(u)) / (2 ||q_j - q_k||).

It is positive inside the teacher cell and negative after a token flip.  The
functions below evaluate all codewords, rather than assuming the second
nearest code is also the closest boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SignedMarginResult:
    margin: torch.Tensor
    competitor_ids: torch.Tensor


def signed_boundary_margin(
    view: torch.Tensor,
    codebook: torch.Tensor,
    teacher_ids: torch.Tensor,
    *,
    frame_chunk: int = 256,
    detach_codebook: bool = True,
    duplicate_eps: float = 1e-7,
) -> SignedMarginResult:
    """Return the exact all-code signed boundary margin for each frame.

    Args:
        view: projected pre-VQ representations, shape ``(T, D)``.
        codebook: deployed VQ codebook, shape ``(V, D)``.
        teacher_ids: one full-context teacher code per frame, shape ``(T,)``.
        frame_chunk: bounds the temporary ``T x V`` matrices.
        detach_codebook: prevents this objective from moving VQ boundaries.
    """

    if view.ndim != 2 or codebook.ndim != 2:
        raise ValueError(f"Expected view=(T,D), codebook=(V,D), got {view.shape}, {codebook.shape}")
    if view.shape[1] != codebook.shape[1]:
        raise ValueError("View/codebook dimension mismatch")
    teacher_ids = teacher_ids.reshape(-1).to(device=view.device, dtype=torch.long)
    if teacher_ids.numel() != view.shape[0]:
        raise ValueError("teacher_ids must contain one ID per frame")
    if frame_chunk <= 0:
        raise ValueError("frame_chunk must be positive")

    u = F.normalize(view.float(), dim=-1)
    q_source = codebook.detach() if detach_codebook else codebook
    q = F.normalize(q_source.float(), dim=-1).to(view.device)
    margins: list[torch.Tensor] = []
    competitors: list[torch.Tensor] = []

    for start in range(0, u.shape[0], frame_chunk):
        stop = min(start + frame_chunk, u.shape[0])
        uc = u[start:stop]
        kc = teacher_ids[start:stop]
        qk = q[kc]

        # Keep the complete squared-distance expression used by AUV.
        q_norm_sq = q.square().sum(dim=1)[None, :]
        dist = uc.square().sum(dim=1, keepdim=True) - 2.0 * (uc @ q.T) + q_norm_sq
        selected_dist = dist.gather(1, kc[:, None])

        sep_sq = (
            qk.square().sum(dim=1, keepdim=True)
            - 2.0 * (qk @ q.T)
            + q_norm_sq
        ).clamp_min(0.0)
        sep = torch.sqrt(sep_sq)
        signed = (dist - selected_dist) / (2.0 * sep.clamp_min(duplicate_eps))

        row = torch.arange(stop - start, device=view.device)
        signed[row, kc] = torch.inf
        duplicate = sep < duplicate_eps
        duplicate[row, kc] = False
        signed = torch.where(duplicate, torch.zeros_like(signed), signed)

        margin, competitor = torch.min(signed, dim=1)
        margins.append(margin)
        competitors.append(competitor)

    return SignedMarginResult(torch.cat(margins), torch.cat(competitors))


def qbc_hinge_loss(
    view: torch.Tensor,
    codebook: torch.Tensor,
    teacher_ids: torch.Tensor,
    *,
    target_margin: float,
    frame_mask: torch.Tensor | None = None,
    frame_weights: torch.Tensor | None = None,
    frame_chunk: int = 256,
) -> tuple[torch.Tensor, SignedMarginResult, torch.Tensor]:
    """Penalize view frames that lack the requested signed VQ margin."""

    result = signed_boundary_margin(
        view,
        codebook,
        teacher_ids,
        frame_chunk=frame_chunk,
        detach_codebook=True,
    )
    per_frame = F.relu(float(target_margin) - result.margin)
    active = torch.ones_like(per_frame, dtype=torch.bool)
    if frame_mask is not None:
        active &= frame_mask.to(device=per_frame.device, dtype=torch.bool).reshape(-1)
    if not torch.any(active):
        return per_frame.sum() * 0.0, result, per_frame

    selected = per_frame[active]
    if frame_weights is None:
        loss = selected.mean()
    else:
        weights = frame_weights.to(device=per_frame.device, dtype=per_frame.dtype).reshape(-1)[active]
        loss = (selected * weights).sum() / weights.sum().clamp_min(1e-8)
    return loss, result, per_frame
