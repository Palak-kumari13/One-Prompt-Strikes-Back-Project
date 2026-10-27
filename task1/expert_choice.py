"""Expert-choice routing from existing SMoPE prompt proxy scores.

Scores: [batch, heads, 1, experts]. Each head is routed independently,
matching original SMoPE head-wise Top-K selection.
"""
import math
import torch


def expert_choice_mask(scores: torch.Tensor, original_topk: int, capacity_factor: float = 1.0):
    """Return boolean assignments [batch, heads, 1, experts] and capacity.

    Every expert selects `capacity` examples for each attention head.
    An input may have 0..E assigned experts; zeros are valid mathematically.
    """
    if scores.ndim != 4 or scores.shape[2] != 1:
        raise ValueError('Expected scores of shape [batch, heads, 1, experts]')
    b, heads, _, experts = scores.shape
    if not (b > 0 and experts > 0 and 0 < original_topk <= experts):
        raise ValueError('Invalid batch size, expert count or original Top-K')
    if not math.isfinite(capacity_factor) or capacity_factor <= 0:
        raise ValueError('capacity_factor must be a positive finite number')
    capacity = min(b, max(1, math.floor(b * original_topk * capacity_factor / experts + 0.5)))
    # [heads, experts, batch]: each expert ranks all inputs in the batch.
    ranked = scores.squeeze(2).permute(1, 2, 0)
    winners = ranked.topk(capacity, dim=-1, sorted=False).indices
    chosen = torch.zeros_like(ranked, dtype=torch.bool)
    chosen.scatter_(-1, winners, True)
    return chosen.permute(2, 0, 1).unsqueeze(2), capacity


def pack_selected_prompts(pk, pv, assignments):
    """Pack selected per-example prompt KV into padded tensors.

    pk, pv: [B,H,E,D]; assignments: [B,H,1,E].
    Returns packed keys/values [B,H,P,D], validity [B,H,1,P].
    Padding does not contribute to attention. If an example has no selected
    experts, its (dummy) prompt slot is masked off later.
    """
    if pk.shape != pv.shape or pk.ndim != 4:
        raise ValueError('pk and pv must be [batch,heads,experts,dim]')
    b, h, e, d = pk.shape
    if assignments.shape != (b, h, 1, e):
        raise ValueError('Assignments shape mismatch')
    active = assignments.squeeze(2)
    counts = active.sum(dim=-1)
    packed_width = max(1, int(counts.max().item()))
    # Sort active indices first; slicing pads shorter rows with inactive slots.
    ordering = torch.argsort(active.to(torch.int32), dim=-1, descending=True, stable=True)
    ix = ordering[..., :packed_width]
    validity = torch.gather(active, -1, ix).unsqueeze(2)
    ix = ix.unsqueeze(-1).expand(-1, -1, -1, d)
    return torch.gather(pk, 2, ix), torch.gather(pv, 2, ix), validity


def utilization_report(mask):
    """JSON-compatible summary for one expert-choice routing batch."""
    counts = mask.squeeze(2).sum(dim=0).float()  # [heads,experts]
    per_input = mask.squeeze(2).sum(dim=-1).float()
    return {
        'activations_per_head_expert': counts.int().cpu().tolist(),
        'total_assignments': int(mask.sum().item()),
        'mean_experts_per_input_head': float(per_input.mean().item()),
        'min_experts_per_input_head': int(per_input.min().item()),
        'max_experts_per_input_head': int(per_input.max().item()),
        'zero_assignment_fraction': float((per_input == 0).float().mean().item()),
        'expert_utilization_cv': float((counts.std(unbiased=False) / counts.mean().clamp_min(1e-12)).item()),
    }
