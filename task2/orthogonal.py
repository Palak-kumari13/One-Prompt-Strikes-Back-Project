"""Task 2: cosine-decorrelation of SMoPE's trainable prefix keys.

Each (ViT layer, attention head) has E key vectors of dimension D.
We penalize off-diagonal squared cosine similarities, NOT the diagonal.
This is sign-invariant: antiparallel experts are also penalized.
"""
import torch
import torch.nn.functional as F


def layer_head_keys(prompt, layer, head):
    """Return [experts, head_dim] with gradients connected to parameters."""
    return torch.cat([getattr(prompt, f'e_pk_{layer}_{expert}_{head}')
                      for expert in range(prompt.num_experts)], dim=0)


def cosine_matrix(keys):
    """Full [E,E] diagnostic cosine similarity matrix."""
    if keys.ndim != 2:
        raise ValueError('Expected [experts, features]')
    return F.normalize(keys.float(), p=2, dim=-1, eps=1e-8) @ F.normalize(
        keys.float(), p=2, dim=-1, eps=1e-8).T


def orthogonal_loss(prompt):
    """Mean off-diagonal cosine^2 averaged across layers and heads.

    E=1 yields a differentiable zero. No detached tensors in the loss path.
    """
    penalties = []
    for layer in prompt.e_layers:
        for head in range(prompt.num_heads):
            keys = layer_head_keys(prompt, layer, head)
            if keys.shape[0] < 2:
                penalties.append(keys.sum() * 0.0)
                continue
            sim = cosine_matrix(keys)
            # Subtract diagonal contributions; normalized diagonal is ~1.
            sq = sim.square()
            offdiag = sq.sum() - sq.diagonal().sum()
            penalties.append(offdiag / (keys.shape[0] * (keys.shape[0] - 1)))
    if not penalties:
        raise ValueError('No prompt layers found')
    return torch.stack(penalties).mean()


@torch.no_grad()
def save_similarity_matrices(prompt, path):
    """Save one matrix per layer/head, plus summary metadata, as NPZ."""
    import numpy as np
    arrays = {}
    for layer in prompt.e_layers:
        for head in range(prompt.num_heads):
            arrays[f'layer{layer}_head{head}'] = cosine_matrix(
                layer_head_keys(prompt, layer, head)).cpu().numpy()
    np.savez_compressed(path, **arrays)
    return path
