"""
task2.py - compare SMoPE with and without the orthogonal prefix-key regularization.
Evaluation only (no training). Needs task1.py (read_acc, usage). Run from the repo root.

Train first:  sh experiments/cifar-100_orth.sh   (the original SMoPE run is the strength-0 reference)

    python task2.py \
        --runs  outputs/cifar-100/10-task/one-prompt \
                outputs/cifar-100/10-task/one-prompt-orth-0.01 \
                outputs/cifar-100/10-task/one-prompt-orth-0.1 \
                outputs/cifar-100/10-task/one-prompt-orth-1.0 \
        --names original 0.01 0.1 1.0

Prints FAA, CAA, the pairwise prefix-key similarity and the expert-activation balance of every run, then
shows the key-similarity matrices (top) and the expert activation frequencies (bottom).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from task1 import read_acc, usage

LAYERS, HEADS = 6, 12  # prompted layers and heads of SMoPE (models/zoo.py)


def prefix_keys(log_dir, repeat):
    """Prefix keys of the final checkpoint, shape (layers, heads, experts, head_dim)."""
    with open(log_dir + "/args.yaml") as f:
        n_experts = int(yaml.load(f, Loader=yaml.Loader)["prompt_param"][0]) // 2
    n_tasks = len(os.listdir(f"{log_dir}/models/repeat-{repeat}"))
    state = torch.load(f"{log_dir}/models/repeat-{repeat}/task-{n_tasks}/class.pth", map_location="cpu")
    return torch.stack(
        [
            torch.stack(
                [torch.cat([state[f"prompt.e_pk_{l}_{i}_{h}"] for i in range(n_experts)]) for h in range(HEADS)]
            )
            for l in range(LAYERS)
        ]
    )


def key_cosine(keys):
    """Pairwise cosine similarity between the experts' prefix keys, shape (layers, heads, experts, experts)."""
    k = F.normalize(keys, dim=-1)
    return k @ k.transpose(-1, -2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="output dirs; the first one is the original SMoPE")
    p.add_argument("--names", nargs="+", default=None, help="one label per run")
    p.add_argument("--repeat", type=int, default=1, help="1-based repeat used for the keys and activations")
    p.add_argument("--layer", type=int, default=0, help="layer shown in the figure (0..5)")
    p.add_argument("--head", type=int, default=0, help="head shown in the key-similarity matrix (0..11)")
    p.add_argument("--png", default=None, help="save the figure here instead of showing it")
    a = p.parse_args()
    names = a.names or [os.path.basename(r.rstrip("/")) for r in a.runs]

    cosines, freqs = [], []
    print(f"\n{'run':>12s}{'FAA (%)':>16s}{'CAA (%)':>16s}{'mean |cos|':>12s}{'max cos':>10s}{'act. CV':>10s}")
    for name, run in zip(names, a.runs):
        faa, caa = read_acc(run)
        cos = key_cosine(prefix_keys(run, a.repeat))
        off = cos[..., ~torch.eye(cos.shape[-1], dtype=torch.bool)]  # different experts only
        freq, s = usage(argparse.Namespace(repeat=a.repeat, capacity_factor=1.0), run, "topk")
        cosines.append(cos)
        freqs.append(freq)
        print(
            f"{name:>12s}{np.mean(faa):>9.2f}+-{np.std(faa):<5.2f}{np.mean(caa):>9.2f}+-{np.std(caa):<5.2f}"
            f"{off.abs().mean().item():>12.3f}{off.max().item():>10.3f}{s['balance_cv']:>10.3f}"
        )
    print("mean |cos| / max cos: similarity of different experts' prefix keys (0 = orthogonal)")
    print("act. CV: spread of how often each expert is activated (0 = perfectly even)\n")

    import matplotlib.pyplot as plt

    n = len(names)
    vmax = max(f[a.layer].max().item() for f in freqs)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
    for k in range(n):
        im1 = axes[0, k].imshow(cosines[k][a.layer, a.head].numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        axes[0, k].set_title(f"{names[k]}\nkey cosine (layer {a.layer}, head {a.head})")
        axes[0, k].set_xlabel("expert")
        axes[0, k].set_ylabel("expert")
        im2 = axes[1, k].imshow(freqs[k][a.layer].numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        axes[1, k].set_title(f"activation frequency (layer {a.layer})")
        axes[1, k].set_xlabel("expert")
        axes[1, k].set_ylabel("head")
    fig.colorbar(im1, ax=axes[0], shrink=0.8, label="cosine similarity")
    fig.colorbar(im2, ax=axes[1], shrink=0.8, label="fraction of test inputs")
    if a.png:
        fig.savefig(a.png, dpi=150)
        print("Saved figure to", a.png)
    else:
        plt.show()
