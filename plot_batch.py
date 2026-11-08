"""
Show one saved batch of prompt-expert scores as a heatmap. Nothing is written to disk.

    python plot_batch.py score_matrices/cifar-100/ckpt-task-10/eval-task-03/batch_00000.pt --layer 0 --head 0
    python plot_batch.py <batch.pt> --layer 0 --head all      # all 12 heads of one layer

Rows = inputs of the batch, columns = expert index (0..24), colour = raw score.
Needs matplotlib:  pip install matplotlib
"""

import argparse

import matplotlib.pyplot as plt
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch_file")
    p.add_argument("--layer", type=int, default=0, help="prompted layer index 0..5 (position in the saved layer axis)")
    p.add_argument("--head", default="0", help="head index 0..11, or 'all'")
    args = p.parse_args()

    d = torch.load(args.batch_file, map_location="cpu")
    S = d["scores"].float()  # (B, L, H, E)
    B, L, H, E = S.shape
    title = (
        f"ckpt task {d['ckpt_task']} | eval task {d['eval_task']} | "
        f"batch {d['batch_idx']} (samples {d['start']}-{d['start'] + B - 1}) | layer {args.layer}"
    )

    if args.head == "all":
        ncols = 4
        nrows = (H + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharex=True, sharey=True)
        vmin, vmax = S[:, args.layer].min().item(), S[:, args.layer].max().item()
        for h, ax in enumerate(axes.flat):
            im = ax.imshow(S[:, args.layer, h, :].numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"head {h}", fontsize=9)
        fig.supxlabel("expert index")
        fig.supylabel("input in batch")
        fig.suptitle(title)
        fig.colorbar(im, ax=axes, shrink=0.8, label="score")
    else:
        h = int(args.head)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(S[:, args.layer, h, :].numpy(), aspect="auto", cmap="viridis")
        ax.set_xlabel("expert index")
        ax.set_ylabel("input in batch")
        ax.set_title(f"{title} | head {h}", fontsize=10)
        fig.colorbar(im, ax=ax, label="score")

    plt.show()


if __name__ == "__main__":
    main()
