"""
task1.py - expert-choice routing for SMoPE (OnePrompt). Run from the repo root.

Task 1  store the prompt-expert score matrix of every minibatch (evaluation only, no training)
    python task1.py scores --log_dir outputs/cifar-100/10-task/one-prompt --out_dir score_matrices/cifar-100

Tasks 2+3  expert-choice routing and expert capacity
    The code lives in models/vit.py (expert_choice_mask, expert_capacity), switched on with the
    environment variable SMOPE_ROUTING=expert_choice. capacity = round(B * K / E) inputs per expert,
    so the average number of active experts per input stays about K (B=128, K=5, E=25 -> 26).

Task 4  train + evaluate with the ORIGINAL pipeline (run.py), only the routing is different
    sh experiments/cifar-100_ec.sh

Task 5  compare the two trained models (FAA, CAA, balance, cost) and plot the activation frequency
    python task1.py compare --log_dir_topk outputs/cifar-100/10-task/one-prompt \
                            --log_dir_ec   outputs/cifar-100/10-task/one-prompt-ec
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


# ----------------------------------------------------------------------------------------
# helpers shared by all tasks
# ----------------------------------------------------------------------------------------
def build(log_dir, repeat):
    """Rebuild Trainer (same class split, datasets and model as in training)."""
    from trainer import Trainer

    with open(log_dir + "/args.yaml") as f:
        args = argparse.Namespace(**yaml.load(f, Loader=yaml.Loader))
    args.log_dir = log_dir
    r = repeat - 1
    seed = args.seeds[r] if r < len(args.seeds) else r
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return Trainer(args, seed, ["acc", "time", "fr"], ["global", "pt"], r), args


def batches(trainer, args, repeat, ckpts, routing, capacity_factor=1.0):
    """Evaluation only (no training). For each checkpoint i in `ckpts` (0-based) and each task j <= i,
    run task j's test images in order and yield, for every minibatch:
        (ckpt_task, eval_task, batch_idx, start, raw, adj)
    raw = prompt-expert scores  q_mean . pk^T          shape (B, layers, heads, experts)
    adj = noise-adjusted scores that the routing uses   shape (B, layers, heads, experts)
    """
    from models import vit

    vit.ROUTING = routing  # which routing the model uses in its forward pass
    vit.CAPACITY_FACTOR = capacity_factor

    learner = trainer.learner
    core = learner.model.module if hasattr(learner.model, "module") else learner.model
    layers = list(core.prompt.e_layers)

    got = {}

    def make_hook(l):
        def hook(module, inputs, output):
            got[l] = output[1]  # (prompt_score_, prompt_score_label_), each (B, heads, 1, experts)

        return hook

    hooks = [core.feat.blocks[l].attn.register_forward_hook(make_hook(l)) for l in layers]
    try:
        for i in ckpts:
            model_dir = f"{args.log_dir}/models/repeat-{repeat}/task-{trainer.task_names[i]}/"
            learner.task_count = i
            learner.valid_out_dim = sum(len(t) for t in trainer.tasks_logits[: i + 1])
            learner.load_model(model_dir)  # loads class.pth, eval mode
            core.task_id = i
            for j in range(i + 1):
                trainer.test_dataset.load_dataset(j, train=True)  # test images of task j only
                loader = DataLoader(
                    trainer.test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=args.workers,
                    pin_memory=True,
                )
                start = 0
                for b, (x, _, _) in enumerate(loader):
                    if learner.gpu:
                        x = x.cuda()
                    with torch.no_grad():
                        core(x)  # forward only; the hooks collect the scores
                    raw = torch.stack([got[l][0].squeeze(2) for l in layers], dim=1)
                    adj = torch.stack([got[l][1].squeeze(2) for l in layers], dim=1)
                    yield i + 1, j + 1, b, start, raw, adj
                    start += x.size(0)
    finally:
        for h in hooks:
            h.remove()


def get_ckpts(spec, n_tasks):
    if spec == "last":
        return [n_tasks - 1]
    if spec == "all":
        return list(range(n_tasks))
    return [int(t) - 1 for t in spec.split(",")]


# ----------------------------------------------------------------------------------------
# Task 1: store the score matrix of every minibatch
# ----------------------------------------------------------------------------------------
def task_scores(a):
    trainer, args = build(a.log_dir, a.repeat)
    ckpts = get_ckpts(a.ckpt_tasks, trainer.max_task)
    dtype = torch.float16 if a.fp16 else torch.float32
    for ck, ev, b, start, raw, adj in batches(trainer, args, a.repeat, ckpts, a.routing, a.capacity_factor):
        folder = f"{a.out_dir}/ckpt-task-{ck:02d}/eval-task-{ev:02d}"
        os.makedirs(folder, exist_ok=True)
        torch.save(
            {"scores": raw.to(dtype).cpu(), "batch_idx": b, "start": start, "eval_task": ev, "ckpt_task": ck},
            f"{folder}/batch_{b:05d}.pt",
        )
        if b == 0:
            print(f"ckpt task {ck}, eval task {ev}: saving batches to {folder}")
    print("Saved score matrices to", a.out_dir)


# ----------------------------------------------------------------------------------------
# Tasks 2, 3, 5: which experts are active, how balanced, how expensive
# ----------------------------------------------------------------------------------------
def assignments(adj, routing, topk):
    """Which experts are active for each input.  adj (B, L, H, E) -> bool (B, L, H, E)."""
    if routing == "topk":  # SMoPE: every input picks its K best experts
        idx = adj.topk(topk, dim=-1).indices
        return torch.zeros_like(adj, dtype=torch.bool).scatter_(-1, idx, True)
    from models import vit  # expert choice: the same function the model uses

    per_layer = [vit.expert_choice_mask(adj[:, l].unsqueeze(2), topk).squeeze(2) for l in range(adj.shape[1])]
    return torch.stack(per_layer, dim=1)


def summarize(freq, zero_frac, head_dim, n_tokens=197):
    """freq (L, H, E) = fraction of inputs for which each expert was active."""
    L, H, E = freq.shape
    active = freq.sum(-1).mean().item()  # average number of active experts per input
    cv = (freq.std(-1, unbiased=False) / freq.mean(-1).clamp_min(1e-12)).mean().item()  # 0 = perfectly even
    # multiply-adds per image in the prompt part: routing scores (E*D) + chosen prompt logits (A*D)
    # + attention-weighted values (n_tokens*A*D), for every head of every prompted layer
    macs = L * H * head_dim * (E + active * (1 + n_tokens))
    return {"active_experts": active, "zero_expert_inputs": zero_frac, "balance_cv": cv, "prompt_macs": macs}


def usage(a, log_dir, routing):
    """Run the last checkpoint on the test images of all tasks and count the active experts."""
    trainer, args = build(log_dir, a.repeat)
    core = trainer.learner.model
    core = core.module if hasattr(core, "module") else core
    topk, head_dim = core.prompt.topk, core.prompt.head_dim
    count, n_inputs, n_zero, n_rows = 0, 0, 0, 0
    for ck, ev, b, start, raw, adj in batches(trainer, args, a.repeat, [trainer.max_task - 1], routing, a.capacity_factor):
        act = assignments(adj.cpu(), routing, topk)  # (B, L, H, E)
        count = count + act.sum(0)
        n_inputs += act.shape[0]
        n_zero += (act.sum(-1) == 0).sum().item()
        n_rows += act.shape[0] * act.shape[1] * act.shape[2]
    freq = count.float() / n_inputs
    return freq, summarize(freq, n_zero / n_rows, head_dim)


def read_acc(log_dir):
    """FAA and CAA of every repeat, from the results that run.py wrote."""
    with open(log_dir + "/results-acc/global.yaml") as f:
        h = np.array(yaml.safe_load(f)["history"])  # (tasks, repeats)
    return h[-1], h.mean(0)


def task_compare(a):
    import matplotlib.pyplot as plt

    names = ["SMoPE top-K", "expert choice"]
    runs = [(a.log_dir_topk, "topk"), (a.log_dir_ec, "expert_choice")]
    freqs, stats, accs = [], [], []
    for log_dir, routing in runs:
        freq, s = usage(a, log_dir, routing)
        freqs.append(freq)
        stats.append(s)
        accs.append(read_acc(log_dir))

    def pm(v):
        return f"{np.mean(v):.2f} +- {np.std(v):.2f}"

    print(f"\n{'':32s}{names[0]:>18s}{names[1]:>18s}")
    print(f"{'FAA (%)':32s}{pm(accs[0][0]):>18s}{pm(accs[1][0]):>18s}")
    print(f"{'CAA (%)':32s}{pm(accs[0][1]):>18s}{pm(accs[1][1]):>18s}")
    print(f"{'active experts per input':32s}{stats[0]['active_experts']:>18.2f}{stats[1]['active_experts']:>18.2f}")
    print(f"{'inputs with 0 experts (%)':32s}{100 * stats[0]['zero_expert_inputs']:>18.2f}{100 * stats[1]['zero_expert_inputs']:>18.2f}")
    print(f"{'balance CV (0 = even)':32s}{stats[0]['balance_cv']:>18.3f}{stats[1]['balance_cv']:>18.3f}")
    print(f"{'prompt MACs per image (M)':32s}{stats[0]['prompt_macs'] / 1e6:>18.2f}{stats[1]['prompt_macs'] / 1e6:>18.2f}")

    # activation frequency of every expert: rows = heads, columns = experts, one layer
    vmax = max(f[a.layer].max().item() for f in freqs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, f, name in zip(axes, freqs, names):
        im = ax.imshow(f[a.layer].numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{name} (layer {a.layer})")
        ax.set_xlabel("expert index")
    axes[0].set_ylabel("head")
    fig.colorbar(im, ax=axes, label="fraction of test inputs for which the expert is active")
    if a.png:
        fig.savefig(a.png, dpi=150)
        print("Saved figure to", a.png)
    else:
        plt.show()


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["scores", "compare"])
    p.add_argument("--repeat", type=int, default=1, help="1-based repeat (seed) to evaluate")
    p.add_argument("--capacity_factor", type=float, default=1.0, help="expert capacity multiplier (must match training)")
    # task 1
    p.add_argument("--log_dir", help="[scores] training output dir")
    p.add_argument("--out_dir", help="[scores] where to store the score matrices")
    p.add_argument("--routing", default="topk", choices=["topk", "expert_choice"], help="[scores] routing of the model")
    p.add_argument("--ckpt_tasks", default="last", help="[scores] 'last', 'all' or e.g. 1,5,10")
    p.add_argument("--fp16", action="store_true", help="[scores] store float16")
    # task 5
    p.add_argument("--log_dir_topk", help="[compare] output dir of the original SMoPE run")
    p.add_argument("--log_dir_ec", help="[compare] output dir of the expert-choice run")
    p.add_argument("--layer", type=int, default=0, help="[compare] layer shown in the heatmap (0..5)")
    p.add_argument("--png", default=None, help="[compare] save the figure here instead of showing it")
    a = p.parse_args()

    if a.mode == "scores":
        task_scores(a)
    else:
        task_compare(a)