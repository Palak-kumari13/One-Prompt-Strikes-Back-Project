# SMoPE research extensions: Task 1 + Task 2

Independent, toggleable modifications for the **official [SMoPE repository](https://github.com/Minhchuyentoancbn/SMoPE)** (branch `master`). **This ZIP is an add-on installer, not a vendored copy of the upstream repository.** Requires the official source, datasets, checkpoints and dependencies described in upstream README.

## Project structure

```text
SMoPE_Dual_Tasks/
  task1/expert_choice.py        # Expert-choice routing (Task 1)
  task2/orthogonal.py           # Cosine-based orthogonality loss (Task 2)
  scripts/install.py            # Safely integrates chosen task into upstream
  scripts/export_similarity.py  # Extract key matrices from checkpoint
  scripts/plot_similarity.py    # Save cosine-heatmap figure
  scripts/plot_experts.py       # Original Task 1 activation visualization
  scripts/compare_results.py   # Original Task 1 comparison utility
  tests/                      # Independent unit tests
```

## Setup

```bash
git clone https://github.com/Minhchuyentoancbn/SMoPE.git
# Extract this ZIP so SMoPE/ and SMoPE_Dual_Tasks/ share a parent folder.
cd SMoPE
python ../SMoPE_Dual_Tasks/scripts/install.py --repo . --task both
```

Install with `--task 1` or `--task 2` for separate modifications. This installer creates `models/vit.py.before_dual_tasks` and `learners/prompt.py.before_dual_tasks` backups before changing the originals, and refuses to patch incompatible/already patched files. Run on a **clean** official checkout, rather than a checkout previously patched with the old Task 1 ZIP. Integration is tested against anchors seen in upstream master; subsequent upstream revisions may change them.

## Pick experiment (Linux shell)

```bash
# Original baseline
SMOPE_ROUTING=input SMOPE_ORTHO_LAMBDA=0 sh experiments/cifar-100.sh

# Task 1 only: expert-choice instead of input-choice
SMOPE_ROUTING=expert SMOPE_ORTHO_LAMBDA=0 sh experiments/cifar-100.sh

# Task 2 only: original router + orthogonal prefix-key loss
SMOPE_ROUTING=input SMOPE_ORTHO_LAMBDA=0.01 sh experiments/cifar-100.sh

# Combined ablation (optional, not a substitute for individual comparisons)
SMOPE_ROUTING=expert SMOPE_ORTHO_LAMBDA=0.01 sh experiments/cifar-100.sh
```

Use the upstream `experiments/cub-200.sh` and `experiments/imagenet-r_all.sh` for other benchmarks. Preserve identical seeds, task order, model, optimizer, training duration, and dataset splits across comparisons. Create **separate output directories** for each run: upstream may otherwise overwrite files. Windows PowerShell users can run Linux commands in WSL, or set `$env:SMOPE_ORTHO_LAMBDA="0.01"` before invoking Python.

### Task 1: what changes?

In `models/vit.py` each expert per head chooses `C ≈ round(batch_size * original_topk * capacity_factor / num_experts)` input samples. Thus expert assignment is balanced per minibatch, but inputs may receive 0..E experts. Unassigned inputs continue to attend to image tokens; masked padded prompts cannot contribute. Routing is non-differentiable just like top-k. All prompt KV values remain trainable. Environment:

- `SMOPE_ROUTING=expert` activates Task 1; default `input` uses upstream.
- `SMOPE_EC_CAPACITY_FACTOR=1.0` controls capacity. Smaller means fewer assignments.
- `SMOPE_EC_TRACE=1` writes JSONL (opt-in, potentially slow). `SMOPE_EC_TRACE_PATH=...` sets path.

**Important:** Original SMoPE auxiliary router loss and frequency counters are still input-topk-based in the upstream `models/zoo.py`. They are **not expert-choice-aware** in this first version, and may counteract Task 1 balancing. Report this limitation; for a rigorous Task 1 reproduction, align the auxiliary objective and frequency tracking with selected expert-choice assignments. Additionally, changing the evaluation minibatch composition changes routing decisions. Evaluate multiple evaluation batch sizes as an ablation.

### Task 2: what changes?

For layer `l`, head `h`, expert keys `K` shaped `[E, D]`, first L2-normalize rows: `U = normalize(K)`, then calculate cosine matrix `S = U @ U.T`. The auxiliary objective is

`L_orth = mean_{layers,heads} [ (sum_{i != j} S[i,j]^2)/(E*(E-1)) ]`.

It penalizes both parallel and antiparallel directions. Diagonal entries are ignored. When `E=1`, the loss is zero. The original cross-entropy, prototype, and router losses remain unchanged. `L_total = L_original + lambda * L_orth`, applied in `learners/prompt.py` **only in `OnePrompt.update_model`**. No extra trainable parameters, and requires no Task 1 changes. Defaults to `SMOPE_ORTHO_LAMBDA=0` (pure baseline).

Try `lambda ∈ {0, 0.001, 0.01, 0.1, 1.0}`. These are suggestions, **not tuned values**. Compare FAA/CAA from upstream evaluation, across multiple seeds, and plot heatmaps; use the same checkpoint for both metric and matrix analysis.

```bash
python ../SMoPE_Dual_Tasks/scripts/export_similarity.py --checkpoint PATH_TO_CHECKPOINT --output similarity.npz
python ../SMoPE_Dual_Tasks/scripts/plot_similarity.py similarity.npz --output similarity.png
```

Checkpoint export accepts plain or nested `state_dict`, `model_state_dict`, or `model` dictionaries. If your checkpoint is a custom object, adapt the extractor accordingly. Compare off-diagonal absolute cosine similarity and squared similarity along with FAA/CAA and expert activation counts. More diverse keys do not guarantee better continual-learning accuracy.

## Testing

From the extracted extension folder:

```bash
python -m unittest discover -s tests -v
```

Requires `torch` for routing/loss tests, `numpy` and `matplotlib` for visualization. No dataset or GPU is necessary for unit tests. End-to-end model training requires an actual upstream installation and downloaded data/checkpoints.

## Research caveats

- This extension does **not** contain trained weights, invented benchmark results, or a claim that all original training runs have completed.
- Capacity and packed prompt width can differ with batch size and input-to-expert overlap; equal expected assignments is not equal FLOPs/latency.
- In Task 2 the regularizer acts on all trainable prefix keys, regardless of which experts were selected in a given batch; this avoids overlooking inactive experts.
- Before submission, run original baseline, task1-only, task2-only, and combined experiments with 3+ seeds. Save their FAA, CAA, runtime, GPU memory, activation distributions, and cosine heatmaps.

Source: [Official SMoPE repository](https://github.com/Minhchuyentoancbn/SMoPE) (MIT). This is independent research code, not an official upstream release.
