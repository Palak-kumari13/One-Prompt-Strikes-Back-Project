"""Plot prompt expert activation counts from SMOPE_EC_TRACE JSONL."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('log', type=Path)
p.add_argument('--out', default='expert_activation.png')
a = p.parse_args()
counts = None
for raw in a.log.read_text().splitlines():
    if not raw.strip():
        continue
    batch = np.asarray(json.loads(raw)['activations_per_head_expert'], dtype=np.int64)
    counts = batch if counts is None else counts + batch
if counts is None:
    raise SystemExit('No trace entries found')
total = counts.sum(axis=0)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(np.arange(len(total)), total)
ax.set(xlabel='Prompt expert ID', ylabel='Activation count', title='Prompt expert activation frequency')
ax.set_xticks(np.arange(len(total)))
fig.tight_layout()
fig.savefig(a.out, dpi=180)
print('Saved', a.out)
