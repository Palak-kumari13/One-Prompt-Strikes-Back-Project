"""Generate a comparison table from measured results; no fabricated scores."""
import argparse
import csv
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--expert', type=Path, required=True)
p.add_argument('--out', type=Path, default=Path('comparison.csv'))
a = p.parse_args()
# Format for both files: metric,value (e.g., FAA,80.5; CAA,75.2).
def load(path):
    with path.open(newline='') as handle:
        return {r['metric']: float(r['value']) for r in csv.DictReader(handle)}
x, y = load(a.baseline), load(a.expert)
metrics = sorted(set(x) | set(y))
with a.out.open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['metric', 'original_smope', 'expert_choice', 'delta'])
    for m in metrics:
        old, new = x.get(m), y.get(m)
        writer.writerow([m, old if old is not None else '', new if new is not None else '',
                         new - old if old is not None and new is not None else ''])
print('Saved', a.out)
