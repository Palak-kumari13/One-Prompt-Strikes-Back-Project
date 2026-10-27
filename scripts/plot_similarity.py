"""Plot layer/head prefix cosine heatmaps from an exported NPZ."""
import argparse
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('npz', help='Output of save_similarity_matrices')
p.add_argument('--output', default='prefix_similarity.png')
p.add_argument('--max-panels', type=int, default=12)
a = p.parse_args()
data = np.load(a.npz)
keys = sorted(data.files)[:a.max_panels]
if not keys: raise SystemExit('No matrices present')
cols = min(4, len(keys)); rows = (len(keys) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(3.3*cols, 3.2*rows))
for ax in axes.flat: ax.axis('off')
for ax, key in zip(axes.flat, keys):
    ax.axis('on'); ax.imshow(data[key], vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_title(key); ax.set_xlabel('Expert'); ax.set_ylabel('Expert')
fig.tight_layout(); fig.savefig(a.output, dpi=180)
print('Saved:', a.output)
