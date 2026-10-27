"""Export prefix-key similarity from a SMoPE training checkpoint.

Example: python scripts/export_similarity.py --checkpoint outputs/xxx.pth --output sim.npz
The checkpoint must contain a standard SMoPE model state_dict or nested key.
"""
import argparse
import sys
from pathlib import Path
import re
import torch
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', default='similarity.npz')
a=p.parse_args()
ckpt=torch.load(a.checkpoint, map_location='cpu', weights_only=False)
if isinstance(ckpt, dict):
    for key in ('state_dict','model_state_dict','model'):
        if isinstance(ckpt.get(key), dict): ckpt=ckpt[key];break
rx=re.compile(r'e_pk_(\d+)_(\d+)_(\d+)$')
entries={}
for name, value in ckpt.items():
    m=rx.search(name)
    if m and torch.is_tensor(value): entries[tuple(map(int,m.groups()))]=value.reshape(-1).float()
if not entries: raise SystemExit('No e_pk_<layer>_<expert>_<head> keys in checkpoint')
matrices={}
for layer,head in sorted({(l,h) for l,e,h in entries}):
    ids=sorted(e for l,e,h in entries if (l,h)==(layer,head))
    k=torch.stack([entries[(layer,e,head)] for e in ids])
    k=torch.nn.functional.normalize(k,dim=-1)
    matrices[f'layer{layer}_head{head}']=(k@k.T).numpy()
np.savez_compressed(a.output, **matrices)
print(f'Saved {len(matrices)} similarity matrices to {a.output}')
