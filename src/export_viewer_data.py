"""Export one tree (positions + prediction + ground truth) as a compact binary
for the web 3D viewer.

Layout (little-endian):
  uint32  N
  float32 pos[3N]   normalised, centred, z = height
  uint8   pred[N]   0=leaf 1=wood
  uint8   gt[N]     0=leaf 1=wood
"""
import argparse
import glob
import os
import struct

import numpy as np
import torch
import torch.nn as nn


def load_pool(featdir, split):
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(featdir, split, "*.npz"))):
        z = np.load(f); X.append(z["feat_sonata"]); Y.append(z["label"].astype(np.int64))
    return np.concatenate(X), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featdir", default="data/processed/feat")
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", default="web/public/data/tree.bin")
    ap.add_argument("--max-points", type=int, default=35000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, Ytr = load_pool(args.featdir, "train")
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    x = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    y = torch.tensor(Ytr, dtype=torch.long, device=device)
    cls, cnt = np.unique(Ytr, return_counts=True); w = np.ones(2, np.float32)
    for c, n in zip(cls, cnt):
        w[c] = len(Ytr) / (2.0 * n)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    clf = nn.Linear(Xtr.shape[1], 2).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad(); crit(clf(x), y).backward(); opt.step()

    z = np.load(args.tree)
    xyz, gt = z["coord"].astype(np.float32), z["label"].astype(np.uint8)
    with torch.no_grad():
        pred = clf(torch.tensor((z["feat_sonata"] - mu) / sd, dtype=torch.float32,
                                device=device)).argmax(1).cpu().numpy().astype(np.uint8)
    if len(xyz) > args.max_points:
        i = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        xyz, gt, pred = xyz[i], gt[i], pred[i]

    # normalise: centre on XY, base z at 0, scale by max extent
    xyz = xyz.copy()
    xyz[:, 0] -= xyz[:, 0].mean(); xyz[:, 1] -= xyz[:, 1].mean()
    xyz[:, 2] -= xyz[:, 2].min()
    xyz /= (np.abs(xyz).max() + 1e-6)
    N = len(xyz)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<I", N))
        f.write(xyz.astype("<f4").tobytes())
        f.write(pred.astype(np.uint8).tobytes())
        f.write(gt.astype(np.uint8).tobytes())
    acc = float((pred == gt).mean())
    print(f"[saved] {args.out}  N={N}  pred-vs-gt agreement={acc:.3f}  "
          f"size={os.path.getsize(args.out)/1e6:.2f}MB")


if __name__ == "__main__":
    main()
