"""Rotating 3D GIF of a predicted leaf-wood tree (signature animation).

Trains the linear probe on the cached Sonata features, predicts on one tree, and
renders a 360-degree rotation coloured by prediction (leaf = green, wood = brown).
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

CMAP = np.array([[0.16, 0.62, 0.20], [0.50, 0.30, 0.10]])


def load_pool(featdir, split):
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(featdir, split, "*.npz"))):
        z = np.load(f); X.append(z["feat_sonata"]); Y.append(z["label"].astype(np.int64))
    return np.concatenate(X), np.concatenate(Y)


def train_head(X, Y, device, epochs=300):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    x = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    y = torch.tensor(Y, dtype=torch.long, device=device)
    cls, cnt = np.unique(Y, return_counts=True); w = np.ones(2, np.float32)
    for c, n in zip(cls, cnt):
        w[c] = len(Y) / (2.0 * n)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    clf = nn.Linear(X.shape[1], 2).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(); crit(clf(x), y).backward(); opt.step()
    return clf, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featdir", default="data/processed/feat")
    ap.add_argument("--tree", required=True)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--out", default="figs/leafwood_rotate.gif")
    ap.add_argument("--max-points", type=int, default=60000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, Ytr = load_pool(args.featdir, "train")
    clf, mu, sd = train_head(Xtr, Ytr, device)
    z = np.load(args.tree)
    xyz, X = z["coord"], (z["feat_sonata"] - mu) / sd
    with torch.no_grad():
        pred = clf(torch.tensor(X, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    if len(xyz) > args.max_points:
        i = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        xyz, pred = xyz[i], pred[i]

    box = (np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2]))
    frames = []
    for k in range(args.frames):
        azim = -180 + 360 * k / args.frames
        fig = plt.figure(figsize=(4, 5), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1.0, c=CMAP[pred],
                   linewidths=0, marker=".")
        ax.set_axis_off(); ax.set_box_aspect(box); ax.view_init(elev=6, azim=azim)
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
        plt.close(fig)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, duration=0.08, loop=0)
    print(f"[saved] {args.out}  ({args.frames} frames, {len(xyz)} pts)")


if __name__ == "__main__":
    main()
