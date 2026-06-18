"""[B6] Occlusion-robustness: leaf-wood mIoU stratified by canopy height.

Upper canopy is where TLS occlusion is worst and where the literature says
separation fails. We train one linear classifier on the train pool and evaluate
on the test pool, then bin test points by their normalised height within each
tree and report per-stratum mIoU for Sonata vs hand-crafted geometry. The honest
result: accuracy degrades with canopy height, and Sonata degrades less.
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


def load_pool(featdir, split, which):
    col = {"sonata": "feat_sonata", "geom": "feat_geom"}[which]
    X, Y, H = [], [], []
    for f in sorted(glob.glob(os.path.join(featdir, split, "*.npz"))):
        z = np.load(f)
        X.append(z[col]); Y.append(z["label"].astype(np.int64))
        zc = z["coord"][:, 2]
        H.append((zc - zc.min()) / (np.ptp(zc) + 1e-6))  # per-tree normalised height
    return np.concatenate(X), np.concatenate(Y), np.concatenate(H)


def train(Xtr, Ytr, device, epochs=300, lr=1e-2, wd=1e-4):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    ytr = torch.tensor(Ytr, dtype=torch.long, device=device)
    cls, cnt = np.unique(Ytr, return_counts=True); w = np.ones(2, np.float32)
    for c, n in zip(cls, cnt):
        w[c] = len(Ytr) / (2.0 * n)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    clf = nn.Linear(Xtr.shape[1], 2).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(); crit(clf(xtr), ytr).backward(); opt.step()
    return clf, mu, sd


def wood_recall(pred, gt):
    """Wood detection completeness = TP / (TP + FN), positive class = wood(1).
    Mirrors the LeWoS 'integrity' metric (trunk 92.8% -> high branch orders 47.9%)."""
    tp = np.sum((pred == 1) & (gt == 1)); fn = np.sum((pred == 0) & (gt == 1))
    return float(tp / max(tp + fn, 1)) if (tp + fn) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featdir", default="data/processed/feat")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--outfig", default="figs/occlusion_strata.png")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    edges = np.linspace(0, 1, args.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    plt.figure(figsize=(7.5, 5))
    for which, color in [("sonata", "#1f77b4"), ("geom", "#888888")]:
        Xtr, Ytr, _ = load_pool(args.featdir, "train", which)
        Xte, Yte, Hte = load_pool(args.featdir, "test", which)
        clf, mu, sd = train(Xtr, Ytr, device)
        with torch.no_grad():
            pred = clf(torch.tensor((Xte - mu) / sd, dtype=torch.float32,
                                    device=device)).argmax(1).cpu().numpy()
        ms = []
        for b in range(args.bins):
            m = (Hte >= edges[b]) & (Hte < edges[b + 1] + (b == args.bins - 1) * 1e-9)
            ms.append(wood_recall(pred[m], Yte[m]) if m.sum() else np.nan)
        label = "Sonata (SSL)" if which == "sonata" else "Geometry (hand)"
        plt.plot(centers, ms, "-o", color=color, lw=2, label=label)
        print(which, "per-stratum wood recall:", [f"{x:.3f}" for x in ms])

    plt.xlabel("normalised canopy height  (0 = base/trunk, 1 = treetop)")
    plt.ylabel("wood detection completeness  (recall)")
    plt.title("Occlusion robustness: wood completeness vs canopy height\n"
              "(upper canopy = most occluded → thin branches missed)")
    plt.grid(alpha=0.3); plt.legend(loc="lower left"); plt.ylim(0.0, 1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.outfig) or ".", exist_ok=True)
    plt.savefig(args.outfig, dpi=150)
    print(f"[saved] {args.outfig}")


if __name__ == "__main__":
    main()
