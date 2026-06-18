"""Leave-one-site-out (LOCO) cross-site transfer matrix.

The labelled trees come from 3 plots/sites (dro / oc / rc). We train a linear
classifier on a SOURCE site's frozen features and evaluate leaf-wood mIoU on a
TARGET site. Cell (i, j) = train on site i, test on site j. Diagonal uses a
disjoint tree-level split within the site (no leakage). Off-diagonal = pure
cross-site transfer. We build the matrix for Sonata SSL features vs hand-crafted
geometry to show SSL features transfer across sites far better.
"""
import argparse
import glob
import os
import re

import numpy as np
import torch
import torch.nn as nn

SITES = ["dro", "oc", "rc"]


def site_of(path):
    return re.match(r"([a-z]+)_", os.path.basename(path)).group(1)


def load_all(featdir):
    """Return {site: list of (feat_sonata, feat_geom, label) per tree}."""
    data = {s: [] for s in SITES}
    for f in sorted(glob.glob(os.path.join(featdir, "*", "*.npz"))):
        s = site_of(f)
        if s not in data:
            continue
        z = np.load(f)
        data[s].append((z["feat_sonata"], z["feat_geom"], z["label"].astype(np.int64)))
    return data


def stack(trees, which):
    col = {"sonata": 0, "geom": 1}[which]
    X = np.concatenate([t[col] for t in trees])
    Y = np.concatenate([t[2] for t in trees])
    return X, Y


def train_eval(Xtr, Ytr, Xte, Yte, device, epochs=300, lr=1e-2, wd=1e-4):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(Ytr, dtype=torch.long, device=device)
    cls, cnt = np.unique(Ytr, return_counts=True)
    w = np.ones(2, np.float32)
    for c, n in zip(cls, cnt):
        w[c] = len(Ytr) / (2.0 * n)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    clf = nn.Linear(Xtr.shape[1], 2).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(); crit(clf(xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        pred = clf(torch.tensor(Xte, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    ious = []
    for c in (0, 1):
        inter = np.sum((pred == c) & (Yte == c)); union = np.sum((pred == c) | (Yte == c))
        ious.append(inter / max(union, 1))
    return float(np.mean(ious))


def matrix(data, which, device):
    M = np.zeros((len(SITES), len(SITES)))
    for i, si in enumerate(SITES):
        for j, sj in enumerate(SITES):
            if si == sj:  # within-site: disjoint tree split (no leakage)
                trees = data[si]; h = len(trees) // 2
                Xtr, Ytr = stack(trees[:h], which)
                Xte, Yte = stack(trees[h:], which)
            else:
                Xtr, Ytr = stack(data[si], which)
                Xte, Yte = stack(data[sj], which)
            M[i, j] = train_eval(Xtr, Ytr, Xte, Yte, device)
            print(f"  {which:7s} {si}->{sj}: mIoU={M[i,j]:.3f}")
    return M


def heat(ax, M, title):
    im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(SITES))); ax.set_xticklabels([s.upper() for s in SITES])
    ax.set_yticks(range(len(SITES))); ax.set_yticklabels([s.upper() for s in SITES])
    ax.set_xlabel("test site"); ax.set_ylabel("train site")
    for i in range(len(SITES)):
        for j in range(len(SITES)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.8 else "black", fontsize=13)
    ax.set_title(title)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featdir", default="data/processed/feat")
    ap.add_argument("--outfig", default="figs/loco_matrix.png")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_all(args.featdir)
    print("trees/site:", {s: len(data[s]) for s in SITES})
    Ms = matrix(data, "sonata", device)
    Mg = matrix(data, "geom", device)
    print(f"Sonata mean off-diagonal (transfer) = "
          f"{(Ms.sum()-np.trace(Ms))/(Ms.size-len(SITES)):.3f}")
    print(f"Geom   mean off-diagonal (transfer) = "
          f"{(Mg.sum()-np.trace(Mg))/(Mg.size-len(SITES)):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    heat(axes[0], Ms, "Sonata (SSL) — cross-site mIoU")
    im = heat(axes[1], Mg, "Geometry (hand) — cross-site mIoU")
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="leaf-wood mIoU")
    fig.suptitle("Leave-one-site-out transfer (TLS tropical plots)", fontsize=15)
    os.makedirs(os.path.dirname(args.outfig) or ".", exist_ok=True)
    fig.savefig(args.outfig, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.outfig}")


if __name__ == "__main__":
    main()
