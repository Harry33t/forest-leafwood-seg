"""Per-point predictive uncertainty for leaf-wood separation (deep-ensemble of linear heads).

Frozen Sonata features + an ensemble of K linear classifiers (different seeds /
bootstrap subsets). On a held-out tree we report, per point:
  - prediction (argmax of mean wood prob)
  - uncertainty = std of wood probability across the ensemble
Uncertainty is expected to peak at leaf-wood boundaries and occluded crown regions,
which is exactly where the literature says separation fails.

Renders a 4-panel figure: prediction | uncertainty | ground truth | errors.
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

LEAFWOOD = np.array([[0.16, 0.62, 0.20], [0.50, 0.30, 0.10]])


def load_pool(featdir, split):
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(featdir, split, "*.npz"))):
        z = np.load(f); X.append(z["feat_sonata"]); Y.append(z["label"].astype(np.int64))
    return np.concatenate(X), np.concatenate(Y)


def train_head(X, Y, device, seed, epochs=250, lr=1e-2, wd=1e-4):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Y), len(Y), replace=True)  # bootstrap
    Xb, Yb = X[idx], Y[idx]
    xb = torch.tensor(Xb, dtype=torch.float32, device=device)
    yb = torch.tensor(Yb, dtype=torch.long, device=device)
    cls, cnt = np.unique(Yb, return_counts=True); w = np.ones(2, np.float32)
    for c, n in zip(cls, cnt):
        w[c] = len(Yb) / (2.0 * n)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    torch.manual_seed(seed)
    clf = nn.Linear(X.shape[1], 2).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(); crit(clf(xb), yb).backward(); opt.step()
    return clf


def panel(ax, xyz, colors, title, s=1.2):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=s, c=colors, linewidths=0, marker=".")
    ax.set_title(title, fontsize=12); ax.set_axis_off()
    ax.set_box_aspect((np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2])))
    ax.view_init(elev=6, azim=-75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featdir", default="data/processed/feat")
    ap.add_argument("--tree", required=True, help="path to a test-tree npz")
    ap.add_argument("--k", type=int, default=8, help="ensemble size")
    ap.add_argument("--outfig", default="figs/uncertainty.png")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, Ytr = load_pool(args.featdir, "train")
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd

    z = np.load(args.tree)
    xyz, Xte, gt = z["coord"], (z["feat_sonata"] - mu) / sd, z["label"].astype(np.int64)
    xte = torch.tensor(Xte, dtype=torch.float32, device=device)

    probs = []
    for k in range(args.k):
        clf = train_head(Xtr, Ytr, device, seed=k)
        with torch.no_grad():
            p = torch.softmax(clf(xte), 1)[:, 1].cpu().numpy()  # wood prob
        probs.append(p)
    probs = np.stack(probs)              # (K, N)
    mean_p = np.clip(probs.mean(0), 1e-6, 1 - 1e-6)
    pred = (mean_p >= 0.5).astype(int)
    # predictive entropy (peaks at the leaf-wood decision boundary) — richer than
    # ensemble std for visualisation; normalised to [0,1] (binary max entropy = ln2)
    unc = -(mean_p * np.log(mean_p) + (1 - mean_p) * np.log(1 - mean_p)) / np.log(2)

    iou = []
    for c in (0, 1):
        inter = np.sum((pred == c) & (gt == c)); union = np.sum((pred == c) | (gt == c))
        iou.append(inter / max(union, 1))
    print(f"tree mIoU={np.mean(iou):.3f}  mean entropy={unc.mean():.3f}  "
          f"unc@error={unc[pred!=gt].mean():.3f} vs unc@correct={unc[pred==gt].mean():.3f}")

    err_c = np.stack([(pred != gt) * 0.9 + (pred == gt) * 0.8,
                      (pred == gt) * 0.8, (pred == gt) * 0.8], 1)  # red error / grey correct
    fig = plt.figure(figsize=(26, 9))
    ax1 = fig.add_subplot(141, projection="3d"); panel(ax1, xyz, LEAFWOOD[pred], "prediction", s=2.5)
    ax2 = fig.add_subplot(142, projection="3d")
    sc = ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2.5, c=unc, cmap="inferno",
                     vmin=0, vmax=max(float(np.quantile(unc, 0.995)), 1e-3),
                     linewidths=0, marker=".")
    ax2.set_title("predictive uncertainty (entropy)", fontsize=14); ax2.set_axis_off()
    ax2.set_box_aspect((np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2])))
    ax2.view_init(elev=6, azim=-75)
    fig.colorbar(sc, ax=ax2, fraction=0.04, pad=0.02)
    ax3 = fig.add_subplot(143, projection="3d"); panel(ax3, xyz, LEAFWOOD[gt], "ground truth", s=2.5)
    ax4 = fig.add_subplot(144, projection="3d"); panel(ax4, xyz, err_c, "errors (red)", s=2.5)
    fig.suptitle(f"Per-point uncertainty · {os.path.basename(args.tree).replace('.npz','')}"
                 f"  (mIoU {np.mean(iou):.2f})", fontsize=16)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.outfig) or ".", exist_ok=True)
    fig.savefig(args.outfig, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.outfig}")


if __name__ == "__main__":
    main()
