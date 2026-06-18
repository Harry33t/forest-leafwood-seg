"""Render manually-annotated leaf-wood ground truth as a 3D gallery (money shot).

Real per-point annotations from the labelled TLS dataset: leaf=green, wood=brown.
No modelling — this is genuine ground truth, used as the headline figure.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CMAP = np.array([[0.16, 0.62, 0.20], [0.50, 0.30, 0.10]])  # leaf, wood


def load(path, max_pts=200_000):
    a = np.loadtxt(path)
    xyz, lab = a[:, :3], a[:, 3].astype(int)
    if len(xyz) > max_pts:
        i = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)
        xyz, lab = xyz[i], lab[i]
    return xyz, lab


def draw(ax, xyz, lab, title, s=0.4):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=s, c=CMAP[lab],
               linewidths=0, marker=".")
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    ax.set_box_aspect((np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2])))
    ax.view_init(elev=6, azim=-75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for *.txt trees")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--outfig", default="figs/gt_gallery.png")
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))[: args.n]
    cols = args.cols
    rows = (len(files) + cols - 1) // cols
    fig = plt.figure(figsize=(5 * cols, 6 * rows), facecolor="white")
    for k, f in enumerate(files):
        xyz, lab = load(f)
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        name = os.path.splitext(os.path.basename(f))[0]
        draw(ax, xyz, lab, f"{name}  (wood {lab.mean()*100:.0f}%)")
    # legend
    h = [plt.Line2D([0], [0], marker="o", ls="", mfc=CMAP[0], mec="none", label="leaf"),
         plt.Line2D([0], [0], marker="o", ls="", mfc=CMAP[1], mec="none", label="wood")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Leaf-wood ground truth · TLS tropical trees", fontsize=16, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    os.makedirs(os.path.dirname(args.outfig) or ".", exist_ok=True)
    fig.savefig(args.outfig, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.outfig}  ({len(files)} trees)")


if __name__ == "__main__":
    main()
