"""Geometric weak labels for leaf-wood separation.

For each point, compute covariance-eigenvalue geometric features (linearity,
verticality, ...) over a local neighbourhood, then threshold to a wood/leaf
pseudo-label:  wood = linear/cylindrical (high linearity), leaf = scattered/planar.

This is the no-training "money shot": run on a labelled tree, compare the geometric
weak label against the manual ground truth, and report point-wise accuracy.

Usage:
    python weak_labels.py --input tree.txt --outfig fig.png \
        --search-radius 0.05 --lin-thresh 0.7 [--vert-thresh 0.5]
"""
import argparse
import os
import sys

import numpy as np

LEAF, WOOD = 0, 1


def load_points(path):
    """Load an x y z [label ...] point cloud (txt/csv/laz). Returns (xyz, gt_or_None)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".laz", ".las"):
        import laspy
        las = laspy.read(path)
        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        return xyz, None
    # text formats: sniff delimiter + header
    with open(path) as f:
        first = f.readline()
    delim = "," if first.count(",") >= first.count(" ") and "," in first else None
    has_header = any(c.isalpha() for c in first.replace("e", "").replace("E", ""))
    arr = np.loadtxt(path, delimiter=delim, skiprows=1 if has_header else 0)
    if arr.ndim == 1:
        arr = arr[None, :]
    xyz = arr[:, :3].astype(np.float64)
    gt = None
    if arr.shape[1] >= 4:
        lbl = arr[:, 3].astype(int)
        # accept binary {0,1} label column
        if set(np.unique(lbl)).issubset({0, 1}):
            gt = lbl
    return xyz, gt


def compute_features(xyz, search_radius):
    """Per-point geometric features via jakteristics. Returns dict of arrays."""
    from jakteristics import compute_features as jak
    names = ["linearity", "planarity", "sphericity", "verticality"]
    feats = jak(xyz, search_radius=search_radius, feature_names=names)
    return {n: feats[:, i] for i, n in enumerate(names)}


def weak_label(feats, lin_thresh, vert_thresh=None):
    """wood = high linearity (optionally AND high verticality)."""
    wood = feats["linearity"] >= lin_thresh
    if vert_thresh is not None:
        wood = wood | (feats["verticality"] >= vert_thresh)
    return wood.astype(int)  # 1=wood, 0=leaf


def score(pred, gt):
    """Binary metrics treating wood(1) as positive."""
    tp = int(np.sum((pred == WOOD) & (gt == WOOD)))
    tn = int(np.sum((pred == LEAF) & (gt == LEAF)))
    fp = int(np.sum((pred == WOOD) & (gt == LEAF)))
    fn = int(np.sum((pred == LEAF) & (gt == WOOD)))
    oa = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    iou_w = tp / max(tp + fp + fn, 1)
    iou_l = tn / max(tn + fp + fn, 1)
    return dict(OA=oa, precision=prec, recall=rec, F1=f1,
                mIoU=(iou_w + iou_l) / 2, IoU_wood=iou_w, IoU_leaf=iou_l)


def render_panels(xyz, panels, outfig, point_size=0.5):
    """panels: list of (title, label_array | None). Saves a side-by-side figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # leaf=green, wood=brown
    cmap = np.array([[0.20, 0.60, 0.20], [0.55, 0.27, 0.07]])
    n = len(panels)
    fig = plt.figure(figsize=(6 * n, 7))
    # downsample for plotting if huge
    idx = np.arange(len(xyz))
    if len(xyz) > 200_000:
        idx = np.random.default_rng(0).choice(len(xyz), 200_000, replace=False)
    p = xyz[idx]
    for k, (title, lab) in enumerate(panels):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        if lab is None:
            c = np.full((len(p), 3), 0.5)
        else:
            c = cmap[lab[idx]]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=point_size, c=c, linewidths=0, marker=".")
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        ax.set_box_aspect((np.ptp(p[:, 0]), np.ptp(p[:, 1]), np.ptp(p[:, 2])))
        ax.view_init(elev=8, azim=-70)
    fig.tight_layout()
    fig.savefig(outfig, dpi=150, bbox_inches="tight")
    print(f"[saved] {outfig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outfig", default="figs/weak_label_demo.png")
    ap.add_argument("--search-radius", type=float, default=0.05)
    ap.add_argument("--lin-thresh", type=float, default=0.7)
    ap.add_argument("--vert-thresh", type=float, default=None)
    args = ap.parse_args()

    xyz, gt = load_points(args.input)
    print(f"[load] {args.input}: {len(xyz):,} points, "
          f"gt={'yes' if gt is not None else 'no'}")
    if gt is not None:
        print(f"       gt wood fraction = {np.mean(gt == WOOD):.3f}")

    print(f"[features] jakteristics r={args.search_radius} ...")
    feats = compute_features(xyz, args.search_radius)
    pred = weak_label(feats, args.lin_thresh, args.vert_thresh)
    print(f"[weak]  pred wood fraction = {np.mean(pred == WOOD):.3f}")

    panels = [("raw", None), ("geometric weak label", pred)]
    if gt is not None:
        m = score(pred, gt)
        print("[metrics vs ground truth]")
        for k, v in m.items():
            print(f"   {k:10s} {v:.4f}")
        panels.append(("ground truth", gt))

    os.makedirs(os.path.dirname(args.outfig) or ".", exist_ok=True)
    render_panels(xyz, panels, args.outfig)


if __name__ == "__main__":
    sys.exit(main())
