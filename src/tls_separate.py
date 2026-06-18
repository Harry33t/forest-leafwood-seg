"""Wood/leaf separation via the TLSeparation library (topology + geometry).

Stronger than single-scale point-wise thresholding: uses path detection + GMM
filtering. Used as the geometric pseudo-label ("weak teacher") for the project.

Returns per-point wood(1)/leaf(0) labels by matching the separated sub-clouds
back to the original points with a KD-tree.
"""
import argparse
import os
import time

import numpy as np


def separate(xyz, voxel_size=0.05):
    import tlseparation as tls
    wood, leaf = tls.scripts.generic_tree(np.ascontiguousarray(xyz),
                                          voxel_size=voxel_size)
    return wood, leaf


def to_pointwise(xyz, wood, leaf):
    """Assign each original point the label of its nearest separated point."""
    from scipy.spatial import cKDTree
    pts = np.vstack([wood, leaf])
    lab = np.concatenate([np.ones(len(wood), int), np.zeros(len(leaf), int)])
    _, idx = cKDTree(pts).query(xyz, k=1)
    return lab[idx]


def score(pred, gt):
    tp = int(np.sum((pred == 1) & (gt == 1))); tn = int(np.sum((pred == 0) & (gt == 0)))
    fp = int(np.sum((pred == 1) & (gt == 0))); fn = int(np.sum((pred == 0) & (gt == 1)))
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(OA=(tp + tn) / len(gt), precision=p, recall=r,
               F1=2 * p * r / max(p + r, 1e-9),
               IoU_wood=tp / max(tp + fp + fn, 1),
               mIoU=(tp / max(tp + fp + fn, 1) + tn / max(tn + fp + fn, 1)) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--save-pred", default=None, help="save xyz+pred+gt npy")
    args = ap.parse_args()

    a = np.loadtxt(args.input)
    xyz, gt = a[:, :3], (a[:, 3].astype(int) if a.shape[1] >= 4 else None)
    print(f"[load] {len(xyz):,} pts; gt={'yes' if gt is not None else 'no'}")

    t0 = time.time()
    wood, leaf = separate(xyz, args.voxel_size)
    print(f"[tls] wood={len(wood):,} leaf={len(leaf):,} ({time.time()-t0:.0f}s)")
    pred = to_pointwise(xyz, wood, leaf)
    print(f"[pred] wood fraction = {pred.mean():.3f}")

    if gt is not None:
        for k, v in score(pred, gt).items():
            print(f"   {k:10s} {v:.4f}")
    if args.save_pred:
        os.makedirs(os.path.dirname(args.save_pred) or ".", exist_ok=True)
        np.save(args.save_pred, np.column_stack([xyz, pred,
                gt if gt is not None else -np.ones(len(xyz))]))
        print(f"[saved] {args.save_pred}")


if __name__ == "__main__":
    main()
