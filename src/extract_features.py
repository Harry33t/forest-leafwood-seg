"""Extract per-point features for the label-efficiency study.

For each labelled tree we cache:
  - feat_sonata : (K, 1232) frozen Sonata/PTv3 self-supervised features
  - feat_geom   : (K, 8)    hand-crafted geometric features (no pretraining baseline)
  - coord       : (K, 3)
  - label       : (K,)      0=leaf, 1=wood
K points are uniformly subsampled per tree to keep the cache bounded.

The two feature sets feed the SAME linear classifier, isolating the value of the
self-supervised representation vs hand-crafted geometry.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

GEOM_NAMES = ["linearity", "planarity", "sphericity", "verticality"]


def estimate_normals(xyz, radius=0.1, max_nn=30):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pcd.normals, dtype=np.float32)


def geom_features(xyz, normal, radius=0.1):
    """8-dim hand-crafted geometric baseline: 4 covariance feats + 3 normal + height."""
    from jakteristics import compute_features as jak
    f = jak(np.ascontiguousarray(xyz.astype(np.float64)),
            search_radius=radius, feature_names=GEOM_NAMES)
    f = np.nan_to_num(f).astype(np.float32)
    h = (xyz[:, 2:3] - xyz[:, 2].min()).astype(np.float32)
    return np.concatenate([f, normal, h], axis=1)  # (N, 8)


def build_transform(grid_size=0.02):
    """Like sonata.transform.default() but keeps `segment` (labels) through GridSample."""
    import sonata
    cfg = [
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse", "segment"),
             feat_keys=("coord", "color", "normal")),
    ]
    return sonata.transform.Compose(cfg)


def sonata_features(model, transform, xyz, normal, gt, color_val=127.5):
    color = np.full_like(xyz, color_val)
    pt = transform(dict(coord=xyz.astype(np.float32), color=color, normal=normal,
                        segment=gt.astype(np.int64)))
    inverse = pt["inverse"]
    with torch.inference_mode():
        for k in pt:
            if isinstance(pt[k], torch.Tensor):
                pt[k] = pt[k].cuda(non_blocking=True)
        out = model(pt)
        while "pooling_parent" in out.keys():
            parent = out.pop("pooling_parent"); inv = out.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, out.feat[inv]], dim=-1); out = parent
        feat = out.feat.float().cpu().numpy()
    return feat, pt["coord"].cpu().numpy(), pt["segment"].cpu().numpy(), inverse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--subsample", type=int, default=20000)
    ap.add_argument("--max-points", type=int, default=1_200_000,
                    help="cap input cloud (random subsample) to fit GPU memory")
    ap.add_argument("--sonata-dir", default="sonata")
    args = ap.parse_args()

    sys.path.insert(0, args.sonata_dir)
    import sonata
    cfg = dict(enc_patch_size=[256] * 5, enable_flash=False)
    model = sonata.load("sonata", repo_id="facebook/sonata", custom_config=cfg).cuda().eval()
    transform = build_transform()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(args.glob))
    rng = np.random.default_rng(0)
    for i, f in enumerate(files):
        name = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(args.outdir, name + ".npz")
        if os.path.exists(out):
            print(f"[skip] {name}"); continue
        t0 = time.time()
        a = np.loadtxt(f).astype(np.float32)
        xyz, gt = a[:, :3], a[:, 3].astype(np.int64)
        if len(xyz) > args.max_points:  # cap giant clouds so the forward fits 24GB
            sub = rng.choice(len(xyz), args.max_points, replace=False)
            xyz, gt = xyz[sub], gt[sub]
            print(f"      [cap] {name}: {len(a):,} -> {len(xyz):,} pts")
        normal = estimate_normals(xyz)
        # sonata features are aligned to the gridsampled cloud; we map back to the
        # gridsampled coords + their labels (segment carried through transform).
        feat_s, coord_g, seg_g, _ = sonata_features(model, transform, xyz, normal, gt)
        # geometric features on the SAME gridsampled coords for a fair comparison
        # (recompute normals on the gridsampled cloud)
        normal_g = estimate_normals(coord_g)
        feat_g = geom_features(coord_g, normal_g)
        n = len(coord_g)
        idx = rng.choice(n, min(args.subsample, n), replace=False)
        np.savez_compressed(out,
                            coord=coord_g[idx].astype(np.float32),
                            feat_sonata=feat_s[idx].astype(np.float32),
                            feat_geom=feat_g[idx].astype(np.float32),
                            label=seg_g[idx].astype(np.int8))
        print(f"[{i+1}/{len(files)}] {name}: {n} pts -> {len(idx)} cached, "
              f"wood={seg_g[idx].mean():.2f} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
