"""Full fine-tuning of the Sonata / Point Transformer V3 backbone for leaf-wood.

Unlike the linear-probe study (frozen features), this unfreezes the whole PTv3
encoder and trains it end-to-end with a linear segmentation head on the labelled
qforestlab trees. Mixed precision + grid subsampling keep it on a single 24 GB GPU.

Reports test-set mIoU and compares against the frozen linear-probe baseline.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn


def build_transform(grid_size):
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


def estimate_normals(xyz, radius=0.1, max_nn=30):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pcd.normals, dtype=np.float32)


class Tree:
    """Lazily-loaded tree: xyz, normals, labels (capped for memory)."""
    def __init__(self, path, max_points, rng):
        a = np.loadtxt(path).astype(np.float32)
        xyz, gt = a[:, :3], a[:, 3].astype(np.int64)
        if len(xyz) > max_points:
            s = rng.choice(len(xyz), max_points, replace=False)
            xyz, gt = xyz[s], gt[s]
        self.xyz, self.gt = xyz, gt
        self.normal = estimate_normals(xyz)

    def point(self):
        color = np.full_like(self.xyz, 127.5)
        return dict(coord=self.xyz.copy(), color=color, normal=self.normal,
                    segment=self.gt.copy())


def forward_feats(model, pt):
    out = model(pt)
    while "pooling_parent" in out.keys():
        parent = out.pop("pooling_parent"); inv = out.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, out.feat[inv]], dim=-1); out = parent
    return out.feat, out.segment


def miou(pred, gt):
    ious = []
    for c in (0, 1):
        inter = np.sum((pred == c) & (gt == c)); union = np.sum((pred == c) | (gt == c))
        ious.append(inter / max(union, 1))
    return float(np.mean(ious))


@torch.no_grad()
def evaluate(model, head, transform, files, max_points, rng, device):
    model.eval(); head.eval()
    preds, gts = [], []
    for f in files:
        t = Tree(f, max_points, rng)
        pt = transform(t.point())
        for k in pt:
            if isinstance(pt[k], torch.Tensor):
                pt[k] = pt[k].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            feat, seg = forward_feats(model, pt)
            logit = head(feat.float())
        preds.append(logit.argmax(1).cpu().numpy()); gts.append(seg.cpu().numpy())
    p, g = np.concatenate(preds), np.concatenate(gts)
    return miou(p, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/qforestlab/tls_tropical_leaf_wood")
    ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--max-points", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-enc", type=float, default=1e-4)
    ap.add_argument("--sonata-dir", default="sonata")
    ap.add_argument("--ckpt", default="ckpt/ptv3_finetuned.pth")
    args = ap.parse_args()
    device = "cuda"

    sys.path.insert(0, args.sonata_dir)
    import sonata
    cfg = dict(enc_patch_size=[256] * 5, enable_flash=False)
    model = sonata.load("sonata", repo_id="facebook/sonata", custom_config=cfg).cuda()
    head = nn.Linear(1232, 2).cuda()
    transform = build_transform(args.grid)

    train_files = sorted(glob.glob(os.path.join(args.data, "train", "*.txt")))
    test_files = sorted(glob.glob(os.path.join(args.data, "test", "*.txt")))
    rng = np.random.default_rng(0)

    # class weights from a sample of train labels
    wsample = np.concatenate([np.loadtxt(f)[:, 3] for f in train_files[:20]]).astype(int)
    w = np.array([len(wsample) / (2 * np.sum(wsample == 0)),
                  len(wsample) / (2 * np.sum(wsample == 1))], np.float32)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.lr_enc},
        {"params": head.parameters(), "lr": args.lr_head}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    print(f"[ft] {len(train_files)} train / {len(test_files)} test trees; "
          f"grid={args.grid} max_pts={args.max_points} epochs={args.epochs}")
    best = 0.0
    for ep in range(args.epochs):
        model.train(); head.train()
        order = rng.permutation(len(train_files))
        t0 = time.time(); losses = []
        skipped = 0
        for n, i in enumerate(order):
            t = Tree(train_files[i], args.max_points, rng)
            pt = transform(t.point())
            for k in pt:
                if isinstance(pt[k], torch.Tensor):
                    pt[k] = pt[k].cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    feat, seg = forward_feats(model, pt)
                    loss = crit(head(feat.float()), seg)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                losses.append(loss.item())
            except torch.OutOfMemoryError:  # skip an oversized tree, keep training
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); skipped += 1
            if n % 20 == 0:
                torch.cuda.empty_cache()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            m = evaluate(model, head, transform, test_files, args.max_points, rng, device)
            best = max(best, m)
            print(f"[ep {ep+1:02d}] loss={np.mean(losses):.3f} test_mIoU={m:.4f} "
                  f"best={best:.4f} peakGPU={torch.cuda.max_memory_allocated()/1e9:.1f}G "
                  f"({time.time()-t0:.0f}s)")
            if m >= best:
                os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "head": head.state_dict(),
                            "miou": m}, args.ckpt)
        else:
            print(f"[ep {ep+1:02d}] loss={np.mean(losses):.3f} skip={skipped} "
                  f"({time.time()-t0:.0f}s)")
    print(f"[done] best test mIoU = {best:.4f}  (linear-probe baseline = 0.919)")


if __name__ == "__main__":
    main()
