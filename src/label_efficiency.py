"""Label-efficiency curve: linear probing on frozen features.

Compares two representations under the SAME linear classifier and SAME label budgets:
  - Sonata  : 1232-d self-supervised PTv3 features
  - Geometry: 8-d hand-crafted covariance features (no pretraining)

For label fractions {1,5,10,100}% of training points we train a class-weighted linear
classifier (3 seeds) and report test-set mIoU (mean of leaf/wood IoU). The gap between
the curves at low label budgets is the headline result: SSL features reach good leaf-wood
mIoU with far fewer labels.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn


def load_split(d):
    X_s, X_g, Y = [], [], []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        X_s.append(z["feat_sonata"]); X_g.append(z["feat_geom"]); Y.append(z["label"])
    return (np.concatenate(X_s), np.concatenate(X_g),
            np.concatenate(Y).astype(np.int64))


def iou_metrics(pred, gt):
    out = {}
    ious = []
    for c in (0, 1):
        inter = np.sum((pred == c) & (gt == c))
        union = np.sum((pred == c) | (gt == c))
        iou = inter / max(union, 1)
        ious.append(iou)
        out[f"IoU_{'leaf' if c==0 else 'wood'}"] = iou
    out["mIoU"] = float(np.mean(ious))
    out["OA"] = float(np.mean(pred == gt))
    return out


def train_linear(Xtr, Ytr, Xte, device, epochs=300, lr=1e-2, wd=1e-4):
    # standardize on the labelled training subset
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(Ytr, dtype=torch.long, device=device)
    xte = torch.tensor(Xte, dtype=torch.float32, device=device)
    # class weights for imbalance
    cls, cnt = np.unique(Ytr, return_counts=True)
    w = np.ones(2, np.float32)
    for c, n in zip(cls, cnt):
        w[c] = len(Ytr) / (2.0 * n)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    clf = nn.Linear(Xtr.shape[1], 2).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(clf(xtr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = clf(xte).argmax(1).cpu().numpy()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featdir", default="data/processed/feat")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--fractions", default="0.01,0.05,0.1,1.0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outfig", default="figs/label_efficiency.png")
    ap.add_argument("--outcsv", default="data/processed/label_efficiency.csv")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xs_tr, Xg_tr, Y_tr = load_split(os.path.join(args.featdir, "train"))
    Xs_te, Xg_te, Y_te = load_split(os.path.join(args.featdir, args.test_split))
    print(f"train {len(Y_tr):,} pts (wood {Y_tr.mean():.3f}) | "
          f"test {len(Y_te):,} pts (wood {Y_te.mean():.3f})")

    fracs = [float(x) for x in args.fractions.split(",")]
    feats = {"Sonata (SSL, 1232-d)": (Xs_tr, Xs_te),
             "Geometry (hand, 8-d)": (Xg_tr, Xg_te)}
    results = {k: {"frac": [], "miou_mean": [], "miou_std": []} for k in feats}
    rows = [("feature", "label_frac", "seed", "mIoU", "IoU_wood", "IoU_leaf", "OA")]

    for name, (Xtr, Xte) in feats.items():
        for fr in fracs:
            mious = []
            for s in range(args.seeds):
                rng = np.random.default_rng(1000 * s + int(fr * 1e4))
                n = max(2, int(len(Y_tr) * fr))
                idx = rng.choice(len(Y_tr), n, replace=False)
                # guarantee both classes present
                if len(np.unique(Y_tr[idx])) < 2:
                    wood = np.where(Y_tr == 1)[0]
                    idx = np.concatenate([idx, rng.choice(wood, 1)])
                pred = train_linear(Xtr[idx], Y_tr[idx], Xte, device)
                m = iou_metrics(pred, Y_te)
                mious.append(m["mIoU"])
                rows.append((name, fr, s, f"{m['mIoU']:.4f}",
                             f"{m['IoU_wood']:.4f}", f"{m['IoU_leaf']:.4f}", f"{m['OA']:.4f}"))
            results[name]["frac"].append(fr)
            results[name]["miou_mean"].append(float(np.mean(mious)))
            results[name]["miou_std"].append(float(np.std(mious)))
            print(f"{name:24s} frac={fr:5.0%}  mIoU={np.mean(mious):.3f}±{np.std(mious):.3f}")

    # save csv
    os.makedirs(os.path.dirname(args.outcsv) or ".", exist_ok=True)
    with open(args.outcsv, "w") as f:
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"[saved] {args.outcsv}")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    colors = {"Sonata (SSL, 1232-d)": "#1f77b4", "Geometry (hand, 8-d)": "#888888"}
    for name, r in results.items():
        x = np.array(r["frac"]) * 100
        m = np.array(r["miou_mean"]); sd = np.array(r["miou_std"])
        plt.plot(x, m, "-o", color=colors[name], label=name, lw=2)
        plt.fill_between(x, m - sd, m + sd, color=colors[name], alpha=0.15)
    plt.xscale("log")
    plt.xticks([1, 5, 10, 100], ["1%", "5%", "10%", "100%"])
    plt.xlabel("labelled training points")
    plt.ylabel("test mIoU (leaf / wood)")
    plt.title("Label efficiency of self-supervised features\nfor leaf-wood separation (TLS)")
    plt.grid(alpha=0.3); plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.outfig) or ".", exist_ok=True)
    plt.savefig(args.outfig, dpi=150)
    print(f"[saved] {args.outfig}")


if __name__ == "__main__":
    main()
