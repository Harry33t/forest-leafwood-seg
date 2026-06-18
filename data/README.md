# Data menu — forest-leafwood-seg

All open. Download into `data/raw/`, convert to Pointcept format under `data/processed/`.

| Dataset | Use | Source |
|---|---|---|
| FOR-species20K (~20k individual-tree laser point clouds) | Self-supervised pre-training corpus + species diversity | Puliti et al. 2025 (Methods in Ecology & Evolution) |
| FOR-instance (UAV-LS instance segmentation) | Cross-platform / cross-site generalisation | open forest point-cloud benchmark series |
| Wytham Woods / NIBIO TLS (optional) | leaf-wood validation subset | public TLS benchmarks |

> Weak labels: geometric features — wood = linear / cylindrical / high verticality; leaf =
> scattered / planar — give pseudo-labels (the point-cloud analogue of the CHM weak label in
> the urban-canopy project). Public leaf-wood-labelled TLS is scarce, which is exactly why the
> geometric-weak-label + self-supervised approach is the contribution.

Tip: start on single trees / a small subset, get the pipeline running, then scale up.
