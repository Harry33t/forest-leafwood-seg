# Motivation

## English (≈200 words, for cold emails / applications)

Separating wood from leaves in terrestrial laser scanning (TLS) point clouds is the
single largest source of uncertainty in plot-scale aboveground biomass estimates, yet it
remains unsolved under canopy occlusion. Manual annotation of 3-D points is the practical
bottleneck, and the few available labelled datasets cover only a handful of forest types,
so methods rarely generalise across sites.

This project asks how far self-supervised pre-training can push *label efficiency* for
leaf-wood separation. Using frozen features from a self-supervised Point Transformer V3
(Sonata) with nothing but a linear classifier, I reach 0.90 leaf-wood mIoU on annotated
tropical-tree TLS using only 1% of labelled points — within ~2 mIoU of the fully supervised
result and +15 mIoU over a hand-crafted geometric baseline. The same representation transfers
across plots in a leave-one-site-out setting with almost no loss (0.89 vs 0.71 for geometry),
and stays markedly more complete in the occluded upper canopy where existing methods collapse.

The result suggests a practical path to scalable, transferable leaf-wood separation that
needs only sparse annotation — a prerequisite for trustworthy TLS biomass and structural
trait retrieval. I would like to extend this to multi-site, multi-sensor forest inventories.

## 中文(草稿,供你改写)

在地面激光雷达(TLS)点云中分离木质与叶片,是样地尺度地上生物量估算最大的不确定性来源,
而在密冠遮挡下至今未解。逐点人工标注是现实瓶颈,且少数带标注数据只覆盖有限林型,方法难以跨站点泛化。

本项目探究自监督预训练能把 leaf-wood 分离的**标签效率**推到多远:仅用自监督 Point Transformer V3
(Sonata)的冻结特征 + 一个线性分类器,在带标注热带树 TLS 上**仅用 1% 标签即达 0.90 mIoU**,
距全监督约 2 mIoU、较手工几何基线高 +15 mIoU。同一表征在留一站点(LOCO)设置下跨样地几乎不掉
(0.89 vs 几何 0.71),并在已有方法崩溃的遮挡上冠层保持显著更高的木点完整度。

这指向一条可扩展、可迁移、只需稀疏标注的 leaf-wood 分离路径——可信 TLS 生物量与结构性状反演的前提。
希望将其拓展到多站点、多传感器的森林清查。
