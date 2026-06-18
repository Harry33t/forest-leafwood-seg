# 技术方案 · forest-leafwood-seg

> Label-Efficient Leaf-Wood Separation in TLS Forest Point Clouds
> 自监督(Sonata/PTv3)+ 几何弱标签,目标 5–10% 标签逼近全监督。

本方案的关键技术声明均来自 2026-06 的多源调研并经对抗验证(见每节引用)。

---

## ① 推荐技术栈与仓库清单

| 角色 | 选型 | 仓库 / 来源 | 说明 |
|---|---|---|---|
| 自监督脊柱 | **Sonata** (CVPR'25) | github.com/facebookresearch/sonata · arXiv:2503.16429 | **原生 PTv3 backbone**;HF `facebook/sonata` 权重一行加载直接初始化下游 backbone。许可 CC-BY-NC 4.0(学术可用) |
| 分割 backbone | **Point Transformer V3** | github.com/Pointcept/PointTransformerV3 · arXiv:2312.10035 | 与 Sonata 同族,无缝衔接 |
| 训练框架 | **Pointcept** (≥v1.6.0) | github.com/Pointcept/Pointcept | Sonata 预训练复现 + PTv3 微调统一入口 |
| 几何特征 | **jakteristics** | github.com/jakarto3d/jakteristics | 逐点协方差特征:linearity/planarity/verticality/法向 |
| 几何分离对照 | **TLSeparation** / **LeWoS** | github.com/TLSeparation/source · github.com/dwang520/LeWoS | 弱标签交叉验证与基线 |
| 点云 IO / 可视化 | laspy · Open3D | — | .laz 读写、3D 查看器、出图/GIF |

**为何弃 Point-MAE**:Point-MAE 用 Point-BERT 式标准 transformer,**无法直接初始化 PTv3**(架构断层);Sonata 原生 PTv3,省掉最易翻车的预训练衔接环节。

---

## ② 数据获取清单

| 数据集 | 用途 | 下载 | 规模/格式 | 标注 |
|---|---|---|---|---|
| **qforestlab** | ⭐有标签微调/评测(救急 money shot) | Zenodo 13759407 (CC-BY 4.0) | 254.8MB · 148 棵热带树 TLS · `.txt` | **逐点叶/木**(第 4 列 0=叶/1=木) |
| **FOR-species20K** | 自监督预训练大语料 | Zenodo 13255198 | ~2 万棵单木 · `.laz` · 33 种 | 仅物种标签(**无** leaf-wood) |
| FOR-instance(可选) | 跨平台/跨站点泛化 | 公开森林点云基准 | UAV-LS 实例分割 | 实例,无叶/木 |
| Wytham Woods / NIBIO(可选) | 跨站点验证 | 公开 TLS 基准(待核实标注) | TLS | 待确认是否含叶/木 |

> 现实:带 leaf-wood 标注的公开 TLS 稀缺 → 用 qforestlab 做有标签锚点,FOR-species20K 做无标签预训练,几何弱标签补全大规模无标注区。

---

## ③ 几何弱标签实现方案

原理(多源证实):**wood = 线性/圆柱(高 linearity)、leaf = 散乱/面状**。判别基础是每点邻域协方差矩阵特征值。

```
对每点 p:
  邻域 KNN/半径 → 协方差矩阵 C → 特征值 λ1≥λ2≥λ3
  linearity   = (λ1 - λ2) / λ1      # 越高越像枝干
  planarity   = (λ2 - λ3) / λ1
  sphericity  = λ3 / λ1             # 越高越像叶团/散乱
  verticality = 1 - |e3·z|          # 树干高
弱标签规则: linearity ≥ τ (并可结合 verticality) → wood, 否则 leaf
```
- **参数**:CWLS 推荐 min linearity≈0.7;LeWoS≈0.125(阈值含义/尺度不同,需按数据标定);邻域尺度需多尺度试(如 r=2/5/10cm)。
- **库**:jakteristics 直接算上述全部特征;TLSeparation/LeWoS 做对照与图正则化精修。
- **已知失败模式**:密冠/遮挡区破碎细枝非线性 → 误判为叶;LeWoS 木点完整度随枝阶下降(树干 92.8% → 5 阶 47.9%)。这正是 demo 要如实报告并用自监督缓解的痛点。

---

## ④ 训练 / 评测流水线

```
阶段0 数据准备: .laz/.txt → 归一化/分块(grid sampling) → Pointcept 格式
阶段1 几何弱标签: jakteristics 算特征 → 规则伪标签(全量,无需人工)
阶段2 自监督预训练: Sonata/PTv3 在 FOR-species20K(无标签)→ 或直接用 HF 预训练权重
阶段3 微调: PTv3 + 分割头, 叶/木二分类, 在 qforestlab 有标签子集
阶段4 评测签名图:
  · [B1] 标签效率曲线: 1/5/10/100% 标签 × (Sonata 初始化 vs 从零)
  · [B2] LOCO 跨站点/跨树种迁移矩阵 (mIoU 热图)
  · [B3] 逐点不确定性 (MC-dropout 优先,工程简单;deep ensemble 备选)
  · [B6] 遮挡鲁棒性分层精度 (按点密度/冠层高度分桶报 mIoU)
```
精度参照带:几何法 OA 0.81–0.95;有监督点 transformer 热带树 mIoU≈92.2%(上限);无监督 DL(GrowSP-ForMS)mIoU 69.6%。

---

## ⑤ 单卡 4090(24GB)算力预算与风险点

> ⚠️ 调研未找到 Sonata+PTv3 在 4090 24GB 的**实测**显存/batch/时长证据 → 以下为待实机标定的工程预算,非引用数字。

- **显存策略**:点云分块(grid/voxel sampling)控制单样本点数(经验起步 ~30k–80k 点/样本),AMP 混合精度,batch 从 2–4 起调;PTv3 用序列化 attention,显存随点数近线性。
- **预训练**:可**先直接用 HF 预训练权重跳过自监督**(demo 最快路径);若自训,FOR-species20K 全量在单卡需数天量级 → MVP 阶段只在子集上短训做对照。
- **瓶颈**:点云预处理(KNN/协方差)吃 CPU/内存而非 GPU;本机 256 核/755GB 内存充裕。
- **风险**:①Python 3.12 + torch2.5.1 下 spconv/pointops/flash-attn 编译兼容(可能需建独立 env);②几何弱标签噪声注入量;③域差(热带单木 vs 整林地)。

---

## ⑥ 4 周可落地里程碑(demo 优先)

- **W1**:环境(Pointcept+Sonata+jakteristics)+ 下 qforestlab & FOR-species20K 子集 + **几何弱标签 vs 真值三联图(首个 money shot,无需训练)** + PTv3 baseline 跑通
- **W2**:载入 Sonata 预训练权重微调叶/木 → **[B1] 标签效率曲线**
- **W3(MVP 红线)**:**[B2] LOCO 矩阵 + [B3] 不确定性热图 + [B6] 遮挡分层精度**
- **W4**:README/网页/GIF 打磨 + 200 字 motivation
