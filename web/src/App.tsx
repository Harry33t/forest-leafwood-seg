import { useEffect, useState } from "react";
import Viewer from "./Viewer.tsx";

const asset = (p: string) => `${import.meta.env.BASE_URL}${p}`;

type Figure = { src: string; title: string; caption: string; wide?: boolean };

const FIGURES: Figure[] = [
  {
    src: "figs/gt_gallery.png",
    title: "Leaf-wood ground truth",
    caption: "Manually annotated tropical-tree TLS point clouds (leaf = green, wood = brown).",
  },
  {
    src: "figs/label_efficiency.png",
    title: "Label-efficiency curve",
    caption: "Test mIoU vs labelled fraction. Self-supervised features stay flat from 1% to 100%; hand-crafted geometry trails by about 15 mIoU.",
  },
  {
    src: "figs/loco_matrix.png",
    title: "Leave-one-site-out transfer",
    caption: "Train-site × test-site mIoU. Self-supervised features transfer near-uniformly across the three plots.",
    wide: true,
  },
  {
    src: "figs/occlusion_strata.png",
    title: "Occlusion robustness",
    caption: "Wood-detection completeness vs canopy height. Both methods fall in the occluded crown; the self-supervised model falls far less.",
  },
  {
    src: "figs/uncertainty_dro_040.png",
    title: "Per-point uncertainty",
    caption: "Prediction · predictive entropy · ground truth · errors. Uncertainty concentrates where separation is hardest.",
    wide: true,
  },
];

function App() {
  const [zoom, setZoom] = useState<string | null>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setZoom(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-inner">
          <p className="kicker">Terrestrial laser scanning · point-cloud segmentation</p>
          <h1>Label-Efficient Leaf-Wood Separation in TLS Forest Point Clouds</h1>
          <p className="byline">
            <span className="byline-name">Guanxiong Huang</span>
            <span className="byline-sep" aria-hidden="true">|</span>
            <span className="byline-org">College of Information Engineering, Northwest A&amp;F University</span>
            <span className="byline-sep" aria-hidden="true">|</span>
            <a href="mailto:harry.huang@nwafu.edu.cn">harry.huang@nwafu.edu.cn</a>
          </p>
          <p className="lede">
            Separating wood from leaves in terrestrial-laser-scanning point clouds is a leading
            source of uncertainty in plot-scale biomass, and remains difficult under canopy
            occlusion. This project studies how far self-supervised pre-training reduces the
            labelling required — reaching near-fully-supervised accuracy from roughly 1% of
            labels, transferring across sites, and reporting where the model is unsure.
          </p>
          <Viewer />
          <p className="viewer-cap">
            A held-out ~30&nbsp;m emergent tree at full scan density (~250k points), predicted leaf
            vs wood. Drag to rotate; toggle prediction, ground truth, and errors. Prediction agrees
            with the human labels on ~97% of points; the few residual errors fall on thin twigs at
            the leaf–wood boundary.
          </p>
        </div>
      </header>

      <main>
        <section className="section">
          <h2>Why leaf-wood separation</h2>
          <p className="prose">
            Wood/leaf separation is the largest single source of uncertainty in TLS-based
            aboveground biomass, yet it remains unsolved under canopy occlusion. Manual 3-D
            annotation is the practical bottleneck, and the few labelled datasets cover only a
            handful of forest types, so methods rarely generalise. We ask how far self-supervised
            pre-training pushes <em>label efficiency</em>, cross-site transfer, and uncertainty
            awareness on this task.
          </p>
        </section>

        <section className="section">
          <h2>Results</h2>
          <div className="table-wrap">
            <table className="results">
              <caption>
                Leaf-wood separation on annotated tropical-tree TLS (148 trees, 3 plots). A single
                linear classifier is trained on frozen features; the two feature sets differ only
                in pre-training.
              </caption>
              <thead>
                <tr>
                  <th>Features</th>
                  <th>mIoU&nbsp;@1%</th>
                  <th>mIoU&nbsp;@100%</th>
                  <th>Cross-site</th>
                  <th>Crown wood recall</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Self-supervised (Sonata / PTv3)</td>
                  <td className="best">0.90</td>
                  <td className="best">0.92</td>
                  <td className="best">0.89</td>
                  <td className="best">0.61</td>
                </tr>
                <tr>
                  <td>Hand-crafted geometry</td>
                  <td>0.75</td>
                  <td>0.75</td>
                  <td>0.71</td>
                  <td>0.20</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section className="section">
          <h2>Figures</h2>
          <div className="figs">
            {FIGURES.map((f) => (
              <figure
                className={f.wide ? "fig fig-wide" : "fig"}
                key={f.src}
                onClick={() => setZoom(asset(f.src))}
              >
                <img src={asset(f.src)} alt={f.title} loading="lazy" />
                <figcaption>
                  <strong>{f.title}.</strong> {f.caption}
                </figcaption>
              </figure>
            ))}
          </div>
        </section>

        <section className="section">
          <h2>Method</h2>
          <pre className="pipeline">{`Sonata (self-supervised Point Transformer V3) — frozen encoder
   -> per-point features (1232-d)
        -> linear classifier  ->  leaf / wood

Baseline   same linear classifier on 8-d hand-crafted covariance features

[B1] label-efficiency curve   1 / 5 / 10 / 100% labels
[B2] leave-one-site-out       cross-site transfer matrix
[B3] uncertainty              ensemble of linear heads, predictive entropy
[B4] occlusion strata         wood completeness vs canopy height`}</pre>
          <p className="prose">
            Evaluated on 148 manually leaf/wood-annotated tropical-tree TLS point clouds
            (RIEGL VZ-400, 3 plots). Hand-crafted geometric weak labels (covariance linearity /
            verticality) serve as a zero-label reference. Feature extraction fits on a single
            24&nbsp;GB GPU.
          </p>
        </section>
      </main>

      <footer className="footer">
        <p className="footer-author">
          Guanxiong Huang · College of Information Engineering, Northwest A&amp;F University ·{" "}
          <a href="mailto:harry.huang@nwafu.edu.cn">harry.huang@nwafu.edu.cn</a>
        </p>
        <p>forest-leafwood-seg — label-efficient leaf-wood separation for TLS forest point clouds</p>
      </footer>

      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)}>
          <img src={zoom} alt="enlarged figure" />
          <div className="lightbox-hint">click anywhere or press Esc to close</div>
        </div>
      )}
    </div>
  );
}

export default App;
