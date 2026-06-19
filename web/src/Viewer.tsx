import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

type Mode = "pred" | "gt" | "error";
type TreeData = { pos: Float32Array; pred: Uint8Array; gt: Uint8Array; n: number };

const LEAF = [0.36, 0.88, 0.42]; // fresh canopy green
const WOOD = [0.85, 0.57, 0.29]; // warm tan — pops against the green
const OK = [0.1, 0.13, 0.11]; // correct points recede into the dark
const ERR = [1.0, 0.32, 0.12]; // errors ignite

const BG = "#060b08";

function parse(buf: ArrayBuffer): TreeData {
  const dv = new DataView(buf);
  const n = dv.getUint32(0, true);
  let off = 4;
  const pos = new Float32Array(buf, off, n * 3);
  off += n * 3 * 4;
  const pred = new Uint8Array(buf, off, n);
  off += n;
  const gt = new Uint8Array(buf, off, n);
  return { pos, pred, gt, n };
}

// soft round sprite so points read as scan returns, not square blocks
function makeSprite(): THREE.Texture {
  const s = 64;
  const c = document.createElement("canvas");
  c.width = c.height = s;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.45, "rgba(255,255,255,0.95)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, s, s);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

function Cloud({ data, mode }: { data: TreeData; mode: Mode }) {
  const ref = useRef<THREE.Points>(null);
  const sprite = useMemo(makeSprite, []);

  // height (data z) -> three Y up; (x, z, -y). also track height range for shading
  const { positions, hNorm } = useMemo(() => {
    const p = new Float32Array(data.n * 3);
    let hmin = Infinity;
    let hmax = -Infinity;
    for (let i = 0; i < data.n; i++) {
      const h = data.pos[i * 3 + 2];
      p[i * 3] = data.pos[i * 3];
      p[i * 3 + 1] = h;
      p[i * 3 + 2] = -data.pos[i * 3 + 1];
      if (h < hmin) hmin = h;
      if (h > hmax) hmax = h;
    }
    const hN = new Float32Array(data.n);
    const span = hmax - hmin || 1;
    for (let i = 0; i < data.n; i++) hN[i] = (data.pos[i * 3 + 2] - hmin) / span;
    return { positions: p, hNorm: hN };
  }, [data]);

  const colors = useMemo(() => {
    const c = new Float32Array(data.n * 3);
    for (let i = 0; i < data.n; i++) {
      let col: number[];
      if (mode === "error") col = data.pred[i] === data.gt[i] ? OK : ERR;
      else col = (mode === "pred" ? data.pred : data.gt)[i] ? WOOD : LEAF;
      // gentle volumetric shading by height (skip in error mode so red stays hot)
      const shade = mode === "error" ? 1 : 0.82 + 0.3 * hNorm[i];
      c[i * 3] = Math.min(1, col[0] * shade);
      c[i * 3 + 1] = Math.min(1, col[1] * shade);
      c[i * 3 + 2] = Math.min(1, col[2] * shade);
    }
    return c;
  }, [data, mode, hNorm]);

  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.16;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={mode === "error" ? 0.0066 : 0.0052}
        map={sprite}
        alphaTest={0.4}
        transparent
        vertexColors
        sizeAttenuation
        depthWrite
      />
    </points>
  );
}

const LABELS: Record<Mode, string> = {
  pred: "Prediction",
  gt: "Ground truth",
  error: "Errors",
};

export default function Viewer() {
  const [data, setData] = useState<TreeData | null>(null);
  const [mode, setMode] = useState<Mode>("pred");
  const [err, setErr] = useState(false);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/tree.bin`)
      .then((r) => r.arrayBuffer())
      .then((b) => setData(parse(b)))
      .catch(() => setErr(true));
  }, []);

  const agreement = useMemo(() => {
    if (!data) return 0;
    let ok = 0;
    for (let i = 0; i < data.n; i++) if (data.pred[i] === data.gt[i]) ok++;
    return ok / data.n;
  }, [data]);

  return (
    <div className="viewer">
      <div className="viewer-bar">
        {(["pred", "gt", "error"] as Mode[]).map((m) => (
          <button
            key={m}
            className={m === mode ? "vbtn vbtn-on" : "vbtn"}
            onClick={() => setMode(m)}
          >
            {LABELS[m]}
          </button>
        ))}
        {data && (
          <span className="viewer-meta">
            {data.n.toLocaleString()} pts · {(agreement * 100).toFixed(1)}% vs truth · drag to rotate
          </span>
        )}
      </div>
      <div className="viewer-canvas">
        {err ? (
          <p className="viewer-msg">3D data unavailable.</p>
        ) : data ? (
          <Canvas camera={{ position: [0, 0.48, 1.42], fov: 45 }} dpr={[1, 2]}>
            <color attach="background" args={[BG]} />
            <fogExp2 attach="fog" args={[BG, 0.16]} />
            <Cloud data={data} mode={mode} />
            <OrbitControls
              enablePan={false}
              enableDamping
              dampingFactor={0.08}
              minDistance={0.6}
              maxDistance={4}
              target={[0, 0.46, 0]}
            />
          </Canvas>
        ) : (
          <p className="viewer-msg">Loading point cloud…</p>
        )}
      </div>
    </div>
  );
}
