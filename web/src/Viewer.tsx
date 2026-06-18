import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

type Mode = "pred" | "gt" | "error";
type TreeData = { pos: Float32Array; pred: Uint8Array; gt: Uint8Array; n: number };

const LEAF = [0.16, 0.62, 0.2];
const WOOD = [0.5, 0.3, 0.1];
const OK = [0.22, 0.27, 0.22];   // dark grey: correct points recede
const ERR = [1.0, 0.12, 0.08];   // bright red: errors pop

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

function Cloud({ data, mode }: { data: TreeData; mode: Mode }) {
  const ref = useRef<THREE.Points>(null);

  // height (data z) -> three Y up;  (x, z, -y)
  const positions = useMemo(() => {
    const p = new Float32Array(data.n * 3);
    for (let i = 0; i < data.n; i++) {
      p[i * 3] = data.pos[i * 3];
      p[i * 3 + 1] = data.pos[i * 3 + 2];
      p[i * 3 + 2] = -data.pos[i * 3 + 1];
    }
    return p;
  }, [data]);

  const colors = useMemo(() => {
    const c = new Float32Array(data.n * 3);
    for (let i = 0; i < data.n; i++) {
      let col: number[];
      if (mode === "error") col = data.pred[i] === data.gt[i] ? OK : ERR;
      else col = (mode === "pred" ? data.pred : data.gt)[i] ? WOOD : LEAF;
      c[i * 3] = col[0]; c[i * 3 + 1] = col[1]; c[i * 3 + 2] = col[2];
    }
    return c;
  }, [data, mode]);

  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.18;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.012} vertexColors sizeAttenuation />
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
          <Canvas camera={{ position: [1.6, 0.9, 1.6], fov: 45 }} dpr={[1, 2]}>
            <color attach="background" args={["#0b0f0a"]} />
            <Cloud data={data} mode={mode} />
            <OrbitControls enablePan={false} minDistance={0.8} maxDistance={4} />
          </Canvas>
        ) : (
          <p className="viewer-msg">Loading point cloud…</p>
        )}
      </div>
    </div>
  );
}
