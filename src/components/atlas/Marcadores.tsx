import { Html } from "@react-three/drei";
import { useFrame, type ThreeEvent } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { CIDADES } from "../../data/cidades";
import { paraCena, projectar, TERRITORIOS, type TerritorioId } from "../../lib/territorios";
import { TIPOLOGIAS } from "../../lib/tipologias";
import { useAtlas, visivel, type MonumentoResumo } from "./store";
import { EXAGERO, useTerreno } from "./Terrain";

const COR_PEDRA = new THREE.Color("#efe6d2");
const COR_ATIVA = new THREE.Color("#c0602f");

// Geometrias partilhadas: uma torre (castelos medievais) e um baluarte em estrela (fortificações modernas)
const haste = new THREE.CylinderGeometry(0.035, 0.035, 1, 6).translate(0, 0.5, 0);
const torre = new THREE.BoxGeometry(0.42, 0.55, 0.42).translate(0, 1.27, 0);
const baluarte = (() => {
  const s = new THREE.Shape();
  for (let i = 0; i < 10; i++) {
    const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
    const r = i % 2 === 0 ? 0.36 : 0.19;
    const p = [Math.cos(a) * r, Math.sin(a) * r] as const;
    i === 0 ? s.moveTo(...p) : s.lineTo(...p);
  }
  return new THREE.ExtrudeGeometry(s, { depth: 0.16, bevelEnabled: false }).rotateX(-Math.PI / 2).translate(0, 1.0, 0);
})();
const base = new THREE.CylinderGeometry(0.16, 0.16, 0.02, 20);

export function Marcadores({ monumentos, territorio }: { monumentos: MonumentoResumo[]; territorio: TerritorioId }) {
  const t = TERRITORIOS[territorio];
  const lista = useMemo(
    () =>
      monumentos
        .filter((m) => m.territorio === territorio)
        .map((m) => {
          const [x, y] = projectar(t, m.lon, m.lat);
          const [sx, sz] = paraCena(t, x, y);
          return { m, sx, sz };
        }),
    [monumentos, territorio, t],
  );
  return (
    <group>
      {lista.map(({ m, sx, sz }) => (
        <Marcador key={m.id} m={m} x={sx} z={sz} />
      ))}
      <Cidades territorio={territorio} />
    </group>
  );
}

function Marcador({ m, x, z }: { m: MonumentoResumo; x: number; z: number }) {
  const ref = useRef<THREE.Group>(null);
  const rotulo = useRef<HTMLDivElement>(null);
  const terreno = useTerreno();
  const mat = useMemo(() => new THREE.MeshStandardMaterial({ color: COR_PEDRA, roughness: 0.7 }), []);
  const forma = TIPOLOGIAS[m.tipologia].forma;
  const v = new THREE.Vector3();

  useFrame(({ camera }) => {
    const g = ref.current;
    if (!g) return;
    const s = useAtlas.getState();
    const ativo = s.selecionado === m.id || s.sobre === m.id;
    const show = visivel(m, s);
    g.visible = show;
    const h = terreno?.altitude(x, z) ?? 0;
    g.position.set(x, (Math.max(h, 0) / 1000) * EXAGERO, z);
    // tamanho constante no ecrã
    const d = camera.position.distanceTo(v.copy(g.position));
    const escala = Math.min(d * 0.022, 6) * (ativo ? 1.35 : 1);
    g.scale.setScalar(escala);
    mat.color.copy(ativo ? COR_ATIVA : COR_PEDRA);
    if (rotulo.current) {
      const perto = d < 140;
      rotulo.current.dataset.visivel = show && (ativo || perto) ? "1" : "0";
      rotulo.current.dataset.ativo = ativo ? "1" : "0";
    }
  });

  const over = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    useAtlas.getState().setSobre(m.id);
    document.body.style.cursor = "pointer";
  };
  const out = () => {
    if (useAtlas.getState().sobre === m.id) useAtlas.getState().setSobre(null);
    document.body.style.cursor = "";
  };
  const click = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    useAtlas.getState().selecionar(m.id);
  };

  return (
    <group ref={ref} onPointerOver={over} onPointerOut={out} onClick={click}>
      <mesh geometry={haste} material={mat} />
      <mesh geometry={forma === "torre" ? torre : baluarte} material={mat} />
      <mesh geometry={base} material={mat} />
      {/* área de toque mais generosa, invisível */}
      <mesh position={[0, 0.8, 0]} visible={false}>
        <sphereGeometry args={[0.7, 8, 8]} />
      </mesh>
      <Html position={[0, 1.9, 0]} center zIndexRange={[20, 10]} style={{ pointerEvents: "none" }}>
        <div ref={rotulo} className="rotulo-monumento" data-visivel="0">
          {m.nome}
        </div>
      </Html>
    </group>
  );
}

function Cidades({ territorio }: { territorio: TerritorioId }) {
  const t = TERRITORIOS[territorio];
  const terreno = useTerreno();
  const lista = useMemo(
    () =>
      CIDADES.filter((c) => c.territorio === territorio).map((c) => {
        const [x, y] = projectar(t, c.lon, c.lat);
        const [sx, sz] = paraCena(t, x, y);
        return { c, sx, sz };
      }),
    [territorio, t],
  );
  return (
    <group>
      {lista.map(({ c, sx, sz }) => (
        <Cidade key={c.nome} nome={c.nome} nivel={c.nivel} x={sx} z={sz} alt={() => terreno?.altitude(sx, sz) ?? 0} />
      ))}
    </group>
  );
}

function Cidade({ nome, nivel, x, z, alt }: { nome: string; nivel: number; x: number; z: number; alt: () => number }) {
  const ref = useRef<THREE.Group>(null);
  const el = useRef<HTMLDivElement>(null);
  useFrame(({ camera }) => {
    if (!ref.current) return;
    ref.current.position.set(x, (Math.max(alt(), 0) / 1000) * EXAGERO + 0.05, z);
    const d = camera.position.distanceTo(ref.current.position);
    if (el.current) el.current.dataset.visivel = nivel === 1 ? (d < 1500 ? "1" : "0") : d < 420 ? "1" : "0";
  });
  return (
    <group ref={ref}>
      <Html center zIndexRange={[5, 0]} style={{ pointerEvents: "none" }}>
        <div ref={el} className={`rotulo-cidade nivel-${nivel}`} data-visivel="0">
          <span className="ponto" />
          {nome}
        </div>
      </Html>
    </group>
  );
}
