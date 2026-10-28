import { Html } from "@react-three/drei";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { CIDADES } from "../../data/cidades";
import { paraCena, projectar, TERRITORIOS, type TerritorioId } from "../../lib/territorios";
import { TIPOLOGIAS } from "../../lib/tipologias";
import { useAtlas, visivel, type MonumentoResumo } from "./store";
import { EXAGERO, useTerreno } from "./Terrain";

// Marcadores como pequenas figuras de torre branca com bandeira vermelha, à maneira do castillos.site:
// torre redonda ameada para os castelos medievais, baluarte em estrela com torre para as fortificações modernas.
// um pouco de luz própria: a figura lê-se branca mesmo do lado da sombra
const MAT_PEDRA = new THREE.MeshStandardMaterial({ color: "#f4efe4", roughness: 0.62, emissive: "#8f897d", emissiveIntensity: 0.55 });
const MAT_BANDEIRA = new THREE.MeshStandardMaterial({ color: "#b3261b", roughness: 0.7, side: THREE.DoubleSide });
const MAT_BASE = new THREE.MeshStandardMaterial({ color: "#5b4a38", roughness: 0.9 });
const MAT_ANEL = new THREE.MeshBasicMaterial({ color: "#c0392b", transparent: true, opacity: 0.85, depthWrite: false });

function ameias(raio: number, y: number, n: number, lado: number) {
  const g: THREE.BufferGeometry[] = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    g.push(new THREE.BoxGeometry(lado, lado * 1.25, lado).rotateY(-a).translate(Math.cos(a) * raio, y + lado * 0.62, Math.sin(a) * raio));
  }
  return g;
}

function torreRedonda(raio: number, base: number, altura: number) {
  const topo = base + altura;
  return [
    new THREE.CylinderGeometry(raio * 0.94, raio, altura, 20).translate(0, base + altura / 2, 0),
    new THREE.CylinderGeometry(raio * 1.18, raio * 0.96, 0.11, 20).translate(0, topo + 0.055, 0),
    ...ameias(raio * 1.02, topo + 0.11, 7, raio * 0.42),
  ];
}

const estrela = (() => {
  const s = new THREE.Shape();
  for (let i = 0; i < 10; i++) {
    const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
    const r = i % 2 === 0 ? 0.62 : 0.36;
    const p = [Math.cos(a) * r, Math.sin(a) * r] as const;
    i === 0 ? s.moveTo(...p) : s.lineTo(...p);
  }
  return new THREE.ExtrudeGeometry(s, { depth: 0.2, bevelEnabled: false }).rotateX(-Math.PI / 2).translate(0, 0.07, 0);
})();

function bandeira(topo: number) {
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.lineTo(0.34, -0.07);
  s.lineTo(0, -0.17);
  return [
    new THREE.CylinderGeometry(0.018, 0.018, 0.5, 6).translate(0, topo + 0.25, 0),
    new THREE.ShapeGeometry(s).translate(0.012, topo + 0.5, 0),
  ];
}

const juntar = (g: THREE.BufferGeometry[]) => mergeGeometries(g.map((x) => x.toNonIndexed()));
const FORMAS = {
  torre: {
    pedra: juntar(torreRedonda(0.3, 0.07, 0.86)),
    mastro: juntar([bandeira(1.08)[0]]),
    bandeira: bandeira(1.08)[1],
    topo: 1.62,
  },
  baluarte: {
    pedra: juntar([estrela, ...torreRedonda(0.2, 0.27, 0.5)]),
    mastro: juntar([bandeira(0.88)[0]]),
    bandeira: bandeira(0.88)[1],
    topo: 1.42,
  },
};
const GEO_BASE = new THREE.CylinderGeometry(0.34, 0.37, 0.06, 24).translate(0, 0.03, 0);
const GEO_ANEL = new THREE.RingGeometry(0.62, 0.8, 40).rotateX(-Math.PI / 2).translate(0, 0.02, 0);

// ——— Gestão dos rótulos ———
// Com o mapa afastado, os nomes não aparecem todos (atropelar-se-iam). Ao passar o rato
// sobre um monumento, mostra-se o nome dele e dos vizinhos próximos no ecrã, em etiquetas
// de papel dispostas de modo a não se sobreporem. Perto do chão, mostram-se todos os que couberem.
type Registo = { m: MonumentoResumo; grupo: THREE.Group; el: HTMLDivElement };
const registos = new Map<string, Registo>();
const RAIO_VIZINHOS = 110; // px à volta do monumento sob o rato
const DIST_TODOS = 140; // km da câmara abaixo dos quais se tentam mostrar todos os nomes
const ALTURA_ROTULO = 24;
const DESLOCAMENTOS = [0, -ALTURA_ROTULO, ALTURA_ROTULO, -2 * ALTURA_ROTULO, 2 * ALTURA_ROTULO];
const RAIO_TOQUE = 18; // px: com o mapa afastado os marcadores são pequenos, por isso o rato "apanha" o mais próximo

function GestorRotulos() {
  const v = useMemo(() => new THREE.Vector3(), []);
  const gl = useThree((s) => s.gl);
  const rato = useRef({ x: 0, y: 0, dentro: false, mexeu: false, botao: false });

  useEffect(() => {
    const el = gl.domElement;
    const mover = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      Object.assign(rato.current, { x: e.clientX - r.left, y: e.clientY - r.top, dentro: true, mexeu: true, botao: e.buttons !== 0 });
    };
    const sair = () => Object.assign(rato.current, { dentro: false, mexeu: true });
    el.addEventListener("pointermove", mover);
    el.addEventListener("pointerleave", sair);
    return () => {
      el.removeEventListener("pointermove", mover);
      el.removeEventListener("pointerleave", sair);
    };
  }, [gl]);

  // corre depois dos marcadores (subscreve-se depois deles); prioridade > 0 desligaria o desenho automático
  useFrame(({ camera, size }) => {
    const s = useAtlas.getState();
    const cand: { r: Registo; x: number; y: number; w: number; prio: number }[] = [];
    let foco: { x: number; y: number } | null = null;
    const proj = (r: Registo) => {
      v.set(0, 1.6, 0);
      r.grupo.localToWorld(v);
      const d = camera.position.distanceTo(v);
      v.project(camera);
      return { x: (v.x * 0.5 + 0.5) * size.width, y: (-v.y * 0.5 + 0.5) * size.height, d, frente: v.z < 1 };
    };
    // monumento sob o rato: o marcador mais próximo do cursor no ecrã, dentro de RAIO_TOQUE
    const r0 = rato.current;
    if (r0.mexeu && !r0.botao) {
      r0.mexeu = false;
      let melhor: string | null = null, dmin = RAIO_TOQUE;
      if (r0.dentro)
        for (const r of registos.values()) {
          if (!r.grupo.visible) continue;
          v.set(0, 0.8, 0);
          r.grupo.localToWorld(v).project(camera);
          if (v.z > 1) continue;
          const dd = Math.hypot((v.x * 0.5 + 0.5) * size.width - r0.x, (-v.y * 0.5 + 0.5) * size.height - r0.y);
          if (dd < dmin) [dmin, melhor] = [dd, r.m.id];
        }
      if (melhor !== s.sobre) {
        s.setSobre(melhor);
        gl.domElement.style.cursor = melhor ? "pointer" : "";
      }
    }
    const destaque = useAtlas.getState().sobre ?? s.selecionado;
    if (destaque && registos.has(destaque)) {
      const p = proj(registos.get(destaque)!);
      foco = { x: p.x, y: p.y };
    }
    const sobre = useAtlas.getState().sobre;
    for (const r of registos.values()) {
      const ativo = r.m.id === sobre || r.m.id === s.selecionado;
      if (!r.grupo.visible) {
        r.el.dataset.visivel = "0";
        continue;
      }
      const p = proj(r);
      if (!p.frente) {
        r.el.dataset.visivel = "0";
        continue;
      }
      const vizinho = foco && Math.hypot(p.x - foco.x, p.y - foco.y) < RAIO_VIZINHOS;
      const perto = p.d < DIST_TODOS;
      if (!ativo && !vizinho && !perto) {
        r.el.dataset.visivel = "0";
        continue;
      }
      const w = ativo ? Math.min(250, Math.max(r.m.nome.length * 8.4, (r.m.localidade.length + TIPOLOGIAS[r.m.tipologia].nome.length + 3) * 7.2) + 24) : r.m.nome.length * 7 + 18;
      const prio = r.m.id === sobre ? 0 : r.m.id === s.selecionado ? 1 : vizinho ? 2 + Math.hypot(p.x - foco!.x, p.y - foco!.y) / 1000 : 3 + p.d / 1e4;
      cand.push({ r, x: p.x, y: p.y, w, prio });
    }
    cand.sort((a, b) => a.prio - b.prio);
    const ocupados: [number, number, number, number][] = [];
    const livre = (x0: number, y0: number, x1: number, y1: number) =>
      ocupados.every(([a, b, c, d]) => x1 < a || x0 > c || y1 < b || y0 > d);
    for (const c of cand) {
      const ativo = c.prio < 2;
      const tentativas = c.prio < 3 ? DESLOCAMENTOS : [0];
      let colocado = false;
      for (const dy of tentativas) {
        // a cartela fica acima da torre: maior (nome e localidade) para o monumento ativo
        const alto = ativo ? 48 : ALTURA_ROTULO;
        const x0 = c.x - c.w / 2, y0 = c.y + dy - (ativo ? 22 : 14) - alto;
        const x1 = x0 + c.w, y1 = y0 + alto;
        if (ativo || livre(x0, y0, x1, y1)) {
          ocupados.push([x0 - 2, y0 - 1, x1 + 2, y1 + 1]);
          c.r.el.style.setProperty("--dy", `${dy}px`);
          colocado = true;
          break;
        }
      }
      c.r.el.dataset.visivel = colocado ? "1" : "0";
      c.r.el.dataset.ativo = ativo ? "1" : "0";
    }
  });
  return null;
}

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
      <GestorRotulos />
    </group>
  );
}

function Marcador({ m, x, z }: { m: MonumentoResumo; x: number; z: number }) {
  const ref = useRef<THREE.Group>(null);
  const anel = useRef<THREE.Mesh>(null);
  const rotulo = useRef<HTMLDivElement>(null);
  const terreno = useTerreno();
  const forma = FORMAS[TIPOLOGIAS[m.tipologia].forma === "torre" ? "torre" : "baluarte"];
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
    const escala = Math.min(d * 0.018, 34) * (ativo ? 1.25 : 1);
    g.scale.setScalar(escala);
    if (anel.current) anel.current.visible = s.selecionado === m.id;
  });

  // o <Html> monta o rótulo noutra raiz React, por isso o registo só é possível quando ele já existe
  useFrame(() => {
    if (ref.current && rotulo.current && registos.get(m.id)?.el !== rotulo.current)
      registos.set(m.id, { m, grupo: ref.current, el: rotulo.current });
  });
  useEffect(() => () => void registos.delete(m.id), [m.id]);

  const click = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    useAtlas.getState().selecionar(m.id);
  };

  return (
    <group ref={ref} onClick={click}>
      <mesh geometry={GEO_BASE} material={MAT_BASE} castShadow />
      <mesh geometry={forma.pedra} material={MAT_PEDRA} castShadow />
      <mesh geometry={forma.mastro} material={MAT_BASE} />
      <mesh geometry={forma.bandeira} material={MAT_BANDEIRA} />
      <mesh ref={anel} geometry={GEO_ANEL} material={MAT_ANEL} visible={false} renderOrder={2} />
      {/* área de toque mais generosa, invisível */}
      <mesh position={[0, 0.8, 0]} visible={false}>
        <sphereGeometry args={[0.7, 8, 8]} />
      </mesh>
      <Html position={[0, forma.topo, 0]} center zIndexRange={[20, 10]} style={{ pointerEvents: "none" }}>
        <div ref={rotulo} className="etiqueta" data-visivel="0" data-ativo="0">
          <b>{m.nome}</b>
          <span>{m.localidade} · {TIPOLOGIAS[m.tipologia].nome}</span>
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
