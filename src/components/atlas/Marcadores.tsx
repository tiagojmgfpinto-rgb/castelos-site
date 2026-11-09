import { Html } from "@react-three/drei";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { CIDADES } from "../../data/cidades";
import { paraCena, projectar, TERRITORIOS, type TerritorioId } from "../../lib/territorios";
import { TIPOLOGIAS } from "../../lib/tipologias";
import { caixasMonumentos, escolherRotulos, registar, retirar, rotulosMapa, sobrepoe, zonasInterface, type Caixa, type Candidato, type RotuloMapa } from "./rotulosMapa";
import { agrupar, MAX_RODA, posicoesRoda } from "./roda";
import { useAtlas, visivel, type MonumentoResumo } from "./store";
import { exagero, useTerreno } from "./Terrain";

// Marcadores como pequenas figuras brancas, à maneira do castillos.site: um ícone por tipo de fortificação
// (torre ameada com pendão para os castelos, pano de muralha entre duas torres para as fortalezas e praças
// abaluartadas, estrela de baluartes com guarita para os fortes costeiros, torre esbelta para as torres),
// brancas com alguma luz própria e um contorno escuro fino, que as distingue do relevo claro.
const MAT = {
  pedra: new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.7, flatShading: true, emissive: "#ffffff", emissiveIntensity: 0.3 }),
  peana: new THREE.MeshStandardMaterial({ color: "#f4f2ec", roughness: 0.5, emissive: "#ffffff", emissiveIntensity: 0.22 }),
  contorno: new THREE.MeshBasicMaterial({ color: "#2b1f14", side: THREE.BackSide }),
  pendao: new THREE.MeshStandardMaterial({ color: "#b4311f", roughness: 0.8, side: THREE.DoubleSide, emissive: "#7a1a10", emissiveIntensity: 0.35 }),
  escuro: new THREE.MeshBasicMaterial({ color: "#2b1f14" }),
  sombra: new THREE.MeshBasicMaterial({
    map: (() => {
      if (typeof document === "undefined") return null;
      const cv = document.createElement("canvas");
      cv.width = cv.height = 64;
      const g = cv.getContext("2d")!;
      const grad = g.createRadialGradient(32, 32, 4, 32, 32, 32);
      grad.addColorStop(0, "rgba(20,12,4,.55)");
      grad.addColorStop(1, "rgba(20,12,4,0)");
      g.fillStyle = grad;
      g.fillRect(0, 0, 64, 64);
      return new THREE.CanvasTexture(cv);
    })(),
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -4,
    polygonOffsetUnits: -4,
  }),
};
const MAT_ANEL = new THREE.MeshBasicMaterial({ color: "#c23a27", transparent: true, opacity: 0.85, depthWrite: false });

// as figuras desenham-se numa escala própria (torre do castelo = 2,95) e reduzem-se a 0,55 no fim
const ESCALA_ICONE = 0.55;
const em = (g: THREE.BufferGeometry, x: number, y: number, z: number) => g.translate(x, y, z);
const juntar = (lista: THREE.BufferGeometry[]) =>
  mergeGeometries(
    lista.map((g) => {
      const n = g.index ? g.toNonIndexed() : g;
      n.deleteAttribute("uv");
      n.clearGroups();
      return n;
    }),
  ).scale(ESCALA_ICONE, ESCALA_ICONE, ESCALA_ICONE);
const peana = (sx = 1, sz = 1) => new THREE.CylinderGeometry(0.66, 0.74, 0.16, 24).translate(0, 0.08, 0).scale(sx, 1, sz);
/** Vão de porta em arco de volta perfeita, a escuro. */
function porta(x: number, y: number, z: number, largura: number, altura: number, rotY = 0) {
  const r = largura / 2;
  return [
    new THREE.BoxGeometry(largura, altura - r, 0.04).translate(0, (altura - r) / 2, 0),
    new THREE.CylinderGeometry(r, r, 0.04, 14).rotateX(Math.PI / 2).translate(0, altura - r, 0),
  ].map((p) => p.rotateY(rotY).translate(x, y, z));
}

interface Icone {
  pedra: THREE.BufferGeometry;
  peana: THREE.BufferGeometry;
  escuro?: THREE.BufferGeometry;
  pendao?: THREE.BufferGeometry;
  alto: number; // altura da figura, já reduzida
  deFrente?: boolean; // figura plana: roda para olhar sempre para a câmara
}

function iconeCastelo(): Icone {
  const pedra = [
    em(new THREE.CylinderGeometry(0.4, 0.5, 1.35, 10), 0, 0.84, 0), // fuste em talude
    em(new THREE.CylinderGeometry(0.57, 0.41, 0.2, 10), 0, 1.61, 0), // balcão de matacães
    em(new THREE.CylinderGeometry(0.57, 0.57, 0.2, 10), 0, 1.81, 0), // adarve
  ];
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    pedra.push(em(new THREE.BoxGeometry(0.26, 0.24, 0.17).rotateY(-a + Math.PI / 2), Math.cos(a) * 0.48, 2.03, Math.sin(a) * 0.48));
  }
  const peanaEMastro = [peana(), em(new THREE.CylinderGeometry(0.028, 0.028, 0.95, 6), 0, 2.4, 0), em(new THREE.SphereGeometry(0.06, 8, 6), 0, 2.9, 0)];
  const s = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(0.72, -0.19), new THREE.Vector2(0, -0.38)]);
  return { pedra: juntar(pedra), peana: juntar(peanaEMastro), pendao: juntar([em(new THREE.ShapeGeometry(s), 0.03, 2.84, 0)]), alto: 2.95 * ESCALA_ICONE };
}

function iconeFortaleza(): Icone {
  const pedra = [
    em(new THREE.BoxGeometry(0.86, 0.98, 0.3), 0, 0.65, 0), // pano de muralha
    em(new THREE.BoxGeometry(0.44, 1.46, 0.44), -0.64, 0.89, 0),
    em(new THREE.BoxGeometry(0.44, 1.46, 0.44), 0.64, 0.89, 0),
    em(new THREE.BoxGeometry(0.52, 0.08, 0.52), -0.64, 1.66, 0),
    em(new THREE.BoxGeometry(0.52, 0.08, 0.52), 0.64, 1.66, 0),
  ];
  for (const x of [-0.84, -0.64, -0.44, 0.44, 0.64, 0.84]) pedra.push(em(new THREE.BoxGeometry(0.12, 0.2, 0.5), x, 1.8, 0));
  for (const x of [-0.28, 0, 0.28]) pedra.push(em(new THREE.BoxGeometry(0.14, 0.18, 0.3), x, 1.23, 0));
  return {
    pedra: juntar(pedra),
    peana: juntar([peana(1.45, 0.8)]),
    escuro: juntar([...porta(0, 0.16, 0.16, 0.26, 0.5), ...porta(0, 0.16, -0.16, 0.26, 0.5)]),
    alto: 1.92 * ESCALA_ICONE,
    deFrente: true,
  };
}

function iconeForte(): Icone {
  // estrela de quatro baluartes, rasa, com uma guarita ao centro
  const s = new THREE.Shape();
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2 + Math.PI / 4;
    const r = i % 2 === 0 ? 0.78 : 0.42;
    i === 0 ? s.moveTo(Math.cos(a) * r, Math.sin(a) * r) : s.lineTo(Math.cos(a) * r, Math.sin(a) * r);
  }
  const pedra = [
    new THREE.ExtrudeGeometry(s, { depth: 0.42, bevelEnabled: false }).rotateX(-Math.PI / 2).translate(0, 0.16, 0),
    em(new THREE.CylinderGeometry(0.2, 0.22, 0.62, 10), 0, 0.89, 0), // guarita
    em(new THREE.ConeGeometry(0.27, 0.36, 10), 0, 1.38, 0),
  ];
  return { pedra: juntar(pedra), peana: juntar([peana(1.05, 1.05)]), escuro: juntar(porta(0, 0.6, 0.2, 0.12, 0.24)), alto: 1.6 * ESCALA_ICONE };
}

function iconeTorre(): Icone {
  const pedra = [em(new THREE.CylinderGeometry(0.27, 0.31, 2.1, 10), 0, 1.21, 0), em(new THREE.CylinderGeometry(0.37, 0.37, 0.12, 10), 0, 2.32, 0)];
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2;
    pedra.push(em(new THREE.BoxGeometry(0.17, 0.2, 0.12).rotateY(-a + Math.PI / 2), Math.cos(a) * 0.3, 2.48, Math.sin(a) * 0.3));
  }
  return { pedra: juntar(pedra), peana: juntar([peana()]), alto: 2.6 * ESCALA_ICONE };
}

const ICONES = { castelo: iconeCastelo(), fortaleza: iconeFortaleza(), forte: iconeForte(), torre: iconeTorre() };
const ICONE_DA_TIPOLOGIA: Record<string, keyof typeof ICONES> = {
  castelo: "castelo",
  "cerca-urbana": "fortaleza",
  fortaleza: "fortaleza",
  "praca-forte": "fortaleza",
  abaluartada: "fortaleza",
  forte: "forte",
  costeira: "forte",
  bateria: "forte",
  outra: "torre",
};
const GEO_SOMBRA = new THREE.CircleGeometry(1.25 * ESCALA_ICONE, 28).rotateX(-Math.PI / 2).translate(0, 0.01, 0);
const GEO_ANEL = new THREE.RingGeometry(0.92 * ESCALA_ICONE, 1.08 * ESCALA_ICONE, 48).rotateX(-Math.PI / 2).translate(0, 0.02, 0);

// Tamanho no ecrã como no castillos: cresce de 24 px (longe) até 54 px (aproximado) em escala logarítmica da
// distância da câmara ao centro da vista; no telemóvel um pouco mais pequenas ao longe. As distâncias seguem as
// proporções do castillos (vista inicial 118, perto 22, longe 300), a partir da vista inicial de cada território.
const ALTO_PX = 54;
const ALTO_MIN_PX = 24;
function altoNoEcra(d: number, vistaInicial: number, larguraPx: number) {
  const perto = (vistaInicial * 22) / 118;
  if (larguraPx < 700) {
    const t = THREE.MathUtils.clamp(Math.log(d / perto) / Math.log((vistaInicial * 250) / 118 / perto), 0, 1);
    return ALTO_PX + (ALTO_MIN_PX * THREE.MathUtils.clamp(larguraPx / 1100, 0.66, 1) - ALTO_PX) * t;
  }
  const t = THREE.MathUtils.clamp(Math.log(d / perto) / Math.log((vistaInicial * 300) / 118 / perto), 0, 1) ** 1.6;
  return ALTO_PX + (ALTO_MIN_PX - ALTO_PX) * t;
}

// ——— Gestão dos rótulos ———
// Os nomes dos monumentos não aparecem todos (atropelar-se-iam e, ao aproximar de um castelo, enchiam o ecrã
// com os nomes dos outros à volta). Mostra-se o do monumento selecionado e, ao passar o rato sobre uma torre,
// só o dela (pedido do Tiago, 9 nov. 2026: com os vizinhos ficava confuso). Numa roda aberta veem-se todos.
type Registo = { m: MonumentoResumo; grupo: THREE.Group; el: HTMLDivElement; base: THREE.Vector3 };
const registos = new Map<string, Registo>();
// câmara e tamanho do último fotograma, para o toque decidir fora do ciclo de desenho
const vista: { camera?: THREE.Camera; altura: number } = { altura: 1 };
const RAIO_RODA = 26; // px: torres mais perto do que isto umas das outras abrem em roda

/**
 * Toque ou clique numa torre: abre a ficha, ou, se houver outras torres coladas a ela no ecrã, afasta-as
 * todas em roda para se escolher a certa. Dentro de uma roda aberta, o toque abre logo a ficha.
 */
export function tocar(id: string) {
  const s = useAtlas.getState();
  const camera = vista.camera;
  if (s.roda?.ids.includes(id) || !camera) return s.selecionar(id);
  const v = new THREE.Vector3();
  const pontos = [...registos.values()]
    .filter((r) => r.grupo.visible)
    .map((r) => {
      v.copy(r.base).project(camera);
      return { id: r.m.id, x: v.x * 0.5 * vista.altura * (camera as THREE.PerspectiveCamera).aspect, y: -v.y * 0.5 * vista.altura, z: v.z };
    })
    .filter((p) => p.z < 1);
  const ids = agrupar(id, pontos, RAIO_RODA);
  if (ids.length < 2) return s.selecionar(id);
  const centro = new THREE.Vector3();
  for (const i of ids) centro.add(registos.get(i)!.base);
  centro.divideScalar(ids.length);
  s.setSobre(null);
  const distancia = camera.position.distanceTo(centro);
  s.setRoda({ ids, x: centro.x, y: centro.y, z: centro.z, distancia });
  // roda junto à borda do ecrã (ou por baixo da barra): o mapa desliza para a pôr ao centro, à mesma distância
  v.copy(centro).project(camera);
  const margem = (posicoesRoda(ids.length)[0][1] * -1 + 70) / (vista.altura / 2);
  if (Math.abs(v.y) > 1 - margem || Math.abs(v.x) > 1 - margem / (camera as THREE.PerspectiveCamera).aspect) {
    const inclinacao = THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp((camera.position.y - centro.y) / distancia, -1, 1)));
    s.voar({ x: centro.x, z: centro.z, distancia, inclinacao });
  }
}
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
  useFrame(({ camera, size, clock }) => {
    const s = useAtlas.getState();
    vista.camera = camera;
    vista.altura = size.height;
    // aproximar ou afastar bastante fecha a roda: as torres já se separam sozinhas, ou voltam a juntar-se
    if (s.roda) {
      const d = camera.position.distanceTo(v.set(s.roda.x, s.roda.y, s.roda.z));
      if (Math.abs(d / s.roda.distancia - 1) > 0.3) s.setRoda(null);
    }
    const zonas = zonasInterface(gl.domElement, clock.elapsedTime * 1000);
    const cand: { r: Registo; x: number; y: number; w: number; prio: number }[] = [];
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
      if (melhor !== useAtlas.getState().sobre) {
        s.setSobre(melhor);
        gl.domElement.style.cursor = melhor ? "pointer" : "";
      }
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
      // as torres de uma roda aberta mostram todas o nome: é para isso que se afastaram
      const vizinho = !!s.roda?.ids.includes(r.m.id);
      if (!ativo && !vizinho) {
        r.el.dataset.visivel = "0";
        continue;
      }
      const w = ativo ? Math.min(250, Math.max(r.m.nome.length * 8.4, (r.m.localidade.length + TIPOLOGIAS[r.m.tipologia].nome.length + 3) * 7.2) + 24) : r.m.nome.length * 7 + 18;
      const prio = r.m.id === sobre ? 0 : r.m.id === s.selecionado ? 1 : vizinho ? 2 : 3 + p.d / 1e4;
      cand.push({ r, x: p.x, y: p.y, w, prio });
    }
    cand.sort((a, b) => a.prio - b.prio);
    const ocupados: Caixa[] = [];
    // as cartelas que não são do monumento ativo também não entram por baixo da barra, da rosa ou da escala
    const livre = (x0: number, y0: number, x1: number, y1: number) =>
      ocupados.every(([a, b, c, d]) => x1 < a || x0 > c || y1 < b || y0 > d) && !zonas.some((z) => sobrepoe([x0, y0, x1, y1], z));
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
    caixasMonumentos.length = 0;
    caixasMonumentos.push(...ocupados);
  });
  return null;
}

// ——— Rótulos cartográficos (cidades, distritos, mares) ———
const DESVIOS_MAPA = [0, -0.6, 0.6, -1.2, 1.2]; // em alturas do rótulo
// Corre depois do GestorRotulos: os nomes do mapa cedem às cartelas dos monumentos e à interface.
function GestorMapa() {
  const v = useMemo(() => new THREE.Vector3(), []);
  const gl = useThree((s) => s.gl);
  const larguras = useMemo(() => new WeakMap<HTMLElement, [number, number]>(), []);
  useFrame(({ camera, size, clock }) => {
    const proibidas = [...zonasInterface(gl.domElement, clock.elapsedTime * 1000), ...caixasMonumentos];
    const lista: RotuloMapa[] = [];
    const cand: Candidato[] = [];
    const alturas: number[] = [];
    for (const r of rotulosMapa) {
      const i = lista.push(r) - 1;
      r.objeto.getWorldPosition(v);
      const d = camera.position.distanceTo(v);
      v.project(camera);
      if (d >= r.ate || d <= r.desde || v.z > 1) continue;
      const x = (v.x * 0.5 + 0.5) * size.width, y = (-v.y * 0.5 + 0.5) * size.height;
      if (x < -200 || x > size.width + 200 || y < -50 || y > size.height + 50) continue;
      // o tamanho do texto não muda: mede-se uma vez (o elemento invisível continua a ocupar espaço)
      let wh = larguras.get(r.el);
      if (!wh || !wh[0]) {
        wh = [r.el.offsetWidth, r.el.offsetHeight];
        larguras.set(r.el, wh);
      }
      const [w, h] = wh;
      // além do sítio exato, o rótulo pode subir ou descer meia altura ou uma altura inteira
      const caixas = (r.fixo ? [0] : DESVIOS_MAPA).map((f): Caixa => [x - w / 2 - 2, y + f * h - h / 2 - 1, x + w / 2 + 2, y + f * h + h / 2 + 1]);
      const jaVisivel = r.el.dataset.visivel === "1";
      cand.push({ id: i, prio: r.prio, caixas, jaVisivel, anterior: jaVisivel ? Number(r.el.dataset.desvio ?? 0) : undefined });
      alturas[i] = h;
    }
    const escolha = escolherRotulos(cand, proibidas);
    lista.forEach((r, i) => {
      const k = escolha.get(i);
      const novo = k === undefined ? "0" : "1";
      if (r.el.dataset.visivel !== novo) r.el.dataset.visivel = novo;
      if (k !== undefined && r.el.dataset.desvio !== String(k)) {
        r.el.dataset.desvio = String(k);
        r.el.style.transform = k ? `translateY(${DESVIOS_MAPA[k] * alturas[i]}px)` : "";
      }
    });
  });
  return null;
}

export function Marcadores({ monumentos, territorio }: { monumentos: MonumentoResumo[]; territorio: TerritorioId }) {
  const t = TERRITORIOS[territorio];
  const lista = useMemo(
    () =>
      monumentos
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
        <Marcador key={m.id} m={m} x={sx} z={sz} longe={t.vista.distancia} />
      ))}
      <FiosRoda />
      <Cidades territorio={territorio} />
      <GestorRotulos />
      <GestorMapa />
    </group>
  );
}

function Marcador({ m, x, z, longe }: { m: MonumentoResumo; x: number; z: number; longe: number }) {
  const ref = useRef<THREE.Group>(null);
  const corpo = useRef<THREE.Group>(null);
  const contorno = useRef<THREE.Group>(null);
  const pendao = useRef<THREE.Mesh>(null);
  const controls = useThree((st) => st.controls);
  const dir = useMemo(() => new THREE.Vector3(), []);
  const ordem = useMemo(() => [...m.id].reduce((a, c) => a + c.charCodeAt(0), 0) % 7, [m.id]);
  const anel = useRef<THREE.Mesh>(null);
  const rotulo = useRef<HTMLDivElement>(null);
  const terreno = useTerreno();
  const icone = ICONES[ICONE_DA_TIPOLOGIA[m.tipologia] ?? "castelo"];
  const v = useMemo(() => new THREE.Vector3(), []);
  const base = useMemo(() => new THREE.Vector3(), []);
  const aux = useMemo(() => ({ centro: new THREE.Vector3(), direita: new THREE.Vector3(), frente: new THREE.Vector3() }), []);

  useFrame(({ camera, size, clock }) => {
    const g = ref.current;
    if (!g) return;
    const s = useAtlas.getState();
    const selecionado = s.selecionado === m.id;
    const ativo = selecionado || s.sobre === m.id;
    const show = visivel(m, s);
    g.visible = show;
    const h = terreno?.altitude(x, z) ?? 0;
    base.set(x, (Math.max(h, 0) / 1000) * exagero(), z);
    g.position.copy(base);
    const k = s.roda ? s.roda.ids.indexOf(m.id) : -1;
    if (k >= 0 && s.roda) {
      // na roda: o desvio em px no ecrã passa a km no chão, à distância do centro da roda
      const [dx, dy] = posicoesRoda(s.roda.ids.length)[k];
      const { centro, direita, frente } = aux;
      centro.set(s.roda.x, s.roda.y, s.roda.z);
      const d = camera.position.distanceTo(centro);
      const kmPx = (2 * d * Math.tan(((camera as THREE.PerspectiveCamera).fov * Math.PI) / 360)) / size.height;
      direita.setFromMatrixColumn(camera.matrixWorld, 0).setY(0).normalize();
      frente.set(direita.z, 0, -direita.x);
      const cosInc = Math.max(0.35, (camera.position.y - centro.y) / d);
      g.position.copy(centro).addScaledVector(direita, dx * kmPx).addScaledVector(frente, (-dy * kmPx) / cosInc);
    }
    // tamanho no ecrã como no castillos; de longe, mais luz própria e contorno mais grosso para continuarem a
    // ler-se como torres brancas ameadas; a selecionada e a que está sob o rato ficam maiores
    const alvo = (controls as unknown as { target?: THREE.Vector3 } | null)?.target;
    const altoPx = altoNoEcra(alvo ? camera.position.distanceTo(alvo) : camera.position.distanceTo(g.position), longe, size.width);
    const longeK = size.width < 700 ? 0 : THREE.MathUtils.clamp((ALTO_PX - altoPx) / (ALTO_PX - 22), 0, 1);
    MAT.pedra.emissiveIntensity = 0.3 + 0.32 * longeK;
    MAT.peana.emissiveIntensity = 0.22 + 0.25 * longeK;
    camera.getWorldDirection(dir);
    const profundidade = Math.max(0.01, v.copy(g.position).sub(camera.position).dot(dir));
    const pxMundo = (2 * Math.tan(((camera as THREE.PerspectiveCamera).fov * Math.PI) / 360)) / size.height;
    const enfase = selecionado ? 1 : ativo ? 0.6 : 0;
    g.scale.setScalar(((altoPx * pxMundo * profundidade) / ICONES.castelo.alto) * (1 + 0.28 * enfase));
    contorno.current?.scale.set(1.12 + 0.13 * longeK, 1.03 + 0.04 * longeK, 1.12 + 0.13 * longeK);
    // as figuras planas olham sempre para a câmara; o pendão ondeia
    if (icone.deFrente && corpo.current) corpo.current.rotation.y = Math.atan2(camera.position.x - g.position.x, camera.position.z - g.position.z);
    if (pendao.current) pendao.current.rotation.y = Math.sin(clock.elapsedTime * 2.4 + ordem) * 0.22 * (0.4 + enfase);
    if (anel.current) anel.current.visible = selecionado;
  });

  // o <Html> monta o rótulo noutra raiz React, por isso o registo só é possível quando ele já existe
  useFrame(() => {
    if (ref.current && rotulo.current && registos.get(m.id)?.el !== rotulo.current)
      registos.set(m.id, { m, grupo: ref.current, el: rotulo.current, base });
  });
  useEffect(() => () => void registos.delete(m.id), [m.id]);

  const click = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    tocar(m.id);
  };

  return (
    <group ref={ref} onClick={click}>
      <mesh geometry={GEO_SOMBRA} material={MAT.sombra} renderOrder={1} />
      <group ref={corpo}>
        <mesh geometry={icone.peana} material={MAT.peana} />
        <mesh geometry={icone.pedra} material={MAT.pedra} />
        {icone.escuro && <mesh geometry={icone.escuro} material={MAT.escuro} />}
        {icone.pendao && <mesh ref={pendao} geometry={icone.pendao} material={MAT.pendao} />}
        {/* contorno: a mesma figura um pouco maior, vista por dentro */}
        <group ref={contorno} scale={[1.12, 1.03, 1.12]} position={[0, -0.01, 0]}>
          <mesh geometry={icone.pedra} material={MAT.contorno} />
          <mesh geometry={icone.peana} material={MAT.contorno} />
        </group>
      </group>
      <mesh ref={anel} geometry={GEO_ANEL} material={MAT_ANEL} visible={false} renderOrder={2} />
      {/* área de toque mais generosa, invisível */}
      <mesh position={[0, icone.alto / 2, 0]} visible={false}>
        <cylinderGeometry args={[0.55, 0.55, icone.alto * 1.1, 8]} />
      </mesh>
      <Html position={[0, icone.alto, 0]} center zIndexRange={[20, 10]} style={{ pointerEvents: "none" }}>
        <div ref={rotulo} className="etiqueta" data-visivel="0" data-ativo="0">
          <b>{m.nome}</b>
          <span>{m.localidade} · {TIPOLOGIAS[m.tipologia].nome}</span>
        </div>
      </Html>
    </group>
  );
}

/** Fios finos do centro da roda a cada torre afastada, para se perceber de onde veio cada uma. */
function FiosRoda() {
  const geo = useMemo(() => new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAX_RODA * 6), 3)), []);
  const ref = useRef<THREE.LineSegments>(null);
  useEffect(() => () => geo.dispose(), [geo]);
  useFrame(() => {
    const l = ref.current;
    if (!l) return;
    const r = useAtlas.getState().roda;
    l.visible = !!r;
    if (!r) return;
    const a = geo.attributes.position as THREE.BufferAttribute;
    let n = 0;
    for (const id of r.ids) {
      const g = registos.get(id)?.grupo;
      if (!g || !g.visible) continue;
      a.setXYZ(n++, r.x, r.y, r.z);
      a.setXYZ(n++, g.position.x, g.position.y, g.position.z);
    }
    geo.setDrawRange(0, n);
    a.needsUpdate = true;
  });
  return (
    <lineSegments ref={ref} geometry={geo} frustumCulled={false} renderOrder={3} visible={false}>
      <lineBasicMaterial color="#2a1c10" transparent opacity={0.85} depthTest={false} />
    </lineSegments>
  );
}

function Cidades({ territorio }: { territorio: TerritorioId }) {
  const t = TERRITORIOS[territorio];
  const terreno = useTerreno();
  const lista = useMemo(
    () =>
      CIDADES.map((c) => {
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
  const registo = useRef<RotuloMapa | null>(null);
  useEffect(() => () => retirar(registo), []);
  useFrame(() => {
    if (!ref.current) return;
    ref.current.position.set(x, (Math.max(alt(), 0) / 1000) * exagero() + 0.05, z);
    // as capitais de distrito (nível 1) vêm primeiro; as outras cidades só de perto e depois dos distritos
    registar(registo, el.current && { el: el.current, objeto: ref.current, prio: nivel === 1 ? 0 : 3, ate: nivel === 1 ? 1500 : 420, desde: 0, fixo: true });
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
