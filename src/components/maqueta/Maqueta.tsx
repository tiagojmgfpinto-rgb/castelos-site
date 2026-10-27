import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { MeshBVH, acceleratedRaycast } from "three-mesh-bvh";
import { BASE } from "../../lib/base";
import { CAMADAS_TERRENO, ESTADOS, elementosComPlaca, type DadosMaqueta, type ElementoMaqueta } from "./dados";
import { alturaLinha, disporPlacas, type OpcoesDisposicao } from "./placas";

interface Controlos {
  target: THREE.Vector3;
  getAzimuthalAngle(): number;
  addEventListener(t: string, f: () => void): void;
  removeEventListener(t: string, f: () => void): void;
  update(): void;
}

interface Alvo {
  centro: THREE.Vector3;
  topo: THREE.Vector3;
  raio: number;
}

const DIRECAO_INICIAL = new THREE.Vector3(-0.62, 0.6, 0.5).normalize(); // de sudoeste, como as fotografias aéreas
const DESTAQUE = new THREE.Color("#c0602f");
const EIXO_Y = new THREE.Vector3(0, 1, 0);
const VEL_GIRO = (2 * Math.PI) / 90; // uma volta em 90 s
const RETOMA_GIRO = 5; // segundos sem interação até o giro recomeçar
const SEM_PLACA_OCULTA = ["vegetacao"]; // as copas não escondem placas (pisca-pisca entre folhas)

/** Estado de cada placa flutuante, atualizado a cada fotograma sem passar pelo React. */
interface Placa {
  id: string;
  ancora: THREE.Vector3;
  el: HTMLElement;
  cartao: HTMLElement;
  linha: HTMLElement;
  w: number;
  h: number;
  op: number;
  dy: number;
  dx: number;
  inicio: number;
  tapada: boolean;
}

const reduzMovimento = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Visor da maqueta 3D: abre logo com o giro e as placas (a capa fica por trás enquanto carrega),
 * com Giro, Placas, Enquadrar, Ampliar, Terreno e seleção de cada elemento com a sua ficha.
 */
export default function Maqueta({ url, capa, dados, provisoria, onVoltarMapa }: {
  url: string;
  capa?: string;
  dados: DadosMaqueta;
  provisoria: boolean;
  onVoltarMapa?: () => void;
}) {
  const [ampliada, setAmpliada] = useState(false);
  const [giro, setGiro] = useState(() => !reduzMovimento());
  const [rotulos, setRotulos] = useState(true);
  const [terreno, setTerreno] = useState(true);
  const [sel, setSel] = useState<string | null>(null);
  const [notas, setNotas] = useState(false);
  const [pedidoEnquadrar, setPedidoEnquadrar] = useState(0);
  const caixa = useRef<HTMLDivElement>(null);
  const bussola = useRef<HTMLDivElement>(null);
  const escala = useRef<HTMLDivElement>(null);
  const escalaTexto = useRef<HTMLSpanElement>(null);
  const camadaPlacas = useRef<HTMLDivElement>(null);
  const pausa = useRef(false);
  const [carregada, setCarregada] = useState(false);
  const marcarCarregada = useCallback(() => setCarregada(true), []);
  const elementos = useMemo(() => new Map(dados.elementos.map((e) => [e.id, e])), [dados]);
  const elemento = sel ? elementos.get(sel) : undefined;
  const placas = useMemo(() => elementosComPlaca(dados), [dados]);

  const ampliar = useCallback(() => {
    const el = caixa.current;
    if (!el) return;
    if (!ampliada) {
      setAmpliada(true);
      el.requestFullscreen?.().catch(() => undefined);
    } else {
      setAmpliada(false);
      if (document.fullscreenElement) document.exitFullscreen().catch(() => undefined);
    }
  }, [ampliada]);

  useEffect(() => {
    const sair = () => !document.fullscreenElement && setAmpliada(false);
    document.addEventListener("fullscreenchange", sair);
    const tecla = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (sel) setSel(null);
      e.stopPropagation();
    };
    caixa.current?.addEventListener("keydown", tecla);
    const c = caixa.current;
    return () => {
      document.removeEventListener("fullscreenchange", sair);
      c?.removeEventListener("keydown", tecla);
    };
  }, [sel]);

  const botoes: [string, boolean | undefined, () => void][] = [
    ["Giro", giro, () => setGiro((v) => !v)],
    ["Placas", rotulos, () => setRotulos((v) => !v)],
    ["Enquadrar", undefined, () => { setSel(null); setPedidoEnquadrar((n) => n + 1); }],
    [ampliada ? "Reduzir" : "Ampliar", undefined, ampliar],
    ["Terreno", terreno, () => setTerreno((v) => !v)],
  ];

  return (
    <div className={`maqueta${ampliada ? " maqueta--ampliada" : ""}`} ref={caixa} tabIndex={-1}>
      <div className="maqueta-palco">
        {capa && !carregada && <img className="maqueta-capa-fundo" src={`${BASE}${capa}`} alt={`Maqueta 3D de ${dados.titulo}, vista de sudoeste`} />}
        <Canvas shadows={{ type: THREE.PCFShadowMap }} dpr={[1, 2]} camera={{ fov: 32, near: 1, far: 4000, position: [-300, 300, 250] }} gl={{ antialias: true, alpha: true }}>
          <Suspense fallback={null}>
            <Cena
              url={url}
              elementos={elementos}
              sel={sel}
              setSel={setSel}
              giro={giro}
              rotulos={rotulos}
              placas={placas}
              distancia={dados.distancia ?? 300}
              camadaPlacas={camadaPlacas}
              pausa={pausa}
              terreno={terreno}
              pedidoEnquadrar={pedidoEnquadrar}
              bussola={bussola}
              escala={escala}
              escalaTexto={escalaTexto}
              onCarregada={marcarCarregada}
            />
          </Suspense>
        </Canvas>
        <div className="maqueta-placas" ref={camadaPlacas} aria-hidden={!rotulos}>
          {placas.map((e, i) => (
            <div key={e.id} className="maqueta-placa" data-placa={e.id} data-ativo={sel === e.id ? "1" : "0"} style={{ zIndex: placas.length - i }}>
              <span className="maqueta-placa-linha" />
              <button
                className="maqueta-placa-cartao"
                tabIndex={rotulos ? 0 : -1}
                aria-label={`${e.nome}: ver descrição`}
                onPointerEnter={() => (pausa.current = true)}
                onPointerLeave={() => (pausa.current = false)}
                onClick={() => setSel(sel === e.id ? null : e.id)}
              >
                <strong>{e.nome.replace(/ \(.*\)$/, "")}</strong>
                {e.resumo && <em>{e.resumo}</em>}
              </button>
            </div>
          ))}
        </div>
        <div className="maqueta-norte" ref={bussola} aria-hidden="true"><span>N</span></div>
        {!carregada && <p className="maqueta-carregar">A carregar a maqueta…</p>}
        <div className="maqueta-escala" aria-hidden="true"><div ref={escala} /><span ref={escalaTexto}>10 m</span></div>
        {provisoria && (
          <button className="maqueta-selo" onClick={() => setNotas((v) => !v)} aria-expanded={notas}>
            Versão provisória · fontes e limitações
          </button>
        )}
        {ampliada && onVoltarMapa && (
          <button className="maqueta-voltar" onClick={() => { ampliar(); onVoltarMapa(); }}>← Voltar ao mapa</button>
        )}
        {elemento && <FichaElemento e={elemento} dados={dados} onFechar={() => setSel(null)} />}
        {notas && (
          <div className="maqueta-notas" role="dialog" aria-label="Fontes e limitações">
            <button className="fechar" aria-label="Fechar" onClick={() => setNotas(false)}>×</button>
            <h4>Fontes e método</h4>
            <p>
              {dados.titulo}, {ESTADOS[dados.variante as ElementoMaqueta["estado"]]?.toLowerCase() ?? dados.variante}, versão {dados.versao} ({dados.data}).
              Modelado em Blender com Python à escala 1:1, georreferenciado ({dados.origem.lat}, {dados.origem.lon}), norte geográfico.
            </p>
            <ul>{dados.limitacoes.map((l) => <li key={l}>{l}</li>)}</ul>
            <ul className="maqueta-fontes">
              {Object.entries(dados.fontes).map(([k, f]) => <li key={k}>{f.url ? <a href={f.url} target="_blank" rel="noopener">{f.titulo}</a> : f.titulo}</li>)}
            </ul>
          </div>
        )}
      </div>
      <div className="ficha-maqueta-botoes maqueta-botoes" role="toolbar" aria-label="Controlos da maqueta">
        {botoes.map(([nome, ativo, f]) => (
          <button key={nome} aria-pressed={ativo} onClick={f}>{nome}</button>
        ))}
      </div>
      <p className="maqueta-ajuda">Arraste para rodar · roda ou dois dedos para aproximar · botão direito para deslocar · clique numa placa ou num elemento para o identificar.</p>
    </div>
  );
}

function FichaElemento({ e, dados, onFechar }: { e: ElementoMaqueta; dados: DadosMaqueta; onFechar: () => void }) {
  return (
    <div className="maqueta-elemento" role="dialog" aria-label={e.nome}>
      <button className="fechar" aria-label="Fechar" onClick={onFechar}>×</button>
      <p className="maqueta-elemento-grupo">{e.grupo}</p>
      <h4>{e.nome}</h4>
      <ul className="maqueta-elemento-estado">
        <li>{ESTADOS[e.estado]}</li>
        <li>Confiança {e.confianca}</li>
      </ul>
      <p>{e.descricao}</p>
      {e.medidas && <p className="maqueta-elemento-medidas">{e.medidas}</p>}
      <ul className="maqueta-fontes">
        {e.fontes.map((k) => {
          const f = dados.fontes[k];
          if (!f) return null;
          return <li key={k}>{f.url ? <a href={f.url} target="_blank" rel="noopener">{f.titulo}</a> : f.titulo}</li>;
        })}
      </ul>
    </div>
  );
}

function Cena({ url, elementos, sel, setSel, giro, rotulos, placas, distancia, camadaPlacas, pausa, terreno, pedidoEnquadrar, bussola, escala, escalaTexto, onCarregada }: {
  url: string;
  elementos: Map<string, ElementoMaqueta>;
  sel: string | null;
  setSel: (id: string | null) => void;
  giro: boolean;
  rotulos: boolean;
  placas: ElementoMaqueta[];
  distancia: number;
  camadaPlacas: React.RefObject<HTMLDivElement | null>;
  pausa: React.RefObject<boolean>;
  terreno: boolean;
  pedidoEnquadrar: number;
  bussola: React.RefObject<HTMLDivElement | null>;
  escala: React.RefObject<HTMLDivElement | null>;
  escalaTexto: React.RefObject<HTMLSpanElement | null>;
  onCarregada: () => void;
}) {
  const { scene } = useGLTF(url, `${BASE}draco/`);
  const { camera, size } = useThree();
  const controlos = useRef<Controlos>(null);
  const objetivo = useRef<{ alvo: THREE.Vector3; pos: THREE.Vector3 } | null>(null);
  const rodar = useRef({ v: 0, agarrada: false, ultima: -Infinity });
  const estados = useRef<Placa[]>([]);
  const entrada = useRef<{ atraso: number; passo: number } | null>({ atraso: 0.6, passo: 0.35 });
  const oclusores = useRef<THREE.Object3D[]>([]);
  const aux = useMemo(() => ({ v: new THREE.Vector3(), dir: new THREE.Vector3(), raio: Object.assign(new THREE.Raycaster(), { firstHitOnly: true }) }), []);

  // Preparação: sombras, materiais próprios por elemento (para o destaque) e caixas de cada elemento.
  const { alvos, centro, terreno: area, porElemento, malhas, idDe } = useMemo(() => {
    const porElemento = new Map<string, THREE.Mesh[]>();
    const malhas: THREE.Mesh[] = [];
    const idDe = new Map<THREE.Object3D, string>();
    scene.updateMatrixWorld(true);
    scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      malhas.push(m);
      const terrenoBase = /terreno|base/i.test(m.name) || /^(Terreno|Base)$/.test(m.parent?.name ?? "");
      m.castShadow = !terrenoBase;
      m.receiveShadow = true;
      let p: THREE.Object3D | null = m;
      while (p && !p.userData.elemento) p = p.parent;
      const id = p?.userData.elemento as string | undefined;
      if (!id) return;
      idDe.set(m, id);
      m.material = Array.isArray(m.material) ? m.material.map((x) => x.clone()) : m.material.clone();
      porElemento.set(id, [...(porElemento.get(id) ?? []), m]);
    });
    const alvos = new Map<string, Alvo>();
    for (const [id, meshes] of porElemento) {
      const b = new THREE.Box3();
      meshes.forEach((m) => b.expandByObject(m));
      const c = b.getCenter(new THREE.Vector3());
      const s = b.getSize(new THREE.Vector3());
      alvos.set(id, { centro: c, topo: new THREE.Vector3(c.x, b.max.y + 1.5, c.z), raio: Math.max(s.x, s.z, s.y) / 2 });
    }
    // O enquadramento inicial abrange as muralhas e os elementos com placa (em Guimarães, também o Paço)
    const muralhas = scene.getObjectByName("Muralhas_grupo");
    const bm = new THREE.Box3().setFromObject(muralhas ?? scene);
    for (const e of placas) porElemento.get(e.id)?.forEach((m) => bm.expandByObject(m));
    const centro = bm.getCenter(new THREE.Vector3());
    // A área da maqueta (terreno) limita o alvo da câmara e a caixa das sombras
    const terreno = new THREE.Box3().setFromObject(scene.getObjectByName("Terreno") ?? scene);
    // Índice espacial para os raios de oclusão das placas (e dos cliques): o terreno tem 140 mil triângulos
    for (const m of malhas) {
      if (SEM_PLACA_OCULTA.includes(idDe.get(m) ?? "")) continue;
      const g = m.geometry as THREE.BufferGeometry & { boundsTree?: MeshBVH };
      g.boundsTree ??= new MeshBVH(g);
      m.raycast = acceleratedRaycast;
    }
    return { alvos, centro, terreno, porElemento, malhas, idDe };
  }, [scene, placas]);

  // Ponto de cada placa: o indicado nos dados ou o topo do elemento junto ao centro da sua planta
  const ancoras = useMemo(() => {
    const r = new Map<string, THREE.Vector3>();
    for (const e of placas) {
      if (e.ancora) { r.set(e.id, new THREE.Vector3(...e.ancora)); continue; }
      const a = alvos.get(e.id);
      const meshes = porElemento.get(e.id);
      if (!a || !meshes) continue;
      const pts: THREE.Vector3[] = [];
      for (const m of meshes) {
        const pos = m.geometry.getAttribute("position");
        for (let i = 0; i < pos.count; i++) pts.push(new THREE.Vector3().fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld));
      }
      const dh = (p: THREE.Vector3, q: THREE.Vector3) => Math.hypot(p.x - q.x, p.z - q.z);
      const perto = pts.reduce((b, p) => (dh(p, a.centro) < dh(b, a.centro) ? p : b), pts[0]);
      const topo = pts.filter((p) => dh(p, perto) < 2.5).reduce((y, p) => Math.max(y, p.y), perto.y);
      r.set(e.id, new THREE.Vector3(perto.x, topo + 0.6, perto.z));
    }
    return r;
  }, [placas, alvos, porElemento]);

  useEffect(() => {
    const camada = camadaPlacas.current;
    if (!camada) return;
    estados.current = placas.flatMap((e) => {
      const el = camada.querySelector<HTMLElement>(`[data-placa="${e.id}"]`);
      const ancora = ancoras.get(e.id);
      if (!el || !ancora) return [];
      return [{ id: e.id, ancora, el, cartao: el.querySelector<HTMLElement>(".maqueta-placa-cartao")!, linha: el.querySelector<HTMLElement>(".maqueta-placa-linha")!, w: 0, h: 0, op: 0, dy: 0, dx: 0, inicio: Infinity, tapada: false }];
    });
    entrada.current = { atraso: 0.6, passo: 0.35 };
  }, [placas, ancoras, camadaPlacas]);

  // Ao voltar a ligar as placas, entram de novo em sequência, mais depressa
  const primeira = useRef(true);
  useEffect(() => {
    if (primeira.current) { primeira.current = false; return; }
    if (rotulos) entrada.current = { atraso: 0, passo: 0.08 };
  }, [rotulos]);

  // Remedir os cartões quando o visor muda de tamanho
  useEffect(() => {
    estados.current.forEach((p) => (p.w = 0));
  }, [size.width, size.height]);

  // Depois de fechar a ficha de um elemento, o giro espera como depois de uma interação
  useEffect(() => {
    if (!sel) rodar.current.ultima = performance.now() / 1000;
  }, [sel]);

  const enquadrar = useCallback((instantaneo = false) => {
    const pos = centro.clone().addScaledVector(DIRECAO_INICIAL, distancia);
    if (instantaneo && controlos.current) {
      camera.position.copy(pos);
      controlos.current.target.copy(centro);
      controlos.current.update();
    } else {
      objetivo.current = { alvo: centro.clone(), pos };
    }
  }, [camera, centro, distancia]);

  useEffect(() => {
    enquadrar(true);
    onCarregada();
  }, [enquadrar, onCarregada]);
  useEffect(() => {
    if (pedidoEnquadrar) enquadrar();
  }, [pedidoEnquadrar, enquadrar]);

  // Aproximar suavemente ao elemento selecionado, mantendo o rumo atual da câmara
  useEffect(() => {
    porElemento.forEach((meshes, id) => {
      for (const m of meshes) {
        for (const mat of (Array.isArray(m.material) ? m.material : [m.material]) as THREE.MeshStandardMaterial[]) {
          mat.emissive?.copy(id === sel ? DESTAQUE : new THREE.Color(0));
          mat.emissiveIntensity = id === sel ? 0.28 : 0;
        }
      }
    });
    const a = sel ? alvos.get(sel) : undefined;
    if (!a || !controlos.current) return;
    const dir = camera.position.clone().sub(controlos.current.target).normalize();
    const dist = THREE.MathUtils.clamp(a.raio * 3.6, 60, 260);
    objetivo.current = { alvo: a.centro.clone(), pos: a.centro.clone().addScaledVector(dir, dist) };
  }, [sel, alvos, porElemento, camera]);

  useEffect(() => {
    CAMADAS_TERRENO.forEach((n) => {
      const o = scene.getObjectByName(n);
      if (o) o.visible = terreno;
    });
    const base = scene.getObjectByName("Base");
    if (base) base.visible = terreno;
    const visivel = (o: THREE.Object3D | null): boolean => !o || (o.visible && visivel(o.parent));
    oclusores.current = malhas.filter((m) => visivel(m) && !SEM_PLACA_OCULTA.includes(idDe.get(m) ?? ""));
  }, [terreno, scene, malhas, idDe]);

  useEffect(() => {
    const c = controlos.current;
    if (!c) return;
    const agarrar = () => {
      objetivo.current = null;
      rodar.current.agarrada = true;
      rodar.current.v = 0;
    };
    const largar = () => {
      rodar.current.agarrada = false;
      rodar.current.ultima = performance.now() / 1000;
    };
    c.addEventListener("start", agarrar);
    c.addEventListener("end", largar);
    return () => {
      c.removeEventListener("start", agarrar);
      c.removeEventListener("end", largar);
    };
  }, []);

  useFrame((estado, dtBruto) => {
    const c = controlos.current;
    if (!c) return;
    const dt = Math.min(dtBruto, 0.1);
    // Giro automático: arranca devagar, para ao agarrar a maqueta e retoma após uns segundos parada
    const g = rodar.current;
    const agora = performance.now() / 1000;
    const quer = giro && !g.agarrada && !pausa.current && !sel && !objetivo.current && agora - g.ultima > RETOMA_GIRO;
    g.v += ((quer ? VEL_GIRO : 0) - g.v) * (1 - Math.exp(-dt * (quer ? 0.7 : 4)));
    if (g.v > 1e-5) {
      aux.v.copy(camera.position).sub(c.target).applyAxisAngle(EIXO_Y, g.v * dt);
      camera.position.copy(c.target).add(aux.v);
    }
    const o = objetivo.current;
    if (o) {
      const k = 1 - Math.exp(-dt * 4);
      camera.position.lerp(o.pos, k);
      c.target.lerp(o.alvo, k);
      if (camera.position.distanceTo(o.pos) < 0.3 && c.target.distanceTo(o.alvo) < 0.1) objetivo.current = null;
    }
    // manter o alvo dentro da maqueta
    c.target.x = THREE.MathUtils.clamp(c.target.x, area.min.x + 40, area.max.x - 40);
    c.target.z = THREE.MathUtils.clamp(c.target.z, area.min.z + 40, area.max.z - 40);
    c.target.y = THREE.MathUtils.clamp(c.target.y, centro.y - 40, centro.y + 40);
    if (bussola.current) bussola.current.style.transform = `rotate(${THREE.MathUtils.radToDeg(c.getAzimuthalAngle())}deg)`;
    if (escala.current) {
      const d = camera.position.distanceTo(c.target);
      const pxPorM = size.height / (2 * d * Math.tan(THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov / 2)));
      const m = [1, 2, 5, 10, 20, 50, 100, 200].filter((v) => v * pxPorM <= 140).pop() ?? 1;
      escala.current.style.width = `${m * pxPorM}px`;
      if (escalaTexto.current) escalaTexto.current.textContent = `${m} m`;
    }
    atualizarPlacas(estado.clock.elapsedTime, dt);
  });

  function atualizarPlacas(t: number, dt: number) {
    const ps = estados.current;
    if (!ps.length) return;
    const e = entrada.current;
    if (e) {
      ps.forEach((p, i) => (p.inicio = t + e.atraso + i * e.passo));
      entrada.current = null;
    }
    const o: OpcoesDisposicao = { linha: size.width < 560 ? 26 : 42, folga: 6, niveis: 2, topo: 44 };
    const { v, dir, raio } = aux;
    const noEcra = ps.map((p) => {
      if (!p.w) { p.w = p.cartao.offsetWidth; p.h = p.cartao.offsetHeight; }
      v.copy(p.ancora).project(camera);
      const x = (v.x + 1) / 2 * size.width;
      const y = (1 - v.y) / 2 * size.height;
      const dentro = v.z < 1 && x > -p.w / 2 && x < size.width + p.w / 2 && y > 0 && y < size.height + 20;
      // Oclusão: um raio da câmara ao ponto da placa; se bater antes noutro elemento, a placa esconde-se
      if (dentro && rotulos) {
        dir.copy(p.ancora).sub(camera.position);
        const dist = dir.length();
        raio.set(camera.position, dir.divideScalar(dist));
        raio.far = dist - 0.8;
        const hit = raio.intersectObjects(oclusores.current, false)[0];
        p.tapada = !!hit && idDe.get(hit.object) !== p.id;
      }
      return { x, y, w: p.w, h: p.h, candidata: rotulos && dentro && !p.tapada && t >= p.inicio };
    });
    const lugares = disporPlacas(noEcra, o);
    const kOp = 1 - Math.exp(-dt * 5);
    const kDy = 1 - Math.exp(-dt * 7);
    ps.forEach((p, i) => {
      const { nivel, lado } = lugares[i];
      const { x, y } = noEcra[i];
      p.op += ((nivel >= 0 ? 1 : 0) - p.op) * kOp;
      if (nivel >= 0) {
        p.dy += (alturaLinha(nivel, p.h, o) - p.dy) * kDy;
        p.dx += (lado * p.w - p.dx) * kDy;
      } else if (p.op < 0.02) {
        p.dy = o.linha * 0.4; // ao reaparecer, a linha cresce de novo
      }
      p.el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
      p.el.style.opacity = p.op.toFixed(3);
      p.el.style.visibility = p.op < 0.02 ? "hidden" : "visible";
      p.linha.style.height = `${p.dy.toFixed(1)}px`;
      p.cartao.style.bottom = `${p.dy.toFixed(1)}px`;
      p.cartao.style.setProperty("--dx", `${p.dx.toFixed(1)}px`);
    });
  }

  const clicar = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > 5) return;
    e.stopPropagation();
    let p: THREE.Object3D | null = e.object;
    while (p && !p.userData.elemento) p = p.parent;
    const id = p?.userData.elemento as string | undefined;
    setSel(id && elementos.has(id) && !["terreno", "vegetacao", "caminhos"].includes(id) ? id : null);
  };

  // A caixa das sombras cobre a área toda da maqueta
  const sol = area.getCenter(new THREE.Vector3());
  const meiaSombra = Math.max(area.max.x - area.min.x, area.max.z - area.min.z) / 2 + 5;

  return (
    <>
      <hemisphereLight args={["#f6f0e2", "#7d6b50", 1.25]} />
      <directionalLight
        position={[sol.x - 160, centro.y + 220, sol.z - 140]}
        intensity={2.4}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-meiaSombra}
        shadow-camera-right={meiaSombra}
        shadow-camera-top={meiaSombra}
        shadow-camera-bottom={-meiaSombra}
        shadow-camera-near={10}
        shadow-camera-far={700}
        shadow-bias={-0.0004}
        shadow-normalBias={0.6}
      >
        <object3D attach="target" position={[sol.x, centro.y, sol.z]} />
      </directionalLight>
      <directionalLight position={[centro.x + 200, centro.y + 120, centro.z + 160]} intensity={0.35} />
      <primitive object={scene} onClick={clicar} onPointerMissed={() => setSel(null)} />
      <OrbitControls
        ref={controlos as never}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={22}
        maxDistance={560}
        minPolarAngle={0.08}
        maxPolarAngle={1.36}
        zoomSpeed={0.9}
      />
    </>
  );
}
