import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html, OrbitControls, useGLTF } from "@react-three/drei";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { BASE } from "../../lib/base";
import { CAMADAS_TERRENO, ESTADOS, ROTULADOS, type DadosMaqueta, type ElementoMaqueta } from "./dados";

interface Controlos {
  target: THREE.Vector3;
  autoRotate: boolean;
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

/**
 * Visor da maqueta 3D: capa com "Explorar em 3D" e, ao abrir, a cena navegável com
 * Giro, Rótulos, Enquadrar, Ampliar, Terreno e seleção de cada elemento com a sua ficha.
 */
export default function Maqueta({ url, capa, dados, provisoria, onVoltarMapa }: {
  url: string;
  capa?: string;
  dados: DadosMaqueta;
  provisoria: boolean;
  onVoltarMapa?: () => void;
}) {
  const [aberta, setAberta] = useState(false);
  const [ampliada, setAmpliada] = useState(false);
  const [giro, setGiro] = useState(false);
  const [rotulos, setRotulos] = useState(true);
  const [terreno, setTerreno] = useState(true);
  const [sel, setSel] = useState<string | null>(null);
  const [notas, setNotas] = useState(false);
  const [pedidoEnquadrar, setPedidoEnquadrar] = useState(0);
  const caixa = useRef<HTMLDivElement>(null);
  const bussola = useRef<HTMLDivElement>(null);
  const escala = useRef<HTMLDivElement>(null);
  const escalaTexto = useRef<HTMLSpanElement>(null);
  const [carregada, setCarregada] = useState(false);
  const marcarCarregada = useCallback(() => setCarregada(true), []);
  const elementos = useMemo(() => new Map(dados.elementos.map((e) => [e.id, e])), [dados]);
  const elemento = sel ? elementos.get(sel) : undefined;

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

  if (!aberta) {
    return (
      <div className="maqueta maqueta-capa">
        {capa && <img src={`${BASE}${capa}`} alt={`Maqueta 3D de ${dados.titulo}, vista de sudoeste`} loading="lazy" />}
        {provisoria && <span className="maqueta-selo">Versão provisória</span>}
        <button className="maqueta-explorar" onClick={() => setAberta(true)}>Explorar em 3D</button>
      </div>
    );
  }

  const botoes: [string, boolean | undefined, () => void][] = [
    ["Giro", giro, () => setGiro((v) => !v)],
    ["Rótulos", rotulos, () => setRotulos((v) => !v)],
    ["Enquadrar", undefined, () => { setSel(null); setPedidoEnquadrar((n) => n + 1); }],
    [ampliada ? "Reduzir" : "Ampliar", undefined, ampliar],
    ["Terreno", terreno, () => setTerreno((v) => !v)],
  ];

  return (
    <div className={`maqueta${ampliada ? " maqueta--ampliada" : ""}`} ref={caixa} tabIndex={-1}>
      <div className="maqueta-palco">
        <Canvas shadows={{ type: THREE.PCFShadowMap }} dpr={[1, 2]} camera={{ fov: 32, near: 1, far: 4000, position: [-300, 300, 250] }} gl={{ antialias: true, alpha: true }}>
          <Suspense fallback={null}>
            <Cena
              url={url}
              elementos={elementos}
              sel={sel}
              setSel={setSel}
              giro={giro}
              rotulos={rotulos}
              terreno={terreno}
              pedidoEnquadrar={pedidoEnquadrar}
              bussola={bussola}
              escala={escala}
              escalaTexto={escalaTexto}
              onCarregada={marcarCarregada}
            />
          </Suspense>
        </Canvas>
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
      <p className="maqueta-ajuda">Arraste para rodar · roda ou dois dedos para aproximar · botão direito para deslocar · clique num elemento para o identificar.</p>
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

function Cena({ url, elementos, sel, setSel, giro, rotulos, terreno, pedidoEnquadrar, bussola, escala, escalaTexto, onCarregada }: {
  url: string;
  elementos: Map<string, ElementoMaqueta>;
  sel: string | null;
  setSel: (id: string | null) => void;
  giro: boolean;
  rotulos: boolean;
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

  // Preparação: sombras, materiais próprios por elemento (para o destaque) e caixas de cada elemento.
  const { alvos, centro, porElemento } = useMemo(() => {
    const porElemento = new Map<string, THREE.Mesh[]>();
    scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      const terrenoBase = /terreno|base/i.test(m.name) || /^(Terreno|Base)$/.test(m.parent?.name ?? "");
      m.castShadow = !terrenoBase;
      m.receiveShadow = true;
      let p: THREE.Object3D | null = m;
      while (p && !p.userData.elemento) p = p.parent;
      const id = p?.userData.elemento as string | undefined;
      if (!id) return;
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
    const muralhas = scene.getObjectByName("Muralhas_grupo");
    const bm = new THREE.Box3().setFromObject(muralhas ?? scene);
    const centro = bm.getCenter(new THREE.Vector3());
    return { alvos, centro, porElemento };
  }, [scene]);

  const enquadrar = useCallback((instantaneo = false) => {
    const pos = centro.clone().addScaledVector(DIRECAO_INICIAL, 300);
    if (instantaneo && controlos.current) {
      camera.position.copy(pos);
      controlos.current.target.copy(centro);
      controlos.current.update();
    } else {
      objetivo.current = { alvo: centro.clone(), pos };
    }
  }, [camera, centro]);

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
  }, [terreno, scene]);

  useEffect(() => {
    const c = controlos.current;
    if (!c) return;
    const parar = () => (objetivo.current = null);
    c.addEventListener("start", parar);
    return () => c.removeEventListener("start", parar);
  }, []);

  useFrame((_, dt) => {
    const c = controlos.current;
    if (!c) return;
    const o = objetivo.current;
    if (o) {
      const k = 1 - Math.exp(-dt * 4);
      camera.position.lerp(o.pos, k);
      c.target.lerp(o.alvo, k);
      if (camera.position.distanceTo(o.pos) < 0.3 && c.target.distanceTo(o.alvo) < 0.1) objetivo.current = null;
    }
    // manter o alvo dentro da maqueta
    c.target.x = THREE.MathUtils.clamp(c.target.x, -120, 120);
    c.target.z = THREE.MathUtils.clamp(c.target.z, -120, 120);
    c.target.y = THREE.MathUtils.clamp(c.target.y, centro.y - 40, centro.y + 40);
    if (bussola.current) bussola.current.style.transform = `rotate(${THREE.MathUtils.radToDeg(c.getAzimuthalAngle())}deg)`;
    if (escala.current) {
      const d = camera.position.distanceTo(c.target);
      const pxPorM = size.height / (2 * d * Math.tan(THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov / 2)));
      const m = [1, 2, 5, 10, 20, 50, 100, 200].filter((v) => v * pxPorM <= 140).pop() ?? 1;
      escala.current.style.width = `${m * pxPorM}px`;
      if (escalaTexto.current) escalaTexto.current.textContent = `${m} m`;
    }
  });

  const clicar = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > 5) return;
    e.stopPropagation();
    let p: THREE.Object3D | null = e.object;
    while (p && !p.userData.elemento) p = p.parent;
    const id = p?.userData.elemento as string | undefined;
    setSel(id && elementos.has(id) && !["terreno", "vegetacao", "caminhos"].includes(id) ? id : null);
  };

  return (
    <>
      <hemisphereLight args={["#f6f0e2", "#7d6b50", 1.25]} />
      <directionalLight
        position={[centro.x - 160, centro.y + 220, centro.z - 140]}
        intensity={2.4}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-170}
        shadow-camera-right={170}
        shadow-camera-top={170}
        shadow-camera-bottom={-170}
        shadow-camera-near={10}
        shadow-camera-far={700}
        shadow-bias={-0.0004}
        shadow-normalBias={0.6}
      >
        <object3D attach="target" position={[centro.x, centro.y, centro.z]} />
      </directionalLight>
      <directionalLight position={[centro.x + 200, centro.y + 120, centro.z + 160]} intensity={0.35} />
      <primitive object={scene} onClick={clicar} onPointerMissed={() => setSel(null)} />
      {rotulos &&
        ROTULADOS.filter((id) => alvos.has(id)).map((id) => (
          <Html key={id} position={alvos.get(id)!.topo} center zIndexRange={[20, 0]} className="maqueta-rotulo-ancora">
            <button
              className="maqueta-rotulo"
              data-ativo={sel === id ? "1" : "0"}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => { e.stopPropagation(); setSel(id); }}
            >
              {elementos.get(id)?.nome.replace(/ \(.*\)$/, "")}
            </button>
          </Html>
        ))}
      <OrbitControls
        ref={controlos as never}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        autoRotate={giro}
        autoRotateSpeed={0.6}
        minDistance={22}
        maxDistance={560}
        minPolarAngle={0.08}
        maxPolarAngle={1.36}
        zoomSpeed={0.9}
      />
    </>
  );
}
