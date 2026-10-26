import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { paraCena, projectar, TERRITORIOS, type TerritorioId } from "../../lib/territorios";
import { Camara, vistaInicial } from "./Camara";
import { Linhas } from "./Linhas";
import { Marcadores } from "./Marcadores";
import { useAtlas, type MonumentoResumo } from "./store";
import { Terreno } from "./Terrain";
import { Interface } from "../ui/Interface";

const COR_FUNDO = "#1b1914";

/** Sol baixo de noroeste, como no relevo sombreado clássico; a sombra acompanha a câmara. */
function Luz() {
  const sol = useRef<THREE.DirectionalLight>(null);
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3 } | null;
  const camera = useThree((s) => s.camera);
  const dir = useMemo(() => new THREE.Vector3(-0.55, 0.62, -0.56).normalize(), []);
  useFrame(() => {
    const l = sol.current;
    if (!l || !controls) return;
    const alvo = controls.target;
    const dist = camera.position.distanceTo(alvo);
    // à escala do país o sombreado vem das normais; sombras projetadas só ao aproximar
    l.castShadow = dist < 160;
    const raio = Math.min(Math.max(dist * 1.25, 3), 260);
    l.position.copy(alvo).addScaledVector(dir, raio * 2);
    l.target.position.copy(alvo);
    l.target.updateMatrixWorld();
    const sc = l.shadow.camera;
    sc.left = -raio;
    sc.right = raio;
    sc.top = raio;
    sc.bottom = -raio;
    sc.near = raio * 0.2;
    sc.far = raio * 4;
    sc.updateProjectionMatrix();
    l.shadow.bias = -0.0005;
    l.shadow.normalBias = raio * 0.004;
  });
  return (
    <>
      <hemisphereLight args={["#f3ead6", "#4a4436", 0.85]} />
      <directionalLight ref={sol} color="#fff1da" intensity={2.4} castShadow shadow-mapSize={[4096, 4096]} />
    </>
  );
}

function Mar() {
  return (
    <mesh rotation-x={-Math.PI / 2} position-y={-1.5}>
      <planeGeometry args={[6000, 6000]} />
      <meshBasicMaterial color={COR_FUNDO} />
    </mesh>
  );
}

/** Sincroniza a seleção com o endereço (#guimaraes), como ligação partilhável. */
function useHash(monumentos: MonumentoResumo[]) {
  const selecionado = useAtlas((s) => s.selecionado);
  useEffect(() => {
    const ler = () => {
      const id = decodeURIComponent(location.hash.slice(1));
      const m = monumentos.find((x) => x.id === id);
      if (m) {
        const s = useAtlas.getState();
        if (s.territorio !== m.territorio) s.setTerritorio(m.territorio);
        s.selecionar(m.id);
      }
    };
    ler();
    addEventListener("hashchange", ler);
    return () => removeEventListener("hashchange", ler);
  }, [monumentos]);
  useEffect(() => {
    const novo = selecionado ? `#${selecionado}` : " ";
    if ((location.hash || " ") !== novo) history.replaceState(null, "", selecionado ? novo : location.pathname + location.search);
  }, [selecionado]);
}

/** Ao selecionar um monumento, a câmara voa até ele. */
function useVooSelecao(monumentos: MonumentoResumo[]) {
  const selecionado = useAtlas((s) => s.selecionado);
  const territorio = useAtlas((s) => s.territorio);
  useEffect(() => {
    if (!selecionado) return;
    const m = monumentos.find((x) => x.id === selecionado);
    if (!m || m.territorio !== territorio) return;
    const t = TERRITORIOS[territorio];
    const [x, y] = projectar(t, m.lon, m.lat);
    const [sx, sz] = paraCena(t, x, y);
    // pequeno atraso para o terreno do novo território existir antes do voo
    const id = setTimeout(() => useAtlas.getState().voar({ x: sx, z: sz, distancia: 7, inclinacao: 58 }), 60);
    return () => clearTimeout(id);
  }, [selecionado, territorio, monumentos]);
}

export default function Atlas({ monumentos }: { monumentos: MonumentoResumo[] }) {
  const territorio = useAtlas((s) => s.territorio);
  useHash(monumentos);
  useVooSelecao(monumentos);

  return (
    <div className="atlas">
      <Canvas
        shadows="percentage"
        dpr={[1, 2]}
        camera={{ fov: 32, near: 0.5, far: 6000, position: [0, 500, 400] }}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        onCreated={({ gl, scene }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.05;
          scene.background = new THREE.Color(COR_FUNDO);
          scene.fog = new THREE.Fog(COR_FUNDO, 900, 2600);
        }}
        onPointerMissed={() => useAtlas.getState().setSobre(null)}
      >
        <Cena territorio={territorio} monumentos={monumentos} />
      </Canvas>
      <Interface monumentos={monumentos} vistaInicial={() => useAtlas.getState().voar(vistaInicial(useAtlas.getState().territorio))} />
    </div>
  );
}

function Cena({ territorio, monumentos }: { territorio: TerritorioId; monumentos: MonumentoResumo[] }) {
  return (
    <>
      <Luz />
      <Mar />
      <Terreno key={territorio} territorio={territorio}>
        <Linhas territorio={territorio} />
        <Marcadores monumentos={monumentos} territorio={territorio} />
        <Camara territorio={territorio} />
      </Terreno>
    </>
  );
}
