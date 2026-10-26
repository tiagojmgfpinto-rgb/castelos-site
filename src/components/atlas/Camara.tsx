import { MapControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { MapControls as MapControlsImpl } from "three-stdlib";
import { paraCena, TERRITORIOS, type TerritorioId } from "../../lib/territorios";
import { useAtlas, type Voo } from "./store";
import { EXAGERO, useTerreno } from "./Terrain";

const DEG = Math.PI / 180;

interface Animacao {
  t0: number;
  dur: number;
  de: { x: number; y: number; z: number; r: number; phi: number; theta: number };
  para: { x: number; y: number; z: number; r: number; phi: number; theta: number };
  arco: number;
}

/** Comandos da câmara para os botões da interface. */
export const camaraApi: { zoom?: (fator: number) => void; norte?: () => void } = {};

const suave = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Ângulo de destino pelo caminho mais curto. */
function angMaisPerto(de: number, para: number) {
  let d = (para - de) % (2 * Math.PI);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return de + d;
}

export function vistaInicial(territorio: TerritorioId): Voo {
  const t = TERRITORIOS[territorio];
  const [x, z] = paraCena(t, t.vista.x, t.vista.y);
  return { x, z, distancia: t.vista.distancia, inclinacao: t.vista.inclinacao, rumo: t.vista.rumo };
}

export function Camara({ territorio }: { territorio: TerritorioId }) {
  const controls = useRef<MapControlsImpl>(null);
  const anim = useRef<Animacao | null>(null);
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const size = useThree((s) => s.size);
  const terreno = useTerreno();
  const voo = useAtlas((s) => s.voo);
  const sph = useRef(new THREE.Spherical());
  const off = useRef(new THREE.Vector3());

  const estadoAtual = () => {
    const c = controls.current!;
    off.current.copy(camera.position).sub(c.target);
    sph.current.setFromVector3(off.current);
    return { x: c.target.x, y: c.target.y, z: c.target.z, r: sph.current.radius, phi: sph.current.phi, theta: sph.current.theta };
  };

  const aplicar = (s: Animacao["de"]) => {
    const c = controls.current!;
    c.target.set(s.x, s.y, s.z);
    sph.current.set(s.r, s.phi, s.theta);
    off.current.setFromSpherical(sph.current);
    camera.position.copy(c.target).add(off.current);
    camera.lookAt(c.target);
  };

  // vista inicial em cada mudança de território
  useEffect(() => {
    if (!controls.current) return;
    const v = vistaInicial(territorio);
    aplicar({ x: v.x, y: 0, z: v.z, r: v.distancia, phi: v.inclinacao * DEG, theta: -(v.rumo ?? 0) * DEG });
    controls.current.update();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [territorio]);

  // novo pedido de voo
  useEffect(() => {
    if (!voo || !controls.current) return;
    const de = estadoAtual();
    const alvoY = ((terreno?.altitude(voo.x, voo.z) ?? 0) / 1000) * EXAGERO;
    const theta = voo.rumo === undefined ? de.theta : angMaisPerto(de.theta, -voo.rumo * DEG);
    const para = { x: voo.x, y: Math.max(alvoY, 0), z: voo.z, r: voo.distancia, phi: voo.inclinacao * DEG, theta };
    if (voo.instantaneo) {
      aplicar(para);
      controls.current.update();
      return;
    }
    const viagem = Math.hypot(para.x - de.x, para.z - de.z);
    const maior = Math.max(de.r, para.r);
    // sobe a meio do percurso quando a viagem é longa em relação à altitude: dá a sensação de sobrevoo
    const arco = Math.max(0, Math.min(2.2, viagem / maior - 0.3));
    const dur = Math.min(4.6, Math.max(1.4, 1.2 + 0.7 * Math.log2(1 + viagem / Math.min(de.r, para.r) / 4)));
    anim.current = { t0: performance.now(), dur: dur * 1000, de, para, arco };
    controls.current.enabled = false;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voo?.pedido]);

  useEffect(() => {
    const animar = (para: Partial<Animacao["para"]>, dur = 700) => {
      if (!controls.current) return;
      const de = estadoAtual();
      anim.current = { t0: performance.now(), dur, de, para: { ...de, ...para }, arco: 0 };
      controls.current.enabled = false;
    };
    camaraApi.zoom = (f) => {
      const de = estadoAtual();
      animar({ r: Math.min(1600, Math.max(0.25, de.r * f)) });
    };
    camaraApi.norte = () => {
      const de = estadoAtual();
      animar({ theta: angMaisPerto(de.theta, 0) }, 900);
    };
    return () => {
      camaraApi.zoom = undefined;
      camaraApi.norte = undefined;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useFrame(() => {
    const c = controls.current;
    if (!c) return;
    const a = anim.current;
    if (a) {
      const t = Math.min(1, (performance.now() - a.t0) / a.dur);
      const e = suave(t);
      const lerp = (p: number, q: number) => p + (q - p) * e;
      // distância interpolada em escala logarítmica, com subida em arco
      const r = Math.exp(lerp(Math.log(a.de.r), Math.log(a.para.r))) * (1 + a.arco * Math.sin(Math.PI * e));
      aplicar({
        x: lerp(a.de.x, a.para.x),
        y: lerp(a.de.y, a.para.y),
        z: lerp(a.de.z, a.para.z),
        r,
        phi: lerp(a.de.phi, a.para.phi),
        theta: lerp(a.de.theta, a.para.theta),
      });
      if (t >= 1) {
        anim.current = null;
        c.enabled = true;
        c.update();
      }
    }
    // planos de recorte e escala adaptados à distância
    const dist = camera.position.distanceTo(c.target);
    camera.near = Math.max(0.01, dist / 400);
    camera.far = Math.max(4000, dist * 8);
    camera.updateProjectionMatrix();
    const kmPorPx = (2 * dist * Math.tan((camera.fov * DEG) / 2)) / size.height;
    off.current.copy(camera.position).sub(c.target);
    const rumo = -Math.atan2(off.current.x, off.current.z) / DEG;
    const s = useAtlas.getState();
    if (Math.abs(s.escalaKmPorPx - kmPorPx) / kmPorPx > 0.02 || Math.abs(s.rumoAtual - rumo) > 0.5) s.setEscala(kmPorPx, rumo);
  });

  return (
    <MapControls
      ref={controls as never}
      makeDefault
      enableDamping
      dampingFactor={0.09}
      screenSpacePanning={false}
      minDistance={0.25}
      maxDistance={1600}
      maxPolarAngle={78 * DEG}
      zoomSpeed={1.1}
      rotateSpeed={0.6}
    />
  );
}
