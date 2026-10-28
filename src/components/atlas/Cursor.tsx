import { useThree } from "@react-three/fiber";
import { useEffect } from "react";
import * as THREE from "three";
import { deCena, TERRITORIOS, type TerritorioId } from "../../lib/territorios";
import { gms } from "../ui/FichaConteudo";
import { useAtlas } from "./store";
import { EXAGERO, useTerreno } from "./Terrain";

/** Coordenadas e altitude sob o rato, para a leitura do canto inferior esquerdo. */
export function Cursor({ territorio }: { territorio: TerritorioId }) {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const terreno = useTerreno();

  useEffect(() => {
    const el = gl.domElement;
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const plano = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const p = new THREE.Vector3();
    let pendente: PointerEvent | null = null;
    let raf = 0;
    const calcular = () => {
      raf = 0;
      const e = pendente;
      if (!e) return;
      const r = el.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      // interseção com o plano à altitude do terreno, refinada duas vezes
      let h = 0;
      for (let i = 0; i < 3; i++) {
        plano.constant = -(Math.max(h, 0) / 1000) * EXAGERO;
        if (!ray.ray.intersectPlane(plano, p)) return;
        h = terreno?.altitude(p.x, p.z) ?? 0;
      }
      const [lon, lat] = deCena(TERRITORIOS[territorio], p.x, p.z);
      const alt = Math.round(Math.max(h, 0));
      useAtlas.getState().setCursor(`${gms(lat, "N", "S")} · ${gms(lon, "E", "O")} · ${alt.toLocaleString("pt-PT")} m`);
    };
    const mover = (e: PointerEvent) => {
      pendente = e;
      if (!raf) raf = requestAnimationFrame(calcular);
    };
    const sair = () => {
      pendente = null;
      useAtlas.getState().setCursor("");
    };
    el.addEventListener("pointermove", mover);
    el.addEventListener("pointerleave", sair);
    return () => {
      el.removeEventListener("pointermove", mover);
      el.removeEventListener("pointerleave", sair);
      cancelAnimationFrame(raf);
    };
  }, [gl, camera, terreno, territorio]);
  return null;
}
