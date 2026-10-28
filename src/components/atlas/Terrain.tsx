import { useFrame, useThree } from "@react-three/fiber";
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import * as THREE from "three";
import { BASE } from "../../lib/base";
import type { TerritorioId } from "../../lib/territorios";
import { useAtlas } from "./store";
import { TerrainManager, type Manifest } from "./terrain/TerrainManager";

export const EXAGERO = 2.2;

const TerrenoCtx = createContext<TerrainManager | null>(null);
export const useTerreno = () => useContext(TerrenoCtx);

export function Terreno({ territorio, children }: { territorio: TerritorioId; children?: ReactNode }) {
  const [mgr, setMgr] = useState<TerrainManager | null>(null);
  const setCarregamento = useAtlas((s) => s.setCarregamento);
  const invalidate = useThree((s) => s.invalidate);

  useEffect(() => {
    let vivo = true;
    let m: TerrainManager | null = null;
    setMgr(null);
    const base = `${BASE}terrain/${territorio}/`;
    fetch(`${base}manifest.json`)
      .then((r) => r.json() as Promise<Manifest>)
      .then(async (manifest) => {
        m = new TerrainManager(manifest, base, EXAGERO);
        let max = 1;
        m.onProgress = (_l, pend) => {
          max = Math.max(max, pend);
          setCarregamento(pend === 0 ? 1 : 1 - pend / (max + 1));
        };
        m.onChange = () => invalidate();
        await m.init();
        if (vivo) setMgr(m);
      })
      .catch((e) => console.error("Falha a carregar o relevo", e));
    return () => {
      vivo = false;
      m?.dispose();
    };
  }, [territorio, setCarregamento, invalidate]);

  useFrame(({ camera }) => {
    mgr?.update(camera);
  });

  if (!mgr) return null;
  return (
    <TerrenoCtx.Provider value={mgr}>
      <primitive object={mgr.group} />
      <Peana lado={mgr.ladoKm} />
      {children}
    </TerrenoCtx.Provider>
  );
}

const ALTURA_PEANA = 16; // km

/** Peana de madeira escura por baixo do relevo, com sombra de contacto na mesa: o mapa é uma maqueta. */
function Peana({ lado }: { lado: number }) {
  const sombra = useMemo(() => {
    const cv = document.createElement("canvas");
    cv.width = cv.height = 256;
    const g = cv.getContext("2d")!;
    const grad = g.createRadialGradient(128, 128, 70, 128, 128, 128);
    grad.addColorStop(0, "rgba(0,0,0,0.62)");
    grad.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, 256, 256);
    return new THREE.CanvasTexture(cv);
  }, []);
  return (
    <group>
      <mesh position={[0, -ALTURA_PEANA / 2 - 1.5, 0]}>
        <boxGeometry args={[lado, ALTURA_PEANA, lado]} />
        <meshStandardMaterial color="#3a2b1d" roughness={0.78} />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position={[0, -ALTURA_PEANA - 1.7, 0]}>
        <planeGeometry args={[lado * 1.5, lado * 1.55]} />
        <meshBasicMaterial map={sombra} transparent depthWrite={false} />
      </mesh>
    </group>
  );
}
