import { useFrame, useThree } from "@react-three/fiber";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
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
      {children}
    </TerrenoCtx.Provider>
  );
}
