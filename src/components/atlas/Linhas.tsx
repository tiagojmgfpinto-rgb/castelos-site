import { Line } from "@react-three/drei";
import { useEffect, useMemo, useState } from "react";
import { BASE } from "../../lib/base";
import { centro, TERRITORIOS, type TerritorioId } from "../../lib/territorios";
import { EXAGERO } from "./Terrain";

type P = [number, number, number];
interface Dados {
  costa: P[][];
  fronteira?: P[][];
  costaEspanha?: P[][];
  distritos: P[][];
  nomesDistritos: { nome: string; x: number; y: number }[];
  rios: { nome: string; pontos: P[] }[];
}

const ELEVAR = 0.06; // km acima do terreno, para as linhas não se enterrarem

export function Linhas({ territorio }: { territorio: TerritorioId }) {
  const [dados, setDados] = useState<Dados | null>(null);
  useEffect(() => {
    setDados(null);
    fetch(`${BASE}data/${territorio}/linhas.json`).then((r) => r.json()).then(setDados).catch(console.error);
  }, [territorio]);

  const conv = useMemo(() => {
    const { cx, cy } = centro(TERRITORIOS[territorio]);
    return (l: P[]) => l.map(([x, y, h]) => [(x - cx) / 1000, (h / 1000) * EXAGERO + ELEVAR, -(y - cy) / 1000] as [number, number, number]);
  }, [territorio]);

  if (!dados) return null;
  return (
    <group>
      {dados.costa.map((l, i) => (
        <Line key={`c${i}`} points={conv(l)} color="#4a3a28" lineWidth={1.1} transparent opacity={0.75} />
      ))}
      {dados.costaEspanha?.map((l, i) => (
        <Line key={`ce${i}`} points={conv(l)} color="#6d6a62" lineWidth={0.8} transparent opacity={0.5} />
      ))}
      {dados.fronteira?.map((l, i) => (
        <Line key={`f${i}`} points={conv(l)} color="#2e2014" lineWidth={2.4} />
      ))}
      {dados.distritos.map((l, i) => (
        <Line key={`d${i}`} points={conv(l)} color="#3b2c1c" lineWidth={1} dashed dashSize={2.2} gapSize={1.4} transparent opacity={0.7} />
      ))}
      {dados.rios.map((r, i) => (
        <Line key={`r${i}`} points={conv(r.pontos)} color="#3f7fa6" lineWidth={1.3} transparent opacity={0.9} />
      ))}
    </group>
  );
}

export { type Dados as DadosLinhas };
