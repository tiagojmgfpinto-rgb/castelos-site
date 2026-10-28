import { Html, Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { BASE } from "../../lib/base";
import { CIDADES } from "../../data/cidades";
import { centro, paraCena, projectar, TERRITORIOS, type TerritorioId } from "../../lib/territorios";
import { EXAGERO, useTerreno } from "./Terrain";

type P = [number, number, number];
interface Dados {
  costa: P[][];
  fronteira?: P[][];
  costaEspanha?: P[][];
  distritos: P[][];
  nomesDistritos: { nome: string; x: number; y: number }[];
  rios: { nome: string; pontos: P[] }[];
}

const HALO = "#fbf1d6";
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
      {/* hierarquia e cores como no castillos.site; o halo claro por baixo faz o traço ler-se igual na várzea e na serra */}
      {dados.costa.map((l, i) => (
        <Line key={`c${i}`} points={conv(l)} color="#274250" lineWidth={1.8} transparent opacity={0.92} />
      ))}
      {dados.costaEspanha?.map((l, i) => (
        <Line key={`ce${i}`} points={conv(l)} color="#4f6470" lineWidth={1} transparent opacity={0.6} />
      ))}
      {dados.rios.map((r, i) => (
        <Line key={`r${i}`} points={conv(r.pontos)} color="#356f94" lineWidth={1.8} transparent opacity={0.92} />
      ))}
      {dados.distritos.map((l, i) => (
        <Line key={`dh${i}`} points={conv(l)} color={HALO} lineWidth={3.6} transparent opacity={0.42} depthWrite={false} />
      ))}
      {dados.distritos.map((l, i) => (
        <Line key={`d${i}`} points={conv(l)} color="#33230f" lineWidth={1.5} dashed dashSize={2.4} gapSize={1.35} transparent opacity={0.9} />
      ))}
      {dados.fronteira?.map((l, i) => (
        <Line key={`fh${i}`} points={conv(l)} color={HALO} lineWidth={6.6} transparent opacity={0.65} depthWrite={false} />
      ))}
      {dados.fronteira?.map((l, i) => (
        <Line key={`f${i}`} points={conv(l)} color="#1c1209" lineWidth={3.1} />
      ))}
      {territorio === "continente" &&
        // o distrito com o nome da sua capital já rotulada como cidade grande não se repete
        dados.nomesDistritos.filter((d) => !CIDADES.some((c) => c.nivel === 1 && c.nome === d.nome)).map((d) => {
          const { cx, cy } = centro(TERRITORIOS[territorio]);
          return <Rotulo key={d.nome} classe="rotulo-distrito" texto={d.nome} x={(d.x - cx) / 1000} z={-(d.y - cy) / 1000} ate={1700} desde={18} />;
        })}
      {MARES[territorio].map((m) => {
        const t = TERRITORIOS[territorio];
        const [x, z] = paraCena(t, ...projectar(t, m.lon, m.lat));
        return <Rotulo key={m.nome} classe="rotulo-mar" texto={m.nome} x={x} z={z} ate={2400} desde={60} mar />;
      })}
    </group>
  );
}

// nomes dos mares, em itálico espaçado sobre a água
const MARES: Record<TerritorioId, { nome: string; lon: number; lat: number }[]> = {
  continente: [{ nome: "Oceano Atlântico", lon: -10.75, lat: 39.0 }, { nome: "Golfo de Cádis", lon: -7.2, lat: 36.55 }],
  acores: [{ nome: "Oceano Atlântico", lon: -27.2, lat: 37.25 }],
  madeira: [{ nome: "Oceano Atlântico", lon: -17.25, lat: 32.45 }],
};

/** Rótulo cartográfico fixo no terreno, visível entre duas distâncias da câmara (km). */
function Rotulo({ classe, texto, x, z, ate, desde, mar = false }: { classe: string; texto: string; x: number; z: number; ate: number; desde: number; mar?: boolean }) {
  const ref = useRef<THREE.Group>(null);
  const el = useRef<HTMLDivElement>(null);
  const terreno = useTerreno();
  useFrame(({ camera }) => {
    if (!ref.current) return;
    const h = mar ? 0 : Math.max(terreno?.altitude(x, z) ?? 0, 0);
    ref.current.position.set(x, (h / 1000) * EXAGERO + 0.05, z);
    const d = camera.position.distanceTo(ref.current.position);
    if (el.current) el.current.dataset.visivel = d < ate && d > desde ? "1" : "0";
  });
  return (
    <group ref={ref}>
      <Html center zIndexRange={[4, 0]} style={{ pointerEvents: "none" }}>
        <div ref={el} className={classe} data-visivel="0">{texto}</div>
      </Html>
    </group>
  );
}

export { type Dados as DadosLinhas };
