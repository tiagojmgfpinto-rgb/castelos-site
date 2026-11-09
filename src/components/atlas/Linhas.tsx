import { Html, Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { BASE } from "../../lib/base";
import { CIDADES } from "../../data/cidades";
import { centro, MAPA, paraCena, projectar, TERRITORIOS } from "../../lib/territorios";
import { registar, retirar, type RotuloMapa } from "./rotulosMapa";
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

// mapa único: os três territórios partilham a mesma projeção e o mesmo quadrado
const T = TERRITORIOS.continente;
const ARQUIPELAGOS: Record<string, string> = { Açores: "Arquipélago dos Açores", Madeira: "Arquipélago da Madeira" };

export function Linhas() {
  const [dados, setDados] = useState<Dados | null>(null);
  useEffect(() => {
    fetch(`${BASE}data/${MAPA}/linhas.json`).then((r) => r.json()).then(setDados).catch(console.error);
  }, []);

  const conv = useMemo(() => {
    const { cx, cy } = centro(T);
    return (l: P[]) => l.map(([x, y, h]) => [(x - cx) / 1000, (h / 1000) * EXAGERO + ELEVAR, -(y - cy) / 1000] as [number, number, number]);
  }, []);

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
      {/* A fronteira desenha-se logo a seguir ao relevo e sem teste de profundidade: os pontos têm c. 3 km entre si
          e seguem os vales do Minho, do Douro e do Guadiana, onde o relevo mais grosseiro de longe fica acima deles e
          enterrava o traço escuro (só se via o halo). Opaca e sem escrever profundidade, as torres continuam por cima. */}
      {dados.fronteira?.map((l, i) => (
        <Line key={`fh${i}`} points={conv(l)} color={HALO} lineWidth={7.6} depthWrite={false} depthTest={false} renderOrder={-6} />
      ))}
      {dados.fronteira?.map((l, i) => (
        <Line key={`f${i}`} points={conv(l)} color="#1c1209" lineWidth={3.6} depthWrite={false} depthTest={false} renderOrder={-5} />
      ))}
      {/* o distrito com o nome da sua capital já rotulada como cidade grande não se repete;
          as regiões autónomas levam o nome do arquipélago e veem-se de mais longe */}
      {dados.nomesDistritos.filter((d) => !CIDADES.some((c) => c.nivel === 1 && c.nome === d.nome)).map((d) => {
        const { cx, cy } = centro(T);
        const arq = ARQUIPELAGOS[d.nome];
        return <Rotulo key={d.nome} classe="rotulo-distrito" texto={arq ?? d.nome} x={(d.x - cx) / 1000} z={-(d.y - cy) / 1000} ate={arq ? 4000 : 1700} desde={arq ? 60 : 18} />;
      })}
      {MARES.map((m) => {
        const [x, z] = paraCena(T, ...projectar(T, m.lon, m.lat));
        return <Rotulo key={m.nome + m.lon} classe="rotulo-mar" texto={m.nome} x={x} z={z} ate={m.ate} desde={60} mar />;
      })}
    </group>
  );
}

// nomes dos mares, em itálico espaçado sobre a água
const MARES: { nome: string; lon: number; lat: number; ate: number }[] = [
  { nome: "Oceano Atlântico", lon: -10.75, lat: 39.0, ate: 2400 },
  { nome: "Golfo de Cádis", lon: -7.2, lat: 36.55, ate: 2400 },
  { nome: "Oceano Atlântico", lon: -18.5, lat: 36.0, ate: 6000 },
];

/** Rótulo cartográfico fixo no terreno, visível entre duas distâncias da câmara (km). */
function Rotulo({ classe, texto, x, z, ate, desde, mar = false }: { classe: string; texto: string; x: number; z: number; ate: number; desde: number; mar?: boolean }) {
  const ref = useRef<THREE.Group>(null);
  const el = useRef<HTMLDivElement>(null);
  const terreno = useTerreno();
  const registo = useRef<RotuloMapa | null>(null);
  useEffect(() => () => retirar(registo), []);
  useFrame(() => {
    if (!ref.current) return;
    const h = mar ? 0 : Math.max(terreno?.altitude(x, z) ?? 0, 0);
    ref.current.position.set(x, (h / 1000) * EXAGERO + 0.05, z);
    // a visibilidade é decidida pelo GestorMapa (distância, interface e sobreposições): mares 1, distritos 2
    registar(registo, el.current && { el: el.current, objeto: ref.current, prio: mar ? 1 : 2, ate, desde });
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
