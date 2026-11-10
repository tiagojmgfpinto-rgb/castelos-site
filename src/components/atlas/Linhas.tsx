import { Html, Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { BASE } from "../../lib/base";
import { CIDADES } from "../../data/cidades";
import { centro, MAPA, paraCena, projectar, TERRITORIOS } from "../../lib/territorios";
import { registar, retirar, type RotuloMapa } from "./rotulosMapa";
import { EXAGERO, exagero, relevo, useTerreno } from "./Terrain";

type P = [number, number, number];
interface Dados {
  costa: P[][];
  fronteira?: P[][];
  costaEspanha?: P[][];
  distritos: P[][];
  concelhos?: P[][];
  nomesDistritos: { nome: string; x: number; y: number }[];
  ilhas?: Ilha[];
  rios: { nome: string; pontos: P[] }[];
}

interface Ilha { nome: string; arquipelago: string; x: number; y: number; caixa: [number, number, number, number] }

const HALO = "#fbf1d6";
const ELEVAR = 0.06; // km acima do terreno, para as linhas não se enterrarem

// mapa único: os três territórios partilham a mesma projeção e o mesmo quadrado
const T = TERRITORIOS.continente;
// Nomes dos arquipélagos sobre o mar, fora das ilhas e dos nomes delas: o dos Açores a norte do Grupo Central
// (a c. 60 km da Graciosa), o da Madeira a norte da ilha, entre ela e o Porto Santo.
const ARQUIPELAGOS = [
  { nome: "Arquipélago dos Açores", lon: -27.85, lat: 39.62 },
  { nome: "Arquipélago da Madeira", lon: -16.78, lat: 33.32 },
];
// De que lado da ilha fica o nome: do lado do mar aberto e longe da sede de concelho principal (Ponta Delgada,
// Angra, Funchal ficam no sul), e o Corvo e as Flores, a 15 km uma da outra, em lados opostos.
const LADO_ILHA: Record<string, "n" | "s"> = {
  "Santa Maria": "s", "São Miguel": "n", Terceira: "n", Graciosa: "n", "São Jorge": "n", Pico: "s", Faial: "n",
  Flores: "s", Corvo: "n", Madeira: "n", "Porto Santo": "n", "Ilhas Desertas": "s", "Ilhas Selvagens": "n",
};

// Concelhos: traço fino que só aparece de perto (abaixo de 520 km da câmara, todo a partir de 300 km), para o mapa
// do país inteiro não ficar coberto de linhas.
const CONCELHOS_ATE = 520;
const CONCELHOS_TODO = 300;

export function Linhas() {
  const [dados, setDados] = useState<Dados | null>(null);
  useEffect(() => {
    fetch(`${BASE}data/${MAPA}/linhas.json`).then((r) => r.json()).then(setDados).catch(console.error);
  }, []);

  const conv = useMemo(() => {
    const { cx, cy } = centro(T);
    return (l: P[]) => l.map(([x, y, h]) => [(x - cx) / 1000, (h / 1000) * EXAGERO + ELEVAR, -(y - cy) / 1000] as [number, number, number]);
  }, []);

  // as linhas acompanham o relevo quando ele baixa ao aproximar, mantendo a mesma folga sobre o chão
  const grupo = useRef<THREE.Group>(null);
  useFrame(() => {
    const g = grupo.current;
    if (!g || g.scale.y === relevo.escala) return;
    g.scale.y = relevo.escala;
    g.position.y = ELEVAR * (1 - relevo.escala);
  });

  // todos os troços de cada classe numa só linha (pares de pontos): dois ou três desenhos em vez de centenas
  const segmentos = (ls: P[][] | undefined) => {
    if (!ls?.length) return null;
    const out: [number, number, number][] = [];
    for (const l of ls) {
      const c = conv(l);
      for (let i = 1; i < c.length; i++) out.push(c[i - 1], c[i]);
    }
    return out;
  };
  const distritos = useMemo(() => segmentos(dados?.distritos), [dados]);
  const concelhos = useMemo(() => segmentos(dados?.concelhos), [dados]);

  if (!dados) return null;
  return (
    <>
      <group ref={grupo}>
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
        {/* Limites oficiais da CAOP 2025 (DGT). Os distritos são a divisão principal do mapa: traço contínuo escuro sobre
            um halo claro largo, que se lê sobre a serra e sobre a várzea. Os concelhos ficam muito mais discretos.
            Como a fronteira, desenham-se logo a seguir ao relevo e sem teste de profundidade (os limites seguem
            muitas vezes rios e vales, onde o relevo grosseiro de longe os enterrava) e sem escrever profundidade,
            para as torres e os rótulos ficarem sempre por cima. */}
        {concelhos && <Concelhos pontos={concelhos} />}
        {distritos && (
          <>
            <Line points={distritos} segments color={HALO} lineWidth={5.6} transparent opacity={0.6} depthWrite={false} depthTest={false} renderOrder={-8} />
            <Line points={distritos} segments color="#3a2410" lineWidth={2.3} transparent opacity={0.95} depthWrite={false} depthTest={false} renderOrder={-7} />
          </>
        )}
        {/* A fronteira desenha-se logo a seguir ao relevo e sem teste de profundidade: os pontos têm c. 3 km entre si
            e seguem os vales do Minho, do Douro e do Guadiana, onde o relevo mais grosseiro de longe fica acima deles e
            enterrava o traço escuro (só se via o halo). Opaca e sem escrever profundidade, as torres continuam por cima. */}
        {dados.fronteira?.map((l, i) => (
          <Line key={`fh${i}`} points={conv(l)} color={HALO} lineWidth={7.6} depthWrite={false} depthTest={false} renderOrder={-6} />
        ))}
        {dados.fronteira?.map((l, i) => (
          <Line key={`f${i}`} points={conv(l)} color="#1c1209" lineWidth={3.6} depthWrite={false} depthTest={false} renderOrder={-5} />
        ))}
      </group>
      {/* o distrito com o nome da sua capital já rotulada como cidade grande não se repete */}
      {dados.nomesDistritos.filter((d) => !CIDADES.some((c) => c.nivel === 1 && c.nome === d.nome)).map((d) => {
        const { cx, cy } = centro(T);
        return <Rotulo key={d.nome} classe="rotulo-distrito" texto={d.nome} x={(d.x - cx) / 1000} z={-(d.y - cy) / 1000} ate={1700} desde={18} />;
      })}
      {ARQUIPELAGOS.map((a) => {
        const [x, z] = paraCena(T, ...projectar(T, a.lon, a.lat));
        return <Rotulo key={a.nome} classe="rotulo-distrito rotulo-arquipelago" texto={a.nome} x={x} z={z} ate={4000} desde={60} mar prio={1} />;
      })}
      {/* cada ilha com o nome junto à costa do lado do mar aberto (o ponto fica na costa; o nome desce ou sobe
          uma altura e pouco, para não tapar a ilha nem as torres) */}
      {dados.ilhas?.map((i) => {
        const lado = LADO_ILHA[i.nome] ?? "s";
        const [x0, y0, x1, y1] = i.caixa;
        const [x, z] = paraCena(T, (x0 + x1) / 2, lado === "n" ? y1 : y0);
        const desvios = lado === "n" ? [-0.9, -1.5, -2.1] : [0.9, 1.5, 2.1];
        return <Rotulo key={i.nome} classe="rotulo-ilha" texto={i.nome} x={x} z={z} ate={1400} desde={0} mar prio={2} desvios={desvios} />;
      })}
      {MARES.map((m) => {
        const [x, z] = paraCena(T, ...projectar(T, m.lon, m.lat));
        return <Rotulo key={m.nome + m.lon} classe="rotulo-mar" texto={m.nome} x={x} z={z} ate={m.ate} desde={60} mar />;
      })}
    </>
  );
}

// nomes dos mares, em itálico espaçado sobre a água
const MARES: { nome: string; lon: number; lat: number; ate: number }[] = [
  { nome: "Oceano Atlântico", lon: -10.75, lat: 39.0, ate: 2400 },
  { nome: "Golfo de Cádis", lon: -7.2, lat: 36.55, ate: 2400 },
  { nome: "Oceano Atlântico", lon: -18.5, lat: 36.0, ate: 6000 },
];

/** Rótulo cartográfico fixo no terreno, visível entre duas distâncias da câmara (km). */
function Rotulo({ classe, texto, x, z, ate, desde, mar = false, prio, desvios }: { classe: string; texto: string; x: number; z: number; ate: number; desde: number; mar?: boolean; prio?: number; desvios?: number[] }) {
  const ref = useRef<THREE.Group>(null);
  const el = useRef<HTMLDivElement>(null);
  const terreno = useTerreno();
  const registo = useRef<RotuloMapa | null>(null);
  useEffect(() => () => retirar(registo), []);
  useFrame(() => {
    if (!ref.current) return;
    const h = mar ? 0 : Math.max(terreno?.altitude(x, z) ?? 0, 0);
    ref.current.position.set(x, (h / 1000) * exagero() + 0.05, z);
    // a visibilidade é decidida pelo GestorMapa (distância, interface e sobreposições): mares 1, distritos 2
    registar(registo, el.current && { el: el.current, objeto: ref.current, prio: prio ?? (mar ? 1 : 2), ate, desde, desvios });
  });
  return (
    <group ref={ref}>
      <Html center zIndexRange={[4, 0]} style={{ pointerEvents: "none" }}>
        <div ref={el} className={classe} data-visivel="0">{texto}</div>
      </Html>
    </group>
  );
}

/** Limites dos concelhos: traço castanho fino e meio transparente, que aparece aos poucos ao aproximar. */
function Concelhos({ pontos }: { pontos: [number, number, number][] }) {
  const ref = useRef<{ material: THREE.Material & { opacity: number } } & THREE.Object3D>(null);
  useFrame(() => {
    const l = ref.current;
    if (!l) return;
    const t = THREE.MathUtils.clamp((CONCELHOS_ATE - relevo.distancia) / (CONCELHOS_ATE - CONCELHOS_TODO), 0, 1);
    const op = 0.6 * t;
    if (l.material.opacity !== op) l.material.opacity = op;
    l.visible = op > 0.01;
  });
  return (
    <Line ref={ref as never} points={pontos} segments color="#3d2c18" lineWidth={1.1} transparent opacity={0}
      depthWrite={false} depthTest={false} renderOrder={-9} />
  );
}

export { type Dados as DadosLinhas };
