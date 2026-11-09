import * as THREE from "three";

/**
 * Rótulos cartográficos (cidades, distritos, mares) e zonas do ecrã que eles não podem tapar.
 *
 * Cada rótulo regista-se aqui; o GestorMapa (Marcadores.tsx) decide a cada fotograma quais se mostram:
 * só os que estão dentro das distâncias da câmara, fora das zonas da interface (barra, rosa, escala, marca,
 * painéis) e sem se sobreporem a um rótulo mais importante. Um rótulo já visível tem prioridade sobre um
 * da mesma classe que ainda não está, para não piscarem ao arrastar o mapa.
 */
export interface RotuloMapa {
  el: HTMLElement;
  objeto: THREE.Object3D;
  /** 0 = mais importante */
  prio: number;
  /** km da câmara: visível entre desde e ate */
  ate: number;
  desde: number;
  /** o rótulo não pode sair do sítio (as cidades: o ponto faz parte do rótulo) */
  fixo?: boolean;
}

export type Caixa = [x0: number, y0: number, x1: number, y1: number];

export const rotulosMapa = new Set<RotuloMapa>();

/** Caixas (px, relativas ao canvas) das cartelas de monumentos já colocadas neste fotograma. */
export const caixasMonumentos: Caixa[] = [];

/** Elementos da interface por cima do mapa que os rótulos não devem tapar. */
const SELETORES_INTERFACE = ".ferramentas, .leitura .rosa-botao, .leitura .escala, .marca, .painel";

let cache: { t: number; caixas: Caixa[] } = { t: -1, caixas: [] };

/** Zonas ocupadas pela interface, relidas do DOM no máximo quatro vezes por segundo. */
export function zonasInterface(canvas: HTMLCanvasElement, agora: number): Caixa[] {
  if (agora - cache.t < 250) return cache.caixas;
  const raiz = canvas.closest(".atlas-mapa") ?? document;
  const r = canvas.getBoundingClientRect();
  const caixas: Caixa[] = [];
  raiz.querySelectorAll<HTMLElement>(SELETORES_INTERFACE).forEach((el) => {
    const b = el.getBoundingClientRect();
    if (b.width && b.height) caixas.push([b.left - r.left - 6, b.top - r.top - 6, b.right - r.left + 6, b.bottom - r.top + 6]);
  });
  cache = { t: agora, caixas };
  return caixas;
}

export const sobrepoe = (a: Caixa, b: Caixa) => !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3]);

export interface Candidato {
  id: number;
  prio: number;
  /** posições possíveis, pela ordem de preferência (a primeira é a do sítio exato) */
  caixas: Caixa[];
  jaVisivel: boolean;
  /** posição usada no fotograma anterior, tentada primeiro para o rótulo não saltar */
  anterior?: number;
}

/**
 * Escolhe os rótulos a mostrar e em que posição: por ordem de prioridade (e, dentro da mesma, os já visíveis
 * primeiro), cada um fica na primeira posição cuja caixa não toque numa zona proibida nem num rótulo já
 * aceite; se nenhuma servir, esconde-se. Devolve id -> índice da posição. Função pura, testada.
 */
export function escolherRotulos(cand: Candidato[], proibidas: Caixa[]): Map<number, number> {
  const ordem = [...cand].sort((a, b) => a.prio - b.prio || Number(b.jaVisivel) - Number(a.jaVisivel));
  const aceites: Caixa[] = [];
  const escolha = new Map<number, number>();
  for (const c of ordem) {
    const indices = c.caixas.map((_, i) => i);
    if (c.anterior !== undefined && c.anterior < indices.length) indices.unshift(...indices.splice(c.anterior, 1));
    for (const i of indices) {
      const caixa = c.caixas[i];
      if (proibidas.some((z) => sobrepoe(caixa, z)) || aceites.some((z) => sobrepoe(caixa, z))) continue;
      aceites.push(caixa);
      escolha.set(c.id, i);
      break;
    }
  }
  return escolha;
}

/**
 * Regista um rótulo quando o <Html> já montou o seu elemento (montagem noutra raiz React, por isso só se sabe
 * no primeiro fotograma) e retira-o ao desmontar. Chama-se dentro de um useFrame.
 */
export function registar(atual: { current: RotuloMapa | null }, dados: RotuloMapa | null) {
  if (!dados) return;
  if (atual.current?.el === dados.el && atual.current.objeto === dados.objeto) return;
  if (atual.current) rotulosMapa.delete(atual.current);
  atual.current = dados;
  rotulosMapa.add(dados);
}

export function retirar(atual: { current: RotuloMapa | null }) {
  if (atual.current) rotulosMapa.delete(atual.current);
  atual.current = null;
}
