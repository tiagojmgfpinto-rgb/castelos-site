/** Disposição das placas flutuantes no ecrã, sem sobreposições (funções puras, testadas). */

export interface PlacaNoEcra {
  /** Ponto do elemento projetado no ecrã, em px. */
  x: number;
  y: number;
  /** Tamanho do cartão, em px. */
  w: number;
  h: number;
  /** Se pode ser mostrada (no ecrã, não tapada pela maqueta e já entrou). */
  candidata: boolean;
}

export interface OpcoesDisposicao {
  /** Comprimento da linha no primeiro nível, em px. */
  linha: number;
  /** Folga entre cartões, em px. */
  folga: number;
  /** Níveis tentados acima do primeiro antes de desistir. */
  niveis: number;
  /** Margem de cima reservada aos botões do visor, em px. */
  topo: number;
  /** Zonas do visor onde nenhum cartão pode ficar (o selo da versão provisória, a bússola), em px. */
  bloqueadas?: Caixa[];
  /**
   * Altura e largura do visor, em px. Com a altura, uma placa que não cabe por cima pode ficar por
   * baixo do ponto (último recurso); com a largura, os cartões desviados não saem pela borda.
   */
  altura?: number;
  largura?: number;
  /** Margem de baixo reservada à legenda do visor, em px. */
  base?: number;
}

/**
 * Lugar de uma placa: nível da linha, desvio do cartão para o lado (fração da largura) e se o cartão
 * fica por baixo do ponto (só quando não cabe por cima).
 */
export interface Lugar { nivel: number; lado: number; baixo?: boolean }

/** Desvios tentados em cada nível: centrado, depois para a esquerda e para a direita, e mais longe. */
const LADOS = [0, -0.38, 0.38, -0.76, 0.76];

export interface Caixa { x0: number; y0: number; x1: number; y1: number }

const cruza = (a: Caixa, b: Caixa) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

/** Altura da linha (do ponto ao fundo do cartão) num dado nível. */
export function alturaLinha(nivel: number, h: number, o: OpcoesDisposicao): number {
  return o.linha + nivel * (h + o.folga);
}

/**
 * Atribui a cada placa um lugar para que nenhum cartão tape outro, por ordem de prioridade: as
 * primeiras ficam com a linha mais curta. Primeiro tenta-se desviar o cartão para o lado e só depois
 * subir um nível. As que não cabem, ou não são candidatas, ficam com nível -1 e escondem-se.
 */
export function disporPlacas(placas: PlacaNoEcra[], o: OpcoesDisposicao): Lugar[] {
  const ocupadas: Caixa[] = [...(o.bloqueadas ?? [])];
  const encaixar = (p: PlacaNoEcra, y0: number, y1: number, nivel: number, baixo: boolean): Lugar | null => {
    for (const lado of LADOS) {
      let x = p.x + lado * p.w;
      // o visor encosta o cartão à borda; a caixa ocupada tem de ser a do sítio onde ele fica de facto
      if (o.largura) x = Math.min(Math.max(x, p.w / 2 + 4), o.largura - p.w / 2 - 4);
      const c = { x0: x - p.w / 2 - o.folga / 2, x1: x + p.w / 2 + o.folga / 2, y0: y0 - o.folga / 2, y1: y1 + o.folga / 2 };
      if (ocupadas.some((q) => cruza(c, q))) continue;
      ocupadas.push(c);
      return baixo ? { nivel, lado, baixo } : { nivel, lado };
    }
    return null;
  };
  return placas.map((p) => {
    if (!p.candidata) return { nivel: -1, lado: 0 };
    for (let n = 0; n <= o.niveis; n++) {
      const fundo = p.y - alturaLinha(n, p.h, o);
      if (n > 0 && fundo - p.h < o.topo) break; // não subir para cima dos botões nem para fora do visor
      const l = encaixar(p, fundo - p.h, fundo, n, false);
      if (l) return l;
    }
    // Sem lugar por cima: o cartão desce para baixo do ponto, com a linha a apontar para cima
    if (o.altura) {
      for (let n = 0; n <= o.niveis; n++) {
        const cimo = p.y + alturaLinha(n, p.h, o);
        if (cimo + p.h > o.altura - (o.base ?? 0)) break;
        const l = encaixar(p, cimo, cimo + p.h, n, true);
        if (l) return l;
      }
    }
    return { nivel: -1, lado: 0 };
  });
}
