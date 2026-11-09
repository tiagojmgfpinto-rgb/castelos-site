/**
 * Roda temporária, como no castillos.site: quando se toca numa torre com outras coladas a ela no ecrã, em vez
 * de abrir uma ao acaso afastam-se todas em círculo à volta do centro delas, ligadas a ele por um fio, e
 * escolhe-se a certa. Funções puras (testadas em roda.test.ts); o desenho está em Marcadores.tsx.
 */

export interface PontoEcra {
  id: string;
  x: number;
  y: number;
}

/** Máximo de torres numa roda: com mais do que isto aproxima-se o mapa. */
export const MAX_RODA = 12;

/**
 * Torres que se tocam com a torre tocada: as que estão a menos de `raio` px dela ou, em cadeia, de outra do
 * grupo. Devolve os ids pela ordem do ângulo à volta do centro (de cima, no sentido dos ponteiros), para a
 * roda não cruzar fios; um só id quando não há vizinhas.
 */
export function agrupar(alvo: string, pontos: PontoEcra[], raio: number): string[] {
  const inicio = pontos.find((p) => p.id === alvo);
  if (!inicio) return [alvo];
  const grupo = [inicio];
  const resto = pontos.filter((p) => p.id !== alvo);
  for (let i = 0; i < grupo.length && grupo.length < MAX_RODA; i++) {
    for (let j = resto.length - 1; j >= 0 && grupo.length < MAX_RODA; j--) {
      if (Math.hypot(resto[j].x - grupo[i].x, resto[j].y - grupo[i].y) < raio) grupo.push(...resto.splice(j, 1));
    }
  }
  if (grupo.length === 1) return [alvo];
  const cx = grupo.reduce((s, p) => s + p.x, 0) / grupo.length;
  const cy = grupo.reduce((s, p) => s + p.y, 0) / grupo.length;
  const ang = (p: PontoEcra) => (Math.atan2(p.x - cx, -(p.y - cy)) + 2 * Math.PI) % (2 * Math.PI);
  return grupo.sort((a, b) => ang(a) - ang(b) || a.id.localeCompare(b.id)).map((p) => p.id);
}

/** Desvio no ecrã (px, y para baixo) de cada torre da roda, a começar em cima e no sentido dos ponteiros. */
export function posicoesRoda(n: number): [number, number][] {
  // o raio cresce com o número de torres para as cartelas e as torres não se tocarem
  const r = Math.max(64, (n * 40) / Math.PI);
  return Array.from({ length: n }, (_, k) => {
    const a = (k / n) * 2 * Math.PI;
    return [Math.sin(a) * r, -Math.cos(a) * r];
  });
}
