// Endereços permanentes das fichas, agrupados pela designação oficial do monumento:
// /castelos/guimaraes, /fortalezas/valenca, /fortes/sao-juliao-da-barra, /fortificacoes/almeida
export const SECCOES = {
  castelos: "Castelos",
  fortalezas: "Fortalezas",
  fortes: "Fortes",
  fortificacoes: "Fortificações",
} as const;

export type Seccao = keyof typeof SECCOES;

export function seccaoDe(nome: string): Seccao {
  if (/^castelo\b/i.test(nome)) return "castelos";
  if (/^fortaleza\b/i.test(nome)) return "fortalezas";
  if (/^forte\b/i.test(nome)) return "fortes";
  return "fortificacoes";
}

/** Caminho da ficha, relativo à base do site, com barra final. */
export function caminhoFicha(m: { id: string; nome: string }) {
  return `${seccaoDe(m.nome)}/${m.id}/`;
}
