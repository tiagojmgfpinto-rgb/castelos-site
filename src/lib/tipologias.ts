// Tipologias e regiões do atlas. As chaves são usadas nos ficheiros de cada monumento.
export const TIPOLOGIAS = {
  castelo: { nome: "Castelo medieval", plural: "Castelos medievais", forma: "torre" },
  fortaleza: { nome: "Fortaleza", plural: "Fortalezas", forma: "baluarte" },
  forte: { nome: "Forte ou fortim", plural: "Fortes e fortins", forma: "baluarte" },
  "praca-forte": { nome: "Praça-forte", plural: "Praças-fortes", forma: "baluarte" },
  "cerca-urbana": { nome: "Cidade ou vila muralhada", plural: "Cidades e vilas muralhadas", forma: "torre" },
  costeira: { nome: "Fortificação costeira", plural: "Fortificações costeiras", forma: "baluarte" },
  abaluartada: { nome: "Fortificação abaluartada", plural: "Fortificações abaluartadas", forma: "baluarte" },
  bateria: { nome: "Bateria defensiva", plural: "Baterias defensivas", forma: "baluarte" },
  outra: { nome: "Outra estrutura militar", plural: "Outras estruturas militares", forma: "torre" },
} as const;

export type Tipologia = keyof typeof TIPOLOGIAS;

export const REGIOES = {
  norte: "Norte",
  centro: "Centro",
  "lisboa-vale-tejo": "Lisboa e Vale do Tejo",
  alentejo: "Alentejo",
  algarve: "Algarve",
  acores: "Açores",
  madeira: "Madeira",
} as const;

export type Regiao = keyof typeof REGIOES;
