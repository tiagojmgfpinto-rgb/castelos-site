/** Leitura das fontes, estado e confiança de cada elemento de uma maqueta (src/data/maquetas/<id>.json). */
export interface ElementoMaqueta {
  id: string;
  nome: string;
  grupo: string;
  estado: "estado-atual" | "reconstituicao" | "hipotese";
  confianca: "alta" | "média" | "baixa";
  descricao: string;
  medidas?: string;
  fontes: string[];
  /** Linha curta para a placa flutuante (tirada da descrição, sem acrescentar factos). */
  resumo?: string;
  /** Ponto da placa em coordenadas do visor (x este, y cota, z sul). Sem ele, usa-se o topo do elemento. */
  ancora?: [number, number, number];
}

export interface DadosMaqueta {
  monumento: string;
  titulo: string;
  variante: string;
  versao: string;
  data: string;
  origem: { lat: number; lon: number; nota: string };
  fontes: Record<string, { titulo: string; url?: string }>;
  limitacoes: string[];
  /** Elementos com placa flutuante, por ordem de prioridade quando não cabem todas. */
  placas?: string[];
  elementos: ElementoMaqueta[];
}

const ficheiros = import.meta.glob<DadosMaqueta>("../../data/maquetas/*.json", { eager: true, import: "default" });

export function dadosMaqueta(monumento: string): DadosMaqueta | undefined {
  return Object.values(ficheiros).find((d) => d.monumento === monumento);
}

export const ESTADOS: Record<ElementoMaqueta["estado"], string> = {
  "estado-atual": "Estado atual",
  reconstituicao: "Reconstituição",
  hipotese: "Hipótese",
};

/**
 * Elementos com placa: os indicados em "placas" ou, numa maqueta sem essa lista, todos os que
 * não são envolvente. Só entram elementos que existem e estão documentados no ficheiro.
 */
export function elementosComPlaca(d: DadosMaqueta): ElementoMaqueta[] {
  const porId = new Map(d.elementos.map((e) => [e.id, e]));
  const ids = d.placas ?? d.elementos.filter((e) => e.grupo !== "Envolvente").map((e) => e.id);
  return ids.map((id) => porId.get(id)).filter((e): e is ElementoMaqueta => !!e);
}

/** Camadas que se escondem com o terreno (ficariam a flutuar sem ele). */
export const CAMADAS_TERRENO = ["Terreno_grupo", "Vegetacao_grupo", "Vila_grupo", "Caminhos_grupo"];
