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

/** Elementos com rótulo permanente quando os rótulos estão ligados. */
export const ROTULADOS = ["castelejo", "igreja", "porta_vila", "porta_traicao", "brecha", "torre_quinaria", "cubelo_noroeste", "casa_paroquial"];

/** Camadas que se escondem com o terreno (ficariam a flutuar sem ele). */
export const CAMADAS_TERRENO = ["Terreno_grupo", "Vegetacao_grupo", "Vila_grupo", "Caminhos_grupo"];
