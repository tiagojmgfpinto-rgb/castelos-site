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
  /** Direção de onde a câmara olha o elemento ao clicar na placa (x este, y cima, z sul), p. ex. para mostrar a fachada. */
  olhar?: [number, number, number];
  /**
   * Ano (aproximado) desde quando o elemento existe, segundo as fontes citadas em `fontes`: decide em que épocas
   * aparece. Sem ele, o elemento não tem data nas fontes e aparece em todas as épocas, assinalado como tal.
   */
  desde?: number;
  /** A mesma data como as fontes a dão (p. ex. "c. 1300", "1656", "século XVIII"). */
  desdeTexto?: string;
}

/** Época histórica de uma maqueta: mostra só os elementos que hoje existem e que já existiam até ao ano `ate`. */
export interface EpocaMaqueta {
  id: string;
  nome: string;
  /** Último ano da época: entram os elementos com `desde` até este ano. */
  ate: number;
  /** Como se escreve o período (p. ex. "séculos XIII–XIV"). */
  periodo: string;
  /** O que se passou nessa época, tirado das fontes (sem acrescentar factos). */
  texto: string;
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
  /** Elementos com placa flutuante, por ordem de prioridade quando não cabem todas. */
  placas?: string[];
  /** Distância inicial da câmara ao centro da maqueta, em metros (por omissão 300). */
  distancia?: number;
  /** Direção inicial da câmara (x este, y para cima, z sul), por omissão de sudoeste. */
  direcao?: [number, number, number];
  /** Direção de onde vem o sol (x para este, y para cima, z para sul); por omissão, de noroeste. */
  sol?: [number, number, number];
  /** Distância mínima da câmara ao alvo, em metros (por omissão 22); menor nas maquetas pequenas. */
  distanciaMinima?: number;
  elementos: ElementoMaqueta[];
  /** Épocas históricas, da mais antiga para a mais recente; a época atual ("Hoje") é implícita. */
  epocas?: EpocaMaqueta[];
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

/** Grupos de contexto (terreno, casario, árvores): são os de hoje e aparecem em todas as épocas. */
export const GRUPOS_CONTEXTO = ["Envolvente", "Terreno"];

export interface VistaEpoca {
  /** Elementos que ainda não existiam na época (escondem-se). */
  ocultos: Set<string>;
  /** Elementos sem data nas fontes (ficam à vista, assinalados). */
  semData: ElementoMaqueta[];
}

/** O que se esconde e o que fica por datar numa época; sem época (hoje), nada se esconde. */
export function vistaEpoca(d: DadosMaqueta, epoca: EpocaMaqueta | null | undefined): VistaEpoca {
  const ocultos = new Set<string>();
  const semData: ElementoMaqueta[] = [];
  if (!epoca) return { ocultos, semData };
  for (const e of d.elementos) {
    if (GRUPOS_CONTEXTO.includes(e.grupo)) continue;
    if (e.desde === undefined) semData.push(e);
    else if (e.desde > epoca.ate) ocultos.add(e.id);
  }
  return { ocultos, semData };
}
