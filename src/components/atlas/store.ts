import { create } from "zustand";
import type { TerritorioId } from "../../lib/territorios";
import type { Regiao, Tipologia } from "../../lib/tipologias";

export interface Maqueta {
  tipo: "estado-atual" | "reconstituicao";
  ficheiro: string; // relativo à raiz pública do site
  epoca?: string;
  provisoria: boolean;
  capa?: string;
}

export interface MonumentoResumo {
  id: string;
  nome: string;
  tipologia: Tipologia;
  territorio: TerritorioId;
  regiao: Regiao;
  distrito: string;
  concelho: string;
  localidade: string;
  lat: number;
  lon: number;
  coordenadasVerificadas: boolean;
  estadoFicha: "em-preparacao" | "publicada";
  seculoOrigem?: string;
  maquetas: Maqueta[];
  notas: string[];
  fontes: Fonte[];
  html: string; // história (corpo do Markdown já convertido no build)
  conteudo: ConteudoFicha;
}

export interface Fonte {
  titulo: string;
  url?: string;
  consultado?: string;
}

/** Secções da ficha, pela ordem do castillos.site. Tudo opcional: o que não tem fonte fica de fora. */
export interface ConteudoFicha {
  outrasDesignacoes: string[];
  subtitulo?: string;
  estado?: string;
  resumo?: string;
  fichaTecnica: Partial<Record<"epoca" | "tipologia" | "estilo" | "conservacao" | "materiais" | "protecao" | "propriedade" | "usoAtual", string>>;
  oQueVer: { nome: string; texto: string }[];
  cronologia: { ano: string; texto: string }[];
  medidas: { conceito: string; valor: string }[];
  curiosidades: string[];
  imagens: { ficheiro: string; legenda: string; autor: string; licenca: string; url?: string }[];
  altitude?: number;
  visita?: Partial<Record<"resumo" | "horario" | "entrada" | "acessibilidade" | "visitasGuiadas" | "contactos", string>> & { site?: Fonte };
  verificadoEm?: string;
}

/** Pedido de voo de câmara: alvo na cena (km) e enquadramento final. */
export interface Voo {
  x: number;
  z: number;
  distancia: number; // km
  inclinacao: number; // graus a partir da vertical
  rumo?: number; // graus; omisso mantém o atual
  instantaneo?: boolean;
}

export type Painel = "indice" | "distritos" | "fontes";

interface Estado {
  territorio: TerritorioId;
  selecionado: string | null;
  sobre: string | null;
  filtrosTipologia: Set<Tipologia>;
  filtroRegiao: Regiao | null;
  indiceAberto: boolean;
  painel: Painel | null;
  distritoAtivo: string | null;
  cursor: string; // coordenadas sob o rato, já formatadas
  voo: (Voo & { pedido: number }) | null;
  carregamento: number; // 0..1
  escalaKmPorPx: number;
  rumoAtual: number;
  setTerritorio: (t: TerritorioId) => void;
  selecionar: (id: string | null) => void;
  setSobre: (id: string | null) => void;
  alternarTipologia: (t: Tipologia) => void;
  setFiltroRegiao: (r: Regiao | null) => void;
  setIndiceAberto: (v: boolean) => void;
  alternarPainel: (p: Painel) => void;
  setDistritoAtivo: (d: string | null) => void;
  setCursor: (c: string) => void;
  voar: (v: Voo) => void;
  setCarregamento: (v: number) => void;
  setEscala: (kmPorPx: number, rumo: number) => void;
}

let pedidos = 0;

export const useAtlas = create<Estado>((set) => ({
  territorio: "continente",
  selecionado: null,
  sobre: null,
  filtrosTipologia: new Set(),
  filtroRegiao: null,
  indiceAberto: false,
  painel: null,
  distritoAtivo: null,
  cursor: "",
  voo: null,
  carregamento: 0,
  escalaKmPorPx: 1,
  rumoAtual: 0,
  setTerritorio: (territorio) => set({ territorio, selecionado: null, sobre: null, voo: null, carregamento: 0 }),
  selecionar: (selecionado) => set({ selecionado }),
  setSobre: (sobre) => set({ sobre }),
  alternarTipologia: (t) =>
    set((s) => {
      const f = new Set(s.filtrosTipologia);
      f.has(t) ? f.delete(t) : f.add(t);
      return { filtrosTipologia: f };
    }),
  setFiltroRegiao: (filtroRegiao) => set({ filtroRegiao }),
  // os painéis de pergaminho abrem um de cada vez, como no castillos.site
  setIndiceAberto: (v) => set({ indiceAberto: v, painel: v ? "indice" : null }),
  alternarPainel: (p) => set((s) => {
    const painel = s.painel === p ? null : p;
    return { painel, indiceAberto: painel === "indice" };
  }),
  setDistritoAtivo: (distritoAtivo) => set({ distritoAtivo }),
  setCursor: (cursor) => set({ cursor }),
  voar: (v) => set({ voo: { ...v, pedido: ++pedidos } }),
  setCarregamento: (carregamento) => set({ carregamento }),
  setEscala: (escalaKmPorPx, rumoAtual) => set({ escalaKmPorPx, rumoAtual }),
}));

export function visivel(m: MonumentoResumo, s: Pick<Estado, "filtrosTipologia" | "filtroRegiao">) {
  if (s.filtrosTipologia.size && !s.filtrosTipologia.has(m.tipologia)) return false;
  if (s.filtroRegiao && m.regiao !== s.filtroRegiao) return false;
  return true;
}

// acesso para testes automáticos no navegador
if (typeof window !== "undefined") (window as unknown as { __atlas: typeof useAtlas }).__atlas = useAtlas;
