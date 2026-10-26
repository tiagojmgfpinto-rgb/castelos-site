import { create } from "zustand";
import type { TerritorioId } from "../../lib/territorios";
import type { Regiao, Tipologia } from "../../lib/tipologias";

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
  maquetas: { tipo: "estado-atual" | "reconstituicao"; ficheiro: string; epoca?: string }[];
  notas: string[];
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

interface Estado {
  territorio: TerritorioId;
  selecionado: string | null;
  sobre: string | null;
  filtrosTipologia: Set<Tipologia>;
  filtroRegiao: Regiao | null;
  indiceAberto: boolean;
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
  setIndiceAberto: (indiceAberto) => set({ indiceAberto }),
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
