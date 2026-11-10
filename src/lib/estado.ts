// Regras de quem aparece onde. Puras (sem acesso a ficheiros), para correrem no build, no browser e nos testes.
//
//  estado da maqueta                   | mapa principal | Próximas adições
//  ------------------------------------|----------------|-----------------
//  pendente (sem maqueta)              | não            | sim
//  em-curso                            | não            | sim
//  concluida, mas sem ficheiro público | não            | sim
//  concluida e publicada               | sim            | não
//
// "Publicada" quer dizer: a variante da maqueta tem data `publicada`, está marcada como validada ou provisória e o
// GLB existe em public/ no momento da compilação. `concluida` sozinho não chega.

export const ESTADOS_MAQUETA = ["pendente", "em-curso", "concluida"] as const;
export type EstadoMaqueta = (typeof ESTADOS_MAQUETA)[number];

/** Fase mostrada ao visitante, que junta o estado declarado e a publicação efetiva do ficheiro. */
export type Fase = "por-fazer" | "em-curso" | "por-publicar" | "publicada";

export const FASES: Record<Fase, string> = {
  "por-fazer": "Maqueta por fazer",
  "em-curso": "Maqueta em construção",
  "por-publicar": "Maqueta concluída, por publicar",
  publicada: "Maqueta 3D publicada",
};

export interface VarianteMaqueta {
  ficheiro: string;
  validada?: boolean;
  provisoria?: boolean;
  publicada?: string;
}

/** A variante está mesmo publicada: tem data, foi aprovada (validada ou provisória) e o ficheiro existe. */
export function variantePublicada(v: VarianteMaqueta, existe: (ficheiro: string) => boolean) {
  return !!v.publicada && (!!v.validada || !!v.provisoria) && existe(v.ficheiro);
}

export function faseDe(estado: EstadoMaqueta, nPublicadas: number): Fase {
  if (estado === "concluida") return nPublicadas > 0 ? "publicada" : "por-publicar";
  return estado === "em-curso" ? "em-curso" : "por-fazer";
}

/** O monumento entra no mapa principal só com a maqueta concluída e publicada. */
export const entraNoMapa = (estado: EstadoMaqueta, nPublicadas: number) => faseDe(estado, nPublicadas) === "publicada";

// ——— Adicionados recentemente ———

/** Cada monumento fica no painel "Adicionados recentemente" 24 horas a contar da sua publicação. */
export const JANELA_RECENTES_MS = 24 * 3600 * 1000;

export interface Adicao {
  id: string;
  adicionado: string; // ISO 8601, UTC
}

/** Os que entraram no site há menos de 24 h, do mais recente para o mais antigo. Nada é apagado: só filtrado. */
export function recentes<T extends Adicao>(lista: T[], agora: number, janela = JANELA_RECENTES_MS): T[] {
  return lista
    .map((x) => ({ x, t: Date.parse(x.adicionado) }))
    .filter(({ t }) => Number.isFinite(t) && agora - t >= 0 && agora - t < janela)
    .sort((a, b) => b.t - a.t || a.x.id.localeCompare(b.x.id))
    .map(({ x }) => x);
}

/** Próximo instante em que a lista muda (alguém expira ou entra, se a data estiver no futuro). */
export function proximaMudanca(lista: Adicao[], agora: number, janela = JANELA_RECENTES_MS): number | null {
  let prox = Infinity;
  for (const x of lista) {
    const t = Date.parse(x.adicionado);
    if (!Number.isFinite(t)) continue;
    if (t > agora) prox = Math.min(prox, t);
    else if (agora - t < janela) prox = Math.min(prox, t + janela);
  }
  return Number.isFinite(prox) ? prox : null;
}

/** "há 5 minutos", "há 2 horas", "há 1 hora". */
export function haQuanto(ms: number) {
  const min = Math.max(0, Math.floor(ms / 60000));
  if (min < 1) return "agora mesmo";
  if (min < 60) return `há ${min} ${min === 1 ? "minuto" : "minutos"}`;
  const h = Math.floor(min / 60);
  if (h < 24) return `há ${h} ${h === 1 ? "hora" : "horas"}`;
  const d = Math.floor(h / 24);
  return `há ${d} ${d === 1 ? "dia" : "dias"}`;
}
