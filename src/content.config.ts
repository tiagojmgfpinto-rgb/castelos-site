import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";
import { REGIOES, TIPOLOGIAS } from "./lib/tipologias";

const chaves = <T extends Record<string, unknown>>(o: T) => Object.keys(o) as [keyof T & string, ...(keyof T & string)[]];

const fonte = z.object({ titulo: z.string(), url: z.string().url().optional() });

// Uma maqueta só entra no atlas depois de validada. Cada variante declara o que representa.
const maqueta = z.object({
  tipo: z.enum(["estado-atual", "reconstituicao"]),
  ficheiro: z.string(),
  epoca: z.string().optional(), // obrigatório na prática para reconstituições
  validada: z.boolean().default(false),
  // Versão provisória: publicada com aviso, com as limitações e estimativas à vista.
  provisoria: z.boolean().default(false),
  capa: z.string().optional(),
  fontes: z.array(fonte).default([]),
});

const monumentos = defineCollection({
  loader: glob({ pattern: "*.md", base: "./src/content/monumentos" }),
  schema: z.object({
    nome: z.string(),
    outrasDesignacoes: z.array(z.string()).default([]),
    tipologia: z.enum(chaves(TIPOLOGIAS)),
    territorio: z.enum(["continente", "acores", "madeira"]),
    regiao: z.enum(chaves(REGIOES)),
    distrito: z.string(),
    concelho: z.string(),
    localidade: z.string(),
    coordenadas: z.object({
      lat: z.number().min(30).max(43),
      lon: z.number().min(-32).max(-6),
      verificadas: z.boolean(),
      fonte: z.string().optional(),
    }),
    seculoOrigem: z.string().optional(),
    classificacao: z.object({ categoria: z.string(), sipa: z.string().optional() }).optional(),
    estadoFicha: z.enum(["em-preparacao", "publicada"]),
    prioritario: z.boolean().default(false),
    maquetas: z.array(maqueta).default([]),
    fontes: z.array(fonte).default([]),
    notas: z.array(z.string()).default([]),
  }),
});

export const collections = { monumentos };
