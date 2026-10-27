import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";
import { REGIOES, TIPOLOGIAS } from "./lib/tipologias";

const chaves = <T extends Record<string, unknown>>(o: T) => Object.keys(o) as [keyof T & string, ...(keyof T & string)[]];

const data = z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "data no formato AAAA-MM-DD");

// "Organização — Título". `consultado` é a data em que a fonte foi lida para a ficha.
const fonte = z.object({ titulo: z.string(), url: z.url().optional(), consultado: data.optional() });

// Estrutura da ficha, pela ordem das fichas do castillos.site: entrada, ficha técnica, história (corpo do
// Markdown), o que ver, cronologia, medidas, sabia que, imagens, onde está, visita e fontes.
// Só entra o que está numa fonte citada; o que falta fica omisso.
const fichaTecnica = z
  .object({
    epoca: z.string(),
    tipologia: z.string(),
    estilo: z.string(),
    conservacao: z.string(),
    materiais: z.string(),
    protecao: z.string(),
    propriedade: z.string(),
    usoAtual: z.string(),
  })
  .partial();

const visita = z
  .object({
    resumo: z.string(), // uma linha para a ficha técnica
    horario: z.string(),
    entrada: z.string(),
    acessibilidade: z.string(),
    visitasGuiadas: z.string(),
    contactos: z.string(),
    site: fonte,
  })
  .partial();

const imagem = z.object({
  ficheiro: z.string(), // relativo à raiz pública do site
  legenda: z.string(),
  autor: z.string(),
  licenca: z.string(),
  url: z.url().optional(),
});

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

    subtitulo: z.string().optional(), // "Castelo medieval dos séculos XI a XVI"
    estado: z.string().optional(), // etiqueta curta do estado de conservação
    resumo: z.string().optional(), // parágrafo de entrada
    fichaTecnica: fichaTecnica.default({}),
    oQueVer: z.array(z.object({ nome: z.string(), texto: z.string() })).default([]),
    cronologia: z.array(z.object({ ano: z.string(), texto: z.string() })).default([]),
    medidas: z.array(z.object({ conceito: z.string(), valor: z.string() })).default([]),
    curiosidades: z.array(z.string()).default([]),
    imagens: z.array(imagem).default([]),
    altitude: z.number().optional(), // metros
    visita: visita.optional(),
    verificadoEm: data.optional(), // última verificação da ficha contra as fontes
  }),
});

export const collections = { monumentos };
