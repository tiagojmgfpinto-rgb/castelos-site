import { getCollection, type CollectionEntry } from "astro:content";
import type { MonumentoResumo } from "../components/atlas/store";
import { REGIOES, type Regiao } from "./tipologias";
import { comVersao } from "./versao";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { adicionadoEm } from "./adicionados";
import { faseDe, variantePublicada } from "./estado";

const existe = (ficheiro: string) => existsSync(join(process.cwd(), "public", ficheiro.split("?")[0]));

/** Variantes de maqueta efetivamente publicadas (data, aprovação e GLB presente). */
export const maquetasPublicadas = (m: Monumento) => m.data.maquetas.filter((x) => variantePublicada(x, existe));
/** Fase da maqueta e se o monumento entra no mapa principal (regras em lib/estado.ts). */
export const faseDoMonumento = (m: Monumento) => faseDe(m.data.estadoMaqueta, maquetasPublicadas(m).length);
export const estaNoMapa = (m: Monumento) => faseDoMonumento(m) === "publicada";
/** Imagem de apresentação: a capa da maqueta publicada ou a primeira fotografia da ficha. */
export const imagemDe = (m: Monumento) => maquetasPublicadas(m).find((x) => x.capa)?.capa ?? m.data.imagens[0]?.ficheiro;

export type Monumento = CollectionEntry<"monumentos">;

export async function todosMonumentos(): Promise<Monumento[]> {
  const lista = await getCollection("monumentos");
  const ordemRegiao = Object.keys(REGIOES) as Regiao[];
  return lista.sort(
    (a, b) => ordemRegiao.indexOf(a.data.regiao) - ordemRegiao.indexOf(b.data.regiao) || a.data.nome.localeCompare(b.data.nome, "pt"),
  );
}

export function resumo(m: Monumento): MonumentoResumo {
  const d = m.data;
  return {
    id: m.id,
    nome: d.nome,
    tipologia: d.tipologia,
    territorio: d.territorio,
    regiao: d.regiao,
    distrito: d.distrito,
    concelho: d.concelho,
    localidade: d.localidade,
    lat: d.coordenadas.lat,
    lon: d.coordenadas.lon,
    coordenadasVerificadas: d.coordenadas.verificadas,
    estadoFicha: d.estadoFicha,
    seculoOrigem: d.seculoOrigem,
    estadoMaqueta: d.estadoMaqueta,
    fase: faseDoMonumento(m),
    noMapa: estaNoMapa(m),
    adicionado: adicionadoEm(m.id, d.adicionado),
    imagem: comVersao(imagemDe(m)),
    // só as publicadas: uma maqueta concluída por publicar não aparece em lado nenhum
    maquetas: maquetasPublicadas(m)
      .map(({ tipo, ficheiro, epoca, provisoria, capa, publicada }) => ({ tipo, ficheiro: comVersao(ficheiro), epoca, provisoria, capa: comVersao(capa), publicada })),
    notas: d.notas,
    fontes: d.fontes,
    html: m.rendered?.html ?? "",
    conteudo: {
      outrasDesignacoes: d.outrasDesignacoes,
      subtitulo: d.subtitulo,
      estado: d.estado,
      resumo: d.resumo,
      fichaTecnica: { protecao: d.classificacao?.categoria, ...d.fichaTecnica },
      oQueVer: d.oQueVer,
      cronologia: d.cronologia,
      medidas: d.medidas,
      curiosidades: d.curiosidades,
      imagens: d.imagens,
      altitude: d.altitude,
      visita: d.visita,
      verificadoEm: d.verificadoEm,
    },
  };
}
