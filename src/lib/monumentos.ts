import { getCollection, type CollectionEntry } from "astro:content";
import type { MonumentoResumo } from "../components/atlas/store";
import { REGIOES, type Regiao } from "./tipologias";

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
    maquetas: d.maquetas
      .filter((x) => x.validada || x.provisoria)
      .map(({ tipo, ficheiro, epoca, provisoria, capa }) => ({ tipo, ficheiro, epoca, provisoria, capa })),
    notas: d.notas,
    fontes: d.fontes,
    html: m.rendered?.html ?? "",
  };
}
