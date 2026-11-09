import { describe, expect, it } from "vitest";
import { visivel, type MonumentoResumo } from "./store";

const m = (extra: Partial<MonumentoResumo>): MonumentoResumo =>
  ({
    id: "x", nome: "X", tipologia: "castelo", territorio: "continente", regiao: "norte", distrito: "Braga", concelho: "Braga", localidade: "Braga",
    lat: 41, lon: -8, coordenadasVerificadas: true, estadoFicha: "publicada", maquetas: [], notas: [], fontes: [], html: "",
    conteudo: { outrasDesignacoes: [], fichaTecnica: {}, oQueVer: [], cronologia: [], medidas: [], curiosidades: [], imagens: [] },
    ...extra,
  }) as MonumentoResumo;

const base = { filtrosTipologia: new Set<never>(), filtroRegiao: null };

describe("visivel com os filtros da ficha", () => {
  const comMaqueta = m({ maquetas: [{ tipo: "estado-atual", ficheiro: "a.glb", provisoria: false }] });
  const comVisita = m({ conteudo: { ...m({}).conteudo, visita: { horario: "Todos os dias" } } });
  const visitaVazia = m({ conteudo: { ...m({}).conteudo, visita: {} } });

  it("sem filtros mostra todos", () => {
    for (const x of [m({}), comMaqueta, comVisita]) expect(visivel(x, base)).toBe(true);
  });
  it("com maqueta 3D", () => {
    expect(visivel(comMaqueta, { ...base, filtroMaqueta: true })).toBe(true);
    expect(visivel(comVisita, { ...base, filtroMaqueta: true })).toBe(false);
  });
  it("com informação de visita (um bloco vazio não conta)", () => {
    expect(visivel(comVisita, { ...base, filtroVisita: true })).toBe(true);
    expect(visivel(visitaVazia, { ...base, filtroVisita: true })).toBe(false);
    expect(visivel(comMaqueta, { ...base, filtroVisita: true })).toBe(false);
  });
});
