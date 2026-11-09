import { describe, expect, it } from "vitest";
import { SEM_CRITERIOS, escreverCriterios, lerCriterios, normalizar, passa, type ItemCatalogo } from "./catalogo";

const moncao: ItemCatalogo = { texto: normalizar("Fortaleza de Monção Monção Monção Viana do Castelo"), regiao: "norte", tipologia: "praca-forte", maqueta: true, visita: false, ficha: true };
const sagres: ItemCatalogo = { texto: normalizar("Fortaleza de Sagres Sagres Vila do Bispo Faro"), regiao: "algarve", tipologia: "fortaleza", maqueta: false, visita: true, ficha: false };

describe("catálogo", () => {
  it("normaliza acentos e maiúsculas", () => {
    expect(normalizar("  Monção  e   Valença ")).toBe("moncao e valenca");
  });
  it("sem critérios passa tudo", () => {
    expect(passa(moncao, SEM_CRITERIOS) && passa(sagres, SEM_CRITERIOS)).toBe(true);
  });
  it("pesquisa por palavras, sem acentos e em qualquer ordem", () => {
    expect(passa(moncao, { ...SEM_CRITERIOS, q: "moncao" })).toBe(true);
    expect(passa(moncao, { ...SEM_CRITERIOS, q: "Viana MONÇÃO" })).toBe(true);
    expect(passa(sagres, { ...SEM_CRITERIOS, q: "monção" })).toBe(false);
  });
  it("filtros de região, tipologia e marcas", () => {
    expect(passa(moncao, { ...SEM_CRITERIOS, regiao: "algarve" })).toBe(false);
    expect(passa(sagres, { ...SEM_CRITERIOS, tipo: "fortaleza" })).toBe(true);
    expect(passa(sagres, { ...SEM_CRITERIOS, maqueta: true })).toBe(false);
    expect(passa(sagres, { ...SEM_CRITERIOS, visita: true })).toBe(true);
    expect(passa(sagres, { ...SEM_CRITERIOS, ficha: true })).toBe(false);
  });
  it("o endereço guarda e repõe os critérios", () => {
    const c = { q: "torre ", regiao: "norte", tipo: "castelo", maqueta: true, visita: false, ficha: true };
    const s = escreverCriterios(c);
    expect(s).toBe("?q=torre&regiao=norte&tipo=castelo&maqueta=1&ficha=1");
    expect(lerCriterios(s)).toEqual({ ...c, q: "torre" });
    expect(escreverCriterios(SEM_CRITERIOS)).toBe("");
  });
});
