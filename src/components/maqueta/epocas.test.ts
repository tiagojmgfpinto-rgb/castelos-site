import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync } from "node:fs";
import { vistaEpoca, GRUPOS_CONTEXTO, type DadosMaqueta } from "./dados";

const e = (id: string, grupo: string, desde?: number) => ({ id, nome: id, grupo, estado: "estado-atual" as const, confianca: "alta" as const, descricao: "", fontes: [], desde });
const d = { elementos: [e("torre", "Torre", 1300), e("igreja", "Igreja", 1680), e("porta", "Portas"), e("vila", "Envolvente")] } as unknown as DadosMaqueta;
const epoca = (ate: number) => ({ id: "x", nome: "x", ate, periodo: "", texto: "", fontes: [] });

describe("vistaEpoca", () => {
  it("hoje não esconde nada", () => {
    expect(vistaEpoca(d, null).ocultos.size).toBe(0);
  });
  it("esconde o que ainda não existia e assinala o que não tem data", () => {
    const v = vistaEpoca(d, epoca(1400));
    expect([...v.ocultos]).toEqual(["igreja"]);
    expect(v.semData.map((x) => x.id)).toEqual(["porta"]);
  });
  it("o contexto (casario, terreno) fica sempre", () => {
    expect(vistaEpoca(d, epoca(1000)).ocultos.has("vila")).toBe(false);
  });
});

// As épocas e datas de todas as maquetas: coerentes com as fontes que citam
const pasta = new URL("../../data/maquetas/", import.meta.url);
for (const f of readdirSync(pasta).filter((x) => x.endsWith(".json"))) {
  const m = JSON.parse(readFileSync(new URL(f, pasta), "utf8")) as DadosMaqueta;
  describe(`épocas de ${m.monumento}`, () => {
    it("estão por ordem, com fontes que existem, e cada data de elemento traz o texto da fonte", () => {
      const eps = m.epocas ?? [];
      for (let i = 1; i < eps.length; i++) expect(eps[i].ate).toBeGreaterThan(eps[i - 1].ate);
      for (const ep of eps) {
        expect(ep.texto.length).toBeGreaterThan(20);
        expect(ep.fontes.length).toBeGreaterThan(0);
        for (const k of ep.fontes) expect(m.fontes[k], `${ep.id}: fonte ${k}`).toBeDefined();
      }
      for (const el of m.elementos) {
        if (el.desde === undefined) continue;
        expect(el.desdeTexto, el.id).toBeTruthy();
        expect(GRUPOS_CONTEXTO.includes(el.grupo), `${el.id} é contexto e não precisa de data`).toBe(false);
      }
    });
  });
}
