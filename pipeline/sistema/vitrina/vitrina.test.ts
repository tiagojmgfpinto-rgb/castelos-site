import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

// A vitrina de testes tem de desenhar com a mesma luz do visor do site. Estes testes falham se Luz.tsx ou Maqueta.tsx
// mudarem a luz, a oclusão ou o enquadramento sem a vitrina acompanhar.
const ler = (p: string) => readFileSync(new URL(p, import.meta.url), "utf8");
const vitrina = ler("./vitrina.html");
const luz = ler("../../../src/components/maqueta/Luz.tsx");
const maqueta = ler("../../../src/components/maqueta/Maqueta.tsx");
const css = ler("../../../src/styles/global.css");

const trecho = (txt: string, re: RegExp) => {
  const m = txt.match(re);
  if (!m) throw new Error(`não encontrado: ${re}`);
  return m[1].replace(/\s+/g, " ").trim();
};

describe("vitrina de testes", () => {
  it("lê a mesma luz (src/data/luz.json)", () => {
    expect(vitrina).toContain("fetch('/luz.json')");
    for (const campo of ["hemisferio.ceu", "ambiente.sigma", "ambiente.intensidade", "sol.intensidade", "sombra.bias", "sombra.raio", "oclusao.intensidade", "exposicao"]) {
      expect(vitrina).toContain(`LUZ.${campo}`);
    }
  });

  it("usa os mesmos parâmetros da oclusão ambiental", () => {
    const pd = /updatePdMaterial\((\{[^}]*\})\)/;
    expect(trecho(vitrina, pd)).toBe(trecho(luz, pd));
    const gtao = /distanceExponent: ([\d.]+)/;
    expect(trecho(vitrina, gtao)).toBe(trecho(luz, gtao));
  });

  it("enquadra o sol com os mesmos grupos de fora", () => {
    const fora = /\^\((Terreno\|[^)]*)\)_grupo\$/;
    expect(trecho(vitrina, fora)).toBe(trecho(maqueta, fora));
  });

  it("tem o fundo do visor", () => {
    const fundo = /(radial-gradient\(ellipse at 50% 42%[^;]*);/;
    expect(trecho(vitrina, fundo)).toBe(trecho(css, fundo));
  });
});
