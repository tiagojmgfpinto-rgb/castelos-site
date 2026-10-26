import { readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { paraCena, projectar, TERRITORIOS, type TerritorioId } from "./territorios";

// Valores de referência calculados com pyproj (PROJ) para as mesmas projeções.
describe("projeções oficiais", () => {
  it("PT-TM06 (Lisboa)", () => {
    const [x, y] = projectar(TERRITORIOS.continente, -9.1393, 38.7223);
    expect(x).toBeCloseTo(-87503.4, 0);
    expect(y).toBeCloseTo(-104538.9, 0);
  });
  it("PTRA08 UTM 26N (Ponta Delgada)", () => {
    const [x, y] = projectar(TERRITORIOS.acores, -25.6756, 37.7412);
    expect(x).toBeCloseTo(616688.9, 0);
    expect(y).toBeCloseTo(4177926.8, 0);
  });
  it("PTRA08 UTM 28N (Funchal)", () => {
    const [x, y] = projectar(TERRITORIOS.madeira, -16.9241, 32.6669);
    expect(x).toBeCloseTo(319572.0, 0);
    expect(y).toBeCloseTo(3615996.0, 0);
  });
});

describe("monumentos", () => {
  const dir = new URL("../content/monumentos/", import.meta.url);
  const ficheiros = readdirSync(dir).filter((f) => f.endsWith(".md"));
  const campo = (txt: string, nome: string) => txt.match(new RegExp(`^\\s*${nome}:\\s*"?([^"\\n]+)"?`, "m"))?.[1];

  it.each(ficheiros)("%s fica dentro do mapa do seu território", (f) => {
    const txt = readFileSync(new URL(f, dir), "utf8");
    const t = TERRITORIOS[campo(txt, "territorio") as TerritorioId];
    expect(t).toBeDefined();
    const [x, y] = projectar(t, Number(campo(txt, "lon")), Number(campo(txt, "lat")));
    const [sx, sz] = paraCena(t, x, y);
    const meio = t.size / 2000;
    expect(Math.abs(sx)).toBeLessThan(meio);
    expect(Math.abs(sz)).toBeLessThan(meio);
  });
});
