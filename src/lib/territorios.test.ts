import { readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { paraCena, projectar, TERRITORIOS, type TerritorioId } from "./territorios";

// Valores de referência calculados com pyproj (PROJ) para a mesma projeção.
describe("mapa único (Lambert, ETRS89)", () => {
  const casos: [string, TerritorioId, number, number, number, number][] = [
    ["Lisboa", "continente", -9.1393, 38.7223, -87357.2, -104403.0],
    ["Ponta Delgada", "acores", -25.6756, 37.7412, -1534405.7, -70076.9],
    ["Funchal", "madeira", -16.9241, 32.6669, -824781.6, -737563.2],
  ];
  it.each(casos)("%s", (_n, t, lon, lat, ex, ey) => {
    const [x, y] = projectar(TERRITORIOS[t], lon, lat);
    expect(x).toBeCloseTo(ex, 0);
    expect(y).toBeCloseTo(ey, 0);
  });
  it("os três territórios partilham o mesmo quadrado", () => {
    const { continente, acores, madeira } = TERRITORIOS;
    for (const t of [acores, madeira]) {
      expect(t.crs).toBe(continente.crs);
      expect([t.x0, t.y0, t.size]).toEqual([continente.x0, continente.y0, continente.size]);
    }
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
