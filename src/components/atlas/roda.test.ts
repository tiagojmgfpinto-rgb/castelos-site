import { describe, expect, it } from "vitest";
import { agrupar, MAX_RODA, posicoesRoda } from "./roda";

describe("agrupar", () => {
  it("uma torre sozinha abre logo", () => {
    expect(agrupar("a", [{ id: "a", x: 0, y: 0 }, { id: "b", x: 100, y: 0 }], 26)).toEqual(["a"]);
  });

  it("junta as vizinhas, também em cadeia, e ordena-as à volta do centro", () => {
    const pts = [
      { id: "a", x: 0, y: 0 },
      { id: "b", x: 20, y: 0 },
      { id: "c", x: 40, y: 0 }, // só toca em b
      { id: "longe", x: 200, y: 0 },
    ];
    const g = agrupar("a", pts, 26);
    expect(g.sort()).toEqual(["a", "b", "c"]);
  });

  it("não passa do máximo", () => {
    const pts = Array.from({ length: 30 }, (_, i) => ({ id: `t${i}`, x: i, y: 0 }));
    expect(agrupar("t0", pts, 26)).toHaveLength(MAX_RODA);
  });
});

describe("posicoesRoda", () => {
  it("põe a primeira em cima e afasta as torres umas das outras", () => {
    const p = posicoesRoda(5);
    expect(p[0][0]).toBeCloseTo(0);
    expect(p[0][1]).toBeLessThan(0);
    for (let i = 0; i < p.length; i++)
      for (let j = i + 1; j < p.length; j++) expect(Math.hypot(p[i][0] - p[j][0], p[i][1] - p[j][1])).toBeGreaterThan(28);
  });
});
