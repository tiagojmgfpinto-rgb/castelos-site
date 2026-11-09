import { describe, expect, it } from "vitest";
import { dadosMaqueta, elementosComPlaca } from "./dados";
import { disporPlacas } from "./placas";

const o = { linha: 40, folga: 6, niveis: 2, topo: 0 };
const placa = (x: number, y: number, candidata = true) => ({ x, y, w: 160, h: 36, candidata });
const niveis = (ps: ReturnType<typeof placa>[]) => disporPlacas(ps, o).map((l) => l.nivel);

describe("disposição das placas", () => {
  it("placas afastadas ficam todas no primeiro nível, centradas", () => {
    expect(disporPlacas([placa(100, 300), placa(400, 300)], o)).toEqual([{ nivel: 0, lado: 0 }, { nivel: 0, lado: 0 }]);
  });

  it("uma placa que tapava outra desvia-se para o lado, e a de mais prioridade fica no lugar", () => {
    expect(disporPlacas([placa(200, 300), placa(330, 300)], o)).toEqual([{ nivel: 0, lado: 0 }, { nivel: 0, lado: 0.38 }]);
  });

  it("sem lugar ao lado, sobe um nível; a que não cabe em nenhum esconde-se", () => {
    expect(niveis(Array.from({ length: 4 }, () => placa(200, 300)))).toEqual([0, 1, 2, -1]);
  });

  it("não sobe para cima dos botões do visor", () => {
    expect(disporPlacas([placa(200, 120), placa(200, 120)], { ...o, topo: 40 }).map((l) => l.nivel)).toEqual([0, -1]);
  });

  it("as que não são candidatas não ocupam lugar", () => {
    expect(niveis([placa(200, 300, false), placa(200, 300)])).toEqual([-1, 0]);
  });
});

describe.each(["penela", "guimaraes"])("placas da maqueta de %s", (monumento) => {
  it("só nomeiam elementos documentados no ficheiro, cada um com resumo", () => {
    const d = dadosMaqueta(monumento)!;
    const placas = elementosComPlaca(d);
    expect(placas.map((e) => e.id)).toEqual(d.placas);
    for (const e of placas) expect(e.resumo, e.id).toBeTruthy();
  });
});

describe("zonas bloqueadas", () => {
  it("um cartão não fica por cima do selo: desvia-se ou sobe", () => {
    const o = { linha: 20, folga: 4, niveis: 2, topo: 0, bloqueadas: [{ x0: 0, y0: 0, x1: 200, y1: 40 }] };
    // o cartão centrado no primeiro nível ficaria entre y 30 e 50, por cima do selo (até 40)
    const [l] = disporPlacas([{ x: 100, y: 70, w: 60, h: 20, candidata: true }], o);
    expect(l.nivel).toBe(-1);
    const [l2] = disporPlacas([{ x: 100, y: 120, w: 60, h: 20, candidata: true }], o);
    expect(l2.nivel).toBe(0);
  });
});
