import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { projectar, TERRITORIOS } from "./territorios";

// Linhas do mapa geradas pelo pipeline (build_tiles.py) a partir da CAOP 2025 da DGT: no CI correm depois de as gerar.
type P = [number, number, number];
const linhas = JSON.parse(readFileSync("public/data/portugal/linhas.json", "utf8")) as {
  fonteLimites: string;
  fronteira: P[][];
  distritos: P[][];
  concelhos: P[][];
  nomesDistritos: { nome: string; x: number; y: number }[];
  ilhas: { nome: string; arquipelago: string; x: number; y: number; caixa: [number, number, number, number] }[];
};
const T = TERRITORIOS.continente;

describe("limites administrativos (CAOP)", () => {
  it("vêm da CAOP da Direção-Geral do Território", () => {
    expect(linhas.fonteLimites).toMatch(/CAOP 20\d\d, Direção-Geral do Território/);
  });

  it("têm os 18 distritos do Continente, cada um com o nome dentro de Portugal Continental", () => {
    const nomes = linhas.nomesDistritos.map((d) => d.nome).sort((a, b) => a.localeCompare(b, "pt"));
    expect(nomes).toEqual(["Aveiro", "Beja", "Braga", "Bragança", "Castelo Branco", "Coimbra", "Évora", "Faro", "Guarda",
      "Leiria", "Lisboa", "Portalegre", "Porto", "Santarém", "Setúbal", "Viana do Castelo", "Vila Real", "Viseu"]);
    const [x0, y0] = projectar(T, -9.6, 36.9);
    const [x1, y1] = projectar(T, -6.1, 42.2);
    for (const d of linhas.nomesDistritos) {
      expect(d.x).toBeGreaterThan(x0);
      expect(d.x).toBeLessThan(x1);
      expect(d.y).toBeGreaterThan(y0);
      expect(d.y).toBeLessThan(y1);
    }
  });

  it("o nome de cada distrito fica perto da sua capital (a identificação não está trocada)", () => {
    const capitais: Record<string, [number, number]> = {
      Porto: [-8.61, 41.15], Lisboa: [-9.14, 38.72], Faro: [-7.93, 37.02], Bragança: [-6.76, 41.81], Évora: [-7.91, 38.57],
      "Viana do Castelo": [-8.83, 41.69], Beja: [-7.86, 38.02], Guarda: [-7.27, 40.54],
    };
    for (const [nome, [lon, lat]] of Object.entries(capitais)) {
      const d = linhas.nomesDistritos.find((x) => x.nome === nome)!;
      const [x, y] = projectar(T, lon, lat);
      // o mais perto de todos os nomes de distrito é o do próprio distrito
      const maisPerto = [...linhas.nomesDistritos].sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y))[0];
      expect(maisPerto.nome, nome).toBe(d.nome);
    }
  });

  it("há fronteira, limites de distrito e de concelho, com altitude em cada ponto", () => {
    expect(linhas.fronteira.length).toBeGreaterThan(0);
    expect(linhas.distritos.length).toBeGreaterThan(10);
    expect(linhas.concelhos.length).toBeGreaterThan(200);
    for (const l of [...linhas.fronteira, ...linhas.distritos, ...linhas.concelhos]) {
      expect(l.length).toBeGreaterThanOrEqual(2);
      for (const p of l) expect(p).toHaveLength(3);
    }
  });

  it("a fronteira vai da foz do Minho à foz do Guadiana", () => {
    const pts = linhas.fronteira.flat();
    const [xm, ym] = projectar(T, -8.87, 41.87);
    const [xg, yg] = projectar(T, -7.41, 37.17);
    const dist = (x: number, y: number) => Math.min(...pts.map((p) => Math.hypot(p[0] - x, p[1] - y)));
    expect(dist(xm, ym)).toBeLessThan(3000);
    expect(dist(xg, yg)).toBeLessThan(3000);
  });
});

describe("ilhas", () => {
  it("todas as ilhas dos Açores e da Madeira têm nome, com a caixa à volta do centro", () => {
    const acores = ["Santa Maria", "São Miguel", "Terceira", "Graciosa", "São Jorge", "Pico", "Faial", "Flores", "Corvo"];
    const madeira = ["Madeira", "Porto Santo", "Ilhas Desertas", "Ilhas Selvagens"];
    expect(linhas.ilhas.filter((i) => i.arquipelago === "Açores").map((i) => i.nome).sort()).toEqual([...acores].sort());
    expect(linhas.ilhas.filter((i) => i.arquipelago === "Madeira").map((i) => i.nome).sort()).toEqual([...madeira].sort());
    for (const i of linhas.ilhas) {
      const [x0, y0, x1, y1] = i.caixa;
      expect(i.x, i.nome).toBeGreaterThanOrEqual(x0);
      expect(i.x, i.nome).toBeLessThanOrEqual(x1);
      expect(i.y, i.nome).toBeGreaterThanOrEqual(y0);
      expect(i.y, i.nome).toBeLessThanOrEqual(y1);
    }
  });

  it("cada ilha está onde deve (centro a menos de 25 km da posição de referência)", () => {
    const ref: Record<string, [number, number]> = {
      "São Miguel": [-25.5, 37.78], Terceira: [-27.22, 38.72], Pico: [-28.33, 38.46], Flores: [-31.2, 39.44],
      Corvo: [-31.1, 39.7], Madeira: [-16.95, 32.75], "Porto Santo": [-16.34, 33.07], "Ilhas Selvagens": [-15.9, 30.14],
    };
    for (const [nome, [lon, lat]] of Object.entries(ref)) {
      const i = linhas.ilhas.find((x) => x.nome === nome)!;
      const [x, y] = projectar(T, lon, lat);
      expect(Math.hypot(i.x - x, i.y - y), nome).toBeLessThan(25000);
    }
  });
});
