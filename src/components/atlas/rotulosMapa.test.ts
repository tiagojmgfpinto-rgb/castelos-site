import { describe, expect, it } from "vitest";
import { escolherRotulos, type Candidato } from "./rotulosMapa";

const c = (id: number, prio: number, x: number, y: number, jaVisivel = false): Candidato => ({ id, prio, caixas: [[x, y, x + 80, y + 16]], jaVisivel });
const ids = (m: Map<number, number>) => [...m.keys()];

describe("escolherRotulos", () => {
  it("não mostra rótulos por baixo da barra de ferramentas", () => {
    const barra: [number, number, number, number] = [0, 900, 400, 960];
    expect(ids(escolherRotulos([c(1, 0, 100, 920), c(2, 0, 100, 400)], [barra]))).toEqual([2]);
  });

  it("entre dois rótulos sobrepostos fica o mais importante", () => {
    // Vila Real (distrito, prio 2) e Bragança (cidade de nível 1, prio 0) lado a lado
    expect(ids(escolherRotulos([c(1, 2, 100, 100), c(2, 0, 150, 105)], []))).toEqual([2]);
  });

  it("na mesma classe, o que já está visível não cede ao que acaba de entrar", () => {
    expect(ids(escolherRotulos([c(1, 2, 100, 100), c(2, 2, 150, 105, true)], []))).toEqual([2]);
  });

  it("um rótulo que pode mover-se sobe para não tapar o vizinho", () => {
    const viseu: Candidato = { id: 2, prio: 2, caixas: [[150, 105, 230, 121], [150, 88, 230, 104]], jaVisivel: false };
    expect([...escolherRotulos([c(1, 2, 100, 110), viseu], []).entries()]).toEqual([[1, 0], [2, 1]]);
  });

  it("a posição anterior é tentada primeiro, para o rótulo não saltar", () => {
    const r: Candidato = { id: 1, prio: 2, caixas: [[0, 20, 80, 36], [0, 0, 80, 16]], jaVisivel: true, anterior: 1 };
    expect(escolherRotulos([r], []).get(1)).toBe(1);
  });

  it("rótulos afastados aparecem todos", () => {
    expect(escolherRotulos([c(1, 2, 0, 0), c(2, 2, 200, 0), c(3, 1, 0, 100)], []).size).toBe(3);
  });
});
