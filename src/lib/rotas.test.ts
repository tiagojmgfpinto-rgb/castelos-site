import { describe, expect, it } from "vitest";
import { caminhoFicha } from "./rotas";

// Exemplos de endereços permanentes da secção 15 do briefing
describe("endereços das fichas", () => {
  it.each([
    ["guimaraes", "Castelo de Guimarães", "castelos/guimaraes/"],
    ["almourol", "Castelo de Almourol", "castelos/almourol/"],
    ["valenca", "Fortaleza de Valença", "fortalezas/valenca/"],
    ["almeida", "Praça-forte de Almeida", "fortificacoes/almeida/"],
    ["sao-juliao-da-barra", "Forte de São Julião da Barra", "fortes/sao-juliao-da-barra/"],
  ])("%s", (id, nome, esperado) => {
    expect(caminhoFicha({ id, nome })).toBe(esperado);
  });
});
