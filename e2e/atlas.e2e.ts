import { expect, test } from "@playwright/test";
import { errosDaPagina } from "./ajuda";

test("atlas: carrega e, ao abrir um castelo, não mostra os nomes dos outros à volta", async ({ page }) => {
  const erros = errosDaPagina(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.route(/\.glb/, () => undefined);
  await page.goto("#guimaraes");
  // o aviso de carregamento do relevo desaparece
  await page.waitForSelector('.estado[data-visivel="0"]', { state: "attached", timeout: 150_000 });
  await expect(page.locator(".etiqueta[data-visivel='1'] b").first()).toHaveText("Castelo de Guimarães", { timeout: 60_000 });
  await page.waitForTimeout(4000);
  const visiveis = await page.$$eval(".etiqueta[data-visivel='1'] b", (bs) => bs.map((b) => b.textContent));
  expect(visiveis).toEqual(["Castelo de Guimarães"]);
  // a ficha abre ao lado
  await expect(page.locator(".ficha")).toBeVisible();
  expect(erros).toEqual([]);
});

test("atlas: ao passar o rato numa torre só aparece o nome dela", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.route(/\.glb/, () => undefined);
  await page.goto("./");
  await page.waitForSelector('.estado[data-visivel="0"]', { state: "attached", timeout: 150_000 });
  // Braga tem Guimarães e Lanhoso coladas no ecrã: antes apareciam os três nomes
  await page.evaluate(() => (window as unknown as { __atlas: { getState(): { setSobre(id: string): void } } }).__atlas.getState().setSobre("braga"));
  await page.waitForTimeout(1500);
  const visiveis = await page.$$eval(".etiqueta[data-visivel='1'] b", (bs) => bs.map((b) => b.textContent));
  expect(visiveis).toHaveLength(1);
});

test("atlas: Novidades com as abas Novos castelos e Próximas adições", async ({ page }) => {
  await page.route(/\.glb/, () => undefined);
  await page.goto("./");
  await page.getByRole("button", { name: "Novidades" }).click();
  const painel = page.locator("#novidades");
  await expect(painel.getByRole("tab", { name: /Novos castelos/ })).toHaveAttribute("aria-selected", "true");
  const novos = await painel.locator(".lista-novos b").allTextContents();
  expect(novos.length).toBeGreaterThan(0);
  await painel.getByRole("tab", { name: /Próximas adições/ }).click();
  const proximos = await painel.locator(".lista-proximos li").allTextContents();
  expect(proximos.length).toBeGreaterThan(0);
  // as duas listas não se repetem: quem tem maqueta não está nas próximas adições
  expect(proximos.filter((n) => novos.includes(n))).toEqual([]);
});

test("atlas: nos Açores aparecem o nome do arquipélago e das nove ilhas", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.route(/\.glb/, () => undefined);
  await page.goto("./");
  await page.waitForSelector('.estado[data-visivel="0"]', { state: "attached", timeout: 150_000 });
  // vista dos Açores (TERRITORIOS.acores) em coordenadas da cena
  await page.evaluate(() => (window as unknown as { __atlas: { getState(): { voar(v: object): void } } }).__atlas.getState().voar({ x: -1000, z: -900, distancia: 880, inclinacao: 36 }));
  await expect(page.locator(".rotulo-arquipelago[data-visivel='1']")).toHaveText("Arquipélago dos Açores", { timeout: 60_000 });
  await page.waitForTimeout(2000);
  const ilhas = await page.$$eval(".rotulo-ilha[data-visivel='1']", (e) => e.map((x) => x.textContent).sort());
  expect(ilhas).toEqual(["Corvo", "Faial", "Flores", "Graciosa", "Pico", "Santa Maria", "São Jorge", "São Miguel", "Terceira"].sort());
});
