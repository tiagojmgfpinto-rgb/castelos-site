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
