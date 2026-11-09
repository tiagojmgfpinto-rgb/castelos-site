import { expect, test } from "@playwright/test";
import { errosDaPagina, paginasComMaqueta, transbordos } from "./ajuda";

test("página do monumento: a maqueta 3D carrega, com épocas e lightbox", async ({ page }) => {
  const erros = errosDaPagina(page);
  const [pagina] = (await paginasComMaqueta(page)).filter((p) => p.includes("penela"));
  expect(pagina).toBeTruthy();
  const glb = page.waitForRequest(/\.glb/, { timeout: 60_000 });
  await page.goto(pagina);
  await page.locator(".maqueta").scrollIntoViewIfNeeded();
  // a ilha client:visible tem de hidratar e pedir o GLB (falhou assim no PR #46)
  await glb;
  await expect(page.locator(".maqueta-carregar")).toHaveCount(0, { timeout: 150_000 });

  // épocas: a primeira esconde elementos e mostra a legenda; «Hoje» tira-a
  const epocas = page.locator(".maqueta-epocas-botoes button");
  expect(await epocas.count()).toBeGreaterThan(1);
  await epocas.first().click();
  await expect(page.locator(".maqueta-epoca")).toBeVisible();
  await expect(page.locator(".maqueta-epoca-nota").first()).toContainText("forma de hoje");
  await epocas.last().click();
  await expect(page.locator(".maqueta-epoca")).toHaveCount(0);

  // lightbox da galeria
  const fig = page.locator(".ficha-imagens figure > a").nth(1);
  await fig.scrollIntoViewIfNeeded();
  await fig.click();
  const dialogo = page.locator("dialog.lightbox");
  await expect(dialogo).toHaveJSProperty("open", true);
  const antes = await page.locator(".lightbox-conta").textContent();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator(".lightbox-conta")).not.toHaveText(antes ?? "");
  await page.keyboard.press("Escape");
  await expect(dialogo).toHaveJSProperty("open", false);
  expect(erros).toEqual([]);
});

for (const [nome, viewport] of [["computador", { width: 1280, height: 860 }], ["telemóvel", { width: 375, height: 667 }]] as const) {
  test(`visor 3D: nenhum texto sai das caixas (${nome})`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const paginas = await paginasComMaqueta(page);
    expect(paginas.length).toBeGreaterThan(5);
    // o GLB fica pendente: chega a interface do visor, que é o que se mede, sem esperar pelo desenho
    await page.route(/\.glb/, () => undefined);
    const problemas: string[] = [];
    for (const p of paginas) {
      await page.goto(p);
      await page.locator(".maqueta").scrollIntoViewIfNeeded();
      await expect(page.locator(".maqueta-botoes")).toBeVisible();
      const medir = async () => (await page.evaluate(transbordos)).forEach((x) => problemas.push(`${p}: ${x}`));
      await medir();
      const placas = page.locator(".maqueta-placa-cartao");
      for (let i = 0; i < (await placas.count()); i++) {
        await placas.nth(i).evaluate((e) => (e as HTMLElement).click());
        await medir();
      }
      const epocas = page.locator(".maqueta-epocas-botoes button");
      for (let i = 0; i < (await epocas.count()); i++) {
        await epocas.nth(i).click();
        await medir();
      }
    }
    expect([...new Set(problemas)]).toEqual([]);
  });
}
