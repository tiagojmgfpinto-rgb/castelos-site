import { expect, test, type Page } from "@playwright/test";
import { errosDaPagina } from "./ajuda";

// Testes A, B e C do pedido: painel dos adicionados nas últimas 24 h, mapa só com maquetas publicadas e
// Próximas adições. Os dados vêm de dados/adicionados.json, a mesma lista que o painel lê de 5 em 5 minutos.

interface Adicionado { id: string; nome: string; adicionado: string; noMapa: boolean }
const H = 3600 * 1000;

async function dados(page: Page): Promise<Adicionado[]> {
  const r = await page.request.get("dados/adicionados.json");
  expect(r.ok()).toBe(true);
  return r.json();
}

test("A: cada monumento fica 24 h no painel e sai sozinho, sem ser apagado", async ({ page }) => {
  const erros = errosDaPagina(page);
  const lista = await dados(page);
  const alvo = lista[0]; // o mais recente
  const t = Date.parse(alvo.adicionado);
  const companheiros = lista.filter((x) => x.adicionado === alvo.adicionado).length;

  // 23 h 58 min depois de publicado ainda lá está, com o tempo decorrido e ligação para a sua página
  await page.clock.install({ time: t + 24 * H - 2 * 60 * 1000 });
  await page.route(/\.glb/, () => undefined);
  await page.goto("./");
  const painel = page.locator("#recentes");
  if ((await painel.getAttribute("data-minimizado")) === "1") await painel.locator(".recentes-titulo").click();
  const item = painel.locator(`li[data-id="${alvo.id}"]`);
  await expect(item).toBeVisible();
  await expect(item).toContainText(alvo.nome);
  await expect(item.locator("time")).toHaveText("Adicionado há 23 horas");
  expect(await item.locator("a").getAttribute("href")).toMatch(new RegExp(`/${alvo.id}/$`));
  await expect(painel.locator(".conta")).toHaveText(String(lista.filter((x) => t + 24 * H - 2 * 60 * 1000 - Date.parse(x.adicionado) < 24 * H && Date.parse(x.adicionado) <= t).length));

  // passadas as 24 h sai sem recarregar a página; se era o último, o painel desaparece
  await page.clock.fastForward(3 * 60 * 1000);
  await expect(item).toHaveCount(0);
  if (companheiros === lista.filter((x) => Date.parse(x.adicionado) > t - 24 * H).length) await expect(painel).toHaveCount(0);

  // continua nos dados e a página dele continua a existir
  expect((await dados(page)).some((x) => x.id === alvo.id)).toBe(true);
  expect(erros).toEqual([]);
});

test("A: um monumento publicado depois de a página abrir aparece sem recarregar", async ({ page }) => {
  const lista = await dados(page);
  const agora = Date.parse(lista[0].adicionado) + 3 * 24 * H; // ninguém nas últimas 24 h
  await page.clock.install({ time: agora });
  await page.route(/\.glb/, () => undefined);
  await page.goto("./");
  await expect(page.locator("#recentes")).toHaveCount(0);

  // nova publicação: o ficheiro de dados passa a ter um monumento de há 1 hora, sem maqueta
  const novo = { ...lista.find((x) => !x.noMapa)!, adicionado: new Date(agora - H).toISOString() };
  await page.route(/dados\/adicionados\.json/, (r) => r.fulfill({ json: [novo, ...lista.filter((x) => x.id !== novo.id)] }));
  await page.clock.fastForward(5 * 60 * 1000 + 1000);
  const painel = page.locator("#recentes");
  if ((await painel.getAttribute("data-minimizado")) === "1") await painel.locator(".recentes-titulo").click();
  await expect(painel.locator(`li[data-id="${novo.id}"]`)).toContainText("Adicionado há 1 hora");

  // minimizar e fechar
  await painel.locator(".recentes-titulo").click();
  await expect(painel.locator(".recentes-lista")).toHaveCount(0);
  await painel.getByRole("button", { name: /Fechar/ }).click();
  await expect(painel).toHaveCount(0);
});

test("B: o mapa só mostra os monumentos com maqueta concluída e publicada", async ({ page }) => {
  const erros = errosDaPagina(page);
  const lista = await dados(page);
  const noMapa = lista.filter((x) => x.noMapa);
  expect(noMapa.length).toBeGreaterThan(0);
  await page.route(/\.glb/, () => undefined);
  await page.goto("./");
  // um marcador (cartela) por monumento no mapa, e o índice só com esses
  await expect(page.locator(".etiqueta")).toHaveCount(noMapa.length, { timeout: 150_000 });
  await page.getByRole("button", { name: /Índice/ }).click();
  const nomes = await page.locator(".indice .lista button b").allTextContents();
  expect(nomes.map((n) => n.trim()).sort()).toEqual(noMapa.map((x) => x.nome).sort());
  await expect(page.locator(".indice-proximas a")).toHaveText(`Ver as ${lista.length - noMapa.length} próximas adições`);

  // a ligação #id de um monumento sem maqueta leva à página dele, que diz que é uma próxima adição
  const fora = lista.find((x) => !x.noMapa)!;
  await page.goto(`./#${fora.id}`);
  await page.waitForURL(new RegExp(`/${fora.id}/$`));
  await expect(page.locator(".ficha-pagina-proxima")).toContainText("Próxima adição ao mapa");
  expect(erros).toEqual([]);
});

test("C: Próximas adições tem todos os que não estão no mapa, e só esses", async ({ page }) => {
  const lista = await dados(page);
  await page.goto("proximas-adicoes/");
  const ids = await page.locator("[data-proxima]").evaluateAll((els) => els.map((e) => e.getAttribute("data-proxima")));
  expect(ids.sort()).toEqual(lista.filter((x) => !x.noMapa).map((x) => x.id).sort());
  // cada cartão diz em que fase está a maqueta e leva a uma ficha que existe
  await expect(page.locator("[data-proxima] .catalogo-etiqueta").first()).toContainText("Maqueta");
  const href = await page.locator("[data-proxima] a").first().getAttribute("href");
  expect((await page.request.get(href!)).status()).toBe(200);
});
