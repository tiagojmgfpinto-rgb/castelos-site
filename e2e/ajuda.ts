import type { Page } from "@playwright/test";

/** Erros de JavaScript da página, para cada teste confirmar no fim que não houve nenhum. */
export function errosDaPagina(pg: Page) {
  const erros: string[] = [];
  pg.on("pageerror", (e) => erros.push(String(e)));
  return erros;
}

/** Endereços das páginas dos monumentos com maqueta, lidos do catálogo. */
export async function paginasComMaqueta(pg: Page): Promise<string[]> {
  await pg.goto("castelos/");
  return pg.$$eval('[data-cartao][data-maqueta="1"] a', (as) => as.map((a) => new URL((a as HTMLAnchorElement).href).pathname));
}

/** Texto que sai das caixas do visor 3D (cartões das placas, fichas, notas, botões, legenda das épocas). */
export function transbordos(): string[] {
  const r: string[] = [];
  const sel = ".maqueta-placa-cartao, .maqueta-elemento, .maqueta-notas, .maqueta-selo, .maqueta-botoes button, .maqueta-epocas-botoes button, .maqueta-epoca";
  for (const c of document.querySelectorAll<HTMLElement>(sel)) {
    const rc = c.getBoundingClientRect();
    if (c.scrollWidth > c.clientWidth + 1) r.push(`${c.className}: ${(c.textContent ?? "").trim().slice(0, 50)}`);
    for (const f of c.querySelectorAll("*")) {
      const rf = f.getBoundingClientRect();
      if (rf.width && (rf.right > rc.right + 1 || rf.left < rc.left - 1)) r.push(`${c.className} > ${f.tagName}: ${(f.textContent ?? "").trim().slice(0, 50)}`);
    }
  }
  return r;
}

/**
 * Nas 36 h depois de entrar uma maqueta nova, as Novidades abrem sozinhas ao carregar o atlas (Interface.tsx)
 * e tapam o painel dos recentes e parte do mapa. Os testes que não são sobre isso fecham-nas.
 */
export async function fecharNovidadesSozinhas(pg: Page) {
  await pg.getByRole("button", { name: "Novidades" }).waitFor();
  await pg.waitForTimeout(500);
  await pg.evaluate(() => {
    const a = (window as unknown as { __atlas?: { getState(): { painel: string | null }; setState(s: object): void } }).__atlas;
    if (a?.getState().painel === "novidades") a.setState({ painel: null });
  });
}
