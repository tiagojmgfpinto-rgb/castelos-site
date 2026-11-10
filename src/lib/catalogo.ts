/**
 * Catálogo de todos os monumentos (página /castelos/). A página é estática e mostra tudo sem JavaScript;
 * com ele, a pesquisa e os filtros escondem os cartões que não servem e o endereço guarda a escolha
 * (?q=…&regiao=…&tipo=…&maqueta=1&visita=1&ficha=1), para se poder partilhar.
 */

export interface ItemCatalogo {
  /** Texto pesquisável, já normalizado (nome, outras designações, localidade, concelho, distrito). */
  texto: string;
  regiao: string;
  tipologia: string;
  maqueta: boolean;
  visita: boolean;
  ficha: boolean;
}

export interface Criterios {
  q: string;
  regiao: string;
  tipo: string;
  maqueta: boolean;
  visita: boolean;
  ficha: boolean;
}

export const SEM_CRITERIOS: Criterios = { q: "", regiao: "", tipo: "", maqueta: false, visita: false, ficha: false };

/** Minúsculas e sem acentos: «Monção» encontra-se com «moncao». */
export const normalizar = (s: string) =>
  s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/\s+/g, " ").trim();

/** Um item passa quando cumpre todos os critérios; na pesquisa, cada palavra tem de aparecer no texto. */
export function passa(item: ItemCatalogo, c: Criterios): boolean {
  if (c.regiao && item.regiao !== c.regiao) return false;
  if (c.tipo && item.tipologia !== c.tipo) return false;
  if (c.maqueta && !item.maqueta) return false;
  if (c.visita && !item.visita) return false;
  if (c.ficha && !item.ficha) return false;
  const palavras = normalizar(c.q).split(" ").filter(Boolean);
  return palavras.every((p) => item.texto.includes(p));
}

export function lerCriterios(pesquisa: string): Criterios {
  const p = new URLSearchParams(pesquisa);
  return {
    q: (p.get("q") ?? "").slice(0, 100),
    regiao: p.get("regiao") ?? "",
    tipo: p.get("tipo") ?? "",
    maqueta: p.get("maqueta") === "1",
    visita: p.get("visita") === "1",
    ficha: p.get("ficha") === "1",
  };
}

export function escreverCriterios(c: Criterios): string {
  const p = new URLSearchParams();
  if (c.q.trim()) p.set("q", c.q.trim());
  if (c.regiao) p.set("regiao", c.regiao);
  if (c.tipo) p.set("tipo", c.tipo);
  if (c.maqueta) p.set("maqueta", "1");
  if (c.visita) p.set("visita", "1");
  if (c.ficha) p.set("ficha", "1");
  const s = p.toString();
  return s ? `?${s}` : "";
}

/** Liga o formulário aos cartões da página. Chamado uma vez pelo <script> de /castelos/. */
export function instalarCatalogo(raiz: HTMLElement) {
  const form = raiz.querySelector<HTMLFormElement>(".catalogo-filtros");
  if (!form) return;
  const cartoes = [...raiz.querySelectorAll<HTMLElement>("[data-cartao]")].map((el) => ({
    el,
    item: {
      texto: el.dataset.texto ?? "",
      regiao: el.dataset.regiao ?? "",
      tipologia: el.dataset.tipologia ?? "",
      maqueta: el.dataset.maqueta === "1",
      visita: el.dataset.visita === "1",
      ficha: el.dataset.ficha === "1",
    } satisfies ItemCatalogo,
  }));
  const seccoes = [...raiz.querySelectorAll<HTMLElement>("[data-seccao]")];
  const conta = raiz.querySelector<HTMLElement>(".catalogo-conta")!;
  const vazio = raiz.querySelector<HTMLElement>(".catalogo-vazio")!;
  const campo = <T extends HTMLInputElement | HTMLSelectElement>(n: string) => form.elements.namedItem(n) as T;

  const ler = (): Criterios => ({
    q: campo<HTMLInputElement>("q").value,
    regiao: campo<HTMLSelectElement>("regiao").value,
    tipo: campo<HTMLSelectElement>("tipo").value,
    maqueta: campo<HTMLInputElement>("maqueta").checked,
    visita: campo<HTMLInputElement>("visita").checked,
    ficha: campo<HTMLInputElement>("ficha").checked,
  });
  const preencher = (c: Criterios) => {
    campo<HTMLInputElement>("q").value = c.q;
    campo<HTMLSelectElement>("regiao").value = c.regiao;
    campo<HTMLSelectElement>("tipo").value = c.tipo;
    campo<HTMLInputElement>("maqueta").checked = c.maqueta;
    campo<HTMLInputElement>("visita").checked = c.visita;
    campo<HTMLInputElement>("ficha").checked = c.ficha;
  };

  const aplicar = (c: Criterios) => {
    let n = 0;
    for (const { el, item } of cartoes) {
      const ok = passa(item, c);
      el.hidden = !ok;
      if (ok) n++;
    }
    for (const s of seccoes) {
      const vis = s.querySelectorAll("[data-cartao]:not([hidden])").length;
      s.hidden = vis === 0;
      const c = s.querySelector<HTMLElement>(".contagem");
      if (c) c.textContent = String(vis) === c.dataset.total ? vis.toString() : `${vis} de ${c.dataset.total}`;
    }
    conta.textContent = n === cartoes.length ? `${n} fortificações` : `${n} de ${cartoes.length} fortificações`;
    vazio.hidden = n > 0;
  };

  const atualizar = () => {
    const c = ler();
    aplicar(c);
    history.replaceState(null, "", `${location.pathname}${escreverCriterios(c)}${location.hash}`);
  };

  preencher(lerCriterios(location.search));
  aplicar(ler());
  form.hidden = false;
  form.addEventListener("input", atualizar);
  form.addEventListener("change", atualizar);
  form.addEventListener("submit", (e) => e.preventDefault());
  form.addEventListener("reset", () => setTimeout(atualizar));
}
