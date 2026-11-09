/**
 * Visor de imagens (lightbox) das galerias das fichas. Sem JavaScript, cada miniatura continua a ser uma
 * ligação para a imagem; com ele, abre-se por cima da página num <dialog>, com legenda e créditos, setas,
 * teclado (← →, Esc) e deslize no telemóvel. Funciona por delegação: serve a ficha do atlas (React) e a
 * página estática do monumento, chamando instalarLightbox() uma vez em cada.
 */

let instalado = false;

interface Item {
  src: string;
  alt: string;
  legenda: HTMLElement | null;
}

export function instalarLightbox() {
  if (instalado || typeof document === "undefined") return;
  instalado = true;

  const dialogo = document.createElement("dialog");
  dialogo.className = "lightbox";
  dialogo.setAttribute("aria-label", "Imagem ampliada");
  dialogo.innerHTML = `
    <figure class="lightbox-figura">
      <img class="lightbox-imagem" alt="" />
      <figcaption class="lightbox-legenda"></figcaption>
    </figure>
    <button type="button" class="lightbox-botao lightbox-anterior" aria-label="Imagem anterior">‹</button>
    <button type="button" class="lightbox-botao lightbox-seguinte" aria-label="Imagem seguinte">›</button>
    <button type="button" class="lightbox-botao lightbox-fechar" aria-label="Fechar">×</button>
    <span class="lightbox-conta" aria-live="polite"></span>`;
  document.body.append(dialogo);
  const img = dialogo.querySelector<HTMLImageElement>(".lightbox-imagem")!;
  const legenda = dialogo.querySelector<HTMLElement>(".lightbox-legenda")!;
  const conta = dialogo.querySelector<HTMLElement>(".lightbox-conta")!;
  let itens: Item[] = [];
  let atual = 0;

  const mostrar = (i: number) => {
    atual = (i + itens.length) % itens.length;
    const it = itens[atual];
    img.src = it.src;
    img.alt = it.alt;
    legenda.replaceChildren(...(it.legenda ? [...it.legenda.cloneNode(true).childNodes] : []));
    conta.textContent = itens.length > 1 ? `${atual + 1} / ${itens.length}` : "";
    dialogo.dataset.varias = itens.length > 1 ? "1" : "0";
  };

  document.addEventListener("click", (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = (e.target as Element).closest?.<HTMLAnchorElement>(".ficha-imagens figure > a");
    if (!a) return;
    const galeria = a.closest(".ficha-imagens")!;
    const ligacoes = [...galeria.querySelectorAll<HTMLAnchorElement>("figure > a")];
    itens = ligacoes.map((l) => ({
      src: l.href,
      alt: l.querySelector("img")?.alt ?? "",
      legenda: l.closest("figure")?.querySelector("figcaption") ?? null,
    }));
    e.preventDefault();
    mostrar(ligacoes.indexOf(a));
    dialogo.showModal();
  });

  dialogo.querySelector(".lightbox-anterior")!.addEventListener("click", () => mostrar(atual - 1));
  dialogo.querySelector(".lightbox-seguinte")!.addEventListener("click", () => mostrar(atual + 1));
  dialogo.querySelector(".lightbox-fechar")!.addEventListener("click", () => dialogo.close());
  // clicar fora da imagem e da legenda fecha
  dialogo.addEventListener("click", (e) => {
    if (e.target === dialogo || (e.target as Element).classList.contains("lightbox-figura")) dialogo.close();
  });
  dialogo.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") mostrar(atual - 1);
    else if (e.key === "ArrowRight") mostrar(atual + 1);
    else return;
    e.preventDefault();
  });
  // Esc fecha o visor mas não a ficha que está por baixo
  dialogo.addEventListener("cancel", (e) => e.stopPropagation());
  dialogo.addEventListener("keydown", (e) => e.key === "Escape" && e.stopPropagation());

  let x0: number | null = null;
  dialogo.addEventListener("pointerdown", (e) => (x0 = e.pointerType === "mouse" ? null : e.clientX));
  dialogo.addEventListener("pointerup", (e) => {
    if (x0 === null) return;
    const dx = e.clientX - x0;
    x0 = null;
    if (Math.abs(dx) > 50 && itens.length > 1) mostrar(atual + (dx < 0 ? 1 : -1));
  });
  dialogo.addEventListener("close", () => img.removeAttribute("src"));
}
