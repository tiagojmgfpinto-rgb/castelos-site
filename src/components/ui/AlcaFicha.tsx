import { useEffect, useRef, useState } from "react";

/**
 * Alça para redimensionar a ficha, como no castillos.site: no computador arrasta-se a borda esquerda (largura),
 * no telemóvel a barra de cima (altura). Também se usa pelo teclado (setas) e um duplo clique repõe o tamanho.
 * O tamanho escolhido fica guardado neste navegador.
 */
const CHAVE = "castelos:ficha";
const estreito = () => matchMedia("(max-width: 860px)").matches;

type Medidas = { largura?: number; altura?: number };
const ler = (): Medidas => {
  try {
    return JSON.parse(localStorage.getItem(CHAVE) ?? "{}") as Medidas;
  } catch {
    return {};
  }
};
const guardar = (m: Medidas) => {
  try {
    localStorage.setItem(CHAVE, JSON.stringify(m));
  } catch {
    /* navegação privada: fica só nesta visita */
  }
};

export const limitesLargura = (janela: number) => [380, Math.max(380, Math.min(janela - 320, 1400))] as const;
export const limitesAltura = (janela: number) => [Math.round(janela * 0.3), Math.round(janela * 0.94)] as const;
const limitar = (v: number, [a, b]: readonly [number, number]) => Math.round(Math.min(b, Math.max(a, v)));

function aplicar(m: Medidas) {
  const r = document.documentElement.style;
  if (m.largura) r.setProperty("--largura-ficha", `${limitar(m.largura, limitesLargura(innerWidth))}px`);
  else r.removeProperty("--largura-ficha");
  if (m.altura) r.setProperty("--altura-ficha", `${limitar(m.altura, limitesAltura(innerHeight))}px`);
  else r.removeProperty("--altura-ficha");
}

export function AlcaFicha({ ficha }: { ficha: React.RefObject<HTMLElement | null> }) {
  const medidas = useRef<Medidas>({});
  const [vertical, setVertical] = useState(false);
  const [valor, setValor] = useState(0);

  useEffect(() => {
    medidas.current = ler();
    aplicar(medidas.current);
    const mudar = () => {
      setVertical(estreito());
      aplicar(medidas.current);
    };
    mudar();
    addEventListener("resize", mudar);
    return () => removeEventListener("resize", mudar);
  }, []);

  const atual = () => {
    const b = ficha.current?.getBoundingClientRect();
    return b ? (estreito() ? b.height : b.width) : 0;
  };
  const definir = (v: number) => {
    const m = estreito() ? { ...medidas.current, altura: limitar(v, limitesAltura(innerHeight)) } : { ...medidas.current, largura: limitar(v, limitesLargura(innerWidth)) };
    medidas.current = m;
    aplicar(m);
    setValor(estreito() ? m.altura! : m.largura!);
  };

  const arrastar = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    document.documentElement.dataset.redimensionar = estreito() ? "y" : "x";
    const mover = (ev: PointerEvent) => definir(estreito() ? innerHeight - ev.clientY : innerWidth - ev.clientX);
    const largar = () => {
      el.removeEventListener("pointermove", mover);
      el.removeEventListener("pointerup", largar);
      el.removeEventListener("pointercancel", largar);
      delete document.documentElement.dataset.redimensionar;
      guardar(medidas.current);
    };
    el.addEventListener("pointermove", mover);
    el.addEventListener("pointerup", largar);
    el.addEventListener("pointercancel", largar);
  };

  const tecla = (e: React.KeyboardEvent) => {
    const passo = e.shiftKey ? 120 : 40;
    const mais = vertical ? "ArrowUp" : "ArrowLeft", menos = vertical ? "ArrowDown" : "ArrowRight";
    if (e.key !== mais && e.key !== menos) return;
    e.preventDefault();
    definir(atual() + (e.key === mais ? passo : -passo));
    guardar(medidas.current);
  };

  const repor = () => {
    medidas.current = estreito() ? { ...medidas.current, altura: undefined } : { ...medidas.current, largura: undefined };
    aplicar(medidas.current);
    guardar(medidas.current);
  };

  return (
    <div
      className="ficha-alca"
      role="separator"
      aria-orientation={vertical ? "horizontal" : "vertical"}
      aria-label={vertical ? "Arrastar para mudar a altura da ficha" : "Arrastar para mudar a largura da ficha"}
      aria-valuenow={valor || undefined}
      title="Arrastar para redimensionar · duplo clique repõe"
      tabIndex={0}
      onPointerDown={arrastar}
      onKeyDown={tecla}
      onDoubleClick={repor}
    >
      <span />
    </div>
  );
}
