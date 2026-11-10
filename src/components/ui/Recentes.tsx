import { useEffect, useMemo, useState } from "react";
import { BASE } from "../../lib/base";
import { haQuanto, proximaMudanca, recentes } from "../../lib/estado";
import { caminhoFicha } from "../../lib/rotas";
import type { Tipologia } from "../../lib/tipologias";
import { useAtlas, type MonumentoResumo } from "../atlas/store";
import { IconeTipo } from "./Icones";

/** O que o painel precisa de cada monumento; é também o formato de dados/adicionados.json. */
export interface Adicionado {
  id: string;
  nome: string;
  tipologia: Tipologia;
  localidade: string;
  concelho: string;
  imagem?: string;
  adicionado: string;
  noMapa: boolean;
}

export const paraAdicionado = (m: MonumentoResumo): Adicionado => ({
  id: m.id,
  nome: m.nome,
  tipologia: m.tipologia,
  localidade: m.localidade,
  concelho: m.concelho,
  imagem: m.imagem,
  adicionado: m.adicionado,
  noMapa: m.noMapa,
});

const ATUALIZAR_DADOS_MS = 5 * 60 * 1000; // procura monumentos novos publicados depois de a página abrir
const RELOGIO_MS = 30 * 1000; // refaz o "há 2 horas"

const ler = (s: Storage | undefined, k: string) => {
  try {
    return s?.getItem(k) ?? null;
  } catch {
    return null;
  }
};
const gravar = (s: Storage | undefined, k: string, v: string | null) => {
  try {
    if (v === null) s?.removeItem(k);
    else s?.setItem(k, v);
  } catch {
    /* sem armazenamento: só dura até sair da página */
  }
};
const local = () => (typeof localStorage === "undefined" ? undefined : localStorage);
const sessao = () => (typeof sessionStorage === "undefined" ? undefined : sessionStorage);

/**
 * Painel flutuante à direita com os monumentos publicados nas últimas 24 horas, como os "castillos nuevos" do
 * castillos.site. Cada um sai sozinho 24 h depois da sua publicação, sem recarregar a página; não depende de ter
 * maqueta. Pode minimizar-se (fica uma pastilha com o número) ou fechar-se até haver outro novo.
 */
export function Recentes({ iniciais }: { iniciais: MonumentoResumo[] }) {
  const [lista, setLista] = useState<Adicionado[]>(() => iniciais.map(paraAdicionado));
  const [agora, setAgora] = useState(() => Date.now());
  const [minimizado, setMinimizado] = useState(() => {
    const g = ler(local(), "recentes-minimizado");
    if (g !== null) return g === "1";
    return typeof matchMedia !== "undefined" && matchMedia("(max-width: 860px)").matches; // no telemóvel começa recolhido
  });
  const [fechadoEm, setFechadoEm] = useState(() => ler(sessao(), "recentes-fechado"));
  const selecionado = useAtlas((s) => s.selecionado);
  const painel = useAtlas((s) => s.painel);

  // relógio: de 30 em 30 s e, exatamente, quando alguém expira
  useEffect(() => {
    const id = setInterval(() => setAgora(Date.now()), RELOGIO_MS);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    const prox = proximaMudanca(lista, agora);
    if (prox === null) return;
    const id = setTimeout(() => setAgora(Date.now()), Math.max(0, Math.min(prox - Date.now() + 50, 2 ** 31 - 1)));
    return () => clearTimeout(id);
  }, [lista, agora]);

  // dados: os monumentos publicados depois de a página abrir entram sem recarregar
  useEffect(() => {
    let vivo = true;
    const buscar = async () => {
      try {
        const r = await fetch(`${BASE}dados/adicionados.json?t=${Date.now()}`, { cache: "no-store" });
        if (!r.ok) return;
        const novos = (await r.json()) as Adicionado[];
        if (vivo && Array.isArray(novos) && novos.length) {
          setLista(novos);
          setAgora(Date.now());
        }
      } catch {
        /* sem rede: fica a lista que veio com a página */
      }
    };
    const id = setInterval(buscar, ATUALIZAR_DADOS_MS);
    const aoVoltar = () => document.visibilityState === "visible" && buscar();
    document.addEventListener("visibilitychange", aoVoltar);
    return () => {
      vivo = false;
      clearInterval(id);
      document.removeEventListener("visibilitychange", aoVoltar);
    };
  }, []);

  const visiveis = useMemo(() => recentes(lista, agora), [lista, agora]);
  const maisRecente = visiveis[0]?.adicionado ?? null;
  // fechado só até aparecer um mais recente do que o que havia quando se fechou
  const fechado = !!fechadoEm && !!maisRecente && maisRecente <= fechadoEm;
  // a ficha e os painéis da direita ocupam o mesmo sítio
  const tapado = !!selecionado || painel === "novidades" || painel === "distritos" || painel === "fontes";

  if (!visiveis.length || fechado) return null;

  const alternar = () => {
    gravar(local(), "recentes-minimizado", minimizado ? "0" : "1");
    setMinimizado(!minimizado);
  };
  const fechar = () => {
    gravar(sessao(), "recentes-fechado", maisRecente);
    setFechadoEm(maisRecente);
  };
  const titulo = visiveis.length === 1 ? "1 monumento novo nas últimas 24 horas" : `${visiveis.length} monumentos novos nas últimas 24 horas`;

  return (
    <aside className="recentes" id="recentes" aria-label="Adicionados nas últimas 24 horas" data-minimizado={minimizado ? "1" : "0"} data-tapado={tapado ? "1" : "0"} hidden={tapado}>
      <header>
        <button type="button" className="recentes-titulo" aria-expanded={!minimizado} aria-controls="recentes-lista" title={minimizado ? "Mostrar os adicionados recentemente" : "Minimizar"} onClick={alternar}>
          <span className="recentes-ponto" aria-hidden="true" />
          <span className="recentes-rotulo">Adicionados recentemente</span>
          <b className="conta" title={titulo}>{visiveis.length}</b>
          <svg className="flecha" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 14l5-5 5 5" /></svg>
        </button>
        <button type="button" className="recentes-fechar" aria-label="Fechar os adicionados recentemente" title="Fechar até haver outro novo" onClick={fechar}>
          ×
        </button>
      </header>
      {!minimizado && (
        <>
          <p className="recentes-nota">{titulo}. Cada um fica aqui 24 horas.</p>
          <ul className="recentes-lista" id="recentes-lista">
            {visiveis.map((m) => (
              <li key={m.id} data-id={m.id}>
                <a href={`${BASE}${caminhoFicha(m)}`}>
                  <span className="recentes-imagem">
                    {m.imagem ? <img src={`${BASE}${m.imagem}`} alt="" loading="lazy" decoding="async" /> : <IconeTipo tipo={m.tipologia} />}
                  </span>
                  <span className="recentes-texto">
                    <b>{m.nome}</b>
                    <span>{[m.localidade, m.concelho].filter((x, i, a) => x && a.indexOf(x) === i).join(", ")}</span>
                    <time dateTime={m.adicionado} title={new Date(m.adicionado).toLocaleString("pt-PT")}>
                      Adicionado {haQuanto(agora - Date.parse(m.adicionado))}
                    </time>
                  </span>
                </a>
              </li>
            ))}
          </ul>
        </>
      )}
    </aside>
  );
}
