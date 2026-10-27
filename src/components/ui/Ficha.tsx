import { lazy, Suspense, useEffect, useRef } from "react";
import { BASE } from "../../lib/base";
import { caminhoFicha } from "../../lib/rotas";
import { REGIOES, TIPOLOGIAS } from "../../lib/tipologias";
import { useAtlas, type MonumentoResumo } from "../atlas/store";
import { dadosMaqueta } from "../maqueta/dados";
import { FichaConteudo } from "./FichaConteudo";

const Maqueta = lazy(() => import("../maqueta/Maqueta"));

/** Ficha completa num painel lateral: o mapa continua visível (e navegável) à esquerda. */
export function Ficha({ m, monumentos, onFechar }: { m: MonumentoResumo; monumentos: MonumentoResumo[]; onFechar: () => void }) {
  const atual = m.maquetas.find((x) => x.tipo === "estado-atual");
  const reconst = m.maquetas.filter((x) => x.tipo === "reconstituicao");
  const dados = dadosMaqueta(m.id);
  const vizinhos = monumentos.filter((x) => x.regiao === m.regiao && x.id !== m.id).slice(0, 8);
  const corpo = useRef<HTMLDivElement>(null);

  useEffect(() => {
    corpo.current?.scrollTo({ top: 0 });
    const tecla = (e: KeyboardEvent) => e.key === "Escape" && !document.querySelector(".maqueta--ampliada, .maqueta-elemento") && onFechar();
    addEventListener("keydown", tecla);
    return () => removeEventListener("keydown", tecla);
  }, [m.id, onFechar]);

  const local = [m.localidade, m.concelho !== m.localidade ? m.concelho : null, m.distrito, REGIOES[m.regiao]].filter(Boolean).join(" · ");

  return (
    <aside className="ficha" aria-label={`Ficha: ${m.nome}`}>
      <header className="ficha-cabeca">
        <p className="ficha-local">{local}</p>
        <h2>{m.nome}</h2>
        {m.conteudo.subtitulo && <p className="ficha-subtitulo">{m.conteudo.subtitulo}</p>}
        <button className="fechar" aria-label="Fechar ficha" title="Fechar (Esc)" onClick={onFechar}>×</button>
      </header>

      <div className="ficha-corpo" ref={corpo}>
        <ul className="ficha-etiquetas">
          {m.conteudo.estado && <li data-estado="conservacao">{m.conteudo.estado}</li>}
          <li>{TIPOLOGIAS[m.tipologia].nome}</li>
          {m.seculoOrigem && <li>{m.seculoOrigem}</li>}
          <li data-estado={m.estadoFicha}>{m.estadoFicha === "publicada" ? "Ficha publicada" : "Ficha em preparação"}</li>
        </ul>

        <figure className="ficha-maqueta">
          {atual && dados ? (
            <Suspense fallback={<div className="ficha-maqueta-palco"><p>A preparar a maqueta…</p></div>}>
              <Maqueta url={`${BASE}${atual.ficheiro}`} capa={atual.capa} dados={dados} provisoria={atual.provisoria} onVoltarMapa={onFechar} />
            </Suspense>
          ) : (
            <>
              <div className="ficha-maqueta-palco">
                <p>
                  <strong>Maqueta 3D em preparação</strong>
                  Só aparece aqui depois de modelada à escala a partir de plantas, ortofotos e fotografias.
                </p>
              </div>
              <div className="ficha-maqueta-botoes" role="toolbar" aria-label="Maqueta">
                {["Giro", "Placas", "Enquadrar", "Ampliar"].map((b) => (
                  <button key={b} disabled>{b}</button>
                ))}
              </div>
            </>
          )}
          {reconst.length > 0 && (
            <figcaption>Reconstituições disponíveis: {reconst.map((r) => r.epoca ?? "sem época").join(", ")}</figcaption>
          )}
        </figure>

        <FichaConteudo m={m} />

        {vizinhos.length > 0 && (
          <section>
            <h3>Mais castelos em {REGIOES[m.regiao]}</h3>
            <ul className="ficha-vizinhos">
              {vizinhos.map((x) => (
                <li key={x.id}>
                  <button onClick={() => useAtlas.getState().selecionar(x.id)}>
                    {x.nome}
                    <span>{x.localidade}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <p className="ficha-permanente">
          <a href={`${BASE}${caminhoFicha(m)}`}>Endereço permanente desta ficha</a>
        </p>
      </div>
    </aside>
  );
}
