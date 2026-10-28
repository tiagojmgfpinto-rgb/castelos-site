import { lazy, Suspense, useEffect, useRef } from "react";
import { BASE } from "../../lib/base";
import { caminhoFicha } from "../../lib/rotas";
import { REGIOES, TIPOLOGIAS } from "../../lib/tipologias";
import { useAtlas, type MonumentoResumo } from "../atlas/store";
import { dadosMaqueta } from "../maqueta/dados";
import { FichaConteudo, gms } from "./FichaConteudo";

const Maqueta = lazy(() => import("../maqueta/Maqueta"));

/** Ficha completa num painel lateral: o mapa continua visível (e navegável) à esquerda. */
export function Ficha({ m, monumentos, onFechar }: { m: MonumentoResumo; monumentos: MonumentoResumo[]; onFechar: () => void }) {
  const atual = m.maquetas.find((x) => x.tipo === "estado-atual");
  const reconst = m.maquetas.filter((x) => x.tipo === "reconstituicao");
  const dados = dadosMaqueta(m.id);
  const vizinhos = monumentos.filter((x) => x.regiao === m.regiao && x.id !== m.id).slice(0, 8);
  const corpo = useRef<HTMLElement>(null);

  useEffect(() => {
    corpo.current?.scrollTo({ top: 0 });
    const tecla = (e: KeyboardEvent) => e.key === "Escape" && !document.querySelector(".maqueta--ampliada, .maqueta-elemento") && onFechar();
    addEventListener("keydown", tecla);
    return () => removeEventListener("keydown", tecla);
  }, [m.id, onFechar]);

  const local = [m.localidade, m.concelho !== m.localidade ? m.concelho : null, m.distrito.replace(/^Região Autónoma d[aos]+ /, ""), REGIOES[m.regiao]].filter((x, i, a) => x && a.indexOf(x) === i).join(" · ");

  return (
    <aside className="ficha" aria-label={`Ficha: ${m.nome}`} ref={corpo}>
      <div className="ficha-moldura">
        <div className="ficha-banda">
          <button className="fechar" type="button" aria-label="Fechar a ficha" title="Fechar (Esc)" onClick={onFechar}>×</button>
          <p className="ficha-lugar">{local}</p>
          <h2 className="ficha-nome">{m.nome}</h2>
        </div>

        <header className="ficha-cabecalho">
          {m.conteudo.outrasDesignacoes.length > 0 && <p className="ficha-tambem">Também: {m.conteudo.outrasDesignacoes.join(", ")}</p>}
          {m.conteudo.subtitulo && <p className="ficha-tambem">{m.conteudo.subtitulo}</p>}
          <ul className="ficha-chips">
            {m.seculoOrigem && <li>{m.seculoOrigem.replace(/^s/, "S")}</li>}
            <li>{TIPOLOGIAS[m.tipologia].nome}</li>
            {m.conteudo.estado && <li className="estado-chip">{m.conteudo.estado}</li>}
            {m.estadoFicha !== "publicada" && <li className="estado-chip estado-pendente">Ficha em preparação</li>}
          </ul>
        </header>

        <section className="vitrina" aria-label="Maqueta 3D interativa">
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
                {["Giro", "Rótulos", "Enquadrar", "Ampliar"].map((b) => (
                  <button key={b} disabled>{b}</button>
                ))}
              </div>
            </>
          )}
          {reconst.length > 0 && (
            <p className="ficha-nota-pequena">Reconstituições disponíveis: {reconst.map((r) => r.epoca ?? "sem época").join(", ")}</p>
          )}
        </section>

        <div className="ficha-corpo-texto">
          <FichaConteudo m={m} lateral />

          {vizinhos.length > 0 && (
            <section className="ficha-seccao">
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

          <footer className="ficha-pe">
            {m.conteudo.verificadoEm ? `Ficha verificada em ${m.conteudo.verificadoEm.split("-").reverse().join("-")}` : "Ficha em revisão"} · {gms(m.lat, "N", "S")} · {gms(m.lon, "E", "O")}
            <div className="acoes">
              <a href={`${BASE}${caminhoFicha(m)}`}>Endereço permanente</a>
              <a href={`https://www.openstreetmap.org/?mlat=${m.lat}&mlon=${m.lon}#map=17/${m.lat}/${m.lon}`} rel="noopener" target="_blank">Abrir no mapa</a>
            </div>
          </footer>
        </div>
      </div>
    </aside>
  );
}
