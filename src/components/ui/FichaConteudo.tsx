import type { ReactNode } from "react";
import type { Fonte, MonumentoResumo } from "../atlas/store";

const MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"];
export const dataPorExtenso = (iso: string) => {
  const [a, m, d] = iso.split("-").map(Number);
  return `${d} de ${MESES[m - 1]} de ${a}`;
};

export const gms = (n: number, pos: string, neg: string) => {
  const a = Math.abs(n);
  const g = Math.floor(a), mm = Math.floor((a - g) * 60), s = Math.round(((a - g) * 60 - mm) * 60);
  return `${g}° ${String(mm).padStart(2, "0")}′ ${String(s).padStart(2, "0")}″ ${n >= 0 ? pos : neg}`;
};

const ROTULOS_TECNICA = {
  epoca: "Época",
  tipologia: "Tipologia",
  estilo: "Estilo",
  conservacao: "Estado de conservação",
  materiais: "Materiais",
  protecao: "Proteção",
  propriedade: "Propriedade",
  usoAtual: "Uso atual",
} as const;

const Ligacao = ({ f }: { f: Fonte }) => (f.url ? <a href={f.url} rel="noopener" target="_blank">{f.titulo}</a> : <>{f.titulo}</>);

/**
 * Corpo da ficha, com as secções pela ordem das fichas do castillos.site. Serve o painel lateral do
 * atlas (títulos h3) e a página permanente do monumento (títulos h2). Secções sem conteúdo não aparecem.
 */
export function FichaConteudo({ m, nivel = 3 }: { m: MonumentoResumo; nivel?: 2 | 3 }) {
  const c = m.conteudo;
  const H = `h${nivel}` as "h2" | "h3";
  const Seccao = ({ titulo, children }: { titulo: string; children: ReactNode }) => (
    <section className="ficha-seccao">
      <H>{titulo}</H>
      {children}
    </section>
  );
  const tecnica = (Object.keys(ROTULOS_TECNICA) as (keyof typeof ROTULOS_TECNICA)[]).filter((k) => c.fichaTecnica[k]);
  const v = c.visita;
  const temVisita = v && (v.horario || v.entrada || v.acessibilidade || v.visitasGuiadas || v.contactos || v.site);

  return (
    <>
      <div className="ficha-texto">
        {c.resumo ? (
          <p className="ficha-entrada">{c.resumo}</p>
        ) : (
          !m.html && (
            <p className="ficha-entrada">
              A ficha histórica deste monumento está a ser preparada. Só será publicada com fontes documentadas, indicadas no fim desta página, e com a data de verificação.
            </p>
          )
        )}
        {c.outrasDesignacoes.length > 0 && (
          <p className="ficha-designacoes"><em>Também chamado:</em> {c.outrasDesignacoes.join("; ")}</p>
        )}
        {m.notas.map((n) => (
          <p key={n} className="ficha-aviso">{n}</p>
        ))}
      </div>

      {(tecnica.length > 0 || v?.resumo) && (
        <Seccao titulo="Ficha técnica">
          <dl className="ficha-dados">
            {tecnica.map((k) => (
              <div key={k} className="ficha-dados-linha"><dt>{ROTULOS_TECNICA[k]}</dt><dd>{c.fichaTecnica[k]}</dd></div>
            ))}
            {v?.resumo && (
              <div className="ficha-dados-linha">
                <dt>Visita</dt>
                <dd>{v.resumo}{v.site?.url && <> · <a href={v.site.url} rel="noopener" target="_blank">sítio oficial</a></>}</dd>
              </div>
            )}
          </dl>
        </Seccao>
      )}

      {m.html && (
        <Seccao titulo="História">
          <div className="ficha-historia" dangerouslySetInnerHTML={{ __html: m.html }} />
        </Seccao>
      )}

      {c.oQueVer.length > 0 && (
        <Seccao titulo="O que ver">
          <ul className="ficha-lista">
            {c.oQueVer.map((x) => <li key={x.nome}><strong>{x.nome}.</strong> {x.texto}</li>)}
          </ul>
        </Seccao>
      )}

      {c.cronologia.length > 0 && (
        <Seccao titulo="Cronologia">
          <ol className="ficha-cronologia">
            {c.cronologia.map((x) => <li key={x.ano + x.texto}><strong>{x.ano}</strong><span>{x.texto}</span></li>)}
          </ol>
        </Seccao>
      )}

      {c.medidas.length > 0 && (
        <Seccao titulo="Medidas">
          <table className="ficha-medidas">
            <tbody>
              {c.medidas.map((x) => <tr key={x.conceito}><th scope="row">{x.conceito}</th><td>{x.valor}</td></tr>)}
            </tbody>
          </table>
        </Seccao>
      )}

      {c.curiosidades.length > 0 && (
        <Seccao titulo="Sabia que…">
          <ul className="ficha-lista">
            {c.curiosidades.map((x) => <li key={x}>{x}</li>)}
          </ul>
        </Seccao>
      )}

      {c.imagens.length > 0 && (
        <Seccao titulo="Imagens">
          <div className="ficha-imagens">
            {c.imagens.map((x) => (
              <figure key={x.ficheiro}>
                <img src={`${import.meta.env.BASE_URL}${x.ficheiro}`} alt={x.legenda} loading="lazy" />
                <figcaption>{x.legenda} <em>{x.url ? <a href={x.url} rel="noopener" target="_blank">{x.autor}</a> : x.autor}, {x.licenca}</em></figcaption>
              </figure>
            ))}
          </div>
        </Seccao>
      )}

      <Seccao titulo="Onde está">
        <dl className="ficha-dados">
          <div className="ficha-dados-linha"><dt>Localidade</dt><dd>{m.localidade}</dd></div>
          <div className="ficha-dados-linha"><dt>Concelho</dt><dd>{m.concelho}</dd></div>
          <div className="ficha-dados-linha"><dt>Distrito</dt><dd>{m.distrito}</dd></div>
          <div className="ficha-dados-linha">
            <dt>Coordenadas</dt>
            <dd>
              {gms(m.lat, "N", "S")} · {gms(m.lon, "E", "O")}
              {!m.coordenadasVerificadas && <span className="ficha-pendente"> por verificar</span>}
            </dd>
          </div>
          {c.altitude !== undefined && <div className="ficha-dados-linha"><dt>Altitude</dt><dd>{c.altitude.toLocaleString("pt-PT")} m</dd></div>}
          <div className="ficha-dados-linha">
            <dt>Mapa</dt>
            <dd><a href={`https://www.openstreetmap.org/?mlat=${m.lat}&mlon=${m.lon}#map=17/${m.lat}/${m.lon}`} rel="noopener" target="_blank">OpenStreetMap</a></dd>
          </div>
        </dl>
      </Seccao>

      {temVisita && (
        <Seccao titulo="Visitar">
          <dl className="ficha-dados">
            {v.horario && <div className="ficha-dados-linha"><dt>Horário</dt><dd>{v.horario}</dd></div>}
            {v.entrada && <div className="ficha-dados-linha"><dt>Entrada</dt><dd>{v.entrada}</dd></div>}
            {v.acessibilidade && <div className="ficha-dados-linha"><dt>Acessibilidade</dt><dd>{v.acessibilidade}</dd></div>}
            {v.visitasGuiadas && <div className="ficha-dados-linha"><dt>Visitas guiadas</dt><dd>{v.visitasGuiadas}</dd></div>}
            {v.contactos && <div className="ficha-dados-linha"><dt>Contactos</dt><dd>{v.contactos}</dd></div>}
            {v.site && <div className="ficha-dados-linha"><dt>Informação</dt><dd><Ligacao f={v.site} /></dd></div>}
          </dl>
          <p className="ficha-nota-pequena">Horários e condições podem mudar. Confirme junto da entidade gestora antes da visita.</p>
        </Seccao>
      )}

      <Seccao titulo="Fontes">
        {m.fontes.length ? (
          <ul className="ficha-fontes">
            {m.fontes.map((f) => (
              <li key={f.titulo}>
                <Ligacao f={f} />
                {f.consultado && <span className="ficha-consulta"> · consultado em {dataPorExtenso(f.consultado)}</span>}
              </li>
            ))}
          </ul>
        ) : (
          <p className="ficha-vazio">Ainda sem fontes registadas.</p>
        )}
        {c.verificadoEm && <p className="ficha-nota-pequena">Ficha verificada contra as fontes em {dataPorExtenso(c.verificadoEm)}.</p>}
      </Seccao>
    </>
  );
}
