import { BASE } from "../../lib/base";
import { caminhoFicha } from "../../lib/rotas";
import { REGIOES, TIPOLOGIAS } from "../../lib/tipologias";
import type { MonumentoResumo } from "../atlas/store";

const fmt = (n: number, pos: string, neg: string) => `${Math.abs(n).toFixed(4)}° ${n >= 0 ? pos : neg}`;

export function Ficha({ m, onFechar }: { m: MonumentoResumo; onFechar: () => void }) {
  const atual = m.maquetas.find((x) => x.tipo === "estado-atual");
  const reconst = m.maquetas.filter((x) => x.tipo === "reconstituicao");
  return (
    <aside className="painel ficha" aria-label={`Ficha: ${m.nome}`}>
      <div className="painel-cabeca">
        <p className="ficha-tipo">{TIPOLOGIAS[m.tipologia].nome}</p>
        <button className="fechar" aria-label="Fechar ficha" onClick={onFechar}>×</button>
      </div>
      <h2>{m.nome}</h2>
      <p className="ficha-local">
        {m.localidade}
        {m.concelho !== m.localidade ? `, ${m.concelho}` : ""} · {m.distrito} · {REGIOES[m.regiao]}
      </p>

      <dl className="ficha-dados">
        {m.seculoOrigem && (<><dt>Origem</dt><dd>{m.seculoOrigem}</dd></>)}
        <dt>Coordenadas</dt>
        <dd>
          {fmt(m.lat, "N", "S")}, {fmt(m.lon, "E", "O")}
          {!m.coordenadasVerificadas && <span className="aviso"> por verificar</span>}
        </dd>
        <dt>Maqueta 3D</dt>
        <dd>
          {atual ? "Estado atual disponível" : "Em preparação"}
          {reconst.length > 0 && ` · reconstituição ${reconst.map((r) => r.epoca).join(", ")}`}
        </dd>
      </dl>

      {m.estadoFicha === "em-preparacao" && (
        <p className="ficha-nota">
          A ficha histórica deste monumento está em preparação. Só será publicada com fontes documentadas.
        </p>
      )}
      {m.notas.map((n) => (
        <p key={n} className="ficha-nota aviso">{n}</p>
      ))}

      <a className="ficha-ligacao" href={`${BASE}${caminhoFicha(m)}`}>Abrir a ficha completa →</a>
    </aside>
  );
}
