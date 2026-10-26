import { useMemo, useState } from "react";
import { BASE } from "../../lib/base";
import { paraCena, projectar, TERRITORIOS, type TerritorioId } from "../../lib/territorios";
import { REGIOES, TIPOLOGIAS, type Regiao, type Tipologia } from "../../lib/tipologias";
import { camaraApi } from "../atlas/Camara";
import { useAtlas, visivel, type MonumentoResumo } from "../atlas/store";
import { Ficha } from "./Ficha";

const normalizar = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

export function Interface({ monumentos, vistaInicial }: { monumentos: MonumentoResumo[]; vistaInicial: () => void }) {
  const territorio = useAtlas((s) => s.territorio);
  const carregamento = useAtlas((s) => s.carregamento);
  const selecionado = useAtlas((s) => s.selecionado);
  const m = monumentos.find((x) => x.id === selecionado) ?? null;

  return (
    <div className="interface">
      <header className="marca">
        <a href={BASE} className="marca-nome">Castelos<span>.site</span></a>
        <p className="marca-sub">Atlas 3D dos Castelos e Fortalezas de Portugal</p>
      </header>

      <Pesquisa monumentos={monumentos} />
      <Indice monumentos={monumentos} />

      <nav className="territorios" aria-label="Território">
        {(Object.keys(TERRITORIOS) as TerritorioId[]).map((id) => (
          <button key={id} aria-pressed={territorio === id} onClick={() => useAtlas.getState().setTerritorio(id)}>
            {id === "continente" ? "Continente" : TERRITORIOS[id].nome}
          </button>
        ))}
      </nav>

      <Controlos vistaInicial={vistaInicial} />
      <Escala />

      {m && <Ficha m={m} onFechar={() => useAtlas.getState().selecionar(null)} />}

      <div className="carregamento" data-visivel={carregamento < 1 ? "1" : "0"} aria-live="polite">
        <span>A erguer o relevo…</span>
        <div className="barra"><div style={{ width: `${Math.round(carregamento * 100)}%` }} /></div>
      </div>
    </div>
  );
}

function Pesquisa({ monumentos }: { monumentos: MonumentoResumo[] }) {
  const [q, setQ] = useState("");
  const [aberto, setAberto] = useState(false);
  const res = useMemo(() => {
    const n = normalizar(q.trim());
    if (n.length < 2) return [];
    return monumentos
      .filter((m) => normalizar(`${m.nome} ${m.localidade} ${m.concelho} ${m.distrito}`).includes(n))
      .slice(0, 8);
  }, [q, monumentos]);
  const escolher = (m: MonumentoResumo) => {
    const s = useAtlas.getState();
    if (s.territorio !== m.territorio) s.setTerritorio(m.territorio);
    s.selecionar(m.id);
    setQ("");
    setAberto(false);
  };
  return (
    <div className="pesquisa">
      <input
        type="search"
        placeholder="Procurar castelo, forte, localidade…"
        aria-label="Procurar monumento"
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setAberto(true);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && res[0]) escolher(res[0]);
          if (e.key === "Escape") setAberto(false);
        }}
      />
      {aberto && res.length > 0 && (
        <ul className="resultados" role="listbox">
          {res.map((m) => (
            <li key={m.id}>
              <button onClick={() => escolher(m)}>
                <strong>{m.nome}</strong>
                <span>{m.localidade} · {TIPOLOGIAS[m.tipologia].nome}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Indice({ monumentos }: { monumentos: MonumentoResumo[] }) {
  const aberto = useAtlas((s) => s.indiceAberto);
  const filtrosTipologia = useAtlas((s) => s.filtrosTipologia);
  const filtroRegiao = useAtlas((s) => s.filtroRegiao);
  const territorio = useAtlas((s) => s.territorio);
  const visiveis = monumentos.filter((m) => visivel(m, { filtrosTipologia, filtroRegiao }));
  const presentes = new Set(monumentos.map((m) => m.tipologia));

  const voarRegiao = (r: Regiao | null) => {
    const s = useAtlas.getState();
    s.setFiltroRegiao(r);
    if (!r) return;
    const lista = monumentos.filter((m) => m.regiao === r);
    if (!lista.length) return;
    const terr = lista[0].territorio;
    if (s.territorio !== terr) s.setTerritorio(terr);
    const t = TERRITORIOS[terr];
    const pts = lista.map((m) => paraCena(t, ...projectar(t, m.lon, m.lat)));
    const xs = pts.map((p) => p[0]), zs = pts.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cz = (Math.min(...zs) + Math.max(...zs)) / 2;
    const ext = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs));
    setTimeout(() => useAtlas.getState().voar({ x: cx, z: cz, distancia: ext * 2.1 + 40, inclinacao: 42 }), 60);
  };

  return (
    <>
      <button className="botao-indice" aria-expanded={aberto} onClick={() => useAtlas.getState().setIndiceAberto(!aberto)}>
        Índice <span className="contador">{visiveis.length}</span>
      </button>
      {aberto && (
        <aside className="painel indice" aria-label="Índice de monumentos">
          <div className="painel-cabeca">
            <h2>Índice</h2>
            <button className="fechar" aria-label="Fechar índice" onClick={() => useAtlas.getState().setIndiceAberto(false)}>×</button>
          </div>
          <section>
            <h3>Regiões</h3>
            <div className="chips">
              <button aria-pressed={!filtroRegiao} onClick={() => voarRegiao(null)}>Todas</button>
              {(Object.keys(REGIOES) as Regiao[]).map((r) => (
                <button key={r} aria-pressed={filtroRegiao === r} onClick={() => voarRegiao(r)}>{REGIOES[r]}</button>
              ))}
            </div>
          </section>
          <section>
            <h3>Tipologias</h3>
            <div className="chips">
              {(Object.keys(TIPOLOGIAS) as Tipologia[]).filter((t) => presentes.has(t)).map((t) => (
                <button key={t} aria-pressed={filtrosTipologia.has(t)} onClick={() => useAtlas.getState().alternarTipologia(t)}>
                  {TIPOLOGIAS[t].plural}
                </button>
              ))}
            </div>
          </section>
          <section className="lista">
            {(Object.keys(REGIOES) as Regiao[]).map((r) => {
              const doGrupo = visiveis.filter((m) => m.regiao === r);
              if (!doGrupo.length) return null;
              return (
                <div key={r}>
                  <h3>{REGIOES[r]}</h3>
                  <ul>
                    {doGrupo.map((m) => (
                      <li key={m.id}>
                        <button
                          onClick={() => {
                            const s = useAtlas.getState();
                            if (s.territorio !== m.territorio) s.setTerritorio(m.territorio);
                            s.selecionar(m.id);
                            if (matchMedia("(max-width: 720px)").matches) s.setIndiceAberto(false);
                          }}
                          data-fora={m.territorio !== territorio ? "1" : "0"}
                        >
                          {m.nome}
                          <span>{m.localidade}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </section>
        </aside>
      )}
    </>
  );
}

function Controlos({ vistaInicial }: { vistaInicial: () => void }) {
  const rumo = useAtlas((s) => s.rumoAtual);
  return (
    <div className="controlos" role="toolbar" aria-label="Navegação">
      <button aria-label="Aproximar" title="Aproximar" onClick={() => camaraApi.zoom?.(0.55)}>+</button>
      <button aria-label="Afastar" title="Afastar" onClick={() => camaraApi.zoom?.(1.8)}>−</button>
      <button aria-label="Orientar a norte" title="Orientar a norte" onClick={() => camaraApi.norte?.()}>
        <svg viewBox="0 0 24 24" width="18" height="18" style={{ transform: `rotate(${-rumo}deg)` }} aria-hidden="true">
          <path d="M12 3l4 9h-8z" fill="currentColor" />
          <path d="M12 21l-4-9h8z" fill="currentColor" opacity="0.35" />
        </svg>
      </button>
      <button
        aria-label="Vista inicial"
        title="Vista inicial"
        onClick={() => {
          useAtlas.getState().selecionar(null);
          useAtlas.getState().setFiltroRegiao(null);
          vistaInicial();
        }}
      >
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path d="M4 11l8-7 8 7v9h-5v-6H9v6H4z" fill="none" stroke="currentColor" strokeWidth="1.6" />
        </svg>
      </button>
    </div>
  );
}

const PASSOS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200];

function Escala() {
  const kmPorPx = useAtlas((s) => s.escalaKmPorPx);
  const territorio = useAtlas((s) => s.territorio);
  const alvo = kmPorPx * 110;
  const km = PASSOS.reduce((a, b) => (b <= alvo ? b : a), PASSOS[0]);
  const px = km / kmPorPx;
  return (
    <div className="escala" aria-hidden="true">
      <div className="regua" style={{ width: `${px}px` }} />
      <span>{km < 1 ? `${km * 1000} m` : `${km} km`}</span>
      <span className="projecao">{TERRITORIOS[territorio].rotuloProjecao}</span>
    </div>
  );
}
