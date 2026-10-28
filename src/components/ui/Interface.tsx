import { useEffect, useMemo, useRef, useState } from "react";
import { BASE } from "../../lib/base";
import { CIDADES } from "../../data/cidades";
import { paraCena, projectar, TERRITORIOS, type TerritorioId } from "../../lib/territorios";
import { REGIOES, TIPOLOGIAS, type Regiao, type Tipologia } from "../../lib/tipologias";
import { camaraApi } from "../atlas/Camara";
import { useAtlas, visivel, type MonumentoResumo } from "../atlas/store";
import { Escudo, IconeTipo, Rosa } from "./Icones";

const normalizar = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

/** "século XII" -> "s. XII", como no índice do castillos.site. */
export const seculoCurto = (s?: string) => (s ? s.replace(/^s[ée]culos?\s+/i, "s. ") : "");

/** Os 18 distritos do Continente e as duas regiões autónomas, pela ordem alfabética. */
const DISTRITOS: { nome: string; chave: string; territorio: TerritorioId }[] = [
  ...["Aveiro", "Beja", "Braga", "Bragança", "Castelo Branco", "Coimbra", "Évora", "Faro", "Guarda", "Leiria", "Lisboa", "Portalegre", "Porto", "Santarém", "Setúbal", "Viana do Castelo", "Vila Real", "Viseu"].map(
    (nome) => ({ nome, chave: nome, territorio: "continente" as TerritorioId }),
  ),
  { nome: "Açores", chave: "Região Autónoma dos Açores", territorio: "acores" },
  { nome: "Madeira", chave: "Região Autónoma da Madeira", territorio: "madeira" },
];

const plural = (n: number) => `${n} ${n === 1 ? "castelo" : "castelos"}`;

export function Interface({ monumentos, vistaInicial }: { monumentos: MonumentoResumo[]; vistaInicial: () => void }) {
  const carregamento = useAtlas((s) => s.carregamento);
  const painel = useAtlas((s) => s.painel);
  const territorio = useAtlas((s) => s.territorio);
  const distritoAtivo = useAtlas((s) => s.distritoAtivo);
  const nav = useRef<HTMLElement>(null);

  // a leitura do canto inferior esquerdo sobe quando a barra salta de linha (ecrãs estreitos)
  useEffect(() => {
    const el = nav.current;
    if (!el) return;
    const ro = new ResizeObserver(() => el.parentElement?.style.setProperty("--alto-ferramentas", `${el.offsetHeight}px`));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const tecla = (e: KeyboardEvent) => {
      if (e.key === "Escape" && useAtlas.getState().painel && !useAtlas.getState().selecionado) useAtlas.setState({ painel: null, indiceAberto: false });
    };
    addEventListener("keydown", tecla);
    return () => removeEventListener("keydown", tecla);
  }, []);

  const centrar = () => {
    const s = useAtlas.getState();
    s.selecionar(null);
    s.setFiltroRegiao(null);
    s.setDistritoAtivo(null);
    if (s.territorio !== "continente") s.setTerritorio("continente");
    else vistaInicial();
  };
  const emVistaGeral = territorio === "continente" && !distritoAtivo;

  return (
    <div className="interface">
      <header className="marca">
        <Escudo />
        <div>
          <a href={BASE}><span className="marca-titulo">Castelos de Portugal</span></a>
        </div>
      </header>

      <nav className="ferramentas" aria-label="Ferramentas do mapa" ref={nav}>
        <button className="botao" type="button" aria-expanded={painel === "indice"} aria-controls="indice" onClick={() => useAtlas.getState().alternarPainel("indice")}>
          <svg viewBox="0 0 24 24"><path d="M5 6h14M5 12h14M5 18h9" /></svg>
          <span>Índice</span>
          <b className="conta">{monumentos.length}</b>
        </button>
        <div className="grupo" role="group" aria-label="Enquadramentos">
          <button className="botao" type="button" aria-pressed={false} title="Ver Portugal Continental inteiro" onClick={centrar} data-ativo={emVistaGeral ? "1" : "0"}>
            Centrar
          </button>
          <button className="botao" type="button" aria-expanded={painel === "distritos"} aria-controls="distritos" onClick={() => useAtlas.getState().alternarPainel("distritos")}>
            Distritos
            <svg className="flecha" viewBox="0 0 24 24"><path d="M7 10l5 5 5-5" /></svg>
          </button>
        </div>
        <button className="botao" type="button" aria-expanded={painel === "fontes"} aria-controls="fontes" title="Fontes cartográficas e método" onClick={() => useAtlas.getState().alternarPainel("fontes")}>
          <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.4v.6" /></svg>
          <span>Fontes</span>
        </button>
      </nav>

      {painel === "indice" && <Indice monumentos={monumentos} />}
      {painel === "distritos" && <Distritos monumentos={monumentos} />}
      {painel === "fontes" && <Fontes />}

      <Leitura />

      <div className="estado" data-visivel={carregamento < 1 ? "1" : "0"} aria-live="polite">
        <span className="estado-roda" />
        <span>A erguer o relevo…</span>
      </div>
    </div>
  );
}

function Fechar({ rotulo }: { rotulo: string }) {
  return (
    <button className="fechar" type="button" aria-label={rotulo} onClick={() => useAtlas.setState({ painel: null, indiceAberto: false })}>
      ×
    </button>
  );
}

function Indice({ monumentos }: { monumentos: MonumentoResumo[] }) {
  const [q, setQ] = useState("");
  const filtrosTipologia = useAtlas((s) => s.filtrosTipologia);
  const selecionado = useAtlas((s) => s.selecionado);
  const busca = useRef<HTMLInputElement>(null);
  useEffect(() => busca.current?.focus({ preventScroll: true }), []);

  const contagem = useMemo(() => {
    const c = new Map<Tipologia, number>();
    for (const m of monumentos) c.set(m.tipologia, (c.get(m.tipologia) ?? 0) + 1);
    return c;
  }, [monumentos]);

  const n = normalizar(q.trim());
  const lista = monumentos.filter(
    (m) =>
      visivel(m, { filtrosTipologia, filtroRegiao: null }) &&
      (!n ||
        normalizar(
          `${m.nome} ${m.localidade} ${m.concelho} ${m.distrito} ${REGIOES[m.regiao]} ${TIPOLOGIAS[m.tipologia].nome} ${m.seculoOrigem ?? ""} ${seculoCurto(m.seculoOrigem)}`,
        ).includes(n)),
  );

  const escolher = (m: MonumentoResumo) => {
    const s = useAtlas.getState();
    if (s.territorio !== m.territorio) s.setTerritorio(m.territorio);
    s.selecionar(m.id);
    if (matchMedia("(max-width: 860px)").matches) s.setIndiceAberto(false);
  };

  const todos = filtrosTipologia.size === 0;
  return (
    <aside id="indice" className="painel indice" aria-label="Índice de fortalezas">
      <header>
        <h2>Índice de fortalezas</h2>
        <Fechar rotulo="Fechar o índice" />
      </header>
      <input
        ref={busca}
        id="buscar"
        type="search"
        placeholder="Procurar por nome, distrito, século…"
        autoComplete="off"
        aria-label="Procurar castelo"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && lista[0] && escolher(lista[0])}
      />
      <div className="filtros" role="group" aria-label="Filtrar por tipo de fortificação">
        <button type="button" className="filtro" aria-pressed={todos} onClick={() => useAtlas.setState({ filtrosTipologia: new Set() })}>
          Todos<small>{monumentos.length}</small>
        </button>
        {(Object.keys(TIPOLOGIAS) as Tipologia[])
          .filter((t) => contagem.has(t))
          .map((t) => (
            <button
              key={t}
              type="button"
              className="filtro"
              aria-pressed={filtrosTipologia.has(t)}
              onClick={() => useAtlas.setState({ filtrosTipologia: filtrosTipologia.has(t) && filtrosTipologia.size === 1 ? new Set() : new Set([t]) })}
            >
              <IconeTipo tipo={t} />
              {TIPOLOGIAS[t].plural}
              <small>{contagem.get(t)}</small>
            </button>
          ))}
      </div>
      <div className="lista">
        {lista.length === 0 && <p className="vazio">Nenhum monumento corresponde à pesquisa.</p>}
        {(Object.keys(REGIOES) as Regiao[]).map((r) => {
          const daRegiao = lista.filter((m) => m.regiao === r);
          if (!daRegiao.length) return null;
          const distritos = [...new Set(daRegiao.map((m) => m.distrito))].sort((a, b) => a.localeCompare(b, "pt"));
          return (
            <div key={r}>
              <h3>{REGIOES[r]}</h3>
              {distritos.map((d) => (
                <div key={d}>
                  {r !== "acores" && r !== "madeira" && <h4>{d}</h4>}
                  {daRegiao
                    .filter((m) => m.distrito === d)
                    .map((m) => (
                      <button key={m.id} type="button" aria-current={m.id === selecionado} onClick={() => escolher(m)}>
                        <b>
                          <IconeTipo tipo={m.tipologia} />
                          {m.nome}
                        </b>
                        <i>{seculoCurto(m.seculoOrigem)}</i>
                        <span>
                          {m.localidade} · {TIPOLOGIAS[m.tipologia].nome}
                        </span>
                      </button>
                    ))}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </aside>
  );
}

function Distritos({ monumentos }: { monumentos: MonumentoResumo[] }) {
  const ativo = useAtlas((s) => s.distritoAtivo);

  const voar = (d: (typeof DISTRITOS)[number]) => {
    const s = useAtlas.getState();
    s.selecionar(null);
    s.setDistritoAtivo(d.chave);
    const t = TERRITORIOS[d.territorio];
    if (d.territorio !== "continente") {
      // nas ilhas, o enquadramento é a vista inicial do arquipélago
      if (s.territorio !== d.territorio) s.setTerritorio(d.territorio);
      else s.voar({ ...vistaDe(d.territorio) });
      return;
    }
    if (s.territorio !== "continente") s.setTerritorio("continente");
    const capital = CIDADES.find((c) => c.nome === d.nome);
    const pts = monumentos.filter((m) => m.distrito === d.chave).map((m) => paraCena(t, ...projectar(t, m.lon, m.lat)));
    if (capital) pts.push(paraCena(t, ...projectar(t, capital.lon, capital.lat)));
    if (!pts.length) return;
    const xs = pts.map((p) => p[0]), zs = pts.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cz = (Math.min(...zs) + Math.max(...zs)) / 2;
    const ext = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs));
    setTimeout(() => useAtlas.getState().voar({ x: cx, z: cz + ext * 0.12, distancia: Math.max(ext * 2.6, 70) + 60, inclinacao: 38 }), 60);
  };

  return (
    <aside id="distritos" className="painel painel-distritos" aria-label="Enquadrar um distrito">
      <header>
        <h2>Distritos</h2>
        <Fechar rotulo="Fechar" />
      </header>
      <div className="lista-distritos">
        {DISTRITOS.map((d) => (
          <button key={d.chave} type="button" aria-pressed={ativo === d.chave} onClick={() => voar(d)}>
            {d.nome}
            <small>{plural(monumentos.filter((m) => m.distrito === d.chave).length)}</small>
          </button>
        ))}
      </div>
    </aside>
  );
}

function vistaDe(t: TerritorioId) {
  const v = TERRITORIOS[t].vista;
  const [x, z] = paraCena(TERRITORIOS[t], v.x, v.y);
  return { x, z, distancia: v.distancia, inclinacao: v.inclinacao, rumo: v.rumo };
}

function Fontes() {
  return (
    <aside id="fontes" className="painel fontes" aria-label="Fontes e método">
      <header>
        <h2>Fontes e método</h2>
        <Fechar rotulo="Fechar" />
      </header>
      <p>
        O mapa é uma maqueta em relevo gerada a partir de dados abertos e projetada no sistema oficial português, <em>PT-TM06/ETRS89</em> (Transversa de Mercator,
        elipsoide GRS80); os Açores e a Madeira usam o <em>PTRA08</em>. Relevo, fronteiras e castelos passam pela mesma projeção: cada fortaleza cai onde dizem as
        suas coordenadas GPS.
      </p>
      <dl>
        <div><dt>Relevo</dt><dd>Copernicus DEM GLO-30 (Agência Espacial Europeia, via AWS Open Data)</dd></div>
        <div><dt>Costa e fronteiras</dt><dd>Natural Earth 1:10m (domínio público)</dd></div>
        <div><dt>Limites dos distritos</dt><dd>Natural Earth, provisório até à CAOP da DGT</dd></div>
        <div><dt>Rios</dt><dd>Natural Earth 1:10m (domínio público)</dd></div>
      </dl>
      <p className="nota">
        As maquetas dos castelos são construídas com scripts de Python em Blender à escala 1:1 (1 unidade = 1 m) a partir do LiDAR e dos ortofotos da DGT, de plantas e
        de fotografias; as alturas não publicadas são estimativas e assim se indica em cada ficha.
      </p>
      <p className="creditos"><a href={`${BASE}creditos/`}>Créditos e licenças</a></p>
    </aside>
  );
}

const PASSOS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200];

function Leitura() {
  const kmPorPx = useAtlas((s) => s.escalaKmPorPx);
  const rumo = useAtlas((s) => s.rumoAtual);
  const cursor = useAtlas((s) => s.cursor);
  const territorio = useAtlas((s) => s.territorio);
  const alvo = kmPorPx * 85;
  const km = PASSOS.reduce((a, b) => (b <= alvo ? b : a), PASSOS[0]);
  const px = km / kmPorPx;
  return (
    <div className="leitura">
      <Rosa rumo={rumo} onClick={() => camaraApi.norte?.()} />
      <div className="escala" aria-hidden="true">
        <i style={{ width: `${px}px` }} />
        <span>{km < 1 ? `${km * 1000} m` : `${km} km`}</span>
      </div>
      <div className="coordenadas" aria-hidden="true">{cursor || TERRITORIOS[territorio].rotuloProjecao}</div>
    </div>
  );
}
