import type { Tipologia } from "../../lib/tipologias";

// Desenhos pequenos da interface: escudo da marca, ícone de cada tipologia e rosa dos ventos.

export function Escudo() {
  return (
    <svg className="escudo" viewBox="0 0 48 56" aria-hidden="true">
      <path className="e-campo" d="M3 3h42v26c0 14-9 21-21 25C12 50 3 43 3 29z" />
      {/* torre de três ameias com porta, como nas armas dos castelos */}
      <path className="e-torre" fillRule="evenodd" d="M14 11h4.5v4h3v-4h5v4h3v-4H34v9h-2.2v21H16.2V20H14zM21.2 41v-6.5a2.8 2.8 0 0 1 5.6 0V41z" />
    </svg>
  );
}

type Forma = "torre" | "muralha" | "baluarte" | "costeira";
const FORMA: Record<Tipologia, Forma> = {
  castelo: "torre",
  "cerca-urbana": "muralha",
  fortaleza: "baluarte",
  "praca-forte": "baluarte",
  abaluartada: "baluarte",
  forte: "baluarte",
  bateria: "baluarte",
  costeira: "costeira",
  outra: "torre",
};

// fortaleza abaluartada: pentágono com cinco baluartes em ponta
const ESTRELA = (() => {
  const pts: string[] = [];
  for (let i = 0; i < 10; i++) {
    const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
    const r = i % 2 === 0 ? 11 : 6.2;
    pts.push(`${(12 + Math.cos(a) * r).toFixed(2)} ${(15 + Math.sin(a) * r).toFixed(2)}`);
  }
  return `M${pts.join("L")}Z`;
})();

const traco = { fill: "var(--papel-claro)", stroke: "currentColor", strokeWidth: 1.3, strokeLinejoin: "round" as const };

export function IconeTipo({ tipo }: { tipo: Tipologia }) {
  const f = FORMA[tipo];
  return (
    <span className="icone-tipo" aria-hidden="true">
      <svg viewBox="0 0 24 27" focusable="false">
        {f === "torre" && (
          <>
            <path {...traco} d="M6 26V9h2.4V6H11v3h2V6h2.6v3H18v17z" />
            <path d="M10 26v-4.5a2 2 0 0 1 4 0V26z" fill="currentColor" />
            <path d="M12 6V1.2" stroke="currentColor" strokeWidth="1.1" />
            <path d="M12 1.2l4.2 1.4L12 4z" fill="var(--gules)" />
          </>
        )}
        {f === "muralha" && (
          <>
            <path {...traco} d="M1.5 26V14h2.2v-2.5h2.2V14h2.4v-2.5h2.2V14h3v-2.5h2.2V14h2.4v-2.5H20V14h2.5v12z" />
            <path {...traco} d="M8.5 26V8.5h2V6h3v2.5h2V26" />
            <path d="M10.4 26v-3.4a1.6 1.6 0 0 1 3.2 0V26z" fill="currentColor" />
          </>
        )}
        {f === "baluarte" && (
          <>
            <path {...traco} d={ESTRELA} />
            <circle cx="12" cy="15" r="3" fill="currentColor" />
          </>
        )}
        {f === "costeira" && (
          <>
            <path {...traco} d="M8 21V9h1.8V6.5h1.6V9h1.2V6.5h1.6V9H16v12z" />
            <path d="M1.5 23.5c2.2 0 2.2-1.6 4.4-1.6s2.2 1.6 4.4 1.6 2.2-1.6 4.4-1.6 2.2 1.6 4.4 1.6 2.2-1.6 3.4-1.6M1.5 26.5c2.2 0 2.2-1.6 4.4-1.6s2.2 1.6 4.4 1.6 2.2-1.6 4.4-1.6 2.2 1.6 4.4 1.6 2.2-1.6 3.4-1.6" fill="none" stroke="currentColor" strokeWidth="1.1" />
          </>
        )}
      </svg>
    </span>
  );
}

/** Rosa dos ventos de oito pontas; o norte a vermelho. Roda com o rumo da câmara. */
export function Rosa({ rumo, onClick }: { rumo: number; onClick: () => void }) {
  const ponta = (a: number, comprida: boolean) => {
    const r = comprida ? 27 : 17;
    const w = comprida ? 4.6 : 3.4;
    const rad = (a * Math.PI) / 180;
    const [sx, sy] = [Math.sin(rad), -Math.cos(rad)];
    const [px, py] = [-sy, sx];
    return `M${32 + sx * r} ${32 + sy * r}L${32 + px * w} ${32 + py * w}L32 32L${32 - px * w} ${32 - py * w}Z`;
  };
  return (
    <button className="rosa-botao" type="button" title="Orientar a norte" aria-label="Orientar o mapa a norte" onClick={onClick}>
      <svg className="rosa" viewBox="0 0 64 64" aria-hidden="true">
        <circle className="r-aro" cx="32" cy="32" r="30" />
        <circle className="r-aro2" cx="32" cy="32" r="24" />
        <g style={{ transform: `rotate(${-rumo}deg)`, transformOrigin: "32px 32px" }}>
          {[45, 135, 225, 315].map((a) => <path key={a} className="r-sec" d={ponta(a, false)} />)}
          <path className="r-eo" d={ponta(90, true)} />
          <path className="r-eo" d={ponta(270, true)} />
          <path className="r-s" d={ponta(180, true)} />
          <path className="r-n" d={ponta(0, true)} />
          <text className="r-letra" x="32" y="3.2" textAnchor="middle">N</text>
        </g>
        <circle className="r-eixo" cx="32" cy="32" r="2.6" />
      </svg>
    </button>
  );
}
