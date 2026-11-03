"""Castelo de Melgaço: mede as muralhas na superfície do LiDAR (dados/superficie.npz, classes 1, 2 e 6) e grava
as estações em dados/muralhas.json, que o construir.py usa.

- Cerca do castelo: uma estação a cada 2° à volta do centro do recinto, com o raio da face de dentro e da face de
  fora (no topo), a cota do adarve, a do parapeito, a do chão de dentro e a do chão de fora.
- Muralha da vila: segue-se uma linha aproximada (desenhada sobre o relevo) e, em cada metro, o perfil
  transversal dá as duas faces, o topo e o chão dos dois lados.

    python pipeline/maquetas/melgaco/tracar.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
S = np.load(AQUI / "dados" / "superficie.npz")
SP, SX0, SY0 = float(S["passo"]), float(S["x0"]), float(S["y0"])
SUP = S["z"] / 100.0 + float(S["base"])
L = np.load(AQUI / "dados" / "lidar.npz")
LP, LX0, LY0 = float(L["passo"]), float(L["x0"]), float(L["y0"])
MDT = L["mdt"] / 100.0 + float(L["base"])

CENTRO = (-0.56, -1.09)  # centro aproximado do recinto


def _amostra(G, x0, y0, p, x, y):
    xa, ya = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    r = ndi.map_coordinates(G, [(ya - y0) / p, (xa - x0) / p], order=1, mode="nearest")
    return r if np.ndim(x) else float(r[0])


def sup(x, y):
    return _amostra(SUP, SX0, SY0, SP, x, y)


def mdt(x, y):
    return _amostra(MDT, LX0, LY0, LP, x, y)


def cerca():
    cx, cy = CENTRO
    est = []
    r = np.arange(10, 42, 0.1)
    for az in np.arange(0, 360, 2.0):
        d = np.array([np.sin(np.radians(az)), np.cos(np.radians(az))])
        z = sup(cx + r * d[0], cy + r * d[1])
        i0 = next((i for i in range(5, len(r)) if z[i] - z[i - 5] > 1.2), None)
        if i0 is None:
            est.append(None)
            continue
        i1 = next((i for i in range(i0 + 3, len(r) - 5) if z[i] - z[i + 5] > 1.5), None)
        if i1 is None:
            est.append(None)
            continue
        topo = z[i0 + 1:i1]
        rin, rout = float(r[i0]), float(r[i1])
        # adarve: mediana da metade de dentro; parapeito: máximo dos 0,6 m de fora
        meio = max(1, len(topo) // 2)
        adarve = float(np.median(topo[:meio]))
        parap = float(topo[-6:].max())
        zin = float(np.median(z[max(i0 - 15, 0):i0 - 4]))
        po = np.array([cx, cy]) + d * (rout + 1.5)
        zout = float(mdt(po[0], po[1]))
        est.append(dict(az=float(az), rin=round(rin, 2), rout=round(rout, 2), adarve=round(adarve, 2),
                        parapeito=round(parap, 2), chao_dentro=round(zin, 2), chao_fora=round(zout, 2)))
    return est


def muralha_vila(pts, passo=1.0):
    """Estações ao longo do eixo da muralha (pontos [x, y, largura] lidos nos perfis transversais da superfície):
    topo = mediana da faixa central; chão de fora e de dentro = mínimo da superfície entre 0,5 e 3 m de cada face."""
    P = np.array(pts, float)
    seg = np.diff(P[:, :2], axis=0)
    sl = np.hypot(*seg.T)
    cum = np.r_[0, np.cumsum(sl)]
    out = []
    for s in np.arange(0, cum[-1] + 1e-6, passo):
        i = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
        t = (s - cum[i]) / sl[i]
        p = P[i, :2] + seg[i] * t
        w = P[i, 2] + (P[i + 1, 2] - P[i, 2]) * t
        d = seg[i] / sl[i]
        n = np.array([-d[1], d[0]])  # à esquerda do sentido da linha = dentro da vila
        oc = np.arange(-w / 2 + 0.5, w / 2 - 0.5 + 1e-6, 0.25)
        q = p[None, :] + oc[:, None] * n[None, :]
        topo = float(np.median(sup(q[:, 0], q[:, 1])))
        of = np.arange(w / 2 + 0.5, w / 2 + 3.0, 0.25)
        qf = p[None, :] - of[:, None] * n[None, :]
        qd = p[None, :] + of[:, None] * n[None, :]
        out.append(dict(s=round(float(s), 1), x=round(float(p[0]), 2), y=round(float(p[1]), 2), largura=round(float(w), 2),
                        nx=round(float(n[0]), 4), ny=round(float(n[1]), 4), topo=round(topo, 2),
                        chao_fora=round(float(np.min(sup(qf[:, 0], qf[:, 1]))), 2),
                        chao_dentro=round(float(np.min(sup(qd[:, 0], qd[:, 1]))), 2)))
    return out


# Eixo da muralha da vila [x, y, largura no topo], lido nos perfis transversais da superfície do LiDAR (a vila fica à
# esquerda): sai da torre noroeste do castelo, segue a norte, dobra a noroeste e desce a sudoeste até à Porta de Baixo.
VILA = [[-29.5, 25.5, 3.4], [-33.0, 26.6, 3.4], [-36.7, 28.0, 3.2], [-40.6, 29.4, 3.0], [-44.5, 30.2, 3.0],
        [-48.5, 30.7, 3.0], [-52.4, 31.0, 3.2], [-56.4, 31.6, 3.2], [-60.4, 31.9, 3.2], [-64.5, 32.3, 3.0],
        [-68.5, 32.5, 3.0], [-72.6, 32.1, 3.2], [-76.3, 30.3, 3.4], [-79.0, 27.3, 3.6], [-80.1, 24.0, 3.7],
        [-81.2, 20.0, 3.8], [-82.0, 16.0, 3.7], [-82.9, 12.0, 3.6], [-84.1, 8.0, 3.2], [-86.1, 4.0, 3.2],
        [-87.4, 0.0, 3.6], [-88.7, -4.0, 3.0], [-90.8, -8.0, 3.5], [-92.9, -12.0, 3.3], [-95.2, -16.0, 3.8],
        [-96.4, -19.0, 3.5], [-97.0, -20.3, 3.5]]


def main():
    dados = {"cerca": cerca(), "vila": muralha_vila(VILA),
             "nota": "Medido na superfície do LiDAR 2025 (classes 1, 2 e 6, 25 cm). Cotas em metros (altitude)."}
    (AQUI / "dados" / "muralhas.json").write_text(json.dumps(dados, ensure_ascii=False, indent=0))
    for e in dados["vila"][::4]:
        print(e)


if __name__ == "__main__":
    main()
