"""Forte de São Sebastião: planta do forte medida no LiDAR (dados/superficie.npz) -> dados/planta.json.

Cada contorno é dado à mão com erro de c. 1,5 m (lido nos mapas de cotas) e depois cada lado é encostado à aresta
mais forte do relevo (máximo do gradiente da superfície de 25 cm, mediana 3×3) numa faixa de ±2 m: em cada 25 cm
do lado procura-se esse máximo e ajusta-se uma reta robusta; os vértices são as interseções das retas vizinhas.
O resultado é a aresta de cima da face de cada muro (a magistral), com erro de 0,1 a 0,3 m.

    python pipeline/maquetas/sao-sebastiao-angra/tracar.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
S = np.load(AQUI / "dados" / "superficie.npz")
SUP = S["z"] / 100.0
X0, Y0, P = float(S["x0"]), float(S["y0"]), float(S["passo"])
Z = ndi.median_filter(SUP, size=3)
G = np.hypot(*np.gradient(Z, P))

# Magistral do forte (recinto alto, com os baluartes e as cortinas), no sentido dos ponteiros do relógio a partir
# do ângulo saliente do baluarte noroeste.
FORTE = [(-40.6, 63.8), (-11.6, 45.2), (-15.9, 36.3), (17.3, 24.6), (19.8, 34.4), (63.8, 33.0), (55.0, 22.5),
         (-0.8, -50.0), (1.2, -55.3), (-7.8, -64.6), (-19.8, -64.6), (-42.0, -79.3), (-47.3, -48.0), (-46.5, -33.0),
         (-46.2, -21.6), (-59.5, -24.3), (-51.8, -1.0), (-43.4, -0.5),
         (-38.3, 41.3), (-43.3, 41.8), (-40.8, 44.5)]
# Recinto baixo a poente e a sul: a rampa que desce do baluarte sudoeste e a bateria baixa (Bateria da
# Heroicidade, 1828). Só os lados de fora; os de dentro são os muros do forte.
BAIXO = [(-46.2, -21.6), (-50.0, -21.0), (-52.6, -60.0), (-53.0, -107.5), (-37.7, -108.2), (-20.3, -72.5),
         (-19.8, -64.6)]
FIXOS = {"FORTE": (), "BAIXO": (0, 6)}


def amostra(A, x, y):
    return ndi.map_coordinates(A, [(np.asarray(y) - Y0) / P, (np.asarray(x) - X0) / P], order=1, mode="nearest")


def lado(a, b, busca=5.0, folga=1.5):
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    t = d / L
    n = np.array([-t[1], t[0]])
    if L < 2 * folga + 1:
        return a, b, None
    # lado de fora: o mais baixo dos dois, 6 m para cada lado
    meio = a + t * L / 2
    pa, pb = meio + 6 * n, meio - 6 * n
    fora = 1.0 if amostra(Z, [pa[0]], [pa[1]])[0] < amostra(Z, [pb[0]], [pb[1]])[0] else -1.0
    offs = np.arange(-busca, busca + 1e-6, 0.25)
    pts = []
    for s in np.arange(folga, L - folga, 0.25):
        # aresta de cima da face de fora: o ponto mais para fora onde a superfície começa a cair mais de 2 m em 2 m
        # (os muros grossos têm um degrau por dentro que não é a face)
        q = (a + t * s)[None, :] + (offs[:, None] * fora) * n
        v = amostra(Z, q[:, 0], q[:, 1])
        g = amostra(G, q[:, 0], q[:, 1])
        alto = v.max() - 1.5
        cand = [k for k in range(len(offs) - 8) if v[k] - v[k + 8] > 2.0 and v[k] - v[k + 2] > 0.4 and v[k] > alto]
        if not cand:
            continue
        k = max(cand)
        while k > 0 and v[k - 1] - v[k] > 0.25:
            k -= 1
        if g[min(k + 1, len(g) - 1)] > 1.0:
            pts.append((s, offs[k] * fora))
    if len(pts) < 4:
        return a, b, None
    B = np.array(pts)
    w = np.ones(len(B), bool)
    for _ in range(3):
        c = np.polyfit(B[w, 0], B[w, 1], 1)
        r = np.abs(np.polyval(c, B[:, 0]) - B[:, 1])
        w = r < max(0.35, np.percentile(r, 70))
    erro = float(np.median(np.abs(np.polyval(c, B[w, 0]) - B[w, 1])))
    return a + n * np.polyval(c, 0), a + t * L + n * np.polyval(c, L), erro


def cruzar(p1, p2, q1, q2):
    d1, d2 = p2 - p1, q2 - q1
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return (p2 + q1) / 2
    return p1 + d1 * np.linalg.solve(A, q1 - p1)[0]


def ajustar(poly, fechado=True, fixos=()):
    P_ = [np.array(v, float) for v in poly]
    n = len(P_)
    E, erros = [], []
    for i in range(n if fechado else n - 1):
        a, b = P_[i], P_[(i + 1) % n]
        if i in fixos:
            E.append((a, b))
            erros.append(None)
            continue
        a2, b2, e = lado(a, b)
        E.append((a2, b2))
        erros.append(e)
    out = []
    for i in range(n):
        if not fechado and i in (0, n - 1):
            out.append(P_[i])
            continue
        q = cruzar(*E[i - 1], *E[i % len(E)])
        out.append(q if np.linalg.norm(q - P_[i]) <= 3.0 else P_[i])
    return np.round(np.array(out), 2), erros


def main():
    res = {"nota": "Arestas de cima das faces dos muros (m; x este, y norte; origem do lidar.py)."}
    fo, ef = ajustar(FORTE)
    res["forte"] = fo.tolist()
    # o recinto baixo começa e acaba nos vértices do forte já ajustados
    b = [fo[14]] + BAIXO[1:-1] + [fo[10]]
    bo, eb = ajustar(b, fechado=False)
    res["baixo"] = bo.tolist()
    res["erro_mediano"] = {"forte": round(float(np.median([e for e in ef if e is not None])), 2),
                           "baixo": round(float(np.median([e for e in eb if e is not None])), 2)}
    (AQUI / "dados" / "planta.json").write_text(json.dumps(res, indent=1))
    print(res["erro_mediano"])


if __name__ == "__main__":
    main()
