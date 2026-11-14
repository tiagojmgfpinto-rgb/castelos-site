"""Topo dos muros em degraus, sem rampas (revisão do perito, 14 nov. 2026).

O tracar.py dá, para cada muro, o contorno com a cota do topo medida em cada vértice. Triangular esse contorno e
interpolar as cotas faz rampas onde o muro muda de nível (o cunhal SE a 471 m contra o pano a 468 m, a cerca a
469,8 m contra o canto do museu, a muralha exterior a 465,5 contra 463,8 m): faces inclinadas que não existem.

Aqui:
- um triângulo cujos cantos diferem mais de T (1,5 m) fica plano, à cota da maioria dos cantos (a mediana);
- onde dois triângulos vizinhos ficam a cotas diferentes, fecha-se o degrau com uma face a prumo;
- as faces de fora saem dos lados da triangulação que só têm um triângulo (o contorno já limpo pelo buffer(0),
  que nalguns muros se cruzava a si próprio) e sobem até ao topo desse triângulo.
Densificar o contorno (passo < 99) e dar aos triângulos planos a cota mediana do LiDAR foi experimentado: com
triângulos pequenos de um lado ao outro dos muros finos, o topo ficava em dentes de serra junto ao museu.
"""
import math

import numpy as np
import shapely
from scipy.spatial import cKDTree

T_DEGRAU = 1.5  # m: acima disto um desnível entre vértices vizinhos é degrau, não rampa
PASSO = 1.0  # m: comprimento máximo dos lados do contorno


def _densificar(c, t, ch, passo=PASSO, T=T_DEGRAU):
    pc, pt, pch = [], [], []
    n = len(c)
    for i in range(n):
        j = (i + 1) % n
        L = float(np.hypot(*(c[j] - c[i])))
        k = max(1, math.ceil(L / passo))
        for s in range(k):
            f = s / k
            pc.append(c[i] + f * (c[j] - c[i]))
            pt.append(t[i] + f * (t[j] - t[i]) if abs(t[j] - t[i]) <= T else (t[i] if f < 0.5 else t[j]))
            pch.append(ch[i] + f * (ch[j] - ch[i]))
    return np.array(pc), np.array(pt), np.array(pch)


def topo_degraus(aneis, T=T_DEGRAU, passo=PASSO):
    """aneis: [{"contorno", "topo", "chao"}] (o primeiro é o de fora). Devolve (topos, paredes, info):
    topos = [(cs 3×2, z 3)], com os cantos em sentido anti-horário;
    paredes = [(a, b, za0, zb0, za1, zb1, fora)]: face a prumo de a para b, da cota 0 à cota 1; fora=True nas faces
    do contorno (desce até ao chão), False nos degraus do topo."""
    aneis_d = [_densificar(np.asarray(a["contorno"], float), np.asarray(a["topo"], float),
                           np.asarray(a["chao"], float), passo, T) for a in aneis]
    pol = shapely.Polygon(aneis_d[0][0], [a[0] for a in aneis_d[1:]])
    if not pol.is_valid:
        pol = pol.buffer(0)
    if pol.geom_type != "Polygon":
        return None
    vs = np.concatenate([a[0] for a in aneis_d])
    ts = np.concatenate([a[1] for a in aneis_d])
    chs = np.concatenate([a[2] for a in aneis_d])
    kd = cKDTree(vs)
    topos, arestas = [], {}
    planos = 0
    chave = lambda p: (round(float(p[0]), 3), round(float(p[1]), 3))
    for tri in shapely.constrained_delaunay_triangles(pol).geoms:
        cs = np.array(tri.exterior.coords)[:3]
        if shapely.Polygon(cs).area < 1e-4:
            continue
        if not shapely.Polygon(cs).exterior.is_ccw:
            cs = cs[::-1]
        _, j = kd.query(cs)  # o buffer(0) pode criar vértices novos: ficam com a cota do vértice medido mais perto
        z = ts[j].copy()
        if np.ptp(z) > T:
            z[:] = np.median(z)  # o nível da maioria dos cantos
            planos += 1
        topos.append((cs, z))
        for e in range(3):
            f = (e + 1) % 3
            ka, kb = chave(cs[e]), chave(cs[f])
            arestas.setdefault(tuple(sorted((ka, kb))), []).append((ka, cs[e], cs[f], z[e], z[f], chs[j[e]], chs[j[f]]))
    paredes = []
    degraus = 0
    for lista in arestas.values():
        if len(lista) == 1:
            # lado do contorno (de fora ou de um furo): o triângulo é anti-horário, por isso "fora" fica à direita
            _, pa, pb, za, zb, ca, cb = lista[0]
            paredes.append((pa, pb, ca, cb, za, zb, True))
        elif len(lista) == 2:
            (ka1, pa, pb, za1, zb1, _, _), (ka2, _, _, za2, zb2, _, _) = lista
            oa, ob = (za2, zb2) if ka2 == ka1 else (zb2, za2)  # cotas do outro triângulo nos mesmos vértices
            if max(abs(za1 - oa), abs(zb1 - ob)) < 0.05:
                continue
            degraus += 1
            # as duas faces (uma para cada lado): um degrau pode ser visto de cima e de baixo
            lo_a, lo_b, hi_a, hi_b = min(za1, oa), min(zb1, ob), max(za1, oa), max(zb1, ob)
            paredes.append((pa, pb, lo_a, lo_b, hi_a, hi_b, False))
            paredes.append((pb, pa, lo_b, lo_a, hi_b, hi_a, False))
    return topos, paredes, {"triangulos_planos": planos, "degraus": degraus}
