"""Classes do chão da maqueta de Almeida -> dados/chao.json (método "metro a metro", novembro de 2026).

1. Proposta automática a 0,5 m (kit/chao.py, grelha retangular do LiDAR): cor da nuvem de pontos (dados/orto.jpg, de
   primavera: os prados dos fossos e da esplanada estão verdes), classes e declive: relva, erva (a erva seca e os
   campos de flores amarelas), mato debaixo das copas, saibro (claro e neutro, plano, longe das casas) e rocha.
2. Correções à mão (lista CORRECOES, por ordem; cada uma manda sobre as anteriores):
   - a vila dentro das muralhas (traçada pela rua que corre por dentro do terrapleno): as ruas, as praças e os largos
     entre as casas são calçada de granito (o chão sem verde até 12 m das casas e as ruas do OSM, que aqui assentam
     bem na ortoimagem); o picadeiro, as ruínas do castelo e os quintais ficam com a classe da cor;
   - as estradas de fora (a EN340 e as ruas dos bairros novos): asfalto, com o traçado do OSM assente na faixa
     cinzenta da ortoimagem e a largura da classe;
   - os passeios e largos dos bairros de fora (o claro até 10 m das casas): calçada;
   - o campo lavrado a nascente da esplanada (castanho): terra.

    python pipeline/maquetas/almeida/chao.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1]))
from kit import chao  # noqa: E402

CFG = json.loads((AQUI.parents[2] / "src" / "data" / "maquetas" / "almeida.json").read_text())
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574


def ll(c):
    return np.array([(c[0] - LON0) * KX, (c[1] - LAT0) * KY])  # CENTRO = (0, 0), como em construir.py


def geom(f):
    g = f["geometria"]
    if g["type"] == "Polygon":
        return shapely.Polygon([ll(c) for c in g["coordinates"][0]]).buffer(0)
    if g["type"] == "LineString":
        return shapely.LineString([ll(c) for c in g["coordinates"]])
    return None


# A vila dentro das muralhas, pela rua que corre por dentro do terrapleno (vista na ortoimagem e no OSM), com 6 m
# para fora para levar a própria rua.
INTRAMUROS = chao.poligono([(37.5, 225), (79, 196), (162, 150), (192, 96), (198, 17), (187, -42), (167, -83),
                            (137, -117), (96, -142), (8, -208), (-42, -200), (-71, -167), (-129, -125), (-135, -79),
                            (-171, -37), (-173, 17), (-162, 54), (-148, 96), (-112, 133), (-54, 179),
                            (-12, 200)]).buffer(6.0)
CASAS = [geom(f) for f in F["edificios"] if f["geometria"]["type"] == "Polygon"]
RUAS_VILA = shapely.union_all([geom(f).buffer(3.0, cap_style="flat") for f in F["caminhos"]
                               if f["geometria"]["type"] == "LineString"
                               and f["classe"] in ("residential", "primary", "tertiary", "unclassified", "pedestrian",
                                                   "service", "secondary", "living_street")]).intersection(INTRAMUROS)
VILA = shapely.union_all([c.buffer(12.0) for c in CASAS if c.intersects(INTRAMUROS)] + [RUAS_VILA]).intersection(INTRAMUROS)
URBANO = shapely.union_all([c.buffer(10.0) for c in CASAS]).difference(INTRAMUROS)
# O campo lavrado a nascente, entre a esplanada e a EN340 (castanho na ortoimagem); os outros castanhos (a noroeste)
# são prado seco e ficam erva
CAMPO_LAVRADO = shapely.box(280.0, -180.0, 400.0, 20.0)
LARGURA = {"primary": 8.0, "secondary": 8.0, "tertiary": 7.0, "unclassified": 5.5, "residential": 6.0, "service": 4.0}
_CACHE = {}


def estradas():
    if "e" in _CACHE:
        return _CACHE["e"]
    masc = chao.mascara_asfalto(AQUI)
    lid = np.load(AQUI / "dados" / "lidar.npz")
    xs, ys, _ = chao.grelha(lid)
    fora = shapely.box(xs[0], ys[0], xs[-1], ys[-1])
    fs = []
    for f in F["caminhos"]:
        if f["classe"] not in LARGURA or f["geometria"]["type"] != "LineString":
            continue
        for g in shapely.get_parts(geom(f).intersection(fora).difference(INTRAMUROS)):
            if g.geom_type == "LineString" and g.length >= 8:
                fs.append(chao.alinhar_osm(g, masc, procura=8.0).buffer(LARGURA[f["classe"]] / 2, cap_style="flat"))
    _CACHE["e"] = shapely.union_all(fs).difference(INTRAMUROS) if fs else shapely.Polygon()
    return _CACHE["e"]


def asfalto(cl):
    return chao.pela_classe(estradas(), "erva", "saibro", "rocha", "relva")(cl)


def lavrados(cl):
    """Os campos lavrados: castanhos (quentes, sem verde), em manchas grandes, fora da vila e das estradas."""
    lid = np.load(AQUI / "dados" / "lidar.npz")
    xs, ys, passo = chao.grelha(lid)
    o = chao._orto(AQUI, len(xs), len(ys))
    q = ndi.uniform_filter(o[..., 0] - o[..., 2], 9)
    v = ndi.uniform_filter(o[..., 1] - (o[..., 0] + o[..., 2]) / 2, 9)
    lum = ndi.uniform_filter(o.mean(2), 9)
    m = (q > 0.12) & (v < 0.0) & (lum > 0.45) & (lum < 0.80)
    m = ndi.binary_opening(ndi.binary_closing(m, iterations=4), iterations=6)
    g = chao.raster_para_poligonos(np.where(m, 0, -1), xs[0], ys[0], passo, ["terra"], area_min=1500.0, alisar=1.5)
    return g.get("terra", shapely.Polygon()).intersection(CAMPO_LAVRADO)


CORRECOES = [
    ("calcada", chao.pela_classe(VILA, "erva", "saibro", "rocha", "terra"),
     "vila dentro das muralhas: ruas, praças e largos entre as casas em calçada de granito (o chão sem verde até 12 m "
     "das casas e as ruas do OSM)"),
    ("asfalto", asfalto,
     "estradas de fora das muralhas (EN340 e bairros novos): o traçado do OSM assente na faixa cinzenta da ortoimagem, "
     "com a largura da classe"),
    ("calcada", chao.pela_classe(URBANO, "saibro"),
     "passeios e largos dos bairros de fora: o chão claro até 10 m das casas é calçada"),
    ("terra", lavrados, "campo lavrado a nascente da esplanada (a mancha castanha da ortoimagem): terra"),
]


def main():
    cl, dom = chao.proposta(AQUI, claro=(0.66, 0.12), casa_folga=2.0, sombra=0.30, area_min=30.0)
    cl = chao.aplicar(cl, CORRECOES, dom)
    chao.gravar(AQUI / "dados" / "chao.json", cl, {"fonte": "cor do LiDAR DGT + classes e MDT + correções à mão",
                                                    "correcoes": [c[2] for c in CORRECOES]})
    for n, g in cl.items():
        print(f"{n:8s} {g.area:9.0f} m²  {len(shapely.get_parts(g)):4d} polígonos")


if __name__ == "__main__":
    main()
