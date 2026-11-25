"""Classes do chão da maqueta de Lindoso -> dados/chao.json (método "metro a metro", novembro de 2026).

1. Proposta automática a 0,5 m (kit/chao.py, grelha do LiDAR 2025 de x0, y0): cor do ortofotomapa (dados/orto.jpg,
   de verão: o monte do castelo e da eira é erva seca e urze, castanha), classes e declive: erva, relva (os lameiros
   a nascente), mato debaixo das copas, saibro (os carreiros claros) e rocha.
2. Correções à mão (lista CORRECOES, por ordem; cada uma manda sobre as anteriores):
   - os afloramentos de granito do monte e da eira dos espigueiros: o MDT acima do relevo à volta, ou cinzento no
     ortofotomapa com algum relevo (como o construir.py fazia face a face);
   - o pátio do castelo é relvado (fotografias de drone), com os afloramentos;
   - a estrada nacional (primary do OSM) em asfalto; as ruas da aldeia em calçada de granito (claras no
     ortofotomapa, ao contrário da nacional, escura); o traçado do OSM assente na faixa clara do ortofotomapa;
   - os carreiros e trilhos do OSM em terra;
   - o claro junto às ruas e às casas (largos, o parque a sul da eira, quintais): calçada.

    python pipeline/maquetas/lindoso/chao.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1]))
sys.path.insert(0, str(AQUI))
from kit import chao  # noqa: E402
from comum import ZONA_CASTELO  # noqa: E402

F = json.loads((AQUI / "dados" / "fontes.json").read_text())  # já em metros da cena
CAIXA = shapely.box(-124.0, -156.0, 74.0, 64.0)  # X0, Y0, X1, Y1 de construir.py
# o pátio, por dentro da cerca medieval (PATIO em construir.py)
PATIO = shapely.Polygon([(-12.8, 2.0), (-2.5, 9.0), (13.5, 7.5), (17.5, -6.0), (14.0, -19.5), (8.0, -25.5),
                         (-6.5, -28.0), (-12.5, -10.0)])
CASTELO = shapely.Polygon(ZONA_CASTELO)
# a eira dos espigueiros: a laje de granito clara e rasa (ortofotomapa 2021, traçada à mão)
EIRA = shapely.Polygon([(-58.0, -72.0), (-50.0, -64.0), (-40.0, -60.0), (-34.0, -58.0), (-33.0, -62.0),
                        (-37.0, -68.0), (-35.0, -76.0), (-32.0, -82.0), (-33.0, -90.0), (-40.0, -90.0),
                        (-45.0, -82.0), (-50.0, -78.0), (-57.0, -77.0)])
CASAS = [shapely.Polygon(f["geometria"]["coordinates"][0]).buffer(0) for f in F["edificios"]
         if f["geometria"]["type"] == "Polygon"]
LARGURA = {"primary": 7.0, "secondary": 6.5, "tertiary": 6.0, "residential": 5.0, "unclassified": 5.0,
           "living_street": 4.0, "service": 3.5}
TRILHOS = {"track": 2.5, "path": 1.6, "footway": 1.8, "steps": 1.8}
_CACHE = {}


def _grelha():
    if "g" not in _CACHE:
        lid = np.load(AQUI / "dados" / "lidar.npz")
        xs, ys, passo = chao.grelha(lid)
        _CACHE["g"] = (lid, xs, ys, passo, chao._orto(AQUI, len(xs), len(ys)))
    return _CACHE["g"]


def afloramentos(cl):
    """Penedos e lajes de granito: relevo local positivo do MDT (> 22 cm) ou cinzento no ortofotomapa com algum
    relevo (> 12 cm); as lajes rasas da eira são cinzentas e lisas, entram pela cor com relevo baixo (> 4 cm)."""
    lid, xs, ys, passo, o = _grelha()
    mdt = lid["mdt"] / 100.0
    rel = ndi.gaussian_filter(mdt - ndi.gaussian_filter(mdt, 3.0), 0.7)
    om = ndi.uniform_filter(o, size=(7, 7, 1))
    sat, lum = om.max(2) - om.min(2), om.mean(2)
    cinza = (sat < 0.07) & (lum > 0.38)
    copa = (lid["classe"] >= 3) & (lid["classe"] <= 5) & (lid["mds"] - lid["mdt"] > 120)
    m = ((rel > 0.22) | (cinza & (rel > 0.04))) & ~copa & (lid["classe"] != 6)
    m = ndi.binary_opening(ndi.binary_closing(m, iterations=1), iterations=1)
    g = chao.raster_para_poligonos(np.where(m, 0, -1), xs[0], ys[0], passo, ["rocha"], area_min=3.0, alisar=0.5)
    return g.get("rocha", shapely.Polygon()).difference(shapely.union_all(CASAS).buffer(1.0))


def estradas():
    if "e" in _CACHE:
        return _CACHE["e"]
    lid, xs, ys, passo, o = _grelha()
    # as ruas da aldeia são claras (granito), não cinzentas como o asfalto: a máscara é o claro sem verde, fora das
    # copas e dos telhados
    om = ndi.uniform_filter(o, size=(3, 3, 1))
    copa = (lid["classe"] >= 3) & (lid["classe"] <= 5) & (lid["mds"] - lid["mdt"] > 120)
    masc = ((om.mean(2) > 0.58) & (om[..., 1] - (om[..., 0] + om[..., 2]) / 2 < 0.0) & ~copa & (lid["classe"] != 6),
            xs, ys, passo)
    asf, cal = [], []
    for f in F["caminhos"]:
        if f["classe"] not in LARGURA or f["geometria"]["type"] != "LineString":
            continue
        for g in shapely.get_parts(shapely.LineString(f["geometria"]["coordinates"]).intersection(CAIXA.buffer(5.0))
                                   .difference(CASTELO)):
            if g.geom_type == "LineString" and g.length >= 6:
                if f["classe"] != "primary":
                    g = chao.alinhar_osm(g, masc, procura=9.0, troco=30.0, meia=1.5)
                (asf if f["classe"] == "primary" else cal).append(g.buffer(LARGURA[f["classe"]] / 2, cap_style="flat"))
    u = lambda gs: shapely.union_all(gs).difference(CASTELO) if gs else shapely.Polygon()  # noqa: E731
    _CACHE["e"] = (u(asf), u(cal))
    return _CACHE["e"]


TRILHO = shapely.union_all([shapely.LineString(f["geometria"]["coordinates"]).buffer(TRILHOS[f["classe"]] / 2,
                                                                                     cap_style="flat")
                            for f in F["caminhos"] if f["geometria"]["type"] == "LineString"
                            and f["classe"] in TRILHOS]).difference(CASTELO)
URBANO = shapely.union_all([c.buffer(8.0) for c in CASAS]).difference(CASTELO)


def rocha_patio(cl):
    return afloramentos(cl).intersection(PATIO)


CORRECOES = [
    ("rocha", afloramentos,
     "afloramentos de granito do monte e lajes da eira dos espigueiros (relevo do MDT, ou cinzento no ortofotomapa "
     "com algum relevo)"),
    ("rocha", chao.pela_classe(EIRA, "erva", "saibro", "relva"),
     "a eira dos espigueiros: a laje de granito clara e rasa, traçada à mão no ortofotomapa"),
    ("relva", PATIO, "pátio do castelo: relvado (fotografias de drone)"),
    ("rocha", rocha_patio, "afloramentos de granito no pátio"),
    ("terra", TRILHO, "carreiros e trilhos do OSM (para o castelo e entre os espigueiros): terra"),
    ("calcada", lambda cl: estradas()[1],
     "ruas da aldeia: calçada de granito (claras no ortofotomapa), com o traçado do OSM assente na faixa clara"),
    ("asfalto", lambda cl: estradas()[0], "estrada nacional (primary do OSM, escura no ortofotomapa): asfalto"),
    ("calcada", lambda cl: chao.pela_classe(estradas()[1].buffer(6.0).difference(CASTELO), "saibro")(cl),
     "a parte clara das ruas que o traçado alinhado não cobre (a rua de nascente é mais larga do que a classe)"),
    ("calcada", chao.pela_classe(URBANO, "saibro"),
     "largos, quintais e o parque a sul da eira: o claro até 8 m das casas é calçada"),
]


def main():
    cl, _ = chao.proposta(AQUI, claro=(0.62, 0.10), casa_folga=2.0, sombra=0.25, area_min=8.0)
    cl = chao.aplicar(cl, CORRECOES, CAIXA)
    chao.gravar(AQUI / "dados" / "chao.json", cl, {"fonte": "cor do ortofotomapa DGT 2021 + classes e MDT do LiDAR "
                                                            "2025 + correções à mão",
                                                    "correcoes": [c[2] for c in CORRECOES]})
    for n, g in cl.items():
        print(f"{n:8s} {g.area:9.0f} m²  {len(shapely.get_parts(g)):4d} polígonos")


if __name__ == "__main__":
    main()
