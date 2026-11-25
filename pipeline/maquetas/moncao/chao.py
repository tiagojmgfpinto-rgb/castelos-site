"""Classes do chão da maqueta de Monção -> dados/chao.json (método "metro a metro", novembro de 2026).

1. Proposta automática a 0,5 m (kit/chao.py, grelha retangular do LiDAR 2025): cor da nuvem de pontos (dados/orto.jpg),
   classes e declive: erva, relva, mato debaixo das copas, saibro (claro e neutro, plano, longe das casas) e rocha.
2. Correções à mão (lista CORRECOES, por ordem; cada uma manda sobre as anteriores):
   - dentro da vila amuralhada (traçada pelos panos e pelas linhas de muralha) o chão sem verde entre as casas é
     calçada de granito, com os jardins e largos verdes do OSM em relva;
   - as ruas e estradas de fora das muralhas são asfalto: o traçado do OSM assente na faixa cinzenta da ortofoto
     (chao.alinhar_osm) e com a largura da classe;
   - fora das muralhas, o chão claro até 12 m das casas (passeios, largos, parques) é calçada;
   - o recinto da feira junto ao rio (as tendas são provisórias e não entram) é terreiro de terra batida;
   - o leito do rio Minho (sem pontos do LiDAR, à cota da água) é "agua", debaixo do plano de água.

    python pipeline/maquetas/moncao/chao.py
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

CFG = json.loads((AQUI.parents[2] / "src" / "data" / "maquetas" / "moncao.json").read_text())
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574


def ll(c):
    return np.array([(c[0] - LON0) * KX, (c[1] - LAT0) * KY])  # CENTRO = (0, 0), como em construir.py


def geom(f):
    g = f["geometria"]
    if g["type"] == "Polygon":
        return shapely.Polygon([ll(c) for c in g["coordinates"][0]])
    if g["type"] == "LineString":
        return shapely.LineString([ll(c) for c in g["coordinates"]])
    return None


# A vila amuralhada, pelo lado de dentro dos panos (dados/escarpas.json) e das linhas de muralha do OSM, vista na
# ortofoto: a frente norte sobre o rio, o baluarte a nordeste, a frente sul pela Rua da Independência e a poente
# pelo Largo da Estação.
INTRAMUROS = chao.poligono([(-349, 190), (-234, 211), (-134, 197), (-106, 186), (9, 165), (80, 147), (109, 126),
                            (109, 83), (80, 61), (9, -60), (-63, -89), (-113, -60), (-220, -67), (-291, -46),
                            (-334, 19), (-377, 104), (-391, 154)])
VERDES = shapely.union_all([geom(f) for f in F["usos"] if f["classe"] in ("garden", "village_green", "park", "grass",
                                                                           "playground")]).intersection(INTRAMUROS)
# arruamentos de fora das muralhas: até 12 m das casas do OSM, o chão claro e neutro não é saibro mas passeio e largo
# calcetado (a proposta só vê "claro")
URBANO = shapely.union_all([geom(f).buffer(12.0) for f in F["edificios"]
                            if f["geometria"]["type"] == "Polygon"]).difference(INTRAMUROS)
FEIRA = shapely.box(222, 82, 332, 148)  # o mesmo de construir.py
NIVEL = 3.0  # o mesmo de construir.py
LARGURA = {"secondary": 8.0, "tertiary": 7.0, "unclassified": 5.5, "residential": 5.5, "service": 4.0}
_ESTRADAS = []


def estradas():
    """Ruas e estradas do OSM fora da vila amuralhada, assentes na faixa cinzenta da ortofoto."""
    if _ESTRADAS:
        return _ESTRADAS[0]
    masc = chao.mascara_asfalto(AQUI)
    fora = shapely.box(-498, -298, 498, 298)
    fs = []
    for f in F["caminhos"]:
        if f["classe"] not in LARGURA or f["geometria"]["type"] != "LineString":
            continue
        for g in shapely.get_parts(geom(f).intersection(fora).difference(INTRAMUROS)):
            if g.geom_type == "LineString" and g.length >= 8:
                fs.append(chao.alinhar_osm(g, masc).buffer(LARGURA[f["classe"]] / 2, cap_style="flat"))
    _ESTRADAS.append(shapely.union_all(fs).difference(INTRAMUROS))
    return _ESTRADAS[0]


def asfalto(cl):
    return chao.pela_classe(estradas(), "erva", "saibro", "rocha", "terra")(cl)


def leito():
    """O rio: sem pontos do LiDAR e à cota da água (como AGUA em construir.py)."""
    lid = np.load(AQUI / "dados" / "lidar.npz")
    xs, ys, passo = chao.grelha(lid)
    mdt = lid["mdt"] / 100.0 + float(lid["base"])
    ag = (lid["classe"] == 0) & (mdt < NIVEL + 0.45)
    lab, k = ndi.label(ag)
    ar = ndi.sum(np.ones_like(mdt), lab, np.arange(1, k + 1)) * passo * passo
    ag = np.isin(lab, 1 + np.nonzero(ar > 400)[0])
    ag = ndi.binary_closing(ag, iterations=3) & (mdt < NIVEL + 0.6)
    g = chao.raster_para_poligonos(np.where(ag, 0, -1), xs[0], ys[0], passo, ["agua"], area_min=50.0)
    return g.get("agua", shapely.Polygon())


CORRECOES = [
    ("calcada", chao.pela_classe(INTRAMUROS, "erva", "saibro", "rocha", "terra", "mato"),
     "vila amuralhada: ruas, praças e largos entre as casas em calçada de granito"),
    ("relva", VERDES, "jardins e largos verdes do OSM dentro da vila"),
    ("asfalto", asfalto,
     "ruas e estradas de fora das muralhas: o traçado do OSM assente na faixa cinzenta da ortofoto, com a largura da "
     "classe; ficam as bermas e os canteiros verdes"),
    ("calcada", chao.pela_classe(URBANO, "saibro"),
     "passeios, largos e parques de fora das muralhas: o chão claro até 12 m das casas é calcetado, não saibro"),
    ("terra", FEIRA, "recinto da feira junto ao rio: terreiro de terra batida (as tendas são provisórias)"),
    ("agua", leito(), "leito do rio Minho, debaixo do plano de água"),
]


def main():
    cl, dom = chao.proposta(AQUI, claro=(0.66, 0.12), casa_folga=2.0, sombra=0.30, area_min=30.0)
    cl = chao.aplicar(cl, CORRECOES, dom)
    chao.gravar(AQUI / "dados" / "chao.json", cl, {"fonte": "cor do LiDAR DGT 2025 + classes e MDT + correções à mão",
                                                    "correcoes": [c[2] for c in CORRECOES]})
    for n, g in cl.items():
        print(f"{n:8s} {g.area:9.0f} m²  {len(shapely.get_parts(g)):4d} polígonos")


if __name__ == "__main__":
    main()
