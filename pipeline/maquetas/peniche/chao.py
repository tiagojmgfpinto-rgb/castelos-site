"""Classes do chão da maqueta de Peniche -> dados/chao.json (método "metro a metro", novembro de 2026).

Em Peniche a proposta genérica do kit não serve: a fortaleza inteira vem como edifício (classe 6) na nuvem de pontos
(os terraplenos, os adarves e os pátios são o topo das plataformas), o mar não tem pontos e a cor por cima dele é
uma interpolação, e o chão é quase todo claro (calcário, saibro, lajes). Por isso a proposta é feita aqui, a 0,5 m:

1. Proposta pela cor da nuvem de pontos (dados/orto.jpg), pela textura (desvio local da luminância) e pelo declive:
   - mar: o MDT abaixo do nível (0,4 m) em manchas grandes, como AGUA em construir.py -> "agua";
   - verde -> relva; castanho-amarelado (a erva seca dos terraplenos e dos fossos) -> erva;
   - claro e áspero (blocos, penedos) ou a pique, fora das plataformas -> rocha (o calcário da arriba);
   - claro e liso -> saibro (o terreiro do Campo da República e os largos de terra);
   - debaixo das casas da vila (classe 6 fora das plataformas) fica a classe vizinha; no topo das plataformas (os
     terraplenos e os pátios da fortaleza) fica a classe da cor.
2. Correções à mão (lista CORRECOES, por ordem; cada uma manda sobre as anteriores):
   - os pátios, as ruas e os adarves da fortaleza (áreas pedonais do OSM) e a praça: o chão não verde é lajeado;
   - os relvados do OSM dentro da fortaleza: o claro passa a erva (a relva seca e rapada dos baluartes);
   - as ruas e estradas de fora: asfalto, com o traçado do OSM assente na faixa cinzenta e a largura da classe;
   - os passeios e largos da vila (o claro até 10 m das casas do OSM, fora da fortaleza) são calçada;
   - o mar no fim (debaixo do plano de água).

    python pipeline/maquetas/peniche/chao.py
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

CFG = json.loads((AQUI.parents[2] / "src" / "data" / "maquetas" / "peniche.json").read_text())
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
PLAT = json.loads((AQUI / "dados" / "plataformas.json").read_text())
LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
NIVEL = 0.4  # o mesmo de construir.py
CONTORNO = shapely.box(-200.0, -140.0, 200.0, 150.0)  # o mesmo de construir.py
FORTIM = shapely.Point(-10.5, -63.3).buffer(12.3)  # o Fortim Redondo (FORTIM em construir.py, com 0,5 m de folga)


def ll(c):
    return np.array([(c[0] - LON0) * KX, (c[1] - LAT0) * KY])  # CENTRO = (0, 0), como em construir.py


def geom(f):
    g = f["geometria"]
    if g["type"] == "Polygon":
        return shapely.Polygon([ll(c) for c in g["coordinates"][0]]).buffer(0)
    if g["type"] == "LineString":
        return shapely.LineString([ll(c) for c in g["coordinates"]])
    return None


PLATAFORMAS = shapely.union_all([shapely.Polygon(p["contorno"], p["buracos"]).buffer(0) for p in PLAT["plataformas"]]
                                + [FORTIM])
# a fortaleza: as plataformas e o que fica entre elas (pátios), pelo contorno das muralhas do OSM
_cw = [geom(f) for f in F["infraestrutura"] if f["classe"] == "city_wall" and f["geometria"]["type"] == "Polygon"]
FORTALEZA = shapely.union_all(_cw + [PLATAFORMAS]) if _cw else PLATAFORMAS
PEDONAL = shapely.union_all([geom(f) for f in F["usos"] if f["classe"] in ("pedestrian", "plaza")
                             and f["geometria"]["type"] == "Polygon"])
RELVADOS_OSM = shapely.union_all([geom(f) for f in F["usos"] if f["classe"] in ("grass", "meadow", "greenfield", "park",
                                                                               "garden", "flowerbed")
                                  and f["geometria"]["type"] == "Polygon"])
URBANO = shapely.union_all([geom(f).buffer(10.0) for f in F["edificios"]
                            if f["geometria"]["type"] == "Polygon"]).difference(FORTALEZA.buffer(3.0))
LARGURA = {"secondary": 8.0, "tertiary": 7.0, "unclassified": 6.0, "residential": 6.0, "service": 4.5}
NOMES = ["agua", "relva", "erva", "rocha", "saibro", "mato"]
_CACHE = {}


def raster():
    """Proposta a 0,5 m: (rótulos, xs, ys, passo)."""
    if "r" in _CACHE:
        return _CACHE["r"]
    lid = np.load(AQUI / "dados" / "lidar.npz")
    xs, ys, passo = chao.grelha(lid)
    mdt, mds, cls = lid["mdt"] / 100.0 + float(lid["base"]), lid["mds"] / 100.0 + float(lid["base"]), lid["classe"]
    o = chao._orto(AQUI, len(xs), len(ys))
    lum, sat = o.mean(2), o.max(2) - o.min(2)
    verde = o[..., 1] - (o[..., 0] + o[..., 2]) / 2
    quente = o[..., 0] - o[..., 2]
    aspero = np.sqrt(np.maximum(ndi.uniform_filter(lum ** 2, 5) - ndi.uniform_filter(lum, 5) ** 2, 0))
    gy, gx = np.gradient(mdt, passo)
    decl = np.degrees(np.arctan(np.hypot(gx, gy)))
    X, Y = np.meshgrid(xs, ys)
    plat = shapely.contains_xy(PLATAFORMAS, X, Y)
    # mar: como AGUA em construir.py
    ag = mdt < NIVEL
    lab, k = ndi.label(ag)
    tam = ndi.sum(np.ones_like(lab), lab, np.arange(1, k + 1)) * passo * passo
    agua = np.isin(lab, np.nonzero(tam > 150)[0] + 1)
    ix = {n: i for i, n in enumerate(NOMES)}
    lum_s, asp_s = ndi.uniform_filter(lum, 5), ndi.uniform_filter(aspero, 5)
    rot = np.full(lum.shape, ix["saibro"])
    # a erva seca é mais escura e quente do que o calcário, as lajes e o saibro (que passam de 0,7)
    rot[(lum_s < 0.64) & (ndi.uniform_filter(quente, 5) > 0.03)] = ix["erva"]
    rot[verde > 0.02] = ix["relva"]
    # o calcário áspero só na faixa da costa (até 45 m do mar): no terreiro, as manchas ásperas são saibro remexido
    costa = ndi.distance_transform_edt(~agua) * passo < 45.0
    rocha = ((asp_s > 0.07) & (lum_s > 0.6) & (verde < 0.02) & costa) | (decl > 45)
    rot[rocha & ~plat] = ix["rocha"]
    copa = (cls >= 3) & (cls <= 5) & (mds - mdt > 1.2)
    rot[copa] = ix["mato"]
    casa = (cls == 6) & ~plat
    if casa.any():
        _, (ri, ci) = ndi.distance_transform_edt(casa, return_indices=True)
        rot = rot[ri, ci]
    rot = chao.maioria(rot, len(NOMES), raio=2)
    rot = chao.maioria(rot, len(NOMES), raio=1)
    rot[agua] = ix["agua"]
    _CACHE["r"] = (rot, xs, ys, passo, agua)
    return _CACHE["r"]


def proposta():
    rot, xs, ys, passo, _ = raster()
    return chao.raster_para_poligonos(rot, xs[0], ys[0], passo, NOMES, area_min=12.0, simplificar=0.4, alisar=0.6)


def mar():
    rot, xs, ys, passo, agua = raster()
    g = chao.raster_para_poligonos(np.where(agua, 0, -1), xs[0], ys[0], passo, ["agua"], area_min=50.0)
    return g.get("agua", shapely.Polygon())


def estradas():
    if "e" in _CACHE:
        return _CACHE["e"]
    fs = []
    for f in F["caminhos"]:
        if f["classe"] not in LARGURA or f["geometria"]["type"] != "LineString":
            continue
        for g in shapely.get_parts(geom(f).intersection(CONTORNO).difference(FORTALEZA)):
            if g.geom_type == "LineString" and g.length >= 8:
                # em Peniche o traçado do OSM assenta bem nas ruas (verificado na ortoimagem) e o chão é quase todo
                # claro: alinhar pela cor desviava-o; fica como está
                fs.append(g.buffer(LARGURA[f["classe"]] / 2, cap_style="flat"))
    _CACHE["e"] = shapely.union_all(fs).difference(FORTALEZA.buffer(1.0)) if fs else shapely.Polygon()
    return _CACHE["e"]


def asfalto(cl):
    return chao.pela_classe(estradas(), "saibro", "erva", "rocha")(cl)


CORRECOES = [
    ("lajeado", chao.pela_classe(PEDONAL, "saibro", "rocha", "erva"),
     "pátios, ruas e adarves da fortaleza (áreas pedonais do OSM) e a praça: o chão sem verde é lajeado de calcário"),
    ("erva", chao.pela_classe(RELVADOS_OSM.intersection(FORTALEZA), "saibro", "rocha"),
     "relvados do OSM nos baluartes: o claro é a erva seca e rapada"),
    ("asfalto", asfalto,
     "ruas e estradas de fora da fortaleza: o traçado do OSM assente na faixa cinzenta da ortoimagem, com a largura "
     "da classe"),
    ("calcada", chao.pela_classe(URBANO, "saibro", "rocha"),
     "passeios e largos da vila: o chão claro até 10 m das casas, fora da fortaleza, é calçada"),
    ("asfalto", lambda cl: chao.pela_cor(AQUI, URBANO.difference(FORTALEZA.buffer(12.0)).difference(mar().buffer(15.0)), lum_min=0.40, lum_max=0.70, sat_max=0.06, verde_max=0.01,
                                          fechar=4, abrir=3, area_min=60.0, alisar=1.0),
     "parques de estacionamento e ruas da vila sem traçado no OSM: o cinzento médio (os carros ficam por baixo)"),
    ("agua", mar(), "o mar (o MDT abaixo de 0,4 m em manchas grandes), debaixo do plano de água"),
]


def main():
    cl = chao.aplicar(proposta(), CORRECOES, CONTORNO)
    chao.gravar(AQUI / "dados" / "chao.json", cl, {"fonte": "cor e textura do LiDAR DGT 2024 + classes e MDT + correções "
                                                            "à mão", "correcoes": [c[2] for c in CORRECOES]})
    for n, g in cl.items():
        print(f"{n:8s} {g.area:9.0f} m²  {len(shapely.get_parts(g)):4d} polígonos")


if __name__ == "__main__":
    main()
