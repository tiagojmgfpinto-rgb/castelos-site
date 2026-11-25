"""Classes do chão da maqueta de Valença -> dados/chao.json (método "metro a metro", novembro de 2026).

1. Proposta automática a 0,5 m (kit/chao.py, grelha retangular do LiDAR 2025): cor da nuvem de pontos (dados/orto.jpg),
   classes e declive. Nas esplanadas e nos terraplenos a erva seca é dourada (erva) e as manchas verdes são relva;
   o chão claro e neutro, plano e longe das casas, é o saibro dos terreiros e dos carreiros; debaixo das copas, mato.
2. Correções à mão (lista CORRECOES, por ordem; cada uma manda sobre as anteriores):
   - dentro das duas vilas (Magistral e Coroada) o chão entre as casas é calçada de granito, com os jardins do OSM
     em relva;
   - as ruas das passagens das portas e das calçadas sobre os fossos, dentro do contorno das muralhas, também;
   - as estradas de fora das muralhas são asfalto: o traçado do OSM, que chega a estar 10–20 m ao lado, assente na
     faixa cinzenta da ortofoto e com a largura medida.

    python pipeline/maquetas/valenca/chao.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import shapely

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1]))
from kit import chao  # noqa: E402

CFG = json.loads((AQUI.parents[2] / "src" / "data" / "maquetas" / "valenca.json").read_text())
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([-40.0, 10.0])  # o mesmo de construir.py


def ll(c):
    return np.array([(c[0] - LON0) * KX, (c[1] - LAT0) * KY]) - CENTRO


def geom(f):
    g = f["geometria"]
    if g["type"] == "Polygon":
        return shapely.Polygon([ll(c) for c in g["coordinates"][0]])
    if g["type"] == "LineString":
        return shapely.LineString([ll(c) for c in g["coordinates"]])
    return None


# As duas vilas dentro das muralhas, traçadas na ortofoto pelo lado de dentro dos terraplenos (onde acabam as casas,
# as ruas e os largos e começa a erva): a Magistral a norte e a Coroada a sul.
MAGISTRAL = chao.poligono([(-16, 308), (-3, 322), (22, 325), (41, 320), (62, 299), (81, 280), (97, 258), (112, 242),
                           (126, 232), (142, 205), (147, 160), (148, 110), (149, 65), (147, 27), (140, -10),
                           (131, -41), (115, -58), (84, -64), (62, -52), (44, -25), (38, 0), (20, 40), (4, 70),
                           (-2, 100), (-2, 180), (-14, 200), (-26, 230), (-28, 265), (-22, 290)])
COROADA = chao.poligono([(10, -104), (50, -100), (87, -108), (92, -125), (92, -206), (78, -240), (60, -270),
                         (33, -284), (-8, -293), (-50, -270), (-81, -243), (-90, -215), (-63, -188), (-31, -179),
                         (-8, -147), (-4, -115)])
VILAS = shapely.union_all([MAGISTRAL, COROADA])
JARDINS = shapely.union_all([geom(f) for f in F["usos"] if f["classe"] in ("grass", "garden", "park", "village_green",
                                                                            "playground")]).intersection(VILAS)
# Estradas de fora das muralhas: o traçado do OSM chega a estar 10–20 m ao lado da faixa que se vê na ortofoto, e
# não sempre para o mesmo lado, por isso cada estrada é primeiro assente na faixa cinzenta da ortofoto
# (chao.alinhar_osm, troços de 40 m). Dentro da faixa com a largura medida, o asfalto é o chão sem verde (as bermas
# de relva ficam). A Ecopista do Minho (antiga linha de comboio, a poente) também é asfaltada. As de serviço
# (corredores dos parques de estacionamento de terra batida) ficam com o chão da cor.
LARGURA = {"motorway": 11.0, "primary": 8.0, "secondary": 8.0, "tertiary": 7.0, "unclassified": 6.0,
           "residential": 7.0, "living_street": 5.0, "cycleway": 4.0}


# contorno das muralhas no OSM: por dentro (passagens das portas, calçadas sobre os fossos) as ruas são de granito
PRACA = next(geom(f) for f in F["infraestrutura"] if f["classe"] == "city_wall" and f["geometria"]["type"] == "Polygon")
_ESTRADAS = []


def estradas():
    if _ESTRADAS:
        return _ESTRADAS[0]
    masc = chao.mascara_asfalto(AQUI)
    fora = shapely.box(-288, -468, 288, 468)
    fs = []
    for f in F["caminhos"]:
        if f["classe"] not in LARGURA or f["geometria"]["type"] != "LineString":
            continue
        for g in shapely.get_parts(geom(f).intersection(fora).difference(VILAS)):
            if g.geom_type == "LineString" and g.length >= 8:
                fs.append(chao.alinhar_osm(g, masc).buffer(LARGURA[f["classe"]] / 2, cap_style="flat"))
    _ESTRADAS.append(shapely.union_all(fs).difference(VILAS.buffer(2.0)))
    return _ESTRADAS[0]


def asfalto(cl):
    return chao.pela_classe(estradas().difference(PRACA), "erva", "saibro", "rocha", "terra")(cl)


def calcada_portas(cl):
    return chao.pela_classe(estradas().intersection(PRACA), "erva", "saibro", "rocha", "terra")(cl)


CORRECOES = [
    ("calcada", chao.pela_classe(VILAS, "erva", "saibro", "rocha", "terra", "mato"),
     "vilas da Magistral e da Coroada: ruas, largos e quintais entre as casas em calçada de granito"),
    ("relva", JARDINS, "jardins e relvados do OSM dentro das vilas"),
    ("asfalto", asfalto,
     "estradas de fora das muralhas: o traçado do OSM assente na faixa cinzenta da ortofoto (chega a estar 10–20 m ao "
     "lado), com a largura medida; ficam as bermas de relva"),
    ("calcada", calcada_portas,
     "ruas dentro do contorno das muralhas fora das vilas (passagens das portas, calçada da Porta do Meio sobre o "
     "fosso): granito, como nas vilas"),
]


def main():
    cl, dom = chao.proposta(AQUI, claro=(0.66, 0.12), casa_folga=2.0, sombra=0.30, maioria_raio=3, area_min=30.0)
    cl = chao.aplicar(cl, CORRECOES, dom)
    chao.gravar(AQUI / "dados" / "chao.json", cl, {"fonte": "cor do LiDAR DGT 2025 + classes e MDT + correções à mão",
                                                    "correcoes": [c[2] for c in CORRECOES]})
    for n, g in cl.items():
        print(f"{n:8s} {g.area:9.0f} m²  {len(shapely.get_parts(g)):4d} polígonos")


if __name__ == "__main__":
    main()
