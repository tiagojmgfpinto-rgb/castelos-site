"""Classes do chão da maqueta de Guimarães -> dados/chao.json (método "metro a metro", novembro de 2026).

1. Proposta automática a 0,5 m (kit/chao.py): cor da nuvem de pontos do LiDAR 2025 (dados/orto.jpg), classes e declive.
   Afinada para a Colina Sagrada: os carreiros do parque são saibro claro (lum > 0,70) e estreitos (maioria de raio 2),
   as sombras das árvores na relva ficam com a classe do vizinho ao sol, e o chão debaixo das copas também (é um parque
   de relva aparada, não mato: Street View).
2. Correções à mão (lista CORRECOES, por ordem; cada uma manda sobre as anteriores), vistas nas quadrículas de 40 m
   (kit/quadricula.py) e no Street View (25 nov. 2026):
   - carreiros do parque: saibro claro e compactado, com relva rente (Street View junto à Igreja de São Miguel);
   - Rua Conde Dom Henrique e Largo Martins Sarmento: cubos de granito (Street View);
   - adro do Paço: cubos de granito em quadros com faixas de lajes (Street View à frente da fachada poente);
   - Rua Dr. Joaquim de Meira, Rua Dona Teresa e a N101 a sueste: asfalto (Street View);
   - à volta da igreja e no pátio do Paço: lajeado de granito.

    python pipeline/maquetas/guimaraes/chao.py
    python pipeline/kit/quadricula.py guimaraes --classes      # ver o resultado quadrícula a quadrícula
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

CFG = json.loads((AQUI.parents[2] / "src" / "data" / "maquetas" / "guimaraes.json").read_text())
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([-40.0, -90.0])
DESVIO_OSM = np.array([1.0, 0.0])  # o mesmo de construir.py


def ll(c):
    return np.array([(c[0] - LON0) * KX, (c[1] - LAT0) * KY]) - CENTRO + DESVIO_OSM


def osm_linha(i, d=(0.0, 0.0)):
    """Eixo de um caminho do OSM, deslocado d metros para o assentar no meio da faixa que se vê na ortofoto."""
    return [tuple(ll(c) + d) for c in F["caminhos"][i]["geometria"]["coordinates"]]


def osm_aneis(prefixo, lista="edificios"):
    f = next(f for f in F[lista] if f["id"].startswith(prefixo))
    return [[tuple(ll(c)) for c in r[:-1]] for r in f["geometria"]["coordinates"]]


def edificio_lidar(centro, raio):
    """Planta de um edifício no LiDAR (células de classe 6 mais de 1,5 m acima do MDT, a maior mancha perto de centro)."""
    lid = np.load(AQUI / "dados" / "lidar.npz")
    meio, passo = float(lid["meio"]), float(lid["passo"])
    xs = np.linspace(-meio, meio, lid["mdt"].shape[0])
    X, Y = np.meshgrid(xs, xs)
    q = (np.hypot(X - centro[0], Y - centro[1]) < raio) & (lid["classe"] == 6) & (lid["mds"] - lid["mdt"] > 150)
    lab, k = ndi.label(q)
    q = lab == (np.argmax(ndi.sum(q, lab, range(1, k + 1))) + 1)
    return shapely.MultiPoint(np.stack([X[q], Y[q]], 1)).buffer(passo * 0.75, quad_segs=2).buffer(-0.2).simplify(0.4)


# A Colina Sagrada (parque do castelo, do Paço e da igreja), traçada na ortofoto pelo lado de dentro das ruas que a
# cercam: Rua Conde Dom Henrique e Largo Martins Sarmento a poente, Rua Dona Teresa a noroeste, Rua Duques de Bragança
# a nascente, a N101 (Rua Dona Constança de Noronha) a sueste e o edifício do antigo convento a sul. Fora dela é vila.
PARQUE = chao.poligono([(-94, 30), (-70, 52), (-50, 72), (-35, 98), (-20, 122), (-5, 138), (15, 152), (40, 160),
                        (80, 175), (105, 135), (103, 80), (108, 62), (118, 38), (133, 12), (145, -5), (175, -20),
                        (175, -50), (150, -55), (118, -95), (98, -115), (78, -135), (50, -175), (-30, -175),
                        (-32, -125), (-98, -125), (-135, -104), (-102, -42), (-94, -36)])
VILA = chao.disco(180).difference(PARQUE)
# adro do Paço: o largo claro entre a Rua Conde Dom Henrique, o terreiro da estátua e a fachada poente do Paço
ADRO = chao.poligono([(-94, -36), (-92, -18), (-80, -12), (-62, -12), (-48, -18), (-36, -26), (-30, -40), (-34, -58),
                      (-50, -64), (-70, -62), (-86, -56), (-102, -42)])
IGREJA = edificio_lidar((-6.0, 19.0), 14.0)
PACO_PATIO = chao.poligono(osm_aneis("f4db2746")[1])

# corredores de 4 m para cada lado dos carreiros do OSM no parque (o traçado chega a estar 3 m fora): dentro deles, o
# carreiro é o que a cor diz (claro, sem verde, fora das copas)
CARREIROS = shapely.union_all([shapely.LineString(osm_linha(i)).buffer(4.0) for i, f in enumerate(F["caminhos"])
                               if f["classe"] in ("footway", "path", "steps")
                               and f["geometria"]["type"] == "LineString"]).intersection(PARQUE).difference(ADRO)
SEM_RELVA = ("asfalto", "erva", "saibro", "mato", "rocha", "terra")  # as copas das árvores de arruamento pendem sobre a rua

CORRECOES = [
    ("saibro", chao.pela_cor(AQUI, CARREIROS, lum_min=0.60, sat_max=0.16, verde_max=0.03),
     "carreiros do parque: saibro claro e compactado (Street View junto à igreja); a forma é a da faixa clara da "
     "ortofoto num corredor de 4 m à volta do traçado do OSM, porque o filtro de maioria apagava os de 2–3 m"),
    ("asfalto", chao.pela_classe(VILA, "erva", "saibro", "rocha", "terra"),
     "vila: o chão sem verde nem copas fora do parque são ruas, passeios e parques de estacionamento (asfalto na "
     "ortofoto e no Street View); a fronteira é a da cor"),
    ("calcada", chao.pela_classe(chao.faixa(osm_linha(83, (-3.0, 0.0)), 15.0, "round"), *SEM_RELVA),
     "Rua Conde Dom Henrique (OSM 83, 3 m a nascente do meio da rua na ortofoto): cubos de granito de parede a "
     "parede, com os carros e o passeio largo (Street View), c. 15 m; os canteiros de relva ficam"),
    ("calcada", chao.pela_classe(chao.faixa(osm_linha(91, (-3.75, 3.3)), 12.0, "round"), *SEM_RELVA),
     "Largo Martins Sarmento (OSM 91, que corre pela berma sueste da rua na ortofoto: eixo 5 m para noroeste): "
     "cubos de granito como a Rua Conde Dom Henrique, c. 12 m"),
    ("calcada", chao.pela_classe(ADRO, "erva", "saibro", "asfalto"),
     "adro do Paço: cubos de granito em quadros com faixas de lajes (Street View); fica a relva dos canteiros"),
    ("lajeado", IGREJA.buffer(2.2, join_style="mitre").difference(IGREJA.buffer(0.3)).union(IGREJA),
     "faixa de lajes de granito à volta da Igreja de São Miguel (Street View: c. 2 m do lado norte)"),
    ("lajeado", PACO_PATIO, "pátio do Paço: lajeado de granito"),
]


def main():
    cl, meio = chao.proposta(AQUI, claro=(0.70, 0.13), rocha_decl=58.0, casa_folga=1.5, sombra=0.33, maioria_raio=2, verde_min=0.03,
                             copa=None)
    cl = chao.aplicar(cl, CORRECOES, meio)
    chao.gravar(AQUI / "dados" / "chao.json", cl, {"fonte": "cor do LiDAR DGT 2025 + classes e MDT + correções à mão",
                                                    "correcoes": [c[2] for c in CORRECOES]})
    for n, g in cl.items():
        print(f"{n:8s} {g.area:9.0f} m²  {len(shapely.get_parts(g)):4d} polígonos")


if __name__ == "__main__":
    main()
