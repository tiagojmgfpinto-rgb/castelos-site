"""Classes do chão da maqueta de Melgaço -> dados/chao.json (método "metro a metro", novembro de 2026).

1. Proposta automática a 0,5 m (kit/chao.py, grelha do LiDAR 2025 de x0, y0): cor do ortofotomapa (dados/orto.jpg),
   classes e declive: relva, erva, mato debaixo das copas (os quintais e a encosta a norte), saibro e rocha.
2. Correções à mão (lista CORRECOES, por ordem; cada uma manda sobre as anteriores):
   - o recinto do castelo é relvado, com os afloramentos de granito onde o chão sobressai do relevo à volta (MDT);
   - o carreiro de lajes da entrada e o pátio da barbacã (fotografia de cima): lajeado;
   - as ruas, vielas, largos e escadas do centro histórico (traçado do OSM, que aqui assenta bem no ortofotomapa,
     com a largura da classe), o chão sem verde até 6 m das casas e o claro de fora do castelo: calçada de granito.

    python pipeline/maquetas/melgaco/chao.py
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

F = json.loads((AQUI / "dados" / "fontes.json").read_text())  # já em metros da cena
MUR = json.loads((AQUI / "dados" / "muralhas.json").read_text())
CAIXA = shapely.box(-112.0, -40.0, 46.0, 46.0)  # X0, Y0, X1, Y1 de construir.py
CENTRO = (-0.56, -1.09)  # centro do recinto (tracar.py, como em construir.py)


def recinto():
    """A face de dentro da cerca, pelas estações de 2 em 2 graus de dados/muralhas.json (rin), com uma mediana de 5
    estações (como em construir.py: as escadas em V e as torres coladas dão leituras falsas)."""
    az = np.array([e["az"] for e in MUR["cerca"] if e and e.get("rin")])
    r = np.array([e["rin"] for e in MUR["cerca"] if e and e.get("rin")])
    rm = np.array([np.median(np.take(r, range(i - 2, i + 3), mode="wrap")) for i in range(len(r))])
    a = np.radians(az)
    return shapely.Polygon(np.column_stack([CENTRO[0] + rm * np.sin(a), CENTRO[1] + rm * np.cos(a)])).buffer(0)


RECINTO = recinto()
CASAS = [shapely.Polygon(f["geometria"]["coordinates"][0]).buffer(0) for f in F["edificios"]
         if f["geometria"]["type"] == "Polygon"]
LARGURA = {"residential": 5.0, "service": 4.0, "pedestrian": 4.0, "unknown": 4.0, "footway": 2.5, "steps": 2.5}
RUAS = shapely.union_all([shapely.LineString(f["geometria"]["coordinates"]).buffer(LARGURA[f["classe"]] / 2,
                                                                                    cap_style="flat")
                          for f in F["caminhos"] if f["geometria"]["type"] == "LineString"
                          and f["classe"] in LARGURA]).difference(RECINTO.buffer(1.0))


def _casas_lidar():
    """Os telhados da classe 6 do LiDAR (há casas que o OSM não tem, entre a Rua Direita e a muralha sul)."""
    lid = np.load(AQUI / "dados" / "lidar.npz")
    xs, ys, passo = chao.grelha(lid)
    m = ndi.binary_opening(lid["classe"] == 6, iterations=2)
    g = chao.raster_para_poligonos(np.where(m, 0, -1), xs[0], ys[0], passo, ["casa"], area_min=12.0, alisar=0.5)
    return g.get("casa", shapely.Polygon())


URBANO = shapely.union_all([c.buffer(6.0) for c in CASAS] + [_casas_lidar().buffer(5.0)]).difference(RECINTO.buffer(1.0))
# carreiro de lajes da entrada (fotografia de cima) e pátio da barbacã, como em construir.py
CARREIRO = shapely.union_all([shapely.LineString([(-3.0, -29.5), (-3.0, -24.0), (-2.0, -19.5)]).buffer(0.9),
                              shapely.Polygon([(-5.5, -28.6), (2.6, -28.6), (2.6, -32.6), (-5.5, -32.6)])])


def afloramentos(cl):
    """Os penedos de granito do recinto: o MDT acima do relevo alisado (sigma de 2 m) por mais de 9 cm, em manchas."""
    lid = np.load(AQUI / "dados" / "lidar.npz")
    xs, ys, passo = chao.grelha(lid)
    mdt = lid["mdt"] / 100.0
    rel = ndi.gaussian_filter(mdt - ndi.gaussian_filter(mdt, 4.0), 0.8)  # sigma em células, como em construir.py
    m = ndi.binary_opening(ndi.gaussian_filter((rel > 0.09).astype(float), 1.2) > 0.5, iterations=1)
    g = chao.raster_para_poligonos(np.where(m, 0, -1), xs[0], ys[0], passo, ["rocha"], area_min=2.0, alisar=0.5)
    return g.get("rocha", shapely.Polygon()).intersection(RECINTO.buffer(-0.5))


CORRECOES = [
    ("relva", RECINTO, "recinto do castelo: relvado"),
    ("rocha", afloramentos, "afloramentos de granito no recinto (onde o MDT sobressai do relevo à volta)"),
    ("lajeado", CARREIRO, "carreiro de lajes da entrada e pátio da barbacã (fotografia de cima)"),
    ("calcada", RUAS, "ruas, vielas, largos e escadas do centro histórico: o traçado do OSM com a largura da classe, "
                      "em calçada de granito"),
    ("calcada", chao.pela_classe(URBANO, "erva", "saibro", "rocha"),
     "pátios e largos entre as casas: o chão sem verde até 6 m das casas (do OSM e da classe 6 do LiDAR) é calçada"),
    ("mato", chao.pela_classe(shapely.box(-72.5, -27.5, -59.0, -12.5), "calcada", "erva"),
     "o quintal murado a sul da Rua Direita: arvoredo cerrado no ortofotomapa (os muros dele são classe 6 no LiDAR)"),
    ("calcada", chao.pela_classe(shapely.box(-112.0, -40.0, -92.0, -12.0), "erva", "rocha"),
     "a rua e o largo à frente da Porta de Baixo (cinzentos com a passadeira no ortofotomapa; a sombra das casas "
     "dava erva)"),
    ("calcada", chao.pela_classe(CAIXA.difference(RECINTO.buffer(1.0)), "saibro"),
     "o claro de fora do castelo (o largo a noroeste, o passeio ao pé da muralha a poente, a rua dos carros a "
     "nordeste): no centro histórico não há saibro, é calçada"),
]


def main():
    cl, _ = chao.proposta(AQUI, claro=(0.62, 0.10), casa_folga=2.0, sombra=0.28, area_min=6.0)
    cl = chao.aplicar(cl, CORRECOES, CAIXA)
    chao.gravar(AQUI / "dados" / "chao.json", cl, {"fonte": "cor do ortofotomapa DGT 2021 + classes e MDT do LiDAR "
                                                            "2025 + correções à mão",
                                                    "correcoes": [c[2] for c in CORRECOES]})
    for n, g in cl.items():
        print(f"{n:8s} {g.area:9.0f} m²  {len(shapely.get_parts(g)):4d} polígonos")


if __name__ == "__main__":
    main()
