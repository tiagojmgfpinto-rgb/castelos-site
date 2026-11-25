"""Classes do chão da maqueta de Castro Laboreiro -> dados/chao.json (método "metro a metro", novembro de 2026).

1. Proposta automática a 0,5 m (kit/chao.py): cor da nuvem de pontos do LiDAR 2025 (dados/orto.jpg), classes e declive.
   No monte não há saibro (o claro e neutro são as lajes de granito); erva seca dourada no recinto e nas encostas,
   relva onde a cor é verde, mato debaixo das copas (carvalhal a poente).
2. Correções à mão (lista CORRECOES): os trilhos de terra do OSM (acesso pela Porta do Sol, carreiros do recinto,
   escadaria da porta norte).

As lajes e os penedos de granito não são classe do chão: ficam no objeto Afloramentos (construir.py), com a pedra
fotografada, e o chão é recortado por baixo deles.

    python pipeline/maquetas/castro-laboreiro/chao.py
    python pipeline/kit/quadricula.py castro-laboreiro --classes
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

CFG = json.loads((AQUI.parents[2] / "src" / "data" / "maquetas" / "castro-laboreiro.json").read_text())
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([0.0, 0.0])  # o mesmo de construir.py


def ll(c):
    return np.array([(c[0] - LON0) * KX, (c[1] - LAT0) * KY]) - CENTRO


def osm_linha(i):
    return [tuple(ll(c)) for c in F["caminhos"][i]["geometria"]["coordinates"]]


CORRECOES = [
    *[("terra", chao.faixa(osm_linha(i), 1.8 if f["classe"] == "steps" else 1.4, "round"),
       f"trilho de terra do OSM n.º {i} ({f['classe']}): faixa batida de c. 1,2–1,8 m na ortofoto")
      for i, f in enumerate(F["caminhos"]) if f["classe"] in ("path", "footway", "steps", "track")
      and f["geometria"]["type"] == "LineString"],
]


def main():
    cl, meio = chao.proposta(AQUI, saibro=False)
    cl = chao.aplicar(cl, CORRECOES, meio)
    chao.gravar(AQUI / "dados" / "chao.json", cl, {"fonte": "cor do LiDAR DGT 2025 + classes e MDT + correções à mão",
                                                    "correcoes": [c[2] for c in CORRECOES]})
    for n, g in cl.items():
        print(f"{n:8s} {g.area:9.0f} m²  {len(shapely.get_parts(g)):4d} polígonos")


if __name__ == "__main__":
    main()
