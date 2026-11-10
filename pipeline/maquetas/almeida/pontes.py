"""Praça-forte de Almeida: mede as pontes de alvenaria sobre os fossos na nuvem de pontos da DGT e grava
dados/pontes.json, que o construir.py lê.

Nas folhas de 2024 as pontes vêm na classe 26 (o MDT passa por baixo delas, à cota do fundo do fosso).
Cada grupo de pontos dá o eixo (componente principal), o comprimento, a largura (percentis 3 e 97 na
perpendicular), a cota do tabuleiro (mediana) e a do topo das guardas (percentil 95).

Uso:
    python pipeline/maquetas/almeida/pontes.py PASTA_COM_OS_LAZ
"""
import json
import math
import sys
from pathlib import Path

import laspy
import numpy as np
from pyproj import Transformer
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
LAT0, LON0 = F["origem"]["lat"], F["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
NOMES = {  # ponto aproximado -> nome (portas no OSM, ver LEIAME)
    "ponte_sao_francisco": (135, -190),
    "ponte_santo_antonio": (-200, -47),
    "ponte_santo_antonio_revelim": (-333, -13),
}


def main():
    inv = Transformer.from_crs(3763, 4326, always_xy=True)
    pts = []
    for f in sorted(Path(sys.argv[1]).glob("LO-*.laz")):
        with laspy.open(f) as r:
            for ch in r.chunk_iterator(3_000_000):
                m = np.asarray(ch.classification) == 26
                if m.any():
                    lon, lat = inv.transform(np.asarray(ch.x)[m], np.asarray(ch.y)[m])
                    pts.append(np.column_stack([(lon - LON0) * KX, (lat - LAT0) * KY, np.asarray(ch.z)[m]]))
    P = np.vstack(pts)
    P = P[(np.abs(P[:, 0]) < 500) & (np.abs(P[:, 1]) < 470)]
    g = np.zeros((470, 500), bool)
    gi, gj = ((P[:, 1] + 470) // 2).astype(int), ((P[:, 0] + 500) // 2).astype(int)
    g[gi, gj] = True
    lab, k = ndi.label(ndi.binary_closing(g, iterations=2))
    out = {}
    for i in range(1, k + 1):
        q = P[lab[gi, gj] == i]
        if len(q) < 300:
            continue
        c = q[:, :2].mean(0)
        _, _, vt = np.linalg.svd(q[:, :2] - c, full_matrices=False)
        ax, pp = vt[0], vt[1]
        s, w = (q[:, :2] - c) @ ax, (q[:, :2] - c) @ pp
        nome = min(NOMES, key=lambda n: np.hypot(*(np.array(NOMES[n]) - c)))
        meio = c + ax * (s.max() + s.min()) / 2 + pp * (np.percentile(w, 97) + np.percentile(w, 3)) / 2
        out[nome] = {
            "centro": np.round(meio, 2).tolist(), "eixo": np.round(ax, 4).tolist(),
            "comprimento": round(float(s.max() - s.min()), 2),
            "largura": round(float(np.percentile(w, 97) - np.percentile(w, 3)), 2),
            "tabuleiro": round(float(np.median(q[:, 2])), 2), "guardas": round(float(np.percentile(q[:, 2], 95)), 2),
            "pontos": int(len(q)),
        }
        print(nome, out[nome])
    (AQUI / "dados" / "pontes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
