"""Castelo de Lindoso: procura os espigueiros no LiDAR e grava dados/espigueiros_lidar.json (o construir.py junta-os
às plantas do OSM, ficando com os que o OSM não tem).

Um espigueiro é uma mancha de 2,5 a 45 m² com 1,6 a 6 m acima do MDT, a sul e sudoeste do castelo (y < -50, x < 20).
Para cada mancha: o retângulo mínimo (comprimento, largura, azimute), a altura (percentil 97 e mediana acima do
chão), o chão e a fração de pontos da classe 6.

Uso:
    python pipeline/maquetas/lindoso/espigueiros.py
"""
import json
from pathlib import Path

import numpy as np
import shapely
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
S = np.load(AQUI / "dados" / "superficie.npz")
z = S["z"] / 100.0 + float(S["base"])
p, x0, y0 = float(S["passo"]), float(S["x0"]), float(S["y0"])
L = np.load(AQUI / "dados" / "lidar.npz")
T = L["mdt"] / 100.0 + float(L["base"])
C = L["classe"]
lp, lx0, ly0 = float(L["passo"]), float(L["x0"]), float(L["y0"])

ny, nx = z.shape
ys, xs = np.mgrid[0:ny, 0:nx]
X, Y = x0 + xs * p, y0 + ys * p
Tm = ndi.map_coordinates(T, [(Y - ly0) / lp, (X - lx0) / lp], order=1, mode="nearest")
Cm = C[np.clip(((Y - ly0) / lp).round().astype(int), 0, C.shape[0] - 1), np.clip(((X - lx0) / lp).round().astype(int), 0, C.shape[1] - 1)]
h = z - Tm
B = (h > 1.6) & (h < 6.0) & (X < 20) & (Y < -50)
B = ndi.binary_opening(B, iterations=1)
lab, n = ndi.label(B)
tr = Affine(p, 0, x0 - p / 2, 0, p, y0 - p / 2)
res = []
for geo, v in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=tr):
    g = shapely.geometry.shape(geo)
    if g.area < 2.5 or g.area > 45:
        continue
    c = np.array(g.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = c[1] - c[0], c[2] - c[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if l1 < l2:
        e1, e2, l1, l2 = e2, e1, l2, l1
    m = lab == int(v)
    hh = h[m]
    res.append(dict(c=np.round(c.mean(0), 2).tolist(), L=round(float(l1), 2), W=round(float(l2), 2),
                    az=round(float(np.degrees(np.arctan2(e1[0], e1[1])) % 180), 1), area=round(g.area, 1),
                    fill=round(g.area / (l1 * l2), 2), hmax=round(float(np.percentile(hh, 97)), 2),
                    hmed=round(float(np.median(hh)), 2), chao=round(float(np.percentile(Tm[m], 20)), 2),
                    cls6=round(float((Cm[m] == 6).mean()), 2)))
res.sort(key=lambda r: (r["c"][1], r["c"][0]))
(AQUI / "dados" / "espigueiros_lidar.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")))
print("manchas", len(res))
