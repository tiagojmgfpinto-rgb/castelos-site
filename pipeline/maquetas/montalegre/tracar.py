"""Castelo de Montalegre: traça no LiDAR as plantas da cerca, da barbacã a nascente e dos muros baixos (ruínas do
pátio) e grava-as em dados/plantas.json (coordenadas da cena, em metros).

Método: células de 25 cm do ponto mais alto das classes 1, 2 e 6 (superficie.npz); relevo local = altura acima do
mínimo num disco de 3 m; muro = relevo > 1,0 m (cerca e barbacã); tiram-se as torres
(comum.TORRES); fecho e abertura morfológicos, polígonos simplificados a 15 cm.

    python pipeline/maquetas/montalegre/tracar.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import shapely
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import comum as C  # noqa: E402

S = np.load(AQUI / "dados" / "superficie.npz")
SUP = S["z"] / 100.0 + float(S["base"])
P, X0, Y0 = float(S["passo"]), float(S["x0"]), float(S["y0"])
TR = Affine(P, 0, X0 - P / 2, 0, P, Y0 - P / 2)
ys = Y0 + np.arange(SUP.shape[0]) * P
xs = X0 + np.arange(SUP.shape[1]) * P
X, Y = np.meshgrid(xs, ys)
fp = np.hypot(*np.mgrid[-12:13, -12:13]) <= 12
H = SUP - ndi.minimum_filter(np.where(S["vazio"], 9999, SUP), footprint=fp)
TORRES = shapely.union_all([shapely.Polygon(C.planta_torre(k, 0.05)) for k in C.TORRES])


def poligonos(msk, area_min, simp=0.15):
    out = []
    for g, _ in features.shapes(msk.astype(np.uint8), mask=msk, transform=TR):
        q = shapely.geometry.shape(g).buffer(0)
        if q.area >= area_min:
            out.append(q.simplify(simp))
    return out


def dentro(pol):
    return shapely.contains_xy(pol, X, Y)


def main():
    # cerca: o anel entre a torre de menagem e as três torres do sul
    anel = shapely.Point(-4.5, -22.5).buffer(26.0).difference(shapely.Point(-4.5, -22.5).buffer(11.0))
    anel = anel.intersection(shapely.box(-30, -45, 15.2, 2.0))
    m = (H > 1.0) & dentro(anel) & ~dentro(TORRES)
    m = ndi.binary_closing(m, iterations=3)
    m = ndi.binary_opening(m, iterations=2) & ~dentro(TORRES.buffer(-0.05))
    cerca = shapely.union_all(poligonos(m, 15.0))
    cerca = cerca.buffer(0.6, join_style="mitre").buffer(-0.6, join_style="mitre").difference(TORRES).simplify(0.15)
    # barbacã de nascente: o muro de suporte a nascente e a sul, de fora da cerca
    zona = shapely.Polygon([(11.5, 14), (19.5, 14), (21.5, -36), (14, -52), (-6, -60), (-8, -53), (8, -47), (13.5, -37)])
    m = (H > 0.9) & dentro(zona) & ~dentro(TORRES.buffer(0.5)) & ~dentro(cerca.buffer(0.6))
    m = ndi.binary_opening(ndi.binary_closing(m, iterations=2), iterations=1)
    barb = [q for q in poligonos(m, 6.0) if q.length > 12]
    # perfil da cerca por azimute (a partir do centro do pátio): face de dentro e de fora do topo e cota do topo
    cx, cy = C.CENTRO_PATIO
    ang = (np.degrees(np.arctan2(X - cx, Y - cy)) + 360) % 360
    rad = np.hypot(X - cx, Y - cy)
    ok = (H > 1.0) & (rad > 12) & (rad < 24) & ~dentro(TORRES.buffer(0.3)) & ~S["vazio"]
    perfil = []
    for a in np.arange(0, 360, 2.5):
        k = ok & (np.abs(((ang - a + 180) % 360) - 180) < 2.5)
        if k.sum() < 12:
            perfil.append([float(a), None, None, None])
            continue
        z, r = SUP[k], rad[k]
        zt = float(np.percentile(z, 92))
        t = z > zt - 0.45
        perfil.append([float(a), round(float(np.percentile(r[t], 4)), 2), round(float(np.percentile(r[t], 96)), 2), round(zt, 2)])
    d = {"nota": "plantas traçadas no LiDAR de 2024 (tracar.py); x para este, y para norte, origem no centro da torre de menagem",
         "cerca": shapely.geometry.mapping(cerca), "barbaca": [shapely.geometry.mapping(q) for q in barb],
         "perfil_cerca": {"centro": [cx, cy], "colunas": ["azimute", "r_dentro", "r_fora", "z_topo"], "linhas": perfil}}
    (AQUI / "dados" / "plantas.json").write_text(json.dumps(d, separators=(",", ":")))
    print("cerca", round(cerca.area, 1), cerca.geom_type, "barbacã", [round(q.area, 1) for q in barb])


if __name__ == "__main__":
    main()
