"""Fortaleza de Peniche: parte a massa construída do LiDAR (células de classe 6 da nuvem de pontos, mais de
1 m acima do MDT) em plataformas, isto é, superfícies contínuas do MDS separadas por degraus de mais de
0,7 m: os terraplenos e adarves das muralhas, os pátios altos e os telhados dos edifícios da prisão. Cada
plataforma fica com o contorno (simplificado) e o plano que melhor se ajusta ao topo; o construir.py
levanta-as em panos e paredes. Grava dados/plataformas.json.

Uso:
    python pipeline/maquetas/peniche/plataformas.py
"""
import json
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
L = np.load(AQUI / "dados" / "lidar.npz")
PASSO = float(L["passo"])
MDT = L["mdt"] / 100.0
MDS = L["mds"] / 100.0
CLS = L["classe"]
XS = np.linspace(-float(L["meiox"]), float(L["meiox"]), MDT.shape[1])
YS = np.linspace(-float(L["meioy"]), float(L["meioy"]), MDT.shape[0])

# Contorno das obras da fortaleza (traçado à mão sobre o MDS): o que fica fora é a cidade, feita do OSM
FORTALEZA = shapely.Polygon([(-182, -52), (-182, 96), (-100, 102), (-45, 128), (40, 128), (112, 122), (162, 66),
                             (162, -22), (124, -22), (60, -12), (0, -98), (-60, -62), (-150, -52)])
CORTE = 0.7     # degrau que separa plataformas (m)
AREA_MIN = 5.0  # m²


def main():
    X, Y = np.meshgrid(XS, YS)
    h = MDS - MDT
    dentro = shapely.contains_xy(FORTALEZA, X, Y)
    b = (CLS == 6) & (h > 1.0) & dentro
    b = ndi.binary_opening(ndi.binary_closing(b, iterations=1), iterations=1)
    # suaviza o MDS só dentro da massa (mediana 3×3): tira o ruído e os merlões finos
    S = ndi.median_filter(np.where(b, MDS, np.nan), size=3, mode="nearest")
    S = np.where(np.isnan(S), MDS, S)
    borda = np.zeros_like(b)
    for ax in (0, 1):
        g = np.abs(np.diff(S, axis=ax)) > CORTE
        if ax == 0:
            borda[:-1] |= g
            borda[1:] |= g
        else:
            borda[:, :-1] |= g
            borda[:, 1:] |= g
    lab, k = ndi.label(b & ~borda)
    # regiões pequenas saem; as células de borda e das regiões pequenas vão para a vizinha mais próxima
    tam = ndi.sum(np.ones_like(lab), lab, np.arange(1, k + 1)) * PASSO * PASSO
    pequenas = np.isin(lab, np.nonzero(tam < AREA_MIN)[0] + 1)
    lab[pequenas] = 0
    _, (ri, ci) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    lab = np.where(b, lab[ri, ci], 0)
    out = []
    import rasterio.features as rf
    from rasterio.transform import Affine
    tf = Affine(PASSO, 0, XS[0] - PASSO / 2, 0, PASSO, YS[0] - PASSO / 2)
    for g, v in rf.shapes(lab.astype(np.int32), mask=lab > 0, transform=tf, connectivity=4):
        i = int(v)
        p = shapely.geometry.shape(g).buffer(0)
        p = p.buffer(0.3, join_style="mitre").buffer(-0.3, join_style="mitre").simplify(0.45)
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda q: q.area)
        if p.is_empty or p.area < AREA_MIN:
            continue
        m = lab == i
        z = S[m]
        A = np.column_stack([np.ones(m.sum()), X[m], Y[m]])
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        res = float(np.percentile(np.abs(A @ coef - z), 90))
        if res > 0.5:  # topo irregular (escadas, telhados de águas): plano horizontal na mediana
            coef = np.array([float(np.median(z)), 0.0, 0.0])
        chao = MDT[m]
        out.append({"id": len(out), "contorno": np.round(np.array(p.exterior.coords)[:-1], 2).tolist(),
                    "buracos": [np.round(np.array(r.coords)[:-1], 2).tolist() for r in p.interiors if shapely.Polygon(r).area > 4],
                    "plano": np.round(coef, 5).tolist(), "residuo": round(res, 2),
                    "topo": round(float(np.median(z)), 2), "chao": round(float(np.percentile(chao, 10)), 2),
                    "area": round(float(p.area), 1)})
    muros = tracar_muros(dentro, b, X, Y)
    (AQUI / "dados" / "plataformas.json").write_text(json.dumps({"plataformas": out, "muros": muros}, separators=(",", ":")))
    print("plataformas", len(out), "área", round(sum(o["area"] for o in out)), "m²", "muros", len(muros),
          "com", round(sum(m["comprimento"] for m in muros)), "m")


def tracar_muros(dentro, macico, X, Y):
    """Parapeitos e muros finos: no relevo local do MDS (MDS menos a abertura 9×9) os parapeitos, os
    merlões e os muros de pátio sobressaem mais de 0,45 m, seja qual for a classe (a cortina da costa vem
    como solo). Esqueleto da máscara e troços seguidos de vizinho em vizinho, simplificados a 0,35 m. Para
    cada troço grava-se, de 0,5 em 0,5 m, a cota do MDS (máximo 3×3) e o relevo, para o construir.py ler a
    altura do parapeito e os merlões."""
    from skimage.morphology import skeletonize
    rel = MDS - ndi.grey_opening(MDS, size=(9, 9))
    veg = (CLS >= 3) & (CLS <= 5)
    fino = (rel > 0.45) & dentro & ~ndi.binary_dilation(veg, iterations=1)
    # os merlões e as canhoneiras juntam-se numa faixa contínua antes do esqueleto
    fino = ndi.binary_closing(fino, iterations=3) & dentro
    fino = ndi.binary_opening(fino, iterations=1)
    esq = skeletonize(fino)
    pix = set(zip(*np.nonzero(esq)))
    viz = lambda p: [(p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (p[0] + a, p[1] + b) in pix]
    usados = set()
    linhas = []
    pontas = [p for p in pix if len(viz(p)) == 1]
    for inicio in pontas + sorted(pix):
        if inicio in usados:
            continue
        cam = [inicio]
        usados.add(inicio)
        while True:
            nx_ = [q for q in viz(cam[-1]) if q not in usados]
            if not nx_:
                break
            q = min(nx_, key=lambda q: abs(q[0] - cam[-1][0]) + abs(q[1] - cam[-1][1]))
            cam.append(q)
            usados.add(q)
        if len(cam) >= 14:
            linhas.append(cam)
    Smax = ndi.maximum_filter(MDS, size=3)
    out = []
    for cam in linhas:
        r, c = np.array(cam).T
        ln = shapely.LineString(np.column_stack([XS[c], YS[r]])).simplify(0.35)
        if ln.length < 5:
            continue
        q = np.array(ln.coords)
        dd = np.arange(0, ln.length + 1e-6, 0.5)
        pts = np.array([ln.interpolate(d).coords[0] for d in dd])
        ci = np.clip(np.searchsorted(XS, pts[:, 0]), 0, len(XS) - 1)
        ri = np.clip(np.searchsorted(YS, pts[:, 1]), 0, len(YS) - 1)
        out.append({"linha": np.round(q, 2).tolist(), "comprimento": round(ln.length, 1),
                    "cota": np.round(Smax[ri, ci], 2).tolist(), "relevo": np.round(rel[ri, ci], 2).tolist(),
                    "classe6": round(float((CLS[ri, ci] == 6).mean()), 2)})
    return out


if __name__ == "__main__":
    main()
