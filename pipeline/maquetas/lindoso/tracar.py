"""Castelo de Lindoso: traça no LiDAR as muralhas (a cerca medieval, a muralha abaluartada do século XVII e o
revelim) e grava dados/muralhas.json, que o construir.py lê.

Os muros do castelo destacam-se do chão dos dois lados (a cerca medieval tem c. 6 m acima do pátio; a muralha
abaluartada, de 1,2 a 2 m de espessura, 2 a 4 m acima da lice e do campo). Aqui:
1. tira-se ao ponto mais alto das classes solo, edifícios e muros (superficie.npz, 25 cm) a sua abertura
   morfológica com um disco de 3 m de raio (white top-hat): ficam os muros, que são mais estreitos do que o disco;
2. os muros são as células com mais de 0,7 m de top-hat e mais de 0,9 m acima do MDT (assim as lices, que o
   MDT tem como chão, não entram, mesmo onde são mais estreitas do que o disco), sem a torre de menagem, os edifícios do museu, a torre da
   porta e o chão (escadas e rampas ficam de fora por terem menos de 0,7 m de degrau); fecham-se buracos de
   menos de 1 m² e tiram-se manchas de menos de 3 m²;
3. cada mancha vira um polígono (contorno alisado e simplificado a 0,45 m, ângulos com mais de 3,5 m de lado afiados);
4. em cada vértice mede-se o topo (percentil 90 do LiDAR num raio de 0,6 m dentro do polígono) e o chão de fora
   (o mais baixo do MDT entre 0,6 e 2,0 m para fora), para as faces descerem até ao chão.

Uso:
    pip install numpy scipy shapely rasterio scikit-image
    python pipeline/maquetas/lindoso/tracar.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import shapely
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage as ndi
from skimage.morphology import remove_small_holes, remove_small_objects

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from comum import EXCLUIR_MUROS, ZONA_CASTELO  # noqa: E402

SUPF = np.load(AQUI / "dados" / "superficie.npz")
SP, SX0, SY0 = float(SUPF["passo"]), float(SUPF["x0"]), float(SUPF["y0"])
SUP = SUPF["z"] / 100.0 + float(SUPF["base"])
LID = np.load(AQUI / "dados" / "lidar.npz")
LP, LX0, LY0 = float(LID["passo"]), float(LID["x0"]), float(LID["y0"])
MDT = LID["mdt"] / 100.0 + float(LID["base"])


def amostra(G, x0, y0, p, x, y, ordem=1):
    return ndi.map_coordinates(G, [(np.asarray(y) - y0) / p, (np.asarray(x) - x0) / p], order=ordem, mode="nearest")


def afiar(lin, minimo=1.2):
    """Troca lados curtos (cantos arredondados pelo raster) pela interseção dos dois lados vizinhos."""
    lin = [np.asarray(p, float) for p in lin]
    mudou = True
    while mudou and len(lin) > 4:
        mudou = False
        n = len(lin)
        segs = [np.linalg.norm(lin[(i + 1) % n] - lin[i]) for i in range(n)]
        for i in sorted(range(n), key=lambda k: segs[k]):
            if segs[i] >= minimo:
                break
            p1, p2, p3, p4 = lin[(i - 1) % n], lin[i], lin[(i + 1) % n], lin[(i + 2) % n]
            if segs[(i - 1) % n] < 2.5 or segs[(i + 1) % n] < 2.5:
                continue
            d1, d2 = p2 - p1, p4 - p3
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-6:
                continue
            t = ((p3 - p1)[0] * d2[1] - (p3 - p1)[1] * d2[0]) / den
            x = p1 + d1 * t
            if np.linalg.norm(x - (p2 + p3) / 2) > 1.5:
                continue
            j = (i + 1) % n
            lin[i] = x
            del lin[j]
            mudou = True
            break
    return np.array(lin)


def main():
    r = int(round(3.0 / SP))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disco = xx ** 2 + yy ** 2 <= r * r
    zf = ndi.median_filter(SUP, 3)
    th = zf - ndi.grey_opening(zf, footprint=disco)
    ny, nx = SUP.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    mdt = amostra(MDT, LX0, LY0, LP, SX0 + xx * SP, SY0 + yy * SP)
    acima = zf - mdt  # altura acima do chão (o MDT tem as lices e o pátio como chão)
    tr = Affine(SP, 0, SX0 - SP / 2, 0, SP, SY0 - SP / 2)
    zona = features.rasterize([shapely.Polygon(ZONA_CASTELO)], out_shape=(ny, nx), transform=tr).astype(bool)
    excl = features.rasterize([shapely.Polygon(p) for p in EXCLUIR_MUROS], out_shape=(ny, nx), transform=tr).astype(bool)
    M = (th > 0.7) & (acima > 0.9) & zona & ~excl
    M = ndi.binary_opening(M, iterations=1)
    M = remove_small_holes(M, max_size=16)
    M = remove_small_objects(M, max_size=48)
    pols = []
    for geo, val in features.shapes(M.astype(np.uint8), mask=M, transform=tr):
        g = shapely.geometry.shape(geo).buffer(0)
        g = g.buffer(0.35, join_style="mitre").buffer(-0.5, join_style="mitre").buffer(0.15, join_style="mitre")
        for q in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            if q.area < 3:
                continue
            furos = [h for h in q.interiors if shapely.Polygon(h).area > 4.0]  # pátios e lices; sem os buracos pequenos
            q = shapely.Polygon(q.exterior, furos).simplify(0.45)
            if q.geom_type != "Polygon" or q.area < 3:
                continue
            ext = afiar(np.array(q.exterior.coords)[:-1])
            ints = [afiar(np.array(h.coords)[:-1]) for h in q.interiors]
            q2 = shapely.Polygon(ext, [h for h in ints if len(h) >= 3]).buffer(0)
            if q2.geom_type == "MultiPolygon":
                q2 = max(q2.geoms, key=lambda g: g.area)
            if q2.geom_type != "Polygon" or q2.area < 3:
                continue
            q2 = shapely.geometry.polygon.orient(q2, 1)  # exterior anti-horário, furos horários
            pols.append([np.array(q2.exterior.coords)[:-1]] + [np.array(h.coords)[:-1] for h in q2.interiors])
    out = []
    for i, aneis in enumerate(pols):
        reg = {"id": "", "aneis": [], "area": round(float(shapely.Polygon(aneis[0], aneis[1:]).area), 1)}
        for c in aneis:
            reg["aneis"].append(medir_anel(c))
        a0 = reg["aneis"][0]
        if np.median(np.array(a0["topo"]) - np.array(a0["chao"])) < 0.9:
            continue  # pedras e muretes com menos de 0,9 m acima do chão de fora
        out.append(reg)
    out.sort(key=lambda e: -e["area"])
    for i, e in enumerate(out):
        e["id"] = f"m{i:02d}"
    (AQUI / "dados" / "muralhas.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("muros", len(out), "área", round(sum(e["area"] for e in out)), "m²")
    for e in out:
        print(e["id"], [len(a["contorno"]) for a in e["aneis"]], e["area"], "topo", min(min(a["topo"]) for a in e["aneis"]),
              max(max(a["topo"]) for a in e["aneis"]))


def medir_anel(c):
    """Topo e chão de fora em cada vértice de um anel (o exterior anti-horário ou um furo horário: o "fora" fica
    sempre à direita do sentido do anel)."""
    if True:
        n = len(c)
        nrm = []
        for k in range(n):
            t1 = c[k] - c[k - 1]
            t2 = c[(k + 1) % n] - c[k]
            a = np.array([t1[1], -t1[0]]) / max(np.linalg.norm(t1), 1e-9)
            b = np.array([t2[1], -t2[0]]) / max(np.linalg.norm(t2), 1e-9)
            nrm.append((a + b) / max(np.linalg.norm(a + b), 1e-9))  # para fora (polígono anti-horário)
        nrm = np.array(nrm)
        topo, chao = [], []
        for k in range(n):
            pts = [c[k] - nrm[k] * d + np.array([dx, dy]) for d in (0.35, 0.6, 0.85) for dx in (-0.25, 0, 0.25) for dy in (-0.25, 0, 0.25)]
            pts = np.array(pts)
            topo.append(float(np.percentile(amostra(SUP, SX0, SY0, SP, pts[:, 0], pts[:, 1], 0), 85)))
            fora = np.array([c[k] + nrm[k] * d for d in np.arange(0.6, 2.01, 0.2)])
            chao.append(float(amostra(MDT, LX0, LY0, LP, fora[:, 0], fora[:, 1]).min()))
        topo = np.array(topo)
        # topo plano por troços: mediana com os vizinhos (o LiDAR tem ervas e pedras soltas no alto dos muros)
        tm = np.array([np.median(topo[[(k - 1) % n, k, (k + 1) % n]]) for k in range(n)])
        return {"contorno": np.round(c, 2).tolist(), "topo": np.round(tm, 2).tolist(), "chao": np.round(chao, 2).tolist()}


if __name__ == "__main__":
    main()
