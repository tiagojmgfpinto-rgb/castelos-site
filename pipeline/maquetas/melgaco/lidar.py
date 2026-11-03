"""Castelo de Melgaço: prepara o LiDAR da DGT (folha 189572, 2025) para a maqueta.

Grava em dados/ (grelha local: x para este, y para norte, em PT-TM06 menos a origem; linha 0 a sul):
- lidar.npz: MDT e MDS de 50 cm (cm acima de BASE) e a classe do ponto mais alto de cada célula;
- superficie.npz: o ponto mais alto das classes 1, 2 e 6 (solo, edifícios e muros, sem vegetação) em células
  de 25 cm, para traçar e medir as muralhas, a torre e as escadas sem as copas das árvores;
- torre.npz: todos os pontos a menos de 9 m do centro da torre (x, y, z, classe).

Ficheiros (Centro de Dados da DGT, cdd.dgterritorio.gov.pt), numa pasta:
    MDT-50cm-189572-07-2025, MDS-50cm-189572-07-2025 e LO-189572-07-2025 (nuvem de pontos).
Uso:
    pip install rasterio pyproj "laspy[lazrs]" numpy scipy
    python pipeline/maquetas/melgaco/lidar.py PASTA
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
LAT0, LON0 = 42.114413, -8.259790
E0, N0 = Transformer.from_crs(4326, 3763, always_xy=True).transform(LON0, LAT0)
X0, X1, Y0, Y1 = -140.0, 60.0, -80.0, 60.0  # caixa da maqueta (m)
BASE = 170.0


def grelha(passo):
    xs = np.arange(X0, X1 + 1e-6, passo)
    ys = np.arange(Y0, Y1 + 1e-6, passo)
    return xs, ys


def main():
    pasta = Path(sys.argv[1])
    xs, ys = grelha(0.5)
    X, Y = np.meshgrid(xs + E0, ys + N0)
    saida = {}
    for prod in ("MDT", "MDS"):
        f = next(pasta.glob(f"{prod}-50cm-189572*.tif"))
        r = rasterio.open(f)
        a = r.read(1).astype(np.float64)
        c = (X - r.transform.c) / r.transform.a - 0.5
        rr = (Y - r.transform.f) / r.transform.e - 0.5
        v = ndi.map_coordinates(a, [rr, c], order=1, mode="nearest")
        saida[prod.lower()] = np.round((v - BASE) * 100).astype(np.int16)
        print(prod, f.name, round(float(v.min()), 2), round(float(v.max()), 2))
    import laspy
    n_y, n_x = len(ys), len(xs)
    alto = np.full(n_y * n_x, -1e9)
    cls = np.zeros(n_y * n_x, np.uint8)
    sx, sy = grelha(0.25)
    sup = np.full(len(sy) * len(sx), np.nan)
    torre = []
    with laspy.open(next(pasta.glob("LO-189572*.laz"))) as r:
        for ch in r.chunk_iterator(3_000_000):
            x, y, z = np.asarray(ch.x) - E0, np.asarray(ch.y) - N0, np.asarray(ch.z)
            c = np.asarray(ch.classification)
            m = (x > X0 - 1) & (x < X1 + 1) & (y > Y0 - 1) & (y < Y1 + 1) & (c >= 1) & (c <= 6)
            if not m.any():
                continue
            t = m & (np.hypot(x, y) < 9)
            torre.append(np.stack([x[t], y[t], z[t], c[t]], 1))
            gx = np.round((x[m] - X0) / 0.5).astype(int)
            gy = np.round((y[m] - Y0) / 0.5).astype(int)
            ok = (gx >= 0) & (gx < n_x) & (gy >= 0) & (gy < n_y)
            i, zz, cc = gy[ok] * n_x + gx[ok], z[m][ok], c[m][ok]
            o = np.argsort(zz)
            i, zz, cc = i[o], zz[o], cc[o]
            sobe = zz > alto[i]
            np.maximum.at(alto, i, zz)
            cls[i[sobe]] = cc[sobe]
            k = m & np.isin(c, (1, 2, 6))
            gx = np.floor((x[k] - X0) / 0.25 + 0.5).astype(int)
            gy = np.floor((y[k] - Y0) / 0.25 + 0.5).astype(int)
            ok = (gx >= 0) & (gx < len(sx)) & (gy >= 0) & (gy < len(sy))
            j = gy[ok] * len(sx) + gx[ok]
            s0 = np.where(np.isnan(sup), -1e9, sup)
            np.maximum.at(s0, j, z[k][ok])
            sup = np.where(s0 < -1e8, np.nan, s0)
    saida["classe"] = cls.reshape(n_y, n_x)
    sup = sup.reshape(len(sy), len(sx))
    vazio = np.isnan(sup)
    idx = ndi.distance_transform_edt(vazio, return_distances=False, return_indices=True)
    sup = sup[tuple(idx)]
    np.savez_compressed(AQUI / "dados" / "superficie.npz", passo=0.25, x0=X0, y0=Y0, base=BASE,
                        z=np.round((sup - BASE) * 100).astype(np.int16), vazio=vazio)
    T = np.concatenate(torre).astype(np.float32)
    np.savez_compressed(AQUI / "dados" / "torre.npz", pontos=T, e0=E0, n0=N0)
    print("pontos da torre", len(T), "superfície", sup.shape)
    np.savez_compressed(AQUI / "dados" / "lidar.npz", passo=0.5, x0=X0, y0=Y0, base=BASE, e0=E0, n0=N0, **saida,
                        fonte="DGT, LiDAR 2025 (MDT, MDS 50 cm e nuvem de pontos), folha 189572")


if __name__ == "__main__":
    main()
