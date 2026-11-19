"""Castelo e muralhas de Silves: prepara o LiDAR da DGT (folhas 172025, 172026, 173025 e 173026, voo de abril de 2024).

O castelo fica no canto comum das quatro folhas. Grava em dados/ (grelha local: x para este, y para norte, em PT-TM06
menos a origem; linha 0 a sul):
- lidar.npz: MDT e MDS de 50 cm (cm acima de BASE) e a classe do ponto mais alto de cada célula;
- superficie.npz: o ponto mais alto das classes 1, 2 e 6 (sem vegetação) em células de 25 cm, para traçar e medir
  as muralhas, as torres e os adarves sem as copas das árvores;
- orto_lidar.jpg: a cor RGB do ponto mais alto de cada célula de 25 cm (todas as classes), um ortofotomapa do dia
  do voo, porque o WMS da DGT não responde a partir do ambiente onde a maqueta se constrói;
- castelo.npz: todos os pontos das classes 1 a 6 nas caixas da alcáçova e da Porta da Cidade (x, y, z, classe, r, g, b).

Uso:
    pip install rasterio pyproj "laspy[lazrs]" numpy scipy
    python pipeline/maquetas/silves/lidar.py PASTA   # com MDT-50cm-*, MDS-50cm-* e LO-* das quatro folhas
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from comum import E0, N0, X0, X1, Y0, Y1, CAIXAS  # noqa: E402

BASE = 0.0
FOLHAS = ("172025", "172026", "173025", "173026")


def grelha(passo):
    return np.arange(X0, X1 + 1e-6, passo), np.arange(Y0, Y1 + 1e-6, passo)


def main():
    pasta = Path(sys.argv[1])
    xs, ys = grelha(0.5)
    X, Y = np.meshgrid(xs + E0, ys + N0)
    saida = {}
    for prod in ("MDT", "MDS"):
        v = np.full(X.shape, np.nan)
        for fl in FOLHAS:
            for f in sorted(pasta.glob(f"{prod}-50cm-{fl}*.tif")):
                r = rasterio.open(f)
                a = r.read(1).astype(np.float64)
                a[a < -100] = np.nan
                c = (X - r.transform.c) / r.transform.a - 0.5
                rr = (Y - r.transform.f) / r.transform.e - 0.5
                dentro = (c >= -0.5) & (c <= a.shape[1] - 0.5) & (rr >= -0.5) & (rr <= a.shape[0] - 0.5)
                w = ndi.map_coordinates(np.nan_to_num(a, nan=-999), [rr, c], order=1, mode="nearest")
                v = np.where(dentro & np.isnan(v) & (w > -100), w, v)
                print(prod, f.name)
        idx = ndi.distance_transform_edt(np.isnan(v), return_distances=False, return_indices=True)
        v = v[tuple(idx)]
        saida[prod.lower()] = np.round((v - BASE) * 100).astype(np.int16)
        print(prod, round(float(np.nanmin(v)), 2), round(float(np.nanmax(v)), 2))
    import laspy
    n_y, n_x = len(ys), len(xs)
    alto = np.full(n_y * n_x, -1e9)
    cls = np.zeros(n_y * n_x, np.uint8)
    sx, sy = grelha(0.25)
    sup = np.full(len(sy) * len(sx), -1e9)
    topo = np.full(len(sy) * len(sx), -1e9)
    cor = np.zeros((len(sy) * len(sx), 3), np.uint8)
    cast = []
    for fl in FOLHAS:
        for laz in sorted(pasta.glob(f"LO-{fl}*.laz")):
            with laspy.open(laz) as r:
                for ch in r.chunk_iterator(3_000_000):
                    x, y, z = np.asarray(ch.x) - E0, np.asarray(ch.y) - N0, np.asarray(ch.z)
                    c = np.asarray(ch.classification)
                    m = (x > X0 - 1) & (x < X1 + 1) & (y > Y0 - 1) & (y < Y1 + 1)
                    if not m.any():
                        continue
                    rgb = np.stack([np.asarray(ch.red), np.asarray(ch.green), np.asarray(ch.blue)], 1)[m] >> 8
                    x, y, z, c = x[m], y[m], z[m], c[m]
                    gx = np.round((x - X0) / 0.5).astype(int)
                    gy = np.round((y - Y0) / 0.5).astype(int)
                    ok = (gx >= 0) & (gx < n_x) & (gy >= 0) & (gy < n_y)
                    i = gy * n_x + gx
                    k = ok & (c >= 1) & (c <= 6)
                    ii, zz, cc = i[k], z[k], c[k]
                    o = np.argsort(zz)
                    ii, zz, cc = ii[o], zz[o], cc[o]
                    sobe = zz > alto[ii]
                    np.maximum.at(alto, ii, zz)
                    cls[ii[sobe]] = cc[sobe]
                    k = np.isin(c, (1, 2, 6))
                    gx = np.floor((x[k] - X0) / 0.25 + 0.5).astype(int)
                    gy = np.floor((y[k] - Y0) / 0.25 + 0.5).astype(int)
                    ok2 = (gx >= 0) & (gx < len(sx)) & (gy >= 0) & (gy < len(sy))
                    np.maximum.at(sup, gy[ok2] * len(sx) + gx[ok2], z[k][ok2])
                    gx = np.floor((x - X0) / 0.25 + 0.5).astype(int)
                    gy = np.floor((y - Y0) / 0.25 + 0.5).astype(int)
                    ok3 = (gx >= 0) & (gx < len(sx)) & (gy >= 0) & (gy < len(sy)) & (c != 7)
                    j3, z3, c3 = gy[ok3] * len(sx) + gx[ok3], z[ok3], rgb[ok3]
                    o = np.argsort(z3)
                    j3, z3, c3 = j3[o], z3[o], c3[o]
                    sobe = z3 > topo[j3]
                    np.maximum.at(topo, j3, z3)
                    cor[j3[sobe]] = c3[sobe]
                    t = np.zeros(len(x), bool)
                    for (a0, a1, b0, b1) in CAIXAS.values():
                        t |= (x > a0) & (x < a1) & (y > b0) & (y < b1)
                    t &= (c >= 1) & (c <= 6)
                    cast.append(np.concatenate([np.stack([x[t], y[t], z[t], c[t]], 1), rgb[t]], 1))
                print("LAZ", laz.name, flush=True)
    saida["classe"] = cls.reshape(n_y, n_x)
    sup = np.where(sup < -1e8, np.nan, sup).reshape(len(sy), len(sx))
    vazio = np.isnan(sup)
    idx = ndi.distance_transform_edt(vazio, return_distances=False, return_indices=True)
    sup = sup[tuple(idx)]
    np.savez_compressed(AQUI / "dados" / "superficie.npz", passo=0.25, x0=X0, y0=Y0, base=BASE,
                        z=np.round((sup - BASE) * 100).astype(np.int16), vazio=vazio)
    from PIL import Image
    cor = cor.reshape(len(sy), len(sx), 3)
    falta = topo.reshape(len(sy), len(sx)) < -1e8
    idx = ndi.distance_transform_edt(falta, return_distances=False, return_indices=True)
    Image.fromarray(cor[tuple(idx)][::-1]).save(AQUI / "dados" / "orto_lidar.jpg", quality=90)
    T = np.concatenate(cast).astype(np.float32)
    np.savez_compressed(AQUI / "dados" / "castelo.npz", pontos=T, e0=E0, n0=N0)
    print("pontos do castelo", len(T), "superfície", sup.shape)
    np.savez_compressed(AQUI / "dados" / "lidar.npz", passo=0.5, x0=X0, y0=Y0, base=BASE, e0=E0, n0=N0, **saida,
                        fonte="DGT, LiDAR 2024 (MDT, MDS 50 cm e nuvem de pontos), folhas 172025, 172026, 173025 e 173026")


if __name__ == "__main__":
    main()
