"""Castelo de Almourol: prepara o LiDAR da DGT (folhas 178278 e 178277, voo de junho de 2024) para a maqueta.

Grava em dados/ (grelha local: x para este, y para norte, em PT-TM06 menos a origem; linha 0 a sul):
- lidar.npz: MDT e MDS de 50 cm (cm acima de BASE), a classe do ponto mais alto de cada célula e a máscara da água
  (células com pontos da classe 9 e sem solo);
- superficie.npz: o ponto mais alto das classes 1, 2 e 6 (solo, não classificado e edifícios, sem vegetação) em
  células de 25 cm, para traçar e medir as muralhas, as torres e os adarves sem as copas das árvores;
- castelo.npz: todos os pontos das classes 1 a 6 na caixa do castelo (x, y, z, classe).

Ficheiros (Centro de Dados da DGT, cdd.dgterritorio.gov.pt), numa pasta:
    MDT-50cm-178278-06-2024, MDS-50cm-178278-06-2024 e LO-178278-06-2024 (nuvem de pontos), e, se existirem,
    os da folha 178277 (a sul: o braço sul do Tejo e a margem sul).
Uso:
    pip install rasterio pyproj "laspy[lazrs]" numpy scipy
    python pipeline/maquetas/almourol/lidar.py PASTA
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
LAT0, LON0 = 39.461973, -8.383981  # centro da torre de menagem (retângulo mínimo dos pontos do topo)
E0, N0 = Transformer.from_crs(4326, 3763, always_xy=True).transform(LON0, LAT0)
X0, X1, Y0, Y1 = -350.0, 110.0, -126.5, 215.0  # caixa (m); sem a folha 178277 acaba a sul em y = -127
CASTELO = (-25.0, 45.0, -30.0, 30.0)
BASE = 0.0


def grelha(passo, y0=None):
    xs = np.arange(X0, X1 + 1e-6, passo)
    ys = np.arange(Y0 if y0 is None else y0, Y1 + 1e-6, passo)
    return xs, ys


def main():
    global Y0
    pasta = Path(sys.argv[1])
    sul = sorted(pasta.glob("MDT-50cm-178277*.tif"))
    if sul:
        Y0 = -245.0
    xs, ys = grelha(0.5)
    X, Y = np.meshgrid(xs + E0, ys + N0)
    saida = {}
    for prod in ("MDT", "MDS"):
        v = np.full(X.shape, np.nan)
        for f in sorted(pasta.glob(f"{prod}-50cm-1782*.tif")):
            r = rasterio.open(f)
            a = r.read(1).astype(np.float64)
            a[a < -100] = np.nan
            c = (X - r.transform.c) / r.transform.a - 0.5
            rr = (Y - r.transform.f) / r.transform.e - 0.5
            dentro = (c >= 0) & (c <= a.shape[1] - 1) & (rr >= 0) & (rr <= a.shape[0] - 1)
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
    n_agua = np.zeros(n_y * n_x, np.int32)
    n_solo = np.zeros(n_y * n_x, np.int32)
    sx, sy = grelha(0.25)
    sup = np.full(len(sy) * len(sx), -1e9)
    cast = []
    for laz in sorted(pasta.glob("LO-1782*.laz")):
        with laspy.open(laz) as r:
            for ch in r.chunk_iterator(3_000_000):
                x, y, z = np.asarray(ch.x) - E0, np.asarray(ch.y) - N0, np.asarray(ch.z)
                c = np.asarray(ch.classification)
                m = (x > X0 - 1) & (x < X1 + 1) & (y > Y0 - 1) & (y < Y1 + 1)
                if not m.any():
                    continue
                x, y, z, c = x[m], y[m], z[m], c[m]
                gx = np.round((x - X0) / 0.5).astype(int)
                gy = np.round((y - Y0) / 0.5).astype(int)
                ok = (gx >= 0) & (gx < n_x) & (gy >= 0) & (gy < n_y)
                i = gy * n_x + gx
                np.add.at(n_agua, i[ok & (c == 9)], 1)
                np.add.at(n_solo, i[ok & (c == 2)], 1)
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
                t = (x > CASTELO[0]) & (x < CASTELO[1]) & (y > CASTELO[2]) & (y < CASTELO[3]) & (c >= 1) & (c <= 6)
                cast.append(np.stack([x[t], y[t], z[t], c[t]], 1))
            print("LAZ", laz.name)
    saida["classe"] = cls.reshape(n_y, n_x)
    agua = (n_agua > 0) & (n_solo == 0)
    saida["agua"] = ndi.binary_closing(agua.reshape(n_y, n_x), iterations=2)
    sup = np.where(sup < -1e8, np.nan, sup).reshape(len(sy), len(sx))
    vazio = np.isnan(sup)
    idx = ndi.distance_transform_edt(vazio, return_distances=False, return_indices=True)
    sup = sup[tuple(idx)]
    np.savez_compressed(AQUI / "dados" / "superficie.npz", passo=0.25, x0=X0, y0=Y0, base=BASE,
                        z=np.round((sup - BASE) * 100).astype(np.int16), vazio=vazio)
    T = np.concatenate(cast).astype(np.float32)
    np.savez_compressed(AQUI / "dados" / "castelo.npz", pontos=T, e0=E0, n0=N0)
    print("pontos do castelo", len(T), "superfície", sup.shape)
    np.savez_compressed(AQUI / "dados" / "lidar.npz", passo=0.5, x0=X0, y0=Y0, base=BASE, e0=E0, n0=N0, **saida,
                        fonte="DGT, LiDAR 2024 (MDT, MDS 50 cm e nuvem de pontos), folhas 178278 e 178277")


if __name__ == "__main__":
    main()
