"""Castelo de Montalegre: prepara o LiDAR da DGT (folha 228540, 2024) para a maqueta.

Grava em dados/ (grelha local: x para este, y para norte, em PT-TM06 menos a origem; linha 0 a sul):
- lidar.npz: MDT e MDS de 50 cm (cm acima de BASE) e a classe do ponto mais alto de cada célula;
- superficie.npz: o ponto mais alto das classes 1, 2 e 6 (solo, edifícios e muros, sem vegetação) em células
  de 25 cm, e a cor RGB desse ponto (a nuvem de 2024 vem colorida), para traçar e medir as torres, a cerca e a
  barbacã sem as copas das árvores;
- torre.npz: todos os pontos a menos de 16 m do centro da torre de menagem (x, y, z, classe, r, g, b).

Ficheiros (Centro de Dados da DGT, cdd.dgterritorio.gov.pt), numa pasta:
    MDT-50cm-228540-09-2024, MDS-50cm-228540-09-2024 e LO-228540-09-2024 (nuvem de pontos).
Uso:
    pip install rasterio pyproj "laspy[lazrs]" numpy scipy
    python pipeline/maquetas/montalegre/lidar.py PASTA
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from comum import BASE, E0, N0, X0, X1, Y0, Y1  # noqa: E402

FOLHA = "228540"


def grelha(passo):
    return np.arange(X0, X1 + 1e-6, passo), np.arange(Y0, Y1 + 1e-6, passo)


def main():
    pasta = Path(sys.argv[1])
    xs, ys = grelha(0.5)
    X, Y = np.meshgrid(xs + E0, ys + N0)
    saida = {}
    for prod in ("MDT", "MDS"):
        f = next(pasta.glob(f"{prod}-50cm-{FOLHA}*.tif"))
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
    ns = len(sy) * len(sx)
    sup = np.full(ns, -1e9)
    cor = np.zeros((ns, 3), np.uint8)
    torre = []
    with laspy.open(next(pasta.glob(f"LO-{FOLHA}*.laz"))) as r:
        for ch in r.chunk_iterator(3_000_000):
            x, y, z = np.asarray(ch.x) - E0, np.asarray(ch.y) - N0, np.asarray(ch.z)
            c = np.asarray(ch.classification)
            m = (x > X0 - 1) & (x < X1 + 1) & (y > Y0 - 1) & (y < Y1 + 1) & (c >= 1) & (c <= 6)
            if not m.any():
                continue
            rgb = np.stack([np.asarray(ch.red), np.asarray(ch.green), np.asarray(ch.blue)], 1)
            rgb = (rgb >> 8).astype(np.uint8) if rgb.max() > 255 else rgb.astype(np.uint8)
            t = m & (np.hypot(x, y) < 16)
            torre.append(np.concatenate([np.stack([x[t], y[t], z[t], c[t]], 1), rgb[t]], 1))
            gx = np.round((x[m] - X0) / 0.5).astype(int)
            gy = np.round((y[m] - Y0) / 0.5).astype(int)
            ok = (gx >= 0) & (gx < n_x) & (gy >= 0) & (gy < n_y)
            i, zz, cc = gy[ok] * n_x + gx[ok], z[m][ok], c[m][ok]
            o = np.argsort(zz)
            i, zz, cc = i[o], zz[o], cc[o]
            np.maximum.at(alto, i, zz)
            cls[i] = cc  # ordenados por z: fica a classe do mais alto
            k = m & np.isin(c, (1, 2, 6))
            gx = np.floor((x[k] - X0) / 0.25 + 0.5).astype(int)
            gy = np.floor((y[k] - Y0) / 0.25 + 0.5).astype(int)
            ok = (gx >= 0) & (gx < len(sx)) & (gy >= 0) & (gy < len(sy))
            j, zk, ck = gy[ok] * len(sx) + gx[ok], z[k][ok], rgb[k][ok]
            o = np.argsort(zk)
            j, zk, ck = j[o], zk[o], ck[o]
            sobe = zk > sup[j]
            np.maximum.at(sup, j, zk)
            cor[j[sobe]] = ck[sobe]
    saida["classe"] = cls.reshape(n_y, n_x)
    sup = sup.reshape(len(sy), len(sx))
    cor = cor.reshape(len(sy), len(sx), 3)
    vazio = sup < -1e8
    idx = ndi.distance_transform_edt(vazio, return_distances=False, return_indices=True)
    sup, cor = sup[tuple(idx)], cor[tuple(idx)]
    np.savez_compressed(AQUI / "dados" / "superficie.npz", passo=0.25, x0=X0, y0=Y0, base=BASE,
                        z=np.round((sup - BASE) * 100).astype(np.int16), vazio=vazio, cor=cor)
    T = np.concatenate(torre).astype(np.float32)
    np.savez_compressed(AQUI / "dados" / "torre.npz", pontos=T, e0=E0, n0=N0)
    print("pontos da torre", len(T), "superfície", sup.shape)
    np.savez_compressed(AQUI / "dados" / "lidar.npz", passo=0.5, x0=X0, y0=Y0, base=BASE, e0=E0, n0=N0, **saida,
                        fonte=f"DGT, LiDAR 2024 (MDT, MDS 50 cm e nuvem de pontos), folha {FOLHA}")


if __name__ == "__main__":
    main()
