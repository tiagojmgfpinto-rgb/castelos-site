"""Forte de São Sebastião (Angra do Heroísmo): prepara o LiDAR de 2024 da ilha Terceira para a maqueta.

Fonte: levantamento LiDAR da Região Autónoma dos Açores de 2024, bloco Angra (24-4-2-5): nuvem de pontos
classificada com cor e infravermelho (2024_TER_ANG_24-4-2-5_CCL_RGBI.laz) e os modelos de 1 m (…_01_MDS.tif e
…_01_MDE.tif), em PTRA08 / UTM 26N (EPSG:5015). A grelha da maqueta é a UTM menos a origem (x para este, y para
norte); a convergência de meridianos aqui é de 0,13°, que se ignora (0,2 m a 100 m).

Classes da nuvem: 1 não classificado, 2 solo, 3–5 vegetação, 6 edifício, 7 ruído, 11 pavimento de estrada,
12 sobreposição de fiadas (todos os pontos da fiada 30, sem outra classe), 17 tabuleiro de ponte, 19 estruturas
suspensas. As muralhas e os terraplenos do forte estão na 2 (solo) e na 6.

Grava em dados/ (linha 0 a sul):
- lidar.npz: MDT (solo, 50 cm: pontos da classe 2 e, onde faltam, o MDE de 1 m), MDS (ponto mais alto sem ruído,
  50 cm), classe do ponto mais alto (sem a 12) e árvores (máximos locais da vegetação);
- superficie.npz: ponto mais alto das classes 1, 2, 6, 11, 17 e 19 em células de 25 cm (muros, adarves e
  telhados sem as copas);
- orto.jpg: cor do ponto mais alto de cada célula de 25 cm (norte em cima), a ortoimagem do próprio voo.

Uso:
    pip install rasterio pyproj "laspy[lazrs]" numpy scipy pillow
    python pipeline/maquetas/sao-sebastiao-angra/lidar.py LAZ MDE.tif
"""
import sys
from pathlib import Path

import laspy
import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
E0, N0 = 481553.4, 4278168.5          # origem (PTRA08 / UTM 26N): 38,651886 N, 27,211995 W, no recinto do forte
X0, X1, Y0, Y1 = -200.0, 200.0, -168.0, 200.0   # a folha acaba a sul em N = 4 278 000 (y = −168,5): mar
SUP_CLS = (1, 2, 6, 11, 17, 19)


def grelha(p):
    return np.arange(X0, X1 + 1e-6, p), np.arange(Y0, Y1 + 1e-6, p)


def main():
    laz, mde = Path(sys.argv[1]), Path(sys.argv[2])
    xs, ys = grelha(0.5)
    nx, ny = len(xs), len(ys)
    sx, sy = grelha(0.25)
    snx, sny = len(sx), len(sy)
    mds = np.full(nx * ny, -1e9)
    cls = np.zeros(nx * ny, np.uint8)
    solo = np.full(nx * ny, np.nan)
    solo_n = np.zeros(nx * ny)
    sup = np.full(snx * sny, -1e9)
    ztop = np.full(snx * sny, -1e9)
    rgb = np.zeros((snx * sny, 3), np.uint8)
    solo_s = np.zeros(nx * ny)
    with laspy.open(laz) as r:
        for ch in r.chunk_iterator(4_000_000):
            x, y, z = np.asarray(ch.x) - E0, np.asarray(ch.y) - N0, np.asarray(ch.z)
            c = np.asarray(ch.classification)
            m = (x > X0 - 0.5) & (x < X1 + 0.5) & (y > Y0 - 0.5) & (y < Y1 + 0.5) & (c != 7)
            if not m.any():
                continue
            x, y, z, c = x[m], y[m], z[m], c[m]
            col = (np.stack([np.asarray(ch.red)[m], np.asarray(ch.green)[m], np.asarray(ch.blue)[m]], 1) // 256).astype(np.uint8)
            o = np.argsort(z)
            x, y, z, c, col = x[o], y[o], z[o], c[o], col[o]
            gx = np.clip(np.round((x - X0) / 0.5).astype(int), 0, nx - 1)
            gy = np.clip(np.round((y - Y0) / 0.5).astype(int), 0, ny - 1)
            i = gy * nx + gx
            sobe = z > mds[i]
            np.maximum.at(mds, i, z)
            k = sobe & (c != 12)
            cls[i[k]] = c[k]
            g = c == 2
            np.add.at(solo_s, i[g], z[g])
            np.add.at(solo_n, i[g], 1)
            gx = np.clip(np.round((x - X0) / 0.25).astype(int), 0, snx - 1)
            gy = np.clip(np.round((y - Y0) / 0.25).astype(int), 0, sny - 1)
            j = gy * snx + gx
            k = np.isin(c, SUP_CLS)
            np.maximum.at(sup, j[k], z[k])
            sobe = z > ztop[j]
            np.maximum.at(ztop, j, z)
            rgb[j[sobe]] = col[sobe]
            print(".", end="", flush=True)
    print()

    def encher(a):
        v = np.isnan(a)
        idx = ndi.distance_transform_edt(v, return_distances=False, return_indices=True)
        return a[tuple(idx)], v

    mds = np.where(mds < -1e8, np.nan, mds).reshape(ny, nx)
    mds, _ = encher(mds)
    solo = np.where(solo_n > 0, solo_s / np.maximum(solo_n, 1), np.nan).reshape(ny, nx)
    # onde não há solo (debaixo dos edifícios e no mar), o MDE de 1 m da mesma campanha
    with rasterio.open(mde) as r:
        a = r.read(1).astype(float)
        a[a < -100] = np.nan
        X, Y = np.meshgrid(xs + E0, ys + N0)
        cc = (X - r.transform.c) / r.transform.a - 0.5
        rr = (Y - r.transform.f) / r.transform.e - 0.5
        w = ndi.map_coordinates(np.nan_to_num(a, nan=-999), [rr, cc], order=1, mode="nearest")
    w = np.where(w < -100, np.nan, w)
    sem = np.isnan(solo) | (cls.reshape(ny, nx) == 6)
    mdt = np.where(sem, w, solo)
    mdt, _ = encher(mdt)
    mdt = np.where(np.isnan(w) & sem, 0.0, mdt)   # mar fora da cobertura: nível do mar
    sup = np.where(sup < -1e8, np.nan, sup).reshape(sny, snx)
    sup, vazio = encher(sup)
    cheio = (ztop > -1e8).reshape(sny, snx)
    _, (ri, ci) = ndi.distance_transform_edt(~cheio, return_indices=True)
    img = rgb.reshape(sny, snx, 3)[ri, ci][::-1]
    Image.fromarray(img).save(AQUI / "dados" / "orto.jpg", quality=90)
    cl = cls.reshape(ny, nx)
    arv = arvores(mdt, mds, cl, xs, ys)
    np.savez_compressed(AQUI / "dados" / "lidar.npz", passo=0.5, x0=X0, y0=Y0, e0=E0, n0=N0,
                        mdt=np.round(mdt * 100).astype(np.int16), mds=np.round(mds * 100).astype(np.int16),
                        classe=cl, arvores=arv,
                        fonte="Região Autónoma dos Açores, LiDAR 2024 da ilha Terceira, bloco 24-4-2-5 (Angra)")
    np.savez_compressed(AQUI / "dados" / "superficie.npz", passo=0.25, x0=X0, y0=Y0,
                        z=np.round(sup * 100).astype(np.int16), vazio=vazio)
    print("MDT", round(float(mdt.min()), 2), round(float(mdt.max()), 2), "MDS", round(float(mds.max()), 2),
          "superfície", sup.shape, "árvores", len(arv))


def arvores(mdt, mds, cls, xs, ys):
    """Copas: máximos locais da altura da vegetação (MDS − MDT nas células de classe 3 a 5)."""
    h = mds - mdt
    veg = (cls >= 3) & (cls <= 5) & (h > 2.0)
    veg = ndi.binary_opening(veg, iterations=1)
    hs = ndi.gaussian_filter(np.where(veg, h, 0.0), 1.2)
    topo = (hs == ndi.maximum_filter(hs, size=7)) & veg & (hs > 2.5)
    marc, k = ndi.label(topo)
    if k == 0:
        return np.zeros((0, 5), np.float32)
    d, (ri, ci) = ndi.distance_transform_edt(marc == 0, return_indices=True)
    lab = np.where(veg & (d * 0.5 < 6.0), marc[ri, ci], 0)
    area = ndi.sum(np.ones_like(h), lab, index=np.arange(1, k + 1)) * 0.25
    pos = np.array(ndi.center_of_mass(topo, marc, np.arange(1, k + 1)))
    r_, c_ = np.round(pos).astype(int).T
    out = np.stack([xs[c_], ys[r_], mds[r_, c_], mdt[r_, c_], np.sqrt(area / np.pi)], 1)
    return out[out[:, 4] >= 0.8].astype(np.float32)


if __name__ == "__main__":
    main()
