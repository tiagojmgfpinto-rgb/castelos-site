"""Torre de Menagem de Braga: prepara o LiDAR e o ortofotomapa da DGT para a maqueta.

Grava em dados/:
- lidar.npz: MDT e MDS de 50 cm (cm acima de BASE) e a classe do ponto mais alto de cada célula, numa
  grelha de ±MEIO m à volta do centro da torre (x para este, y para norte, em PT-TM06; linha 0 a sul);
- torre.npz: todos os pontos da nuvem a menos de 16 m do centro (x, y, z, classe), para as medições;
- orto.jpg: o ortofotomapa de 2021 (25 cm, RGB) recortado na mesma área (linha 0 a norte).

Ficheiros (Centro de Dados da DGT, cdd.dgterritorio.gov.pt), numa pasta:
    MDT-50cm-175510-07-2025 e MDS-50cm-175510-07-2025 (LiDAR 2025), LO-175510-07-2025 (nuvem de pontos),
    ortos2021_cog_25cm_rgbi_jpg_070-1 (Ortofotomapa 2021).
Uso:
    pip install rasterio pyproj "laspy[lazrs]" numpy pillow
    python pipeline/maquetas/braga/lidar.py PASTA
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

AQUI = Path(__file__).resolve().parent
LAT0, LON0 = 41.551322, -8.423892
E0, N0 = Transformer.from_crs(4326, 3763, always_xy=True).transform(LON0, LAT0)
MEIO = 70.0
PASSO = 0.5
BASE = 180.0


def main():
    pasta = Path(sys.argv[1])
    n = int(2 * MEIO / PASSO) + 1
    xs = np.linspace(-MEIO, MEIO, n)
    saida = {}
    for prod in ("MDT", "MDS"):
        f = next(pasta.glob(f"*{prod}-50cm-175510*.tif"))
        r = rasterio.open(f)
        a = r.read(1).astype(np.float64)
        # centros das células do raster de 50 cm coincidem com a grelha (E0, N0 não são múltiplos: bilinear)
        X, Y = np.meshgrid(xs + E0, xs + N0)
        c = (X - r.transform.c) / r.transform.a - 0.5
        rr = (Y - r.transform.f) / r.transform.e - 0.5
        c0, r0 = np.floor(c).astype(int), np.floor(rr).astype(int)
        fc, fr = c - c0, rr - r0
        v = (a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c0 + 1] * fc * (1 - fr) + a[r0 + 1, c0] * (1 - fc) * fr
             + a[r0 + 1, c0 + 1] * fc * fr)
        saida[prod.lower()] = np.round((v - BASE) * 100).astype(np.int16)
        print(prod, f.name, round(float(v.min()), 2), round(float(v.max()), 2))
    import laspy
    alto = np.full(n * n, -1e9)
    cls = np.zeros(n * n, np.uint8)
    torre = []
    with laspy.open(next(pasta.glob("LO-175510*.laz"))) as r:
        for ch in r.chunk_iterator(3_000_000):
            x, y, z = np.asarray(ch.x) - E0, np.asarray(ch.y) - N0, np.asarray(ch.z)
            c = np.asarray(ch.classification)
            m = (np.abs(x) < MEIO + 1) & (np.abs(y) < MEIO + 1) & (c >= 2) & (c <= 6)
            if not m.any():
                continue
            t = m & (np.hypot(x, y) < 16)
            torre.append(np.stack([x[t], y[t], z[t], c[t]], 1))
            gx = np.round((x[m] + MEIO) / PASSO).astype(int)
            gy = np.round((y[m] + MEIO) / PASSO).astype(int)
            ok = (gx >= 0) & (gx < n) & (gy >= 0) & (gy < n)
            i, zz, cc = gy[ok] * n + gx[ok], z[m][ok], c[m][ok]
            o = np.argsort(zz)
            i, zz, cc = i[o], zz[o], cc[o]
            sobe = zz > alto[i]
            np.maximum.at(alto, i, zz)
            cls[i[sobe]] = cc[sobe]
    saida["classe"] = cls.reshape(n, n)
    T = np.concatenate(torre).astype(np.float32)
    np.savez_compressed(AQUI / "dados" / "torre.npz", pontos=T, e0=E0, n0=N0)
    print("pontos da torre", len(T))
    np.savez_compressed(AQUI / "dados" / "lidar.npz", passo=PASSO, meio=MEIO, base=BASE, e0=E0, n0=N0, **saida,
                        fonte="DGT, LiDAR 2025 (MDT, MDS 50 cm e nuvem de pontos), folha 175510")
    from PIL import Image
    o = rasterio.open(next(pasta.glob("ortos2021_cog_25cm*.tif")))
    w = from_bounds(E0 - MEIO, N0 - MEIO, E0 + MEIO, N0 + MEIO, o.transform)
    a = o.read([1, 2, 3], window=w, out_shape=(3, 560, 560), resampling=rasterio.enums.Resampling.bilinear)
    Image.fromarray(np.moveaxis(a, 0, 2)).save(AQUI / "dados" / "orto.jpg", quality=90)
    print("orto", a.shape)


if __name__ == "__main__":
    main()
