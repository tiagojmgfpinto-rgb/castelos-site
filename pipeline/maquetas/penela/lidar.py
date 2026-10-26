"""Reamostra o LiDAR da DGT (MDT e MDS de 50 cm, PT-TM06/ETRS89) para a grelha local da maqueta
e grava dados/lidar.npz, que o construir.py lê sem precisar de GDAL nem pyproj.

As folhas descarregam-se no Centro de Dados da DGT (cdd.dgterritorio.gov.pt):
- "LiDAR - Modelos Digitais do Relevo", MDT-50cm e MDS-50cm (folhas 177341 e 178341, voo de 2024);
- "LiDAR - Nuvem de pontos" (LO-177341 e LO-178341, 2024), de onde se tira a classe de cada célula
  (solo, vegetação, edifício), para separar árvores de casas e muralhas sem adivinhar;
- ORTOS 2025 (25 cm, RGB+IV), recortada para dados/orto.jpg: serve para a verificação em planta.

Uso:
    python pipeline/maquetas/penela/lidar.py PASTA_COM_AS_FOLHAS
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.merge import merge

AQUI = Path(__file__).resolve().parent
CFG = json.loads((AQUI.parents[2] / "src" / "data" / "maquetas" / "penela.json").read_text())
LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([-40.0, 0.0])  # igual ao construir.py
MEIO = 160.0  # metros para cada lado do centro
PASSO = 0.5


def mosaico(ficheiros):
    fontes = [rasterio.open(f) for f in ficheiros]
    a, t = merge(fontes, nodata=-999.0)
    return a[0].astype(np.float64), t


def amostrar(a, t, ex, ny):
    """Interpolação bilinear num raster (transformação afim sem rotação)."""
    c = (ex - t.c) / t.a - 0.5
    r = (ny - t.f) / t.e - 0.5
    c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
    fc, fr = c - c0, r - r0
    c0 = np.clip(c0, 0, a.shape[1] - 2)
    r0 = np.clip(r0, 0, a.shape[0] - 2)
    v = (a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c0 + 1] * fc * (1 - fr)
         + a[r0 + 1, c0] * (1 - fc) * fr + a[r0 + 1, c0 + 1] * fc * fr)
    return v


def classes(pasta, tf, n):
    """Classe ASPRS do ponto mais alto de cada célula: 2 solo, 3-5 vegetação, 6 edifício, 0 sem pontos."""
    import laspy
    inv = Transformer.from_crs(3763, 4326, always_xy=True)
    fs = sorted(pasta.glob("LO-*.laz"))
    if not fs:
        return None
    # caixa da grelha em PT-TM06, com folga
    cx = LON0 + np.array([-MEIO, MEIO]) / KX + CENTRO[0] / KX
    cy = LAT0 + np.array([-MEIO, MEIO]) / KY + CENTRO[1] / KY
    ex, ny = tf.transform(*np.meshgrid(cx, cy))
    w, e, s, nn = ex.min() - 5, ex.max() + 5, ny.min() - 5, ny.max() + 5
    alto = np.full(n * n, -1e9)
    cls = np.zeros(n * n, np.uint8)
    for f in fs:
        with laspy.open(f) as r:
            for ch in r.chunk_iterator(2_000_000):
                x, y = np.asarray(ch.x), np.asarray(ch.y)
                m = (x > w) & (x < e) & (y > s) & (y < nn)
                c = np.asarray(ch.classification)[m]
                m2 = (c >= 2) & (c <= 6)
                if not m2.any():
                    continue
                lon, lat = inv.transform(x[m][m2], y[m][m2])
                z, c = np.asarray(ch.z)[m][m2], c[m2]
                gx = np.round(((lon - LON0) * KX - CENTRO[0] + MEIO) / PASSO).astype(int)
                gy = np.round(((lat - LAT0) * KY - CENTRO[1] + MEIO) / PASSO).astype(int)
                ok = (gx >= 0) & (gx < n) & (gy >= 0) & (gy < n)
                i = gy[ok] * n + gx[ok]
                z, c = z[ok], c[ok]
                o = np.argsort(z)  # o último a escrever em cada célula é o mais alto
                i, z, c = i[o], z[o], c[o]
                sobe = z > alto[i]
                np.maximum.at(alto, i, z)
                cls[i[sobe]] = c[sobe]  # com repetidos, fica o último (o mais alto do bloco)
    print("classes", {int(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))})
    return cls.reshape(n, n)


def arvores(mdt, mds, cls, xs):
    """Copas individuais: máximos locais da altura da vegetação (MDS - MDT, pontos de classe 3 a 5)
    e raio da copa pela área das células de vegetação mais próximas de cada máximo."""
    from scipy import ndimage as ndi
    h = mds - mdt
    veg = (cls >= 3) & (cls <= 5) & (h > 2.0)
    veg = ndi.binary_opening(veg, iterations=1)
    hs = ndi.gaussian_filter(np.where(veg, h, 0.0), 1.2)
    topo = (hs == ndi.maximum_filter(hs, size=7)) & veg & (hs > 2.5)
    marc, k = ndi.label(topo)
    if k == 0:
        return np.zeros((0, 5), np.float32)
    # cada célula de vegetação vai para o máximo mais próximo (até 6 m)
    d, (ri, ci) = ndi.distance_transform_edt(marc == 0, return_indices=True)
    lab = np.where(veg & (d * PASSO < 6.0), marc[ri, ci], 0)
    area = ndi.sum(np.ones_like(h), lab, index=np.arange(1, k + 1)) * PASSO * PASSO
    pos = np.array(ndi.center_of_mass(topo, marc, np.arange(1, k + 1)))
    r_, c_ = np.round(pos).astype(int).T
    out = np.stack([xs[c_], xs[r_], mds[r_, c_], mdt[r_, c_], np.sqrt(area / np.pi)], 1)
    out = out[out[:, 4] >= 0.8]
    print("árvores", len(out), "altura mediana", round(float(np.median(out[:, 2] - out[:, 3])), 1),
          "raio mediano", round(float(np.median(out[:, 4])), 1))
    return out.astype(np.float32)


def ortofoto(pasta, tf):
    """Recorte RGB da ortofoto a 25 cm na mesma área, norte para cima, gravado em dados/orto.jpg."""
    from PIL import Image
    fs = sorted(pasta.glob("ortos*.tif"))
    if not fs:
        return
    k = int(2 * MEIO / 0.25)
    xs = np.linspace(-MEIO, MEIO, k)
    X, Y = np.meshgrid(xs, xs[::-1])  # linha 0 = norte
    ex, ny = tf.transform(LON0 + (X + CENTRO[0]) / KX, LAT0 + (Y + CENTRO[1]) / KY)
    with rasterio.open(fs[0]) as src:
        from rasterio.windows import from_bounds
        jan = from_bounds(ex.min() - 2, ny.min() - 2, ex.max() + 2, ny.max() + 2, src.transform).round_offsets().round_lengths()
        a = src.read((1, 2, 3), window=jan).astype(np.float64)
        t = src.window_transform(jan)
    img = np.stack([amostrar(a[b], t, ex, ny) for b in range(3)], -1)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(AQUI / "dados" / "orto.jpg", quality=88)
    print("ortofoto", fs[0].name, img.shape)


def main():
    pasta = Path(sys.argv[1])
    tf = Transformer.from_crs(4326, 3763, always_xy=True)
    n = int(2 * MEIO / PASSO) + 1
    xs = np.linspace(-MEIO, MEIO, n)
    X, Y = np.meshgrid(xs, xs)  # coordenadas da cena (y para norte; linha 0 = sul)
    lon = LON0 + (X + CENTRO[0]) / KX
    lat = LAT0 + (Y + CENTRO[1]) / KY
    ex, ny = tf.transform(lon, lat)
    saida = {}
    for prod in ("MDT", "MDS"):
        fs = sorted(pasta.glob(f"*{prod}-50cm-*.tif"))
        if not fs:
            raise SystemExit(f"faltam folhas {prod}-50cm em {pasta}")
        a, t = mosaico(fs)
        if (a < -900).any():
            a[a < -900] = np.nan
        v = amostrar(a, t, ex, ny)
        if np.isnan(v).any():
            raise SystemExit(f"{prod}: a área da maqueta não está toda coberta pelas folhas")
        saida[prod.lower()] = np.round((v - 200.0) * 100).astype(np.int16)  # centímetros acima de 200 m
        print(prod, [f.name[-30:] for f in fs], round(float(v.min()), 2), round(float(v.max()), 2))
    c = classes(pasta, tf, n)
    if c is not None:
        saida["classe"] = c
        # x, y (cena), topo da copa, chão (cotas absolutas) e raio da copa
        saida["arvores"] = arvores(saida["mdt"] / 100 + 200, saida["mds"] / 100 + 200, c, xs)
    ortofoto(pasta, tf)
    np.savez_compressed(AQUI / "dados" / "lidar.npz", passo=PASSO, meio=MEIO, base=200.0, **saida,
                        fonte="DGT, LiDAR 2024 (MDT, MDS 50 cm e nuvem de pontos), folhas 177341 e 178341")


if __name__ == "__main__":
    main()
