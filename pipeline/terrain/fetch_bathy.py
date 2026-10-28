"""Descarrega batimetria (e relevo de enquadramento) dos Terrain Tiles da Mapzen
(AWS Open Data, formato terrarium; no mar provêm do ETOPO1 / GEBCO) e grava um mosaico
Web Mercator em cache. Só é usado onde não há terra: o relevo terrestre vem do Copernicus.

Uso:
    python pipeline/terrain/fetch_bathy.py --region continente --zoom 9
    python pipeline/terrain/fetch_bathy.py --region continente-moldura --zoom 8
"""
import argparse
import concurrent.futures as cf
import io
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
CACHE = Path(os.environ.get("CASTELOS_CACHE", Path(__file__).resolve().parents[2] / ".cache"))
REGIONS = {
    # lon_min, lat_min, lon_max, lat_max (com margem à volta do quadrado de mosaicos)
    "continente": (-12.5, 35.8, -3.0, 43.2),
    # moldura larga, de menor resolução: quase toda a Península, o Atlântico e as pontas de França e de Marrocos
    "continente-moldura": (-16.0, 33.0, 1.6, 46.6),
    "acores": (-32.5, 35.5, -23.5, 41.0),
    "madeira": (-17.7, 32.0, -15.8, 33.6),
}
R = 6378137.0


def tile_xy(lon, lat, z):
    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def fetch(z, x, y):
    with urllib.request.urlopen(URL.format(z=z, x=x, y=y), timeout=60) as r:
        a = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB")).astype(np.float32)
    return x, y, a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="continente", choices=REGIONS)
    ap.add_argument("--zoom", type=int, default=9)
    args = ap.parse_args()
    lon0, lat0, lon1, lat1 = REGIONS[args.region]
    z = args.zoom
    x0, y0 = tile_xy(lon0, lat1, z)
    x1, y1 = tile_xy(lon1, lat0, z)
    out = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), dtype=np.float32)
    jobs = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for x, y, a in ex.map(lambda j: fetch(z, *j), jobs):
            out[(y - y0) * 256 : (y - y0 + 1) * 256, (x - x0) * 256 : (x - x0 + 1) * 256] = a
    size = 2 * math.pi * R / 2**z  # metros por mosaico
    left = -math.pi * R + x0 * size
    top = math.pi * R - y0 * size
    tr = rasterio.transform.from_origin(left, top, size / 256, size / 256)
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"bathy_{args.region}_z{z}.tif"
    with rasterio.open(dst, "w", driver="GTiff", width=out.shape[1], height=out.shape[0], count=1, dtype="float32",
                       crs="EPSG:3857", transform=tr, compress="deflate", tiled=True) as f:
        f.write(out, 1)
    print(f"gravado {dst} ({len(jobs)} mosaicos, {out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
