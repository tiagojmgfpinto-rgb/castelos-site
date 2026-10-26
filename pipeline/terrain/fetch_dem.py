"""Descarrega o Copernicus DEM GLO-30 para Portugal Continental (e margem de Espanha)
e grava um mosaico WGS84 de resolução reduzida em cache.

Os ficheiros GLO-30 são COG públicos no AWS Open Data; lemos só a vista geral
(overview) necessária por pedidos HTTP parciais, sem descarregar cada ficheiro inteiro.

Uso:
    python pipeline/terrain/fetch_dem.py --region continente --arcsec 3
"""
import argparse
import concurrent.futures as cf
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

REGIONS = {
    # lat_min, lat_max, lon_min, lon_max (graus inteiros, canto SW de cada ficheiro)
    "continente": (36, 43, -10, -3),
    "acores": (36, 40, -32, -24),
    "madeira": (32, 34, -18, -16),
}

CACHE = Path(os.environ.get("CASTELOS_CACHE", Path(__file__).resolve().parents[2] / ".cache"))


def tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def read_tile(lat: int, lon: int, px: int):
    name = tile_name(lat, lon)
    url = f"/vsicurl/{BASE}/{name}/{name}.tif"
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            with rasterio.open(url) as src:
                # A largura dos ficheiros varia com a latitude; reamostramos para px x px.
                return lat, lon, src.read(1, out_shape=(px, px), resampling=Resampling.average).astype(np.float32)
    except rasterio.errors.RasterioIOError:
        return lat, lon, None  # ficheiro inexistente: só mar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="continente", choices=REGIONS)
    ap.add_argument("--arcsec", type=float, default=3.0, help="resolução do mosaico em segundos de arco")
    args = ap.parse_args()

    lat0, lat1, lon0, lon1 = REGIONS[args.region]
    px = int(round(3600 / args.arcsec))
    out = np.zeros(((lat1 - lat0) * px, (lon1 - lon0) * px), dtype=np.float32)

    jobs = [(la, lo) for la in range(lat0, lat1) for lo in range(lon0, lon1)]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for la, lo, arr in ex.map(lambda j: read_tile(*j, px), jobs):
            status = "mar" if arr is None else "ok"
            print(f"{tile_name(la, lo)}: {status}", flush=True)
            if arr is None:
                continue
            r = (lat1 - 1 - la) * px
            c = (lo - lon0) * px
            out[r : r + px, c : c + px] = arr

    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"dem_{args.region}_{args.arcsec:g}s.tif"
    res = 1.0 / px
    transform = rasterio.transform.from_origin(lon0, lat1, res, res)
    with rasterio.open(
        dst, "w", driver="GTiff", width=out.shape[1], height=out.shape[0], count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, compress="deflate", tiled=True,
    ) as f:
        f.write(out, 1)
    print(f"gravado {dst} ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
