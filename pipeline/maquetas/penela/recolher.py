"""Recolhe os dados abertos georreferenciados para a maqueta do Castelo de Penela
e grava-os em dados/fontes.json, para que a maqueta se possa reconstruir sem rede.

Fontes:
- OpenStreetMap (ODbL), lido através do Overture Maps (release fixada abaixo):
  traçado das muralhas (barrier=city_wall), torreões (defensive), portas,
  Igreja de São Miguel e torre sineira, edifícios da vila e caminhos.
- Copernicus DEM GLO-30 (© DLR e Airbus, distribuído pela ESA): modelo digital
  de superfície de 30 m. É o único MDT acessível neste ambiente; o LiDAR da DGT
  (MDT de 50 cm) deve substituí-lo quando estiver disponível.

Uso (Python com pyarrow, shapely e rasterio):
    python pipeline/maquetas/penela/recolher.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pfs
import rasterio
import shapely
from rasterio.windows import from_bounds

AQUI = Path(__file__).resolve().parent
RELEASE = "2026-10-21.0"
# Coordenadas oficiais indicadas para o monumento (origem da maqueta).
LAT0, LON0 = 40.031162, -8.389862
RAIO = 260  # m à volta da origem que se guardam (a maqueta usa menos)

KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574


def caixa():
    dlon, dlat = RAIO / KX, RAIO / KY
    return LON0 - dlon, LAT0 - dlat, LON0 + dlon, LAT0 + dlat


def overture(tema: str, tipo: str):
    fs = pfs.S3FileSystem(anonymous=True, region="us-west-2")
    d = ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/theme={tema}/type={tipo}/", filesystem=fs, format="parquet")
    w, s, e, n = caixa()
    f = (pc.field("bbox", "xmin") < e) & (pc.field("bbox", "xmax") > w) & (pc.field("bbox", "ymin") < n) & (pc.field("bbox", "ymax") > s)
    cols = [c for c in ("id", "geometry", "class", "subtype", "names", "height", "num_floors") if c in d.schema.names]
    out = []
    for r in d.to_table(filter=f, columns=cols).to_pylist():
        g = shapely.from_wkb(r["geometry"])
        if g.distance(shapely.Point(LON0, LAT0)) * KX > RAIO:
            continue
        out.append({
            "id": r["id"],
            "classe": r.get("class"),
            "subtipo": r.get("subtype"),
            "nome": (r.get("names") or {}).get("primary"),
            "altura": r.get("height"),
            "pisos": r.get("num_floors"),
            "geometria": shapely.geometry.mapping(shapely.set_precision(g, 1e-7)),
        })
    return out


def mdt():
    nome = "Copernicus_DSM_COG_10_N40_00_W009_00_DEM"
    with rasterio.open(f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{nome}/{nome}.tif") as src:
        w, s, e, n = caixa()
        janela = from_bounds(w - 0.001, s - 0.001, e + 0.001, n + 0.001, src.transform)
        a = src.read(1, window=janela)
        t = src.window_transform(janela)
    return {
        "fonte": f"Copernicus DEM GLO-30 ({nome})",
        "lon0": t.c, "lat0": t.f, "dlon": t.a, "dlat": t.e,  # canto NO e passo do píxel
        "alturas": np.round(a, 2).tolist(),
    }


def main():
    infra = [f for f in overture("base", "infrastructure") if f["subtipo"] in ("barrier", "tower")]
    dados = {
        "origem": {"lat": LAT0, "lon": LON0, "nota": "Coordenadas oficiais do monumento; x = este, y = norte, em metros"},
        "overture": RELEASE,
        "infraestrutura": infra,
        "edificios": overture("buildings", "building"),
        "caminhos": overture("transportation", "segment"),
        "mdt": mdt(),
    }
    (AQUI / "dados").mkdir(exist_ok=True)
    (AQUI / "dados" / "fontes.json").write_text(json.dumps(dados, ensure_ascii=False, separators=(",", ":")))
    print({k: len(v) for k, v in dados.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
