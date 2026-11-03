"""Recolhe do OpenStreetMap (via Overture Maps) as plantas dos edifícios, as ruas e as marcas de muralha à volta
do Castelo de Lanhoso e do Santuário do Pilar e grava-as em dados/fontes.json (coordenadas locais em metros), para que a maqueta se
reconstrua sem rede.

Uso (Python com pyarrow, shapely e pyproj):
    python pipeline/maquetas/lanhoso/recolher.py
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pfs
import shapely
from pyproj import Transformer

AQUI = Path(__file__).resolve().parent
RELEASE = "2026-10-21.0"
import sys

sys.path.insert(0, str(AQUI))
from comum import E0, LAT0, LON0, N0  # noqa: E402

RAIO = 160  # m
TF = Transformer.from_crs(4326, 3763, always_xy=True)


def local(a):
    x, y = TF.transform(a[:, 0], a[:, 1])
    return np.stack([x - E0, y - N0], 1)


def overture(tema, tipo, filtro=None):
    fs = pfs.S3FileSystem(anonymous=True, region="us-west-2")
    d = ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/theme={tema}/type={tipo}/", filesystem=fs, format="parquet")
    dl, dt = RAIO / 82000, RAIO / 111000
    w, e, s, n = LON0 - dl, LON0 + dl, LAT0 - dt, LAT0 + dt
    f = (pc.field("bbox", "xmin") < e) & (pc.field("bbox", "xmax") > w) & (pc.field("bbox", "ymin") < n) & (pc.field("bbox", "ymax") > s)
    cols = [c for c in ("id", "geometry", "class", "subtype", "names", "height", "num_floors") if c in d.schema.names]
    out = []
    for r in d.to_table(filter=f, columns=cols).to_pylist():
        if filtro and not filtro(r):
            continue
        g = shapely.transform(shapely.from_wkb(r["geometry"]), local)
        if g.distance(shapely.Point(0, 0)) > RAIO:
            continue
        out.append({"id": r["id"], "classe": r.get("class"), "subtipo": r.get("subtype"),
                    "nome": (r.get("names") or {}).get("primary"), "altura": r.get("height"),
                    "geometria": shapely.geometry.mapping(shapely.set_precision(g, 0.01))})
    return out


def main():
    dados = {"origem": {"lat": LAT0, "lon": LON0, "nota": "centro da torre; x = este, y = norte (PT-TM06), em metros"},
             "overture": RELEASE, "edificios": overture("buildings", "building"),
             "caminhos": overture("transportation", "segment"),
             "infraestrutura": overture("base", "infrastructure", lambda r: r.get("subtype") in ("barrier", "tower", "utility", "storage", "agricultural") or r.get("class") in ("city_wall", "silo", "granary", "storage_tank"))}
    (AQUI / "dados").mkdir(exist_ok=True)
    (AQUI / "dados" / "fontes.json").write_text(json.dumps(dados, ensure_ascii=False, separators=(",", ":")))
    print({k: len(v) for k, v in dados.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
