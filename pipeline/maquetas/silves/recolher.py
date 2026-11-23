"""Recolhe do OpenStreetMap (via Overture Maps) as plantas dos edifícios da vila e os troços de muralha
(«city_wall») à volta do Castelo de Silves e grava-os em dados/fontes.json, para a maqueta se reconstruir sem rede.

Uso (Python com pyarrow e shapely):
    python pipeline/maquetas/silves/recolher.py
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pfs
import shapely
from pyproj import Transformer

from comum import E0, N0, LAT0, LON0, CENTRO_BASE, RAIO_BASE

AQUI = Path(__file__).resolve().parent
RELEASE = "2026-10-21.0"
TF = Transformer.from_crs(4326, 3763, always_xy=True)


def _local(a):
    x, y = TF.transform(a[:, 0], a[:, 1])
    return np.stack([np.asarray(x) - E0, np.asarray(y) - N0], 1)


def overture(tema, tipo, filtro=None):
    fs = pfs.S3FileSystem(anonymous=True, region="us-west-2")
    d = ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/theme={tema}/type={tipo}/", filesystem=fs, format="parquet")
    lon_c, lat_c = Transformer.from_crs(3763, 4326, always_xy=True).transform(E0 + CENTRO_BASE[0], N0 + CENTRO_BASE[1])
    dl, dt = RAIO_BASE / 83000, RAIO_BASE / 111000
    w, e, s, n = lon_c - dl, lon_c + dl, lat_c - dt, lat_c + dt
    f = (pc.field("bbox", "xmin") < e) & (pc.field("bbox", "xmax") > w) & (pc.field("bbox", "ymin") < n) & (pc.field("bbox", "ymax") > s)
    cols = [c for c in ("id", "geometry", "class", "subtype", "names", "height", "num_floors") if c in d.schema.names]
    out = []
    centro = shapely.Point(*CENTRO_BASE)
    for r in d.to_table(filter=f, columns=cols).to_pylist():
        if filtro and not filtro(r):
            continue
        g = shapely.transform(shapely.from_wkb(r["geometry"]), _local)
        if g.distance(centro) > RAIO_BASE:
            continue
        out.append({"id": r["id"], "classe": r.get("class"), "subtipo": r.get("subtype"),
                    "nome": (r.get("names") or {}).get("primary"), "altura": r.get("height"),
                    "geometria": shapely.geometry.mapping(shapely.set_precision(g, 0.01))})
    return out


def main():
    dados = {"origem": {"lat": LAT0, "lon": LON0, "nota": "centro da alcáçova; x = este, y = norte (PT-TM06), em metros"},
             "overture": RELEASE, "edificios": overture("buildings", "building"),
             "muralhas": overture("base", "infrastructure", lambda r: r.get("class") in ("city_wall", "wall", "gate", "tower"))}
    (AQUI / "dados").mkdir(exist_ok=True)
    (AQUI / "dados" / "fontes.json").write_text(json.dumps(dados, ensure_ascii=False, separators=(",", ":")))
    print({k: len(v) for k, v in dados.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
