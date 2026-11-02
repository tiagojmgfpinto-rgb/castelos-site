"""Recolhe do OpenStreetMap (via Overture Maps) as plantas dos edifícios e as ruas à volta da Torre de
Menagem de Braga e grava-as em dados/fontes.json, para que a maqueta se reconstrua sem rede.

Uso (Python com pyarrow e shapely):
    python pipeline/maquetas/braga/recolher.py
"""
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pfs
import shapely
from pyproj import Transformer

AQUI = Path(__file__).resolve().parent
RELEASE = "2026-10-21.0"
LAT0, LON0 = 41.551322, -8.423892  # centro da torre (LiDAR)
RAIO = 120  # m

TF = Transformer.from_crs(4326, 3763, always_xy=True)
E0, N0 = TF.transform(LON0, LAT0)


def overture(tema, tipo):
    fs = pfs.S3FileSystem(anonymous=True, region="us-west-2")
    d = ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/theme={tema}/type={tipo}/", filesystem=fs, format="parquet")
    dl, dt = RAIO / 83000, RAIO / 111000
    w, e, s, n = LON0 - dl, LON0 + dl, LAT0 - dt, LAT0 + dt
    f = (pc.field("bbox", "xmin") < e) & (pc.field("bbox", "xmax") > w) & (pc.field("bbox", "ymin") < n) & (pc.field("bbox", "ymax") > s)
    cols = [c for c in ("id", "geometry", "class", "subtype", "names", "height", "num_floors") if c in d.schema.names]
    out = []
    for r in d.to_table(filter=f, columns=cols).to_pylist():
        g = shapely.from_wkb(r["geometry"])
        g = shapely.transform(g, lambda a: (lambda x, y: __import__("numpy").stack([x - E0, y - N0], 1))(*TF.transform(a[:, 0], a[:, 1])))
        if g.distance(shapely.Point(0, 0)) > RAIO:
            continue
        out.append({"id": r["id"], "classe": r.get("class"), "subtipo": r.get("subtype"),
                    "nome": (r.get("names") or {}).get("primary"), "altura": r.get("height"),
                    "geometria": shapely.geometry.mapping(shapely.set_precision(g, 0.01))})
    return out


def main():
    dados = {"origem": {"lat": LAT0, "lon": LON0, "nota": "centro da torre; x = este, y = norte (PT-TM06), em metros"},
             "overture": RELEASE, "edificios": overture("buildings", "building"),
             "caminhos": overture("transportation", "segment")}
    (AQUI / "dados").mkdir(exist_ok=True)
    (AQUI / "dados" / "fontes.json").write_text(json.dumps(dados, ensure_ascii=False, separators=(",", ":")))
    print({k: len(v) for k, v in dados.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
