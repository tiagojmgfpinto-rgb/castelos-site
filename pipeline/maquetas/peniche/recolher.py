"""Recolhe os dados abertos georreferenciados para a maqueta da Fortaleza de Peniche
e grava-os em dados/fontes.json, para que a maqueta se possa reconstruir sem rede.

Fontes:
- OpenStreetMap (ODbL), lido através do Overture Maps (release fixada abaixo):
  muralhas da praça (barrier=city_wall), portas (barrier=sally_port/gate), casario, igrejas e ruas.
O relevo, as cotas e as alturas vêm do LiDAR da DGT, preparado à parte pelo lidar.py.

Uso (Python com pyarrow e shapely):
    python pipeline/maquetas/peniche/recolher.py
"""
import json
import math
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pfs
import shapely

AQUI = Path(__file__).resolve().parent
RELEASE = "2026-10-21.0"
# Origem da maqueta: centróide do polígono das fortificações no OSM (praça e Coroada).
LAT0, LON0 = 39.3539, -9.3814
RAIO = 600  # m à volta da origem que se guardam (a maqueta usa menos)

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


def main():
    infra = [f for f in overture("base", "infrastructure") if f["subtipo"] in ("barrier", "tower")]
    dados = {
        "origem": {"lat": LAT0, "lon": LON0, "nota": "Centróide do polígono das fortificações; x = este, y = norte, em metros"},
        "overture": RELEASE,
        "infraestrutura": infra,
        "edificios": overture("buildings", "building"),
        "caminhos": overture("transportation", "segment"),
        "usos": overture("base", "land_use"),
    }
    (AQUI / "dados").mkdir(exist_ok=True)
    (AQUI / "dados" / "fontes.json").write_text(json.dumps(dados, ensure_ascii=False, separators=(",", ":")))
    print({k: len(v) for k, v in dados.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
