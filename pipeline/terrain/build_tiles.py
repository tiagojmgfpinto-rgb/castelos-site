"""Gera os mosaicos de relevo (quadtree) e as linhas cartográficas de cada território.

Entrada: mosaico WGS84 criado por fetch_dem.py e Natural Earth 10m (GeoJSON).
Saída:   public/terrain/<territorio>/<z>/<x>_<y>.png + manifest.json
         public/data/<territorio>/linhas.json

Codificação de cada píxel PNG (RGB, sem perdas):
    R*256 + G = (altitude_m + 1000) * 10      -> precisão de 0,1 m
    B         = classe (0 mar, 1 Portugal, 2 Espanha)

Cada mosaico tem 257x257 amostras: a última linha e coluna repetem a primeira
do mosaico vizinho, para que as malhas encaixem sem fendas.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from shapely.geometry import shape, mapping, LineString, MultiLineString, box
from shapely.ops import transform as shp_transform, unary_union, linemerge

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("CASTELOS_CACHE", ROOT / ".cache"))
NE = Path(os.environ.get("NATURAL_EARTH_DIR", CACHE / "naturalearth"))
TILE = 256

TERRITORIES = {
    # crs, canto superior esquerdo (m), lado do quadrado (m), nível máximo,
    # nível a partir do qual só se gera onde há Portugal
    "continente": dict(crs="EPSG:3763", x0=-260000, y0=300000, size=614400, max_level=5, pt_only_from=4,
                       dem="dem_continente_3s.tif", name="Portugal Continental"),
    "acores": dict(crs="EPSG:5015", x0=100000, y0=4560000, size=614400, max_level=6, pt_only_from=0,
                   dem="dem_acores_3s.tif", name="Açores"),
    "madeira": dict(crs="EPSG:5016", x0=260000, y0=3700000, size=153600, max_level=4, pt_only_from=0,
                    dem="dem_madeira_3s.tif", name="Madeira"),
}


def load_ne(name):
    with open(NE / f"{name}.geojson") as f:
        return json.load(f)["features"]


def projector(crs):
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return lambda geom: shp_transform(lambda x, y, z=None: t.transform(x, y), geom)


def countries(crs):
    proj = projector(crs)
    out = {}
    for f in load_ne("ne_10m_admin_0_countries"):
        a3 = f["properties"]["ADM0_A3"]
        if a3 in ("PRT", "ESP"):
            out[a3] = proj(shape(f["geometry"]))
    return out


def grid_transform(cfg, level):
    res = cfg["size"] / (TILE * 2**level)
    # amostras nos vértices: o centro do píxel (0,0) fica exatamente em (x0, y0)
    return res, rasterio.transform.from_origin(cfg["x0"] - res / 2, cfg["y0"] + res / 2, res, res)


def build_level(cfg, level, src, cmap):
    res, tr = grid_transform(cfg, level)
    n = TILE * 2**level + 1
    h = np.zeros((n, n), dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1), destination=h, dst_transform=tr, dst_crs=cfg["crs"],
        resampling=Resampling.average if res > 90 else Resampling.bilinear, dst_nodata=0,
    )
    cls = np.zeros((n, n), dtype=np.uint8)
    shapes = []
    if "ESP" in cmap:
        shapes.append((mapping(cmap["ESP"]), 2))
    shapes.append((mapping(cmap["PRT"]), 1))
    cls = rasterize(shapes, out_shape=(n, n), transform=tr, fill=0, dtype="uint8", all_touched=False)
    # Terra que o Natural Earth não apanha (ilhéus, costa recortada) fica com a classe vizinha mais comum.
    land_missing = (cls == 0) & (h > 2)
    cls[land_missing] = 1 if "ESP" not in cmap else 0
    h[cls == 0] = np.minimum(h[cls == 0], 0)
    return res, h, cls


def encode(h, cls):
    v = np.clip(np.round((h + 1000.0) * 10.0), 0, 65535).astype(np.uint16)
    rgb = np.stack([(v >> 8).astype(np.uint8), (v & 255).astype(np.uint8), cls], axis=-1)
    return Image.fromarray(rgb, "RGB")


def build_tiles(key, cfg):
    out_dir = ROOT / "public" / "terrain" / key
    cmap = countries(cfg["crs"])
    if key != "continente":
        cmap.pop("ESP", None)
    manifest = dict(
        id=key, name=cfg["name"], crs=cfg["crs"], x0=cfg["x0"], y0=cfg["y0"], size=cfg["size"],
        tileSize=TILE, maxLevel=cfg["max_level"], encoding="rg16_dm_plus1000_b_class", levels={},
    )
    hmin, hmax = 1e9, -1e9
    with rasterio.open(CACHE / cfg["dem"]) as src:
        for level in range(cfg["max_level"] + 1):
            res, h, cls = build_level(cfg, level, src, cmap)
            tiles = []
            for ty in range(2**level):
                for tx in range(2**level):
                    sl = (slice(ty * TILE, ty * TILE + TILE + 1), slice(tx * TILE, tx * TILE + TILE + 1))
                    c = cls[sl]
                    if not c.any():
                        continue
                    if level >= cfg["pt_only_from"] and not (c == 1).any():
                        continue
                    hh = h[sl]
                    hmin, hmax = min(hmin, float(hh.min())), max(hmax, float(hh.max()))
                    d = out_dir / str(level)
                    d.mkdir(parents=True, exist_ok=True)
                    encode(hh, c).save(d / f"{tx}_{ty}.png", optimize=True)
                    tiles.append([tx, ty])
            manifest["levels"][level] = tiles
            print(f"{key} nível {level}: {len(tiles)} mosaicos, {res:.1f} m/amostra", flush=True)
    manifest["heightRange"] = [round(hmin, 1), round(hmax, 1)]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    return cmap


def sample_heights(key, cfg, coords):
    """Altitude (m) em cada vértice, lida do nível mais fino gerado."""
    with rasterio.open(CACHE / cfg["dem"]) as src:
        t = Transformer.from_crs(cfg["crs"], "EPSG:4326", always_xy=True)
        lon, lat = t.transform(np.array([c[0] for c in coords]), np.array([c[1] for c in coords]))
        vals = np.array([v[0] for v in src.sample(list(zip(lon, lat)))], dtype=np.float32)
    return np.maximum(vals, 0)


def lines_out(geom, key, cfg, simplify):
    if geom.is_empty:
        return []
    geom = geom.simplify(simplify)
    if isinstance(geom, LineString):
        parts = [geom]
    elif hasattr(geom, "geoms"):
        parts = [g for g in geom.geoms if isinstance(g, LineString)]
    else:
        parts = []
    out = []
    for p in parts:
        cs = list(p.coords)
        if len(cs) < 2:
            continue
        zs = sample_heights(key, cfg, cs)
        out.append([[round(x), round(y), round(float(z))] for (x, y), z in zip(cs, zs)])
    return out


def as_lines(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type in ("LineString", "MultiLineString"):
        return linemerge(geom) if geom.geom_type == "MultiLineString" else geom
    if geom.geom_type == "GeometryCollection":
        ls = [g for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString")]
        return linemerge(unary_union(ls)) if ls else LineString()
    return geom.boundary


def build_lines(key, cfg, cmap):
    proj = projector(cfg["crs"])
    frame = box(cfg["x0"], cfg["y0"] - cfg["size"], cfg["x0"] + cfg["size"], cfg["y0"])
    pt = cmap["PRT"].intersection(frame)
    data = {}
    if "ESP" in cmap:
        es = cmap["ESP"].intersection(frame)
        border = as_lines(pt.boundary.intersection(es.buffer(800)))
        coast = as_lines(pt.boundary.difference(es.buffer(800)))
        data["fronteira"] = lines_out(border, key, cfg, 150)
        es_coast = es.boundary.difference(pt.buffer(800)).difference(frame.exterior.buffer(500))
        data["costaEspanha"] = lines_out(as_lines(es_coast), key, cfg, 300)
    else:
        coast = pt.boundary
    data["costa"] = lines_out(coast, key, cfg, 100)

    # Distritos (limites internos) e regiões autónomas
    dists = []
    for f in load_ne("ne_10m_admin_1_states_provinces"):
        p = f["properties"]
        if p["adm0_a3"] != "PRT":
            continue
        g = proj(shape(f["geometry"])).intersection(frame)
        if g.is_empty:
            continue
        dists.append((p["name"].replace(" District", "").replace(" Autonomous region", ""), g))
    inner = unary_union([g.boundary for _, g in dists]).difference(pt.boundary.buffer(600))
    data["distritos"] = lines_out(as_lines(inner), key, cfg, 150)
    tinv = Transformer.from_crs(cfg["crs"], "EPSG:4326", always_xy=True)
    data["nomesDistritos"] = []
    for name, g in dists:
        c = g.representative_point()
        data["nomesDistritos"].append({"nome": name, "x": round(c.x), "y": round(c.y)})

    # Rios
    rivers = []
    for src_name in ("ne_10m_rivers_lake_centerlines", "ne_10m_rivers_europe"):
        for f in load_ne(src_name):
            g = proj(shape(f["geometry"]))
            g = g.intersection(frame)
            if g.is_empty or not g.intersects(pt.buffer(30000)):
                continue
            nm = f["properties"].get("name") or ""
            for line in lines_out(as_lines(g), key, cfg, 120):
                rivers.append({"nome": nm, "pontos": line})
    data["rios"] = rivers

    d = ROOT / "public" / "data" / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "linhas.json").write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(f"{key}: linhas gravadas ({sum(len(v) for v in data.values() if isinstance(v, list))} elementos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("territorios", nargs="*", default=list(TERRITORIES))
    args = ap.parse_args()
    for key in args.territorios:
        cfg = TERRITORIES[key]
        cmap = build_tiles(key, cfg)
        build_lines(key, cfg, cmap)


if __name__ == "__main__":
    main()
