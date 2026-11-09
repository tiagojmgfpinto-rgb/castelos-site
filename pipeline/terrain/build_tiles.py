"""Gera os mosaicos de relevo (quadtree) e as linhas cartográficas de cada território.

Entrada: mosaico WGS84 criado por fetch_dem.py (terra), batimetria de fetch_bathy.py (mar)
         e Natural Earth 10m (GeoJSON).
Saída:   public/terrain/<territorio>/<z>/<x>_<y>.png + manifest.json
         public/data/<territorio>/linhas.json

Codificação de cada píxel PNG (RGB, sem perdas):
    R*256 + G = (altitude_m + 8000) * 5       -> de -8000 a +5107 m, precisão de 0,2 m
    B         = classe (0 mar, 1 Portugal, 2 Espanha) + 8 nas manchas urbanas (Natural Earth urban areas)

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
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("CASTELOS_CACHE", ROOT / ".cache"))
NE = Path(os.environ.get("NATURAL_EARTH_DIR", CACHE / "naturalearth"))
TILE = 256

# Projeção do mapa único (como o castillos.site, que leva as Canárias no mesmo mapa da Península): cónica
# conforme de Lambert com o centro em Portugal Continental, paralelos 34° e 41°. O Continente fica como estava
# (meridiano central e origem do PT-TM06) e os Açores e a Madeira aparecem na posição real, um pouco rodados.
PT_LCC = ("+proj=lcc +lat_0=39.6682583333333 +lon_0=-8.13310833333333 +lat_1=34 +lat_2=41 "
          "+x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs")

TERRITORIES = {
    # crs, canto superior esquerdo (m), lado do quadrado (m), nível máximo,
    # nível a partir do qual só se gera onde há Portugal
    # sea_max: até este nível geram-se também os mosaicos só de mar (batimetria)
    # terra_km: entre sea_max e pt_only_from geram-se os mosaicos com terra a menos desta distância de Portugal
    # Quadrado de 3072 km, das Flores à Catalunha e das Landes às Canárias; fora do Copernicus o relevo vem das molduras.
    "portugal": dict(crs=PT_LCC, x0=-2300000, y0=700000, size=3200000, max_level=7, pt_only_from=6, sea_max=4, terra_km=450,
                     dem=["dem_continente_3s.tif", "dem_acores_3s.tif", "dem_madeira_3s.tif"],
                     bathy=["bathy_continente_z9.tif", "bathy_acores_z9.tif", "bathy_madeira_z9.tif"],
                     moldura=["bathy_continente-moldura_z8.tif", "bathy_portugal-moldura_z7.tif"],
                     name="Portugal"),
}


def load_ne(name):
    with open(NE / f"{name}.geojson") as f:
        return json.load(f)["features"]


def projector(crs):
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return lambda geom: shp_transform(lambda x, y, z=None: t.transform(x, y), geom)


def countries(crs, frame_wgs84=None):
    """Portugal ("PRT") e, juntos, os outros países que caem no quadrado ("OUT")."""
    proj = projector(crs)
    out, outros = {}, []
    for f in load_ne("ne_10m_admin_0_countries"):
        a3 = f["properties"]["ADM0_A3"]
        g = shape(f["geometry"])
        if a3 == "PRT":
            out["PRT"] = proj(g)
        elif frame_wgs84 is not None and g.intersects(frame_wgs84):
            outros.append(proj(g.intersection(frame_wgs84)))
        if a3 == "ESP":
            out["ESP"] = proj(g)
    if outros:
        out["OUT"] = unary_union(outros)
    return out


def urban(crs, frame_wgs84):
    """Manchas urbanas do Natural Earth dentro do quadrado, tingidas no relevo como no castillos.site."""
    urb = [g for g in (shape(f["geometry"]) for f in load_ne("ne_10m_urban_areas")) if g.intersects(frame_wgs84)]
    return projector(crs)(unary_union(urb).buffer(0)) if urb else None


def tile_transform(cfg, level, tx, ty):
    res = cfg["size"] / (TILE * 2**level)
    # amostras nos vértices: o centro do píxel (0,0) fica exatamente no canto do mosaico
    left = cfg["x0"] + tx * TILE * res
    top = cfg["y0"] - ty * TILE * res
    return res, rasterio.transform.from_origin(left - res / 2, top + res / 2, res, res)


def warp(src, tr, crs, res, n):
    """Reprojeta uma fonte para a grelha do mosaico; fora da fonte fica NaN."""
    a = np.full((n, n), np.nan, dtype=np.float32)
    if src is None:
        return a
    reproject(
        source=rasterio.band(src, 1), destination=a, dst_transform=tr, dst_crs=crs,
        resampling=Resampling.average if res > 90 else Resampling.bilinear, dst_nodata=np.nan, init_dest_nodata=True,
    )
    return a


def build_tile(cfg, level, tx, ty, srcs, cmap):
    res, tr = tile_transform(cfg, level, tx, ty)
    n = TILE + 1
    dems, bathys, molduras = srcs

    def primeiro(fontes):
        """Cada amostra vem da primeira fonte que a cobre."""
        a = np.full((n, n), np.nan, dtype=np.float32)
        for f in fontes:
            if f is None or not np.isnan(a).any():
                continue
            a = np.where(np.isnan(a), warp(f, tr, cfg["crs"], res, n), a)
        return a

    larga = primeiro(molduras)
    h = primeiro(dems)
    h = np.where(np.isnan(h), larga, h)
    b = primeiro(bathys)
    b = np.where(np.isnan(b), larga, b)
    h, b = np.nan_to_num(h), np.nan_to_num(b)
    shapes = []
    if "OUT" in cmap:
        shapes.append((mapping(cmap["OUT"]), 2))
    shapes.append((mapping(cmap["PRT"]), 1))
    cls = rasterize(shapes, out_shape=(n, n), transform=tr, fill=0, dtype="uint8", all_touched=False)
    # Terra que o Natural Earth não apanha (ilhéus, costa recortada) fica com a classe vizinha mais comum.
    land_missing = (cls == 0) & (h > 2)
    cls[land_missing] = 1 if "OUT" not in cmap else 0
    if "URB" in cmap:
        urb = rasterize([(mapping(cmap["URB"]), 1)], out_shape=(n, n), transform=tr, fill=0, dtype="uint8", all_touched=False)
        cls[(urb == 1) & (cls > 0)] += 8
    # no mar, a profundidade vem da batimetria
    sea = cls == 0
    h[sea] = np.minimum(np.minimum(h[sea], 0), np.minimum(b[sea], -1))
    return res, h, cls


def encode(h, cls):
    v = np.clip(np.round((h + 8000.0) * 5.0), 0, 65535).astype(np.uint16)
    rgb = np.stack([(v >> 8).astype(np.uint8), (v & 255).astype(np.uint8), cls], axis=-1)
    return Image.fromarray(rgb, "RGB")


def build_tiles(key, cfg):
    out_dir = ROOT / "public" / "terrain" / key
    frame = box(cfg["x0"], cfg["y0"] - cfg["size"], cfg["x0"] + cfg["size"], cfg["y0"])
    tinv = Transformer.from_crs(cfg["crs"], "EPSG:4326", always_xy=True)
    frame_wgs84 = shp_transform(lambda x, y, z=None: tinv.transform(x, y), frame.segmentize(5000))
    cmap = countries(cfg["crs"], frame_wgs84)
    urb = urban(cfg["crs"], frame_wgs84)
    if urb is not None:
        cmap["URB"] = urb
    pt = prep(cmap["PRT"])
    perto = cmap["PRT"].buffer(cfg.get("terra_km", 1e5) * 1000)
    terra = prep(unary_union([g for g in (cmap.get("PRT"), cmap.get("OUT")) if g is not None]).intersection(perto))
    manifest = dict(
        id=key, name=cfg["name"], crs=cfg["crs"], x0=cfg["x0"], y0=cfg["y0"], size=cfg["size"],
        tileSize=TILE, maxLevel=cfg["max_level"], encoding="rg16_5x_plus8000_b_class", levels={},
    )
    hmin, hmax = 1e9, -1e9
    abrir = lambda nome: rasterio.open(CACHE / nome) if nome and (CACHE / nome).exists() else None
    srcs = [[abrir(f) for f in cfg["dem"]], [abrir(f) for f in cfg["bathy"]], [abrir(f) for f in cfg.get("moldura", [])]]
    try:
        for level in range(cfg["max_level"] + 1):
            tiles = []
            lado = cfg["size"] / 2**level
            for ty in range(2**level):
                for tx in range(2**level):
                    caixa = box(cfg["x0"] + tx * lado, cfg["y0"] - (ty + 1) * lado, cfg["x0"] + (tx + 1) * lado, cfg["y0"] - ty * lado)
                    # nos níveis finos só se gera onde há terra, e a partir de pt_only_from só onde há Portugal
                    if level > cfg["sea_max"] and not terra.intersects(caixa):
                        continue
                    if level > cfg["sea_max"] and level >= cfg["pt_only_from"] and not pt.intersects(caixa):
                        continue
                    res, hh, c = build_tile(cfg, level, tx, ty, srcs, cmap)
                    if level > cfg["sea_max"] and level >= cfg["pt_only_from"] and not (c == 1).any():
                        continue
                    hmin, hmax = min(hmin, float(hh.min())), max(hmax, float(hh.max()))
                    d = out_dir / str(level)
                    d.mkdir(parents=True, exist_ok=True)
                    encode(hh, c).save(d / f"{tx}_{ty}.png", optimize=True)
                    tiles.append([tx, ty])
            manifest["levels"][level] = tiles
            print(f"{key} nível {level}: {len(tiles)} mosaicos, {cfg['size'] / (TILE * 2**level):.1f} m/amostra", flush=True)
    finally:
        for s_ in (x for grupo in srcs for x in grupo):
            if s_ is not None:
                s_.close()
    manifest["heightRange"] = [round(hmin, 1), round(hmax, 1)]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    return cmap


def sample_heights(key, cfg, coords):
    """Altitude (m) em cada vértice, lida do nível mais fino gerado."""
    t = Transformer.from_crs(cfg["crs"], "EPSG:4326", always_xy=True)
    lon, lat = t.transform(np.array([c[0] for c in coords]), np.array([c[1] for c in coords]))
    vals = np.zeros(len(coords), dtype=np.float32)
    for nome in cfg["dem"]:
        if not (CACHE / nome).exists():
            continue
        with rasterio.open(CACHE / nome) as src:
            v = np.array([x[0] for x in src.sample(list(zip(lon, lat)))], dtype=np.float32)
        vals = np.maximum(vals, np.nan_to_num(v, nan=0.0))
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
    if "OUT" in cmap:
        es = cmap["ESP"].intersection(frame)
        border = as_lines(pt.boundary.intersection(es.buffer(800)))
        coast = as_lines(pt.boundary.difference(es.buffer(800)))
        data["fronteira"] = lines_out(border, key, cfg, 150)
        # costas e fronteiras dos outros países, mais finas
        outros = cmap["OUT"].intersection(frame).intersection(cmap["PRT"].buffer(600000))
        es_coast = outros.boundary.difference(pt.buffer(800)).difference(frame.exterior.buffer(500))
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
        nome = p["name"].replace(" District", "").replace(" Autonomous region", "")
        dists.append(({"Azores": "Açores"}.get(nome, nome), g))
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
