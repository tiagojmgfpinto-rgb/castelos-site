"""Castro Laboreiro: reamostra o LiDAR da DGT (MDT e MDS de 50 cm, PT-TM06/ETRS89) para a grelha local da maqueta
e grava dados/lidar.npz, que o construir.py lê sem precisar de GDAL nem pyproj.

As folhas descarregam-se no Centro de Dados da DGT (cdd.dgterritorio.gov.pt):
- "LiDAR - Modelos Digitais do Relevo", MDT-50cm e MDS-50cm (folhas 197562 e 198562, voo de 2025);
- "LiDAR - Nuvem de pontos" (LO-197562 e LO-198562, 2025), de onde se tira a classe de cada célula
  (solo, vegetação, edifício), para separar árvores de casas e muralhas sem adivinhar. Os pontos
  de 2025 trazem cor (RGB): a cor do ponto mais alto de cada célula de 50 cm faz uma ortoimagem
  (dados/orto.jpg) para a verificação em planta e a cor do terreno, sem precisar da ortofoto.

Uso:
    python pipeline/maquetas/castro-laboreiro/lidar.py PASTA_COM_AS_FOLHAS
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
CFG = json.loads((AQUI.parents[2] / "src" / "data" / "maquetas" / "castro-laboreiro.json").read_text())
LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([0.0, 0.0])  # igual ao construir.py
MEIO = 150.0  # metros para cada lado do centro
PASSO = 0.5
BASE = 900.0


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
    k = n - 1  # grelha de cor a 50 cm (c. 4 pontos/m²), norte em cima
    alto_c = np.full(k * k, -1e9)
    rgb = np.zeros((k * k, 3), np.uint8)
    for f in fs:
        with laspy.open(f) as r:
            for ch in r.chunk_iterator(2_000_000):
                x, y = np.asarray(ch.x), np.asarray(ch.y)
                m = (x > w) & (x < e) & (y > s) & (y < nn)
                c = np.asarray(ch.classification)[m]
                cor_ok = (c >= 2) & (c <= 6)
                if cor_ok.any():
                    lon, lat = inv.transform(x[m][cor_ok], y[m][cor_ok])
                    z = np.asarray(ch.z)[m][cor_ok]
                    cc = np.stack([np.asarray(ch.red)[m][cor_ok], np.asarray(ch.green)[m][cor_ok], np.asarray(ch.blue)[m][cor_ok]], 1) // 256
                    gx = np.floor(((lon - LON0) * KX - CENTRO[0] + MEIO) / PASSO).astype(int)
                    gy = np.floor((MEIO - ((lat - LAT0) * KY - CENTRO[1])) / PASSO).astype(int)
                    ok = (gx >= 0) & (gx < k) & (gy >= 0) & (gy < k)
                    i, z, cc = gy[ok] * k + gx[ok], z[ok], cc[ok]
                    o = np.argsort(z)
                    i, z, cc = i[o], z[o], cc[o]
                    sobe = z > alto_c[i]
                    np.maximum.at(alto_c, i, z)
                    rgb[i[sobe]] = cc[sobe]
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
    print("classes", {int(a): int(b) for a, b in zip(*np.unique(cls, return_counts=True))})
    ortoimagem(rgb.reshape(k, k, 3), alto_c.reshape(k, k) > -1e8)
    return cls.reshape(n, n)


def ortoimagem(rgb, cheio):
    """Cor do ponto mais alto de cada célula de 50 cm (norte em cima), com as falhas preenchidas
    pela célula com pontos mais próxima. Gravada em dados/orto.jpg."""
    from PIL import Image
    from scipy import ndimage as ndi
    _, (ri, ci) = ndi.distance_transform_edt(~cheio, return_indices=True)
    Image.fromarray(rgb[ri, ci]).save(AQUI / "dados" / "orto.jpg", quality=88)
    print("ortoimagem da nuvem de pontos", rgb.shape, f"{cheio.mean():.0%} de células com pontos")


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


def casas(mdt, mds, cls, xs, corte=1.2, area_min=12.0):
    """Plantas dos edifícios que faltam no OSM, a partir das células de classe 6 (edifício) da nuvem
    de pontos. As casas encostadas umas às outras separam-se pelos degraus do telhado (> `corte` m
    entre células vizinhas); as que o OSM já tem ficam de fora. Devolve uma lista de
    {planta, cume, chao} (cotas absolutas)."""
    import shapely
    from scipy import ndimage as ndi
    # plantas do OSM (dados/fontes.json), na mesma grelha da cena
    osm = None
    fj = AQUI / "dados" / "fontes.json"
    if fj.exists():
        polys = []
        for f in json.loads(fj.read_text())["edificios"]:
            if f["geometria"]["type"] == "Polygon":
                a = np.array(f["geometria"]["coordinates"][0])
                polys.append(shapely.Polygon(np.stack([(a[:, 0] - LON0) * KX - CENTRO[0], (a[:, 1] - LAT0) * KY - CENTRO[1]], 1)).buffer(0))
        osm = shapely.union_all(polys)
    h = mds - mdt
    b = (cls == 6) & (h > 2.0)
    b = ndi.binary_opening(ndi.binary_closing(b, iterations=1), iterations=1)
    d = np.zeros_like(b)
    for ax in (0, 1):
        g = np.abs(np.diff(mds, axis=ax)) > corte
        if ax == 0:
            d[:-1] |= g
            d[1:] |= g
        else:
            d[:, :-1] |= g
            d[:, 1:] |= g
    lab, k = ndi.label(b & ~d)
    # as células de fronteira voltam para a casa vizinha mais próxima
    _, (ri, ci) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    lab = np.where(b, lab[ri, ci], 0)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        if m.sum() * PASSO * PASSO < area_min:
            continue
        r, c = np.nonzero(m)
        r, c = r + sl[0].start, c + sl[1].start
        pts = shapely.MultiPoint(np.stack([xs[c], xs[r]], 1))
        ret = pts.minimum_rotated_rectangle.buffer(PASSO / 2, join_style="mitre")
        area = m.sum() * PASSO * PASSO
        if area / ret.area > 0.78:
            g = ret
        else:
            g = shapely.concave_hull(pts, ratio=0.15).buffer(PASSO / 2, join_style="mitre").simplify(0.6)
        if osm is not None and g.intersects(osm):
            if g.intersection(osm).area > 0.3 * g.area:
                continue  # já está no OSM, que tem plantas mais limpas
            g = g.difference(osm.buffer(0.3))
            if g.geom_type == "MultiPolygon":
                g = max(g.geoms, key=lambda x: x.area)
            g = g.simplify(0.6)
        if g.geom_type != "Polygon" or g.area < area_min:
            continue
        mrr = np.array(g.minimum_rotated_rectangle.exterior.coords)
        if min(np.linalg.norm(mrr[1] - mrr[0]), np.linalg.norm(mrr[2] - mrr[1])) < 3.0:
            continue  # tiras estreitas: muros e muralhas, não casas
        cx = np.array(g.exterior.coords)[:-1]
        out.append({"planta": np.round(cx, 2).tolist(), "cume": round(float(np.percentile(mds[r, c], 95)), 2),
                    "chao": round(float(np.percentile(mdt[r, c], 10)), 2)})
    print("casas do LiDAR", len(out))
    return out


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
        saida[prod.lower()] = np.round((v - BASE) * 100).astype(np.int16)  # centímetros acima de BASE
        print(prod, [f.name[-30:] for f in fs], round(float(v.min()), 2), round(float(v.max()), 2))
    c = classes(pasta, tf, n)
    if c is not None:
        saida["classe"] = c
        # x, y (cena), topo da copa, chão (cotas absolutas) e raio da copa
        saida["arvores"] = arvores(saida["mdt"] / 100 + BASE, saida["mds"] / 100 + BASE, c, xs)
        (AQUI / "dados" / "casas.json").write_text(json.dumps(
            casas(saida["mdt"] / 100 + BASE, saida["mds"] / 100 + BASE, c, xs), separators=(",", ":")))
    if not c is None and not (AQUI / "dados" / "orto.jpg").exists():
        ortofoto(pasta, tf)
    np.savez_compressed(AQUI / "dados" / "lidar.npz", passo=PASSO, meio=MEIO, base=BASE, **saida,
                        fonte="DGT, LiDAR 2025 (MDT, MDS 50 cm e nuvem de pontos), folhas 197562 e 198562")


if __name__ == "__main__":
    main()
