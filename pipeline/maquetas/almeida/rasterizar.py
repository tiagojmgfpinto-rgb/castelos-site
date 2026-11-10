"""Praça-forte de Almeida: faz o MDT e o MDS de 50 cm das folhas que faltam a partir da nuvem de pontos da DGT.

Das quatro folhas que cobrem a praça (303418, 303419, 304418, 304419, voo de 2024) havia a nuvem de pontos
(LAZ) de todas, mas o MDT de 303418 e o MDS de 303418 e 303419 não vieram. Este script grava-os em GeoTIFF
igual aos da DGT (PT-TM06, 50 cm, nodata -999), com o nome `MDT-50cm-<folha>-LAZ.tif`, na mesma pasta:

- MDT: média dos pontos de solo (classe 2) em cada célula, com as falhas (casas, muralhas na classe 6)
  preenchidas por interpolação linear na triangulação das células com solo;
- MDS: ponto mais alto das classes 1 a 6 em cada célula, com as falhas preenchidas da mesma maneira.

Para conferir o método, `--conferir FOLHA` refaz uma folha que veio da DGT e mostra a diferença.

Uso:
    python pipeline/maquetas/almeida/rasterizar.py PASTA [--conferir 304419]
"""
import sys
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import LinearNDInterpolator

PASSO = 0.5
N = 2000  # células por lado (folha de 1 km)


def grelhas(laz):
    """Média do solo e máximo da superfície por célula de 50 cm (linha 0 a norte)."""
    h = laspy.open(laz).header
    x0, y1 = float(np.floor(h.mins[0] / 1000) * 1000), float(np.floor(h.mins[1] / 1000) * 1000 + 1000)
    soma = np.zeros(N * N)
    cont = np.zeros(N * N)
    alto = np.full(N * N, -np.inf)
    with laspy.open(laz) as r:
        for ch in r.chunk_iterator(3_000_000):
            x, y, z = np.asarray(ch.x), np.asarray(ch.y), np.asarray(ch.z)
            c = np.asarray(ch.classification)
            col = np.clip(((x - x0) / PASSO).astype(int), 0, N - 1)
            lin = np.clip(((y1 - y) / PASSO).astype(int), 0, N - 1)
            i = lin * N + col
            s = c == 2
            np.add.at(soma, i[s], z[s])
            np.add.at(cont, i[s], 1)
            m = (c >= 1) & (c <= 6)
            np.maximum.at(alto, i[m], z[m])
    mdt = np.where(cont > 0, soma / np.maximum(cont, 1), np.nan).reshape(N, N)
    mds = np.where(np.isfinite(alto), alto, np.nan).reshape(N, N)
    return mdt, mds, x0, y1


def preencher(a):
    ok = np.isfinite(a)
    if ok.all():
        return a
    r, c = np.nonzero(ok)
    f = LinearNDInterpolator(np.column_stack([r, c]), a[ok])
    rr, cc = np.nonzero(~ok)
    out = a.copy()
    out[~ok] = f(rr, cc)
    # bordas fora da triangulação: vizinho mais próximo
    if np.isnan(out).any():
        from scipy import ndimage as ndi
        _, (ri, ci) = ndi.distance_transform_edt(np.isnan(out), return_indices=True)
        out = out[ri, ci]
    return out


def gravar(a, x0, y1, destino):
    with rasterio.open(destino, "w", driver="GTiff", height=N, width=N, count=1, dtype="float32", crs="EPSG:3763",
                       transform=from_origin(x0, y1, PASSO, PASSO), nodata=-999.0, compress="deflate") as w:
        w.write(a.astype(np.float32), 1)
    print("gravado", destino)


def main():
    pasta = Path(sys.argv[1])
    if "--conferir" in sys.argv:
        folha = sys.argv[sys.argv.index("--conferir") + 1]
        laz = next(pasta.glob(f"LO-{folha}-*.laz"))
        mdt, mds, _, _ = grelhas(laz)
        for prod, a in (("MDT", mdt), ("MDS", mds)):
            fs = list(pasta.glob(f"{prod}-50cm-{folha}-*_v01.tif"))
            if not fs:
                continue
            ref = rasterio.open(fs[0]).read(1)
            d = preencher(a) - ref
            print(prod, folha, "diferença à DGT: mediana", round(float(np.median(d)), 3), "p05", round(float(np.percentile(d, 5)), 2),
                  "p95", round(float(np.percentile(d, 95)), 2), "|d|<0,25 m:", f"{(np.abs(d) < 0.25).mean():.1%}")
        return
    for laz in sorted(pasta.glob("LO-*.laz")):
        folha = laz.name.split("-")[1]
        falta = [p for p in ("MDT", "MDS") if not list(pasta.glob(f"{p}-50cm-{folha}-*.tif"))]
        if not falta:
            continue
        mdt, mds, x0, y1 = grelhas(laz)
        for p in falta:
            gravar(preencher(mdt if p == "MDT" else mds), x0, y1, pasta / f"{p}-50cm-{folha}-LAZ.tif")


if __name__ == "__main__":
    main()
