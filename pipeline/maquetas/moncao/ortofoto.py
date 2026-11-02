"""Fortaleza de Monção: descarrega o ortofotomapa da DGT (OrtoSat 2023, cor verdadeira, serviço WMS) e
reamostra-o para a grelha local da maqueta, a 25 cm, em dados/ortofoto.jpg (norte em cima, a mesma caixa do
dados/orto.jpg do LiDAR: 1000 x 600 m em torno da origem). É a camada "Ortofotomapa 2023" do WebSIG da Câmara
de Monção (websig.cm-moncao.pt), a que já mostra o passadiço de madeira da encosta norte.

    python pipeline/maquetas/moncao/ortofoto.py

Serviço: https://ortos.dgterritorio.gov.pt/wms/ortosat2023, camada ortoSat2023-CorVerdadeira (DGT, CC BY 4.0).
"""
import io
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer

AQUI = Path(__file__).resolve().parent
CFG = json.loads((AQUI.parents[2] / "src" / "data" / "maquetas" / "moncao.json").read_text())
LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
MEIOX, MEIOY = 500.0, 300.0  # igual ao lidar.py
PASSO = 0.25  # saída
PASSO_WMS = 0.25  # pedido ao serviço (m/pixel em PT-TM06)
URL = "https://ortos.dgterritorio.gov.pt/wms/ortosat2023"
CAMADA = "ortoSat2023-CorVerdadeira"
BLOCO = 1000  # pixels por pedido
# O OrtoSat 2023 está c. 2,25 m a este e 0,5 m a norte da nuvem de pontos do LiDAR 2025 (medido pela correlação
# dos pormenores do chão, classe 2, entre as duas imagens). O LiDAR manda: amostra-se o orto deslocado.
DESVIO_E, DESVIO_N = 2.25, 0.5


def pedir(w, s, e, n, nx, ny):
    q = (f"{URL}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={CAMADA}&STYLES=&CRS=EPSG:3763"
         f"&BBOX={w},{s},{e},{n}&WIDTH={nx}&HEIGHT={ny}&FORMAT=image/png")
    for tent in range(5):
        try:
            with urllib.request.urlopen(q, timeout=120) as r:
                return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
        except Exception as ex:  # noqa: BLE001
            print("  falhou", ex, "- nova tentativa")
            time.sleep(2 ** tent)
    raise RuntimeError("o WMS da DGT não respondeu")


def main():
    fw = Transformer.from_crs(4326, 3763, always_xy=True)
    # píxeis de saída (centros), norte em cima
    W, H = int(2 * MEIOX / PASSO), int(2 * MEIOY / PASSO)
    x = -MEIOX + (np.arange(W) + 0.5) * PASSO
    y = MEIOY - (np.arange(H) + 0.5) * PASSO
    X, Y = np.meshgrid(x, y)
    E, N = fw.transform(LON0 + X / KX, LAT0 + Y / KY)
    E, N = E + DESVIO_E, N + DESVIO_N
    w, e = math.floor(E.min()) - 5, math.ceil(E.max()) + 5
    s, n = math.floor(N.min()) - 5, math.ceil(N.max()) + 5
    nx, ny = int(round((e - w) / PASSO_WMS)), int(round((n - s) / PASSO_WMS))
    print(f"PT-TM06 {w},{s} - {e},{n}: {nx} x {ny} px")
    mos = np.zeros((ny, nx, 3), np.uint8)
    for r0 in range(0, ny, BLOCO):
        for c0 in range(0, nx, BLOCO):
            bx, by = min(BLOCO, nx - c0), min(BLOCO, ny - r0)
            bw, bn = w + c0 * PASSO_WMS, n - r0 * PASSO_WMS
            mos[r0:r0 + by, c0:c0 + bx] = pedir(bw, bn - by * PASSO_WMS, bw + bx * PASSO_WMS, bn, bx, by)
            print(f"  bloco {r0 // BLOCO},{c0 // BLOCO}")
    # bilinear do mosaico para a grelha local
    c = (E - w) / PASSO_WMS - 0.5
    r = (n - N) / PASSO_WMS - 0.5
    c0, r0 = np.floor(c).astype(int).clip(0, nx - 2), np.floor(r).astype(int).clip(0, ny - 2)
    fc, fr = (c - c0)[..., None], (r - r0)[..., None]
    a = mos.astype(np.float32)
    out = (a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c0 + 1] * fc * (1 - fr)
           + a[r0 + 1, c0] * (1 - fc) * fr + a[r0 + 1, c0 + 1] * fc * fr)
    Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(AQUI / "dados" / "ortofoto.jpg", quality=90)
    print("dados/ortofoto.jpg", out.shape)


if __name__ == "__main__":
    main()
