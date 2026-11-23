"""Castelo e muralhas de Silves: descarrega o Ortofotomapa 2025 da DGT (25 cm, serviço WMS Ortos2025-RGB) na caixa da
maqueta e grava-o em dados/orto.jpg (norte em cima, 25 cm por píxel), para conferir a planta e as cores do chão.

    python pipeline/maquetas/silves/ortofoto.py

Serviço: https://cartografia.dgterritorio.gov.pt/wms/ortos2025, camada Ortos2025-RGB (DGT, CC BY 4.0).
"""
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from comum import E0, N0, X0, X1, Y0, Y1  # noqa: E402

PASSO = 0.25
URL = "https://cartografia.dgterritorio.gov.pt/wms/ortos2025"
Image.MAX_IMAGE_PIXELS = None


def pedaco(w, s, e, n, nx, ny):
    q = (f"{URL}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=Ortos2025-RGB&STYLES=&CRS=EPSG:3763"
         f"&BBOX={w},{s},{e},{n}&WIDTH={nx}&HEIGHT={ny}&FORMAT=image/jpeg")
    for tent in range(6):
        try:
            with urllib.request.urlopen(q, timeout=180) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception as ex:  # noqa: BLE001
            print("falhou", ex)
            time.sleep(2 ** tent)
    raise RuntimeError("o WMS da DGT não respondeu")


def main():
    nx, ny = int(round((X1 - X0) / PASSO)), int(round((Y1 - Y0) / PASSO))
    im = Image.new("RGB", (nx, ny))
    lado = 1240  # píxeis por pedido (310 m)
    for j in range(0, ny, lado):
        for i in range(0, nx, lado):
            a, b = min(lado, nx - i), min(lado, ny - j)
            w = E0 + X0 + i * PASSO
            n = N0 + Y1 - j * PASSO
            im.paste(pedaco(w, n - b * PASSO, w + a * PASSO, n, a, b), (i, j))
            print("pedaço", i, j, flush=True)
    im.save(AQUI / "dados" / "orto.jpg", quality=90)
    print("dados/orto.jpg", np.asarray(im).shape)


if __name__ == "__main__":
    main()
