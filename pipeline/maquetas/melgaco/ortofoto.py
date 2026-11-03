"""Castelo de Melgaço: descarrega o Ortofotomapa 2021 da DGT (25 cm, serviço WMS Ortos2021-RGB) na caixa da
maqueta e grava-o em dados/orto.jpg (norte em cima, 25 cm por píxel), para as cores do chão e para conferir.

    python pipeline/maquetas/melgaco/ortofoto.py

Serviço: https://cartografia.dgterritorio.gov.pt/wms/ortos2021, camada Ortos2021-RGB (DGT, CC BY 4.0).
"""
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer

AQUI = Path(__file__).resolve().parent
LAT0, LON0 = 42.114413, -8.259790
E0, N0 = Transformer.from_crs(4326, 3763, always_xy=True).transform(LON0, LAT0)
X0, X1, Y0, Y1 = -140.0, 60.0, -80.0, 60.0
PASSO = 0.25
URL = "https://cartografia.dgterritorio.gov.pt/wms/ortos2021"


def main():
    w, s, e, n = E0 + X0, N0 + Y0, E0 + X1, N0 + Y1
    nx, ny = int(round((X1 - X0) / PASSO)), int(round((Y1 - Y0) / PASSO))
    q = (f"{URL}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=Ortos2021-RGB&STYLES=&CRS=EPSG:3763"
         f"&BBOX={w},{s},{e},{n}&WIDTH={nx}&HEIGHT={ny}&FORMAT=image/png")
    for tent in range(5):
        try:
            with urllib.request.urlopen(q, timeout=180) as r:
                im = Image.open(io.BytesIO(r.read())).convert("RGB")
            break
        except Exception as ex:  # noqa: BLE001
            print("falhou", ex)
            time.sleep(2 ** tent)
    else:
        raise RuntimeError("o WMS da DGT não respondeu")
    im.save(AQUI / "dados" / "orto.jpg", quality=90)
    print("dados/orto.jpg", np.asarray(im).shape)


if __name__ == "__main__":
    main()
