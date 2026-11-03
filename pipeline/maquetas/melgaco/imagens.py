"""Imagens da ficha de Melgaço: as fotografias cedidas ao castelos.site, o ortofotomapa de 2021 da DGT e o relevo
sombreado do MDS, à volta do castelo.

Uso: python3 pipeline/maquetas/melgaco/imagens.py PASTA_DAS_FOTOS
     →  public/imagens/melgaco/*.jpg

As fotografias com marca de água de outro autor (m04 e m08) ficam de fora.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI.parents[2] / "public" / "imagens" / "melgaco"
SAIDA.mkdir(parents=True, exist_ok=True)

FOTOS = {
    "m11": "castelo-drone-sul",
    "m12": "castelo-aereo-sudoeste",
    "m09": "torre-face-sul",
    "m03": "torre-porta-escada",
    "m02": "recinto-e-torre",
    "m05": "entrada-e-barbaca",
    "m01": "castelo-da-rua",
    "m10": "torre-e-igreja-matriz",
    "m07": "porta-de-baixo-frente",
    "m06": "porta-de-baixo",
}
if len(sys.argv) > 1:
    pasta = Path(sys.argv[1])
    for k, nome in FOTOS.items():
        f = next(pasta.glob(f"{k}.*"))
        im = Image.open(f).convert("RGB")
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600), Image.LANCZOS)
        im.save(SAIDA / f"{nome}.jpg", quality=88)
        print(nome, im.size)

# ortofotomapa e relevo: 150 × 100 m, do castelo à Porta de Baixo (norte para cima)
d = np.load(AQUI / "dados" / "lidar.npz")
passo, x0, y0 = float(d["passo"]), float(d["x0"]), float(d["y0"])
mds = d["mds"] / 100.0 + float(d["base"])
XA, XB, YA, YB = -110.0, 40.0, -45.0, 45.0
orto = Image.open(AQUI / "dados" / "orto.jpg").convert("RGB")  # 25 cm, x -140..60, y -80..60
op = 0.25
orto.crop((int((XA + 140) / op), int((60 - YB) / op), int((XB + 140) / op), int((60 - YA) / op))).resize((1500, 900), Image.LANCZOS).save(
    SAIDA / "ortofotomapa-2021.jpg", quality=86)
c0, c1 = int((XA - x0) / passo), int((XB - x0) / passo)
r0, r1 = int((YA - y0) / passo), int((YB - y0) / passo)
m = np.flipud(mds[r0:r1, c0:c1])  # linha 0 = norte
gy, gx = np.gradient(ndimage.gaussian_filter(m, 0.6), passo)
gy = -gy
az, alt = np.radians(315), np.radians(45)
declive = np.arctan(np.hypot(gx, gy))
aspecto = np.arctan2(-gx, gy)
sombra = np.clip(np.sin(alt) * np.cos(declive) + np.cos(alt) * np.sin(declive) * np.cos(az - aspecto), 0, 1)
alt_n = np.clip((m - 176) / 36, 0, 1)
cor = np.stack([0.55 + 0.4 * alt_n, 0.6 + 0.3 * alt_n, 0.62 + 0.18 * alt_n], -1)
img = (cor * (0.25 + 0.85 * sombra[..., None])).clip(0, 1)
Image.fromarray((img * 255).astype(np.uint8)).resize((1500, 900), Image.LANCZOS).save(SAIDA / "relevo-lidar.jpg", quality=86)
print("ok")
