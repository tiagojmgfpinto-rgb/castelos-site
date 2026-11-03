"""Imagens da ficha de Lindoso: as fotografias cedidas ao castelos.site, o ortofotomapa de 2021 da DGT e o relevo
sombreado do MDS, do castelo aos espigueiros.

Uso: python3 pipeline/maquetas/lindoso/imagens.py PASTA_DAS_FOTOS
     →  public/imagens/lindoso/*.jpg

As fotografias com marca de água de outro autor (duas aéreas e uma do pátio), a infografia de jornal e a imagem de
satélite do Google ficam de fora.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI.parents[2] / "public" / "imagens" / "lindoso"
SAIDA.mkdir(parents=True, exist_ok=True)

FOTOS = {
    "l01": "castelo-drone-sul",
    "l02": "castelo-e-albufeira",
    "l03": "castelo-e-espigueiros",
    "l04": "torre-face-sul",
    "l05": "torre-e-postigo",
    "l06": "porta-norte",
    "l07": "patio-e-museu",
    "l08": "eira-dos-espigueiros",
    "l09": "espigueiro",
    "l10": "espigueiro-e-castelo",
    "l11": "guarita-sueste",
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

# ortofotomapa e relevo: 150 × 170 m, do castelo à eira dos espigueiros (norte para cima)
d = np.load(AQUI / "dados" / "lidar.npz")
passo, x0, y0 = float(d["passo"]), float(d["x0"]), float(d["y0"])
mds = d["mds"] / 100.0 + float(d["base"])
XA, XB, YA, YB = -115.0, 50.0, -150.0, 45.0
orto = Image.open(AQUI / "dados" / "orto.jpg").convert("RGB")  # 25 cm, x -130..80, y -160..70
op = 0.25
tam = (1320, 1560)
orto.crop((int((XA + 130) / op), int((70 - YB) / op), int((XB + 130) / op), int((70 - YA) / op))).resize(tam, Image.LANCZOS).save(
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
alt_n = np.clip((m - 440) / 40, 0, 1)
cor = np.stack([0.55 + 0.4 * alt_n, 0.6 + 0.3 * alt_n, 0.62 + 0.18 * alt_n], -1)
img = (cor * (0.25 + 0.85 * sombra[..., None])).clip(0, 1)
Image.fromarray((img * 255).astype(np.uint8)).resize(tam, Image.LANCZOS).save(SAIDA / "relevo-lidar.jpg", quality=86)
print("ok")
