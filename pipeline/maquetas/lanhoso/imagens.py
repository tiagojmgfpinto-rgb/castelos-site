"""Imagens da ficha de Lanhoso: as fotografias cedidas ao castelos.site, o ortofotomapa de 2021 da DGT e o relevo
sombreado do MDS, do castelo ao santuário.

Uso: python3 pipeline/maquetas/lanhoso/imagens.py [PASTA_DAS_FOTOS]
     →  public/imagens/lanhoso/*.jpg

As fotografias pequenas (tiradas de páginas da internet) e a captura do geoportal ficam de fora.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI.parents[2] / "public" / "imagens" / "lanhoso"
SAIDA.mkdir(parents=True, exist_ok=True)

FOTOS = {
    "d74a3f84-image.png": "torre-drone-sul",
    "image 11.png": "porta-e-penedo",
    "image 12.png": "porta-vista-de-dentro",
    "image 9.png": "adarve-e-patio",
}
if len(sys.argv) > 1:
    for pasta in sys.argv[1:]:
        for k, nome in FOTOS.items():
            f = Path(pasta) / k
            if not f.exists():
                continue
            im = Image.open(f).convert("RGB")
            if max(im.size) > 1600:
                im.thumbnail((1600, 1600), Image.LANCZOS)
            im.save(SAIDA / f"{nome}.jpg", quality=84)
            print(nome, im.size)

# ortofotomapa e relevo: 150 × 120 m, do castelo ao santuário (norte para cima), a área da maqueta
d = np.load(AQUI / "dados" / "lidar.npz")
passo, x0, y0 = float(d["passo"]), float(d["x0"]), float(d["y0"])
mds = d["mds"] / 100.0 + float(d["base"])
XA, XB, YA, YB = -70.0, 80.0, -63.0, 57.0
orto = Image.open(AQUI / "dados" / "orto.jpg").convert("RGB")  # 25 cm, x -95..85, y -63,5..76
op = 0.25
tam = (1500, 1200)
orto.crop((int((XA + 95) / op), int((76 - YB) / op), int((XB + 95) / op), int((76 - YA) / op))).resize(tam, Image.LANCZOS).save(
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
alt_n = np.clip((m - 360) / 40, 0, 1)
cor = np.stack([0.55 + 0.4 * alt_n, 0.6 + 0.3 * alt_n, 0.62 + 0.18 * alt_n], -1)
img = (cor * (0.25 + 0.85 * sombra[..., None])).clip(0, 1)
Image.fromarray((img * 255).astype(np.uint8)).resize(tam, Image.LANCZOS).save(SAIDA / "relevo-lidar.jpg", quality=86)
print("ok")
