"""Imagens da ficha de Montalegre: fotografias do Wikimedia Commons (licença livre), o ortofotomapa de 2021 da DGT
e o relevo sombreado do MDS, na área da maqueta.

Uso: python3 pipeline/maquetas/montalegre/imagens.py [PASTA_DAS_FOTOS_DO_COMMONS]
     →  public/imagens/montalegre/*.jpg

As fotografias enviadas, de origem por confirmar, ficam de fora (só serviram de referência para a maqueta).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI.parents[2] / "public" / "imagens" / "montalegre"
SAIDA.mkdir(parents=True, exist_ok=True)

COMMONS = {  # ficheiro original do Commons → nome na galeria
    "Castelo_de_Montalegre_2020_01.jpg": "castelo-2020",
    "Castelo_de_Montalegre_-_Portugal_%2811024635733%29.jpg": "castelo-2013",
    "Torre_de_Menagem_do_Castelo_de_Montalegre.jpg": "torre-de-menagem",
    "Castelo_de_Montalegre_2020_05.jpg": "patio-2020",
}
for pasta in sys.argv[1:]:
    for k, nome in COMMONS.items():
        f = Path(pasta) / k
        if not f.exists():
            continue
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        im.thumbnail((1600, 1600), Image.LANCZOS)
        im.save(SAIDA / f"{nome}.jpg", quality=84)
        print(nome, im.size)

# ortofotomapa e relevo: 110 × 110 m, a área da maqueta (norte para cima)
d = np.load(AQUI / "dados" / "lidar.npz")
passo, x0, y0 = float(d["passo"]), float(d["x0"]), float(d["y0"])
mds = d["mds"] / 100.0 + float(d["base"])
XA, XB, YA, YB = -58.0, 52.0, -78.0, 32.0
orto = Image.open(AQUI / "dados" / "orto.jpg").convert("RGB")  # 25 cm, x -85..75, y -95..55
op = 0.25
tam = (1200, 1200)
orto.crop((int((XA + 85) / op), int((55 - YB) / op), int((XB + 85) / op), int((55 - YA) / op))).resize(tam, Image.LANCZOS).save(
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
alt_n = np.clip((m - 1000) / 40, 0, 1)
cor = np.stack([0.55 + 0.4 * alt_n, 0.6 + 0.3 * alt_n, 0.62 + 0.18 * alt_n], -1)
img = (cor * (0.25 + 0.85 * sombra[..., None])).clip(0, 1)
Image.fromarray((img * 255).astype(np.uint8)).resize(tam, Image.LANCZOS).save(SAIDA / "relevo-lidar.jpg", quality=86)
print("ok")
