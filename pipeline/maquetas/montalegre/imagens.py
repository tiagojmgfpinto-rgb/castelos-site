"""Imagens da ficha de Montalegre: fotografias do Wikimedia Commons (licença livre), o ortofotomapa de 2021 da DGT
e o relevo sombreado do MDS, na área da maqueta.

Uso: python3 pipeline/maquetas/montalegre/imagens.py [PASTA_DAS_FOTOS_DO_COMMONS]
     →  public/imagens/montalegre/*.jpg

Com PASTA_DAS_FOTOS_ENVIADAS (/mnt/project-files/uploads/hearth), copia também as fotografias cedidas ao castelos.site
(autorização do autor do projeto, 2026-11-04); o cartaz e as capturas de sites ficam de fora.
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
ENVIADAS = {
    "62d1c50a-d581-4260-bcd5-6011b7a8702b": "drone-sul",
    "4f30befa-80d5-4b31-9814-c59afcbf847b": "drone-nascente",
    "f05bfa9d-a399-41bf-8702-a15b79c3650b": "do-alto-da-torre-sueste",
    "05d9fb42-46e0-4fed-9495-13d2183b16c7": "passadico-norte",
    "0e257373-bd53-4bd9-a21b-e0adaefdf1b9": "torre-e-escada",
    "3b33ddbf-6155-42d4-b9ca-9689df52dafd": "de-poente",
}
for pasta in sys.argv[1:]:
    for k, nome in {**COMMONS, **ENVIADAS}.items():
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
