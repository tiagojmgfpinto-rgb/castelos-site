"""Imagens da ficha: o ortofotomapa de 2021 da DGT e o relevo sombreado do MDS, à volta da torre.

Uso: python3 pipeline/maquetas/braga/imagens.py  →  public/imagens/braga/{ortofotomapa-2021,relevo-lidar}.jpg
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI.parents[2] / "public" / "imagens" / "braga"
SAIDA.mkdir(parents=True, exist_ok=True)
d = np.load(AQUI / "dados" / "lidar.npz")
mds = np.flipud(d["mds"] / 100.0 + float(d["base"]))  # linha 0 = norte, como o ortofotomapa
passo = float(d["passo"])
n = mds.shape[0]
c = n // 2
k = int(50 / passo)  # 100 × 100 m à volta da torre
orto = Image.open(AQUI / "dados" / "orto.jpg").convert("RGB")
w = orto.size[0]
f = w / (2 * float(d["meio"]))
orto.crop((int(w / 2 - 50 * f), int(w / 2 - 50 * f), int(w / 2 + 50 * f), int(w / 2 + 50 * f))).resize((1000, 1000), Image.LANCZOS).save(
    SAIDA / "ortofotomapa-2021.jpg", quality=86)
m = mds[c - k:c + k, c - k:c + k]
gy, gx = np.gradient(ndimage.gaussian_filter(m, 0.6), passo)
az, alt = np.radians(315), np.radians(45)
declive = np.arctan(np.hypot(gx, gy))
aspecto = np.arctan2(-gx, -gy)
sombra = np.clip(np.sin(alt) * np.cos(declive) + np.cos(alt) * np.sin(declive) * np.cos(az - aspecto), 0, 1)
alt_n = np.clip((m - 190) / 30, 0, 1)
cor = np.stack([0.55 + 0.4 * alt_n, 0.6 + 0.3 * alt_n, 0.62 + 0.18 * alt_n], -1)
img = (cor * (0.25 + 0.85 * sombra[..., None])).clip(0, 1)
Image.fromarray((img * 255).astype(np.uint8)).resize((1000, 1000), Image.LANCZOS).save(SAIDA / "relevo-lidar.jpg", quality=86)
print("ok")
