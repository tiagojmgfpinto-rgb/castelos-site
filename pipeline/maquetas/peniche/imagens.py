"""Imagens da ficha a partir do LiDAR: ortoimagem (cor dos pontos) e relevo sombreado do MDS.

Uso: python3 imagens.py  →  public/imagens/peniche/{ortoimagem-lidar-2024,relevo-lidar}.jpg
"""
from pathlib import Path
import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI.parents[2] / "public" / "imagens" / "peniche"
SAIDA.mkdir(parents=True, exist_ok=True)
d = np.load(AQUI / "dados" / "lidar.npz")
mds = np.flipud(d["mds"].astype(float))[:-1, :-1]  # linha 0 = norte, como a ortoimagem
classe = np.flipud(d["classe"])[:-1, :-1]
passo = float(d["passo"])

orto = Image.open(AQUI / "dados" / "orto.jpg").convert("RGB")
w, h = orto.size
o = np.asarray(orto).astype(float) / 255
mar = (mds < 0.9) | (classe == 9)
from scipy import ndimage
mar = ndimage.binary_opening(mar, iterations=2)
prof = np.clip(ndimage.distance_transform_edt(mar) / 60, 0, 1)[..., None]
agua = np.array([0.36, 0.62, 0.66]) * (1 - prof) + np.array([0.16, 0.36, 0.45]) * prof
o[mar] = agua[mar]
orto = Image.fromarray((o * 255).astype(np.uint8))
orto.resize((w * 2, h * 2), Image.LANCZOS).save(SAIDA / "ortoimagem-lidar-2024.jpg", quality=86)

# relevo sombreado, luz de noroeste (linha 0 = norte, como a ortoimagem)
gy, gx = np.gradient(ndimage.gaussian_filter(mds, 0.8) * 0.3, passo)
az, alt = np.radians(315), np.radians(48)
dz_dx, dz_dn = gx, -gy  # norte para cima
declive = np.arctan(np.hypot(dz_dx, dz_dn))
aspecto = np.arctan2(-dz_dx, dz_dn)
sombra = np.sin(alt) * np.cos(declive) + np.cos(alt) * np.sin(declive) * np.cos(az - aspecto)
sombra = np.clip(sombra, 0, 1)
alt_n = np.clip((mds - 0) / 40, 0, 1)
cor = np.stack([0.55 + 0.4 * alt_n, 0.6 + 0.3 * alt_n, 0.62 + 0.18 * alt_n], -1)
cor[mar] = [0.45, 0.62, 0.72]
img = (cor * (0.25 + 0.85 * sombra[..., None])).clip(0, 1)
Image.fromarray((img * 255).astype(np.uint8)).resize((w * 2, h * 2), Image.LANCZOS).save(SAIDA / "relevo-lidar.jpg", quality=86)
print("ok", w, h)
