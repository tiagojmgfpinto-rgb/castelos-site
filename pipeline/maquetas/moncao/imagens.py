"""Imagens da ficha a partir do LiDAR: ortoimagem (cor dos pontos) e relevo sombreado do MDS.

Uso: python3 imagens.py  →  public/imagens/moncao/{ortoimagem-lidar-2025,relevo-lidar}.jpg
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI.parents[2] / "public" / "imagens" / "moncao"
SAIDA.mkdir(parents=True, exist_ok=True)
d = np.load(AQUI / "dados" / "lidar.npz")
mds = np.flipud(d["mds"] / 100.0 + float(d["base"]))[:-1, :-1]  # linha 0 = norte, como a ortoimagem
mdt = np.flipud(d["mdt"] / 100.0 + float(d["base"]))[:-1, :-1]
classe = np.flipud(d["classe"])[:-1, :-1]
passo = float(d["passo"])

orto = Image.open(AQUI / "dados" / "orto.jpg").convert("RGB")
w, h = orto.size
o = np.asarray(orto).astype(float) / 255
# o rio Minho: o LiDAR não tem pontos sobre a água (classe 0) e o MDT fica a c. 3 m
rio = (classe == 0) & (mdt < 3.45)
rio = ndimage.binary_opening(rio, iterations=2)
lab, n = ndimage.label(rio)
tam = ndimage.sum(rio, lab, range(1, n + 1))
rio = np.isin(lab, 1 + np.nonzero(tam * passo**2 > 400)[0])
prof = np.clip(ndimage.distance_transform_edt(rio) / 50, 0, 1)[..., None]
agua = np.array([0.33, 0.40, 0.33]) * (1 - prof) + np.array([0.12, 0.22, 0.24]) * prof
o[rio] = agua[rio]
Image.fromarray((o * 255).astype(np.uint8)).resize((w * 2, h * 2), Image.LANCZOS).save(
    SAIDA / "ortoimagem-lidar-2025.jpg", quality=86)

# relevo sombreado, luz de noroeste
gy, gx = np.gradient(ndimage.gaussian_filter(mds, 0.8) * 0.3, passo)
az, alt = np.radians(315), np.radians(48)
dz_dx, dz_dn = gx, -gy
declive = np.arctan(np.hypot(dz_dx, dz_dn))
aspecto = np.arctan2(-dz_dx, dz_dn)
sombra = np.clip(np.sin(alt) * np.cos(declive) + np.cos(alt) * np.sin(declive) * np.cos(az - aspecto), 0, 1)
alt_n = np.clip((mds - 3) / 45, 0, 1)
cor = np.stack([0.55 + 0.4 * alt_n, 0.6 + 0.3 * alt_n, 0.62 + 0.18 * alt_n], -1)
cor[rio] = [0.45, 0.6, 0.68]
img = (cor * (0.25 + 0.85 * sombra[..., None])).clip(0, 1)
Image.fromarray((img * 255).astype(np.uint8)).resize((w * 2, h * 2), Image.LANCZOS).save(
    SAIDA / "relevo-lidar.jpg", quality=86)
print("ok", w, h)
