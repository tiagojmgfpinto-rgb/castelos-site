"""Superfície das muralhas a partir da nuvem de pontos: em Monção os parapeitos e muitas faces vêm na classe 1
(não classificado) e o MDT da DGT passa por baixo deles. Grava, na grelha do lidar.npz (50 cm, linha 0 a sul):
  sup   - ponto mais alto das classes 1, 2 e 6 (sem árvores)
  c1    - ponto mais alto da classe 1
  solo  - ponto mais baixo da classe 2
em centímetros (int16, -32768 = sem pontos) em dados/superficie.npz.

Uso: python3 pontos.py PASTA   (a pasta com LO-170568-07-2025_v05.laz e LO-171568-07-2025_v05.laz)
"""
import glob
import sys
from pathlib import Path
import laspy
import numpy as np

AQUI = Path(__file__).resolve().parent
L = np.load(AQUI / "dados" / "lidar.npz")
MX, MY, PA = float(L["meiox"]), float(L["meioy"]), float(L["passo"])
NX, NY = L["mdt"].shape[1], L["mdt"].shape[0]
X0, Y0 = -28721.75, 267647.02  # PT-TM06 da origem (42.07785, -8.48020)

sup = np.full(NX * NY, -1e9)
c1 = np.full(NX * NY, -1e9)
solo = np.full(NX * NY, 1e9)
for f in sorted(glob.glob(str(Path(sys.argv[1]) / "LO-1*568-07-2025_v05.laz"))):
    with laspy.open(f) as r:
        for ch in r.chunk_iterator(3_000_000):
            x = np.asarray(ch.x) - X0
            y = np.asarray(ch.y) - Y0
            z = np.asarray(ch.z)
            c = np.asarray(ch.classification)
            k = (np.abs(x) <= MX) & (np.abs(y) <= MY)
            i = np.round((x[k] + MX) / PA).astype(int).clip(0, NX - 1)
            j = np.round((y[k] + MY) / PA).astype(int).clip(0, NY - 1)
            idx, zk, ck = j * NX + i, z[k], c[k]
            m = (ck == 1) | (ck == 2) | (ck == 6)
            np.maximum.at(sup, idx[m], zk[m])
            m = ck == 1
            np.maximum.at(c1, idx[m], zk[m])
            m = ck == 2
            np.minimum.at(solo, idx[m], zk[m])


def cm(a, vazio):
    out = np.where(np.abs(a) < 1e8, np.round(a * 100), -32768).astype(np.int16)
    return out.reshape(NY, NX)


np.savez_compressed(AQUI / "dados" / "superficie.npz", sup=cm(sup, -1e9), c1=cm(c1, -1e9), solo=cm(solo, 1e9))
print("superficie.npz", NX, NY)
