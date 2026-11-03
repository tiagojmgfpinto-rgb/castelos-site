"""Perfil da cerca por azimute (dados/plantas.json, de tracar.py), limpo: as linhas que caem nas torres, nos degraus
de aço do adarve e nos pontos soltos são tiradas e preenchidas por interpolação; mediana de 5 amostras.

Devolve a planta da cerca (polígono) e a cota do topo do adarve em função do azimute a partir do centro do pátio."""
import json
import math
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage as ndi

import comum as C

AQUI = Path(__file__).resolve().parent


def perfil():
    d = json.loads((AQUI / "dados" / "plantas.json").read_text())["perfil_cerca"]["linhas"]
    a = np.array([r[0] for r in d])
    rin = np.array([np.nan if r[1] is None else r[1] for r in d])
    rout = np.array([np.nan if r[2] is None else r[2] for r in d])
    z = np.array([np.nan if r[3] is None else r[3] for r in d])
    mau = np.isnan(z) | (z > 1020.6) | ((rout - rin) < 1.2)
    # a cerca só existe de 27,5° (encosta à torre de menagem, lado de nascente) a 352,5° (lado de poente)
    mau |= (a < 37.5) | (a > 345)
    # degraus de aço do adarve (262–278) e escada de dentro (302–312): o topo do muro está por baixo
    mau |= ((a >= 262.5) & (a <= 277.5)) | ((a >= 302.5) & (a <= 312.5))
    ok = ~mau
    out = {}
    for nome, v in (("rin", rin), ("rout", rout), ("z", z)):
        w = np.interp(a, a[ok], v[ok])
        w = ndi.median_filter(w, size=5, mode="nearest")
        out[nome] = w
    # na zona dos degraus a cota passa em rampa do adarve poente (1019,4) para o terraço noroeste (1016,25)
    k = (a >= 262.5) & (a <= 280)
    out["z"][k] = np.interp(a[k], [262.5, 280.0], [1019.40, 1016.25])
    return a, out["rin"], out["rout"], out["z"]


A, RIN, ROUT, ZT = perfil()
AZ0, AZ1 = 27.0, 352.0   # a cerca entra na torre de menagem dos dois lados


def _r(az, v):
    return float(np.interp(az, A, v))


def ztopo(x, y):
    cx, cy = C.CENTRO_PATIO
    az = (math.degrees(math.atan2(x - cx, y - cy)) + 360) % 360
    if az < 15:
        az += 360
    az = min(max(az, 37.5), 345.0)
    return _r(az, ZT)


def ponto(az, r):
    cx, cy = C.CENTRO_PATIO
    t = math.radians(az)
    return (cx + r * math.sin(t), cy + r * math.cos(t))


def planta(passo=1.25):
    az = np.arange(AZ0, AZ1 + 1e-6, passo)
    fora = [ponto(a_, _r(min(max(a_, 37.5), 345), ROUT)) for a_ in az]
    dentro = [ponto(a_, _r(min(max(a_, 37.5), 345), RIN)) for a_ in az[::-1]]
    return shapely.Polygon(fora + dentro).buffer(0)
