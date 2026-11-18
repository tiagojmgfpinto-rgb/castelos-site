"""Castelo de Almourol: mede as muralhas e as torres no LiDAR (superficie.npz, células de 25 cm) e grava
dados/muralhas.json.

Cada pano é um eixo desenhado à mão sobre a planta do relevo (pontos abaixo, x este / y norte em metros, origem no
centro da torre de menagem), com o exterior à esquerda de quem anda no sentido dos pontos. Em estações a cada
0,5 m mede-se, num perfil perpendicular de ±4,5 m, a faixa de alvenaria (células mais de 1,8 m acima do chão
local): face de fora e face de dentro, o adarve (percentil 35 do topo), as pontas dos merlões (percentil 92) e o
chão de cada lado. As torres (cubelos) são círculos ajustados ao contorno da alvenaria.

    python pipeline/maquetas/almourol/tracar.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
S = np.load(AQUI / "dados" / "superficie.npz")
Z = S["z"] / 100.0 + float(S["base"])
P_, X0_, Y0_ = float(S["passo"]), float(S["x0"]), float(S["y0"])
L = np.load(AQUI / "dados" / "lidar.npz")
MDT = L["mdt"] / 100.0 + float(L["base"])
LP, LX0, LY0 = float(L["passo"]), float(L["x0"]), float(L["y0"])

# chão local: abertura morfológica de 7 m (tira muros e torres); alvenaria = mais de 1,8 m acima dele
CH = ndi.grey_opening(Z, size=(29, 29))
ALV = ndi.binary_opening((Z - CH) > 1.8, iterations=1)


def amostra(G, x0, y0, p, x, y, ordem=1):
    c = (np.asarray(x) - x0) / p
    r = (np.asarray(y) - y0) / p
    return ndi.map_coordinates(G.astype(float), [r, c], order=ordem, mode="nearest")


# eixos dos panos (exterior à esquerda do sentido de marcha: o circuito exterior vai no sentido dos ponteiros)
PANOS = {
    # cerca exterior, do cubelo sudoeste para norte e à volta, no sentido dos ponteiros do relógio
    "oeste_sul": [(-13.0, -13.6), (-11.4, -10.6), (-10.4, -8.4), (-9.0, -5.6), (-8.6, -4.4)],
    "oeste_norte": [(-8.6, -4.4), (-9.4, -2.4), (-9.6, 1.0), (-9.2, 4.0), (-8.4, 6.6), (-7.6, 8.8)],
    "norte": [(-7.6, 8.8), (-4.0, 8.6), (1.0, 9.8), (6.0, 11.0), (10.0, 12.0), (12.4, 12.4), (15.0, 11.0),
              (19.0, 9.2), (23.4, 7.6)],
    "nordeste": [(23.4, 7.6), (27.0, 6.4), (30.0, 5.4), (33.4, 4.4)],
    "leste": [(33.4, 4.4), (30.0, 1.6), (26.6, -3.6), (23.0, -9.4), (20.0, -14.0), (17.4, -18.4)],
    "sul": [(17.4, -18.4), (14.0, -16.6), (8.0, -16.4), (2.0, -16.9), (-4.0, -17.6), (-8.0, -16.8), (-11.0, -16.0),
            (-13.0, -13.6)],
    # muro interior (separa o recinto de cima, a norte, do de baixo): o "exterior" é o recinto de baixo (sul)
    "interior_oeste": [(-8.6, -4.4), (-5.0, -4.3), (-2.0, -4.3)],
    "interior_leste": [(4.6, -2.0), (6.6, -4.2), (8.8, -5.0), (10.4, -3.2), (11.4, -0.4), (13.6, 2.4), (17.0, 4.2),
                       (20.4, 5.4), (23.4, 7.6)],
}
TORRES0 = {
    "sudoeste": (-13.4, -14.0), "oeste_sul": (-12.6, -8.0), "oeste": (-11.8, -0.9), "noroeste": (-7.6, 9.0),
    "norte": (12.4, 13.6), "nordeste": (23.6, 7.8), "leste": (33.6, 4.2), "sueste": (17.2, -18.8),
}


def densificar(pts, passo):
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / passo)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def medir_pano(pts):
    q = densificar(pts, 0.5)
    t = np.gradient(q, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    nl = np.stack([-t[:, 1], t[:, 0]], 1)  # à esquerda = fora
    ds = np.arange(-4.5, 4.51, 0.125)
    est = []
    for i in range(len(q)):
        xy = q[i][None, :] + nl[i][None, :] * ds[:, None]
        a = amostra(ALV, X0_, Y0_, P_, xy[:, 0], xy[:, 1], ordem=0) > 0.5
        z = amostra(Z, X0_, Y0_, P_, xy[:, 0], xy[:, 1], ordem=0)
        # corrida de alvenaria mais próxima do eixo
        j0 = len(ds) // 2
        if not a.any():
            est.append(None)
            continue
        idx = np.nonzero(a)[0]
        k = idx[np.argmin(np.abs(idx - j0))]
        lo = k
        while lo > 0 and a[lo - 1]:
            lo -= 1
        hi = k
        while hi < len(ds) - 1 and a[hi + 1]:
            hi += 1
        fora, dentro = ds[hi], ds[lo]
        zt = z[lo:hi + 1]
        pf, pd = q[i] + nl[i] * (fora + 1.2), q[i] + nl[i] * (dentro - 1.2)
        zf = amostra(MDT, LX0, LY0, LP, [pf[0]], [pf[1]])[0]
        zd = amostra(MDT, LX0, LY0, LP, [pd[0]], [pd[1]])[0]
        est.append(dict(c=q[i].round(3).tolist(), n=nl[i].round(4).tolist(), fora=float(fora), dentro=float(dentro),
                        adarve=float(np.percentile(zt, 35)), topo=float(np.percentile(zt, 92)),
                        chao_fora=float(zf), chao_dentro=float(zd)))
    return est


def medir_torre(c0):
    """Círculo pelas células de alvenaria ligadas à volta de c0 (até 4 m): centro, raio pelo contorno, topo."""
    c0 = np.array(c0)
    r = np.arange(0.0, 4.5, 0.125)
    az = np.radians(np.arange(0, 360, 5))
    raios, pts = [], []
    for a in az:
        d = np.array([math.cos(a), math.sin(a)])
        xy = c0[None, :] + r[:, None] * d[None, :]
        al = amostra(ALV, X0_, Y0_, P_, xy[:, 0], xy[:, 1], ordem=0) > 0.5
        k = np.nonzero(~al)[0]
        rr = r[k[0]] if len(k) else r[-1]
        raios.append(rr)
        pts.append(c0 + d * rr)
    pts = np.array(pts)
    raios = np.array(raios)
    # ajuste de círculo só com as direções em que a torre sai do muro (raio abaixo da mediana + 1 m)
    ok = raios < np.median(raios) + 0.6
    A = np.c_[2 * pts[ok], np.ones(ok.sum())]
    b = (pts[ok] ** 2).sum(1)
    cx, cy, k = np.linalg.lstsq(A, b, rcond=None)[0]
    R = math.sqrt(k + cx * cx + cy * cy)
    yy, xx = np.mgrid[-5:5.01:0.25, -5:5.01:0.25]
    dd = np.hypot(xx, yy)
    zz = amostra(Z, X0_, Y0_, P_, cx + xx, cy + yy, ordem=0)
    anel = (dd > R - 0.9) & (dd < R - 0.1)
    meio = dd < R - 1.0
    ch = amostra(MDT, LX0, LY0, LP, cx + (R + 1.5) * np.cos(az), cy + (R + 1.5) * np.sin(az))
    return dict(c=[round(cx, 3), round(cy, 3)], r=round(R, 3), topo=float(np.percentile(zz[anel], 92)),
                parapeito=float(np.percentile(zz[anel], 40)), piso=float(np.percentile(zz[meio], 50)) if meio.any() else None,
                chao_min=float(ch.min()), chao_med=float(np.median(ch)))


def main():
    out = {"panos": {}, "torres": {}}
    for k, pts in PANOS.items():
        e = medir_pano(pts)
        out["panos"][k] = {"eixo": pts, "estacoes": e}
        v = [x for x in e if x]
        print(f"{k:16s} {len(e):3d} est.  esp {np.median([x['fora'] - x['dentro'] for x in v]):.2f}  "
              f"adarve {np.median([x['adarve'] for x in v]):.2f}  topo {np.median([x['topo'] for x in v]):.2f}  "
              f"chão fora {np.median([x['chao_fora'] for x in v]):.2f} dentro {np.median([x['chao_dentro'] for x in v]):.2f}  "
              f"desvio {np.median([(x['fora'] + x['dentro']) / 2 for x in v]):+.2f}  falhas {len(e) - len(v)}")
    for k, c in TORRES0.items():
        t = medir_torre(c)
        out["torres"][k] = t
        print(f"torre {k:10s}", t)
    (AQUI / "dados" / "muralhas.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
