"""Castelo de Silves: mede as muralhas e as torres da alcáçova no LiDAR (superficie.npz, células de 25 cm) e grava
dados/muralhas.json.

Cada pano é um eixo aproximado desenhado à mão sobre a planta do relevo (x este / y norte em metros, origem no centro
da alcáçova), percorrido no sentido dos ponteiros do relógio, com o exterior à esquerda. Em estações a cada 0,5 m
mede-se, num perfil perpendicular de ±4,5 m, a faixa de alvenaria (células mais de 1,2 m acima do chão local): face
de fora e face de dentro, o adarve (percentil 35 do topo), as pontas dos merlões (percentil 92) e o chão de cada lado.
As torres são retângulos mínimos das células altas (abertura de 3,75 m, que apaga os panos) à volta de cada semente.

    python pipeline/maquetas/silves/tracar.py
"""
import json
import math
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
S = np.load(AQUI / "dados" / "superficie.npz")
Z = S["z"] / 100.0 + float(S["base"])
P_, X0_, Y0_ = float(S["passo"]), float(S["x0"]), float(S["y0"])
L = np.load(AQUI / "dados" / "lidar.npz")
MDT = L["mdt"] / 100.0 + float(L["base"])
LP, LX0, LY0 = float(L["passo"]), float(L["x0"]), float(L["y0"])

# só a zona da alcáçova (para a abertura morfológica não correr a grelha toda)
RX0, RX1, RY0, RY1 = -80.0, 75.0, -95.0, 90.0
_c0, _c1 = int((RX0 - X0_) / P_), int((RX1 - X0_) / P_)
_r0, _r1 = int((RY0 - Y0_) / P_), int((RY1 - Y0_) / P_)
ZR = Z[_r0:_r1, _c0:_c1]
ZX0, ZY0 = X0_ + _c0 * P_, Y0_ + _r0 * P_
CH = ndi.grey_opening(ZR, size=(41, 41))           # chão local: abertura de 10 m (tira muros e torres)
ALV = ndi.binary_opening((ZR - CH) > 1.2, iterations=1)
ALTO = ndi.binary_opening((ZR - CH) > 1.5, structure=np.ones((15, 15)))  # corpos com 3,75 m ou mais: torres


def amostra(G, x0, y0, p, x, y, ordem=1):
    c = (np.asarray(x) - x0) / p
    r = (np.asarray(y) - y0) / p
    return ndi.map_coordinates(G.astype(float), [r, c], order=ordem, mode="nearest")


# eixos aproximados dos panos (sentido dos ponteiros do relógio: exterior à esquerda)
PANOS = {
    "norte_oeste": [(-45.0, 44.8), (-34.4, 44.8)],
    "norte_diagonal": [(-34.4, 44.8), (-28.4, 53.8), (-25.5, 57.8)],
    "norte": [(-25.5, 57.8), (-20.0, 61.4), (-10.0, 61.6), (0.0, 61.3), (14.5, 61.2)],
    "nordeste": [(23.5, 57.5), (35.0, 49.0), (42.2, 40.0), (46.8, 33.0), (49.6, 28.2)],
    "leste": [(49.6, 28.2), (47.6, 19.8), (45.5, 10.2), (42.6, 0.0), (40.2, -10.0), (38.5, -18.0)],
    "sueste": [(37.0, -24.0), (32.5, -33.0), (30.2, -38.1), (28.0, -43.5), (26.0, -49.5), (24.0, -55.5)],
    "sul": [(16.0, -60.8), (0.0, -62.0), (-11.0, -63.2)],
    "sul_oeste": [(-27.5, -59.4), (-37.5, -61.2)],
    "oeste": [(-43.4, -54.0), (-45.0, -44.0), (-46.6, -30.0), (-49.1, -17.0), (-51.0, -7.0), (-52.3, 0.0),
              (-51.5, 6.5), (-48.4, 18.0), (-47.5, 25.0), (-45.5, 35.0), (-45.0, 44.8)],
}
# sementes das torres (centro aproximado) e nomes (OSM: Menagem, Segredo, Mulheres, Centro, Albarrãs)
# (centro, cota mínima das células do corpo da torre, raio de procura): a cota fica abaixo do terraço e acima do
# adarve dos panos que lá chegam, para o retângulo apanhar só a torre
TORRES0 = {
    "menagem": ((20.0, 67.5), 62.0, 9.0), "segredo": ((-49.0, 50.0), 60.0, 6.5), "norte": ((-28.3, 57.6), 59.6, 4.0),
    "albarra_nordeste": ((54.8, 33.2), 58.6, 4.0), "centro": ((40.3, -23.3), 56.0, 6.5),
    "albarra_sueste": ((35.5, -49.0), 52.0, 8.5), "sueste": ((21.0, -60.3), 54.6, 5.0),
    "porta": ((-28.0, -71.3), 57.0, 3.5), "mulheres": ((-41.6, -61.6), 60.0, 6.5),
    "oeste": ((-55.8, -8.2), 57.5, 4.0),
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
        a = amostra(ALV, ZX0, ZY0, P_, xy[:, 0], xy[:, 1], ordem=0) > 0.5
        z = amostra(ZR, ZX0, ZY0, P_, xy[:, 0], xy[:, 1], ordem=0)
        j0 = len(ds) // 2
        if not a.any():
            est.append(None)
            continue
        idx = np.nonzero(a)[0]
        k = idx[np.argmin(np.abs(idx - j0))]
        if abs(ds[k]) > 2.5:
            est.append(None)
            continue
        lo = k
        while lo > 0 and a[lo - 1]:
            lo -= 1
        hi = k
        while hi < len(ds) - 1 and a[hi + 1]:
            hi += 1
        fora, dentro = ds[hi], ds[lo]
        zt = z[lo:hi + 1]
        pf, pd = q[i] + nl[i] * (fora + 1.5), q[i] + nl[i] * (dentro - 1.5)
        zf = float(amostra(ZR, ZX0, ZY0, P_, [pf[0]], [pf[1]], ordem=0)[0])
        zd = float(amostra(ZR, ZX0, ZY0, P_, [pd[0]], [pd[1]], ordem=0)[0])
        # parapeito: a faixa de 0,7 m junto à face de fora
        par = z[max(hi - 5, lo):hi + 1]
        est.append(dict(c=q[i].round(3).tolist(), n=nl[i].round(4).tolist(), fora=float(fora), dentro=float(dentro),
                        adarve=float(np.percentile(zt, 35)), topo=float(np.percentile(zt, 92)),
                        par_max=float(par.max()), par_min=float(par.min()),
                        chao_fora=zf, chao_dentro=zd))
    return est


def medir_torre(c0, zmin, raio):
    xs_ = np.arange(ZR.shape[1]) * P_ + ZX0
    ys_ = np.arange(ZR.shape[0]) * P_ + ZY0
    XX, YY = np.meshgrid(xs_, ys_)
    m0 = (ZR > zmin) & (np.hypot(XX - c0[0], YY - c0[1]) < raio)
    lab, n = ndi.label(m0)
    xs = (np.arange(ALTO.shape[1]) * P_ + ZX0)
    ys = (np.arange(ALTO.shape[0]) * P_ + ZY0)
    ci, ri = int((c0[0] - ZX0) / P_), int((c0[1] - ZY0) / P_)
    w = lab[max(ri - 8, 0):ri + 9, max(ci - 8, 0):ci + 9]
    v = np.bincount(w[w > 0].ravel())
    if not len(v) or v.max() == 0:
        return None
    k = int(np.argmax(v))
    m = ndi.binary_fill_holes(lab == k)
    rr, cc = np.nonzero(m)
    px, py = xs[cc], ys[rr]
    # retângulo robusto: para cada ângulo, extensão entre os percentis 1,5 e 98,5 das células projetadas; fica o de
    # menor área (o retângulo mínimo exato agarra-se a merlões e pedras soltas)
    melhor = None
    for a in np.radians(np.arange(0.0, 90.0, 0.5)):
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        pu, pv = np.c_[px, py] @ u, np.c_[px, py] @ v
        u0, u1 = np.percentile(pu, [1.5, 98.5])
        v0, v1 = np.percentile(pv, [1.5, 98.5])
        ar = (u1 - u0 + P_) * (v1 - v0 + P_)
        if melhor is None or ar < melhor[0]:
            melhor = (ar, u, v, u0 - P_ / 2, u1 + P_ / 2, v0 - P_ / 2, v1 + P_ / 2)
    _, u, v, u0, u1, v0, v1 = melhor
    xy = np.array([u * u0 + v * v0, u * u1 + v * v0, u * u1 + v * v1, u * u0 + v * v1])
    zz = ZR[rr, cc]
    g = CH[rr, cc]
    return dict(cantos=np.round(xy, 2).tolist(), topo=float(np.percentile(zz, 95)), meio=float(np.percentile(zz, 50)),
                terraco=float(np.percentile(zz, 25)), chao_local=float(np.median(g)))


# ---------------------------------------------------------------- cerca da vila
# Eixos dos troços que sobrevivem (polígonos «city_wall» do OpenStreetMap, conferidos no relevo do LiDAR), no sentido
# dos ponteiros do relógio à volta da vila (exterior à esquerda). Em estações a cada metro mede-se o topo (percentil
# 85 das células dentro do polígono a menos de 1 m da estação), a largura do polígono e o chão dos dois lados (MDT a
# 4 m da face).
CERCA = {
    "cerca_norte": [(-285.2, -88.6), (-279.0, -75.5), (-268.0, -63.0), (-218.5, -12.0), (-176.5, 29.0), (-154.2, 40.8),
                    (-134.5, 47.3), (-109.6, 50.5), (-104.3, 53.9), (-78.0, 51.3), (-52.4, 46.5)],
    "cerca_oeste": [(-247.0, -258.0), (-252.0, -240.0), (-266.8, -221.0), (-275.5, -204.5), (-285.3, -194.0),
                    (-301.3, -172.0), (-304.0, -166.8), (-306.3, -161.5), (-312.5, -149.0), (-305.5, -131.0),
                    (-300.3, -124.6)],
    "cerca_sudoeste": [(-199.0, -261.0), (-206.0, -258.5), (-241.0, -268.0), (-247.0, -258.0)],
    "cerca_porta": [(-134.2, -244.0), (-138.7, -242.2), (-149.5, -242.3), (-158.5, -243.3), (-165.6, -244.8)],
    "cerca_sul": [(-21.2, -214.0), (-29.0, -219.3), (-50.0, -227.5), (-79.5, -235.0), (-94.0, -241.3),
                  (-104.0, -244.5), (-112.8, -244.5)],
    "cerca_leste": [(16.4, -62.2), (5.8, -93.6)],
}
# torres da cerca (polígonos do OSM): cantos do retângulo e cotas do LiDAR
CERCA_TORRES = {
    "norte_1": [(-106.9, 53.9), (-101.9, 53.9), (-101.9, 64.4), (-106.9, 64.4)],
    "norte_2": [(-154.0, 46.1), (-156.2, 50.7), (-161.0, 48.3), (-158.7, 43.7)],
    "norte_3": [(-204.4, 14.5), (-207.6, 10.9), (-204.3, 8.1), (-201.4, 11.5)],
    "norte_4": [(-217.3, -10.8), (-220.1, -8.5), (-222.4, -11.0), (-219.7, -13.6)],
    "norte_5": [(-239.9, -27.8), (-242.0, -30.0), (-239.6, -32.2), (-237.5, -30.0)],
    "oeste_1": [(-285.2, -202.9), (-290.1, -206.1), (-286.7, -211.0), (-282.0, -207.6)],
    "oeste_2": [(-312.2, -168.1), (-320.0, -171.2), (-315.6, -177.4), (-308.9, -172.4)],
    "sudoeste_1": [(-236.9, -276.7), (-233.5, -277.3), (-225.7, -273.3), (-224.7, -268.5), (-238.0, -266.5)],
    "sudoeste_2": [(-202.3, -262.3), (-199.8, -269.3), (-194.0, -267.2), (-196.7, -259.8)],
    "sul_1": [(-106.7, -243.9), (-106.3, -251.6), (-101.3, -250.4), (-102.5, -244.0)],
    "porta_cidade": [(-149.8, -243.3), (-149.4, -246.7), (-145.9, -263.3), (-134.5, -261.5), (-136.0, -245.7), (-136.1, -243.6)],
    "leste_1": [(20.1, -121.9), (15.0, -134.5), (20.8, -136.0), (25.7, -123.2)],
    "leste_2": [(5.7, -165.6), (4.8, -172.5), (11.5, -173.4), (12.4, -166.5)],
    "leste_3": [(-0.3, -195.4), (-5.3, -202.5), (2.6, -206.2), (6.3, -197.9)],
}


def medir_cerca(pts):
    q = densificar(pts, 1.0)
    t = np.gradient(q, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    nl = np.stack([-t[:, 1], t[:, 0]], 1)
    est = []
    for i in range(len(q)):
        ds = np.arange(-2.0, 2.01, 0.25)
        lin = q[i][None, :] + nl[i][None, :] * ds[:, None]
        zz = []
        for dl in (-0.5, 0.0, 0.5):
            p = lin + t[i][None, :] * dl
            zz.append(amostra(Z, X0_, Y0_, P_, p[:, 0], p[:, 1], ordem=0))
        zz = np.concatenate(zz)
        pf, pd = q[i] + nl[i] * 4.0, q[i] - nl[i] * 4.0
        zf = float(amostra(MDT, LX0, LY0, LP, [pf[0]], [pf[1]])[0])
        zd = float(amostra(MDT, LX0, LY0, LP, [pd[0]], [pd[1]])[0])
        ze = float(amostra(MDT, LX0, LY0, LP, [q[i][0]], [q[i][1]])[0])
        est.append(dict(c=q[i].round(3).tolist(), n=nl[i].round(4).tolist(), topo=float(np.percentile(zz, 85)),
                        chao_fora=zf, chao_dentro=zd, chao_eixo=ze))
    return est


def medir_torre_pol(pol):
    P = shapely.Polygon(pol).buffer(0)
    r = shapely.minimum_rotated_rectangle(P)
    x0, y0, x1, y1 = P.bounds
    gx, gy = np.meshgrid(np.arange(x0, x1, P_), np.arange(y0, y1, P_))
    d = shapely.contains_xy(P.buffer(-0.5), gx, gy)
    zz = amostra(Z, X0_, Y0_, P_, gx[d], gy[d], ordem=0)
    zt = amostra(MDT, LX0, LY0, LP, gx[d], gy[d])
    return dict(cantos=np.round(np.array(r.exterior.coords)[:-1], 2).tolist(), poligono=pol,
                topo=float(np.percentile(zz, 90)), meio=float(np.percentile(zz, 50)), chao=float(np.percentile(zt, 10)))


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
    for k, (c, zmin, raio) in TORRES0.items():
        t = medir_torre(c, zmin, raio)
        out["torres"][k] = t
        if t:
            xy = np.array(t["cantos"])
            e = np.diff(np.vstack([xy, xy[:1]]), axis=0)
            print(f"torre {k:16s} {np.hypot(*e[0]):5.2f} x {np.hypot(*e[1]):5.2f}  az {math.degrees(math.atan2(e[0][0], e[0][1])) % 180:5.1f}  "
                  f"topo {t['topo']:.2f} terraço {t['terraco']:.2f} chão {t['chao_local']:.2f}")
        else:
            print("torre", k, "não encontrada")
    out["cerca"], out["cerca_torres"] = {}, {}
    for k, pts in CERCA.items():
        e = medir_cerca(pts)
        out["cerca"][k] = {"eixo": pts, "estacoes": e}
        tp = np.array([x["topo"] for x in e])
        print(f"{k:16s} {len(e):3d} est.  topo {np.percentile(tp, 25):.1f}/{np.median(tp):.1f}/{np.percentile(tp, 75):.1f}  "
              f"fora {np.median([x['chao_fora'] for x in e]):.1f} dentro {np.median([x['chao_dentro'] for x in e]):.1f}")
    for k, pol in CERCA_TORRES.items():
        t = medir_torre_pol(pol)
        out["cerca_torres"][k] = t
        print(f"torre {k:14s} topo {t['topo']:.2f} meio {t['meio']:.2f} chão {t['chao']:.2f}")
    (AQUI / "dados" / "muralhas.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
