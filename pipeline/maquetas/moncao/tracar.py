"""Fortaleza de Monção: traça no MDT da DGT as escarpas de alvenaria (os panos dos baluartes, cortinas,
revelins e contraguardas e as contraescarpas do fosso) e grava dados/escarpas.json, que o construir.py lê.

Uma praça abaluartada é feita de terra revestida de cantaria: o MDT (só solo) apanha os terraplenos, os
parapeitos, o fosso, a estrada coberta e a esplanada, e as escarpas aparecem como faixas quase verticais.
Aqui as faixas de declive forte (> 58°) com mais de 2 m de desnível numa janela de 3 m reduzem-se a uma
linha (esqueleto), que se parte em troços entre cruzamentos e se simplifica (Douglas-Peucker, 0,6 m), para
as faces saírem retas como as de cantaria. Cada troço fica orientado com o fosso (o lado baixo) à direita,
e em cada vértice mede-se o topo (lado alto) e o pé (lado baixo) no MDT.

Só entram os troços na faixa das fortificações (perto das linhas de muralha do OSM) e com pelo menos 8 m.

Uso:
    python pipeline/maquetas/moncao/tracar.py
"""
import json
import math
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, skeletonize

AQUI = Path(__file__).resolve().parent
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
LAT0, LON0 = F["origem"]["lat"], F["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([0.0, 0.0])  # igual ao lidar.py e ao construir.py

L = np.load(AQUI / "dados" / "lidar.npz")
PASSO = float(L["passo"])
MDT0 = L["mdt"] / 100.0 + float(L["base"])
MDS = L["mds"] / 100.0 + float(L["base"])
CLS = L["classe"]
XS = np.linspace(-float(L["meiox"]), float(L["meiox"]), MDT0.shape[1])
YS = np.linspace(-float(L["meioy"]), float(L["meioy"]), MDT0.shape[0])


def ll(c):
    return np.array([(c[0] - LON0) * KX, (c[1] - LAT0) * KY]) - CENTRO


def amostra(G, x, y):
    c = (np.asarray(x, float) - XS[0]) / PASSO
    r = (np.asarray(y, float) - YS[0]) / PASSO
    c0 = np.clip(np.floor(c).astype(int), 0, G.shape[1] - 2)
    r0 = np.clip(np.floor(r).astype(int), 0, G.shape[0] - 2)
    fc, fr = np.clip(c - c0, 0, 1), np.clip(r - r0, 0, 1)
    return (G[r0, c0] * (1 - fc) * (1 - fr) + G[r0, c0 + 1] * fc * (1 - fr)
            + G[r0 + 1, c0] * (1 - fc) * fr + G[r0 + 1, c0 + 1] * fc * fr)


# Socalcos e taludes que não são muralha (lidos na ortoimagem e no relevo): a encosta do rio a norte da
# Porta de Salvaterra e abaixo da frente norte, e dois taludes de estrada a poente e a nascente.
EXCLUIR = [shapely.Polygon(p) for p in (
    [(-258, 262), (-150, 262), (-150, 212), (-200, 232), (-230, 214), (-258, 214)],
    [(-85, 205), (-15, 205), (-15, 177), (-85, 181)],
    [(-385, -40), (-355, -40), (-355, -65), (-385, -65)],
    [(295, -45), (325, -45), (325, -70), (295, -70)],
    [(-86, 36), (-70, 36), (-70, 46), (-86, 46)],  # degraus da Misericórdia, no largo
    # encosta norte, entre a muralha da vila e o rio: socalcos e o passadiço de madeira em ziguezague
    # (ortofotomapa da DGT no WebSIG da Câmara), não são panos
    [(-104, 192), (-50, 168), (-13, 180), (5, 170), (57, 152), (85, 143), (90, 150), (90, 250), (-104, 250)],
)]


def edificios_mascara():
    """Plantas das casas (OSM e LiDAR), alargadas 1 m: os beirados não são escarpas."""
    pl = []
    for f in F["edificios"]:
        g = f["geometria"]
        if g["type"] == "Polygon":
            pl.append(shapely.Polygon([ll(c) for c in g["coordinates"][0]]))
        elif g["type"] == "MultiPolygon":
            pl += [shapely.Polygon([ll(c) for c in q[0]]) for q in g["coordinates"]]
    pl += [shapely.Polygon(c["planta"]) for c in json.loads((AQUI / "dados" / "casas.json").read_text())]
    u = shapely.union_all([q.buffer(1.0) for q in pl if q.is_valid])
    X, Y = np.meshgrid(XS, YS)
    return shapely.contains_xy(u, X, Y)


def superficie(zona_):
    """Em Monção os parapeitos e muitas faces vêm na classe 1 (não classificado) ou 6 da nuvem de pontos, e o
    MDT passa por baixo deles. Dentro da zona das fortificações e fora das casas usa-se o ponto mais alto das
    classes 1, 2 e 6 (dados/superficie.npz, do pontos.py); no resto fica o MDT."""
    S = np.load(AQUI / "dados" / "superficie.npz")
    sup = S["sup"].astype(float)
    tem = sup > -32000
    sup = np.where(tem, sup / 100.0, -1e9)
    X, Y = np.meshgrid(XS, YS)
    dentro = shapely.contains_xy(zona_, X, Y) & ~edificios_mascara()
    return np.where(dentro & tem, np.maximum(sup, MDT0), MDT0)


def zona():
    """Zona das fortificações: o polígono que envolve as linhas de muralha do OSM, alargado 40 m (em Monção
    as linhas do OSM têm falhas, por isso não chega uma faixa à volta delas), sem os socalcos do rio e os
    cortes das estradas (EXCLUIR)."""
    linhas = []
    for f in F["infraestrutura"]:
        if f["classe"] == "city_wall" and f["geometria"]["type"] == "LineString":
            linhas.append(shapely.LineString([ll(c) for c in f["geometria"]["coordinates"]]))
    u = shapely.union_all(linhas).convex_hull.buffer(40)
    u = u.intersection(shapely.box(-480, -290, 400, 290))  # a leste de x = 400 são socalcos e quintas
    for p in EXCLUIR:
        u = u.difference(p)
    return u, linhas


def tracos(sk):
    """Parte o esqueleto em troços entre pontas e cruzamentos (vizinhança de 8)."""
    viz = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    viz = np.where(sk, viz, 0)
    nos = sk & (viz != 2)
    visto = np.zeros_like(sk)
    R, C = sk.shape
    passos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    out = []

    def seguir(r, c, pr, pc):
        cam = [(r, c), (pr, pc)]
        visto[pr, pc] = True
        while not nos[cam[-1]]:
            r0, c0 = cam[-1]
            nxt = None
            for dr, dc in passos:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < R and 0 <= cc < C and sk[rr, cc] and (rr, cc) != cam[-2] and not (visto[rr, cc] and not nos[rr, cc]):
                    if (rr, cc) in cam[-3:]:
                        continue
                    nxt = (rr, cc)
                    break
            if nxt is None:
                break
            if not nos[nxt]:
                visto[nxt] = True
            cam.append(nxt)
        return cam

    for r, c in zip(*np.nonzero(nos)):
        for dr, dc in passos:
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C and sk[rr, cc] and not visto[rr, cc] and not nos[rr, cc]:
                out.append(seguir(r, c, rr, cc))
    # anéis fechados sem nós
    for r, c in zip(*np.nonzero(sk & ~visto & ~nos)):
        if visto[r, c]:
            continue
        visto[r, c] = True
        for dr, dc in passos:
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C and sk[rr, cc] and not visto[rr, cc]:
                out.append(seguir(r, c, rr, cc))
                break
    return out


def juntar(linhas, tol=3.0):
    """Junta troços que se tocam pelas pontas (o esqueleto parte-se nos esporões das esquinas)."""
    linhas = [list(map(tuple, l)) for l in linhas]
    mudou = True
    while mudou:
        mudou = False
        for i in range(len(linhas)):
            for j in range(len(linhas)):
                if i == j:
                    continue
                a, b = linhas[i], linhas[j]
                for (pa, ra), (pb, rb) in (((a[-1], False), (b[0], False)), ((a[-1], False), (b[-1], True)),
                                           ((a[0], True), (b[0], False)), ((a[0], True), (b[-1], True))):
                    if math.dist(pa, pb) < tol:
                        aa = a[::-1] if ra else a
                        bb = b[::-1] if rb else b
                        # só junta se não fizer um cotovelo para trás (ângulo > 30°)
                        t1 = np.subtract(aa[-1], aa[max(0, len(aa) - 4)])
                        t2 = np.subtract(bb[min(3, len(bb) - 1)], bb[0])
                        if np.linalg.norm(t1) > 0 and np.linalg.norm(t2) > 0 and t1 @ t2 / np.linalg.norm(t1) / np.linalg.norm(t2) < -0.5:
                            continue
                        linhas[i] = aa + bb[1:]
                        linhas.pop(j)
                        mudou = True
                        break
                if mudou:
                    break
            if mudou:
                break
    return linhas


def afiar(lin, minimo=3.5):
    """Os ângulos dos baluartes saem arredondados no esqueleto (e com um troço curto depois do
    Douglas-Peucker): troca cada troço curto pela interseção dos dois troços vizinhos, para a esquina
    ficar viva, como na cantaria."""
    lin = [np.asarray(p, float) for p in lin]
    mudou = True
    while mudou and len(lin) > 3:
        mudou = False
        segs = [np.linalg.norm(lin[i + 1] - lin[i]) for i in range(len(lin) - 1)]
        for i in sorted(range(1, len(lin) - 2), key=lambda k: segs[k]):
            if segs[i] >= minimo:
                break
            p1, p2, p3, p4 = lin[i - 1], lin[i], lin[i + 1], lin[i + 2]
            d1, d2 = p2 - p1, p4 - p3
            den = d1[0] * d2[1] - d1[1] * d2[0]
            meio = (p2 + p3) / 2
            q = meio
            if abs(den) > 1e-6:
                t = ((p3 - p1)[0] * d2[1] - (p3 - p1)[1] * d2[0]) / den
                x = p1 + d1 * t
                if np.linalg.norm(x - meio) < 4.0:
                    q = x
            lin[i:i + 2] = [q]
            mudou = True
            break
    return np.array(lin)


def esquinas(linhas, raio=7.0):
    """Nos ângulos salientes o esqueleto parte-se muitas vezes em dois troços que não se tocam: prolongam-se
    as pontas até à interseção das duas retas e juntam-se num só traço, com a esquina viva."""
    linhas = [np.asarray(l, float) for l in linhas]
    mudou = True
    while mudou:
        mudou = False
        melhor = None
        for i in range(len(linhas)):
            for j in range(len(linhas)):
                if i == j:
                    continue
                a, b = linhas[i], linhas[j]
                for ra in (False, True):
                    for rb in (False, True):
                        aa = a[::-1] if ra else a
                        bb = b[::-1] if rb else b
                        pa, pb = aa[-1], bb[0]
                        dist = np.linalg.norm(pa - pb)
                        if dist > raio:
                            continue
                        d1 = unit_(aa[-1] - aa[-2])
                        d2 = unit_(bb[1] - bb[0])
                        den = d1[0] * d2[1] - d1[1] * d2[0]
                        if abs(den) < 0.2:  # quase paralelos: junta só se estiverem alinhados e perto
                            if dist < 2.0 and d1 @ d2 > 0.9:
                                cand = (dist, i, j, ra, rb, None)
                            else:
                                continue
                        else:
                            t = ((pb - pa)[0] * d2[1] - (pb - pa)[1] * d2[0]) / den
                            u = ((pb - pa)[0] * d1[1] - (pb - pa)[1] * d1[0]) / den
                            # a interseção tem de ficar à frente da ponta de a e atrás da de b
                            if not (-1.5 < t < 6.0 and -6.0 < u < 1.5):
                                continue
                            cand = (dist, i, j, ra, rb, pa + d1 * t)
                        if melhor is None or cand[0] < melhor[0]:
                            melhor = cand
        if melhor is not None:
            _, i, j, ra, rb, x = melhor
            aa = linhas[i][::-1] if ra else linhas[i]
            bb = linhas[j][::-1] if rb else linhas[j]
            nova = np.vstack([aa[:-1], [x], bb[1:]]) if x is not None else np.vstack([aa, bb])
            linhas = [l for k, l in enumerate(linhas) if k not in (i, j)] + [afiar(nova)]
            mudou = True
    return linhas


def partir_lado(lin, minimo=6.0):
    """Um traço juntado pode trocar de lado baixo a meio (duas escarpas encostadas que o esqueleto ligou):
    parte-se onde o lado baixo muda, para cada troço ficar com o fosso sempre do mesmo lado."""
    g = shapely.LineString(lin)
    n = max(2, int(math.ceil(g.length / 1.0)) + 1)
    pts = np.array([g.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)])
    t = np.gradient(pts, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nr = np.stack([t[:, 1], -t[:, 0]], 1)
    zd = np.mean([amostra(MDT, *(pts + nr * d).T) for d in (1.5, 2.5, 3.5)], axis=0)
    ze = np.mean([amostra(MDT, *(pts - nr * d).T) for d in (1.5, 2.5, 3.5)], axis=0)
    sg = np.sign(ndi.median_filter(ze - zd, 7, mode="nearest"))
    cortes = [0] + [i for i in range(1, n) if sg[i] != sg[i - 1]] + [n]
    partes = []
    for i0, i1 in zip(cortes[:-1], cortes[1:]):
        if (i1 - i0) >= minimo:
            q = shapely.LineString(pts[i0:i1]).simplify(0.6)
            partes.append(afiar(np.array(q.coords)))
    return partes or [lin]


def unit_(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def perto(G, x, y):
    c = np.clip(np.round((np.asarray(x, float) - XS[0]) / PASSO).astype(int), 0, G.shape[1] - 1)
    r = np.clip(np.round((np.asarray(y, float) - YS[0]) / PASSO).astype(int), 0, G.shape[0] - 1)
    return G[r, c]


def plataforma(pts, topo):
    """Fração do troço que tem por cima um terrapleno ou um parapeito: pontos de solo (classe 2) até 2,5 m
    abaixo do topo, ou da classe 1 até 1,5 m abaixo, entre 1,5 e 6 m para dentro. Distingue as muralhas
    (com adarve e parapeito) das fachadas e dos beirados das casas, que só têm telhado por cima."""
    S = np.load(AQUI / "dados" / "superficie.npz")
    solo = np.where(S["solo"] > -32000, S["solo"] / 100.0, -1e9)
    c1 = np.where(S["c1"] > -32000, S["c1"] / 100.0, -1e9)
    t = np.gradient(pts, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nr = np.stack([t[:, 1], -t[:, 0]], 1)
    ok = np.zeros(len(pts), bool)
    for d in np.arange(-1.5, -6.01, -0.5):
        q = pts + nr * d
        ok |= (perto(solo, *q.T) >= topo - 2.5) | (perto(c1, *q.T) >= topo - 1.5)
    return float(ok.mean())


def medir(linha, virado=False):
    """Orienta o troço com o lado baixo à direita e mede em cada vértice (densificado a 1 m) o topo, o pé,
    e onde começa a escarpa por cima (aresta) e acaba por baixo, a partir da linha do esqueleto."""
    g = shapely.LineString(linha)
    n = max(2, int(math.ceil(g.length / 1.0)) + 1)
    pts = np.array([g.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)])
    t = np.gradient(pts, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nr = np.stack([t[:, 1], -t[:, 0]], 1)  # à direita
    d = np.arange(-6.0, 8.01, 0.25)
    xy = pts[:, None, :] + nr[:, None, :] * d[None, :, None]
    z = amostra(MDT, xy[..., 0], xy[..., 1])
    esq = z[:, d < -1.0].mean()
    dir_ = z[:, d > 1.0].mean()
    if dir_ > esq and not virado:  # o lado baixo tem de ficar à direita
        return medir(linha[::-1], True)
    i0 = int(np.argmin(np.abs(d)))
    topo = np.zeros(n)
    pe = np.zeros(n)
    aresta = np.zeros(n)
    fim = np.zeros(n)
    for k in range(n):
        zz = z[k]
        # topo: o mais alto até 6 m para dentro (o parapeito de terra é largo); pé: o mais baixo entre
        # 0,5 e 6 m para fora
        lado_alto = zz[(d <= 0) & (d >= -6.0)]
        topo[k] = lado_alto.max()
        sel = (d >= 0.5) & (d <= 6.0)
        pe[k] = np.percentile(zz[sel], 15)
        h = topo[k] - pe[k]
        # aresta: primeiro ponto (de dentro para fora) abaixo do topo - 6 % da altura
        ii = np.nonzero((d <= 1.5) & (zz < topo[k] - max(0.25, 0.06 * h)) & (d >= -4.0))[0]
        aresta[k] = d[ii[0]] if len(ii) else 0.0
        jj = np.nonzero((d >= aresta[k]) & (zz <= pe[k] + max(0.2, 0.05 * h)))[0]
        fim[k] = d[jj[0]] if len(jj) else 2.0
    return pts, topo, pe, aresta, fim


def main():
    global MDT
    global LINHAS_OSM
    z_, linhas_osm = zona()
    LINHAS_OSM = shapely.union_all(linhas_osm)
    MDT = superficie(z_)
    gy, gx = np.gradient(ndi.gaussian_filter(MDT, 0.6), PASSO)
    decl = np.degrees(np.arctan(np.hypot(gx, gy)))
    dz = ndi.maximum_filter(MDT, 7) - ndi.minimum_filter(MDT, 7)
    S = (decl > 58) & (dz > 2.0)
    S = remove_small_objects(S, max_size=40)
    sk = skeletonize(ndi.binary_closing(S, iterations=2))
    brutos = []
    for cam in tracos(sk):
        if len(cam) < 6:
            continue
        r, c = np.array(cam).T
        brutos.append(np.stack([XS[c], YS[r]], 1))
    # tira os esporões curtos e junta o resto
    brutos = [b for b in brutos if shapely.LineString(b).length >= 3.0]
    juntos = juntar(brutos)
    linhas = []
    for lin in juntos:
        g = shapely.LineString(lin).simplify(0.6)
        if g.length < 8.0:
            continue
        if g.intersection(z_).length < 0.6 * g.length:
            continue
        if any(g.intersection(p).length > 0.5 * g.length for p in EXCLUIR):
            continue
        linhas.append(afiar(np.array(g.coords)))
    linhas = esquinas(linhas)
    linhas = [p for l in linhas for p in partir_lado(l)]
    out = []
    for lin in linhas:
        g = shapely.LineString(lin)
        pts, topo, pe, aresta, fim = medir(lin)
        h = topo - pe
        if np.median(h) < 2.2:
            continue
        pl = plataforma(pts, topo)
        osm = g.buffer(6.0).intersection(LINHAS_OSM).length / max(g.length, 1e-6)
        if pl < 0.4 and osm < 0.5:
            continue
        # o traço final: os vértices do troço simplificado, com as medidas filtradas (mediana de 5 m)
        k = 5

        def suave(v):
            return np.array([np.median(v[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(v))])
        topo, pe, aresta, fim = suave(topo), suave(pe), suave(aresta), suave(fim)
        simp = afiar(np.array(shapely.LineString(pts).simplify(0.6).coords))
        idx = [int(np.argmin(np.hypot(*(pts - q).T))) for q in simp]
        out.append({
            "linha": np.round(simp, 2).tolist(),
            "topo": np.round(topo[idx], 2).tolist(),
            "pe": np.round(pe[idx], 2).tolist(),
            "aresta": np.round(aresta[idx], 2).tolist(),
            "fim": np.round(fim[idx], 2).tolist(),
            "comprimento": round(float(g.length), 1),
            "altura_mediana": round(float(np.median(h)), 2),
            "plataforma": round(pl, 2),
        })
    out.sort(key=lambda e: -e["comprimento"])
    for i, e in enumerate(out):
        e["id"] = f"e{i:03d}"
    (AQUI / "dados" / "escarpas.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("escarpas", len(out), "comprimento total", round(sum(e["comprimento"] for e in out)), "m",
          "altura mediana", round(float(np.median([e["altura_mediana"] for e in out])), 1), "m")


if __name__ == "__main__":
    main()
