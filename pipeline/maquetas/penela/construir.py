"""Constrói a maqueta 3D do Castelo de Penela (estado atual) em Blender e exporta-a
para public/maquetas/penela/penela.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy
    python pipeline/maquetas/penela/construir.py            # GLB
    python pipeline/maquetas/penela/construir.py --vistas   # GLB + capa + vistas de verificação

Dados: dados/fontes.json (recolher.py: traçado do OSM), dados/lidar.npz (lidar.py: MDT, MDS,
classes da nuvem de pontos e copas, do LiDAR 2024 da DGT) e src/data/maquetas/penela.json, que tem
a leitura das fontes e o estado de cada elemento. As cotas e alturas medem-se no LiDAR; o que
o LiDAR não mede (espessura, ameias, coruchéu, vãos) vem do JSON e das fotografias.
Coordenadas da cena: x para este, y para norte, z para cima, em metros (1:1).
O glTF converte para y para cima (-z é o norte no visor).
"""
import json
import math
import sys
from pathlib import Path

import bpy  # noqa: I001 (o bpy tem de vir antes do bmesh)
import bmesh
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "penela.json").read_text())
EL = {e["id"]: e for e in CFG["elementos"]}
SAIDA = RAIZ / "public" / "maquetas" / "penela"

LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
# Centro da maqueta em metros a partir da origem oficial (centro do recinto amuralhado)
CENTRO = np.array([-40.0, 0.0])
RAIO = 150.0
ZREF = 250.0  # cota que fica em z = 0
rng = np.random.default_rng(1064)

MUR = CFG["muralha"]
PARAPEITO, MERLAO = MUR["parapeito"], MUR["merlao"]


# ---------------------------------------------------------------- geografia

def P(x, y):
    """Metros a partir da origem oficial -> coordenadas da cena."""
    return np.array([x, y], dtype=float) - CENTRO


def ll(c):
    return np.array([(c[0] - LON0) * KX, (c[1] - LAT0) * KY]) - CENTRO


def geom(f):
    g = f["geometria"]
    if g["type"] == "Polygon":
        return np.array([ll(c) for c in g["coordinates"][0][:-1]])
    if g["type"] == "LineString":
        return np.array([ll(c) for c in g["coordinates"]])
    if g["type"] == "Point":
        return ll(g["coordinates"])
    return None


def osm(prefixo, lista="infraestrutura"):
    return next(f for f in F[lista] if f["id"].startswith(prefixo))


# LiDAR da DGT reamostrado para a grelha da cena (lidar.py): linha 0 = sul, coluna 0 = oeste
LID = np.load(AQUI / "dados" / "lidar.npz")
LPASSO, LMEIO = float(LID["passo"]), float(LID["meio"])
MDT = LID["mdt"] / 100.0 + float(LID["base"]) - ZREF
MDS = LID["mds"] / 100.0 + float(LID["base"]) - ZREF
CLS = LID["classe"]
VEG = (CLS >= 3) & (CLS <= 5)
LXS = np.linspace(-LMEIO, LMEIO, MDT.shape[0])


def amostra(G, x, y):
    """Interpolação bilinear numa grelha do LiDAR, em z da cena."""
    c = (np.asarray(x, float) + LMEIO) / LPASSO
    r = (np.asarray(y, float) + LMEIO) / LPASSO
    c0 = np.clip(np.floor(c).astype(int), 0, G.shape[1] - 2)
    r0 = np.clip(np.floor(r).astype(int), 0, G.shape[0] - 2)
    fc, fr = np.clip(c - c0, 0, 1), np.clip(r - r0, 0, 1)
    return (G[r0, c0] * (1 - fc) * (1 - fr) + G[r0, c0 + 1] * fc * (1 - fr)
            + G[r0 + 1, c0] * (1 - fc) * fr + G[r0 + 1, c0 + 1] * fc * fr)


def cota(x, y):
    """Terreno: MDT LiDAR de 50 cm (solo sem edifícios nem vegetação)."""
    return amostra(MDT, x, y)


def celulas(poly=None, centro=None, raio=None, sem_veg=True):
    """Valores do MDS nas células dentro de um polígono ou de um círculo (sem copas de árvores)."""
    if poly is not None:
        x0, y0 = poly.min(0) - 1
        x1, y1 = poly.max(0) + 1
    else:
        x0, y0 = centro - raio - 1
        x1, y1 = centro + raio + 1
    c0, c1 = np.searchsorted(LXS, [x0, x1])
    r0, r1 = np.searchsorted(LXS, [y0, y1])
    X, Y = np.meshgrid(LXS[c0:c1], LXS[r0:r1])
    ok = dentro(X, Y, poly) if poly is not None else np.hypot(X - centro[0], Y - centro[1]) <= raio
    if sem_veg:
        ok &= ~VEG[r0:r1, c0:c1]
    return MDS[r0:r1, c0:c1][ok]


def medir(poly=None, centro=None, raio=None, q=95):
    v = celulas(poly, centro, raio)
    return float(np.percentile(v, q)) if len(v) >= 3 else float("nan")


def dentro(px, py, poly):
    px, py = np.asarray(px, float), np.asarray(py, float)
    res = np.zeros(px.shape, bool)
    x, y = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = ((y[i] > py) != (y[j] > py)) & (px < (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i] + 1e-12) + x[i])
        res ^= cond
        j = i
    return res


def distancia(px, py, linha, fechada=False):
    """Distância mínima de pontos a uma polilinha e o ponto mais próximo."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    pts = np.vstack([linha, linha[:1]]) if fechada else linha
    best = np.full(px.shape, np.inf)
    qx, qy = np.zeros(px.shape), np.zeros(px.shape)
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        t = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / max(d @ d, 1e-9), 0, 1)
        cx, cy = a[0] + t * d[0], a[1] + t * d[1]
        dd = np.hypot(px - cx, py - cy)
        m = dd < best
        best[m], qx[m], qy[m] = dd[m], cx[m], cy[m]
    return best, qx, qy


def area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# ---------------------------------------------------------------- planta

TORRES = {
    "torreao_poente_1": "2536ee3e",
    "torreao_poente_2": "51630ee0",
    "torreao_poente_3": "79fb475c",
    "porta_traicao": "f2a8d2cb",
    "torre_quinaria": "507542ce",
}
torres = {k: geom(osm(v)) for k, v in TORRES.items()}

principal = geom(osm("e8b44ca9"))  # do castelejo, por norte, poente e sul, até à Brecha
sueste = geom(osm("06f1dca5"))  # da Brecha (torre quinária) até ao castelejo
castelejo_int = geom(osm("9501d9c4"))  # muro do castelejo virado ao recinto
# Face exterior do castelejo: não está no OSM; fecha-se pela planta do edifício 0de35856.
castelejo_ext = np.array([P(0.4, -2.0), P(0.6, 5.1)])
castelejo = np.vstack([castelejo_int, castelejo_ext])
if area(castelejo) < 0:
    castelejo = castelejo[::-1]


def sem_torres(linha):
    keep = []
    for p in linha:
        perto = any(distancia(np.array([p[0]]), np.array([p[1]]), t, True)[0][0] < 0.05 or dentro(p[0], p[1], t) for t in torres.values())
        keep.append(not perto)
    keep[0] = keep[-1] = True
    return linha[np.array(keep)]


def perto_de(linha, alvo):
    return int(np.argmin(np.hypot(*(linha - alvo).T)))


principal_l = sem_torres(principal)
sueste_l = sem_torres(sueste)
i_trai = perto_de(principal_l, P(-44, 54))
i_cub = perto_de(principal_l, P(-85.7, 30.4))
i_sw = perto_de(principal_l, P(-49.3, -42.7))
SECTORES = {
    "pano_norte": principal_l[: i_trai + 1],
    "pano_noroeste": principal_l[i_trai: i_cub + 1],
    "pano_poente": principal_l[i_cub: i_sw + 1],
    "pano_sul": principal_l[i_sw:],
    "pano_nascente": sueste_l,
}
perimetro = np.vstack([principal, sueste, castelejo_ext])
if area(perimetro) < 0:
    raise SystemExit("perímetro devia estar no sentido anti-horário")

PORTA_VILA = geom(osm("1e35ebc6"))
BRECHA = geom(osm("28537527"))
CUBELO = P(-84.0, 31.4)

# Muro de suporte do terreiro da igreja (fotografia aérea 1; o degrau confere-se no MDT)
MURO_TERREIRO = np.array([P(-47, 46), P(-47, 24), P(-43, 8), P(-34, 0), P(-20, -3), P(-10.6, -1)])


# ---------------------------------------------------------------- terreno


def ruido(x, y, escala, semente):
    g = np.random.default_rng(semente).random((64, 64))
    u, v = x / escala, y / escala
    i, j = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i, v - j
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a, b = g[i % 64, j % 64], g[(i + 1) % 64, j % 64]
    c, d = g[i % 64, (j + 1) % 64], g[(i + 1) % 64, (j + 1) % 64]
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def fbm(x, y, escala, semente, oit=4):
    s, amp, tot = 0.0, 1.0, 0.0
    for o in range(oit):
        s = s + amp * ruido(x, y, escala / 2**o, semente + o)
        tot += amp
        amp *= 0.5
    return s / tot


_ORTO = []


def ortofoto(x, y):
    """Cor (sRGB 0-1) da ortofoto DGT 2025 recortada em dados/orto.jpg (norte em cima)."""
    if not _ORTO:
        img = bpy.data.images.load(str(AQUI / "dados" / "orto.jpg"))
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        _ORTO.append(px.reshape(h, w, 4)[..., :3])  # o Blender guarda a linha 0 em baixo (sul)
        bpy.data.images.remove(img)
    a = _ORTO[0]
    k = a.shape[0]
    c = np.clip(np.round((np.asarray(x) + LMEIO) / (2 * LMEIO) * (k - 1)).astype(int), 0, k - 1)
    r = np.clip(np.round((np.asarray(y) + LMEIO) / (2 * LMEIO) * (k - 1)).astype(int), 0, k - 1)
    return a[r, c].astype(float)


def construir_terreno():
    passo = 1.0
    n = int(2 * RAIO / passo) + 1
    xs = np.linspace(-RAIO, RAIO, n)
    X, Y = np.meshgrid(xs, xs)
    Z = cota(X, Y)
    gy, gx = np.gradient(Z, passo)
    decl = np.degrees(np.arctan(np.hypot(gx, gy)))

    centro_in = np.hypot(X[:-1, :-1] + passo / 2, Y[:-1, :-1] + passo / 2) < RAIO - passo * 0.3
    idx = -np.ones(X.shape, int)
    usados = np.zeros(X.shape, bool)
    usados[:-1, :-1] |= centro_in
    usados[1:, :-1] |= centro_in
    usados[:-1, 1:] |= centro_in
    usados[1:, 1:] |= centro_in
    idx[usados] = np.arange(usados.sum())
    V = np.stack([X[usados], Y[usados], Z[usados]], 1)
    I, J = np.nonzero(centro_in)
    faces = np.stack([idx[I, J], idx[I, J + 1], idx[I + 1, J + 1], idx[I + 1, J]], 1)

    # cores por vértice (sRGB): paleta de maqueta guiada pela ortofoto de 2025 (relva, terra,
    # calçada e rocha onde de facto estão), com as sombras e o ruído da fotografia atenuados
    xv, yv = V[:, 0], V[:, 1]
    dv = decl[usados]
    orto = ortofoto(xv, yv)
    lum = orto.mean(1)
    sat = orto.max(1) - orto.min(1)
    verde = np.clip((orto[:, 1] - (orto[:, 0] + orto[:, 2]) / 2) * 8 + 0.3, 0, 1)
    # debaixo das copas (nuvem de pontos) a ortofoto só mostra a árvore e a sombra: chão de mato
    ci = np.clip(np.round((xv + LMEIO) / LPASSO).astype(int), 0, VEG.shape[1] - 1)
    ri = np.clip(np.round((yv + LMEIO) / LPASSO).astype(int), 0, VEG.shape[0] - 1)
    sob = np.zeros(len(xv))
    for dr in (-2, 0, 2):
        for dc in (-2, 0, 2):
            sob += VEG[np.clip(ri + dr, 0, VEG.shape[0] - 1), np.clip(ci + dc, 0, VEG.shape[1] - 1)]
    sob = np.clip(sob / 9 * 1.5, 0, 1)
    verde = np.maximum(verde, sob * 0.85)
    erva = np.array([0.52, 0.55, 0.35])
    terra = np.array([0.76, 0.70, 0.57])
    calcada = np.array([0.80, 0.78, 0.73])
    cor = erva * verde[:, None] + terra * (1 - verde[:, None])
    claro = np.clip((lum - 0.55) * 6, 0, 1) * np.clip((0.12 - sat) * 12, 0, 1) * (1 - sob)
    cor = cor * (1 - claro[:, None]) + calcada * claro[:, None]
    cor = cor * (0.94 + 0.12 * fbm(xv, yv, 6.0, 3))[:, None]
    roc = np.array([0.76, 0.69, 0.55]) * (0.9 + 0.2 * fbm(xv, yv, 4.0, 5)[:, None])
    tr = np.clip((dv - 38) / 12, 0, 1)[:, None]
    cor = cor * (1 - tr) + roc * tr

    me = bpy.data.meshes.new("terreno")
    me.from_pydata(V.tolist(), [], faces.tolist())
    me.update()
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    # borda circular exata
    borda = [v for v in bm.verts if v.is_boundary]
    for v in borda:
        r = math.hypot(v.co.x, v.co.y)
        v.co.x, v.co.y = v.co.x * RAIO / r, v.co.y * RAIO / r
    bxy = np.array([[v.co.x, v.co.y] for v in borda])
    bz = cota(bxy[:, 0], bxy[:, 1])
    for v, z in zip(borda, bz):
        v.co.z = float(z)
    bm.to_mesh(me)
    bm.free()
    cor_lin = np.clip(cor, 0, 1) ** 2.2
    ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    rgba = np.hstack([cor_lin, np.ones((len(V), 1))]).astype(np.float32)
    ca.data.foreach_set("color", rgba.ravel())
    me.color_attributes.active_color = ca
    me.shade_smooth()
    ob = novo_objeto("Terreno", me, "terreno")
    uv_mundo(ob, 8.0)

    # base de maqueta de museu: saia e fundo
    ang = np.arctan2(bxy[:, 1], bxy[:, 0])
    ordem = np.argsort(ang)
    anel = np.stack([bxy[ordem, 0], bxy[ordem, 1], bz[ordem]], 1)
    fundo = float(bz.min()) - 14.0
    k = len(anel)
    verts = anel.tolist() + [[p[0], p[1], fundo] for p in anel]
    fs = [[i, (i + 1) % k, k + (i + 1) % k, k + i] for i in range(k)]
    fs.append(list(range(2 * k - 1, k - 1, -1)))
    mb = bpy.data.meshes.new("base")
    mb.from_pydata(verts, [], fs)
    mb.update()
    novo_objeto("Base", mb, None, mat="base")
    return Z


# ---------------------------------------------------------------- utilitários de malha


MATS = {}


def novo_objeto(nome, me, elemento, mat="terreno", pai=None):
    ob = bpy.data.objects.new(nome, me)
    bpy.context.scene.collection.objects.link(ob)
    if mat:
        me.materials.append(MATS[mat])
    if elemento:
        ob["elemento"] = elemento
    ob.parent = pai or GRUPOS.get(nome) or RAIZ_OB
    return ob


def uv_mundo(ob, escala):
    me = ob.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    uv = me.uv_layers.active.data
    for poly in me.polygons:
        n = poly.normal
        for li in poly.loop_indices:
            co = ob.matrix_world @ me.vertices[me.loops[li].vertex_index].co
            if abs(n.z) > 0.7:
                uv[li].uv = (co.x / escala, co.y / escala)
            elif abs(n.x) > abs(n.y):
                uv[li].uv = (co.y / escala, co.z / escala)
            else:
                uv[li].uv = (co.x / escala, co.z / escala)


class Malha:
    """Acumula vértices e faces (com material por face) para um objeto."""

    def __init__(self):
        self.v, self.f, self.m = [], [], []

    def face(self, pts, mat=0):
        b = len(self.v)
        self.v.extend([tuple(map(float, p)) for p in pts])
        self.f.append(list(range(b, b + len(pts))))
        self.m.append(mat)

    def prisma(self, base, z0, z1, mat=0, topo=True, fundo=False):
        """Prisma vertical sobre um polígono (anti-horário)."""
        n = len(base)
        for i in range(n):
            a, b = base[i], base[(i + 1) % n]
            za0 = z0[i] if np.ndim(z0) else z0
            zb0 = z0[(i + 1) % n] if np.ndim(z0) else z0
            za1 = z1[i] if np.ndim(z1) else z1
            zb1 = z1[(i + 1) % n] if np.ndim(z1) else z1
            self.face([(a[0], a[1], za0), (b[0], b[1], zb0), (b[0], b[1], zb1), (a[0], a[1], za1)], mat)
        if topo:
            self.face([(p[0], p[1], z1[i] if np.ndim(z1) else z1) for i, p in enumerate(base)], mat)
        if fundo:
            self.face([(p[0], p[1], z0[i] if np.ndim(z0) else z0) for i, p in reversed(list(enumerate(base)))], mat)

    def caixa(self, c, u, largura, prof, z0, z1, mat=0):
        """Caixa centrada em c, com eixo u (unitário) no plano, largura ao longo de u."""
        u = np.asarray(u, float)
        w = np.array([-u[1], u[0]])
        hl, hp = largura / 2, prof / 2
        q = [c - u * hl - w * hp, c + u * hl - w * hp, c + u * hl + w * hp, c - u * hl + w * hp]
        self.prisma(q, z0, z1, mat, topo=True, fundo=False)

    def objeto(self, nome, elemento, mats, pai=None, uv=3.0, liso=False):
        me = bpy.data.meshes.new(nome)
        me.from_pydata(self.v, [], self.f)
        me.update()
        for m in mats:
            me.materials.append(MATS[m])
        me.polygons.foreach_set("material_index", self.m)
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.002)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        if liso:
            me.shade_smooth()
        ob = bpy.data.objects.new(nome, me)
        bpy.context.scene.collection.objects.link(ob)
        if elemento:
            ob["elemento"] = elemento
        ob.parent = pai or RAIZ_OB
        if uv:
            uv_mundo(ob, uv)
        return ob


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def desvio(linha, d, fechada=False):
    """Polilinha paralela à distância d (positivo = à esquerda), com esquinas em mitra."""
    n = len(linha)
    out = []
    for i in range(n):
        if fechada or 0 < i < n - 1:
            t1 = unit(linha[i] - linha[i - 1])
            t2 = unit(linha[(i + 1) % n] - linha[i])
        elif i == 0:
            t1 = t2 = unit(linha[1] - linha[0])
        else:
            t1 = t2 = unit(linha[-1] - linha[-2])
        n1 = np.array([-t1[1], t1[0]])
        nm = unit(np.array([-(t1 + t2)[1], (t1 + t2)[0]]))
        if np.linalg.norm(nm) < 1e-6:
            nm = n1
        esc = 1.0 / max(0.7, float(nm @ n1))
        out.append(linha[i] + nm * d * esc)
    return np.array(out)


def densificar(linha, passo=2.0, fechada=False):
    pts = np.vstack([linha, linha[:1]]) if fechada else linha
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / passo)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    out = np.array(out)
    return out[:-1] if fechada else out


def mediana_movel(v, k=5, fechada=False):
    h = k // 2
    vv = np.concatenate([v[-h:], v, v[:h]]) if fechada else np.concatenate([np.repeat(v[:1], h), v, np.repeat(v[-1:], h)])
    return np.array([np.median(vv[i:i + k]) for i in range(len(v))])


def perfil(linha, esp, fechada=False, raio=1.6):
    """Mede uma muralha no LiDAR, vértice a vértice: topo das ameias no MDS (percentil 95 das células
    sem vegetação num raio de 1,6 m da linha, filtrado ao longo do pano), pé exterior e chão interior
    no MDT. Devolve a linha densificada, topo, pé exterior e chão interior (z da cena)."""
    linha, desl = ajustar(linha, fechada)
    if np.abs(desl).max() > 0.5:
        print(f"  linha ajustada ao MDS: deslocamento máx. {np.abs(desl).max():.1f} m, mediano {np.median(np.abs(desl)):.1f} m")
    topo = np.array([medir(centro=p, raio=raio) for p in linha])
    ok = ~np.isnan(topo)
    idx = np.arange(len(linha))
    topo = np.interp(idx, idx[ok], topo[ok], period=len(linha) if fechada else None)
    topo = mediana_movel(topo, 5, fechada)
    fora = np.min([cota(*desvio(linha, -(esp / 2 + d), fechada).T) for d in (0.5, 1.0, 1.5)], axis=0)
    dentro_ = cota(*desvio(linha, esp / 2 + 1.0, fechada).T)
    return linha, topo, fora, dentro_


def ajustar(linha, fechada=False, maximo=4.5):
    """Encosta a linha do OSM à crista da muralha no MDS. Em cada vértice procura, na normal e até
    `maximo` metros, o ponto mais alto (sem vegetação); só desloca quando a crista está claramente
    acima (> 1 m) do que está sob a linha. Os deslocamentos são filtrados ao longo do pano."""
    linha = densificar(linha, 2.0, fechada)
    nrm = desvio(linha, 1.0, fechada) - linha
    ds = np.arange(-maximo, maximo + 0.01, 0.25)
    xy = linha[:, None, :] + nrm[:, None, :] * ds[None, :, None]
    z = amostra(MDS, xy[..., 0], xy[..., 1])
    ci = np.clip(np.round((xy[..., 0] + LMEIO) / LPASSO).astype(int), 0, VEG.shape[1] - 1)
    ri = np.clip(np.round((xy[..., 1] + LMEIO) / LPASSO).astype(int), 0, VEG.shape[0] - 1)
    z = np.where(VEG[ri, ci], -1e3, z) - 0.15 * np.abs(ds)[None, :]
    k = np.argmax(z, axis=1)
    z0 = z[:, len(ds) // 2]
    d = np.where(z[np.arange(len(linha)), k] - z0 > 1.0, ds[k], 0.0)
    if not fechada:
        d[0] = d[-1] = 0.0  # as pontas ligam a torres e a outros panos: ficam onde estão
    d = mediana_movel(d, 5, fechada)
    d = np.convolve(np.pad(d, 2, mode="wrap" if fechada else "edge"), np.ones(5) / 5, mode="valid")
    return linha + nrm * d[:, None], d


def muralha(m, linha, esp, topo, fora, dentro_, fechada=False, ameias=True, mat=0):
    """Pano de muralha ao longo da linha central (o exterior fica à direita), com o topo
    medido vértice a vértice. Os panos descem 1,5 m abaixo do terreno para não ficarem no ar."""
    adarve = topo - (PARAPEITO + MERLAO["altura"] if ameias else PARAPEITO)
    adarve = np.maximum(adarve, dentro_)  # onde o chão de dentro chega ao adarve, o pano é só parapeito
    ext = desvio(linha, -esp / 2, fechada)
    inn = desvio(linha, esp / 2, fechada)
    n = len(linha)
    ze = np.minimum(fora, adarve) - 1.5
    zi = np.minimum(dentro_, adarve) - 1.5
    a = adarve
    seg = range(n) if fechada else range(n - 1)
    for i in seg:
        j = (i + 1) % n
        m.face([(*ext[i], ze[i]), (*ext[j], ze[j]), (*ext[j], a[j]), (*ext[i], a[i])], mat)
        m.face([(*inn[j], zi[j]), (*inn[i], zi[i]), (*inn[i], a[i]), (*inn[j], a[j])], mat)
        m.face([(*ext[i], a[i]), (*ext[j], a[j]), (*inn[j], a[j]), (*inn[i], a[i])], mat)
        m.face([(*ext[j], ze[j]), (*ext[i], ze[i]), (*inn[i], zi[i]), (*inn[j], zi[j])], mat)
    if not fechada:
        for i in (0, n - 1):
            m.face([(*ext[i], ze[i]), (*inn[i], zi[i]), (*inn[i], a[i]), (*ext[i], a[i])], mat)
    # parapeito do lado exterior e ameias por cima
    par_int = desvio(linha, -esp / 2 + 0.6, fechada)
    for i in seg:
        j = (i + 1) % n
        q = [ext[i], ext[j], par_int[j], par_int[i]]
        m.prisma(q, [a[i], a[j], a[j], a[i]], [a[i] + PARAPEITO, a[j] + PARAPEITO, a[j] + PARAPEITO, a[i] + PARAPEITO], mat)
    if ameias:
        ameias_ao_longo(m, desvio(linha, -esp / 2 + 0.3, fechada), a + PARAPEITO, fechada, mat)
    return adarve


def ameias_ao_longo(m, linha, z, fechada=False, mat=0, prof=0.6):
    """Merlões ao longo de uma polilinha; z é a cota do parapeito (escalar ou por vértice).
    O ritmo segue o pano todo, não cada segmento, para a linha densificada não mudar o desenho."""
    pts = np.vstack([linha, linha[:1]]) if fechada else linha
    zs = np.full(len(pts), z) if np.ndim(z) == 0 else (np.append(z, z[:1]) if fechada else np.asarray(z))
    passo = MERLAO["largura"] + MERLAO["intervalo"]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    acum = np.concatenate([[0], np.cumsum(seg)])
    L = acum[-1]
    k = max(1, int((L + MERLAO["intervalo"]) // passo))
    sobra = L - (k * passo - MERLAO["intervalo"])
    for i in range(k):
        s = sobra / 2 + i * passo + MERLAO["largura"] / 2
        j = min(int(np.searchsorted(acum, s)) - 1, len(seg) - 1)
        j = max(j, 0)
        t = (s - acum[j]) / max(seg[j], 1e-9)
        c = pts[j] + (pts[j + 1] - pts[j]) * t
        zc = zs[j] + (zs[j + 1] - zs[j]) * t
        u = unit(pts[j + 1] - pts[j])
        m.caixa(c, u, MERLAO["largura"], prof, zc, zc + MERLAO["altura"], mat)


def torre(m, base, pe, topo, mat=0):
    """Torre maciça com parapeito e ameias no topo."""
    if area(base) < 0:
        base = base[::-1]
    terraco = topo - PARAPEITO - MERLAO["altura"]
    m.prisma(base, pe - 1.5, terraco, mat)
    interior = desvio(base, 0.6, True)
    for i in range(len(base)):
        j = (i + 1) % len(base)
        m.prisma([base[i], base[j], interior[j], interior[i]], terraco, terraco + PARAPEITO, mat)
    ameias_ao_longo(m, desvio(base, 0.3, True), terraco + PARAPEITO, True, mat)


def piramide(m, c, r, z0, z1, mat):
    """Pináculo: pirâmide de base quadrada."""
    b = [c + np.array([math.cos(k * math.pi / 2 + math.pi / 4), math.sin(k * math.pi / 2 + math.pi / 4)]) * r for k in range(4)]
    for k in range(4):
        m.face([(*b[k], z0), (*b[(k + 1) % 4], z0), (*c, z1)], mat)


def arco(m, centro, u, normal, largura, soleira, nascenca, prof, mat):
    """Vão em arco pleno, como reentrância escura na face (normal aponta para fora)."""
    r = largura / 2
    pts = [(-r, soleira), (r, soleira), (r, soleira + nascenca)]
    for k in range(1, 12):
        a = math.pi * k / 12
        pts.append((r * math.cos(a), soleira + nascenca + r * math.sin(a)))
    pts.append((-r, soleira + nascenca))
    frente = [centro + u * px + normal * 0.03 for px, _ in pts]
    m.face([(*p, z) for p, (_, z) in zip(frente, pts)], mat)


# ---------------------------------------------------------------- edifícios


def _retangulo(base):
    b = np.array(base)
    if area(b) < 0:
        b = b[::-1]
    e = [np.linalg.norm(b[(i + 1) % 4] - b[i]) for i in range(4)]
    k = 0 if e[0] + e[2] >= e[1] + e[3] else 1  # arestas k e k+2 são as compridas
    b = np.roll(b, -k, axis=0)
    L = (np.linalg.norm(b[1] - b[0]) + np.linalg.norm(b[2] - b[3])) / 2
    W = (np.linalg.norm(b[3] - b[0]) + np.linalg.norm(b[2] - b[1])) / 2
    return b, L, W


def altura_telhado(poly, inclinacao=24):
    s = simplificar(poly)
    if len(s) == 4 and abs(area(s)) > 0.85 * abs(area(poly)):
        return _retangulo(s)[2] / 2 * math.tan(math.radians(inclinacao))
    return min(2.2, 0.25 * math.sqrt(abs(area(poly))))


def telhado_4aguas(m, base, z, inclinacao=24, mat=1):
    """Telhado de quatro águas sobre um polígono quase retangular de 4 vértices."""
    b, L, W = _retangulo(base)
    h = W / 2 * math.tan(math.radians(inclinacao))
    mid_a = (b[0] + b[3]) / 2
    mid_b = (b[1] + b[2]) / 2
    u = unit(mid_b - mid_a)
    r1 = mid_a + u * min(W / 2, L / 2)
    r2 = mid_b - u * min(W / 2, L / 2)
    R1, R2 = (*r1, z + h), (*r2, z + h)
    B = [(*p, z) for p in b]
    m.face([B[0], B[1], R2, R1], mat)
    m.face([B[2], B[3], R1, R2], mat)
    m.face([B[1], B[2], R2], mat)
    m.face([B[3], B[0], R1], mat)


def telhado_generico(m, base, z, mat=1):
    b = np.array(base)
    if area(b) < 0:
        b = b[::-1]
    c = b.mean(0)
    inset = c + (b - c) * 0.45
    h = min(2.2, 0.25 * math.sqrt(abs(area(b))))
    n = len(b)
    for i in range(n):
        j = (i + 1) % n
        m.face([(*b[i], z), (*b[j], z), (*inset[j], z + h), (*inset[i], z + h)], mat)
    m.face([(*p, z + h) for p in inset], mat)


def simplificar(poly, tol=0.6):
    """Remove vértices quase colineares (para reconhecer retângulos)."""
    pts = list(poly)
    mudou = True
    while mudou and len(pts) > 4:
        mudou = False
        for i in range(len(pts)):
            a, b, c = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[(i + 1) % len(pts)])
            d = abs((c - a)[0] * (b - a)[1] - (c - a)[1] * (b - a)[0]) / max(np.linalg.norm(c - a), 1e-9)
            if d < tol:
                pts.pop(i)
                mudou = True
                break
    return np.array(pts)


def casa(m, poly, altura, mat_par=0, mat_tel=1):
    """Volume simplificado com o cume à cota medida no MDS (percentil 95 do telhado dentro da planta).
    Sem medição (planta sem eco de edifício no LiDAR), usa a altura dada acima do terreno.
    Devolve a cota da cornija e se foi medida."""
    if area(poly) < 0:
        poly = poly[::-1]
    zt = cota(poly[:, 0], poly[:, 1])
    z0 = float(zt.min()) - 1.0
    cume = medir(desvio(poly, 0.4, True) if abs(area(poly)) > 12 else poly)
    medido = not math.isnan(cume) and cume > float(np.median(zt)) + 2.0
    z1 = max(cume - altura_telhado(poly), float(np.median(zt)) + 2.0) if medido else float(np.median(zt)) + altura
    m.prisma(poly, z0, z1, mat_par, topo=False)
    s = simplificar(poly)
    if len(s) == 4 and abs(area(s)) > 0.85 * abs(area(poly)):
        telhado_4aguas(m, s, z1, mat=mat_tel)
    else:
        telhado_generico(m, poly, z1, mat=mat_tel)
    return z1, medido


MEDIDAS = {}  # o que se mediu no LiDAR, para conferir e citar no penela.json
PERFIS = {}  # perfis das muralhas (saida/perfis.npz), para conferência


def igreja():
    """Igreja de São Miguel: planta do OSM; cumeeiras e torre medidas no MDS; vãos e coruchéu
    pelas fotografias."""
    m = Malha()  # 0 cal, 1 telha, 2 sombra, 3 cantaria
    poly = geom(osm("3904babc", "edificios"))
    if area(poly) < 0:
        poly = poly[::-1]
    fa, fb = P(-30.6, 9.6), P(-39.5, 23.3)  # cunhais da fachada (sul e oeste)
    larg = np.linalg.norm(fb - fa)
    v = unit(fb - fa)  # ao longo da fachada, de sul para norte
    u = np.array([v[1], -v[0]])  # eixo da igreja, da fachada para a cabeceira
    if u @ (P(-11, 22) - fa) < 0:
        u = -u
    adro = float(cota(*np.array([fa, fb, (fa + fb) / 2]).T).mean())  # chão diante da fachada
    z0 = float(cota(*poly.T).min()) - 0.5
    uv = np.stack([(poly - fa) @ u, (poly - fa) @ v], 1)
    comp = 23.2  # corpo principal: u de 0 a 23,2 m e v de 0 à largura da fachada (planta OSM)

    def quad(v0, v1, u0, u1):
        c = np.array([fa + v * v0 + u * u0, fa + v * v0 + u * u1, fa + v * v1 + u * u1, fa + v * v1 + u * u0])
        return c if area(c) > 0 else c[::-1]

    def faixa(v0, v1, u0=1.0, u1=comp - 1.0, q=50):
        return medir(quad(v0, v1, u0, u1), q=q)

    # Telhado do corpo principal (três naves sob um só telhado, como mostra o MDS): cumeeira onde a
    # faixa ao longo do eixo é mais alta; beirados sul e norte medidos junto às paredes.
    vs = np.arange(2.0, larg - 1.5, 0.5)
    perfil_v = np.array([faixa(x - 0.25, x + 0.25, q=50) for x in vs])
    v_cume = float(vs[int(np.nanargmax(perfil_v))])
    cume = faixa(v_cume - 0.75, v_cume + 0.75, q=90)
    beiral_s = faixa(0.5, 1.5)
    beiral_n = faixa(larg - 1.5, larg - 0.5)
    p = lambda vv, uu, z: (*(fa + v * vv + u * uu), z)
    m.face([p(0, 0, beiral_s), p(0, comp, beiral_s), p(v_cume, comp, cume), p(v_cume, 0, cume)], 1)
    m.face([p(v_cume, 0, cume), p(v_cume, comp, cume), p(larg, comp, beiral_n), p(larg, 0, beiral_n)], 1)
    m.face([p(0, 0, beiral_s), p(v_cume, 0, cume), p(larg, 0, beiral_n)], 0)  # empena da fachada
    m.face([p(larg, comp, beiral_n), p(v_cume, comp, cume), p(0, comp, beiral_s)], 0)
    for vv, zz in ((0, beiral_s), (larg, beiral_n)):  # paredes laterais até ao beirado
        m.face([p(vv, 0, z0), p(vv, comp, z0), p(vv, comp, zz), p(vv, 0, zz)], 0)
    for uu in (0, comp):
        m.face([p(0, uu, z0), p(larg, uu, z0), p(larg, uu, beiral_n), p(0, uu, beiral_s)], 0)

    # anexo norte (u de 11 a 23,2 m, v até 20,7 m na planta OSM): telhado de uma água, medido
    v_max = float(uv[:, 1].max())
    an_in, an_out = faixa(larg + 0.3, larg + 1.3, 11.5, comp - 0.5), faixa(v_max - 1.2, v_max - 0.4, 11.5, comp - 0.5)
    an_in = min(an_in, beiral_n)
    m.face([p(larg, 11, z0), p(v_max, 11, z0), p(v_max, 11, an_out), p(larg, 11, an_in)], 0)
    m.face([p(v_max, comp, z0), p(larg, comp, z0), p(larg, comp, an_in), p(v_max, comp, an_out)], 0)
    m.face([p(v_max, 11, z0), p(v_max, comp, z0), p(v_max, comp, an_out), p(v_max, 11, an_out)], 0)
    m.face([p(larg, 11, an_in), p(v_max, 11, an_out), p(v_max, comp, an_out), p(larg, comp, an_in)], 1)

    # cabeceira (u de 23,2 a 25,5 m, v de 3,2 a 11,3 m): volume baixo com telhado medido
    cab = quad(3.2, 11.3, comp, float(uv[:, 0].max()))
    cab_topo = medir(desvio(cab, 0.3, True), q=90)
    if math.isnan(cab_topo) or cab_topo < adro + 2.5:
        cab_topo = adro + 4.0
    m.prisma(cab, z0, cab_topo - 1.0, 0, topo=False)
    telhado_generico(m, cab, cab_topo - 1.0, mat=1)

    # Fachada (fotografia da fachada, out. 2026): porta de madeira retangular com aro de cantaria no
    # eixo da nave, janela retangular por cima, friso a meia altura, cunhais e pináculos de cantaria
    # e cruz no vértice do frontão.
    nf = -u
    pc = fa + v * v_cume

    def ret(c, larg_, z0_, z1_, mat, sai=0.03):
        q = c + nf * sai
        m.face([(*(q - v * larg_ / 2), z0_), (*(q + v * larg_ / 2), z0_), (*(q + v * larg_ / 2), z1_), (*(q - v * larg_ / 2), z1_)], mat)

    ret(pc, 2.0, adro, adro + 3.4, 3, 0.04)  # aro
    ret(pc, 1.5, adro, adro + 3.0, 4, 0.06)  # porta
    ret(pc + v * 0.0, 1.5, adro + 4.1, adro + 5.9, 3, 0.04)
    ret(pc, 1.0, adro + 4.4, adro + 5.6, 2, 0.06)  # janela
    m.caixa(fa + v * larg / 2 + nf * 0.12, v, larg + 0.3, 0.3, adro + 6.0, adro + 6.35, 3)  # friso
    for vv, zz in ((0.0, beiral_s), (larg, beiral_n)):
        cc_ = fa + v * vv + nf * 0.1 + v * (0.3 if vv == 0 else -0.3)
        m.caixa(cc_, v, 0.6, 0.3, z0, zz + 0.2, 3)  # cunhal
        m.caixa(cc_, v, 0.35, 0.35, zz + 0.2, zz + 0.5, 3)
        m.prisma([cc_ + np.array([math.cos(k * math.pi / 2 + math.pi / 4), math.sin(k * math.pi / 2 + math.pi / 4)]) * 0.16 for k in range(4)],
                 zz + 0.5, zz + 0.6, 3)
        piramide(m, cc_, 0.15, zz + 0.6, zz + 1.5, 3)
    # cruz no vértice do frontão
    topo = pc + nf * 0.1
    m.caixa(topo, v, 0.18, 0.18, cume, cume + 1.3, 3)
    m.caixa(topo, v, 0.75, 0.18, cume + 0.75, cume + 0.93, 3)

    # Torre sineira (planta OSM). No MDS o topo das paredes é um patamar e o vértice do coruchéu o
    # ponto mais alto; sineiras, relógio, janela, pináculos e cata-vento pela fotografia da fachada.
    t = geom(osm("1d555fe5"))
    t = simplificar(t, 0.8)
    if area(t) < 0:
        t = t[::-1]
    vt = celulas(t)
    vertice = float(vt.max())
    cornija = float(np.percentile(vt, 30))
    if not (vertice - 7 < cornija < vertice - 2):
        cornija = vertice - 4.5
    m.prisma(t, z0, cornija, 0, topo=True)
    ct = t.mean(0)
    face_fr = min(range(len(t)), key=lambda i: ((t[i] + t[(i + 1) % len(t)]) / 2 - ct) @ u)
    a, b = t[face_fr], t[(face_fr + 1) % len(t)]
    ft = unit(b - a)
    out = unit(np.array([(b - a)[1], -(b - a)[0]]))
    mid = (a + b) / 2
    alt_t = cornija - adro
    for dx in (-0.8, 0.8):  # duas sineiras lado a lado
        arco(m, mid + ft * dx, ft, out, 0.9, adro + 0.48 * alt_t, 0.9, 0.1, 2)
    arco(m, mid, ft, out, 0.5, adro + 0.66 * alt_t, 0.6, 0.1, 2)  # janela pequena
    cc = mid + out * 0.04
    zr = cornija - 0.06 * alt_t - 0.1
    m.face([(*(cc + ft * 0.62 * math.cos(k * math.pi / 8)), zr + 0.62 * math.sin(k * math.pi / 8)) for k in range(16)], 3)  # relógio
    # cornija, coruchéu caiado, pináculos nos cunhais e cata-vento
    m.prisma(desvio(t, -0.15, True), cornija, cornija + 0.4, 3)
    ap = (*ct, vertice)
    tt = desvio(t, 0.35, True)
    for i in range(len(tt)):
        m.face([(*tt[i], cornija + 0.4), (*tt[(i + 1) % len(tt)], cornija + 0.4), ap], 0)
    for p_ in t:
        piramide(m, p_ + unit(ct - p_) * 0.25, 0.18, cornija + 0.4, cornija + 1.6, 3)
    m.caixa(ct, ft, 0.08, 0.08, vertice, vertice + 1.5, 3)
    m.caixa(ct, ft, 0.6, 0.08, vertice + 1.0, vertice + 1.1, 3)
    m.objeto("Igreja_Sao_Miguel", "igreja", ["cal", "telha", "sombra", "cantaria", "madeira"], GRUPOS["Igreja_Sao_Miguel"], uv=2.0)
    MEDIDAS["igreja"] = {"adro": adro + ZREF, "cumeeira": cume + ZREF, "beirado_sul": beiral_s + ZREF,
                         "beirado_norte": beiral_n + ZREF, "anexo_norte": [an_out + ZREF, an_in + ZREF],
                         "cabeceira": cab_topo + ZREF, "torre_cornija": cornija + ZREF, "torre_vertice": vertice + ZREF,
                         "cumeeira_acima_adro": cume - adro, "torre_acima_adro": vertice - adro}


# ---------------------------------------------------------------- materiais e texturas


def imagem(nome, arr):
    h, w, _ = arr.shape
    img = bpy.data.images.new(nome, w, h, alpha=False)
    rgba = np.concatenate([arr, np.ones((h, w, 1))], 2).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.file_format = "JPEG"
    img.pack()
    return img


def textura_pedra(n=512):
    """Aparelho irregular de calcário/arenito em fiadas (3 m por repetição)."""
    r = np.random.default_rng(1310)
    tex = np.zeros((n, n, 3))
    m_px = n / 3.0
    y = 0
    while y < n:
        hf = int(r.uniform(0.22, 0.42) * m_px)
        x = -int(r.uniform(0, 0.6) * m_px)
        while x < n:
            wf = int(r.uniform(0.35, 0.95) * m_px)
            base = np.array([0.72, 0.66, 0.54]) * r.uniform(0.86, 1.10) + r.normal(0, 0.015, 3)
            x0, x1 = max(x, 0), min(x + wf, n)
            y1 = min(y + hf, n)
            if x1 > x0:
                tex[y:y1, x0:x1] = base
                tex[y:y1, x0:min(x0 + 2, x1)] *= 0.82
            x += wf
        tex[y:min(y + 2, n)] *= 0.8
        y += hf
    yy, xx = np.mgrid[0:n, 0:n] / n
    grao = fbm(xx * 40, yy * 40, 1.5, 21)[..., None]
    tex = tex * (0.86 + 0.24 * grao)
    return np.clip(tex, 0, 1)


def textura_grao(n=256):
    yy, xx = np.mgrid[0:n, 0:n] / n
    g = fbm(xx * 30, yy * 30, 2.0, 31)
    return np.repeat((0.84 + 0.22 * g)[..., None], 3, 2).clip(0, 1)


def material(nome, cor, rug=0.9, tex=None, vcor=False):
    mt = bpy.data.materials.new(nome)
    mt.use_nodes = True
    nt = mt.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*[c ** 2.2 for c in cor], 1)
    bsdf.inputs["Roughness"].default_value = rug
    if tex is not None:
        ti = nt.nodes.new("ShaderNodeTexImage")
        ti.image = tex
        if vcor:
            va = nt.nodes.new("ShaderNodeVertexColor")
            va.layer_name = "Col"
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            nt.links.new(ti.outputs["Color"], mix.inputs[6])
            nt.links.new(va.outputs["Color"], mix.inputs[7])
            nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        else:
            nt.links.new(ti.outputs["Color"], bsdf.inputs["Base Color"])
    elif vcor:
        va = nt.nodes.new("ShaderNodeVertexColor")
        va.layer_name = "Col"
        nt.links.new(va.outputs["Color"], bsdf.inputs["Base Color"])
    MATS[nome] = mt
    return mt


# ---------------------------------------------------------------- cena


def limpar():
    bpy.ops.wm.read_factory_settings(use_empty=True)


limpar()
RAIZ_OB = bpy.data.objects.new("Castelo_Penela", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = LAT0, LON0
RAIZ_OB["centro_m"] = [float(CENTRO[0]), float(CENTRO[1])]
RAIZ_OB["cota_zero"] = ZREF
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
for g in ["Terreno", "Muralhas", "Castelejo", "Torreoes", "Porta_Vila", "Porta_Traicao", "Brecha_Desaparecidas",
          "Igreja_Sao_Miguel", "Recinto", "Caminhos", "Vegetacao", "Vila"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

pedra = imagem("pedra", textura_pedra())
grao = imagem("grao", textura_grao())
material("pedra", (1, 1, 1), 0.92, pedra)
material("sombra", (0.13, 0.11, 0.09), 1.0)
material("cal", (0.94, 0.93, 0.89), 0.85)
material("telha", (0.68, 0.35, 0.22), 0.8)
material("cantaria", (0.82, 0.79, 0.72), 0.9)
material("terreno", (1, 1, 1), 0.97, grao, vcor=True)
material("base", (0.33, 0.26, 0.20), 0.95)
material("caminho", (0.74, 0.72, 0.67), 0.95)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)
material("apoio", (0.42, 0.33, 0.25), 0.9)
material("madeira", (0.42, 0.25, 0.13), 0.8)

print("terreno…")
construir_terreno()
for ob in bpy.data.objects:
    if ob.name in ("Terreno", "Base"):
        ob.parent = GRUPOS["Terreno"]

print("muralhas…")
for s_, linha in SECTORES.items():
    m = Malha()
    lin, topo, fora, dentro_ = perfil(linha, MUR["espessura"])
    adarve = muralha(m, lin, MUR["espessura"], topo, fora, dentro_)
    PERFIS[s_] = np.stack([lin[:, 0], lin[:, 1], topo, fora, dentro_], 1)
    if s_ == "pano_poente":
        i = perto_de(linha, PORTA_VILA)
        a, b = linha[max(i - 1, 0)], linha[min(i + 1, len(linha) - 1)]
        u = unit(b - a)
        out = np.array([u[1], -u[0]])
        g = PORTA_VILA + out * MUR["espessura"] / 2
        SOLEIRA_VILA = float(cota(*(g + out * 1.0)))
        arco(m, g, u, out, 2.6, SOLEIRA_VILA, 2.6, 0.1, 1)
    m.objeto(s_, s_, ["pedra", "sombra"], GRUPOS["Muralhas"])
    alt = topo - fora
    MEDIDAS[s_] = {"topo": [float(topo.min() + ZREF), float(topo.max() + ZREF)],
                   "altura_exterior": [float(np.percentile(alt, 10)), float(np.percentile(alt, 90))],
                   "acima_do_chao_interior": [float(np.percentile(topo - dentro_, 10)), float(np.percentile(topo - dentro_, 90))]}

print("castelejo…")
m = Malha()
lin, topo, fora, dentro_ = perfil(castelejo, 1.8, fechada=True)
muralha(m, lin, 1.8, topo, fora, dentro_, fechada=True)
PERFIS["castelejo"] = np.stack([lin[:, 0], lin[:, 1], topo, fora, dentro_], 1)
m.objeto("Castelejo", "castelejo", ["pedra", "sombra"], GRUPOS["Castelejo"])
MEDIDAS["castelejo"] = {"topo": [float(topo.min() + ZREF), float(topo.max() + ZREF)],
                        "chao_interior": float(np.median(cota(*densificar(desvio(castelejo, 3.0, True), 2, True).T)) + ZREF),
                        "altura_exterior": [float(np.percentile(topo - fora, 10)), float(np.percentile(topo - fora, 90))],
                        "acima_do_chao_interior": [float(np.percentile(topo - dentro_, 10)), float(np.percentile(topo - dentro_, 90))]}

print("torres…")
for tid, base in torres.items():
    m = Malha()
    b = base if area(base) > 0 else base[::-1]
    topo = medir(desvio(b, 0.3, True) if abs(area(b)) > 6 else b, q=95)
    pe = float(np.min(cota(*densificar(desvio(b, -1.0, True), 1.0, True).T)))
    torre(m, b, pe, topo)
    PERFIS["torre_" + tid] = b
    MEDIDAS[tid] = {"topo": topo + ZREF, "pe": pe + ZREF, "altura_exterior": topo - pe}
    if tid == "porta_traicao":
        # porta em cotovelo: abertura exterior na face virada a nordeste e interior na face virada a sueste
        faces = [(b[i], b[(i + 1) % len(b)]) for i in range(len(b))]
        def face_para(direcao):
            return max(faces, key=lambda f: unit(np.array([(f[1] - f[0])[1], -(f[1] - f[0])[0]])) @ direcao * min(1, np.linalg.norm(f[1] - f[0]) / 2))
        for dirc in (np.array([0.7, 0.7]), np.array([0.7, -0.7])):
            a, c = face_para(dirc)
            u = unit(c - a)
            out = np.array([u[1], -u[0]])
            sole = float(cota(*((a + c) / 2 + out * 0.8)))
            arco(m, (a + c) / 2, u, out, 1.8, sole, 2.0, 0.1, 1)
        nome, grupo = "Porta_Traicao", GRUPOS["Porta_Traicao"]
    elif tid == "torre_quinaria":
        nome, grupo = "Torre_Quinaria", GRUPOS["Torreoes"]
    else:
        nome, grupo = tid.replace("torreao_poente_", "Torreao_Poente_"), GRUPOS["Torreoes"]
    m.objeto(nome, tid, ["pedra", "sombra"], grupo)

m = Malha()
circ = np.array([CUBELO + 2.8 * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 18, endpoint=False)])
topo = medir(centro=CUBELO, raio=2.4)
pe = float(np.min(cota(*desvio(circ, -1.0, True).T)))
torre(m, circ, pe, topo)
MEDIDAS["cubelo_noroeste"] = {"topo": topo + ZREF, "pe": pe + ZREF, "altura_exterior": topo - pe}
m.objeto("Cubelo_Noroeste", "cubelo_noroeste", ["pedra"], GRUPOS["Torreoes"])

# Porta da Vila: aro de cantaria em volta do vão (o vão faz parte do pano poente)
m = Malha()
i = perto_de(SECTORES["pano_poente"], PORTA_VILA)
linha = SECTORES["pano_poente"]
a, b = linha[max(i - 1, 0)], linha[min(i + 1, len(linha) - 1)]
u = unit(b - a)
out = np.array([u[1], -u[0]])
g = PORTA_VILA + out * (MUR["espessura"] / 2 + 0.12)
for sx in (-1.55, 1.55):
    m.caixa(g + u * sx, u, 0.5, 0.25, SOLEIRA_VILA - 0.2, SOLEIRA_VILA + 4.3)
m.caixa(g, u, 3.6, 0.25, SOLEIRA_VILA + 4.3, SOLEIRA_VILA + 4.8)
m.objeto("Porta_Vila", "porta_vila", ["cantaria"], GRUPOS["Porta_Vila"])
MEDIDAS["porta_vila"] = {"soleira": SOLEIRA_VILA + ZREF}

# Brecha: chão do acesso atual entre o pano sul e a torre quinária
m = Malha()
a_ = SECTORES["pano_sul"][-1]
b_ = sueste[0]
u = unit(b_ - a_)
w = np.array([-u[1], u[0]])
c = (a_ + b_) / 2
L = np.linalg.norm(b_ - a_)
q = [c - u * L / 2 - w * 5, c + u * L / 2 - w * 5, c + u * L / 2 + w * 5, c - u * L / 2 + w * 5]
zq = cota(np.array([p[0] for p in q]), np.array([p[1] for p in q]))
m.prisma(q, zq - 0.5, zq + 0.08, 0)
m.objeto("Brecha_Desaparecidas", "brecha", ["caminho"], GRUPOS["Brecha_Desaparecidas"], uv=None)
MEDIDAS["brecha"] = {"chao": float(cota(*c)) + ZREF, "abertura": float(L)}

print("igreja e recinto…")
igreja()
m = Malha()
pc_ = geom(osm("69a1ea12", "edificios"))
z1, med = casa(m, pc_, 7.0)
m.objeto("Casa_Paroquial", "casa_paroquial", ["cal", "telha"], GRUPOS["Recinto"], uv=None)
MEDIDAS["casa_paroquial"] = {"cornija": z1 + ZREF, "cume": medir(desvio(pc_ if area(pc_) > 0 else pc_[::-1], 0.4, True)) + ZREF,
                             "chao": float(np.median(cota(*pc_.T))) + ZREF, "medido": med}
m = Malha()
for pid in ("9ed07352", "dc7a9b83"):
    poly = geom(osm(pid, "edificios"))
    if area(poly) < 0:
        poly = poly[::-1]
    z = cota(poly[:, 0], poly[:, 1])
    tp = medir(poly)
    tp = tp if not math.isnan(tp) and tp > float(z.max()) + 1.5 else float(z.max()) + 3.0
    m.prisma(poly, float(z.min()) - 0.5, tp, 0)
m.objeto("Apoio", None, ["apoio"], GRUPOS["Recinto"], uv=None)
m = Malha()
lin = densificar(MURO_TERREIRO, 2.0)
dentro_ = mediana_movel(cota(*desvio(lin, 1.2).T), 5)  # terreiro (à esquerda da linha), sem picos nas esquinas
fora = cota(*desvio(lin, -1.2).T)  # pátio
# só onde o MDT mostra o degrau (> 0,6 m); no resto do traçado o terreiro e o pátio estão ao mesmo nível
degrau = mediana_movel(dentro_ - fora, 3) > 0.6
for i in range(1, len(degrau) - 2):  # fecha falhas de um ou dois vértices
    if not degrau[i] and degrau[i - 1] and (degrau[i + 1] or degrau[i + 2]):
        degrau[i] = True
ini = None
for i, k in enumerate(list(degrau) + [False]):
    if k and ini is None:
        ini = i
    if not k and ini is not None:
        if i - ini >= 2:
            sl = slice(ini, i)
            muralha(m, lin[sl], 0.9, dentro_[sl] + 0.9, fora[sl], dentro_[sl], ameias=False)
        ini = None
m.objeto("Muro_Terreiro", "muro_terreiro", ["pedra"], GRUPOS["Recinto"])
PERFIS["muro_terreiro"] = np.stack([lin[:, 0], lin[:, 1], dentro_ - fora], 1)
MEDIDAS["muro_terreiro"] = {"desnivel_max": float((dentro_ - fora).max()), "comprimento": float(degrau.sum() * 2.0)}

print("vila…")
proprios = {"3904babc", "69a1ea12", "9ed07352", "dc7a9b83", "0de35856"}
m = Malha()
pegadas = []
n_med = 0
for f in F["edificios"]:
    if f["id"][:8] in proprios or f["geometria"]["type"] != "Polygon":
        continue
    poly = geom(f)
    c = poly.mean(0)
    if np.hypot(*c) > RAIO - 6 or np.hypot(*poly.T).max() > RAIO - 1:
        continue
    if dentro(c[0], c[1], perimetro) or any(np.linalg.norm(c - t.mean(0)) < 4 for t in torres.values()):
        continue
    if distancia(np.array([c[0]]), np.array([c[1]]), principal)[0][0] < 2.0:
        continue
    h = f.get("altura") or ((f.get("pisos") or 2) * 3.0 + 0.6)
    n_med += casa(m, poly, h)[1]
    pegadas.append(poly)
n_osm = len(pegadas)
# Casas que o OSM não tem: plantas tiradas da nuvem de pontos (lidar.py -> dados/casas.json)
muros = [principal, sueste, castelejo]
for c_ in json.loads((AQUI / "dados" / "casas.json").read_text()):
    poly = np.array(c_["planta"], float)
    if area(poly) < 0:
        poly = poly[::-1]
    c = poly.mean(0)
    if np.hypot(*poly.T).max() > RAIO - 1 or dentro(c[0], c[1], perimetro):
        continue
    if min(distancia(poly[:, 0], poly[:, 1], w_)[0].min() for w_ in muros) < 1.5:
        continue  # o que encosta às muralhas é a própria muralha
    if any(np.linalg.norm(c - t.mean(0)) < 5 for t in torres.values()):
        continue
    n_med += casa(m, poly, 6.0)[1]
    pegadas.append(poly)
m.objeto("Edificios_Vila", "vila", ["cal", "telha"], GRUPOS["Vila"], uv=None)
MEDIDAS["vila"] = {"edificios": len(pegadas), "do_osm": n_osm, "do_lidar": len(pegadas) - n_osm, "medidos": n_med}

print("caminhos…")
LARG = {"primary": 7, "secondary": 7, "tertiary": 6, "residential": 5, "unclassified": 4.5, "service": 3.5,
        "living_street": 4, "pedestrian": 4, "track": 3, "footway": 1.6, "path": 1.4, "steps": 1.6}
m = Malha()
for f in F["caminhos"]:
    if f["subtipo"] != "road":
        continue
    lw = LARG.get(f["classe"] or "", 3)
    linha = geom(f)
    # densificar
    pts = [linha[0]]
    for a, b in zip(linha[:-1], linha[1:]):
        k = max(1, int(np.linalg.norm(b - a) / 2.0))
        pts += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    pts = np.array(pts)
    ok = (np.hypot(*pts.T) < RAIO - 1.5) & ~dentro(pts[:, 0], pts[:, 1], perimetro)
    folga = 6.0 if lw < 2.5 else 1.2
    ok &= distancia(pts[:, 0], pts[:, 1], perimetro, True)[0] > lw / 2 + folga
    e_ = 1.0
    inc = np.hypot(cota(pts[:, 0] + e_, pts[:, 1]) - cota(pts[:, 0] - e_, pts[:, 1]),
                   cota(pts[:, 0], pts[:, 1] + e_) - cota(pts[:, 0], pts[:, 1] - e_)) / (2 * e_)
    ok &= inc < 0.6
    troco = []
    for p, k in zip(pts, ok):
        if k:
            troco.append(p)
        if (not k or p is pts[-1]) and len(troco) > 1:
            t = np.array(troco)
            L_ = desvio(t, lw / 2)
            R_ = desvio(t, -lw / 2)
            zl = cota(L_[:, 0], L_[:, 1]) + 0.15
            zr = cota(R_[:, 0], R_[:, 1]) + 0.15
            for i in range(len(t) - 1):
                m.face([(*R_[i], zr[i]), (*R_[i + 1], zr[i + 1]), (*L_[i + 1], zl[i + 1]), (*L_[i], zl[i])])
        if not k:
            troco = []
m.objeto("Caminhos", "caminhos", ["caminho"], GRUPOS["Caminhos"], uv=None, liso=True)

print("vegetação…")


def arvore_modelo(subdiv):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    v = np.array([x.co[:] for x in bm.verts])
    f = [[x.index for x in fa.verts] for fa in bm.faces]
    bm.free()
    return v, f


ico0, ico0f = arvore_modelo(1)
mv, mf, mc = [], [], []


def plantar(x, y, z, raio, alt, copa_cor, tronco=1.0):
    b = len(mv)
    # tronco: prisma de 5 lados
    for k in range(5):
        a = 2 * math.pi * k / 5
        mv.append((x + 0.18 * math.cos(a), y + 0.18 * math.sin(a), z - 0.3))
        mv.append((x + 0.14 * math.cos(a), y + 0.14 * math.sin(a), z + tronco + 0.3))
    for k in range(5):
        j = (k + 1) % 5
        mf.append([b + 2 * k, b + 2 * j, b + 2 * j + 1, b + 2 * k + 1])
    mc.extend([(0.33, 0.26, 0.18)] * 10)
    b = len(mv)
    jit = rng.normal(0, 0.08, ico0.shape)
    pts = (ico0 + jit) * np.array([raio, raio, alt / 2]) + np.array([x, y, z + tronco + alt / 2])
    mv.extend(map(tuple, pts))
    mf.extend([[b + i for i in fa] for fa in ico0f])
    sh = 0.9 + 0.2 * rng.random()
    mc.extend([tuple(np.clip(np.array(copa_cor) * sh, 0, 1))] * len(pts))


# Copas medidas no LiDAR (lidar.py): posição, topo e raio de cada árvore; a forma é esquemática.
ARV = LID["arvores"]
n_arv = 0
for x, y, topo, _, r in ARV:
    if np.hypot(x, y) > RAIO - 2:
        continue
    z = float(cota(x, y))
    h = float(topo) - ZREF - z
    if h < 2.0:
        continue
    r = float(np.clip(r, 1.0, 6.0))
    alt = float(np.clip(h * 0.62, 1.4, 2.6 * r))
    no_recinto = bool(dentro(x, y, perimetro))
    if no_recinto and h < 8:
        cor = (0.48, 0.52, 0.38)  # oliveiras (fotografias aéreas 1 e 2)
    else:
        cor = (0.30, 0.38, 0.22) if rng.random() < 0.65 else (0.42, 0.47, 0.30)
    plantar(float(x), float(y), z, r, alt, cor, tronco=max(h - alt, 0.6))
    n_arv += 1
MEDIDAS["vegetacao"] = {"arvores": n_arv}

me = bpy.data.meshes.new("vegetacao")
me.from_pydata(mv, [], mf)
me.update()
cl = np.array(mc) ** 2.2
ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
ca.data.foreach_set("color", np.hstack([cl, np.ones((len(cl), 1))]).astype(np.float32).ravel())
me.color_attributes.active_color = ca
novo_objeto("Vegetacao", me, "vegetacao", mat="arvore", pai=GRUPOS["Vegetacao"])


def _arred(v):
    if isinstance(v, float):
        return round(v, 1)
    if isinstance(v, list):
        return [_arred(x) for x in v]
    if isinstance(v, dict):
        return {k: _arred(x) for k, x in v.items()}
    return v


(AQUI / "saida").mkdir(exist_ok=True)
np.savez(AQUI / "saida" / "perfis.npz", **PERFIS)
(AQUI / "dados" / "medidas.json").write_text(json.dumps(_arred(MEDIDAS), ensure_ascii=False, indent=1))
print(json.dumps(_arred(MEDIDAS), ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
(AQUI / "saida").mkdir(exist_ok=True)
glb = AQUI / "saida" / "penela-bruto.glb"
props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
opcoes = dict(
    filepath=str(glb), export_format="GLB", export_extras=True, export_yup=True, export_apply=True,
    export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=7,
    export_draco_position_quantization=15, export_draco_normal_quantization=10,
    export_draco_texcoord_quantization=12, export_draco_color_quantization=8,
    export_image_format="JPEG", export_jpeg_quality=82,
)
if "export_vertex_color" in props:
    opcoes["export_vertex_color"] = "ACTIVE"
if "export_active_vertex_color_when_no_material" in props:
    opcoes["export_active_vertex_color_when_no_material"] = True
bpy.ops.export_scene.gltf(**{k: v for k, v in opcoes.items() if k in props})
print("GLB:", glb, round(glb.stat().st_size / 1e6, 2), "MB")
# O bpy do PyPI não traz o Draco: comprime-se com o glTF Transform.
import subprocess  # noqa: E402

final = SAIDA / "penela.glb"
subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "penela.blend"))

# ---------------------------------------------------------------- capa e vistas de verificação

if "--vistas" in sys.argv:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--amostras=")), 24))
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Base Contrast" if "AgX - Base Contrast" in [l.identifier for l in sc.view_settings.bl_rna.properties["look"].enum_items] else "None"
    w = bpy.data.worlds.new("mundo")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.80, 0.76, 0.68, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.65
    sol = bpy.data.lights.new("sol", "SUN")
    sol.energy = 3.6
    sol.angle = math.radians(3)
    so = bpy.data.objects.new("sol", sol)
    sc.collection.objects.link(so)
    # sol a noroeste, como no relevo do atlas (azimute 315°, 50° de altura)
    from mathutils import Euler, Vector
    ALVO = float(cota(0, 0))
    alvo = Vector((0, 0, ALVO + 4))
    d = Vector((-math.sin(math.radians(45)), math.cos(math.radians(45)), 0)) * math.cos(math.radians(50))
    d.z = math.sin(math.radians(50))
    so.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    cam = bpy.data.cameras.new("cam")
    co = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    pasta = AQUI / "vistas"
    pasta.mkdir(exist_ok=True)

    def vista(nome, azimute, elev, dist, lente=50, orto=None, res=(1200, 800), destino=None, alvo_z=None, alvo_xy=(0, 0)):
        a, e = math.radians(azimute), math.radians(elev)
        al = Vector((alvo_xy[0], alvo_xy[1], alvo.z if alvo_z is None else alvo_z))
        pos = al + Vector((math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))) * dist
        co.location = pos
        co.rotation_euler = (al - pos).to_track_quat("-Z", "Y").to_euler()
        if orto:
            cam.type = "ORTHO"
            cam.ortho_scale = orto
        else:
            cam.type = "PERSP"
            cam.lens = lente
        cam.clip_end = 5000
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.render.filepath = str(destino or (pasta / f"{nome}.jpg"))
        bpy.ops.render.render(write_still=True)
        print("vista", nome)

    so_capa = "--capa" in sys.argv
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 84
    vista("capa", 225, 33, 540, lente=50, res=(1280, 800), destino=SAIDA / "capa.jpg", alvo_z=ALVO - 34)
    if not so_capa:
        vista("topo", 180, 89.9, 400, orto=200, res=(1000, 1000))
        vista("norte", 0, 30, 330)
        vista("sul", 180, 30, 330)
        vista("nascente", 90, 30, 330)
        vista("poente", 270, 30, 330)
        vista("obliqua_foto1", 268, 38, 220, lente=50)
        vista("obliqua_foto2", 312, 32, 230, lente=50)
        vista("muro_poente", 262, 12, 120, lente=40)
        vista("igreja_fachada", 228, 14, 48, lente=35, alvo_xy=tuple(P(-30, 10)), alvo_z=float(cota(*P(-30, 10))) + 8)
