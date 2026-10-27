"""Constrói a maqueta 3D do Castelo de Guimarães (estado atual), com o Paço dos Duques de Bragança,
a Igreja de São Miguel do Castelo e a estátua de D. Afonso Henriques, em Blender, e exporta-a para
public/maquetas/guimaraes/guimaraes.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely
    python pipeline/maquetas/guimaraes/construir.py            # GLB
    python pipeline/maquetas/guimaraes/construir.py --vistas   # GLB + capa + vistas de verificação

Dados: dados/fontes.json (recolher.py: plantas do OSM), dados/lidar.npz e dados/orto.jpg (lidar.py:
MDT, MDS, classes e cor da nuvem de pontos do LiDAR 2025 da DGT) e src/data/maquetas/guimaraes.json,
com a leitura das fontes e o estado de cada elemento. O castelo é traçado no próprio LiDAR (torres,
torre de menagem e panos), porque o OSM tem só o contorno exterior; as cotas e alturas medem-se no
LiDAR e o que ele não mede (espessuras, ameias, vãos, a estátua) vem do JSON e das fotografias.
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
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "guimaraes.json").read_text())
EL = {e["id"]: e for e in CFG["elementos"]}
SAIDA = RAIZ / "public" / "maquetas" / "guimaraes"

LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
# Centro da maqueta em metros a partir da origem (torre de menagem): entre o castelo, a Igreja de
# São Miguel, o Paço dos Duques e a estátua de D. Afonso Henriques, que ficam todos dentro do círculo
CENTRO = np.array([-40.0, -90.0])
RAIO = 165.0
ZREF = 200.0  # cota que fica em z = 0
rng = np.random.default_rng(1128)

MUR = CFG["muralha"]
PARAPEITO, MERLAO = MUR["parapeito"], MUR["merlao"]
# Os merlões de Guimarães são chanfrados (rematam em ponta, SIPA e fotografias): o topo medido no
# LiDAR fica a meio da ponta, porque a nuvem de pontos (c. 4 pontos/m²) raramente apanha o vértice.
ALT_MERLAO = MERLAO["altura"] + 0.5 * MERLAO.get("ponta", 0.0)
# O traçado do OSM está desviado c. 1 m para oeste em relação às plantas dos edifícios no LiDAR
# (melhor sobreposição das plantas do OSM com as células de classe edifício, medida no lidar.py)
DESVIO_OSM = np.array([1.0, 0.0])


# ---------------------------------------------------------------- geografia

def P(x, y):
    """Metros a partir da origem oficial -> coordenadas da cena."""
    return np.array([x, y], dtype=float) - CENTRO


def ll(c):
    return np.array([(c[0] - LON0) * KX, (c[1] - LAT0) * KY]) - CENTRO + DESVIO_OSM


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


def aneis(f):
    """Anéis de um polígono do OSM (exterior primeiro), em coordenadas da cena."""
    return [np.array([ll(c) for c in r[:-1]]) for r in f["geometria"]["coordinates"]]


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




# ---------------------------------------------------------------- terreno

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
    adarve = topo - (PARAPEITO + ALT_MERLAO if ameias else PARAPEITO)
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


def ameias_ao_longo(m, linha, z, fechada=False, mat=0, prof=0.6, dims=None):
    """Merlões ao longo de uma polilinha; z é a cota do parapeito (escalar ou por vértice).
    O ritmo segue o pano todo, não cada segmento, para a linha densificada não mudar o desenho.
    Com "ponta" nas medidas, cada merlão remata numa pirâmide (merlão chanfrado)."""
    MERLAO = dims or globals()["MERLAO"]
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
        if MERLAO.get("ponta"):
            w = np.array([-u[1], u[0]])
            hl, hp = MERLAO["largura"] / 2, prof / 2
            q = [c - u * hl - w * hp, c + u * hl - w * hp, c + u * hl + w * hp, c - u * hl + w * hp]
            zt = zc + MERLAO["altura"]
            for k in range(4):
                m.face([(*q[k], zt), (*q[(k + 1) % 4], zt), (*c, zt + MERLAO["ponta"])], mat)


def torre(m, base, pe, topo, mat=0):
    """Torre maciça com parapeito e ameias no topo."""
    if area(base) < 0:
        base = base[::-1]
    terraco = topo - PARAPEITO - ALT_MERLAO
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




# ---------------------------------------------------------------- materiais e texturas

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
    """Cantaria de granito em fiadas regulares (3 m por repetição), como nas fotografias."""
    r = np.random.default_rng(1310)
    tex = np.zeros((n, n, 3))
    m_px = n / 3.0
    y = 0
    while y < n:
        hf = int(r.uniform(0.40, 0.55) * m_px)
        x = -int(r.uniform(0, 0.6) * m_px)
        while x < n:
            wf = int(r.uniform(0.60, 1.20) * m_px)
            base = np.array([0.70, 0.68, 0.64]) * r.uniform(0.92, 1.06) + r.normal(0, 0.006, 3)
            x0, x1 = max(x, 0), min(x + wf, n)
            y1 = min(y + hf, n)
            if x1 > x0:
                tex[y:y1, x0:x1] = base
                tex[y:y1, x0:min(x0 + 2, x1)] *= 0.88
            x += wf
        tex[y:min(y + 2, n)] *= 0.86
        y += hf
    yy, xx = np.mgrid[0:n, 0:n] / n
    grao = fbm(xx * 40, yy * 40, 1.5, 21)[..., None]
    tex = tex * (0.92 + 0.12 * grao)
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



# ---------------------------------------------------------------- planta do castelo (traçada no LiDAR)


def poligono_lidar(semente, raio, limiar, folga=0.25):
    """Planta retangular de uma torre: retângulo mínimo das células do MDS acima de `limiar` (cota
    absoluta), sem vegetação, ligadas à célula mais próxima da `semente` (metros a partir da origem)."""
    import shapely
    from scipy import ndimage as ndi
    s = P(*semente)
    X, Y = np.meshgrid(LXS, LXS)
    m = (np.hypot(X - s[0], Y - s[1]) < raio) & (MDS + ZREF > limiar) & ~VEG
    lab, _ = ndi.label(m)
    i = np.argmin(np.where(m, np.hypot(X - s[0], Y - s[1]), 1e9))
    q = lab == lab.ravel()[i]
    g = shapely.MultiPoint(np.stack([X[q], Y[q]], 1)).minimum_rotated_rectangle.buffer(folga, join_style="mitre")
    b = np.array(g.exterior.coords)[:-1]
    return b if area(b) > 0 else b[::-1]


def retangulo(centro, larg, prof, ang):
    """Retângulo (anti-horário) centrado num ponto (metros a partir da origem), rodado `ang` graus."""
    a = math.radians(ang)
    u, w = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c = P(*centro)
    return np.array([c - u * larg / 2 - w * prof / 2, c + u * larg / 2 - w * prof / 2,
                     c + u * larg / 2 + w * prof / 2, c - u * larg / 2 + w * prof / 2])


# Torres que se destacam bem das muralhas no MDS: semente, raio de procura e cota mínima
TORRES_LIDAR = {
    "torre_nordeste": ((7.8, 21.3), 6, 254.5),
    "torre_sul": ((18.9, -25.4), 6, 254.5),
    "torre_sudoeste": ((-5.6, -19.9), 5, 254.5),
    "torre_noroeste": ((-16.5, 15.1), 5, 253.5),
}
torres = {k: poligono_lidar(*v) for k, v in TORRES_LIDAR.items()}
# As torres das duas portas quase não sobem acima do adarve: plantas desenhadas sobre o MDS
# (c. 4,5 m de lado, como no contorno do OSM) e topo medido.
torres["porta_principal_n"] = retangulo((18.0, 9.0), 4.2, 4.2, 20)
torres["porta_principal_s"] = retangulo((20.0, 2.8), 4.2, 4.2, 20)
torres["porta_traicao_n"] = retangulo((-21.4, 2.8), 4.6, 4.6, 33)
torres["porta_traicao_s"] = retangulo((-17.0, -4.6), 4.6, 4.6, 33)
MENAGEM = poligono_lidar((1.7, -0.3), 10, 262.0)

# Panos da cerca, linha do meio de cada um (metros a partir da origem), desenhada sobre o MDS de
# 50 cm de torre em torre, no sentido anti-horário (o exterior fica à direita); o perfil() encosta-a
# depois à crista medida.
PANOS = {
    "muralha_nascente": ([(19.3, -21.0), (19.4, -14.0), (19.2, -6.0), (19.2, 0.8)], 2.6),
    "muralha_nascente_norte": ([(17.4, 10.8), (15.0, 13.6), (12.3, 17.2)], 2.6),
    "alcacova": ([(4.0, 21.6), (-4.0, 19.5), (-12.6, 17.2)], 2.0),
    "muralha_noroeste": ([(-19.0, 12.4), (-21.6, 8.6), (-22.6, 5.0)], 2.6),
    "muralha_sudoeste": ([(-15.9, -6.6), (-11.0, -12.0), (-7.4, -16.6)], 2.6),
    "muralha_sul": ([(-3.0, -21.2), (2.0, -23.3), (8.0, -25.3), (14.6, -26.5)], 2.6),
}
CONTORNO = np.array([ll(c) for c in osm("739e68f7")["geometria"]["coordinates"][0][:-1]])  # face exterior (OSM)
if area(CONTORNO) < 0:
    CONTORNO = CONTORNO[::-1]
PORTA_PRINCIPAL = (P(18.6, 6.0), 20)  # vão entre as duas torres, virado a nascente
PORTA_TRAICAO = (P(-19.3, -0.8), 33 + 90)
ESTATUA = P(-108.6, -112.4)  # pedestal e figura no MDS (5,5 m acima do chão), onde o Overture a situa


def faces_retangulo(b):
    """Arestas de um retângulo com a normal exterior de cada uma."""
    out = []
    for i in range(len(b)):
        a, c = b[i], b[(i + 1) % len(b)]
        t = unit(c - a)
        out.append((a, c, t, np.array([t[1], -t[0]])))
    return out


def fresta(m, a, c, n, s, z, alt=1.4, larg=0.22, mat=1):
    """Fresta (seteira) escura numa face, a uma fração s da aresta a-c."""
    t = unit(c - a)
    p = a + (c - a) * s + n * 0.03
    m.face([(*(p - t * larg / 2), z), (*(p + t * larg / 2), z), (*(p + t * larg / 2), z + alt), (*(p - t * larg / 2), z + alt)], mat)


def telhado_ate(m, base, z, cume, mat=1):
    """Telhado de quatro águas com o cume à cota dada (medida no MDS)."""
    b, L, W = _retangulo(base)
    h = max(cume - z, 0.6)
    telhado_4aguas(m, b, z, math.degrees(math.atan(h / (W / 2))), mat)


def arco_quebrado(m, centro, u, normal, largura, soleira, nascenca, mat):
    """Vão em arco quebrado (ogival), como reentrância escura na face."""
    r = largura
    pts = [(-largura / 2, soleira), (largura / 2, soleira), (largura / 2, soleira + nascenca)]
    for k in range(1, 7):  # arco do lado direito, centro no arranque esquerdo
        a = math.pi / 3 * k / 6
        pts.append((-largura / 2 + r * math.cos(a), soleira + nascenca + r * math.sin(a)))
    for k in range(5, 0, -1):  # lado esquerdo, simétrico
        a = math.pi / 3 * k / 6
        pts.append((largura / 2 - r * math.cos(a), soleira + nascenca + r * math.sin(a)))
    pts.append((-largura / 2, soleira + nascenca))
    m.face([(*(centro + u * px + normal * 0.03), z) for px, z in pts], mat)


# ---------------------------------------------------------------- Paço dos Duques


def anel_ccw(a, ccw=True):
    return a if (area(a) > 0) == ccw else a[::-1]


def paco():
    """Paço dos Duques de Bragança: planta do OSM (corpo e pátio); altura das fachadas, telhados e
    chaminés medidos no MDS; ameias, janelas, portas e arcarias do pátio pelas fotografias."""
    import shapely
    from matplotlib.path import Path as Caminho
    from scipy import ndimage as ndi
    from scipy.spatial import Delaunay
    ext, pat = aneis(osm("f4db2746", "edificios"))
    ext = anel_ccw(simplificar(ext, 0.3))
    pat = anel_ccw(simplificar(pat, 0.3), ccw=False)  # o pátio no sentido horário: o edifício à esquerda
    ESP, PAR = 1.2, 0.5  # largura do caminho de ronda junto às ameias e do parapeito (estimativas)
    m = Malha()  # 0 pedra, 1 telha, 2 sombra, 3 cantaria, 4 madeira, 5 tijolo
    med = {}

    def medir_anel(anel):
        """Topo das ameias de 1,5 em 1,5 m: o mais alto de três faixas junto à face (onde a planta do OSM
        sai da fachada, a primeira faixa cai fora do edifício), filtrado ao longo da fachada."""
        lin = densificar(anel, 1.5, True)
        topo = np.nanmax([[medir(centro=p, raio=0.8, q=90) for p in desvio(lin, d, True)] for d in (0.5, 1.0, 1.6)], axis=0)
        ok = ~np.isnan(topo)
        idx = np.arange(len(lin))
        topo = np.interp(idx, idx[ok], topo[ok], period=len(lin))
        mm = mediana_movel(topo, 9, True)
        topo = mediana_movel(np.where(topo < mm - 1.5, mm, topo), 5, True)
        chao = cota(*desvio(lin, -1.0, True).T)
        return lin, topo, chao

    for nome, anel in (("fachadas", ext), ("patio", pat)):
        lin, topo, chao = medir_anel(anel)
        n = len(lin)
        z0 = float(chao.min()) - 1.5
        caminho = topo - MERLAO_PACO["altura"] - PAR - 0.3
        inn = desvio(lin, ESP, True)
        par = desvio(lin, PAR, True)
        for i in range(n):
            j = (i + 1) % n
            m.face([(*lin[i], z0), (*lin[j], z0), (*lin[j], topo[j] - MERLAO_PACO["altura"]), (*lin[i], topo[i] - MERLAO_PACO["altura"])], 0)
            m.face([(*par[i], caminho[i]), (*par[j], caminho[j]), (*inn[j], caminho[j]), (*inn[i], caminho[i])], 0)
            m.face([(*par[j], caminho[j]), (*par[i], caminho[i]), (*par[i], topo[i] - MERLAO_PACO["altura"]), (*par[j], topo[j] - MERLAO_PACO["altura"])], 0)
            m.face([(*lin[i], topo[i] - MERLAO_PACO["altura"]), (*lin[j], topo[j] - MERLAO_PACO["altura"]), (*par[j], topo[j] - MERLAO_PACO["altura"]), (*par[i], topo[i] - MERLAO_PACO["altura"])], 0)
        ameias_ao_longo(m, desvio(lin, 0.22, True), topo - MERLAO_PACO["altura"], True, 0, prof=0.42, dims=MERLAO_PACO)
        med[nome] = {"topo": [float(topo.min() + ZREF), float(topo.max() + ZREF)],
                     "altura": [float(np.percentile(topo - chao, 10)), float(np.percentile(topo - chao, 90))]}
        if nome == "fachadas":
            CAM_EXT = (lin, caminho)
        else:
            CAM_PAT = (lin, caminho)

    # Telhados: o Paço tem um telhado de quatro águas (ou pirâmide) por corpo, separados por paredes.
    # Cada corpo é uma bacia do MDS alisado (sem chaminés) à volta de um cume; em cada bacia o telhado
    # sobe por igual desde a borda até ao cume medido, como num telhado de águas iguais.
    from rasterio import features
    from scipy.spatial import cKDTree
    from skimage.segmentation import watershed
    dominio = shapely.Polygon(desvio(ext, ESP + 0.05, True), [desvio(pat, ESP + 0.05, True)])
    X, Y = np.meshgrid(LXS, LXS)
    xy = np.stack([X.ravel(), Y.ravel()], 1)
    caminho_dentro = Caminho(np.array(dominio.exterior.coords))
    pat_c = Caminho(np.array(dominio.interiors[0].coords))
    dom = (caminho_dentro.contains_points(xy) & ~pat_c.contains_points(xy)).reshape(X.shape)
    sup = ndi.gaussian_filter(ndi.median_filter(MDS, size=5), 1.0)
    cumes = (sup == ndi.maximum_filter(sup, size=9)) & dom
    marc, _ = ndi.label(ndi.binary_dilation(cumes, iterations=4) & dom)
    seg = watershed(-sup, marc, mask=dom)
    # borda do telhado: o caminho de ronda mais próximo
    bl = np.vstack([CAM_EXT[0], CAM_PAT[0]])
    bz = np.concatenate([CAM_EXT[1], CAM_PAT[1]])
    arvore_b = cKDTree(bl)
    tf = __import__("rasterio").transform.Affine(LPASSO, 0, LXS[0] - LPASSO / 2, 0, LPASSO, LXS[0] - LPASSO / 2)
    V, T = [], []
    corpos = []
    for k in range(1, seg.max() + 1):
        msk = seg == k
        if msk.sum() * LPASSO ** 2 < 12:
            continue
        geo = [shapely.geometry.shape(g_) for g_, v_ in features.shapes(msk.astype(np.uint8), mask=msk, transform=tf) if v_ == 1]
        G = max(geo, key=lambda x: x.area).simplify(0.6).intersection(dominio)
        if G.geom_type != "Polygon" or G.area < 12:
            continue
        G = shapely.Polygon(G.exterior)
        bd = densificar(np.array(G.exterior.coords)[:-1], 0.7, True)
        _, kb = arvore_b.query(bd)
        zbase = float(np.median(bz[kb]))
        cume = float(np.percentile(sup[msk], 97))
        if cume < zbase + 0.8:
            cume = zbase + 0.8
        R = G.minimum_rotated_rectangle
        if G.area / R.area > 0.8:
            # corpo retangular: telhado de quatro águas exato, com o cume medido
            r4 = np.array(R.exterior.coords)[:-1]
            r4 = r4 if area(r4) > 0 else r4[::-1]
            mt = Malha()
            telhado_ate(mt, r4, zbase, cume, mat=0)
            b0 = len(V)
            V.extend([list(v_) for v_ in mt.v])
            T.extend([[b0 + i for i in f_] for f_ in mt.f])
            b1 = len(V)
            V.extend([[q[0], q[1], zbase] for q in r4] + [[q[0], q[1], zbase - 1.6] for q in r4])
            for i in range(4):
                j = (i + 1) % 4
                T.append([b1 + i, b1 + j, b1 + 4 + j, b1 + 4 + i])
            corpos.append((round(G.area), round(zbase + ZREF, 1), round(cume + ZREF, 1)))
            continue
        x0, y0, x1, y1 = G.bounds
        gx, gy = np.meshgrid(np.arange(x0, x1, 0.7), np.arange(y0, y1, 0.7))
        g = np.stack([gx.ravel(), gy.ravel()], 1)
        cg = Caminho(np.array(G.exterior.coords))
        g = g[cg.contains_points(g, radius=-0.25 if area(np.array(G.exterior.coords)) > 0 else 0.25)]
        pts = np.vstack([bd, g])
        d = np.array([G.exterior.distance(shapely.Point(*q)) for q in pts])
        z = zbase + (cume - zbase) * d / max(d.max(), 1e-6)
        tri = Delaunay(pts).simplices
        tri = tri[cg.contains_points(pts[tri].mean(1))]
        b0 = len(V)
        V.extend(np.column_stack([pts, z]).tolist())
        T.extend((tri + b0).tolist())
        # saia vertical na borda, para não abrir frestas entre corpos com bordas a cotas diferentes
        n = len(bd)
        b1 = len(V)
        V.extend([[q[0], q[1], zbase - 1.6] for q in bd])
        for i in range(n):
            j = (i + 1) % n
            T.append([b0 + i, b0 + j, b1 + j])
            T.append([b0 + i, b1 + j, b1 + i])
        corpos.append((round(G.area), round(zbase + ZREF, 1), round(cume + ZREF, 1)))
    me = bpy.data.meshes.new("telhados_paco")
    me.from_pydata(V, [], T)
    me.update()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.01)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(2.0), verts=bm.verts, edges=bm.edges)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f_ in bm.faces:
        if f_.normal.z < -0.5:
            f_.normal_flip()
    bm.to_mesh(me)
    bm.free()
    me.materials.append(MATS["telha"])
    ob = bpy.data.objects.new("Paco_Telhados", me)
    bpy.context.scene.collection.objects.link(ob)
    ob["elemento"] = "paco"
    ob.parent = GRUPOS["Paco"]
    print("  paço:", len(corpos), "corpos de telhado (área, borda, cume):", corpos)
    med["corpos_telhado"] = len(corpos)
    V = np.array(V)

    # Chaminés: picos do MDS que sobem mais de 1,2 m acima de tudo num anel de 2 a 3,5 m à volta
    # (assim não se confundem com cumeeiras nem com as ameias). Torreões cónicos: os picos largos
    # junto aos cunhais exteriores.
    yy, xx = np.mgrid[-7:8, -7:8]
    dd = np.hypot(xx, yy) * LPASSO
    viz = ndi.maximum_filter(MDS, footprint=(dd >= 2.0) & (dd <= 3.5))
    pico = (MDS == ndi.maximum_filter(MDS, size=3)) & (MDS > viz + 1.2)
    corpo = Caminho(ext).contains_points(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)
    corpo &= ~Caminho(pat).contains_points(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)
    pico &= corpo
    lab, k = ndi.label(MDS - sup > 0.8)
    torreoes, chamines = [], []
    for r_, c_ in zip(*np.nonzero(pico)):
        p = np.array([LXS[c_], LXS[r_]])
        larga = (lab == lab[r_, c_]).sum() * LPASSO ** 2 if lab[r_, c_] else 0
        canto = min(np.hypot(*(ext - p).T))
        if canto < 2.2 and larga > 2.5:
            torreoes.append((p, float(MDS[r_, c_])))
        else:
            chamines.append((p, float(MDS[r_, c_])))
    chamines = [c for c in chamines if all(np.linalg.norm(c[0] - t[0]) > 2.0 for t in torreoes)]
    mc = Malha()
    for p, topo in chamines:
        z0 = float(amostra(sup, *p)) - 0.6
        oct_ = [p + 0.42 * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        mc.prisma(oct_, z0, topo - 0.25, 0)
        mc.prisma([p + 0.52 * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)], topo - 0.25, topo, 0)
    for p, topo in torreoes:
        _, qx, qy = distancia(np.array([p[0]]), np.array([p[1]]), ext, True)
        zp = float(medir(centro=np.array([qx[0], qy[0]]), raio=1.2, q=50))
        circ = [p + 1.0 * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
        mc.prisma(circ, zp - 4.0, zp, 1)
        for i in range(12):
            mc.face([(*circ[i], zp), (*circ[(i + 1) % 12], zp), (*p, topo)], 2)
    mc.objeto("Paco_Chamines", "paco", ["tijolo", "pedra", "telha_viva"], GRUPOS["Paco"], uv=None)
    med["chamines"] = len(chamines)
    med["torreoes"] = len(torreoes)
    med["cumeeiras"] = float(np.percentile(V[:, 2], 99) + ZREF)

    # Janelas das fachadas exteriores (fotografias): no andar nobre, janelas de cruz com moldura de
    # cantaria; em baixo, frestas pequenas. Porta principal em arco no meio da fachada noroeste.
    lin, topo, chao = medir_anel(ext)
    for i in range(len(ext)):
        a, c = ext[i], ext[(i + 1) % len(ext)]
        L = np.linalg.norm(c - a)
        if L < 6:
            continue
        t = unit(c - a)
        n_ = np.array([t[1], -t[0]])
        k = int((L - 3) // 4.6)
        zc = float(np.median(chao[[int(np.argmin(np.hypot(*(lin - q).T))) for q in (a, c)]]))
        zt = float(np.median(topo[[int(np.argmin(np.hypot(*(lin - q).T))) for q in (a, (a + c) / 2, c)]]))
        h = zt - zc
        for s in (np.linspace(0, 1, k + 2)[1:-1] if k > 0 else []):
            q = a + (c - a) * s
            for zz, lj, hj in ((zc + h * 0.6, 1.3, 2.0), (zc + h * 0.33, 1.1, 1.6), (zc + h * 0.12, 0.45, 0.65)):
                if zz + hj > zt - 1.5:
                    continue
                m.face([(*(q - t * (lj / 2 + 0.18) + n_ * 0.03), zz - 0.18), (*(q + t * (lj / 2 + 0.18) + n_ * 0.03), zz - 0.18),
                        (*(q + t * (lj / 2 + 0.18) + n_ * 0.03), zz + hj + 0.18), (*(q - t * (lj / 2 + 0.18) + n_ * 0.03), zz + hj + 0.18)], 3)
                m.face([(*(q - t * lj / 2 + n_ * 0.05), zz), (*(q + t * lj / 2 + n_ * 0.05), zz),
                        (*(q + t * lj / 2 + n_ * 0.05), zz + hj), (*(q - t * lj / 2 + n_ * 0.05), zz + hj)], 2)
    ip = max(range(len(ext)), key=lambda i: np.linalg.norm(ext[(i + 1) % len(ext)] - ext[i]) * (unit(np.array([(ext[(i + 1) % len(ext)] - ext[i])[1], -(ext[(i + 1) % len(ext)] - ext[i])[0]])) @ unit(np.array([-1.0, 0.6])) > 0.5))
    a, c = ext[ip], ext[(ip + 1) % len(ext)]
    t = unit(c - a)
    n_ = np.array([t[1], -t[0]])
    pm = (a + c) / 2
    sole = float(cota(*(pm + n_ * 1.5)))
    arco(m, pm + n_ * 0.03, t, n_, 3.0, sole, 2.6, 0.1, 3)
    arco(m, pm + n_ * 0.06, t, n_, 2.3, sole, 2.3, 0.1, 2)
    med["porta_principal_soleira"] = sole + ZREF

    # Arcarias do pátio (fotografias): arcos quebrados no piso térreo e galeria no andar de cima
    lin_p, topo_p, chao_p = medir_anel(pat)
    for i in range(len(pat)):
        a, c = pat[i], pat[(i + 1) % len(pat)]
        L = np.linalg.norm(c - a)
        t = unit(c - a)
        n_ = np.array([t[1], -t[0]])  # para o pátio (o anel do pátio está no sentido horário)
        k = max(1, int(L // 3.6))
        zc = float(np.median(cota(*np.array([a, c, (a + c) / 2]).T + n_[:, None] * 1.5)))
        for s in (np.arange(k) + 0.5) / k:
            q = a + (c - a) * s
            arco_quebrado(m, q, t, n_, 2.4, zc, 1.6, 2)
            m.caixa(q + n_ * 0.04, t, 1.8, 0.05, zc + 5.6, zc + 8.0, 2)
            m.caixa(q + n_ * 0.12 + t * (L / k / 2), t, 0.35, 0.3, zc + 5.4, zc + 8.2, 3)  # colunas da galeria
    m.objeto("Paco_Corpo", "paco", ["pedra", "telha", "sombra", "cantaria", "madeira"], GRUPOS["Paco"], uv=3.0)
    MEDIDAS["paco"] = med
    return ext, pat


# ---------------------------------------------------------------- Igreja de São Miguel do Castelo


def igreja():
    """Planta e cumeeiras no LiDAR (células de classe edifício): nave e capela-mor mais baixa a
    nascente. Portal, fresta, cruz e porta lateral pelas fotografias."""
    import shapely
    from scipy import ndimage as ndi
    X, Y = np.meshgrid(LXS, LXS)
    c0 = P(-46.0, -71.0)
    q = (np.hypot(X - c0[0], Y - c0[1]) < 14) & (MDS - MDT > 1.5) & (CLS == 6)
    lab, k = ndi.label(q)
    q = lab == (np.argmax(ndi.sum(q, lab, range(1, k + 1))) + 1)
    r_ = np.array(shapely.MultiPoint(np.stack([X[q], Y[q]], 1)).minimum_rotated_rectangle.exterior.coords)[:4]
    e = [np.linalg.norm(r_[(i + 1) % 4] - r_[i]) for i in range(4)]
    i0 = int(np.argmax(e))
    a0, a1 = r_[i0], r_[(i0 + 1) % 4]
    u = unit(a1 - a0)
    v = np.array([-u[1], u[0]])
    if (r_.mean(0) - a0) @ v < 0:
        a0, a1 = a0 + (r_.mean(0) - a0) @ v * 2 * v, a1 + (r_.mean(0) - a0) @ v * 2 * v
    # a capela-mor é a ponta mais baixa: fica em a0 (u cresce para a fachada)
    A = np.stack([X[q], Y[q]], 1) - a0
    Lr = np.linalg.norm(a1 - a0)
    if np.percentile(MDS[q][A @ u < 3], 95) > np.percentile(MDS[q][A @ u > Lr - 3], 95):
        a0, a1 = a1, a0
        u = -u
        v = -v
    if (r_.mean(0) - a0) @ v < 0:
        v = -v
    L = np.linalg.norm(a1 - a0)
    W = e[(i0 + 1) % 4]
    print("  igreja: planta", round(L, 1), "x", round(W, 1))
    p = lambda uu, vv, z: (*(a0 + u * uu + v * vv), z)
    A = np.stack([X[q], Y[q]], 1) - a0
    au, av, zz = A @ u, A @ v, MDS[q]

    def faixa(u0, u1, v0, v1, qq=90):
        s = (au >= u0) & (au < u1) & (av >= v0) & (av < v1)
        return float(np.percentile(zz[s], qq)) if s.sum() > 2 else float("nan")

    # capela-mor: onde o cume é mais baixo, do lado nascente
    perf = np.array([faixa(s, s + 1, 0, W, 98) for s in np.arange(0, L - 1, 1.0)])
    alta = np.nanmax(perf)
    cm = float(np.argmax(perf > alta - 0.8))  # comprimento da capela-mor
    chao = float(np.percentile(MDT[q], 10))
    z0 = chao - 1.0
    m = Malha()  # 0 pedra, 1 telha, 2 sombra, 3 cantaria, 4 madeira
    res = {}
    for nome, (u0, u1, v0, v1) in (("nave", (cm, L - 0.2, 0.35, W - 0.35)), ("capela_mor", (0.2, cm + 0.3, 1.6, W - 1.6))):
        cume = faixa(u0 + 0.5, u1 - 0.5, (v0 + v1) / 2 - 0.8, (v0 + v1) / 2 + 0.8, 95)
        beiral = min(faixa(u0 + 0.5, u1 - 0.5, v0, v0 + 0.8, 50), faixa(u0 + 0.5, u1 - 0.5, v1 - 0.8, v1, 50))
        if math.isnan(beiral) or beiral > cume - 1.0:
            beiral = cume - 0.32 * (v1 - v0)  # pendente de c. 33 %
        vm = (v0 + v1) / 2
        ins = 0.25  # o beirado sai um pouco das paredes
        m.prisma([np.array(p(u0, v0 + ins, 0)[:2]), np.array(p(u1, v0 + ins, 0)[:2]), np.array(p(u1, v1 - ins, 0)[:2]), np.array(p(u0, v1 - ins, 0)[:2])], z0, beiral, 0, topo=False)
        m.face([p(u0, v0, beiral), p(u1, v0, beiral), p(u1, vm, cume), p(u0, vm, cume)], 1)
        m.face([p(u0, vm, cume), p(u1, vm, cume), p(u1, v1, beiral), p(u0, v1, beiral)], 1)
        for uu in (u0, u1):  # empenas
            m.face([p(uu, v0 + ins, beiral), p(uu, v1 - ins, beiral), p(uu, vm, cume)], 0)
        res[nome] = {"cumeeira": cume + ZREF, "beirado": beiral + ZREF, "comprimento": u1 - u0, "largura": v1 - v0}
    # fachada poente (fotografias): portal de arco pleno com arquivolta e porta verde, fresta por cima,
    # cruz no vértice da empena; porta lateral a sul.
    nf = u
    fc = a0 + u * (L - 0.2) + v * (W / 2)
    arco(m, fc + nf * 0.02, v, nf, 2.9, chao, 2.4, 0.1, 3)
    arco(m, fc + nf * 0.05, v, nf, 1.7, chao, 2.1, 0.1, 4)
    m.caixa(fc + nf * 0.04, v, 0.18, 0.05, chao + 4.4, chao + 5.4, 2)
    ct = res["nave"]["cumeeira"] - ZREF
    m.caixa(fc + nf * 0.1, v, 0.16, 0.16, ct, ct + 1.0, 3)
    m.caixa(fc + nf * 0.1, v, 0.6, 0.16, ct + 0.55, ct + 0.72, 3)
    ls = a0 + u * (cm + (L - cm) * 0.55) + v * 0.33
    m.caixa(ls - v * 0.03, u, 1.1, 0.05, chao, chao + 2.2, 2)
    m.objeto("Igreja_Sao_Miguel", "igreja", ["pedra", "telha", "sombra", "cantaria", "madeira_verde"], GRUPOS["Igreja"], uv=3.0)
    res["chao"] = chao + ZREF
    MEDIDAS["igreja"] = res
    return np.array([a0, a0 + u * L, a0 + u * L + v * W, a0 + v * W])


# ---------------------------------------------------------------- estátua de D. Afonso Henriques


def estatua():
    """Pedestal de granito e figura de bronze: posição e altura total no LiDAR (MDS 5,5 m acima do
    chão); a divisão entre pedestal e figura e a forma da figura são esquemáticas (fotografia)."""
    chao = float(cota(*ESTATUA))
    topo = float(np.max(celulas(centro=ESTATUA, raio=0.8, sem_veg=False)))  # a nuvem classifica-a como vegetação
    alt = topo - chao
    m = Malha()  # 0 cantaria, 1 bronze
    c = ESTATUA
    frente = unit(P(-47, -162) - c)  # virada para o Paço (fotografia)
    lado = np.array([-frente[1], frente[0]])
    m.caixa(c, frente, 3.4, 3.4, chao - 0.5, chao + 0.3, 0)
    m.caixa(c, frente, 2.6, 2.6, chao + 0.3, chao + 0.6, 0)
    zp = chao + alt * 0.56  # topo do pedestal
    m.caixa(c, frente, 1.7, 1.7, chao + 0.6, zp - 0.35, 0)
    m.caixa(c, frente, 2.0, 2.0, zp - 0.35, zp, 0)
    hf = topo - zp  # figura: corpo com manto, cabeça com elmo, escudo e espada
    corpo = [c + frente * dx + lado * dy for dx, dy in ((-0.35, -0.42), (0.35, -0.42), (0.35, 0.42), (-0.35, 0.42))]
    cima = [c + frente * dx * 0.7 + lado * dy * 0.75 for dx, dy in ((-0.35, -0.42), (0.35, -0.42), (0.35, 0.42), (-0.35, 0.42))]
    zo = zp + hf * 0.8
    for i in range(4):
        j = (i + 1) % 4
        m.face([(*corpo[i], zp), (*corpo[j], zp), (*cima[j], zo), (*cima[i], zo)], 1)
    m.face([(*p_, zo) for p_ in cima], 1)
    m.caixa(c, frente, 0.34, 0.34, zo, topo - 0.05, 1)
    piramide(m, c, 0.22, topo - 0.1, topo, 1)
    m.caixa(c + frente * 0.45 + lado * 0.35, lado, 0.7, 0.1, zp + hf * 0.25, zp + hf * 0.7, 1)  # escudo
    m.caixa(c + frente * 0.3 - lado * 0.45, lado, 0.1, 0.1, zp + 0.1, zp + hf * 0.55, 1)  # espada
    m.objeto("Estatua_Afonso_Henriques", "estatua", ["cantaria", "bronze"], GRUPOS["Estatua"], uv=2.0)
    MEDIDAS["estatua"] = {"chao": chao + ZREF, "topo": topo + ZREF, "altura_total": alt}


# ---------------------------------------------------------------- cena


def limpar():
    bpy.ops.wm.read_factory_settings(use_empty=True)


limpar()
RAIZ_OB = bpy.data.objects.new("Castelo_Guimaraes", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = LAT0, LON0
RAIZ_OB["centro_m"] = [float(CENTRO[0]), float(CENTRO[1])]
RAIZ_OB["cota_zero"] = ZREF
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
for g in ["Terreno", "Muralhas", "Torres", "Torre_Menagem", "Portas", "Paco", "Igreja", "Estatua",
          "Caminhos", "Vegetacao", "Vila"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

MERLAO_PACO = {"altura": 0.55, "largura": 0.55, "intervalo": 0.45}  # ameias miúdas do Paço (fotografias)

pedra = imagem("pedra", textura_pedra())
grao = imagem("grao", textura_grao())
material("pedra", (1, 1, 1), 0.92, pedra)
material("sombra", (0.12, 0.11, 0.10), 1.0)
material("cal", (0.94, 0.93, 0.89), 0.85)
material("telha", (0.66, 0.36, 0.25), 0.8)
material("telha_viva", (0.80, 0.42, 0.24), 0.8)
material("cantaria", (0.80, 0.78, 0.73), 0.9)
material("terreno", (1, 1, 1), 0.97, grao, vcor=True)
material("base", (0.33, 0.26, 0.20), 0.95)
material("caminho", (0.80, 0.76, 0.67), 0.95)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)
material("madeira", (0.42, 0.27, 0.15), 0.8)
material("madeira_verde", (0.20, 0.45, 0.35), 0.7)
material("tijolo", (0.62, 0.34, 0.24), 0.85)
material("bronze", (0.24, 0.26, 0.22), 0.5)
MATS["bronze"].node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.6

MEDIDAS = {}  # o que se mediu no LiDAR, para conferir e citar no guimaraes.json
PERFIS = {}  # perfis das muralhas (saida/perfis.npz), para conferência

print("terreno…")
construir_terreno()
for ob in bpy.data.objects:
    if ob.name in ("Terreno", "Base"):
        ob.parent = GRUPOS["Terreno"]

print("muralhas…")
for s_, (pts, esp) in PANOS.items():
    m = Malha()
    lin = np.array([P(*p) for p in pts])
    lin = np.vstack([lin[0] + unit(lin[0] - lin[1]) * 1.8, lin[1:-1], lin[-1] + unit(lin[-1] - lin[-2]) * 1.8]) if len(lin) > 2 else lin
    lin, topo, fora, dentro_ = perfil(lin, esp)  # as pontas entram 1,8 m nas torres
    adarve = muralha(m, lin, esp, topo, fora, dentro_)
    PERFIS[s_] = np.stack([lin[:, 0], lin[:, 1], topo, fora, dentro_], 1)
    if s_ == "alcacova":
        # Alcáçova (SIPA): paredes rasgadas por grandes janelas e chaminés, em dois registos, viradas
        # ao pátio. Vãos esquemáticos na face interior do pano norte.
        inn = desvio(lin, esp / 2, False)
        for s in (0.22, 0.42, 0.62, 0.82):
            i = int(s * (len(lin) - 1))
            a, c = inn[max(i - 1, 0)], inn[min(i + 1, len(inn) - 1)]
            t = unit(a - c)
            n_ = np.array([t[1], -t[0]])
            for zz, h in ((dentro_[i] + 1.2, 2.0), (dentro_[i] + 4.6, 2.2)):
                if zz + h < adarve[i] - 0.4:
                    m.caixa(inn[i] + n_ * 0.03, t, 1.3, 0.05, zz, zz + h, 1)
    m.objeto(s_, s_, ["pedra", "sombra"], GRUPOS["Muralhas"])
    alt = topo - fora
    MEDIDAS[s_] = {"topo": [float(topo.min() + ZREF), float(topo.max() + ZREF)],
                   "altura_exterior": [float(np.percentile(alt, 10)), float(np.percentile(alt, 90))],
                   "acima_do_chao_interior": [float(np.percentile(topo - dentro_, 10)), float(np.percentile(topo - dentro_, 90))],
                   "adarve": [float(adarve.min() + ZREF), float(adarve.max() + ZREF)],
                   "adarve_mediano": float(np.median(adarve) + ZREF)}

print("torres…")
TELHADOS = {"torre_nordeste", "torre_sul"}  # torres com telhado de quatro águas (SIPA, ortoimagem)
for tid, b in torres.items():
    m = Malha()
    topo = medir(desvio(b, 0.3, True) if abs(area(b)) > 6 else b, q=95)
    pe = float(np.min(cota(*densificar(desvio(b, -1.0, True), 1.0, True).T)))
    torre(m, b, pe, topo)
    terraco = topo - PARAPEITO - ALT_MERLAO
    med = {"topo": topo + ZREF, "pe": pe + ZREF, "altura_exterior": topo - pe}
    if tid in TELHADOS:
        cume = float(np.max(celulas(desvio(b, 1.2, True))))
        if cume < terraco + 0.6:
            cume = terraco + 1.6
        telhado_ate(m, desvio(b, 0.6, True), terraco, cume, mat=2)
        med["cume_telhado"] = cume + ZREF
    for a, c, t, n_ in faces_retangulo(b):
        if np.linalg.norm(c - a) > 3.5:
            fresta(m, a, c, n_, 0.5, pe + (terraco - pe) * 0.62)
    PERFIS["torre_" + tid] = b
    MEDIDAS[tid] = med
    if tid.startswith("porta_"):
        nome, grupo, el = tid.title(), GRUPOS["Portas"], tid.rsplit("_", 1)[0]
    else:
        nome, grupo, el = tid.title(), GRUPOS["Torres"], tid
    m.objeto(nome, el, ["pedra", "sombra", "telha"], grupo)

print("torre de menagem…")
m = Malha()
b = MENAGEM
topo = medir(desvio(b, 0.3, True), q=95)
pe = float(np.min(cota(*densificar(desvio(b, -1.0, True), 1.0, True).T)))
chao_med = float(np.median(cota(*densificar(desvio(b, -2.0, True), 1.0, True).T)))
torre(m, b, pe, topo)
terraco = topo - PARAPEITO - ALT_MERLAO
cume = float(np.percentile(celulas(desvio(b, 2.0, True), sem_veg=False), 99))  # sem o mastro da bandeira
telhado_ate(m, desvio(b, 0.6, True), terraco, cume, mat=2)
# três registos de frestas (SIPA) e a porta no segundo registo, na face virada ao adarve poente
fs = faces_retangulo(b)
for a, c, t, n_ in fs:
    for k, zz in enumerate((chao_med + 6.0, chao_med + 12.5, chao_med + 18.5)):
        for s in ((0.5,) if k != 1 else (0.32, 0.68)):
            fresta(m, a, c, n_, s, zz, alt=1.5)
fp = max(fs, key=lambda f: f[3] @ unit(np.array([-1.0, 0.8])))  # face noroeste
porta_z = chao_med + 7.4
m.caixa((fp[0] + fp[1]) / 2 + fp[3] * 0.03, fp[2], 1.1, 0.05, porta_z, porta_z + 2.1, 1)
m.objeto("Torre_Menagem", "torre_menagem", ["pedra", "sombra", "telha"], GRUPOS["Torre_Menagem"])
MEDIDAS["torre_menagem"] = {"topo_ameias": topo + ZREF, "cume": cume + ZREF, "pe": pe + ZREF, "chao": chao_med + ZREF,
                            "altura": cume - chao_med, "altura_ameias": topo - chao_med,
                            "planta": [float(np.linalg.norm(b[1] - b[0])), float(np.linalg.norm(b[2] - b[1]))]}
# passadiço de madeira do adarve poente à porta da torre (SIPA); traçado e cota estimados
m = Malha()
pm = (fp[0] + fp[1]) / 2 + fp[3] * 0.1
alvo = P(-14.5, 8.5)
d = unit(alvo - pm)
w = np.array([-d[1], d[0]])
L = np.linalg.norm(alvo - pm)
z_adarve = MEDIDAS["muralha_noroeste"]["adarve_mediano"] - ZREF
for i, s in enumerate(np.linspace(0, 1, 8)[:-1]):
    a_ = pm + d * L * s
    b_ = pm + d * L * (s + 1 / 7)
    za = porta_z + (z_adarve - porta_z) * s
    zb_ = porta_z + (z_adarve - porta_z) * (s + 1 / 7)
    m.prisma([a_ - w * 0.7, b_ - w * 0.7, b_ + w * 0.7, a_ + w * 0.7], [za - 0.25, zb_ - 0.25, zb_ - 0.25, za - 0.25], [za, zb_, zb_, za], 0)
for s in np.linspace(0.15, 0.85, 3):  # esteios até ao chão
    q = pm + d * L * s
    zq = porta_z + (z_adarve - porta_z) * s
    m.caixa(q, d, 0.25, 0.25, float(cota(*q)) - 0.3, zq - 0.25, 0)
m.objeto("Passadico", "passadico", ["madeira"], GRUPOS["Torre_Menagem"], uv=None)

print("portas…")
# Porta principal: vão em arco entre as duas torres, com a escadaria semicircular à frente (SIPA)
m = Malha()
c, ang = PORTA_PRINCIPAL
u = np.array([math.cos(math.radians(ang + 90)), math.sin(math.radians(ang + 90))])  # ao longo do pano
out = np.array([u[1], -u[0]])
if out @ np.array([1.0, 0]) < 0:
    out, u = -out, -u
sole = float(np.median(cota(*np.array([c - out * 1.5, c, c - out * 0.5]).T)))
fora_ = float(cota(*(c + out * 4.0)))
adarve = MEDIDAS["muralha_nascente"]["adarve_mediano"] - ZREF
vao = min(distancia(np.array([c[0]]), np.array([c[1]]), torres[t_], True)[0][0] for t_ in ("porta_principal_n", "porta_principal_s"))
m.caixa(c - out * 0.2, u, 2 * vao + 0.6, 2.4, sole - 1.0, adarve, 0)
arco(m, c + out * 1.0, u, out, 2.0, sole, 2.4, 0.1, 1)
arco(m, c - out * 1.42, u, -out, 2.0, sole, 2.4, 0.1, 1)
n_deg = max(2, int(round((sole - fora_) / 0.3)))
for k in range(n_deg):  # degraus em meia-lua
    r_ = 1.4 + (k + 1) * 0.55
    z1 = sole - k * (sole - fora_) / n_deg
    arc = [c + out * 1.1 + (u * math.cos(a) + out * math.sin(a)) * r_ for a in np.linspace(0, math.pi, 13)]
    m.prisma(arc, fora_ - 0.5, z1, 2)
m.objeto("Porta_Principal", "porta_principal", ["pedra", "sombra", "cantaria"], GRUPOS["Portas"])
MEDIDAS["porta_principal"] = {"soleira": sole + ZREF, "chao_fora": fora_ + ZREF, "degraus": n_deg}
# Porta da Traição: arco quebrado entre as duas torres do pano poente (SIPA)
m = Malha()
# o bloco vai de centro a centro das duas torres, para ficar preso a ambas
cn, cs = (np.asarray(torres[t_], float).mean(axis=0) for t_ in ("porta_traicao_n", "porta_traicao_s"))
c = (cn + cs) / 2
u = (cn - cs) / np.linalg.norm(cn - cs)
out = np.array([u[1], -u[0]])
if out @ np.array([-1.0, 0]) < 0:
    out, u = -out, -u
sole = float(np.median(cota(*np.array([c, c + out * 0.5, c - out * 0.5]).T)))
adarve = MEDIDAS["muralha_noroeste"]["adarve_mediano"] - ZREF
m.caixa(c, u, float(np.linalg.norm(cn - cs)), 2.6, sole - 1.0, adarve, 0)
arco_quebrado(m, c + out * 1.3, u, out, 1.5, sole, 1.6, 1)
arco_quebrado(m, c - out * 1.3, u, -out, 1.5, sole, 1.6, 1)
m.objeto("Porta_Traicao", "porta_traicao", ["pedra", "sombra"], GRUPOS["Portas"])
MEDIDAS["porta_traicao"] = {"soleira": sole + ZREF}

print("paço, igreja e estátua…")
PACO_EXT, PACO_PAT = paco()
IGREJA = igreja()
estatua()

print("vila…")
proprios = {"f4db2746", "cb136145", "0939c270", "739e68f7"}
CASTELO = CONTORNO
evitar = [CASTELO, PACO_EXT, IGREJA]
m = Malha()
pegadas = []
n_med = 0


def recortar(poly):
    """Corta pela borda da maqueta os edifícios que a atravessam (o centro tem de ficar dentro)."""
    import shapely
    if np.hypot(*poly.mean(0)) > RAIO - 4:
        return None
    if np.hypot(*poly.T).max() <= RAIO - 1:
        return poly
    g = shapely.Polygon(poly).intersection(shapely.Point(0, 0).buffer(RAIO - 1.0, 64))
    if g.geom_type != "Polygon" or g.area < 12:
        return None
    b = np.array(g.exterior.coords)[:-1]
    return b if area(b) > 0 else b[::-1]


def livre(poly):
    c = poly.mean(0)
    for z in evitar:
        if dentro(c[0], c[1], z) or distancia(poly[:, 0], poly[:, 1], z, True)[0].min() < 1.5:
            return False
    return np.linalg.norm(c - ESTATUA) > 4


for f in F["edificios"]:
    if f["id"][:8] in proprios or f["geometria"]["type"] != "Polygon":
        continue
    poly = recortar(geom(f))
    if poly is None or not livre(poly):
        continue
    h = f.get("altura") or ((f.get("pisos") or 2) * 3.0 + 0.6)
    n_med += casa(m, poly, h)[1]
    pegadas.append(poly)
n_osm = len(pegadas)
for c_ in json.loads((AQUI / "dados" / "casas.json").read_text()):
    poly = np.array(c_["planta"], float)
    if area(poly) < 0:
        poly = poly[::-1]
    poly = recortar(poly)
    if poly is None or not livre(poly):
        continue
    n_med += casa(m, poly, 6.0)[1]
    pegadas.append(poly)
m.objeto("Edificios_Vila", "vila", ["cal", "telha"], GRUPOS["Vila"], uv=None)
MEDIDAS["vila"] = {"edificios": len(pegadas), "do_osm": n_osm, "do_lidar": len(pegadas) - n_osm, "medidos": n_med}

print("caminhos…")
LARG = {"primary": 7, "secondary": 7, "tertiary": 6, "residential": 5, "unclassified": 4.5, "service": 3.5,
        "living_street": 4, "pedestrian": 4, "track": 3, "footway": 1.8, "path": 1.6, "steps": 1.8}
m = Malha()
for f in F["caminhos"]:
    if f["subtipo"] != "road" or f["geometria"]["type"] != "LineString":
        continue
    lw = LARG.get(f["classe"] or "", 3)
    linha = geom(f)
    pts = [linha[0]]
    for a, b in zip(linha[:-1], linha[1:]):
        k = max(1, int(np.linalg.norm(b - a) / 2.0))
        pts += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    pts = np.array(pts)
    ok = np.hypot(*pts.T) < RAIO - 1.5
    for z in evitar:
        ok &= ~dentro(pts[:, 0], pts[:, 1], z) & (distancia(pts[:, 0], pts[:, 1], z, True)[0] > lw / 2 + 0.8)
    e_ = 1.0
    inc = np.hypot(cota(pts[:, 0] + e_, pts[:, 1]) - cota(pts[:, 0] - e_, pts[:, 1]),
                   cota(pts[:, 0], pts[:, 1] + e_) - cota(pts[:, 0], pts[:, 1] - e_)) / (2 * e_)
    ok &= inc < 0.6
    troco = []
    for i, (p, k) in enumerate(zip(pts, ok)):
        if k:
            troco.append(p)
        if (not k or i == len(pts) - 1) and len(troco) > 1:
            t = np.array(troco)
            L_ = desvio(t, lw / 2)
            R_ = desvio(t, -lw / 2)
            zl = cota(L_[:, 0], L_[:, 1]) + 0.12
            zr = cota(R_[:, 0], R_[:, 1]) + 0.12
            for j in range(len(t) - 1):
                m.face([(*R_[j], zr[j]), (*R_[j + 1], zr[j + 1]), (*L_[j + 1], zl[j + 1]), (*L_[j], zl[j])])
        if not k:
            troco = []
m.objeto("Caminhos", "caminhos", ["caminho"], GRUPOS["Caminhos"], uv=None, liso=True)

print("vegetação…")
print("vegetação…")


def arvore_modelo(subdiv):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    v = np.array([x.co[:] for x in bm.verts])
    f = [[x.index for x in fa.verts] for fa in bm.faces]
    bm.free()
    return v, f


ico1, ico1f = arvore_modelo(1)
ico2, ico2f = arvore_modelo(2)
mv, mf, mc = [], [], []


def _lobo(cx, cy, cz, rx, rz, cor, base, topo, grande, irregular):
    """Um lobo da copa: esfera achatada e deformada, com a cor a escurecer para baixo (sombra própria)."""
    v0, f0 = (ico2, ico2f) if grande else (ico1, ico1f)
    b = len(mv)
    jit = rng.normal(0, irregular, v0.shape)
    pts = (v0 + jit) * np.array([rx, rx, rz]) + np.array([cx, cy, cz])
    mv.extend(map(tuple, pts))
    mf.extend([[b + i for i in fa] for fa in f0])
    t = np.clip((pts[:, 2] - base) / max(topo - base, 0.1), 0, 1)
    luz = 0.62 + 0.5 * t ** 0.8
    mc.extend(map(tuple, np.clip(np.array(cor)[None, :] * luz[:, None], 0, 1)))


def plantar(x, y, z, raio, alt, copa_cor, tronco=1.0, oliveira=False):
    """Árvore com a posição, o topo e o raio de copa medidos; a forma da copa é esquemática:
    um lobo principal e, nas maiores, dois a quatro lobos à volta, como nas maquetas de referência."""
    b = len(mv)
    rt = 0.12 + 0.035 * raio
    # tronco: prisma de 6 lados, a afinar para cima
    for k in range(6):
        a = 2 * math.pi * k / 6
        mv.append((x + rt * math.cos(a), y + rt * math.sin(a), z - 0.3))
        mv.append((x + 0.7 * rt * math.cos(a), y + 0.7 * rt * math.sin(a), z + tronco + alt * 0.4))
    for k in range(6):
        j = (k + 1) % 6
        mf.append([b + 2 * k, b + 2 * j, b + 2 * j + 1, b + 2 * k + 1])
    mc.extend([(0.36, 0.28, 0.2)] * 12)
    base, topo = z + tronco, z + tronco + alt
    sh = 0.88 + 0.24 * rng.random()
    cor = tuple(np.clip(np.array(copa_cor) * sh, 0, 1))
    irregular = 0.11 if oliveira else 0.06
    n = 1 if raio < 1.8 else (3 if raio < 3.2 else 4)
    if n == 1:
        _lobo(x, y, base + alt / 2, raio, alt / 2, cor, base, topo, False, irregular)
        return
    # lobo principal, um pouco abaixo do topo medido, e lobos menores à volta sem passar do raio medido
    _lobo(x, y, topo - alt * 0.42, raio * 0.68, alt * 0.42, cor, base, topo, raio > 3, irregular)
    a0 = rng.random() * 2 * math.pi
    for k in range(n):
        a = a0 + 2 * math.pi * k / n + rng.normal(0, 0.3)
        d = raio * (0.4 + 0.08 * rng.random())
        rl = raio * (0.5 + 0.08 * rng.random())
        zl = base + alt * (0.36 + 0.12 * rng.random())
        c2 = tuple(np.clip(np.array(cor) * (0.92 + 0.14 * rng.random()), 0, 1))
        _lobo(x + d * math.cos(a), y + d * math.sin(a), zl, rl, min(rl, alt * 0.36), c2, base, topo, False, irregular)


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
    if any(dentro(x, y, z_) for z_ in evitar):
        continue  # copas sobre os telhados ou dentro do castelo: o LiDAR mede-as, mas são poucas e escondem o monumento
    cor = (0.30, 0.38, 0.22) if rng.random() < 0.65 else (0.42, 0.47, 0.30)
    plantar(float(x), float(y), z, r, alt, cor, tronco=max(h - alt, 0.6))
    n_arv += 1
MEDIDAS["vegetacao"] = {"arvores": n_arv}

me = bpy.data.meshes.new("vegetacao")
me.from_pydata(mv, [], mf)
me.update()
me.shade_smooth()  # copas redondas, sem facetas
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
glb = AQUI / "saida" / "guimaraes-bruto.glb"
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

final = SAIDA / "guimaraes.glb"
subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "guimaraes.blend"))

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
    ALVO = float(cota(*P(0, 0)))
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
    CAST = P(2, -2)
    zc = float(cota(*CAST))
    vista("capa", 215, 30, 470, lente=50, res=(1280, 800), destino=SAIDA / "capa.jpg", alvo_xy=tuple(CAST * 0.55 + P(-47, -150) * 0.45), alvo_z=zc - 22)
    if not so_capa:
        vista("topo", 180, 89.9, 400, orto=330, res=(1100, 1100))
        vista("norte", 0, 30, 380)
        vista("sul", 180, 30, 380)
        vista("nascente", 90, 30, 380)
        vista("poente", 270, 30, 380)
        vista("castelo_so", 200, 28, 150, lente=50, alvo_xy=tuple(CAST), alvo_z=zc + 6)
        vista("castelo_ne", 40, 25, 150, lente=50, alvo_xy=tuple(CAST), alvo_z=zc + 6)
        vista("castelo_rente", 245, 6, 90, lente=35, alvo_xy=tuple(CAST), alvo_z=zc + 8)
        vista("porta_principal", 75, 8, 45, lente=35, alvo_xy=tuple(P(20, 6)), alvo_z=zc + 6)
        vista("paco_aereo", 200, 38, 170, lente=50, alvo_xy=tuple(P(-47, -162)), alvo_z=float(cota(*P(-47, -162))) + 10)
        vista("paco_estatua", 305, 10, 110, lente=35, alvo_xy=tuple(P(-60, -145)), alvo_z=float(cota(*P(-60, -145))) + 9)
        vista("igreja_fachada", 248, 10, 38, lente=35, alvo_xy=tuple(P(-52, -78)), alvo_z=float(cota(*P(-52, -78))) + 4)
