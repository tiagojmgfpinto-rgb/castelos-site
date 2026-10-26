"""Constrói a maqueta 3D do Castelo de Penela (estado atual) em Blender e exporta-a
para public/maquetas/penela/penela.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy
    python pipeline/maquetas/penela/construir.py            # GLB
    python pipeline/maquetas/penela/construir.py --vistas   # GLB + capa + vistas de verificação

Dados: dados/fontes.json (recolher.py) e src/data/maquetas/penela.json, que tem a
leitura das fontes, as cotas e alturas estimadas e o estado de cada elemento.
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

NIV = {k: v["cota"] - ZREF for k, v in CFG["niveis"].items()}
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


M = F["mdt"]
A = np.array(M["alturas"], dtype=float)


def mdt(x, y):
    """Copernicus GLO-30 interpolado (Catmull-Rom), em z da cena."""
    lon = LON0 + (x + CENTRO[0]) / KX
    lat = LAT0 + (y + CENTRO[1]) / KY
    c = (lon - M["lon0"]) / M["dlon"] - 0.5
    r = (lat - M["lat0"]) / M["dlat"] - 0.5
    c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)

    def w(t):
        return [((-t + 2) * t - 1) * t / 2, ((3 * t - 5) * t * t + 2) / 2, ((-3 * t + 4) * t + 1) * t / 2, (t - 1) * t * t / 2]

    wc, wr = w(c - c0), w(r - r0)
    out = 0.0
    for i in range(4):
        rr = np.clip(r0 - 1 + i, 0, A.shape[0] - 1)
        for j in range(4):
            cc = np.clip(c0 - 1 + j, 0, A.shape[1] - 1)
            out = out + wr[i] * wc[j] * A[rr, cc]
    return out - ZREF


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

# Terreiro da igreja e do castelejo (fotografia aérea 1); o muro de suporte segue a 1.ª parte.
MURO_TERREIRO = np.array([P(-47, 46), P(-47, 24), P(-43, 8), P(-34, 0), P(-20, -3), P(-10.6, -1)])
TERREIRO = np.vstack([MURO_TERREIRO, [P(-10, 20), P(5, 30), P(5, 40), P(-30, 52), P(-47, 52)]])

TOPO = {s: EL[s]["adarve"] - ZREF + PARAPEITO + MERLAO["altura"] for s in SECTORES}
PE = {s: TOPO[s] - EL[s]["altura"] for s in SECTORES}
CAST_ADARVE = NIV["castelejo"] + 4.0
CAST_PE = CAST_ADARVE + PARAPEITO + MERLAO["altura"] - 6.0
CAST_ESCARPA = 8.0  # rochedo do castelejo sobre a encosta nascente (estimativa)


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


def cota_interior(x, y):
    z = np.where(dentro(x, y, TERREIRO), NIV["terreiro_igreja"], NIV["patio"])
    dc, _, _ = distancia(x, y, castelejo, True)
    no_cast = dentro(x, y, castelejo)
    # rochedo do castelejo: desce em escarpa para o recinto
    z = np.maximum(z, np.where(no_cast, NIV["castelejo"], NIV["castelejo"] - 1.4 * dc))
    return z


def cota(x, y):
    """Terreno final: MDT, recinto em plataformas, base das muralhas e escarpas."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    z0 = mdt(x, y)
    dentro_c = dentro(x, y, perimetro)
    melhor = np.full(x.shape, np.inf)
    pe = np.zeros(x.shape)
    esc = np.zeros(x.shape)
    zq = np.zeros(x.shape)
    linhas = list(SECTORES.items()) + [("castelejo", castelejo_ext)]
    for s, linha in linhas:
        d, qx, qy = distancia(x, y, linha)
        m = d < melhor
        melhor[m] = d[m]
        pe[m] = PE[s] if s in PE else CAST_PE
        esc[m] = EL[s]["escarpa"] if s in SECTORES else CAST_ESCARPA
        zq[m] = mdt(qx[m], qy[m])
    # A Brecha abre-se ao nível do pátio; na Brecha e na Porta da Vila não há escarpa
    for ponto, nivel in ((BRECHA, NIV["patio"] - 0.15), (PORTA_VILA, None)):
        dp = np.hypot(x - ponto[0], y - ponto[1])
        wp = np.clip(1 - (dp - 5) / 10, 0, 1)
        wp = wp * wp * (3 - 2 * wp)
        if nivel is not None:
            pe = pe * (1 - wp) + nivel * wp
        esc = esc * (1 - wp)
    s1 = np.clip((melhor - 1.0) / 5.0, 0, 1)
    zc = pe - esc * s1 * s1 * (3 - 2 * s1)  # pé da muralha e escarpa de rocha
    t = np.clip((melhor - 6.0) / 16.0, 0, 1)
    w = 1 - t * t * (3 - 2 * t)
    fora = z0 + (zc - zq) * w
    # sob a espessura da muralha o terreno já é o de fora (evita degraus à vista)
    recinto = dentro_c & (np.minimum(melhor, distancia(x, y, perimetro, True)[0]) > MUR["espessura"] / 2 + 0.3)
    return np.where(recinto, cota_interior(x, y), fora)


def construir_terreno():
    passo = 1.25
    n = int(2 * RAIO / passo) + 1
    xs = np.linspace(-RAIO, RAIO, n)
    X, Y = np.meshgrid(xs, xs)
    Z = cota(X, Y)
    # rugosidade da rocha nas encostas íngremes
    gy, gx = np.gradient(Z, passo)
    decl = np.degrees(np.arctan(np.hypot(gx, gy)))
    fora = ~dentro(X, Y, perimetro)
    rocha = np.clip((decl - 32) / 12, 0, 1) * fora
    rocha *= np.clip((distancia(X, Y, perimetro, True)[0] - 2.5) / 3.0, 0, 1)
    Z = Z + rocha * (fbm(X + 500, Y + 500, 7.0, 7) - 0.5) * 2.2
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

    # cores por vértice (sRGB)
    xv, yv = V[:, 0], V[:, 1]
    dv = decl[usados]
    nz = fbm(xv, yv, 22.0, 3)
    erva = np.array([0.50, 0.53, 0.33]) * (0.85 + 0.3 * nz[:, None])
    seco = np.array([0.66, 0.62, 0.44])
    mix = np.clip((fbm(xv + 90, yv, 35.0, 11) - 0.55) * 3, 0, 1)[:, None] * 0.7
    cor = erva * (1 - mix) + seco * mix
    roc = np.array([0.76, 0.68, 0.52]) * (0.9 + 0.2 * fbm(xv, yv, 4.0, 5)[:, None])
    tr = np.clip((dv - 34) / 10, 0, 1)[:, None]
    cor = cor * (1 - tr) + roc * tr
    din = dentro(xv, yv, perimetro)
    dter = dentro(xv, yv, TERREIRO) & din
    dcas = dentro(xv, yv, castelejo)
    terra = np.array([0.74, 0.68, 0.55]) * (0.9 + 0.14 * nz[:, None])
    relva = np.clip((fbm(xv + 7, yv - 3, 9.0, 13) - 0.5) * 3, 0, 1)[:, None] * 0.6
    patio = terra * (1 - relva) + np.array([0.55, 0.56, 0.36]) * relva
    cor[din] = patio[din]
    cor[dter] = np.array([0.78, 0.75, 0.68])
    roc_in = din & (dv > 30)
    cor[roc_in] = roc[roc_in]
    cor[dcas] = np.array([0.72, 0.66, 0.55])

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


def muralha(m, linha, esp, adarve, pe, interior_z, fechada=False, ameias=True, mat=0):
    """Pano de muralha ao longo da linha central (o exterior fica à direita)."""
    ext = desvio(linha, -esp / 2, fechada)
    inn = desvio(linha, esp / 2, fechada)
    n = len(linha)
    zi = np.minimum(np.minimum(interior_z, adarve), pe) - 1.5
    ze = np.full(n, pe - 1.5)
    seg = range(n) if fechada else range(n - 1)
    for i in seg:
        j = (i + 1) % n
        m.face([(*ext[i], ze[i]), (*ext[j], ze[j]), (*ext[j], adarve), (*ext[i], adarve)], mat)
        m.face([(*inn[j], zi[j]), (*inn[i], zi[i]), (*inn[i], adarve), (*inn[j], adarve)], mat)
        m.face([(*ext[i], adarve), (*ext[j], adarve), (*inn[j], adarve), (*inn[i], adarve)], mat)
        m.face([(*ext[j], ze[j]), (*ext[i], ze[i]), (*inn[i], zi[i]), (*inn[j], zi[j])], mat)
    if not fechada:
        for i in (0, n - 1):
            m.face([(*ext[i], ze[i]), (*inn[i], zi[i]), (*inn[i], adarve), (*ext[i], adarve)], mat)
    # parapeito do lado exterior e ameias por cima
    par_int = desvio(linha, -esp / 2 + 0.6, fechada)
    for i in seg:
        j = (i + 1) % n
        q = [ext[i], ext[j], par_int[j], par_int[i]]
        m.prisma(q, adarve, adarve + PARAPEITO, mat)
    if ameias:
        ameias_ao_longo(m, desvio(linha, -esp / 2 + 0.3, fechada), adarve + PARAPEITO, fechada, mat)


def ameias_ao_longo(m, linha, z, fechada=False, mat=0, prof=0.6):
    pts = np.vstack([linha, linha[:1]]) if fechada else linha
    passo = MERLAO["largura"] + MERLAO["intervalo"]
    for a, b in zip(pts[:-1], pts[1:]):
        L = np.linalg.norm(b - a)
        if L < MERLAO["largura"]:
            continue
        u = (b - a) / L
        k = max(1, int((L + MERLAO["intervalo"]) // passo))
        sobra = L - (k * passo - MERLAO["intervalo"])
        for i in range(k):
            s = sobra / 2 + i * passo + MERLAO["largura"] / 2
            m.caixa(a + u * s, u, MERLAO["largura"], prof, z, z + MERLAO["altura"], mat)


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


def telhado_4aguas(m, base, z, inclinacao=24, mat=1):
    """Telhado de quatro águas sobre um polígono quase retangular de 4 vértices."""
    b = np.array(base)
    if area(b) < 0:
        b = b[::-1]
    e = [np.linalg.norm(b[(i + 1) % 4] - b[i]) for i in range(4)]
    k = 0 if e[0] + e[2] >= e[1] + e[3] else 1  # arestas k e k+2 são as compridas
    b = np.roll(b, -k, axis=0)
    L = (np.linalg.norm(b[1] - b[0]) + np.linalg.norm(b[2] - b[3])) / 2
    W = (np.linalg.norm(b[3] - b[0]) + np.linalg.norm(b[2] - b[1])) / 2
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
    if area(poly) < 0:
        poly = poly[::-1]
    zt = cota(poly[:, 0], poly[:, 1])
    z0, z1 = float(zt.min()) - 1.0, float(np.median(zt)) + altura
    m.prisma(poly, z0, z1, mat_par, topo=False)
    s = simplificar(poly)
    if len(s) == 4 and abs(area(s)) > 0.85 * abs(area(poly)):
        telhado_4aguas(m, s, z1, mat=mat_tel)
    else:
        telhado_generico(m, poly, z1, mat=mat_tel)
    return z1


def igreja():
    """Igreja de São Miguel: planta do OSM; volumes e alturas das fotografias."""
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
    comp = float(max((p - fa) @ u for p in poly))
    nave_c = 23.3
    z0 = float(cota(*np.array([fa, fb]).T).min()) - 0.5
    # corpo baixo de toda a planta (capelas, sacristia)
    m.prisma(poly, z0, NIV["terreiro_igreja"] + 6.0, 0, topo=False)
    telhado_generico(m, poly, NIV["terreiro_igreja"] + 6.0, mat=1)

    def bloco(v0, v1, u0, u1, beiral, cume):
        c = [fa + v * v0 + u * u0, fa + v * v0 + u * u1, fa + v * v1 + u * u1, fa + v * v1 + u * u0]
        c = np.array(c)
        if area(c) < 0:
            c = c[::-1]
        m.prisma(c, z0, NIV["terreiro_igreja"] + beiral, 0, topo=False)
        zb = NIV["terreiro_igreja"] + beiral
        zc = NIV["terreiro_igreja"] + cume
        vm = (v0 + v1) / 2
        p = lambda vv, uu, z: (*(fa + v * vv + u * uu), z)
        m.face([p(v0, u0, zb), p(v0, u1, zb), p(vm, u1, zc), p(vm, u0, zc)], 1)
        m.face([p(vm, u0, zc), p(vm, u1, zc), p(v1, u1, zb), p(v1, u0, zb)], 1)
        m.face([p(v0, u0, zb), p(vm, u0, zc), p(v1, u0, zb)], 0)  # empena da fachada
        m.face([p(v1, u1, zb), p(vm, u1, zc), p(v0, u1, zb)], 0)
        return p

    # nave central e colateral sul sob o telhado alto; colateral norte mais baixa
    p = bloco(0.0, 10.2, 0.0, nave_c, 8.6, 11.6)
    bloco(10.2, larg, 0.0, nave_c - 1.0, 6.2, 8.0)
    # porta principal e janelas da fachada
    nf = -u
    arco(m, fa + v * 5.1, v, nf, 2.0, NIV["terreiro_igreja"], 2.6, 0.1, 2)
    for vv in (2.6, 7.6):
        c = fa + v * vv + nf * 0.03
        m.face([(*(c - v * 0.5), NIV["terreiro_igreja"] + 5.0), (*(c + v * 0.5), NIV["terreiro_igreja"] + 5.0),
                (*(c + v * 0.5), NIV["terreiro_igreja"] + 6.6), (*(c - v * 0.5), NIV["terreiro_igreja"] + 6.6)], 2)
    arco(m, fa + v * 13.2, v, nf, 1.4, NIV["terreiro_igreja"], 1.9, 0.1, 2)
    # cruz no vértice da empena
    topo = fa + v * 5.1 + nf * 0.1
    zc = NIV["terreiro_igreja"] + 11.6
    m.caixa(topo, v, 0.18, 0.18, zc, zc + 1.3, 3)
    m.caixa(topo, v, 0.75, 0.18, zc + 0.75, zc + 0.93, 3)

    # torre sineira (planta OSM), coruchéu piramidal e pináculos (fotografias 1, 2 e 4)
    t = geom(osm("1d555fe5"))
    t = simplificar(t, 0.8)
    if area(t) < 0:
        t = t[::-1]
    zt0 = z0
    cornija = NIV["terreiro_igreja"] + 16.5
    m.prisma(t, zt0, cornija, 0, topo=True)
    ct = t.mean(0)
    for i in range(len(t)):
        a, b = t[i], t[(i + 1) % len(t)]
        mid = (a + b) / 2
        out = unit(np.array([(b - a)[1], -(b - a)[0]]))
        arco(m, mid, unit(b - a), out, 1.3, cornija - 4.2, 1.9, 0.1, 2)
    # relógio no alçado da fachada
    face_relogio = min(range(len(t)), key=lambda i: ((t[i] + t[(i + 1) % len(t)]) / 2 - ct) @ u)
    a, b = t[face_relogio], t[(face_relogio + 1) % len(t)]
    out = unit(np.array([(b - a)[1], -(b - a)[0]]))
    cc = (a + b) / 2 + out * 0.04
    pts = [(*(cc + unit(b - a) * 0.8 * math.cos(k * math.pi / 8)), cornija - 6.6 + 0.8 * math.sin(k * math.pi / 8)) for k in range(16)]
    m.face(pts, 3)
    # coruchéu
    m.prisma(t, cornija, cornija + 0.5, 3)
    ap = (*ct, cornija + 6.2)
    for i in range(len(t)):
        a, b = t[i], t[(i + 1) % len(t)]
        m.face([(*a, cornija + 0.5), (*b, cornija + 0.5), ap], 3)
        pin = a + unit(ct - a) * 0.35
        base = [pin + np.array([math.cos(k * 2 * math.pi / 4 + math.pi / 4), math.sin(k * 2 * math.pi / 4 + math.pi / 4)]) * 0.3 for k in range(4)]
        for k in range(4):
            m.face([(*base[k], cornija + 0.5), (*base[(k + 1) % 4], cornija + 0.5), (*pin, cornija + 1.9)], 3)
    m.objeto("Igreja_Sao_Miguel", "igreja", ["cal", "telha", "sombra", "cantaria"], GRUPOS["Igreja_Sao_Miguel"], uv=2.0)


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

print("terreno…")
construir_terreno()
for ob in bpy.data.objects:
    if ob.name in ("Terreno", "Base"):
        ob.parent = GRUPOS["Terreno"]

print("muralhas…")
interior = lambda l: cota_interior(l[:, 0], l[:, 1])
for s, linha in SECTORES.items():
    m = Malha()
    ad = EL[s]["adarve"] - ZREF
    muralha(m, linha, MUR["espessura"], ad, PE[s], interior(linha))
    if s == "pano_poente":
        i = perto_de(linha, PORTA_VILA)
        a, b = linha[max(i - 1, 0)], linha[min(i + 1, len(linha) - 1)]
        u = unit(b - a)
        out = np.array([u[1], -u[0]])
        g = PORTA_VILA + out * MUR["espessura"] / 2
        arco(m, g, u, out, 2.6, PE[s], 2.6, 0.1, 1)
    ob = m.objeto(s, s, ["pedra", "sombra"], GRUPOS["Muralhas"])

print("castelejo…")
m = Malha()
muralha(m, castelejo, 1.8, CAST_ADARVE, CAST_PE, np.full(len(castelejo), NIV["castelejo"]), fechada=True)
m.objeto("Castelejo", "castelejo", ["pedra", "sombra"], GRUPOS["Castelejo"])

print("torres…")
for tid, base in torres.items():
    m = Malha()
    perto = min(SECTORES, key=lambda s: distancia(*base.mean(0)[:, None], SECTORES[s])[0][0])
    pe = PE[perto]
    extra = {"torre_quinaria": None, "porta_traicao": 3.0}.get(tid, 2.5)
    topo = (pe + 14.0) if tid == "torre_quinaria" else TOPO[perto] + extra
    torre(m, base, pe, topo)
    if tid == "porta_traicao":
        b = base if area(base) > 0 else base[::-1]
        # porta em cotovelo: abertura exterior na face virada a nordeste e interior na face virada a sueste
        faces = [(b[i], b[(i + 1) % len(b)]) for i in range(len(b))]
        def face_para(direcao):
            return max(faces, key=lambda f: unit(np.array([(f[1] - f[0])[1], -(f[1] - f[0])[0]])) @ direcao * min(1, np.linalg.norm(f[1] - f[0]) / 2))
        for dirc, sole in ((np.array([0.7, 0.7]), pe), (np.array([0.7, -0.7]), NIV["terreiro_igreja"])):
            a, c = face_para(dirc)
            u = unit(c - a)
            out = np.array([u[1], -u[0]])
            arco(m, (a + c) / 2, u, out, 1.8, sole, 2.0, 0.1, 1)
        nome, grupo = "Porta_Traicao", GRUPOS["Porta_Traicao"]
    elif tid == "torre_quinaria":
        nome, grupo = "Torre_Quinaria", GRUPOS["Torreoes"]
    else:
        nome, grupo = tid.replace("torreao_poente_", "Torreao_Poente_"), GRUPOS["Torreoes"]
    m.objeto(nome, tid, ["pedra", "sombra"], grupo)

m = Malha()
circ = np.array([CUBELO + 2.8 * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 18, endpoint=False)])
torre(m, circ, PE["pano_noroeste"], TOPO["pano_noroeste"] + 2.0)
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
    m.caixa(g + u * sx, u, 0.5, 0.25, PE["pano_poente"] - 0.2, PE["pano_poente"] + 4.3)
m.caixa(g, u, 3.6, 0.25, PE["pano_poente"] + 4.3, PE["pano_poente"] + 4.8)
m.objeto("Porta_Vila", "porta_vila", ["cantaria"], GRUPOS["Porta_Vila"])

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

print("igreja e recinto…")
igreja()
m = Malha()
casa(m, geom(osm("69a1ea12", "edificios")), 7.0)
m.objeto("Casa_Paroquial", "casa_paroquial", ["cal", "telha"], GRUPOS["Recinto"], uv=None)
m = Malha()
for pid in ("9ed07352", "dc7a9b83"):
    poly = geom(osm(pid, "edificios"))
    if area(poly) < 0:
        poly = poly[::-1]
    z = cota(poly[:, 0], poly[:, 1])
    m.prisma(poly, float(z.min()) - 0.5, float(z.max()) + 3.0, 0)
m.objeto("Apoio", None, ["apoio"], GRUPOS["Recinto"], uv=None)
m = Malha()
muralha(m, MURO_TERREIRO, 0.9, NIV["terreiro_igreja"] + 0.9 - PARAPEITO, NIV["patio"], np.full(len(MURO_TERREIRO), NIV["terreiro_igreja"]), ameias=False)
m.objeto("Muro_Terreiro", "muro_terreiro", ["pedra"], GRUPOS["Recinto"])

print("vila…")
proprios = {"3904babc", "69a1ea12", "9ed07352", "dc7a9b83", "0de35856"}
m = Malha()
pegadas = []
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
    casa(m, poly, h)
    pegadas.append(poly)
m.objeto("Edificios_Vila", "vila", ["cal", "telha"], GRUPOS["Vila"], uv=None)

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
caminhos_ob = bpy.data.objects["Caminhos"]
cam_xy = np.array([v.co[:2] for v in caminhos_ob.data.vertices]) if caminhos_ob.data.vertices else np.zeros((0, 2))


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


pedacos = [p for p in pegadas]
cand = []
passo = 6.5
for gx in np.arange(-RAIO, RAIO, passo):
    for gy in np.arange(-RAIO, RAIO, passo):
        cand.append((gx + rng.uniform(0, passo), gy + rng.uniform(0, passo)))
cand = np.array(cand)
cx, cy = cand[:, 0], cand[:, 1]
cz = cota(cx, cy)
ok = np.hypot(cx, cy) < RAIO - 3
ok &= ~dentro(cx, cy, perimetro)
ok &= distancia(cx, cy, perimetro, True)[0] > 6
dens = fbm(cx, cy, 30.0, 41)
ok &= dens > 0.42
# declive local
e = 1.5
gx_ = (cota(cx + e, cy) - cota(cx - e, cy)) / (2 * e)
gy_ = (cota(cx, cy + e) - cota(cx, cy - e)) / (2 * e)
ok &= np.degrees(np.arctan(np.hypot(gx_, gy_))) < 36
if len(cam_xy):
    for i in np.nonzero(ok)[0]:
        if np.min(np.hypot(*(cam_xy - cand[i]).T)) < 4.0:
            ok[i] = False
for poly in pegadas:
    c = poly.mean(0)
    rmax = np.hypot(*(poly - c).T).max() + 3
    near = np.hypot(cx - c[0], cy - c[1]) < rmax
    ok &= ~near
for i in np.nonzero(ok)[0]:
    escuro = rng.random() < 0.65
    r = rng.uniform(2.0, 3.4)
    cor = (0.30, 0.38, 0.22) if escuro else (0.42, 0.47, 0.30)
    plantar(cx[i], cy[i], cz[i], r, r * rng.uniform(1.3, 1.9), cor, tronco=rng.uniform(1.0, 2.2))

# oliveiras do recinto (fotografia aérea 2): pátio inferior, longe dos muros e edifícios
olv = []
for gx in np.arange(-60, 50, 5.5):
    for gy in np.arange(-70, 70, 5.5):
        olv.append((gx + rng.uniform(0, 2.5), gy + rng.uniform(0, 2.5)))
olv = np.array(olv)
ox, oy = olv[:, 0], olv[:, 1]
ok = dentro(ox, oy, perimetro) & ~dentro(ox, oy, castelejo)
ok &= distancia(ox, oy, perimetro, True)[0] > 3.5
ok &= distancia(ox, oy, MURO_TERREIRO)[0] > 2.5
ok &= distancia(ox, oy, castelejo, True)[0] > 7
ok &= np.hypot(ox - BRECHA[0], oy - BRECHA[1]) > 14
for pid in ("3904babc", "69a1ea12", "9ed07352", "dc7a9b83"):
    poly = geom(osm(pid, "edificios"))
    ok &= distancia(ox, oy, poly, True)[0] > 3.5
    ok &= ~dentro(ox, oy, poly)
fa_ = P(-35, 16)
ok &= np.hypot(ox - fa_[0], oy - fa_[1]) > 12  # adro da igreja livre
for i in np.nonzero(ok)[0]:
    if rng.random() < 0.35 and (oy[i] > P(0, -12)[1] or ox[i] < P(-58, 0)[0]):
        z = cota_interior(np.array([ox[i]]), np.array([oy[i]]))[0]
        r = rng.uniform(1.6, 2.2)
        plantar(ox[i], oy[i], z, r, r * 1.1, (0.48, 0.52, 0.38), tronco=1.2)

me = bpy.data.meshes.new("vegetacao")
me.from_pydata(mv, [], mf)
me.update()
cl = np.array(mc) ** 2.2
ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
ca.data.foreach_set("color", np.hstack([cl, np.ones((len(cl), 1))]).astype(np.float32).ravel())
me.color_attributes.active_color = ca
novo_objeto("Vegetacao", me, "vegetacao", mat="arvore", pai=GRUPOS["Vegetacao"])

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
    alvo = Vector((0, 0, NIV["patio"] + 4))
    d = Vector((-math.sin(math.radians(45)), math.cos(math.radians(45)), 0)) * math.cos(math.radians(50))
    d.z = math.sin(math.radians(50))
    so.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    cam = bpy.data.cameras.new("cam")
    co = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    pasta = AQUI / "vistas"
    pasta.mkdir(exist_ok=True)

    def vista(nome, azimute, elev, dist, lente=50, orto=None, res=(1200, 800), destino=None, alvo_z=None):
        a, e = math.radians(azimute), math.radians(elev)
        al = alvo if alvo_z is None else Vector((0, 0, alvo_z))
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
    vista("capa", 225, 33, 540, lente=50, res=(1280, 800), destino=SAIDA / "capa.jpg", alvo_z=NIV["patio"] - 34)
    if not so_capa:
        vista("topo", 180, 89.9, 400, orto=200, res=(1000, 1000))
        vista("norte", 0, 30, 330)
        vista("sul", 180, 30, 330)
        vista("nascente", 90, 30, 330)
        vista("poente", 270, 30, 330)
        vista("obliqua_foto1", 268, 38, 220, lente=50)
        vista("obliqua_foto2", 312, 32, 230, lente=50)
        vista("muro_poente", 262, 12, 120, lente=40)
