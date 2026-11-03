"""Castelo de Montalegre (Terras de Barroso): gera a maqueta 3D do estado atual (public/maquetas/montalegre/
montalegre.glb), a capa e as vistas de verificação.

Fontes: LiDAR da DGT de 2024 (folha 228540: MDS e MDT de 50 cm e nuvem de pontos classificada), ortofotomapa da
DGT de 2021 e do geoportal da Câmara de Montalegre (2011), OpenStreetMap/Overture e fotografias. Medidas e
decisões em LEIAME.md.

    python pipeline/maquetas/montalegre/construir.py [--vistas] [--capa] [--so=nome,…] [--sem-glb] [--blend]
"""
import json
import math
import sys
from pathlib import Path

import bpy  # noqa: I001 (o bpy tem de vir antes do bmesh)
import bmesh
import numpy as np
import shapely
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
sys.path.insert(0, str(AQUI))
import comum as C  # noqa: E402

_cfg = RAIZ / "src" / "data" / "maquetas" / "montalegre.json"
CFG = json.loads(_cfg.read_text()) if _cfg.exists() else {"origem": {"lat": C.LAT0, "lon": C.LON0}, "variante": "estado-atual", "versao": "0.1"}
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "montalegre"
LID = np.load(AQUI / "dados" / "lidar.npz")
LP, LX0, LY0 = float(LID["passo"]), float(LID["x0"]), float(LID["y0"])
MDT = LID["mdt"] / 100.0 + float(LID["base"])
MDS = LID["mds"] / 100.0 + float(LID["base"])
CLS = LID["classe"]
SUPF = np.load(AQUI / "dados" / "superficie.npz")
SP, SX0, SY0 = float(SUPF["passo"]), float(SUPF["x0"]), float(SUPF["y0"])
SUP = SUPF["z"] / 100.0 + float(SUPF["base"])
ORTO = np.asarray(Image.open(AQUI / "dados" / "orto.jpg").convert("RGB"), float) / 255.0  # 25 cm, norte em cima
OX0, OY1, OP = C.X0, C.Y1, 0.25

# a maqueta: 110 × 110 m à volta do castelo
X0, X1, Y0, Y1 = -58.0, 52.0, -78.0, 32.0
rng = np.random.default_rng(1273)  # 1273: foral de D. Afonso III

EU = np.array(C.EU)
EV = np.array(C.EV)


def W(u, v):
    """Ponto do referencial da torre (u, v) para a cena (x este, y norte)."""
    return EU * u + EV * v


def WL(pts):
    return [tuple(W(*p)) for p in pts]


# ---------------------------------------------------------------- utilidades
def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def amostra(G, x0, y0, p, x, y, ordem=1):
    xa, ya = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    r = ndi.map_coordinates(G, [(ya - y0) / p, (xa - x0) / p], order=ordem, mode="nearest")
    return r if np.ndim(x) else float(r[0])


def cota(x, y):
    """Chão (MDT de 50 cm)."""
    return amostra(MDT, LX0, LY0, LP, x, y)


def superficie(x, y):
    """O ponto mais alto das classes solo, edifícios e muros (sem vegetação), em células de 25 cm."""
    return amostra(SUP, SX0, SY0, SP, x, y, ordem=0)


# ---------------------------------------------------------------- ruído
# ---------------------------------------------------------------- ruído

def ruido(x, y, escala, semente):
    g = np.random.default_rng(semente).random((64, 64))
    u, v = np.asarray(x) / escala, np.asarray(y) / escala
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


# ---------------------------------------------------------------- malhas
# ---------------------------------------------------------------- malhas

MATS = {}


def uv_mundo(ob, escala):
    """UV em metros: faces horizontais em planta, verticais ao longo da face (as fiadas seguem a cota)."""
    me = ob.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    n_l = len(me.loops)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    vi = np.empty(n_l, int)
    me.loops.foreach_get("vertex_index", vi)
    nrm = np.empty(len(me.polygons) * 3)
    me.polygons.foreach_get("normal", nrm)
    nrm = nrm.reshape(-1, 3)
    lt = np.empty(len(me.polygons), int)
    me.polygons.foreach_get("loop_total", lt)
    pol = np.repeat(np.arange(len(me.polygons)), lt)
    n = nrm[pol]
    p = co[vi]
    hz = np.abs(n[:, 2]) > 0.7
    h = np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-6)
    u = np.where(hz, p[:, 0], (p[:, 0] * -n[:, 1] + p[:, 1] * n[:, 0]) / h)
    v = np.where(hz, p[:, 1], p[:, 2])
    me.uv_layers.active.data.foreach_set("uv", (np.stack([u, v], 1) / escala).ravel().astype(np.float32))


class Malha:
    """Acumula vértices e faces (com material por face) para um objeto."""

    def __init__(self, mats):
        self.v, self.f, self.m = [], [], []
        self.mats = list(mats)

    def i(self, nome):
        return self.mats.index(nome)

    def face(self, pts, mat=0):
        if isinstance(mat, str):
            mat = self.i(mat)
        b = len(self.v)
        self.v.extend([tuple(map(float, p)) for p in pts])
        self.f.append(list(range(b, b + len(pts))))
        self.m.append(mat)

    def prisma(self, base, z0, z1, mat=0, topo=True, fundo=False, mat_topo=None):
        n = len(base)
        for i in range(n):
            a, b = base[i], base[(i + 1) % n]
            za0 = z0[i] if np.ndim(z0) else z0
            zb0 = z0[(i + 1) % n] if np.ndim(z0) else z0
            za1 = z1[i] if np.ndim(z1) else z1
            zb1 = z1[(i + 1) % n] if np.ndim(z1) else z1
            self.face([(a[0], a[1], za0), (b[0], b[1], zb0), (b[0], b[1], zb1), (a[0], a[1], za1)], mat)
        if topo:
            self.face([(p[0], p[1], z1[i] if np.ndim(z1) else z1) for i, p in enumerate(base)], mat if mat_topo is None else mat_topo)
        if fundo:
            self.face([(p[0], p[1], z0[i] if np.ndim(z0) else z0) for i, p in reversed(list(enumerate(base)))], mat)

    def caixa_uv(self, u0, u1, v0, v1, z0, z1, mat=0, topo=True, fundo=False):
        """Caixa alinhada com a torre (limites em u e v)."""
        q = [W(u0, v0), W(u1, v0), W(u1, v1), W(u0, v1)]
        self.prisma(q, z0, z1, mat, topo=topo, fundo=fundo)

    def caixa_face(self, f, s0, s1, d0, d1, z0, z1, mat=0, topo=True, fundo=False):
        """Caixa encostada a um plano f: de s0 a s1 ao longo, de d0 a d1 para fora."""
        q = [f.xy(s0, d0), f.xy(s1, d0), f.xy(s1, d1), f.xy(s0, d1)]
        self.prisma(q, z0, z1, mat, topo=topo, fundo=fundo)

    def piramide(self, base, z0, zp, mat=0):
        c = np.mean(base, 0)
        n = len(base)
        for i in range(n):
            a, b = base[i], base[(i + 1) % n]
            self.face([(a[0], a[1], z0), (b[0], b[1], z0), (c[0], c[1], zp)], mat)

    def cilindro(self, c, r, z0, z1, mat=0, k=12, topo=True, r1=None):
        r1 = r if r1 is None else r1
        a = np.linspace(0, 2 * math.pi, k, endpoint=False)
        b0 = [(c[0] + r * math.cos(t), c[1] + r * math.sin(t), z0) for t in a]
        b1 = [(c[0] + r1 * math.cos(t), c[1] + r1 * math.sin(t), z1) for t in a]
        for i in range(k):
            j = (i + 1) % k
            self.face([b0[i], b0[j], b1[j], b1[i]], mat)
        if topo and r1 > 0.01:
            self.face(b1, mat)

    def barra(self, p0, p1, r, mat=0, k=6):
        """Barra (tubo) de p0 a p1 (3D), de secção poligonal."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        ax = unit(p1 - p0)
        a1 = unit(np.cross(ax, [0, 0, 1] if abs(ax[2]) < 0.9 else [1, 0, 0]))
        a2 = np.cross(ax, a1)
        ang = np.linspace(0, 2 * math.pi, k, endpoint=False)
        c0 = [p0 + r * (math.cos(t) * a1 + math.sin(t) * a2) for t in ang]
        c1 = [p1 + r * (math.cos(t) * a1 + math.sin(t) * a2) for t in ang]
        for i in range(k):
            j = (i + 1) % k
            self.face([c0[i], c0[j], c1[j], c1[i]], mat)

    def objeto(self, nome, elemento, pai, uv=3.0, liso=False, vcor=None, recalc=True):
        me = bpy.data.meshes.new(nome)
        me.from_pydata(self.v, [], self.f)
        me.update()
        for m in self.mats:
            me.materials.append(MATS[m])
        me.polygons.foreach_set("material_index", self.m)
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.002)
        if recalc:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        if liso:
            me.shade_smooth()
        ob = bpy.data.objects.new(nome, me)
        bpy.context.scene.collection.objects.link(ob)
        if elemento:
            ob["elemento"] = elemento
        ob.parent = pai
        if uv:
            uv_mundo(ob, uv)
        if vcor is not None:  # cor por vértice (patine da pedra), multiplicada pela textura
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            cv = vcor(co.reshape(-1, 3))
            at = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
            at.data.foreach_set("color", np.concatenate([cv, np.ones((len(cv), 1))], 1).astype(np.float32).ravel())
        return ob


def malha_rapida(nome, V, Fq, mat, pai, elemento=None):
    me = bpy.data.meshes.new(nome)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(Fq.size)
    me.loops.foreach_set("vertex_index", Fq.astype(np.int32).ravel())
    me.polygons.add(len(Fq))
    me.polygons.foreach_set("loop_start", (np.arange(len(Fq)) * 4).astype(np.int32))
    me.update(calc_edges=True)
    me.materials.append(MATS[mat])
    ob = bpy.data.objects.new(nome, me)
    bpy.context.scene.collection.objects.link(ob)
    if elemento:
        ob["elemento"] = elemento
    ob.parent = pai
    return ob


# ---------------------------------------------------------------- materiais e texturas

def imagem(nome, arr):
    h, w, _ = arr.shape
    img = bpy.data.images.new(nome, w, h, alpha=False)
    rgba = np.concatenate([arr, np.ones((h, w, 1))], 2).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.file_format = "JPEG"
    img.pack()
    return img


# ---------------------------------------------------------------- materiais e texturas
def material(nome, cor, rug=0.9, tex=None, vcor=False, normal=None, forca=1.0, escala=None):
    mt = bpy.data.materials.new(nome)
    mt.use_nodes = True
    nt = mt.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    mapa = None
    if escala is not None:
        uvn = nt.nodes.new("ShaderNodeUVMap")
        mapa = nt.nodes.new("ShaderNodeMapping")
        mapa.inputs["Scale"].default_value = (escala, escala, 1.0)
        nt.links.new(uvn.outputs["UV"], mapa.inputs["Vector"])
    if normal is not None:
        normal.colorspace_settings.name = "Non-Color"
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = normal
        if mapa is not None:
            nt.links.new(mapa.outputs["Vector"], tn.inputs["Vector"])
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = forca
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Base Color"].default_value = (*[c ** 2.2 for c in cor], 1)
    bsdf.inputs["Roughness"].default_value = rug
    if tex is not None:
        ti = nt.nodes.new("ShaderNodeTexImage")
        ti.image = tex
        if mapa is not None:
            nt.links.new(mapa.outputs["Vector"], ti.inputs["Vector"])
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


def normais(alt, forca):
    gx = (np.roll(alt, -1, 1) - np.roll(alt, 1, 1)) * forca
    gy = (np.roll(alt, -1, 0) - np.roll(alt, 1, 0)) * forca
    nv = np.stack([-gx, -gy, np.ones_like(alt)], 2)
    nv /= np.linalg.norm(nv, axis=2, keepdims=True)
    return nv * 0.5 + 0.5


def textura_cantaria(n=1024, metros=6.0, semente=1170, fiada=(0.30, 0.48), comp=(0.55, 1.6), liquen=0.80,
                     tinta=(1.0, 1.0, 1.0), paleta=None, prob=None, junta_m=0.005, irregular=0.0, var=(0.86, 1.12),
                     liq_forca=0.55):
    """Aparelho de granito em fiadas (6 m por repetição): silhares de fiada e comprimento dados, juntas finas,
    arestas gastas, grão de biotite e feldspato, manchas de óxido, escorrências e líquenes. Com irregular > 0
    as juntas verticais inclinam e as fiadas ondulam (alvenaria da cerca e da muralha). Devolve a cor e o relevo."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    if paleta is None:
        paleta = np.array([
            (0.600, 0.565, 0.500),  # cinzento-dourado (a maioria)
            (0.660, 0.630, 0.570),  # claro
            (0.500, 0.475, 0.430),  # escuro, gasto
            (0.640, 0.565, 0.460),  # ocre
            (0.575, 0.505, 0.420),  # ocre-acastanhado
            (0.560, 0.555, 0.535),  # cinzento frio
        ])
        prob = [0.40, 0.16, 0.16, 0.15, 0.05, 0.08]
    prob = np.array(prob) / np.sum(prob)
    cel = np.zeros((n, n), int)
    cores = []
    y = 0.0
    k = 0
    while y < n - 1:
        hf = r.uniform(*fiada) * m_px
        if y + hf > n - fiada[0] * 0.6 * m_px:
            hf = n - y
        y0, y1 = int(round(y)), int(round(y + hf))
        x = r.uniform(0, 1.0) * m_px
        x_ini = x
        while x < x_ini + n:
            wf = r.uniform(*comp) * m_px
            xs = (np.arange(int(round(x)), int(round(x + wf)))) % n
            cel[y0:y1, xs] = k
            cores.append(paleta[r.choice(len(paleta), p=prob)] * r.uniform(*var))
            k += 1
            x += wf
        y += hf
    cores = np.array(cores)
    yy, xx = np.mgrid[0:n, 0:n]
    amp = 0.04 + irregular
    wx = ((fbm(xx, yy, 0.2 * m_px, semente + 3) - 0.5) * amp * m_px).astype(int)
    wy = ((fbm(xx, yy, 0.2 * m_px, semente + 4) - 0.5) * (0.03 + irregular * 0.6) * m_px).astype(int)
    celw = cel[(yy + wy) % n, (xx + wx) % n]
    tex = cores[celw]
    borda = np.zeros((n, n), bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        borda |= celw != np.roll(np.roll(celw, dy, 0), dx, 1)
    dist = ndi.distance_transform_edt(~borda) / m_px
    junta = dist < junta_m
    gasto = np.clip(dist / (0.04 + irregular * 0.5), 0, 1)
    alt = np.sqrt(gasto) * (~junta)
    tex = tex * (0.86 + 0.14 * alt[..., None])
    tex = np.where(junta[..., None], tex * 0.55 + np.array([0.36, 0.35, 0.33]) * 0.45, tex)
    tex = tex * (0.92 + 0.16 * fbm(xx, yy, 0.30 * m_px, semente + 5))[..., None]
    g1 = ndi.gaussian_filter(r.random((n, n)), 0.6, mode="wrap")
    g1 = (g1 - g1.mean()) / g1.std()
    g2 = r.random((n, n))
    tex = tex * (1 + 0.05 * g1[..., None])
    tex = np.where((g2 < 0.045)[..., None], tex * 0.62, tex)
    tex = np.where((g2 > 0.93)[..., None], tex * 0.80 + 0.17, tex)
    ox = np.clip((fbm(xx, yy, 0.6 * m_px, semente + 8) - 0.58) * 3.0, 0, 1)[..., None]
    tex = tex * (1 - 0.30 * ox) + np.array([0.62, 0.50, 0.36]) * 0.30 * ox
    esc = fbm(xx * 6.0, yy * 0.25, 0.9 * m_px, semente + 9)
    esc = np.clip((esc - 0.55) * 2.2, 0, 1)[..., None]
    tex = tex * (1 - 0.22 * esc)
    # líquenes: manchas brancas acinzentadas (muito visíveis nas fotografias, sobretudo na torre e no alto
    # dos muros) e algumas amarelo-esverdeadas
    liq = fbm(xx, yy, 0.25 * m_px, semente + 6)
    tex = np.where((liq > liquen)[..., None], tex * (1 - liq_forca) + np.array([0.86, 0.86, 0.82]) * liq_forca, tex)
    liq2 = fbm(xx, yy, 0.18 * m_px, semente + 11)
    tex = np.where((liq2 > liquen + 0.04)[..., None], tex * 0.6 + np.array([0.66, 0.66, 0.42]) * 0.4, tex)
    return np.clip(tex * np.array(tinta), 0, 1), ndi.gaussian_filter(alt, 0.8) + 0.05 * g1


def textura_telha(n=256, metros=2.0):
    yy, xx = np.mgrid[0:n, 0:n] / n * metros
    onda = 0.5 + 0.5 * np.cos(2 * math.pi * xx / 0.22)
    fiada = (yy % 0.42) / 0.42
    v = 0.78 + 0.22 * onda - 0.18 * (fiada > 0.9)
    g = fbm(xx * 8, yy * 8, 1.0, 77)
    v = v * (0.9 + 0.2 * g)
    return np.repeat(v[..., None], 3, 2).clip(0, 1), onda - 0.4 * (fiada > 0.9)


def textura_cubos(n=1024, metros=8.0, semente=91, base=(0.58, 0.57, 0.54)):
    """Calçada de cubos de granito (cubos de c. 10 cm, juntas de areia)."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    c = 0.105 * m_px
    gi = (xx // c).astype(int) + 1000 * ((yy + (xx // (3 * c)).astype(int) % 2 * c * 0.5) // c).astype(int)
    tom = r.uniform(0.85, 1.12, gi.max() + 1)[gi]
    fx, fy = (xx % c) / c, ((yy + (xx // (3 * c)).astype(int) % 2 * c * 0.5) % c) / c
    junta = (fx < 0.12) | (fy < 0.12)
    tex = np.array(base) * tom[..., None] * (0.9 + 0.2 * fbm(xx, yy, 1.5 * m_px, semente + 1))[..., None]
    tex = np.where(junta[..., None], np.array([0.40, 0.38, 0.35]), tex)
    return np.clip(tex, 0, 1), (~junta).astype(float)


def textura_relva(n=512, metros=8.0, semente=7):
    """Relvado do recinto (fotografias: verde médio, com manchas amareladas e terra à vista nos carreiros)."""
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    g = fbm(xx, yy, 1.2 * m_px, semente)
    g2 = fbm(xx, yy, 0.15 * m_px, semente + 5)
    base = np.array([0.36, 0.46, 0.22])
    amarelo = np.array([0.55, 0.55, 0.30])
    t = np.clip((g - 0.45) * 2.5, 0, 1)[..., None]
    tex = base * (1 - t) + amarelo * t
    tex = tex * (0.85 + 0.3 * g2[..., None])
    pinta = np.random.default_rng(semente).random((n, n))
    tex = np.where((pinta > 0.9)[..., None], tex * 1.15, tex)
    return np.clip(tex, 0, 1), g2


def textura_rocha(n=512, metros=6.0, semente=13):
    """Afloramentos de granito do recinto: cinzento claro, gasto, com líquenes e fendas."""
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    g = fbm(xx, yy, 0.8 * m_px, semente)
    tex = np.array([0.60, 0.58, 0.54]) * (0.8 + 0.4 * g)[..., None]
    fenda = np.abs(fbm(xx, yy, 1.5 * m_px, semente + 3) - 0.5) < 0.012
    tex = np.where(fenda[..., None], tex * 0.55, tex)
    liq = fbm(xx, yy, 0.2 * m_px, semente + 6)
    tex = np.where((liq > 0.66)[..., None], tex * 0.5 + 0.42, tex)
    return np.clip(tex, 0, 1), g


# ---------------------------------------------------------------- planos e vãos
# ---------------------------------------------------------------- planos e vãos

class Plano:
    """Plano vertical: o = ponto de origem (x, y), t = eixo ao longo, n = normal para fora (por omissão, à
    esquerda de t). p(s, d, z): s ao longo, d para fora (negativo = para dentro da parede)."""

    def __init__(self, o, t, n=None, meia=None):
        self.o = np.asarray(o, float)
        self.t = unit(t)
        self.n = unit(n) if n is not None else np.array([-self.t[1], self.t[0]])
        self.meia = meia

    def xy(self, s, d):
        return self.o + self.t * s + self.n * d

    def p(self, s, d, z):
        q = self.xy(s, d)
        return (q[0], q[1], z)






def quad(m, f, s0, s1, z0, z1, d=0.0, mat=0):
    m.face([f.p(s0, d, z0), f.p(s1, d, z0), f.p(s1, d, z1), f.p(s0, d, z1)], mat)


def arco_pts(sc, zc, r, k=10, a0=0.0, a1=math.pi):
    return [(sc + r * math.cos(a), zc + r * math.sin(a)) for a in np.linspace(a0, a1, k + 1)]


def parede(m, f, s0, s1, z0, z1, vaos, mat=0, passo_z=1.5, d=0.0):
    """Plano da face entre s0..s1 e z0..z1, com vãos retangulares ou em arco abertos.
    vaos: lista de (sa, sb, za, zb, arco); arco=True: zb é o fecho do arco de volta perfeita."""
    ss = sorted({s0, s1, *[v_[0] for v_ in vaos], *[v_[1] for v_ in vaos]})
    ss = [s for s in ss if s0 <= s <= s1]
    for a, b in zip(ss[:-1], ss[1:]):
        sm = (a + b) / 2
        cort = sorted([(v_[2], v_[3]) for v_ in vaos if v_[0] <= sm <= v_[1]])
        zs = [z0]
        for za, zb in cort:
            zs += [za, zb]
        zs.append(z1)
        for za, zb in zip(zs[0::2], zs[1::2]):
            if zb - za < 1e-4:
                continue
            nz = max(1, int(math.ceil((zb - za) / passo_z)))
            for i in range(nz):
                quad(m, f, a, b, za + (zb - za) * i / nz, za + (zb - za) * (i + 1) / nz, d=d, mat=mat)
    for sa, sb, za, zb, arco in vaos:
        if not arco:
            continue
        r = (sb - sa) / 2
        zc = zb - r
        pts = arco_pts((sa + sb) / 2, zc, r, 12)
        meio = len(pts) // 2
        m.face([f.p(s, d, z) for s, z in pts[:meio + 1]] + [f.p(sb, d, zb)], mat)
        m.face([f.p(s, d, z) for s, z in pts[meio:]] + [f.p(sa, d, zb)], mat)


def contorno_vao(sc, w, za, zb, arco=True, k=12):
    r = w / 2
    sa, sb = sc - r, sc + r
    c = [(sa, za), (sb, za)]
    if arco:
        c += arco_pts(sc, zb - r, r, k)
    else:
        c += [(sb, zb), (sa, zb)]
    return c


def vao(m, f, sc, w, za, zb, prof, arco=True, mat_lado="cantaria_torre", mat_fundo="sombra", k=12, d0=0.0):
    """Ombreiras, soleira e intradorso do vão (para dentro da parede até prof) e o fundo."""
    contorno = contorno_vao(sc, w, za, zb, arco, k)
    pts = contorno + [contorno[0]]
    for (s1, z1), (s2, z2) in zip(pts[:-1], pts[1:]):
        m.face([f.p(s1, d0, z1), f.p(s2, d0, z2), f.p(s2, d0 - prof, z2), f.p(s1, d0 - prof, z1)], mat_lado)
    m.face([f.p(s, d0 - prof, z) for s, z in contorno], mat_fundo)
    return contorno


def aduelas(m, f, sc, w, za, zb, larg, sal, n=9, mat="cantaria_lisa", ombreiras=True, d0=0.0):
    """Aduelas do arco (e ombreiras de cantaria), ligeiramente salientes da face."""
    r = w / 2
    zc = zb - r
    R = r + larg
    for i in range(n):
        a0, a1 = math.pi * i / n, math.pi * (i + 1) / n
        g = 0.010
        pts = [(sc + r * math.cos(a0 + g), zc + r * math.sin(a0 + g)), (sc + R * math.cos(a0 + g), zc + R * math.sin(a0 + g)),
               (sc + R * math.cos(a1 - g), zc + R * math.sin(a1 - g)), (sc + r * math.cos(a1 - g), zc + r * math.sin(a1 - g))]
        m.face([f.p(s, d0 + sal, z) for s, z in pts], mat)
        for (s1, z1), (s2, z2) in zip(pts, pts[1:] + pts[:1]):
            m.face([f.p(s1, d0, z1), f.p(s2, d0, z2), f.p(s2, d0 + sal, z2), f.p(s1, d0 + sal, z1)], mat)
    if ombreiras:
        for s0, s1 in ((sc - r - larg, sc - r), (sc + r, sc + r + larg)):
            hz = zc - za
            k = max(1, int(round(hz / 0.45)))
            for j in range(k):
                z0_, z1_ = za + hz * j / k + 0.008, za + hz * (j + 1) / k - 0.008
                m.caixa_face(f, s0, s1, d0, d0 + sal, z0_, z1_, mat)


def passagem(m, f, s0, s1, prof, z_base, z_topo, w, alt, chao_f, chao_d, mat="muro", mat_topo="muro_topo",
             mat_intra="intradorso", degrau=None, arco_dentro=None, topo=True, lados=True):
    """Bloco de muro com uma passagem em arco de volta perfeita, de lado a lado. f: plano da face de fora (d = 0,
    normal para fora); o bloco vai de s0 a s1 ao longo e de d = 0 a d = -prof; o eixo da passagem é s = 0.
    w, alt: largura e altura (do chão ao fecho) do vão; chao_f, chao_d: chão da passagem fora e dentro.
    degrau = (w2, alt2, p2): arquivolta de fora mais larga (w2 × alt2) até p2 de profundidade.
    arco_dentro = (w3, alt3): vão da face de dentro, se for diferente."""
    k = 14

    def face_com_vao(d, ww, aa, chao):
        r = ww / 2
        zap = chao + aa
        zs = zap - r
        # faixas dos lados, do pé ao topo
        quad(m, f, s0, -r, z_base, z_topo, d=d, mat=mat)
        quad(m, f, r, s1, z_base, z_topo, d=d, mat=mat)
        if chao > z_base + 0.01:
            quad(m, f, -r, r, z_base, chao, d=d, mat=mat)
        # por cima do arco: duas metades, do arranque ao fecho e até ao topo
        pts = arco_pts(0.0, zs, r, k)
        meio = len(pts) // 2
        m.face([f.p(s, d, z) for s, z in pts[:meio + 1]] + [f.p(0.0, d, z_topo), f.p(r, d, z_topo)], mat)
        m.face([f.p(s, d, z) for s, z in pts[meio:]] + [f.p(-r, d, z_topo), f.p(0.0, d, z_topo)], mat)
        return contorno_vao(0.0, ww, chao, zap, True, k)

    def forro(c_a, d_a, c_b, d_b):
        pa, pb = c_a + [c_a[0]], c_b + [c_b[0]]
        for i in range(len(c_a)):
            (sa1, za1), (sa2, za2) = pa[i], pa[i + 1]
            (sb1, zb1), (sb2, zb2) = pb[i], pb[i + 1]
            mt = mat_intra if i > 0 else "calcada"
            m.face([f.p(sa1, d_a, za1), f.p(sa2, d_a, za2), f.p(sb2, d_b, zb2), f.p(sb1, d_b, zb1)], mt)

    if degrau is None:
        c_fora = face_com_vao(0.0, w, alt, chao_f)
        d_ini = 0.0
    else:
        w2, alt2, p2 = degrau
        c2 = face_com_vao(0.0, w2, alt2, chao_f)
        chao_p = chao_f + (chao_d - chao_f) * p2 / prof
        c2b = contorno_vao(0.0, w2, chao_p, chao_p + alt2, True, k)
        forro(c2, 0.0, c2b, -p2)
        # face do degrau (entre a arquivolta e o vão), a p2 de profundidade
        c_fora = contorno_vao(0.0, w, chao_p, chao_p + alt, True, k)
        for i in range(len(c2b)):
            j = (i + 1) % len(c2b)
            if i == 0:
                continue
            m.face([f.p(*_sd(c2b[i], -p2)), f.p(*_sd(c2b[j], -p2)), f.p(*_sd(c_fora[j], -p2)), f.p(*_sd(c_fora[i], -p2))], mat)
        d_ini = -p2
    if arco_dentro is None:
        c_dentro = face_com_vao(-prof, w, alt, chao_d)
        forro(c_fora, d_ini, c_dentro, -prof)
    else:
        w3, alt3 = arco_dentro
        c_d = face_com_vao(-prof, w3, alt3, chao_d)
        c_m = contorno_vao(0.0, w, chao_d, chao_d + alt, True, k)
        forro(c_fora, d_ini, c_m, -prof + 0.4)
        forro(c_m, -prof + 0.4, c_d, -prof)  # abre em derrame para o arco de dentro
    if topo:
        m.face([f.p(s0, 0, z_topo), f.p(s1, 0, z_topo), f.p(s1, -prof, z_topo), f.p(s0, -prof, z_topo)], mat_topo)
    if lados:
        for s_ in (s0, s1):
            m.face([f.p(s_, 0, z_base), f.p(s_, -prof, z_base), f.p(s_, -prof, z_topo), f.p(s_, 0, z_topo)], mat)


def _sd(sz, d):
    return (sz[0], d, sz[1])


def textura_erva(n=512, metros=8.0, semente=17):
    """Erva seca (julho): palha, verde apagado e terra."""
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    g = fbm(xx, yy, 1.5 * m_px, semente)
    g2 = fbm(xx, yy, 0.12 * m_px, semente + 5)
    palha = np.array([0.55, 0.50, 0.37])  # ortofotomapa: monte de erva seca, castanho-azeitona
    verde = np.array([0.40, 0.42, 0.27])
    t = np.clip((g - 0.42) * 2.2, 0, 1)[..., None]
    tex = palha * (1 - t) + verde * t
    tex = tex * (0.82 + 0.36 * g2[..., None])
    return np.clip(tex, 0, 1), g2


def textura_terra(n=512, metros=6.0, semente=23):
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    g = fbm(xx, yy, 0.6 * m_px, semente)
    tex = np.array([0.60, 0.52, 0.42]) * (0.85 + 0.3 * g)[..., None]
    pedra = np.random.default_rng(semente).random((n, n)) > 0.985
    tex = np.where(ndi.binary_dilation(pedra, iterations=2)[..., None], tex * 1.15, tex)
    return np.clip(tex, 0, 1), g



# ---------------------------------------------------------------- utilidades

def orientar(pts, ref):
    """Põe a face com a normal para o lado de ref (vetor 3D)."""
    p = np.asarray(pts, float)
    nrm = np.zeros(3)
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        nrm += np.cross(a, b)
    return list(pts) if np.dot(nrm, ref) >= 0 else list(pts)[::-1]


def painel(m, f, contorno, furos=(), mat=0, d=0.0):
    """Face plana do plano f (em s, z, a d para fora), com furos; triangulada (Delaunay restringida)."""
    pol = shapely.Polygon(contorno, [h for h in furos if len(h) >= 3])
    if not pol.is_valid:
        pol = pol.buffer(0)
    ref = np.array([f.n[0], f.n[1], 0.0])
    geoms = pol.geoms if hasattr(pol, "geoms") else [pol]
    for g in geoms:
        for t in shapely.constrained_delaunay_triangles(g).geoms:
            cs = np.array(t.exterior.coords)[:3]
            if shapely.Polygon(cs).area < 1e-6:
                continue
            m.face(orientar([f.p(s, d, z) for s, z in cs], ref), mat)


def tunel(m, f, contorno, d0, d1, mat, ref_dentro=True):
    """Faces laterais de um vão (o contorno em s, z), de d0 a d1."""
    pts = list(contorno) + [contorno[0]]
    for (s1, z1), (s2, z2) in zip(pts[:-1], pts[1:]):
        m.face([f.p(s1, d0, z1), f.p(s2, d0, z2), f.p(s2, d1, z2), f.p(s1, d1, z1)], mat)


def ogiva(sc, w, za, h_arranque, k=8, raio=0.8):
    """Contorno de um vão em arco quebrado (ogival): de za ao arranque h_arranque acima, e o arco com raio = raio × w
    (de baixo à esquerda, no sentido anti-horário)."""
    r = raio * w
    sa, sb = sc - w / 2, sc + w / 2
    zs = za + h_arranque
    th = math.acos((w / 2 - r) / r)
    out = [(sa, za), (sb, za), (sb, zs)]
    # arco direito: centro em (sb - r, zs), do ângulo 0 a th
    for a in np.linspace(0, th, k + 1)[1:]:
        out.append((sb - r + r * math.cos(a), zs + r * math.sin(a)))
    # arco esquerdo: centro em (sa + r, zs), de pi - th a pi
    for a in np.linspace(math.pi - th, math.pi, k + 1)[1:]:
        out.append((sa + r + r * math.cos(a), zs + r * math.sin(a)))
    return out


def redondo(sc, w, za, h_arranque, k=10):
    r = w / 2
    zs = za + h_arranque
    out = [(sc - r, za), (sc + r, za)]
    out += [(sc + r * math.cos(a), zs + r * math.sin(a)) for a in np.linspace(0, math.pi, k + 1)]
    return out


def reto(sc, w, za, zb):
    return [(sc - w / 2, za), (sc + w / 2, za), (sc + w / 2, zb), (sc - w / 2, zb)]


def ccw(pts):
    pts = [tuple(map(float, p)) for p in pts]
    return pts if shapely.Polygon(pts).exterior.is_ccw else pts[::-1]


def anel_aduelas(m, f, contorno_arco, larg, sal, mat, d0=0.0, n=None):
    """Aduelas à volta da parte curva de um vão: blocos entre o contorno e o contorno afastado larg para fora."""
    c = np.array(contorno_arco, float)
    cen = c.mean(0)
    seg = []
    for i in range(len(c) - 1):
        a, b = c[i], c[i + 1]
        t = unit(b - a)
        nn = np.array([t[1], -t[0]])
        if np.dot(nn, (a + b) / 2 - cen) < 0:
            nn = -nn
        seg.append((a, b, nn))
    for i, (a, b, nn) in enumerate(seg):
        na = nn if i == 0 else unit(nn + seg[i - 1][2])
        nb = nn if i == len(seg) - 1 else unit(nn + seg[i + 1][2])
        g = 0.006
        a2, b2 = a + (b - a) * g / max(np.linalg.norm(b - a), 1e-6) * 2, b - (b - a) * g / max(np.linalg.norm(b - a), 1e-6) * 2
        q = [a2, b2, b2 + nb * larg, a2 + na * larg]
        m.face(orientar([f.p(s, d0 + sal, z) for s, z in q], np.array([f.n[0], f.n[1], 0])), mat)
        for p1, p2 in zip(q, q[1:] + q[:1]):
            m.face([f.p(p1[0], d0, p1[1]), f.p(p2[0], d0, p2[1]), f.p(p2[0], d0 + sal, p2[1]), f.p(p1[0], d0 + sal, p1[1])], mat)


def degraus(m, p0, p1, larg, z0, z1, n, mat, mat_espelho=None, fundo=None, lado=0.0):
    """Escada reta de p0 (cota z0) a p1 (z1), com n degraus de largura larg (eixo p0-p1). Cada degrau é um bloco
    do fundo (fundo, ou z0 - 0,5) ao cobertor."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = unit(p1 - p0)
    nn = np.array([-t[1], t[0]])
    L = float(np.linalg.norm(p1 - p0))
    zf = (min(z0, z1) - 0.6) if fundo is None else fundo
    for i in range(n):
        a = p0 + t * L * i / n
        b = p0 + t * L * (i + 1) / n
        zt = z0 + (z1 - z0) * (i + 1) / n
        base = [a - nn * larg / 2 + nn * lado, b - nn * larg / 2 + nn * lado, b + nn * larg / 2 + nn * lado, a + nn * larg / 2 + nn * lado]
        m.prisma(ccw(base), zf, zt, mat, mat_topo=mat)


def corrimao(m, pts3, alt=0.95, mat="ferro", passo=1.1):
    """Corrimão de ferro: prumos e duas barras ao longo de uma linha 3D (o pé dos prumos)."""
    pts3 = [np.asarray(p, float) for p in pts3]
    tot = []
    for a, b in zip(pts3[:-1], pts3[1:]):
        L = np.linalg.norm(b - a)
        k = max(1, int(math.ceil(L / passo)))
        for i in range(k):
            tot.append(a + (b - a) * i / k)
    tot.append(pts3[-1])
    for p in tot:
        m.barra(p, p + np.array([0, 0, alt]), 0.022, mat, k=5)
    for h in (alt, alt * 0.5):
        for a, b in zip(pts3[:-1], pts3[1:]):
            m.barra(a + np.array([0, 0, h]), b + np.array([0, 0, h]), 0.02, mat, k=5)


def merlao(m, base, z0, z1, zp, mat, seteira=None):
    """Merlão: prisma de z0 a z1 e remate piramidal até zp; seteira = (plano, s, z) de um vão em arco na face."""
    base = ccw(base)
    m.prisma(base, z0, z1, mat, topo=False)
    m.piramide(base, z1, zp, mat)


def parapeito(m, linha, pol, z0, z_sill, esp, mat, mat_topo, ameias=None, fechado=False):
    """Parapeito ao longo de uma linha (x, y) do bordo de fora de um muro (pol), para dentro com a espessura esp, do
    adarve z0 ao peitoril z_sill; ameias = dict(larg, passo, z1, zp, seteiras) põe os merlões por cima."""
    pts = [np.asarray(p, float) for p in linha]
    if fechado:
        pts = pts + [pts[0]]
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        t = unit(b - a)
        nn = np.array([-t[1], t[0]])
        if not pol.buffer(0.05).contains(shapely.Point(*((a + b) / 2 + nn * 0.3))):
            nn = -nn
        segs.append((a, b, t, nn))
    for i, (a, b, t, nn) in enumerate(segs):
        na = nn if i == 0 and not fechado else unit(nn + segs[i - 1][2 + 1]) if True else nn
        # junta em esquadria com o troço anterior e o seguinte
        n_ant = segs[i - 1][3] if (i > 0 or fechado) else nn
        n_seg = segs[(i + 1) % len(segs)][3] if (i < len(segs) - 1 or fechado) else nn
        ma = unit(nn + n_ant)
        mb = unit(nn + n_seg)
        ka = esp / max(np.dot(ma, nn), 0.3)
        kb = esp / max(np.dot(mb, nn), 0.3)
        q = [a, b, b + mb * kb, a + ma * ka]
        m.prisma(ccw(q), z0, z_sill, mat, mat_topo=mat_topo)
    if not ameias:
        return 0
    # merlões: os dos cantos e, entre eles, ao passo dado
    w, passo = ameias["larg"], ameias["passo"]
    n_m = 0
    cantos = [pts[0]] + [s[1] for s in segs]
    for i, (a, b, t, nn) in enumerate(segs):
        L = float(np.linalg.norm(b - a))
        k = max(1, int(round((L - w) / passo)))
        ss = np.linspace(w / 2, L - w / 2, k + 1)
        if i > 0 or fechado:
            ss = ss[1:]  # o do canto já vem do troço anterior
        for j, s in enumerate(ss):
            c = a + t * s
            base = [c - t * w / 2, c + t * w / 2, c + t * w / 2 + nn * esp, c - t * w / 2 + nn * esp]
            merlao(m, base, z_sill, ameias["z1"], ameias["zp"], mat)
            n_m += 1
            st = ameias.get("seteiras")
            if st and (j % st == 1) and 0.2 < s < L - 0.2:
                # seteira em arco no pé do merlão, para fora (fotografias: buraco escuro de c. 0,2 × 0,35 m)
                f = Plano(c, t, n=-nn)
                cont = redondo(0.0, 0.2, z_sill + 0.05, 0.22, k=6)
                m.face(orientar([f.p(s_, 0.01, z_) for s_, z_ in cont], np.array([-nn[0], -nn[1], 0])), "sombra")
    return n_m


# ---------------------------------------------------------------- cena e materiais

for _ob in list(bpy.data.objects):  # a cena de arranque do bpy traz um cubo, uma luz e uma câmara
    bpy.data.objects.remove(_ob)
RAIZ_OB = bpy.data.objects.new("Castelo_Montalegre", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = CFG["origem"]["lat"], CFG["origem"]["lon"]
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
# "Muralhas_grupo" é o que o visor enquadra ao abrir; o terreno chama-se "Terreno"
for g in ["Terreno", "Torre", "Torres", "Muralhas", "Envolvente", "Vegetacao"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

# Granito de Montalegre nas fotografias: cinzento médio, neutro a ligeiramente esverdeado com o céu encoberto
# (c. 100, 103, 95 em sRGB) e cinzento-claro amarelado ao sol de perto (187, 177, 159); muitos líquenes brancos e
# manchas escuras de escorrência; silhares de fiadas regulares (0,30–0,45 m) nas torres, alvenaria mais miúda e
# irregular na cerca (fotografias de 2005, 2018 e 2023).
PAL_T = np.array([(0.560, 0.555, 0.530), (0.600, 0.592, 0.565), (0.515, 0.512, 0.495), (0.575, 0.556, 0.520),
                  (0.535, 0.540, 0.522)])
TEX, ALT = textura_cantaria(semente=1273, fiada=(0.30, 0.45), comp=(0.45, 1.15), liquen=0.70, paleta=PAL_T,
                            prob=[0.40, 0.18, 0.18, 0.10, 0.14], var=(0.94, 1.05), liq_forca=0.30, tinta=(1.0, 1.0, 0.985))
material("cantaria_torre", (1, 1, 1), 0.9, imagem("cantaria_torre", TEX), normal=imagem("cantaria_torre_n", normais(ALT, 3.0)),
         forca=0.8, escala=0.5, vcor=True)
TEX, ALT = textura_cantaria(semente=1331, fiada=(0.20, 0.40), comp=(0.30, 0.85), liquen=0.68, paleta=PAL_T,
                            prob=[0.38, 0.16, 0.22, 0.10, 0.14], junta_m=0.012, irregular=0.06, var=(0.90, 1.08),
                            liq_forca=0.32, tinta=(0.99, 1.0, 0.975))
material("muro", (1, 1, 1), 0.93, imagem("muro", TEX), normal=imagem("muro_n", normais(ALT, 3.4)), forca=0.95, escala=0.5, vcor=True)
TEX, ALT = textura_cantaria(semente=1332, fiada=(0.35, 0.70), comp=(0.40, 0.95), liquen=0.62, paleta=PAL_T,
                            prob=[0.38, 0.18, 0.18, 0.12, 0.14], junta_m=0.016, irregular=0.07, var=(0.92, 1.07),
                            liq_forca=0.42, tinta=(0.99, 1.0, 0.97))
material("muro_topo", (1, 1, 1), 0.95, imagem("muro_topo", TEX), normal=imagem("muro_topo_n", normais(ALT, 3.5)), forca=1.0, escala=0.5, vcor=True)
material("cantaria_lisa", (0.50, 0.495, 0.47), 0.85)
material("intradorso", (0.34, 0.335, 0.32), 0.95)
material("sombra", (0.04, 0.04, 0.04), 1.0)
material("madeira", (0.34, 0.25, 0.17), 0.8)
material("madeira_porta", (0.26, 0.17, 0.11), 0.75)
material("tabuado", (0.50, 0.47, 0.42), 0.85)      # passadiço: tábuas de madeira acinzentada (fotografia de 2019)
material("ferro", (0.13, 0.13, 0.14), 0.5)
material("corten", (0.50, 0.25, 0.13), 0.75)       # aço corten da escada, dos guarda-corpos e do caminho do adarve (fotografias)
material("corten_escuro", (0.36, 0.18, 0.10), 0.8)
material("bandeira_verde", (0.0, 0.40, 0.18), 0.8)
material("bandeira_vermelha", (0.80, 0.08, 0.10), 0.8)
material("bandeira_amarela", (0.95, 0.80, 0.10), 0.8)
TEX, ALT = textura_telha()
# telhados das torres: telha vermelho-alaranjada (fotografias de drone: c. 195, 114, 80)
material("telha", (1, 1, 1), 0.8, imagem("telha", TEX * np.array([0.80, 0.42, 0.28])), normal=imagem("telha_n", normais(ALT, 2.0)), forca=0.6)
TEX, ALT = textura_cubos(base=(0.70, 0.69, 0.66))
material("lajeado", (1, 1, 1), 0.92, imagem("lajeado", TEX), normal=imagem("lajeado_n", normais(ALT, 1.5)), forca=0.5)  # caminho claro da barbacã
TEX, ALT = textura_relva()
COR_RELVA = tuple(float(v) for v in (np.clip(TEX, 0, 1) ** 2.2).mean((0, 1)))
_l = TEX.mean(2, keepdims=True)
_l = np.clip(0.9 + 0.35 * (_l / _l.mean() - 1.0), 0, 1)   # c. 0,9 (sRGB) = 0,79 linear; compensado na cor
material("chao", (1, 1, 1), 0.93, imagem("chao", np.repeat(_l, 3, 2)), normal=imagem("chao_n", normais(ALT, 1.0)), forca=0.35, vcor=True)
material("asfalto", (0.36, 0.36, 0.37), 0.95)
material("cinza_viz", (0.74, 0.74, 0.73), 0.9)       # casas vizinhas: um só cinzento claro
material("base", (0.33, 0.26, 0.20), 0.95)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)
material("agua", (0.10, 0.12, 0.11), 0.3)

MEDIDAS = {}


def patine_torre(co):
    """A cantaria escurece para o alto e nos cunhais (líquenes e escorrências) e junto ao chão (musgo)."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x * 0.7 + 300, y * 0.7 + z * 0.6, 4.0, 77)
    g2 = fbm(x * 2.0 + z * 0.15, y * 2.0 - z * 0.15, 3.0, 91)       # escorrências verticais
    v = 0.90 + 0.18 * g - 0.10 * np.clip((g2 - 0.55) * 3, 0, 1)
    pe = np.clip(1 - (z - cota(x, y)) / 1.6, 0, 1) * 0.18
    c = np.stack([v, v * 1.005, v * 0.985], 1)
    return np.clip(c * (1 - pe)[:, None], 0, 1)


def patine_muro(co):
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x + 500, y + 500 + z * 0.7, 6.0, 31)
    v = 0.86 + 0.26 * g
    pe = np.clip(1 - (z - cota(x, y)) / 1.4, 0, 1) * 0.18
    c = np.stack([v * 0.99, v * (1.0 + 0.02 * (g - 0.5)), v * 0.97], 1)
    return np.clip(c * (1 - pe)[:, None], 0, 1)


# ---------------------------------------------------------------- torres (genérico)

class Torre:
    """Torre retangular de comum.TORRES: centro c, eixos eu (azimute az) e ev (90° à esquerda), meias medidas."""

    def __init__(self, nome):
        t = C.TORRES[nome]
        self.nome = nome
        self.c = np.array(t["c"], float)
        eu, ev = C.eixos(t["az"])
        self.eu, self.ev = np.array(eu), np.array(ev)
        self.mu, self.mv = t["mu"], t["mv"]

    def W(self, u, v):
        return self.c + self.eu * u + self.ev * v

    def rect(self, ru, rv):
        return [self.W(-ru, -rv), self.W(ru, -rv), self.W(ru, rv), self.W(-ru, rv)]

    def faces(self, folga=0.0):
        """Planos das quatro faces: s ao longo da face, da esquerda para a direita de quem olha de fora."""
        mu, mv = self.mu + folga, self.mv + folga
        return {"+u": (Plano(self.W(mu, 0), self.ev, n=self.eu), mv),
                "+v": (Plano(self.W(0, mv), -self.eu, n=self.ev), mu),
                "-u": (Plano(self.W(-mu, 0), -self.ev, n=-self.eu), mv),
                "-v": (Plano(self.W(0, -mv), self.eu, n=-self.ev), mu)}

    def chao(self, d=1.2):
        """Chão (MDT) a d metros de cada face: o mínimo e o mediano."""
        zs = []
        for f, meia in self.faces().values():
            for s in np.linspace(-meia + 0.5, meia - 0.5, 9):
                zs.append(cota(*f.xy(s, d)))
        return float(np.min(zs)), float(np.median(zs))

    def pol(self, folga=0.0):
        return shapely.Polygon(self.rect(self.mu + folga, self.mv + folga))


def corpo(m, T, z0, z1, vaos=None, mat=0, mat_vao="cantaria_lisa"):
    """As quatro faces da torre de z0 a z1, com os vãos (tipo, s, largura, z do peitoril, altura) por face."""
    vaos = vaos or {}
    for nome, (f, meia) in T.faces().items():
        furos, extra = [], []
        for tipo, s, w, za, h in vaos.get(nome, []):
            if tipo == "porta":
                c = redondo(s, w, za, h - w / 2, k=12)
            elif tipo == "janela":
                c = reto(s, w, za, za + h)
            else:
                c = redondo(s, w, za, h - w / 2, k=4)
            furos.append(c)
            extra.append((tipo, s, w, za, h, c))
        painel(m, f, [(-meia, z0), (meia, z0), (meia, z1), (-meia, z1)], furos, mat)
        for tipo, s, w, za, h, c in extra:
            prof = 0.95 if tipo == "porta" else 0.7
            tunel(m, f, c, 0.0, -prof, mat_vao if tipo != "seteira" else "intradorso")
            painel(m, f, c, (), "madeira_porta" if tipo == "porta" else "sombra", d=-prof)
            if tipo in ("porta", "janela"):
                # ombreiras e lintel/aduelas de cantaria, ligeiramente salientes
                if tipo == "porta":
                    arco = [p for p in c if p[1] >= za + h - w / 2 - 1e-6]
                    anel_aduelas(m, f, arco, 0.30, 0.04, mat_vao)
                    zr = za + h - w / 2
                else:
                    m.caixa_face(f, s - w / 2 - 0.22, s + w / 2 + 0.22, 0.0, 0.04, za + h, za + h + 0.28, mat_vao)
                    zr = za + h
                for s0_, s1_ in ((s - w / 2 - 0.22, s - w / 2), (s + w / 2, s + w / 2 + 0.22)):
                    m.caixa_face(f, s0_, s1_, 0.0, 0.04, za, zr, mat_vao)
                m.caixa_face(f, s - w / 2 - 0.15, s + w / 2 + 0.15, 0.0, 0.10, za - 0.12, za, mat_vao)


def coroamento(m, T, esp, z_adarve, z_sill, merl=None, mat=0, mat_topo="cantaria_lisa", mat_chao="muro_topo"):
    """Parapeito de espessura esp por cima das faces (de z_adarve a z_sill), o adarve por dentro e os merlões:
    merl = dict(n=por face contando os dos cunhais, larg, z1, zp)."""
    fs = T.faces()
    qo = T.rect(T.mu, T.mv)
    qi = T.rect(T.mu - esp, T.mv - esp)
    for i in range(4):
        a, b, c_, d = qo[i], qo[(i + 1) % 4], qi[(i + 1) % 4], qi[i]
        m.face(orientar([(*a, z_sill), (*b, z_sill), (*c_, z_sill), (*d, z_sill)], np.array([0, 0, 1.0])), mat_topo)
        m.face(orientar([(*d, z_adarve), (*c_, z_adarve), (*c_, z_sill), (*d, z_sill)], -np.array([*unit((a + b) / 2 - T.c), 0])), mat)
    n = 0
    if merl:
        for nome, (f, meia) in fs.items():
            L = 2 * meia
            k = merl["n"]
            w = merl["larg"]
            passo = (L - esp) / (k - 1)
            for j in range(k - 1):   # o do cunhal da direita é o da esquerda da face seguinte
                s0 = -meia + j * passo
                if j == 0:
                    base = [f.xy(s0, 0.0), f.xy(s0 + esp, 0.0), f.xy(s0 + esp, -esp), f.xy(s0, -esp)]
                else:
                    sc = -meia + esp / 2 + j * passo
                    base = [f.xy(sc - w / 2, 0.0), f.xy(sc + w / 2, 0.0), f.xy(sc + w / 2, -esp), f.xy(sc - w / 2, -esp)]
                merlao(m, base, z_sill, merl["z1"], merl["zp"], mat)
                n += 1
    return n


def chao_anel(m, T, r_fora, r_dentro, z, mat):
    """Anel horizontal entre dois retângulos (adarve, caleira)."""
    qo, qi = T.rect(T.mu - r_fora, T.mv - r_fora), T.rect(T.mu - r_dentro, T.mv - r_dentro)
    for i in range(4):
        a, b, c_, d = qo[i], qo[(i + 1) % 4], qi[(i + 1) % 4], qi[i]
        m.face(orientar([(*a, z), (*b, z), (*c_, z), (*d, z)], np.array([0, 0, 1.0])), mat)


def telhado(m, T, recuo, z_beiral, z_cume, mat="telha", beiral=0.18):
    """Telhado de quatro águas de telha por dentro do parapeito (recuo das faces), com cumeeira se a planta não for
    quadrada, beiral saliente e telhões nas arestas e na cumeeira."""
    a, b = T.mu - recuo + beiral, T.mv - recuo + beiral
    zb = z_beiral - beiral * (z_cume - z_beiral) / max(min(a, b) - beiral, 0.5)
    if a >= b:
        r0, r1 = T.W(-(a - b), 0), T.W(a - b, 0)
    else:
        r0, r1 = T.W(0, -(b - a)), T.W(0, b - a)
    c00, c10, c11, c01 = T.rect(a, b)
    R0, R1 = (*r0, z_cume), (*r1, z_cume)
    P = lambda p: (p[0], p[1], zb)  # noqa: E731
    if a >= b:   # cumeeira ao longo de u: águas grandes a -v e +v, triângulos nos topos ±u
        m.face([P(c00), P(c10), R1, R0], mat)
        m.face([P(c11), P(c01), R0, R1], mat)
        m.face([P(c10), P(c11), R1], mat)
        m.face([P(c01), P(c00), R0], mat)
    else:
        m.face([P(c10), P(c11), R1, R0], mat)
        m.face([P(c01), P(c00), R0, R1], mat)
        m.face([P(c00), P(c10), R0], mat)
        m.face([P(c11), P(c01), R1], mat)
    m.face([P(c00), P(c01), P(c11), P(c10)], mat)   # face de baixo do beiral
    for p, r in ((c00, R0), (c01, R0), (c10, R1), (c11, R1)):
        m.barra((p[0], p[1], zb + 0.04), (r[0], r[1], r[2] + 0.04), 0.07, mat, k=5)
    if np.linalg.norm(np.array(r1) - np.array(r0)) > 0.05:
        m.barra((*r0, z_cume + 0.04), (*r1, z_cume + 0.04), 0.08, mat, k=5)


def seteira_face(m, f, s, z0, h, w=0.14, prof=0.6):
    c = redondo(s, w, z0, h - w / 2, k=4)
    return c


def misula(m, f, s, z_topo, larg=0.38, sal=0.85, n=3, alt=0.42, mat="cantaria_lisa"):
    """Mísula (cachorro) em degraus: n fiadas que saem cada vez mais da face até sal, do alto z_topo para baixo."""
    for i in range(n):
        za = z_topo - (i + 1) * alt
        zb = z_topo - i * alt
        d1 = sal * (n - i) / n
        m.caixa_face(f, s - larg / 2, s + larg / 2, -0.05, d1, za, zb, mat, fundo=True)


def balcao(m, f, s, w, z_piso, sal, z_par, merloes=2, z_m1=None, z_mp=None, n_mis=3, mat=0, esp=0.28):
    """Balcão com matacães sobre mísulas: caixa de largura w saliente sal da face, piso a z_piso, parapeito até z_par
    e merlões piramidais por cima; entre as mísulas, os buracos dos matacães (sombra)."""
    # mísulas
    ss = np.linspace(s - w / 2 + 0.2, s + w / 2 - 0.2, n_mis)
    for sm in ss:
        misula(m, f, sm, z_piso, larg=0.36, sal=sal, n=3, alt=0.38)
    # piso (laje) com os buracos dos matacães entre as mísulas
    m.caixa_face(f, s - w / 2, s + w / 2, 0.0, sal, z_piso - 0.14, z_piso, "cantaria_lisa", fundo=True)
    for a_, b_ in zip(ss[:-1], ss[1:]):
        q = [f.p(a_ + 0.2, 0.05, z_piso - 0.145), f.p(b_ - 0.2, 0.05, z_piso - 0.145), f.p(b_ - 0.2, sal - esp, z_piso - 0.145), f.p(a_ + 0.2, sal - esp, z_piso - 0.145)]
        m.face(orientar(q, np.array([0, 0, -1.0])), "sombra")
    # parapeito: frente e lados
    m.caixa_face(f, s - w / 2, s + w / 2, sal - esp, sal, z_piso, z_par, mat)
    m.caixa_face(f, s - w / 2, s - w / 2 + esp, 0.0, sal - esp, z_piso, z_par, mat)
    m.caixa_face(f, s + w / 2 - esp, s + w / 2, 0.0, sal - esp, z_piso, z_par, mat)
    # cobertura de lajes inclinada (para a face) por cima do vão
    m.face([f.p(s - w / 2, 0.0, z_par + 0.35), f.p(s + w / 2, 0.0, z_par + 0.35), f.p(s + w / 2, sal - esp, z_par), f.p(s - w / 2, sal - esp, z_par)], "cantaria_lisa")
    if merloes:
        z1 = z_m1 or z_par + 0.55
        zp = z_mp or z1 + 0.6
        for sm in np.linspace(s - w / 2 + 0.25, s + w / 2 - 0.25, merloes):
            merlao(m, [f.xy(sm - 0.25, sal), f.xy(sm + 0.25, sal), f.xy(sm + 0.25, sal - esp), f.xy(sm - 0.25, sal - esp)], z_par, z1, zp, mat)


def guarita(m, T, iu, iv, z_piso, lado, z_par, z_m1, z_mp, mat=0, esp=0.28):
    """Guarita (balcão de canto) sobre mísulas no cunhal (iu, iv = ±1): caixa quadrada de lado `lado` centrada no
    cunhal, a sair c. 0,9 m de cada face, com dois merlões piramidais por lado de fora."""
    cn = T.W(iu * T.mu, iv * T.mv)
    du, dv = T.eu * iu, T.ev * iv
    sal = lado / 2 + 0.10
    c0 = cn + (du + dv) * (sal - lado / 2)
    q = [c0 + du * (sal) + dv * (sal), c0 + du * (sal) - dv * (lado - sal), c0 - du * (lado - sal) - dv * (lado - sal), c0 - du * (lado - sal) + dv * (sal)]
    q = ccw(q)
    # mísulas em degraus por baixo, três fiadas a recuar para o cunhal
    for i in range(3):
        f_ = 1.0 - (i + 1) * 0.28
        qi = [cn + (np.asarray(p) - cn) * f_ for p in q]
        m.prisma(ccw(qi), z_piso - 0.14 - (i + 1) * 0.40, z_piso - 0.14 - i * 0.40, "cantaria_lisa", fundo=True)
    m.prisma(q, z_piso - 0.14, z_piso, "cantaria_lisa", fundo=True)
    # paredes da guarita (anel) e merlões nos dois lados de fora
    qi = ccw([c0 + (np.asarray(p) - c0) * (1 - 2 * esp / lado) for p in q])
    for i in range(4):
        a, b = np.asarray(q[i]), np.asarray(q[(i + 1) % 4])
        ai, bi = np.asarray(qi[i]), np.asarray(qi[(i + 1) % 4])
        m.prisma(ccw([a, b, bi, ai]), z_piso, z_par, mat, mat_topo="cantaria_lisa")
    for lado_vec, outro in ((du, dv), (dv, du)):
        for t in (0.28, 0.72):
            pc = c0 + lado_vec * (sal - esp / 2) + outro * (sal - lado * t)
            base = [pc - outro * 0.25 - lado_vec * esp / 2, pc + outro * 0.25 - lado_vec * esp / 2, pc + outro * 0.25 + lado_vec * esp / 2, pc - outro * 0.25 + lado_vec * esp / 2]
            merlao(m, base, z_par, z_m1, z_mp, mat)
    # seteira em cada lado de fora
    for lado_vec, outro in ((du, dv), (dv, du)):
        pc = c0 + lado_vec * (sal + 0.012) + outro * (sal - lado / 2)
        f = Plano(pc, outro if np.cross(np.append(lado_vec, 0), np.append(outro, 0))[2] < 0 else -outro, n=lado_vec)
        m.face(orientar([f.p(s_, 0, z_) for s_, z_ in redondo(0.0, 0.14, z_piso + 0.35, 0.55, k=4)], np.array([*lado_vec, 0])), "sombra")

# ---------------------------------------------------------------- peças genéricas

import perfil as PF  # noqa: E402


def placa(m, f, cont, esp, mat, d=0.0):
    """Chapa plana (aço) no plano f, com o contorno (s, z) e a espessura esp para dentro (d até d - esp)."""
    painel(m, f, cont, (), mat, d=d)
    pol = shapely.Polygon(cont)
    if not pol.is_valid:
        pol = pol.buffer(0)
    ref = -np.array([f.n[0], f.n[1], 0.0])
    for t in shapely.constrained_delaunay_triangles(pol).geoms:
        cs = np.array(t.exterior.coords)[:3]
        if shapely.Polygon(cs).area > 1e-6:
            m.face(orientar([f.p(s, d - esp, z) for s, z in cs], ref), mat)
    tunel(m, f, cont, d, d - esp, mat)


def prisma_z(m, pol, z0, z1, mat, mat_topo):
    """Prisma de um polígono com o fundo e o topo por vértice (z0 e z1 são funções de x, y); o topo é triangulado."""
    pol = shapely.geometry.polygon.orient(pol, 1)
    for anel in [pol.exterior] + list(pol.interiors):
        b = np.array(anel.coords)[:-1]
        if not anel.is_ccw and anel is pol.exterior:
            b = b[::-1]
        zz0 = [z0(*p) for p in b]
        zz1 = [z1(*p) for p in b]
        n = len(b)
        for i in range(n):
            a, c = b[i], b[(i + 1) % n]
            m.face([(a[0], a[1], zz0[i]), (c[0], c[1], zz0[(i + 1) % n]), (c[0], c[1], zz1[(i + 1) % n]), (a[0], a[1], zz1[i])], mat)
    for t in shapely.constrained_delaunay_triangles(pol).geoms:
        cs = np.array(t.exterior.coords)[:3]
        if shapely.Polygon(cs).area < 1e-5:
            continue
        m.face(orientar([(x, y, z1(x, y)) for x, y in cs], np.array([0, 0, 1.0])), mat_topo)


def chao_min(pol, folga=1.5):
    b = pol.buffer(folga)
    xs, ys = np.array(b.exterior.coords).T
    return float(np.min(cota(xs, ys)))


def setor(a0, a1, r0=8.0, r1=30.0, passo=1.0):
    """Setor de coroa circular à volta do centro do pátio, entre os azimutes a0 e a1."""
    az = np.arange(a0, a1 + 1e-6, passo)
    if az[-1] < a1:
        az = np.append(az, a1)
    return shapely.Polygon([PF.ponto(a, r1) for a in az] + [PF.ponto(a, r0) for a in az[::-1]])


# ---------------------------------------------------------------- torre de menagem

TM = Torre("menagem")
Z_ADARVE_TM, Z_SILL_TM, Z_M1_TM, Z_MP_TM = 1037.53, 1038.35, 1039.05, 1039.72
Z_PLAT = 1017.60     # patamar de pedra da porta (face sul), estimado nas fotografias
Z_PISO_BAL = 1034.0  # piso dos balcões e das guaritas (LiDAR: pontos dos parapeitos a 1035,4)


def torre_menagem():
    """Torre de menagem (LiDAR 2024: 12,1 × 12,0 m, adarve a 1037,53 m, pontas dos merlões a 1039,72 m):
    - nove merlões piramidais por face, contando os dos cunhais (fotografias de 2005, 2019 e 2023);
    - um balcão com matacães a meio de cada face e uma guarita sobre mísulas em cada cunhal, a c. 1034 m;
    - telhado de quatro águas de telha por dentro do adarve, a casota da escada e o mastro da bandeira;
    - porta alta em arco de volta perfeita na face sul, ao nível do patamar de pedra, e seteiras."""
    T = TM
    m = Malha(["cantaria_torre", "cantaria_lisa", "sombra", "madeira_porta", "muro_topo", "intradorso"])
    z_min, z_med = T.chao()
    z0 = z_min - 0.8
    vaos = {
        "-v": [("porta", -0.3, 1.0, Z_PLAT, 2.15), ("seteira", 3.0, 0.14, 1024.0, 0.95), ("seteira", -2.6, 0.14, 1029.6, 0.95)],
        "+v": [("seteira", 1.5, 0.14, 1021.5, 0.95), ("seteira", -1.8, 0.14, 1027.5, 0.95), ("seteira", 0.6, 0.14, 1031.0, 0.85)],
        "+u": [("seteira", -2.0, 0.14, 1020.5, 0.95), ("seteira", 2.2, 0.14, 1029.8, 0.95)],
        "-u": [("seteira", 2.5, 0.14, 1020.0, 0.95), ("seteira", -2.0, 0.14, 1025.6, 0.95), ("seteira", 1.2, 0.14, 1031.2, 0.85)],
    }
    corpo(m, T, z0, Z_SILL_TM, vaos, mat=0, mat_vao="cantaria_lisa")
    pitch = (2 * T.mu - 0.72) / 8
    n_m = coroamento(m, T, 0.72, Z_ADARVE_TM, Z_SILL_TM, merl=dict(n=9, larg=0.74, z1=Z_M1_TM, zp=Z_MP_TM), mat=0)
    chao_anel(m, T, 0.72, 2.1, Z_ADARVE_TM, "muro_topo")
    # balcões com matacães a meio das faces (posições medidas nos pontos do LiDAR que saem das faces)
    fs = T.faces()
    for nome, s in (("-v", 0.0), ("+v", -0.25), ("+u", 0.25), ("-u", 2.25)):
        f, _ = fs[nome]
        balcao(m, f, s, 2.4, Z_PISO_BAL, 0.9, 1035.4, merloes=2, z_m1=1035.78, z_mp=1036.32, mat=0)
    for iu in (-1, 1):
        for iv in (-1, 1):
            guarita(m, T, iu, iv, Z_PISO_BAL, 1.6, 1035.4, 1035.78, 1036.32, mat=0)
    # par de mísulas soltas a meia altura na face nascente (fotografia de 2018 da escada)
    f, _ = fs["+u"]
    for s in (-0.55, 0.55):
        misula(m, f, s, 1026.9, larg=0.36, sal=0.6, n=2, alt=0.36)
    # patamar de pedra da porta, com cornija (fotografia da escada de aço)
    f, _ = fs["-v"]
    zp0 = chao_min(shapely.Polygon([f.xy(-6.05, 0), f.xy(1.5, 0), f.xy(1.5, 2.3), f.xy(-6.05, 2.3)]), 0.5) - 0.6
    m.caixa_face(f, -6.05, 1.5, -0.05, 2.3, zp0, Z_PLAT - 0.25, 0, topo=False)
    m.caixa_face(f, -6.05, 1.62, -0.05, 2.42, Z_PLAT - 0.25, Z_PLAT, 1, topo=False)
    m.face(orientar([f.p(-6.05, -0.05, Z_PLAT), f.p(1.62, -0.05, Z_PLAT), f.p(1.62, 2.42, Z_PLAT), f.p(-6.05, 2.42, Z_PLAT)], np.array([0, 0, 1.0])), 4)
    m.objeto("torre_menagem", "torre", GRUPOS["Torre"], uv=3.0, vcor=patine_torre)
    # telhado e casota da escada
    t = Malha(["telha", "cantaria_lisa", "sombra", "madeira_porta"])
    telhado(t, T, 2.05, 1037.62, 1039.6)
    q = [T.W(-1.85, -4.6), T.W(-0.15, -4.6), T.W(-0.15, -1.0), T.W(-1.85, -1.0)]
    t.prisma(ccw(q), Z_ADARVE_TM, 1040.0, 1)
    t.prisma(ccw([T.W(-1.97, -4.72), T.W(-0.03, -4.72), T.W(-0.03, -0.88), T.W(-1.97, -0.88)]), 1040.0, 1040.15, 1, fundo=True)
    fc = Plano(T.W(-1.0, -4.6), T.eu, n=-T.ev)
    t.face(orientar([fc.p(s_, 0.012, z_) for s_, z_ in reto(0.0, 0.78, Z_ADARVE_TM, Z_ADARVE_TM + 1.95)], np.array([*(-T.ev), 0])), 3)
    t.objeto("torre_telhado", "torre", GRUPOS["Torre"], uv=1.0)
    bandeira(T.W(-0.6, -4.3))
    MEDIDAS["torre"] = {"lado_u": round(2 * T.mu, 2), "lado_v": round(2 * T.mv, 2), "azimute_faces": [97.5, 7.5], "chao": round(z_min, 2),
                        "adarve": Z_ADARVE_TM, "peitoril": Z_SILL_TM, "pontas": Z_MP_TM, "cume": 1039.6, "merloes": n_m,
                        "merloes_por_face": 9, "passo_merloes": round(pitch, 2), "balcoes": 4, "guaritas": 4, "porta": Z_PLAT,
                        "altura": round(Z_MP_TM - z_min, 1)}


def bandeira(p):
    """Mastro com a bandeira nacional por cima da casota da escada (fotografias de drone)."""
    m = Malha(["ferro", "bandeira_verde", "bandeira_vermelha", "bandeira_amarela"])
    m.cilindro(p, 0.045, 1040.15, 1043.8, 0, k=8)
    d = unit(TM.eu)
    h0, h1 = 1042.7, 1043.7
    Lb = 1.5
    for (a, b, mt) in ((0.0, 0.4, 1), (0.4, 1.0, 2)):
        pa, pb = p + d * Lb * a, p + d * Lb * b
        m.face([(*pa, h0), (*pb, h0), (*pb, h1), (*pa, h1)], mt)
        m.face([(*pa, h1), (*pb, h1), (*pb, h0), (*pa, h0)], mt)
    pc = p + d * Lb * 0.4
    for s in (1, -1):
        m.cilindro((pc[0] - TM.ev[0] * 0.012 * s, pc[1] - TM.ev[1] * 0.012 * s), 0.16, 1043.04, 1043.36, 3, k=10)
    m.objeto("bandeira", "torre", GRUPOS["Torre"], uv=None, recalc=False)


def escada_aco():
    """Escada de aço corten encostada à face sul da torre, a subir de nascente (pátio) para o patamar da porta:
    degraus de chapa dobrada (perfil em dente de serra) entre duas chapas cheias de guarda (fotografias de 2018
    e de drone); posição e inclinação estimadas nas fotografias (a face sul está tapada no LiDAR)."""
    T = TM
    m = Malha(["corten", "corten_escuro"])
    f, _ = T.faces()["-v"]   # s = +u, d para fora (sul)
    s_pe, s_topo = 5.95, 1.5
    z_pe = float(cota(*f.xy(s_pe + 0.4, 0.6)))
    n = int(round((Z_PLAT - z_pe) / 0.185))
    esp_d = (Z_PLAT - z_pe) / n
    run = (s_pe - s_topo) / n
    d0, d1 = 0.08, 1.12
    for i in range(n):
        sa = s_pe - i * run
        zt = z_pe + (i + 1) * esp_d
        m.caixa_face(f, sa - run, sa + 0.03, d0, d1, zt - 0.045, zt, 0, fundo=True)          # cobertor
        m.caixa_face(f, sa - 0.03, sa, d0, d1, zt - esp_d, zt - 0.045, 1, fundo=False)     # espelho
    # chapas de guarda dos dois lados: dente de serra por baixo e bordo de cima paralelo à escada
    for d in (d0, d1 + 0.025):
        cont = []
        for i in range(n):
            sa = s_pe - i * run
            cont += [(sa, z_pe + i * esp_d - 0.1), (sa, z_pe + (i + 1) * esp_d - 0.1)]
        cont += [(s_topo - 0.05, Z_PLAT - 0.1), (s_topo - 0.05, Z_PLAT + 1.0), (s_pe + 0.05, z_pe + 1.0), (s_pe + 0.05, z_pe - 0.1)]
        cont = [(-s, z) for s, z in cont]      # o painel quer contornos válidos; s ao contrário para ficar anti-horário
        fr = Plano(f.xy(0, d), -f.t, n=f.n)
        placa(m, fr, cont, 0.025, 0)
    m.objeto("escada_aco", "escada_torre", GRUPOS["Torre"], uv=1.0)
    MEDIDAS["escada_torre"] = {"degraus": n, "pe": round(z_pe, 2), "patamar": Z_PLAT, "espelho": round(esp_d, 3), "largura": round(d1 - d0, 2)}


# ---------------------------------------------------------------- torres do sul

def torre_simples(nome, elemento, z_sill, esp, z_adarve, telh, merl=None, vaos=None, rotulo=""):
    T = Torre(nome)
    m = Malha(["cantaria_torre", "cantaria_lisa", "sombra", "madeira_porta", "muro_topo", "intradorso"])
    z_min, _ = T.chao()
    corpo(m, T, z_min - 0.8, z_sill, vaos or {}, mat=0)
    n_m = coroamento(m, T, esp, z_adarve, z_sill, merl=merl, mat=0)
    chao_anel(m, T, esp, telh[0], z_adarve, "muro_topo")
    m.objeto("torre_" + nome, elemento, GRUPOS["Torres"], uv=3.0, vcor=patine_torre)
    t = Malha(["telha"])
    telhado(t, T, telh[0], telh[1], telh[2], beiral=telh[3] if len(telh) > 3 else 0.18)
    t.objeto("telhado_" + nome, elemento, GRUPOS["Torres"], uv=1.0)
    MEDIDAS["torre_" + nome] = {"lados": [round(2 * T.mu, 2), round(2 * T.mv, 2)], "chao": round(z_min, 2), "peitoril": z_sill,
                                "pontas": merl["zp"] if merl else z_sill, "cume": telh[2], "merloes": n_m, "altura": round((merl["zp"] if merl else telh[2]) - z_min, 1)}
    return T


def torres_sul():
    # torre sueste: a maior das três, com onze merlões por face (LiDAR: pontas a 1029,8 m)
    torre_simples("se", "torre_se", 1028.45, 0.75, 1027.55, (1.5, 1027.62, 1029.2),
                  merl=dict(n=11, larg=0.56, z1=1029.2, zp=1029.8),
                  vaos={"+u": [("seteira", 0.8, 0.14, 1019.5, 0.95)], "+v": [("seteira", -1.0, 0.14, 1021.0, 0.95), ("seteira", 2.0, 0.14, 1015.0, 0.95)],
                        "-v": [("seteira", 1.6, 0.14, 1023.4, 0.9)]})
    # torre sul: parapeito liso e telhado por dentro (LiDAR: topo a 1025,95 m)
    torre_simples("s", "torre_s", 1025.95, 0.5, 1024.9, (0.65, 1025.08, 1026.3, 0.12),
                  vaos={"+u": [("seteira", 0.0, 0.14, 1016.8, 1.0), ("seteira", -1.2, 0.14, 1022.2, 0.9)], "-v": [("seteira", 0.5, 0.14, 1019.0, 0.9)]})
    # torre pequena de sudoeste: rebordo fino e pirâmide baixa; porta em arco para o adarve (fotografia)
    torre_simples("so", "torre_so", 1023.15, 0.3, 1022.85, (0.32, 1022.9, 1023.7, 0.1),
                  vaos={"+v": [("porta", -0.3, 0.8, 1019.45, 1.9)], "-v": [("seteira", 0.0, 0.14, 1021.2, 0.85)],
                        "-u": [("seteira", 0.3, 0.14, 1015.5, 0.9)]})


# ---------------------------------------------------------------- cerca

POL_TORRES = {k: Torre(k).pol() for k in C.TORRES}
AZ_NO = (279.5, 352.0)      # troço noroeste: muro arruinado com o corredor de aço corten por cima
Z_CORTEN = 1016.25


def zbase(x, y):
    return float(cota(x, y)) - 0.7


def cerca():
    """A cerca (LiDAR: perfil por azimute do centro do pátio, faces de dentro e de fora do topo e cota do adarve):
    - a nascente, entre a torre de menagem e a torre sueste, o adarve a 1016,6 m sobe em rampa para 1019,7–1020,1 m;
    - a sul e a poente, entre as três torres, a 1019,5–1019,8 m, com guarda de ferro pelo lado de fora (fotografias);
    - entre a torre pequena e o troço noroeste, a escada de aço desce do adarve (1019,4) para o corredor (1016,25);
    - poterna em arco na base da face de fora, junto à torre sueste, e porta em arco entre a torre sul e a de sudoeste."""
    m = Malha(["muro", "muro_topo", "cantaria_lisa", "sombra", "madeira_porta"])
    pl = PF.planta(passo=1.0)
    tor = shapely.union_all([p.buffer(-0.06) for p in POL_TORRES.values()])
    pedra = pl.difference(setor(*AZ_NO)).difference(tor)
    for p in (pedra.geoms if hasattr(pedra, "geoms") else [pedra]):
        if p.geom_type == "Polygon" and p.area > 0.2:
            prisma_z(m, p.simplify(0.02), zbase, PF.ztopo, 0, 1)
    # portas em arco na base da face de fora (fotografias de nascente e de sul): recuadas, com portas de madeira
    portas = {}
    for nome, az in (("poterna", 137.5), ("porta_sul", 222.0)):
        ro = float(np.interp(az, PF.A, PF.ROUT))
        p0, p1 = np.array(PF.ponto(az - 1.2, ro)), np.array(PF.ponto(az + 1.2, ro))
        t = unit(p1 - p0)
        c = (p0 + p1) / 2
        f = Plano(c, t, n=unit(c - np.array(C.CENTRO_PATIO)))
        za = float(cota(*f.xy(0, 1.0)))
        cont = redondo(0.0, 1.0, za, 1.5, k=10)
        tunel(m, f, cont, 0.02, -0.55, 2)
        painel(m, f, cont, (), 4, d=-0.55)
        anel_aduelas(m, f, [p for p in cont if p[1] >= za + 1.5 - 1e-6], 0.32, 0.04, 2, d0=0.0)
        painel(m, f, cont, (), 3, d=0.021)    # o vão escuro tapa a face do muro (o muro não tem furo)
        portas[nome] = [round(float(c[0]), 2), round(float(c[1]), 2), round(za, 2)]
    m.objeto("cerca", "cerca", GRUPOS["Muralhas"], uv=3.0, vcor=patine_muro)
    # guardas de ferro pelo lado de fora dos adarves altos
    g = Malha(["ferro"])
    for a0, a1 in ((114.0, 151.0), (176.5, 198.0), (219.5, 250.0), (257.0, 262.5)):
        pts = []
        for a in np.arange(a0, a1 + 1e-6, 1.25):
            ro = float(np.interp(a, PF.A, PF.ROUT)) - 0.12
            x, y = PF.ponto(a, ro)
            if any(p.buffer(0.15).contains(shapely.Point(x, y)) for p in POL_TORRES.values()):
                continue
            pts.append((x, y, PF.ztopo(x, y)))
        if len(pts) > 1:
            corrimao(g, pts, alt=1.0, mat=0, passo=1.4)
    g.objeto("guardas", "cerca", GRUPOS["Muralhas"], uv=None)
    # escada de aço do adarve poente para o corredor noroeste (fotografias de sul e de norte)
    e = Malha(["corten", "ferro"])
    ra = float(np.interp(270, PF.A, (PF.RIN + PF.ROUT) / 2))
    p_top, p_pe = np.array(PF.ponto(263.0, ra)), np.array(PF.ponto(279.0, ra))
    n_d = 17
    t_d = unit(p_top - p_pe)
    n_l = np.array([-t_d[1], t_d[0]])
    L_d = float(np.linalg.norm(p_top - p_pe))
    for i in range(n_d):
        a_, b_ = p_pe + t_d * L_d * i / n_d, p_pe + t_d * L_d * (i + 1) / n_d
        zt = Z_CORTEN + (1019.42 - Z_CORTEN) * (i + 1) / n_d
        zf = min(PF.ztopo(*a_), PF.ztopo(*b_)) - 0.15
        e.prisma(ccw([a_ - n_l * 0.5, b_ - n_l * 0.5, b_ + n_l * 0.5, a_ + n_l * 0.5]), min(zf, zt - 0.06), zt, 0)
    nn = unit(np.array([-(p_top - p_pe)[1], (p_top - p_pe)[0]]))
    if np.dot(nn, np.array(C.CENTRO_PATIO) - p_pe) < 0:
        nn = -nn
    corrimao(e, [(*(p_pe + nn * 0.5), Z_CORTEN), (*(p_top + nn * 0.5), 1019.42)], alt=0.95, mat=1, passo=1.2)
    e.objeto("escada_adarve", "cerca", GRUPOS["Muralhas"], uv=1.0)
    MEDIDAS["cerca"] = {"area_m2": round(pedra.area, 1), "adarves": {"nascente": 1016.6, "sueste": 1020.08, "sul": 1019.83, "poente": 1019.48, "corredor_no": Z_CORTEN},
                        "portas": portas}
    return pedra


def corredor_corten():
    """Troço noroeste: o muro arruinado (pedra, c. 0,5 m acima do pátio) e, por cima, o corredor de entrada de aço
    corten com cobertura plana a 1016,25 m (LiDAR: pontos de edifício avermelhados, 1,6–2 m de largura; faixa
    vermelha no ortofotomapa de 2021); porta de chapa ao fundo do passadiço de madeira (fotografia de norte)."""
    m = Malha(["muro", "muro_topo", "corten", "corten_escuro", "cantaria_lisa", "cinza_viz", "sombra"])
    pl = PF.planta(passo=1.0).intersection(setor(*AZ_NO)).difference(POL_TORRES["menagem"].buffer(-0.06))
    pl = max(pl.geoms, key=lambda g: g.area) if hasattr(pl, "geoms") else pl
    base = pl.buffer(0.18, join_style="mitre").intersection(setor(AZ_NO[0] - 0.4, AZ_NO[1]))
    base = base.difference(POL_TORRES["menagem"].buffer(-0.06))
    base = max(base.geoms, key=lambda g: g.area) if hasattr(base, "geoms") else base

    def z_muro(x, y):
        cx, cy = C.CENTRO_PATIO
        az = (math.degrees(math.atan2(x - cx, y - cy)) + 360) % 360
        ri = float(np.interp(az, PF.A, PF.RIN))
        return float(cota(*PF.ponto(az, ri - 1.5))) + 0.55

    prisma_z(m, base.simplify(0.02), zbase, z_muro, 0, 1)
    caixa = pl.buffer(-0.05, join_style="mitre")
    prisma_z(m, caixa.simplify(0.02), lambda x, y: z_muro(x, y) - 0.05, lambda x, y: Z_CORTEN, 2, 2)
    # juntas verticais das chapas (frisos escuros a cada 1,2 m), dos dois lados
    for a in np.arange(AZ_NO[0] + 1.0, AZ_NO[1] - 2.0, 3.6):
        for col, sinal in ((PF.ROUT, 1), (PF.RIN, -1)):
            r = float(np.interp(a, PF.A, col)) - 0.05 * sinal + 0.012 * sinal
            x, y = PF.ponto(a, r)
            if POL_TORRES["menagem"].buffer(0.2).contains(shapely.Point(x, y)):
                continue
            t = unit(np.array(PF.ponto(a + 0.5, r)) - np.array(PF.ponto(a - 0.5, r)))
            f = Plano((x, y), t, n=sinal * unit(np.array((x, y)) - np.array(C.CENTRO_PATIO)))
            zb = z_muro(x, y)
            m.face(orientar([f.p(-0.025, 0, zb), f.p(0.025, 0, zb), f.p(0.025, 0, Z_CORTEN - 0.05), f.p(-0.025, 0, Z_CORTEN - 0.05)], np.array([*f.n, 0])), 3)
    # porta de chapa (fechada) ao fundo do passadiço, com a placa branca à direita
    az = AZ_PORTA
    ro = float(np.interp(az, PF.A, PF.ROUT)) - 0.05
    c = np.array(PF.ponto(az, ro))
    t = unit(np.array(PF.ponto(az + 0.5, ro)) - np.array(PF.ponto(az - 0.5, ro)))
    nf = unit(c - np.array(C.CENTRO_PATIO))
    f = Plano(c, -t if np.cross(np.append(t, 0), np.append(nf, 0))[2] < 0 else t, n=nf)
    z_sol = Z_SOLEIRA
    m.caixa_face(f, -0.62, 0.62, 0.0, 0.035, z_sol, z_sol + 2.35, 3, fundo=False)
    m.caixa_face(f, -0.52, 0.52, 0.035, 0.05, z_sol + 0.1, z_sol + 2.25, 2)
    m.caixa_face(f, 0.62, 0.66, 0.0, 0.06, z_sol, z_sol + 2.35, 6)
    m.caixa_face(f, -0.66, -0.62, 0.0, 0.06, z_sol, z_sol + 2.35, 6)
    s_placa = 1.9 if f.t @ unit(np.array(PF.ponto(az - 5, ro)) - c) > 0 else -1.9
    m.caixa_face(f, s_placa - 0.55, s_placa + 0.55, 0.0, 0.03, z_sol + 1.0, z_sol + 1.75, 5)
    # saída para o pátio no topo nascente do corredor, junto à torre (fotografias de nascente e de drone)
    az_i = 340.0
    ri = float(np.interp(az_i, PF.A, PF.RIN)) + 0.05
    ci = np.array(PF.ponto(az_i, ri))
    ti = unit(np.array(PF.ponto(az_i + 0.5, ri)) - np.array(PF.ponto(az_i - 0.5, ri)))
    fi = Plano(ci, ti, n=-unit(ci - np.array(C.CENTRO_PATIO)))
    zi = float(cota(*fi.xy(0, 1.2)))
    m.face(orientar([fi.p(s_, 0.012, z_) for s_, z_ in reto(0.0, 1.4, zi, zi + 2.4)], np.array([*fi.n, 0])), 6)
    m.caixa_face(fi, -0.75, 0.75, 0.0, 0.04, zi + 2.4, zi + 2.48, 3)
    # degraus de granito do passadiço para a soleira
    zc = float(cota(*f.xy(0, 1.6)))
    for i in range(3):
        d0 = 1.35 - i * 0.42
        m.caixa_face(f, -0.85, 0.85, d0 - 0.42, d0, zc - 0.3, zc + (z_sol - zc) * (i + 1) / 3, 4)
    m.objeto("corredor_corten", "corredor", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)
    MEDIDAS["corredor"] = {"cobertura": Z_CORTEN, "soleira": round(z_sol, 2), "area_m2": round(pl.area, 1),
                           "porta": [round(float(c[0]), 2), round(float(c[1]), 2)]}
    return base, f


# porta do corredor e passadiço (fotografia de norte: o passadiço chega a direito de norte à porta)
AZ_PORTA = 323.0
Z_SOLEIRA = float(cota(*PF.ponto(AZ_PORTA, float(np.interp(AZ_PORTA, PF.A, PF.ROUT)) + 1.5))) + 0.5
PASSADICO = [(-18.3, 15.6), (-16.3, -4.5)]   # faixa de pontos mais claros e lisos do LiDAR entre as ruínas


def passadico(f_porta):
    """Passadiço de tábuas de madeira acinzentada, assente no relvado, com balizadores baixos dos dois lados
    (fotografia de norte); traçado estimado: o ortofotomapa de 2021 ainda não o mostra."""
    m = Malha(["tabuado", "ferro", "corten_escuro"])
    p_fim = np.array(f_porta.xy(0, 1.55))
    p_ini = np.array(PASSADICO[0])
    t = unit(p_fim - p_ini)
    nn = np.array([-t[1], t[0]])
    L = float(np.linalg.norm(p_fim - p_ini))
    k = int(math.ceil(L / 0.6))
    w = 1.5
    for i in range(k):
        a = p_ini + t * L * i / k
        b = p_ini + t * L * (i + 1) / k
        za, zb = float(cota(*a)) + 0.12, float(cota(*b)) + 0.12
        q = [a - nn * w / 2, b - nn * w / 2, b + nn * w / 2, a + nn * w / 2]
        m.prisma(ccw(q), [za - 0.15, zb - 0.15, zb - 0.15, za - 0.15], [za, zb, zb, za], 0)
    for i in range(1, int(L / 2.4)):
        for s in (-1, 1):
            p = p_ini + t * 2.4 * i + nn * s * (w / 2 + 0.35)
            z = float(cota(*p))
            m.cilindro(p, 0.045, z - 0.1, z + 0.30, 1, k=8)
    m.objeto("passadico", "passadico", GRUPOS["Muralhas"], uv=1.0)
    MEDIDAS["passadico"] = {"comprimento": round(L, 1), "largura": w}
    return shapely.LineString([p_ini, p_fim]).buffer(w / 2 + 0.5, cap_style="flat")


def cisterna():
    """Cisterna quadrada no pátio (ortofotomapa e LiDAR: c. 5 × 5 m), com bocal de pedra baixo e o fundo escuro."""
    m = Malha(["muro", "muro_topo", "sombra"])
    q = shapely.box(-12.2, -30.4, -7.2, -25.4)
    zc = float(np.median([cota(x, y) for x, y in np.array(q.buffer(1.0).exterior.coords)]))
    anel = q.difference(q.buffer(-0.45, join_style="mitre"))
    prisma_z(m, anel, lambda x, y: zc - 2.2, lambda x, y: zc + 0.55, 0, 1)
    inn = q.buffer(-0.45, join_style="mitre")
    b = np.array(inn.exterior.coords)[:-1]
    m.face(orientar([(x, y, zc - 1.6) for x, y in b], np.array([0, 0, 1.0])), 2)
    m.objeto("cisterna", "cisterna", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)
    MEDIDAS["cisterna"] = {"lado": 5.0, "bocal": round(zc + 0.55, 2)}
    return q


RUINAS = [((-21.3, 11.4), 2.65, 1.0, 90.0), ((-12.4, 15.2), 3.0, 0.9, 110.0)]


def ruinas():
    """Os dois muros baixos em ferradura a norte da torre de menagem (LiDAR: c. 1 m acima do chão, abertos para
    nascente), restos de construções da vila velha; altura e abertura do LiDAR, a pedra das fotografias."""
    m = Malha(["muro", "muro_topo"])
    out = []
    for c, r, esp, az_ab in RUINAS:
        anel = shapely.Point(c).buffer(r, 24).difference(shapely.Point(c).buffer(r - esp, 24))
        corte = shapely.Polygon([c, (c[0] + 9 * math.sin(math.radians(az_ab - 45)), c[1] + 9 * math.cos(math.radians(az_ab - 45))),
                                 (c[0] + 9 * math.sin(math.radians(az_ab + 45)), c[1] + 9 * math.cos(math.radians(az_ab + 45)))])
        g = anel.difference(corte)
        zt = float(np.max([cota(x, y) for x, y in np.array(shapely.Point(c).buffer(r).exterior.coords)])) + 0.75
        for p in (g.geoms if hasattr(g, "geoms") else [g]):
            prisma_z(m, p, zbase, lambda x, y, zt=zt: zt, 0, 1)
        out.append(g)
    m.objeto("ruinas", "ruinas", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)
    return shapely.union_all(out)


# bordo de fora do caminho da barbacã: onde o MDT cai do patamar (cortes de 10 em 10 m no MDT de 50 cm e no
# ortofotomapa, em que o caminho é a faixa clara de c. 2 m por dentro do bordo)
BARBACA = [(10.4, 15.6), (12.3, 12.0), (14.6, 8.0), (16.0, 5.0), (15.6, -5.0), (16.6, -15.0), (17.5, -25.0), (18.6, -35.0),
           (20.6, -38.2), (22.5, -40.0), (22.5, -46.0), (20.5, -49.6), (15.0, -55.0), (10.0, -58.8), (5.0, -59.5), (0.0, -60.0),
           (-5.0, -61.4)]


def barbaca():
    """Barbacã: o muro de suporte do patamar do castelo a nascente e a sul (LiDAR: o chão cai 2,5 a 4 m no bordo),
    com o caminho lajeado claro por cima, por dentro do bordo (ortofotomapa), e a vedação de madeira pelo lado de
    fora (fotografia de drone de nascente)."""
    linha = shapely.LineString(BARBACA)
    # a linha segue de norte para sul com o patamar à direita (buffer negativo de um só lado)
    muro = linha.buffer(-0.6, single_sided=True, join_style="mitre")
    caminho = linha.buffer(-2.6, single_sided=True, join_style="mitre").difference(muro)
    m = Malha(["muro", "muro_topo"])

    def _lado(x, y):
        """Ponto do bordo mais próximo e a normal para dentro (para o patamar)."""
        t0 = linha.project(shapely.Point(x, y))
        p0, p1 = linha.interpolate(max(t0 - 0.3, 0)), linha.interpolate(min(t0 + 0.3, linha.length))
        t = unit(np.array([p1.x - p0.x, p1.y - p0.y]))
        p = linha.interpolate(t0)
        return np.array([p.x, p.y]), np.array([t[1], -t[0]])

    def z_patamar(x, y):
        p, n = _lado(x, y)
        return float(cota(*(p + n * 1.6))) + 0.12

    def z_pe(x, y):
        p, n = _lado(x, y)
        return min(float(cota(*(p - n * 1.5))), float(cota(x, y))) - 0.6

    prisma_z(m, muro.simplify(0.02), z_pe, z_patamar, 0, 1)
    m.objeto("barbaca", "barbaca", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)
    # vedação de madeira no bordo de fora (prumos de 2 em 2 m e duas travessas)
    v = Malha(["madeira"])
    L = linha.length
    ps = [linha.interpolate(t) for t in np.arange(0, L + 1e-6, L / math.ceil(L / 2.0))]
    zs = []
    for p in ps:
        q = np.array([p.x, p.y])
        z = z_patamar(*q)
        zs.append(z)
        v.prisma(ccw([q + np.array(d) * 0.05 for d in ((-1, -1), (1, -1), (1, 1), (-1, 1))]), z - 0.1, z + 1.0, 0)
    for (a, za), (c, zc_) in zip(zip(ps[:-1], zs[:-1]), zip(ps[1:], zs[1:])):
        for h in (0.45, 0.9):
            v.barra((a.x, a.y, za + h), (c.x, c.y, zc_ + h), 0.035, 0, k=4)
    v.objeto("vedacao", "barbaca", GRUPOS["Muralhas"], uv=None)
    MEDIDAS["barbaca"] = {"comprimento": round(L, 1), "altura_max": round(max(z_patamar(*np.array(c)) - float(cota(*c)) for c in BARBACA), 1)}
    return muro, caminho

# ---------------------------------------------------------------- terreno, envolvente e árvores

def _patio():
    az = np.arange(PF.AZ0, PF.AZ1 + 1e-6, 1.0)
    p = shapely.Polygon([PF.ponto(a, float(np.interp(min(max(a, 37.5), 345), PF.A, PF.RIN))) for a in az]).buffer(0)
    return p.difference(POL_TORRES["menagem"])


def _srgb(c):
    return (np.asarray(c, float) / 255.0) ** 2.2


def construir_terreno(mon, cist, caminho):
    """Relevo do MDT de 50 cm, com a cor do chão por vértice: relva (verde das fotografias de drone), terra batida
    clara no pátio (ortofotomapa e fotografias), caminhos lajeados claros onde o ortofotomapa é claro e cinzento."""
    passo = 0.5
    xs = np.arange(X0, X1 + 1e-6, passo)
    ys = np.arange(Y0, Y1 + 1e-6, passo)
    X, Y = np.meshgrid(xs, ys)
    Z = cota(X, Y)
    pts_all = shapely.points(X.ravel(), Y.ravel())
    mur = shapely.contains(mon.buffer(0.2), pts_all).reshape(X.shape)
    # junto às faces o MDT sobe para o topo dos muros (interpolação por baixo deles): baixa-se até 1,5 m das faces
    banda = ndi.binary_dilation(mur, iterations=3) & ~mur
    zl = np.where(mur, 1e4, Z)
    zmin = ndi.minimum_filter(zl, size=5)
    Z = np.where(banda, np.minimum(Z, np.maximum(zmin, Z - 1.5)), Z)
    ny, nx = X.shape
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    idx = np.arange(ny * nx).reshape(ny, nx)
    Fq = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    # o buraco da cisterna
    cx = V[Fq].mean(1)
    fora_cist = ~shapely.contains(cist.buffer(-0.3), shapely.points(cx[:, 0], cx[:, 1]))
    Fq = Fq[fora_cist]
    cx = cx[fora_cist]
    ob = malha_rapida("Terreno", V, Fq, "chao", GRUPOS["Terreno"], "terreno")
    ob.data.materials.append(MATS["asfalto"])
    Xv, Yv = V[:, 0].reshape(ny, nx), V[:, 1].reshape(ny, nx)
    ci = np.clip(((Xv - OX0) / OP).astype(int), 0, ORTO.shape[1] - 1)
    ri = np.clip(((OY1 - Yv) / OP).astype(int), 0, ORTO.shape[0] - 1)
    o = ndi.uniform_filter(ORTO, size=(5, 5, 1))[ri, ci]
    lum = o.mean(2)
    sat = o.max(2) - o.min(2)
    pts_v = shapely.points(Xv.ravel(), Yv.ravel())
    patio = shapely.contains(_patio().buffer(0.6), pts_v).reshape(ny, nx)
    cls_v = CLS[np.clip(np.round((Yv - LY0) / LP).astype(int), 0, CLS.shape[0] - 1), np.clip(np.round((Xv - LX0) / LP).astype(int), 0, CLS.shape[1] - 1)]
    # caminhos lajeados: claros e pouco saturados no ortofotomapa (de 2021), fora do castelo; a classe 1 do LiDAR
    # (não solo) com o chão liso marca o mesmo caminho por baixo das sombras
    gy, gx = np.gradient(Z, passo)
    decl = np.degrees(np.arctan(np.hypot(gx, gy)))
    claro = (lum > 0.55) & (sat < 0.12) & ~patio & ~mur & (decl < 14)
    claro = ndi.binary_opening(ndi.binary_closing(claro, iterations=2), iterations=1)
    lab, k = ndi.label(claro)
    if k:
        tam = ndi.sum(claro, lab, range(1, k + 1)) * passo * passo
        claro = np.isin(lab, 1 + np.flatnonzero(tam >= 6.0))
    claro |= shapely.contains(caminho, pts_v).reshape(ny, nx)
    g = fbm(Xv * 0.9, Yv * 0.9, 6.0, 71)
    g2 = fbm(Xv * 3.0 + 7, Yv * 3.0, 2.0, 73)
    # relva das fotografias de drone (sRGB c. 95, 110, 42 ao sol; 78, 79, 34 à sombra), com manchas mais secas
    relva = _srgb((92, 108, 40))[None, None, :] * (0.80 + 0.40 * g)[..., None]
    seca = np.clip((fbm(Xv * 0.4 + 3, Yv * 0.4, 9.0, 75) - 0.6) / 0.2, 0, 1) * 0.5
    relva = relva * (1 - seca[..., None]) + _srgb((120, 112, 62))[None, None, :] * seca[..., None]
    # terra batida do pátio: castanho claro acinzentado (ortofotomapa c. 150, 135, 110), com erva rala nas bordas
    terra = _srgb((150, 134, 108))[None, None, :] * (0.85 + 0.3 * g2)[..., None]
    lajes = _srgb((196, 192, 182))[None, None, :] * (0.92 + 0.12 * g2)[..., None]
    w_p = np.clip((ndi.gaussian_filter(patio.astype(float), 1.0) - 0.2) / 0.6, 0, 1)
    rala = np.clip((g - 0.62) / 0.15, 0, 1) * 0.6 * w_p
    cor = relva * (1 - w_p[..., None]) + terra * w_p[..., None]
    cor = cor * (1 - rala[..., None]) + relva * rala[..., None]
    w_l = np.clip((ndi.gaussian_filter(claro.astype(float), 0.6) - 0.25) / 0.5, 0, 1)
    cor = cor * (1 - w_l[..., None]) + lajes * w_l[..., None]
    cor = np.clip(cor.reshape(-1, 3) / 0.79, 0, 1)
    pts = shapely.points(cx[:, 0], cx[:, 1])
    mi_ = np.zeros(len(Fq), np.int32)
    asf = shapely.Polygon()
    for f in F["caminhos"]:
        if f["geometria"]["type"] == "LineString" and f["classe"] in ("residential", "tertiary", "unclassified", "secondary", "service"):
            asf = asf.union(shapely.LineString(f["geometria"]["coordinates"]).buffer(2.8, cap_style="flat"))
    if not asf.is_empty:
        mi_[shapely.contains(asf, pts)] = 1
    ob.data.polygons.foreach_set("material_index", mi_)
    cc = cor[Fq]
    cc[mi_ != 0] = 1.0
    at = ob.data.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
    at.data.foreach_set("color", np.concatenate([cc.reshape(-1, 3), np.ones((cc.size // 3, 1))], 1).astype(np.float32).ravel())
    ob.data.color_attributes.active_color = at
    uv_mundo(ob, 2.0)
    m = Malha(["base"])
    zb = float(Z.min()) - 1.5
    borda = [(X[0, i], Y[0, i], Z[0, i]) for i in range(nx)] + [(X[i, -1], Y[i, -1], Z[i, -1]) for i in range(1, ny)] + \
            [(X[-1, i], Y[-1, i], Z[-1, i]) for i in range(nx - 2, -1, -1)] + [(X[i, 0], Y[i, 0], Z[i, 0]) for i in range(ny - 2, 0, -1)]
    for a, b in zip(borda, borda[1:] + borda[:1]):
        m.face([(a[0], a[1], zb), (b[0], b[1], zb), b, a], 0)
    m.objeto("base", None, GRUPOS["Terreno"], uv=None)
    MEDIDAS["terreno"] = {"m2": {"patio": round(float((w_p > 0.5).sum()) * passo * passo), "lajeado": round(float((w_l > 0.5).sum()) * passo * passo)},
                          "patio_cota": [round(float(np.percentile(Z[patio], 5)), 2), round(float(np.percentile(Z[patio], 95)), 2)]}


def casas_lidar(tirar, osm):
    """Volumes que o OSM não tem: manchas da classe 6 do LiDAR (edifícios) com mais de 2,5 m acima do chão."""
    from rasterio import features
    from rasterio.transform import Affine
    h = MDS - MDT
    ed = (CLS == 6) & (h > 2.5)
    ed = ndi.binary_opening(ed, iterations=1)
    lab, k = ndi.label(ed)
    tr = Affine(LP, 0, LX0 - LP / 2, 0, LP, LY0 - LP / 2)
    out = []
    excl = shapely.union_all([tirar.buffer(1.5), osm.buffer(0.6)])
    caixa = shapely.box(X0 + 0.3, Y0 + 0.3, X1 - 0.3, Y1 - 0.3)
    for geo, val in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=tr):
        p = shapely.geometry.shape(geo).buffer(0).intersection(caixa).difference(excl)
        for q in (p.geoms if p.geom_type in ("MultiPolygon", "GeometryCollection") else [p]):
            if q.geom_type != "Polygon" or q.area < 12:
                continue
            q = q.buffer(0.4, join_style="mitre").buffer(-0.4, join_style="mitre").simplify(0.45)
            if q.geom_type == "Polygon" and q.area >= 12:
                out.append(q)
    return out


def _casa(m, p):
    interior = p.buffer(-0.6)
    if interior.is_empty:
        return False
    minx, miny, maxx, maxy = interior.bounds
    gx, gy = np.meshgrid(np.arange(minx, maxx, 0.5), np.arange(miny, maxy, 0.5))
    dentro = shapely.contains_xy(interior, gx.ravel(), gy.ravel())
    if dentro.sum() < 4:
        return False
    zs = amostra(MDS, LX0, LY0, LP, gx.ravel()[dentro], gy.ravel()[dentro])
    zt = amostra(MDT, LX0, LY0, LP, gx.ravel()[dentro], gy.ravel()[dentro])
    chao = float(np.percentile(zt, 10))
    cume = float(np.percentile(zs, 92))
    cornija = min(max(float(np.percentile(zs, 25)), chao + 2.6), cume - 0.4)
    if cume - chao < 2.5:
        return False
    base = np.array(shapely.geometry.polygon.orient(p, 1).exterior.coords)[:-1]
    rec = (cume - cornija) / math.tan(math.radians(28))
    top = p.buffer(-rec, join_style="mitre")
    if top.is_empty or top.geom_type != "Polygon" or rec < 0.3 or p.convex_hull.area > 1.25 * p.area or len(base) > 14:
        zc = cornija + 0.5 * (cume - cornija)
        m.prisma(list(base), chao - 0.8, zc, 0)
        return True
    m.prisma(list(base), chao - 0.8, cornija, 0, topo=False)
    tb = np.array(shapely.geometry.polygon.orient(top, 1).exterior.coords)[:-1]
    _, j = cKDTree(tb).query(base)
    for i in range(len(base)):
        a, b = base[i], base[(i + 1) % len(base)]
        ja, jb = j[i], j[(i + 1) % len(base)]
        seq = [ja]
        k = ja
        while k != jb and len(seq) < len(tb) + 1:
            k = (k + 1) % len(tb)
            seq.append(k)
        if len(seq) > len(tb):
            seq = [ja, jb]
        m.face([(*a, cornija), (*b, cornija)] + [(*tb[s], cume) for s in seq[::-1]], 0)
    m.face([(*q, cume) for q in tb], 0)
    return True


def construir_envolvente(tirar):
    """Os edifícios da vila à volta do castelo, em volumes simples de um só cinzento claro."""
    caixa = shapely.box(X0 + 0.3, Y0 + 0.3, X1 - 0.3, Y1 - 0.3)
    m = Malha(["cinza_viz"])
    plantas = []
    for f in F["edificios"]:
        g = shapely.geometry.shape(f["geometria"])
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
        if g.geom_type != "Polygon" or g.area < 6:
            continue
        g = g.buffer(0)
        if g.intersection(tirar.buffer(1.0)).area > 0.3 * g.area:
            continue  # as torres também estão no OSM
        g = g.intersection(caixa).difference(tirar.buffer(1.0))
        for p in (g.geoms if g.geom_type in ("MultiPolygon", "GeometryCollection") else [g]):
            if p.geom_type == "Polygon" and p.area >= 6:
                plantas.append(p.simplify(0.3))
    osm = shapely.union_all(plantas) if plantas else shapely.Polygon()
    n_osm = len(plantas)
    plantas += casas_lidar(tirar, osm)
    n = sum(_casa(m, p) for p in plantas)
    print("casas", n, "(OSM", n_osm, ")")
    MEDIDAS["casas"] = {"total": n, "osm": n_osm}
    if n:
        m.objeto("envolvente", "envolvente", GRUPOS["Envolvente"], uv=2.0, recalc=False)


def construir_arvores(tirar):
    """Árvores: máximos locais da vegetação (classes 3 a 5) fora do castelo; nenhuma copa a atravessar os muros."""
    h = MDS - MDT
    veg = (CLS >= 3) & (CLS <= 5) & (h > 2.0)
    veg = ndi.binary_opening(veg, iterations=1)
    hs = ndi.gaussian_filter(np.where(veg, h, 0.0), 1.2)
    topo = (hs == ndi.maximum_filter(hs, size=11)) & veg & (hs > 2.5)
    lab, k = ndi.label(topo)
    V, Fc, Cc = [], [], []
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    v0 = np.array([x.co[:] for x in bm.verts])
    f0 = [[x.index for x in fa.verts] for fa in bm.faces]
    bm.free()
    mon_l = tirar.buffer(0.8)
    n = 0
    for (r, c) in ndi.center_of_mass(topo, lab, range(1, k + 1)):
        r, c = int(round(r)), int(round(c))
        x, y = LX0 + c * LP, LY0 + r * LP
        if not (X0 + 2 < x < X1 - 2 and Y0 + 2 < y < Y1 - 2):
            continue
        pt = shapely.Point(x, y)
        if mon_l.contains(pt):
            continue
        zt, zc = float(cota(x, y)), MDS[r, c]
        alt = zc - zt
        if alt < 2.5:
            continue
        rad = float(np.clip(np.sqrt(veg[max(r - 10, 0):r + 11, max(c - 10, 0):c + 11].sum() * 0.25 / math.pi), 1.3, 5.5))
        rad = min(rad, tirar.distance(pt) - 0.3)
        if rad < 1.0:
            continue
        for (dx, dy, dz, s) in ((0, 0, -0.35, 1.0), (0.4, 0.2, -0.5, 0.7), (-0.35, -0.3, -0.55, 0.7), (0.1, -0.4, -0.3, 0.6)):
            b = len(V)
            pts = (v0 + rng.normal(0, 0.06, v0.shape)) * np.array([rad * s, rad * s, alt * 0.30 * s]) + np.array([x + dx * rad * 0.6, y + dy * rad * 0.6, zc + dz * alt * 0.45])
            V.extend(map(tuple, pts))
            Fc.extend([[b + i for i in fa] for fa in f0])
            tt = np.clip((pts[:, 2] - zt) / alt, 0, 1)
            Cc.extend(map(tuple, np.array([0.20, 0.31, 0.13])[None, :] * (0.55 + 0.5 * tt)[:, None]))
        b = len(V)
        for j in range(6):
            a = 2 * math.pi * j / 6
            V.append((x + 0.22 * math.cos(a), y + 0.22 * math.sin(a), zt - 0.2))
            V.append((x + 0.16 * math.cos(a), y + 0.16 * math.sin(a), zc - alt * 0.45))
        for j in range(6):
            jj = (j + 1) % 6
            Fc.append([b + 2 * j, b + 2 * jj, b + 2 * jj + 1, b + 2 * j + 1])
        Cc.extend([(0.36, 0.30, 0.24)] * 12)
        n += 1
    print("árvores", n)
    MEDIDAS["arvores"] = n
    if not V:
        return
    me = bpy.data.meshes.new("arvores")
    me.from_pydata(V, [], Fc)
    me.update()
    me.materials.append(MATS["arvore"])
    at = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    at.data.foreach_set("color", np.concatenate([np.array(Cc), np.ones((len(Cc), 1))], 1).astype(np.float32).ravel())
    me.shade_smooth()
    ob = bpy.data.objects.new("arvores", me)
    bpy.context.scene.collection.objects.link(ob)
    ob["elemento"] = "vegetacao"
    ob.parent = GRUPOS["Vegetacao"]


# ---------------------------------------------------------------- construção

print("torre…")
torre_menagem()
escada_aco()
print("torres do sul…")
torres_sul()
print("cerca…")
POL_CERCA = cerca()
POL_NO, F_PORTA = corredor_corten()
POL_PASS = passadico(F_PORTA)
POL_CIST = cisterna()
POL_RUINAS = ruinas()
POL_BARB, POL_CAMINHO = barbaca()
POL_MON = shapely.union_all([POL_CERCA, POL_NO, POL_BARB, POL_RUINAS] + list(POL_TORRES.values()))
print("terreno…")
construir_terreno(shapely.union_all([POL_CERCA, POL_NO, POL_BARB] + list(POL_TORRES.values())), POL_CIST, POL_CAMINHO)
TIRAR = shapely.union_all([POL_MON, POL_PASS, _patio(), TM.pol(2.0)])
print("envolvente…")
construir_envolvente(TIRAR)
construir_arvores(TIRAR)


def _anc(x, y, z):
    return [round(float(x), 1), round(float(z), 1), round(float(-y), 1)]


def _olhar(x, y, up=0.35):
    d = unit((x, y))
    return [round(float(d[0]), 2), up, round(float(-d[1]), 2)]


_fs = TM.faces()
_TSE, _TS, _TSO = Torre("se"), Torre("s"), Torre("so")
MEDIDAS["ancoras"] = {
    "torre": _anc(0, 0, Z_MP_TM + 1.5),
    "coroamento": _anc(*TM.W(TM.mu + 0.6, -TM.mv - 0.6), Z_MP_TM - 0.3),
    "balcoes": _anc(*_fs["-v"][0].xy(0.0, 1.2), Z_PISO_BAL + 1.2),
    "porta_torre": _anc(*_fs["-v"][0].xy(-0.3, 2.6), Z_PLAT + 2.6),
    "escada_torre": _anc(*_fs["-v"][0].xy(4.0, 1.5), Z_PLAT - 0.5),
    "torre_se": _anc(*_TSE.c, 1031.2),
    "torre_s": _anc(*_TS.c, 1027.6),
    "torre_so": _anc(*_TSO.c, 1025.0),
    "cerca": _anc(*PF.ponto(75.0, 16.0), 1018.2),
    "corredor": _anc(*PF.ponto(AZ_PORTA, 19.0), Z_CORTEN + 1.5),
    "passadico": _anc(*PASSADICO[0], float(cota(*PASSADICO[0])) + 1.4),
    "cisterna": _anc(-9.7, -27.9, MEDIDAS["cisterna"]["bocal"] + 1.0),
    "ruinas": _anc(*RUINAS[0][0], float(cota(*RUINAS[0][0])) + 1.8),
    "barbaca": _anc(17.0, -10.0, 1015.5),
    "patio": _anc(-4.0, -22.5, 1014.5),
}
MEDIDAS["olhar"] = {
    "coroamento": _olhar(*TM.W(1, -1), 0.25),
    "balcoes": _olhar(*(-TM.ev), 0.2),
    "porta_torre": _olhar(*(-TM.ev), 0.15),
    "escada_torre": _olhar(*(-TM.ev + 0.5 * TM.eu), 0.2),
    "torre_se": _olhar(0.3, -1, 0.3),
    "torre_s": _olhar(-0.5, -1, 0.3),
    "torre_so": _olhar(-1, -0.3, 0.3),
    "cerca": _olhar(1, 0.1, 0.3),
    "corredor": _olhar(-0.2, 1, 0.2),
    "passadico": _olhar(-0.2, 1, 0.3),
    "barbaca": _olhar(1, 0.2, 0.4),
    "patio": _olhar(0.1, -1, 0.8),
}
(AQUI / "dados" / "medidas.json").write_text(json.dumps(MEDIDAS, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in MEDIDAS.items() if k not in ("ancoras", "olhar")}, ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
(AQUI / "saida").mkdir(exist_ok=True)
glb = AQUI / "saida" / "montalegre-bruto.glb"
props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
opcoes = dict(filepath=str(glb), export_format="GLB", export_extras=True, export_yup=True, export_apply=True,
              export_image_format="JPEG", export_jpeg_quality=85)
if "export_vertex_color" in props:
    opcoes["export_vertex_color"] = "ACTIVE"
if "export_active_vertex_color_when_no_material" in props:
    opcoes["export_active_vertex_color_when_no_material"] = True
if "--sem-glb" not in sys.argv:
    bpy.ops.export_scene.gltf(**{k: v for k, v in opcoes.items() if k in props})
    print("GLB:", glb, round(glb.stat().st_size / 1e6, 2), "MB")
    import subprocess  # noqa: E402
    final = SAIDA / "montalegre.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "montalegre.blend"))

# ---------------------------------------------------------------- capa e vistas de verificação

if "--vistas" in sys.argv:
    from mathutils import Vector
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--amostras=")), 32))
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "AgX"
    w = bpy.data.worlds.new("mundo")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.62, 0.72, 0.86, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
    sol = bpy.data.lights.new("sol", "SUN")
    sol.energy = 3.4
    sol.angle = math.radians(2)
    so = bpy.data.objects.new("sol", sol)
    sc.collection.objects.link(so)
    AZ_SOL = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--sol=")), 160))
    d = Vector((math.sin(math.radians(AZ_SOL)), math.cos(math.radians(AZ_SOL)), 0)) * math.cos(math.radians(50))
    d.z = math.sin(math.radians(50))
    so.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    cam = bpy.data.cameras.new("cam")
    co = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    pasta = AQUI / "vistas"
    pasta.mkdir(exist_ok=True)
    SO = next((a_.split("=")[1].split(",") for a_ in sys.argv if a_.startswith("--so=")), None)
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 86

    def vista(nome, cam_xy, cam_z, alvo_xy, alvo_z, lente=35, res=(1000, 1000), destino=None, perto=0.1):
        if SO and nome not in SO:
            return
        co.location = Vector((cam_xy[0], cam_xy[1], cam_z))
        alvo = Vector((alvo_xy[0], alvo_xy[1], alvo_z))
        co.rotation_euler = (alvo - co.location).to_track_quat("-Z", "Y").to_euler()
        cam.type = "PERSP"
        cam.lens = lente
        cam.clip_start = perto
        cam.clip_end = 3000
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.render.filepath = str(destino or (pasta / f"{nome}.jpg"))
        bpy.ops.render.render(write_still=True)
        print("vista", nome)

    def chao(x, y, h=1.6):
        return float(cota(x, y)) + h

    # capa: de sudoeste, do alto, com as três torres do sul à frente e a torre de menagem atrás
    vista("capa", (-62, -88), 1058, (-4, -20), 1018, lente=34, res=(1280, 800), destino=SAIDA / "capa.jpg")
    if "--capa" not in sys.argv:
        # 62d1: drone a sul, a olhar para norte (as três torres do sul à frente)
        vista("62d1_sul", (-40, -84), 1040, (-6, -22), 1020, lente=36, res=(1041, 968))
        # 3b33: do relvado a poente, de baixo
        vista("3b33_poente", (-56, -16), chao(-56, -16), (-8, -20), 1019, lente=30, res=(1280, 960), perto=0.3)
        # 05d9: de norte, do caminho, ao longo do passadiço
        vista("05d9_norte", (-18.6, 18.5), chao(-18.6, 18.5), (-14.0, -8), 1019.0, lente=20, res=(900, 600), perto=0.3)
        # 0e257: de baixo da escada de aço, para a face sul da torre
        vista("0e257_escada", (8.5, -14.0), chao(8.5, -14.0), (-0.5, -4.0), 1031.0, lente=18, res=(651, 868), perto=0.2)
        # 4f30: drone de nascente-sueste
        vista("4f30_nascente", (62, -62), 1050, (-3, -22), 1016, lente=30, res=(1000, 750))
        # f05: do alto da torre sueste para noroeste
        vista("f05_torre_se", (2.0, -46.0), 1031.3, (-14, -12), 1016, lente=22, res=(1000, 750), perto=0.3)
        # coroamento da torre de menagem de perto, de sudoeste
        vista("coroamento", tuple(TM.W(-10.0, -12.0)), 1042.0, (0.0, 0.0), 1036.0, lente=30, res=(1000, 750), perto=0.3)
        # planta (ortográfica, norte para cima), para comparar com o ortofotomapa
        co.location = Vector(((X0 + X1) / 2, (Y0 + Y1) / 2, 1300))
        co.rotation_euler = (0, 0, 0)
        cam.type = "ORTHO"
        cam.ortho_scale = X1 - X0
        sc.render.resolution_x, sc.render.resolution_y = 1500, int(1500 * (Y1 - Y0) / (X1 - X0))
        sc.render.filepath = str(pasta / "topo.jpg")
        if not SO or "topo" in SO:
            bpy.ops.render.render(write_still=True)
