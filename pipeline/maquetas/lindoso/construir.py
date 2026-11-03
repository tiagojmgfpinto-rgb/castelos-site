"""Constrói a maqueta 3D do Castelo de Lindoso e dos espigueiros (estado atual) em Blender e exporta-a para
public/maquetas/lindoso/lindoso.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow rasterio
    python pipeline/maquetas/lindoso/construir.py            # GLB
    python pipeline/maquetas/lindoso/construir.py --vistas   # GLB + capa + vistas de verificação

Dados (em dados/): lidar.npz, superficie.npz e torre.npz (lidar.py: LiDAR 2025 da DGT, folha 194545),
muralhas.json (tracar.py: os muros traçados no LiDAR), orto.jpg (ortofoto.py: ortofotomapa 2021 da DGT),
fontes.json (recolher.py: plantas, ruas e espigueiros do OSM via Overture).

O que se modela, e de onde vem:
- a torre de menagem, rodada 64,25° (as faces olham a ENE, NNO, OSO e SSE), com a planta e as cotas do LiDAR e o
  coroamento das fotografias: sete merlões de remate piramidal por face, contando os dos cunhais, telhado de
  quatro águas de telha por dentro do parapeito, a janela da face SSE, a porta e as seteiras;
- os muros do castelo como o LiDAR os mede (tracar.py): a cerca medieval, com c. 6 m acima do pátio, a muralha
  abaluartada do século XVII à volta, com os baluartes e as lices, e o revelim a norte; faces a prumo do chão de
  fora ao topo medido, em cantaria na cerca e em alvenaria na muralha abaluartada;
- a torre da porta, a sul, com a passagem em arco de volta perfeita, a Porta Norte na ponta poente da cortina
  norte, o postigo ao pé da torre, as guaritas e os edifícios do museu no pátio;
- os espigueiros (c. 60, do LiDAR e do OSM), cada um com os pés e as mesas de granito, o corpo de esteios e
  grades, as empenas, o telhado de lajes e a cruz ou o pináculo nos topos;
- as casas à volta em volumes simples de um só cinzento, as ruas, os afloramentos de granito e as árvores do
  LiDAR (nenhuma sobre o monumento).
Coordenadas da cena: x para este, y para norte (PT-TM06 menos a origem), z = altitude, em metros (1:1).
O glTF converte para y para cima (-z é o norte no visor).
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
from comum import (ALA_OESTE, ALA_SUL, CORREDOR, LAT0, LON0, MEIA_TORRE, PASSAGEM_SUL, PORTA_NORTE,  # noqa: E402
                   ROT_TORRE, TORRE_DA_PORTA, ZONA_CASTELO, _bloco)

_cfg = RAIZ / "src" / "data" / "maquetas" / "lindoso.json"
CFG = json.loads(_cfg.read_text()) if _cfg.exists() else {"origem": {"lat": LAT0, "lon": LON0}, "variante": "estado-atual", "versao": "0.1"}
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
MUR = json.loads((AQUI / "dados" / "muralhas.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "lindoso"
LID = np.load(AQUI / "dados" / "lidar.npz")
LP, LX0, LY0 = float(LID["passo"]), float(LID["x0"]), float(LID["y0"])
MDT = LID["mdt"] / 100.0 + float(LID["base"])
MDS = LID["mds"] / 100.0 + float(LID["base"])
CLS = LID["classe"]
SUPF = np.load(AQUI / "dados" / "superficie.npz")
SP, SX0, SY0 = float(SUPF["passo"]), float(SUPF["x0"]), float(SUPF["y0"])
SUP = SUPF["z"] / 100.0 + float(SUPF["base"])
ORTO = np.asarray(Image.open(AQUI / "dados" / "orto.jpg").convert("RGB"), float) / 255.0  # 25 cm, norte em cima
OX0, OY1, OP = -130.0, 70.0, 0.25

X0, X1, Y0, Y1 = -124.0, 74.0, -156.0, 64.0  # a maqueta: 198 × 220 m, do castelo aos espigueiros
rng = np.random.default_rng(1258)

# ---------------------------------------------------------------- referencial da torre (medido no LiDAR)
# As faces da torre olham a 64,25° (ENE), 334,25° (NNO), 244,25° (OSO) e 154,25° (SSE): u = azimute 64,25°,
# v = azimute 334,25°. O centro da torre é a origem da cena.
_r = math.radians(ROT_TORRE)
EU = np.array([math.sin(_r), math.cos(_r)])
EV = np.array([-math.cos(_r), math.sin(_r)])
A = MEIA_TORRE


def W(u, v):
    """Ponto do referencial da torre (u, v) para a cena (x este, y norte)."""
    return EU * u + EV * v


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
        # granito de Melgaço nas fotografias: cinzento quente, dourado ao sol, com silhares mais claros e outros
        # tingidos de ocre (óxido de ferro)
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


def face_torre(nuv, off):
    nuv = np.array(nuv, float)
    tuv = np.array([nuv[1], -nuv[0]])
    return Plano(W(*(nuv * off)), W(*tuv), W(*nuv), meia=A)




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


def textura_tabuas(n=512, metros=2.0, semente=21):
    """Grades de madeira dos espigueiros: ripas verticais de c. 6 cm com frestas, madeira velha acinzentada."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    larg = 0.085 * m_px
    k = (xx // larg).astype(int)
    f = (xx % larg) / larg
    tom = r.uniform(0.82, 1.15, k.max() + 1)[k]
    veio = fbm(xx * 0.3, yy * 4.0, 0.5 * m_px, semente + 2)
    base = np.array([0.50, 0.47, 0.42])
    tex = base * (tom * (0.85 + 0.3 * veio))[..., None]
    fresta = f > 0.72
    tex = np.where(fresta[..., None], np.array([0.06, 0.05, 0.045]), tex)
    liq = fbm(xx, yy, 0.2 * m_px, semente + 5)
    tex = np.where((liq > 0.7)[..., None], tex * 0.7 + 0.22, tex)
    return np.clip(tex, 0, 1), (~fresta).astype(float) * (0.8 + 0.2 * veio)


def textura_lajes(n=512, metros=2.0, semente=31):
    """Telhado de lajes de granito dos espigueiros: lajes em fiadas que descem a água, cinzentas, com musgo e
    líquenes (fotografias)."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    alt_f = 0.42 * m_px
    fi = (yy // alt_f).astype(int)
    desl = (fi % 2) * 0.3 * m_px
    larg = 0.6 * m_px
    ci = ((xx + desl) // larg).astype(int)
    cel = fi * 1000 + ci
    tom = r.uniform(0.80, 1.12, cel.max() + 1)[cel]
    fy = (yy % alt_f) / alt_f
    fx = ((xx + desl) % larg) / larg
    base = np.array([0.50, 0.50, 0.47])
    tex = base * tom[..., None] * (0.86 + 0.28 * fbm(xx, yy, 0.4 * m_px, semente + 1))[..., None]
    borda = (fy > 0.9) | (fx < 0.03)
    tex = np.where(borda[..., None], tex * 0.45, tex)
    musgo = fbm(xx, yy, 0.5 * m_px, semente + 3)
    tex = np.where((musgo > 0.58)[..., None], tex * 0.55 + np.array([0.30, 0.34, 0.16]) * 0.45, tex)
    liq = fbm(xx, yy, 0.12 * m_px, semente + 6)
    tex = np.where((liq > 0.72)[..., None], tex * 0.5 + 0.40, tex)
    alt = (1 - fy) * (~borda)
    return np.clip(tex, 0, 1), alt


def textura_erva(n=512, metros=8.0, semente=17):
    """Erva seca do monte dos espigueiros (julho): palha, verde apagado e terra."""
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


# ---------------------------------------------------------------- cena

for _ob in list(bpy.data.objects):  # a cena de arranque do bpy traz um cubo, uma luz e uma câmara
    bpy.data.objects.remove(_ob)
RAIZ_OB = bpy.data.objects.new("Castelo_Lindoso", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = CFG["origem"]["lat"], CFG["origem"]["lon"]
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
# "Muralhas_grupo" é o que o visor enquadra ao abrir (com os elementos com placa); o terreno chama-se "Terreno"
for g in ["Terreno", "Torre", "Muralhas", "Museu", "Espigueiros", "Envolvente", "Vegetacao"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

# Granito de Lindoso nas fotografias: cinzento claro, um pouco quente ao sol, com muitos líquenes brancos e
# manchas escuras de musgo nas juntas. A torre e a cerca medieval são de silhares grandes em fiadas (0,35 a
# 0,55 m); a muralha abaluartada é de alvenaria miúda, mais cinzenta e com mais líquenes.
PAL_T = np.array([(0.615, 0.600, 0.565), (0.670, 0.655, 0.620), (0.545, 0.530, 0.500), (0.640, 0.605, 0.535),
                  (0.590, 0.585, 0.570)])
TEX, ALT = textura_cantaria(fiada=(0.36, 0.52), comp=(0.55, 1.35), liquen=0.74, paleta=PAL_T, prob=[0.45, 0.17, 0.16, 0.12, 0.10],
                            var=(0.93, 1.07), liq_forca=0.45, tinta=(1.0, 1.0, 0.985))
material("cantaria_torre", (1, 1, 1), 0.9, imagem("cantaria_torre", TEX), normal=imagem("cantaria_torre_n", normais(ALT, 3.0)),
         forca=0.8, escala=0.5, vcor=True)
TEX, ALT = textura_cantaria(semente=1258, fiada=(0.32, 0.56), comp=(0.45, 1.5), liquen=0.70, paleta=PAL_T,
                            prob=[0.42, 0.16, 0.20, 0.12, 0.10], junta_m=0.010, irregular=0.03, var=(0.90, 1.09),
                            liq_forca=0.45, tinta=(1.0, 1.0, 0.98))
material("muro_medieval", (1, 1, 1), 0.92, imagem("muro_medieval", TEX), normal=imagem("muro_medieval_n", normais(ALT, 3.2)),
         forca=0.9, escala=0.5, vcor=True)
PAL_M = np.array([(0.575, 0.565, 0.540), (0.650, 0.640, 0.610), (0.460, 0.450, 0.430), (0.600, 0.570, 0.510),
                  (0.530, 0.525, 0.510)])
TEX, ALT = textura_cantaria(semente=1662, fiada=(0.16, 0.34), comp=(0.22, 0.65), liquen=0.66, paleta=PAL_M,
                            prob=[0.36, 0.18, 0.22, 0.10, 0.14], junta_m=0.016, irregular=0.09, tinta=(1.03, 1.0, 0.95),
                            var=(0.88, 1.10), liq_forca=0.48)
material("alvenaria", (1, 1, 1), 0.95, imagem("alvenaria", TEX), normal=imagem("alvenaria_n", normais(ALT, 4.0)), forca=1.0, escala=0.8, vcor=True)
TEX, ALT = textura_cantaria(semente=1663, fiada=(0.20, 0.45), comp=(0.25, 0.80), liquen=0.60, paleta=PAL_M,
                            prob=[0.38, 0.18, 0.18, 0.12, 0.14], junta_m=0.018, irregular=0.08, tinta=(0.98, 1.02, 0.93),
                            var=(0.92, 1.07), liq_forca=0.40)
material("muro_topo", (1, 1, 1), 0.95, imagem("muro_topo", TEX), normal=imagem("muro_topo_n", normais(ALT, 3.5)), forca=1.0, escala=0.5, vcor=True)
# espigueiros: granito mais claro e liso (cantaria fina dos esteios, das mesas e das empenas), muito liquenado
PAL_E = np.array([(0.660, 0.650, 0.615), (0.700, 0.690, 0.660), (0.590, 0.580, 0.555), (0.640, 0.615, 0.560)])
TEX, ALT = textura_cantaria(n=512, metros=3.0, semente=1700, fiada=(0.5, 0.9), comp=(0.8, 2.0), liquen=0.62, paleta=PAL_E,
                            prob=[0.4, 0.25, 0.2, 0.15], junta_m=0.004, var=(0.95, 1.05), liq_forca=0.5)
material("granito_esp", (1, 1, 1), 0.9, imagem("granito_esp", TEX), normal=imagem("granito_esp_n", normais(ALT, 2.0)), forca=0.5, escala=1.0, vcor=True)
TEX, ALT = textura_tabuas()
material("grade_esp", (1, 1, 1), 0.9, imagem("grade_esp", TEX), normal=imagem("grade_esp_n", normais(ALT, 2.5)), forca=0.8)
TEX, ALT = textura_lajes()
material("lajes_esp", (1, 1, 1), 0.95, imagem("lajes_esp", TEX), normal=imagem("lajes_esp_n", normais(ALT, 2.0)), forca=0.8)
material("cantaria_lisa", (0.60, 0.585, 0.55), 0.85)
material("intradorso", (0.34, 0.33, 0.31), 0.95)
material("sombra", (0.06, 0.055, 0.05), 1.0)
material("madeira", (0.30, 0.22, 0.15), 0.8)
material("madeira_verde", (0.16, 0.30, 0.22), 0.7)   # portadas da janela da torre (fotografia)
material("madeira_porta", (0.36, 0.17, 0.12), 0.7)   # portas do museu
material("ferro", (0.16, 0.16, 0.17), 0.5)
material("reboco", (0.90, 0.89, 0.86), 0.9)
material("vidro", (0.20, 0.25, 0.28), 0.15)
material("metal", (0.38, 0.39, 0.40), 0.5)
material("bandeira_verde", (0.0, 0.40, 0.18), 0.8)
material("bandeira_vermelha", (0.80, 0.08, 0.10), 0.8)
TEX, ALT = textura_telha()
material("telha", (1, 1, 1), 0.8, imagem("telha", TEX * np.array([0.80, 0.46, 0.31])), normal=imagem("telha_n", normais(ALT, 2.0)), forca=0.6)
TEX, ALT = textura_cubos()
material("calcada", (1, 1, 1), 0.92, imagem("calcada", TEX), normal=imagem("calcada_n", normais(ALT, 1.5)), forca=0.5)
TEX, ALT = textura_relva()
material("relva", (1, 1, 1), 0.95, imagem("relva", TEX), normal=imagem("relva_n", normais(ALT, 1.0)), forca=0.4)
TEX, ALT = textura_erva()
material("erva", (1, 1, 1), 0.95, imagem("erva", TEX), normal=imagem("erva_n", normais(ALT, 1.0)), forca=0.4)
TEX, ALT = textura_terra()
material("terra", (1, 1, 1), 0.95, imagem("terra", TEX), normal=imagem("terra_n", normais(ALT, 1.0)), forca=0.4)
TEX, ALT = textura_rocha()
material("rocha", (1, 1, 1), 0.85, imagem("rocha", TEX), normal=imagem("rocha_n", normais(ALT, 2.0)), forca=0.6)
material("terreiro", (0.62, 0.60, 0.55), 0.95)
material("verde_viz", (0.48, 0.54, 0.38), 0.95)
material("asfalto", (0.33, 0.33, 0.34), 0.95)
# casas vizinhas: um só cinzento claro, sem cor nem textura, para o destaque ficar todo no castelo e nos espigueiros
material("cinza_viz", (0.74, 0.74, 0.73), 0.9)
material("base", (0.33, 0.26, 0.20), 0.95)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)

MEDIDAS = {}


def patine_torre(co):
    """A cantaria da torre escurece para o alto (líquenes e musgo) e junto ao chão (salpicos)."""
    z = co[:, 2]
    t = np.clip((z - 466.0) / 9.0, 0, 1) ** 1.3
    base = np.stack([1.0 - 0.08 * t, 1.0 - 0.06 * t, 1.0 - 0.07 * t], 1)
    pe = np.clip(1 - (z - 461.8) / 1.5, 0, 1)[:, None] * 0.15
    return np.clip(base * (1 - pe), 0, 1)


def patine_muro(co):
    """Os muros têm manchas grandes mais claras e mais escuras (pedra reposta e pedra velha, musgo) e são mais
    escuros ao pé."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x + 500, y + 500 + z * 0.7, 7.0, 31)
    v = 0.86 + 0.26 * g
    pe = np.clip(1 - (z - cota(x, y)) / 1.4, 0, 1) * 0.14
    c = np.stack([v, v * (1.0 + 0.02 * (g - 0.5)), v * 0.99], 1)
    return np.clip(c * (1 - pe)[:, None], 0, 1)


def patine_esp(co):
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x * 3 + 100, y * 3 + z, 2.0, 41)
    v = 0.86 + 0.24 * g
    return np.clip(np.stack([v, v, v * 0.98], 1), 0, 1)


FACES = {"ENE": face_torre((1, 0), A), "NNO": face_torre((0, 1), A), "OSO": face_torre((-1, 0), A), "SSE": face_torre((0, -1), A)}


# ---------------------------------------------------------------- torre de menagem

PT = np.load(AQUI / "dados" / "torre.npz")["pontos"]
_uv = np.stack([PT[:, :2] @ EU, PT[:, :2] @ EV], 1)
_dentro = (np.abs(_uv[:, 0]) < A + 0.3) & (np.abs(_uv[:, 1]) < A + 0.3) & ~np.isin(PT[:, 3], (3, 4, 5, 7))
_anel = _dentro & ((np.abs(_uv[:, 0]) > A - 0.9) | (np.abs(_uv[:, 1]) > A - 0.9))
_cent = _dentro & (np.abs(_uv[:, 0]) < 0.7) & (np.abs(_uv[:, 1]) < 0.7)
_chao = {}
for _nm, (_a, _b) in {"ENE": (1, 0), "NNO": (0, 1), "OSO": (-1, 0), "SSE": (0, -1)}.items():
    _s = (PT[:, 3] == 2) & ((_a * _uv[:, 0] > A + 0.4) & (_a * _uv[:, 0] < A + 2.5) & (np.abs(_uv[:, 1]) < A - 1) if _a else
                            (_b * _uv[:, 1] > A + 0.4) & (_b * _uv[:, 1] < A + 2.5) & (np.abs(_uv[:, 0]) < A - 1))
    _chao[_nm] = round(float(np.percentile(PT[_s, 2], 30)), 2)
CHAO_T = _chao                                   # chão junto a cada face (LiDAR)
Z_PONTA = round(float(np.percentile(PT[_anel & (PT[:, 2] < 476.5), 2], 98.5)), 2)   # pontas dos merlões (sem o mastro)
Z_CUME = round(float(np.percentile(PT[_cent, 2], 90)), 2)      # vértice do telhado de quatro águas
Z_AMEIA = 473.45         # soleira das ameias (LiDAR: o anel tem o modo a 473,1–473,5 entre os merlões)
Z_MERLAO = Z_PONTA - 0.55  # corpo dos merlões; o remate piramidal tem 0,55 m (fotografias)
Z_BEIRAL = 473.05        # beiral do telhado, por dentro do parapeito (LiDAR: 473,1 a 4 m do centro)
Z_ADARVE_T = 472.55      # chão do adarve da torre, por dentro do parapeito
ESP_PAR = 0.75           # espessura do parapeito
N_MERL = 7               # merlões por face, contando os dos cunhais (fotografias)
LM, LC = 0.78, 0.95      # largura dos merlões (os dos cunhais são um pouco mais largos)
print("torre: chão", CHAO_T, "pontas", Z_PONTA, "cume", Z_CUME)

# vãos por face (s ao longo da face: o eixo t de face_torre aponta para a ESQUERDA de quem olha a face de fora)
# SSE (para o pátio): janela retangular a meia altura, à direita do eixo, com portadas verdes (fotografia do pátio:
#      centro a 60 % da largura e a 52 % da altura até às ameias); sem seteira por cima
# OSO: a porta, ao rés do pátio, perto do cunhal sul (estimativa: a fotografia do pátio mostra uma porta em arco à
#      esquerda da torre, onde encosta a ala poente do museu)
# NNO e ENE (para as lices): seteiras
VAOS = {
    "SSE": [(-1.55, -0.55, 467.45, 469.0, False, "janela")],
    "OSO": [(-3.55, -2.55, CHAO_T["OSO"] + 0.05, CHAO_T["OSO"] + 2.35, True, "porta"), (0.5, 0.62, 468.8, 469.9, False, "seteira")],
    "NNO": [(-0.06, 0.06, 466.4, 467.4, False, "seteira"), (-0.06, 0.06, 470.2, 471.2, False, "seteira")],
    "ENE": [(-0.06, 0.06, 467.0, 468.0, False, "seteira"), (-0.06, 0.06, 470.4, 471.4, False, "seteira")],
}


def merloes_pos():
    """Posições dos merlões ao longo de uma face (s de -A a A): sete, os dos cunhais a encostar à esquina."""
    vao = (2 * A - 2 * LC - (N_MERL - 2) * LM) / (N_MERL - 1)
    out = [(-A, -A + LC)]
    s = -A + LC + vao
    for _ in range(N_MERL - 2):
        out.append((s, s + LM))
        s += LM + vao
    out.append((A - LC, A))
    return out, vao


def torre():
    m = Malha(["cantaria_torre", "cantaria_lisa", "sombra", "madeira", "ferro", "telha", "madeira_verde"])
    z0 = min(CHAO_T.values()) - 0.8
    for nome, f in FACES.items():
        vaos = [(a, b, za, zb, arco) for a, b, za, zb, arco, _ in VAOS[nome]]
        parede(m, f, -A, A, z0, Z_AMEIA, vaos, mat=0)
        for a, b, za, zb, arco, tipo in VAOS[nome]:
            w = b - a
            sc = (a + b) / 2
            if tipo == "porta":
                vao(m, f, sc, w, za, zb, 0.55, arco=True, mat_lado="cantaria_lisa", mat_fundo="madeira")
                aduelas(m, f, sc, w, za, zb, 0.28, 0.05, n=11)
            elif tipo == "janela":
                vao(m, f, sc, w, za, zb, 0.25, arco=False, mat_lado="cantaria_lisa", mat_fundo="madeira_verde")
                # moldura de cantaria e peitoril
                m.caixa_face(f, a - 0.14, b + 0.14, 0.0, 0.05, zb, zb + 0.14, 1)
                m.caixa_face(f, a - 0.18, b + 0.18, 0.0, 0.09, za - 0.12, za, 1)
                for s0_, s1_ in ((a - 0.14, a), (b, b + 0.14)):
                    m.caixa_face(f, s0_, s1_, 0.0, 0.05, za, zb, 1)
                # as duas portadas e a junta entre elas
                m.caixa_face(f, sc - 0.015, sc + 0.015, -0.24, -0.20, za, zb, 2)
            else:
                vao(m, f, sc, w, za, zb, 0.6, arco=False, mat_lado="cantaria_lisa", mat_fundo="sombra")
    # parapeito: face de fora já vem de parede() até à soleira das ameias; aqui o adarve, a face de dentro e os merlões
    r = A - ESP_PAR
    q_in = [W(-r, -r), W(r, -r), W(r, r), W(-r, r)]
    q_out = [W(-A, -A), W(A, -A), W(A, A), W(-A, A)]
    for i in range(4):  # topo do parapeito entre os merlões (soleira das ameias)
        a, b, c, d = q_out[i], q_out[(i + 1) % 4], q_in[(i + 1) % 4], q_in[i]
        m.face([(*a, Z_AMEIA), (*b, Z_AMEIA), (*c, Z_AMEIA), (*d, Z_AMEIA)], 1)
        m.face([(*d, Z_ADARVE_T), (*c, Z_ADARVE_T), (*c, Z_AMEIA), (*d, Z_AMEIA)], 0)
    pos, _ = merloes_pos()
    for nome, f in FACES.items():
        for sa, sb in pos[1:-1]:
            merlao(m, f, sa, sb)
    for nuv in ((1, 1), (1, -1), (-1, -1), (-1, 1)):  # merlões dos cunhais, em L
        c = W(nuv[0] * A, nuv[1] * A)
        du, dv = -nuv[0] * LC, -nuv[1] * LC
        base = [c, c + EU * du, c + EU * du + EV * dv * (ESP_PAR / LC), c + EU * du * (ESP_PAR / LC) + EV * dv * (ESP_PAR / LC),
                c + EU * du * (ESP_PAR / LC) + EV * dv, c + EV * dv]
        base = np.array(base)
        if shapely.Polygon(base).exterior.is_ccw is False:
            base = base[::-1]
        m.prisma(list(base), Z_AMEIA, Z_MERLAO, 0, topo=False)
        _remate(m, list(base), Z_MERLAO, Z_PONTA)
    # telhado de quatro águas por dentro do parapeito
    rt = r - 0.05
    beiral = [W(-rt, -rt), W(rt, -rt), W(rt, rt), W(-rt, rt)]
    m.piramide(beiral, Z_BEIRAL, Z_CUME, 5)
    m.face([(*p, Z_BEIRAL - 0.02) for p in beiral[::-1]], 1)
    ob = m.objeto("torre_menagem", "torre", GRUPOS["Torre"], uv=3.0, vcor=patine_torre)
    MEDIDAS["torre"] = {"lado": 2 * A, "azimute_faces": [ROT_TORRE, (ROT_TORRE + 90) % 360], "chao": CHAO_T,
                        "soleira_ameias": Z_AMEIA, "pontas_merloes": Z_PONTA, "cume_telhado": Z_CUME,
                        "altura_sse": round(Z_PONTA - CHAO_T["SSE"], 2), "merloes_por_face": N_MERL}
    return ob


def _remate(m, base, z0, z1):
    """Remate piramidal de um merlão (fotografias: pirâmide baixa de quatro faces sobre o corpo)."""
    m.piramide(base, z0, z1, 0)


def merlao(m, f, sa, sb):
    base = [f.xy(sa, 0.0), f.xy(sb, 0.0), f.xy(sb, -ESP_PAR), f.xy(sa, -ESP_PAR)]
    if shapely.Polygon(base).exterior.is_ccw is False:
        base = base[::-1]
    m.prisma(base, Z_AMEIA, Z_MERLAO, 0, topo=False)
    _remate(m, base, Z_MERLAO, Z_PONTA)


def bandeira():
    """Mastro com a bandeira nacional no parapeito NNO (o LiDAR tem pontos até 479 m em u -1,9 e as fotografias
    mostram a bandeira)."""
    m = Malha(["ferro", "bandeira_verde", "bandeira_vermelha"])
    p = W(-1.9, A - 0.4)
    m.cilindro(p, 0.04, Z_ADARVE_T, 479.0, 0, k=8)
    d = unit(W(-1, 0))
    h0, h1 = 477.9, 478.9
    L = 1.5
    for (a, b, mt) in ((0.0, 0.4, 1), (0.4, 1.0, 2)):
        pa, pb = p + d * L * a, p + d * L * b
        m.face([(*pa, h0), (*pb, h0), (*pb, h1), (*pa, h1)], mt)
        m.face([(*pa, h1), (*pb, h1), (*pb, h0), (*pa, h0)], mt)
    m.objeto("bandeira", "torre", GRUPOS["Torre"], uv=None, recalc=False)


# ---------------------------------------------------------------- muros (tracar.py)

ALTO_MEDIEVAL = 465.8  # acima disto o muro é da cerca medieval (cantaria); abaixo, muralha abaluartada (alvenaria)


def poligono_muro(reg):
    a = reg["aneis"]
    return shapely.Polygon(a[0]["contorno"], [r["contorno"] for r in a[1:]])


POL_MUROS = shapely.union_all([poligono_muro(r).buffer(0) for r in MUR])


def muros():
    """Os muros como o LiDAR os mede: topo triangulado (Delaunay restringida à planta) com a cota medida em cada
    vértice, faces a prumo até abaixo do chão de fora. Dois objetos: a cerca medieval (topo acima de 465,8 m) e a
    muralha abaluartada."""
    mm = {True: Malha(["muro_medieval", "alvenaria", "muro_topo", "erva"]), False: Malha(["muro_medieval", "alvenaria", "muro_topo", "erva"])}
    for reg in MUR:
        pol = poligono_muro(reg).buffer(0)
        if pol.geom_type != "Polygon":
            continue
        vs = np.concatenate([np.array(a["contorno"]) for a in reg["aneis"]])
        ts = np.concatenate([np.array(a["topo"]) for a in reg["aneis"]])
        kd = cKDTree(vs)
        for t in shapely.constrained_delaunay_triangles(pol).geoms:
            cs = np.array(t.exterior.coords)[:3]
            if shapely.Polygon(cs).area < 1e-4:
                continue
            if not shapely.Polygon(cs).exterior.is_ccw:
                cs = cs[::-1]
            _, j = kd.query(cs)
            # os terraplenos dos baluartes (topo largo) têm erva; o alto dos muros e o adarve, lajes
            largo = pol.exterior.distance(shapely.Point(cs.mean(0))) > 1.7 and all(
                shapely.LinearRing(h).distance(shapely.Point(cs.mean(0))) > 1.7 for h in pol.interiors)
            zt = ts[j]
            mm[bool(zt.mean() > ALTO_MEDIEVAL)].face([(x, y, z) for (x, y), z in zip(cs, zt)], 3 if largo else 2)
        for a in reg["aneis"]:
            c = np.array(a["contorno"])
            tp = np.array(a["topo"])
            ch = np.minimum(np.array(a["chao"]), cota(c[:, 0], c[:, 1])) - 0.6
            n = len(c)
            for i in range(n):
                j = (i + 1) % n
                alto = bool((tp[i] + tp[j]) / 2 > ALTO_MEDIEVAL)
                mm[alto].face([(*c[i], ch[i]), (*c[j], ch[j]), (*c[j], tp[j]), (*c[i], tp[i])], 0 if alto else 1)
    ob = mm[True].objeto("cerca_medieval", "cerca", GRUPOS["Muralhas"], uv=3.0, vcor=patine_muro)
    mm[False].objeto("muralha_abaluartada", "baluartes", GRUPOS["Muralhas"], uv=3.0, vcor=patine_muro)
    MEDIDAS["muros"] = {"regioes": len(MUR), "area_m2": round(POL_MUROS.area),
                        "topo_max": round(max(max(a["topo"]) for r in MUR for a in r["aneis"]), 2)}
    return ob


def topo_em(pol, pct=90):
    """Cota do LiDAR (classes solo, edifícios e muros) dentro de um polígono."""
    minx, miny, maxx, maxy = pol.bounds
    gx, gy = np.meshgrid(np.arange(minx, maxx, 0.25), np.arange(miny, maxy, 0.25))
    d = shapely.contains_xy(pol, gx.ravel(), gy.ravel())
    return float(np.percentile(superficie(gx.ravel()[d], gy.ravel()[d]), pct))


# ---------------------------------------------------------------- portas

def _eixo(p, q):
    return unit(np.asarray(q, float) - np.asarray(p, float))


def torre_porta():
    """Torre da porta, a sul: um bloco de cantaria saliente da cerca (planta e topo do LiDAR) com a passagem em
    arco de volta perfeita (fotografia de drone de sul) que segue pela cerca para o pátio."""
    tp = np.array(TORRE_DA_PORTA)
    # face sul (de fora): de (6.5, -32.7) a (10.8, -32.6); a normal aponta para sul
    a, b = tp[3], tp[2]
    t = _eixo(a, b)
    f = Plano((a + b) / 2, t, n=np.array([t[1], -t[0]]))
    larg = float(np.linalg.norm(b - a))
    prof = float(np.linalg.norm((tp[0] + tp[1]) / 2 - (a + b) / 2))
    z_topo = topo_em(shapely.Polygon(tp).buffer(-0.4), 80)
    chao_f, chao_d = 461.2, 461.6
    m = Malha(["muro_medieval", "muro_topo", "intradorso", "calcada", "cantaria_lisa", "madeira_porta", "ferro"])
    # a passagem: o bloco da torre e o troço da cerca por trás (até ao pátio), com o mesmo vão
    passagem(m, f, -larg / 2, larg / 2, prof, chao_f - 1.0, z_topo, 2.0, 3.2, chao_f, chao_f + (chao_d - chao_f) * 0.6,
             mat="muro_medieval", mat_topo="muro_topo", degrau=(2.5, 3.55, 0.35))
    ps = np.array(PASSAGEM_SUL)
    f2 = Plano(f.xy(0, -prof), t, n=f.n)
    prof2 = float(np.linalg.norm((ps[2] + ps[3]) / 2 - (ps[0] + ps[1]) / 2)) + 0.3
    z2 = topo_em(shapely.Polygon(ps).buffer(-0.3), 80)
    passagem(m, f2, -1.35, 1.35, prof2, chao_f - 1.0, z2, 2.0, 3.2, chao_f + (chao_d - chao_f) * 0.6, chao_d,
             mat="muro_medieval", mat_topo="muro_topo", lados=True)
    aduelas(m, f, 0.0, 2.5, chao_f, chao_f + 3.55, 0.32, 0.04, n=13, ombreiras=False, d0=0.0)
    # parapeito baixo à volta do topo da torre (fotografia: o topo é plano, com um murete)
    for i in range(4):
        p, q = tp[i], tp[(i + 1) % 4]
        d = unit(q - p)
        nn = np.array([-d[1], d[0]])
        if np.dot(nn, shapely.Polygon(tp).centroid.coords[0] - p) < 0:
            nn = -nn
        base = [p, q, q + nn * 0.45, p + nn * 0.45]
        if not shapely.Polygon(base).exterior.is_ccw:
            base = base[::-1]
        m.prisma(base, z_topo - 0.05, z_topo + 0.55, 0, mat_topo=1)
    m.objeto("torre_porta", "porta_sul", GRUPOS["Muralhas"], uv=3.0, vcor=patine_muro)
    # porta de madeira aberta encostada ao vão, por dentro (de dia está aberta)
    MEDIDAS["torre_porta"] = {"topo": round(z_topo, 2), "vao": [2.0, 3.2], "arquivolta": [2.5, 3.55], "chao_fora": chao_f,
                              "chao_dentro": chao_d}
    return f, z_topo


def porta_norte():
    """Porta Norte (fotografia com a torre por trás): vão em arco de volta perfeita com aduelas de granito, portão de
    grade de ferro e estrado de madeira à frente; atravessa a ponta poente da cortina norte."""
    c = np.array(PORTA_NORTE["c"])
    az = math.radians(PORTA_NORTE["az"])
    t = np.array([math.sin(az), math.cos(az)])
    n = np.array([-t[1], t[0]])  # para fora (norte)
    meia = PORTA_NORTE["meia"]
    f = Plano(c + n * meia, t, n=n)
    s0, s1 = PORTA_NORTE["s0"], PORTA_NORTE["s1"]
    blk = shapely.Polygon([f.xy(s0, 0), f.xy(s1, 0), f.xy(s1, -2 * meia), f.xy(s0, -2 * meia)])
    z_topo = max(topo_em(blk.buffer(-0.2), 85), 464.0)
    chao_f, chao_d = 461.0, 461.8
    m = Malha(["alvenaria", "muro_topo", "intradorso", "calcada", "cantaria_lisa", "madeira", "ferro"])
    passagem(m, f, s0, s1, 2 * meia, chao_f - 1.0, z_topo, 1.7, 2.75, chao_f + 0.25, chao_d, mat="alvenaria",
             mat_topo="muro_topo")
    aduelas(m, f, 0.0, 1.7, chao_f + 0.25, chao_f + 0.25 + 2.75, 0.30, 0.05, n=11)
    # grade de ferro no vão, recuada 0,25 m
    r = 0.85
    zc = chao_f + 0.25 + 2.75 - r
    for s in np.arange(-r + 0.1, r - 0.05, 0.14):
        ztop = zc + math.sqrt(max(r * r - s * s, 0)) - 0.03
        m.barra(f.p(s, -0.25, chao_f + 0.27), f.p(s, -0.25, ztop), 0.012, 6, k=4)
    for z in (chao_f + 0.4, chao_f + 1.3, zc):
        w = r if z <= zc else 0.5
        m.barra(f.p(-w + 0.02, -0.25, z), f.p(w - 0.02, -0.25, z), 0.02, 6, k=4)
    # estrado de madeira à frente da porta (fotografia), sobre o chão de fora
    m.caixa_face(f, -1.25, 1.25, 0.0, 1.5, chao_f + 0.05, chao_f + 0.25, 5)
    for s in (-1.1, 1.1):
        for d in (0.15, 1.35):
            m.caixa_face(f, s - 0.06, s + 0.06, d - 0.06, d + 0.06, chao_f - 0.6, chao_f + 0.05, 5)
    m.objeto("porta_norte", "porta_norte", GRUPOS["Muralhas"], uv=3.0, vcor=patine_muro)
    MEDIDAS["porta_norte"] = {"topo": round(z_topo, 2), "vao": [1.7, 2.75], "chao_fora": chao_f, "chao_dentro": chao_d}
    return f


def postigo():
    """Porta da traição: arco estreito na face de fora da cerca NO, c. 2,5 m a SO do cunhal oeste da torre
    (fotografia). Vão de 0,8 × 1,9 m, emparedado ao fundo."""
    # a face de fora da cerca a NO corre na direção da face NNO da torre (u), recuada para a lice
    t = unit(W(1, 0))
    nrm = unit(W(0, 1))
    o = np.array((-10.1, 1.2))
    # chão e face do muro medidos: procura a face (o salto do MDS) ao longo da normal
    d_ = np.arange(-2.0, 2.01, 0.1)
    zs = superficie(o[0] + nrm[0] * d_, o[1] + nrm[1] * d_)
    k = int(np.argmax(np.diff(zs) < -1.5)) if np.any(np.diff(zs) < -1.5) else 20
    o = o + nrm * d_[k]
    f = Plano(o, -t, n=nrm)
    ch = float(cota(*(o + nrm * 1.0)))
    m = Malha(["cantaria_lisa", "sombra", "madeira"])
    vao(m, f, 0.0, 0.8, ch, ch + 1.9, 0.6, arco=True, mat_lado="cantaria_lisa", mat_fundo="sombra", d0=0.02)
    aduelas(m, f, 0.0, 0.8, ch, ch + 1.9, 0.22, 0.04, n=7, d0=0.02)
    m.objeto("postigo", "postigo", GRUPOS["Muralhas"], uv=2.0)
    MEDIDAS["postigo"] = {"centro": [round(float(o[0]), 2), round(float(o[1]), 2)], "chao": round(ch, 2), "vao": [0.8, 1.9]}
    return o, ch


def guarita(m, c, z0, dir_fora, r=0.62, alt=2.1):
    """Guarita redonda (fotografia do cunhal SE): mísula em cone invertido com gárgula, corpo cilíndrico com
    seteiras, cornija, cúpula de pedra e pináculo."""
    dfo = unit(dir_fora)
    m.cilindro(c, 0.15, z0 - 1.25, z0 - 0.05, 0, k=14, topo=False, r1=r)
    m.cilindro(c, r, z0 - 0.05, z0 + 0.12, 0, k=16, r1=r + 0.06)                     # cordão
    m.cilindro(c, r, z0 + 0.12, z0 + alt, 0, k=16, topo=False)                       # corpo
    m.cilindro(c, r + 0.07, z0 + alt, z0 + alt + 0.14, 0, k=16, topo=False)          # cornija
    m.face([(c[0] + (r + 0.07) * math.cos(a), c[1] + (r + 0.07) * math.sin(a), z0 + alt) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)][::-1], 0)
    # cúpula em três anéis
    zz = z0 + alt + 0.14
    rs = [r + 0.03, r * 0.85, r * 0.55, 0.12]
    hs = [0.0, 0.2, 0.38, 0.5]
    for i in range(3):
        m.cilindro(c, rs[i], zz + hs[i], zz + hs[i + 1], 0, k=16, topo=False, r1=rs[i + 1])
    m.cilindro(c, 0.12, zz + 0.5, zz + 0.62, 0, k=8, r1=0.0)                         # pináculo
    m.cilindro(c, 0.06, zz + 0.62, zz + 0.74, 0, k=8)
    # seteiras (frestas escuras) para fora e para os lados
    for ang in (-60, 0, 60):
        a = math.atan2(dfo[1], dfo[0]) + math.radians(ang)
        p = c + np.array([math.cos(a), math.sin(a)]) * (r + 0.005)
        tt = np.array([-math.sin(a), math.cos(a)])
        w = 0.11 if ang else 0.16
        z_a, z_b = (z0 + 1.15, z0 + 1.55) if ang else (z0 + 1.2, z0 + 1.6)
        q0, q1 = p - tt * w, p + tt * w
        m.face([(*q0, z_a), (*q1, z_a), (*q1, z_b), (*q0, z_b)], 1)
    # gárgula de pedra por baixo, para fora (fotografia)
    pg = c + dfo * (r * 0.4)
    m.barra((*pg, z0 - 0.75), (*(pg + dfo * 0.75), z0 - 0.85), 0.11, 0, k=6)


def guaritas():
    m = Malha(["muro_medieval", "sombra"])
    out = []
    # cunhal SE do bloco alto da cerca (fotografia): no canto (19,8, -30,2), por cima do topo
    vs = np.concatenate([np.array(a["contorno"]) for r in MUR for a in r["aneis"]])
    ts = np.concatenate([np.array(a["topo"]) for r in MUR for a in r["aneis"]])
    kd = cKDTree(vs)
    for c, dfo in (((19.8, -30.2), (0.7, -0.7)), ((-31.0, 3.0), (-1.0, 0.25))):
        # assenta no vértice do muro mais próximo do cunhal, recuada 0,35 m para dentro; cota do topo medido
        _, j = kd.query(c)
        c = vs[j] - unit(dfo) * 0.35
        z0 = float(np.max(ts[kd.query_ball_point(vs[j], 1.5)])) + 0.1
        guarita(m, c, z0, dfo)
        out.append([round(float(c[0]), 2), round(float(c[1]), 2), round(z0, 2)])
    m.objeto("guaritas", "guaritas", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)
    MEDIDAS["guaritas"] = out


# ---------------------------------------------------------------- museu (no pátio)

def _hip(m, quad, z_beiral, z_cume, beiral=0.45, mat=0, mat_sob=None):
    """Telhado de quatro águas sobre um quadrilátero (cume ao longo dos lados maiores), com beiral."""
    q = shapely.Polygon(quad)
    if not q.exterior.is_ccw:
        quad = quad[::-1]
    q = np.array(shapely.Polygon(quad).buffer(beiral, join_style="mitre").exterior.coords)[:-1]
    if len(q) != 4:
        q = np.array(quad, float)
    if not shapely.Polygon(q).exterior.is_ccw:
        q = q[::-1]
    l01 = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[3] - q[2])
    l12 = np.linalg.norm(q[2] - q[1]) + np.linalg.norm(q[0] - q[3])
    if l12 > l01:
        q = np.roll(q, -1, 0)
    # p0p1 e p2p3 são os lados compridos; p1p2 e p3p0 os topos
    m_a = (q[3] + q[0]) / 2
    m_b = (q[1] + q[2]) / 2
    ax = unit(m_b - m_a)
    hw = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    L = np.linalg.norm(m_b - m_a)
    k = min(hw, 0.45 * L)
    ra, rb = m_a + ax * k, m_b - ax * k
    zb = z_beiral - 0.0
    P = lambda p, z: (p[0], p[1], z)  # noqa: E731
    m.face([P(q[0], zb), P(q[1], zb), P(rb, z_cume), P(ra, z_cume)], mat)
    m.face([P(q[1], zb), P(q[2], zb), P(rb, z_cume)], mat)
    m.face([P(q[2], zb), P(q[3], zb), P(ra, z_cume), P(rb, z_cume)], mat)
    m.face([P(q[3], zb), P(q[0], zb), P(ra, z_cume)], mat)
    m.face([P(p, zb - 0.02) for p in q[::-1]], mat if mat_sob is None else mat_sob)
    # beirado (testa do beiral)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        m.face([P(a, zb - 0.12), P(b, zb - 0.12), P(b, zb), P(a, zb)], mat if mat_sob is None else mat_sob)


def _janelas(m, a, b, z0, z1, passo, w, h, z_peit, porta=None):
    """Janelas (vidro escuro em caixilho recuado) e uma porta, numa fachada de a para b (vista de fora)."""
    t = unit(np.asarray(b) - np.asarray(a))
    f = Plano(a, t, n=np.array([t[1], -t[0]]))
    L = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
    n = max(1, int((L - 1.5) // passo))
    ss = np.linspace(-(n - 1) / 2 * passo, (n - 1) / 2 * passo, n) + L / 2
    for k, s in enumerate(ss):
        if porta is not None and k == porta:
            vao(m, f, s, 1.1, z0 + 0.02, z0 + 2.3, 0.18, arco=False, mat_lado="reboco", mat_fundo="madeira_porta", d0=0.0)
            m.caixa_face(f, s - 0.7, s + 0.7, 0.0, 0.04, z0 + 2.3, z0 + 2.45, "cantaria_lisa")
            continue
        vao(m, f, s, w, z_peit, z_peit + h, 0.14, arco=False, mat_lado="reboco", mat_fundo="vidro", d0=0.0)
        m.caixa_face(f, s - w / 2 - 0.08, s + w / 2 + 0.08, 0.0, 0.06, z_peit - 0.08, z_peit, "cantaria_lisa")


def museu():
    """Os edifícios do museu no pátio (fotografias de drone e satélite): duas alas de paredes rebocadas a branco
    com soco de granito e telhados de quatro águas de telha, e entre elas a galeria envidraçada de cobertura plana.
    Planta do LiDAR e do satélite; cornijas e cumes do LiDAR."""
    m = Malha(["reboco", "telha", "cantaria_lisa", "vidro", "madeira_porta", "metal", "sombra"])
    out = {}
    for nome, quad in (("ala_oeste", ALA_OESTE), ("ala_sul", ALA_SUL)):
        q = np.array(quad, float)
        if not shapely.Polygon(q).exterior.is_ccw:
            q = q[::-1]
        pol = shapely.Polygon(q)
        chao = float(np.percentile(cota(*np.array(pol.buffer(1.0).exterior.coords).T), 15))
        z_cume = topo_em(pol.buffer(-0.5), 97)
        anel = pol.buffer(-0.3).difference(pol.buffer(-1.2))
        z_beir = topo_em(anel, 15)
        z_beir = min(max(z_beir, chao + 2.7), z_cume - 0.8)
        m.prisma(list(q), chao - 0.8, chao + 0.45, 2, topo=False)      # soco de granito
        b = list(np.array(pol.buffer(-0.02, join_style="mitre").exterior.coords)[:-1])
        if not shapely.Polygon(b).exterior.is_ccw:
            b = b[::-1]
        m.prisma(b, chao + 0.45, z_beir, 0, topo=False)
        _hip(m, list(q), z_beir, z_cume, beiral=0.45, mat=1, mat_sob=0)
        out[nome] = {"chao": round(chao, 2), "beiral": round(z_beir, 2), "cume": round(z_cume, 2)}
    # fachadas para o pátio: ala oeste a nascente, ala sul a norte (fotografia: janelas e portas)
    qo = np.array(ALA_OESTE)
    ch = out["ala_oeste"]["chao"]
    _janelas(m, qo[1], qo[2], ch, out["ala_oeste"]["beiral"], 2.8, 0.9, 1.2, ch + 1.0, porta=1)
    qs = np.array(ALA_SUL)
    ch = out["ala_sul"]["chao"]
    _janelas(m, qs[1], qs[0], ch, out["ala_sul"]["beiral"], 2.8, 0.9, 1.2, ch + 1.0, porta=2)
    # galeria envidraçada
    q = np.array(CORREDOR, float)
    if not shapely.Polygon(q).exterior.is_ccw:
        q = q[::-1]
    pol = shapely.Polygon(q)
    chao = float(np.percentile(cota(*np.array(pol.exterior.coords).T), 20))
    zt = min(topo_em(pol.buffer(-0.4), 60), chao + 3.4)
    m.prisma(list(q), chao - 0.5, chao + 0.3, 2, topo=False)
    m.prisma(list(q), chao + 0.3, zt - 0.35, 3, topo=False)
    tt = list(np.array(pol.buffer(0.25, join_style="mitre").exterior.coords)[:-1])
    if not shapely.Polygon(tt).exterior.is_ccw:
        tt = tt[::-1]
    m.prisma(tt, zt - 0.35, zt - 0.1, 5)
    m.face([(*p, zt - 0.1) for p in tt], 3)
    # montantes de metal da galeria
    for i in range(4):
        a_, b_ = q[i], q[(i + 1) % 4]
        L = np.linalg.norm(b_ - a_)
        for s in np.arange(0, L + 1e-6, max(L / max(1, round(L / 1.6)), 0.5)):
            p = a_ + (b_ - a_) * (s / L)
            m.barra((*p, chao + 0.3), (*p, zt - 0.35), 0.04, 5, k=4)
    out["galeria"] = {"chao": round(chao, 2), "cobertura": round(zt, 2)}
    m.objeto("museu", "museu", GRUPOS["Museu"], uv=2.0)
    MEDIDAS["museu"] = out


# ---------------------------------------------------------------- espigueiros

def lista_espigueiros():
    """Espigueiros: as plantas estreitas do OSM (c. 5 × 2 m) na eira a sul e sudoeste do castelo, e as manchas do
    LiDAR com a forma de um espigueiro que o OSM não tem. Comprimento, largura e orientação da planta; cume do
    LiDAR (o ponto mais alto da classe 6 dentro da planta)."""
    lst = []
    for f in F["edificios"]:
        g = shapely.geometry.shape(f["geometria"])
        if g.geom_type != "Polygon":
            continue
        c = g.centroid
        if not (c.y < -45 and c.x < 25 and X0 + 2.5 < c.x < X1 - 2.5 and Y0 + 2.5 < c.y < Y1 - 2.5):
            continue
        r = np.array(g.minimum_rotated_rectangle.exterior.coords)[:4]
        e = [np.linalg.norm(r[1] - r[0]), np.linalg.norm(r[2] - r[1])]
        if min(e) > 3.2 or g.area > 40 or max(e) < 2.6:
            continue
        k = int(np.argmax(e))
        d = r[k + 1] - r[k]
        lst.append({"c": (c.x, c.y), "L": max(e), "W": min(e), "az": math.degrees(math.atan2(d[0], d[1])) % 180, "fonte": "OSM"})
    pts = np.array([e["c"] for e in lst]) if lst else np.zeros((0, 2))
    for e in json.loads((AQUI / "dados" / "espigueiros_lidar.json").read_text()):
        c = np.array(e["c"])
        if len(pts) and np.min(np.linalg.norm(pts - c, axis=1)) < 3.0:
            continue
        if not (1.6 <= e["W"] <= 3.4 and 3.0 <= e["L"] <= 9.0 and e["hmax"] > 2.2 and e["cls6"] > 0.5):
            continue
        if not (c[1] < -45 and c[0] < 25 and X0 + 2.5 < c[0] < X1 - 2.5 and Y0 + 2.5 < c[1] < Y1 - 2.5):
            continue
        lst.append({"c": tuple(c), "L": e["L"], "W": max(e["W"] - 0.4, 1.8), "az": e["az"] % 180, "fonte": "LiDAR"})
    for e in lst:
        c = np.array(e["c"])
        t = np.array([math.sin(math.radians(e["az"])), math.cos(math.radians(e["az"]))])
        n = np.array([-t[1], t[0]])
        ss, dd = np.meshgrid(np.linspace(-e["L"] / 2 + 0.3, e["L"] / 2 - 0.3, 9), np.linspace(-0.5, 0.5, 5))
        p = c + t * ss.ravel()[:, None] + n * dd.ravel()[:, None]
        e["cume"] = float(np.percentile(amostra(MDS, LX0, LY0, LP, p[:, 0], p[:, 1], 0), 95))
        pc = c + t * np.linspace(-e["L"] / 2, e["L"] / 2, 7)[:, None]
        zc = cota(pc[:, 0], pc[:, 1])
        e["chao_min"], e["chao_max"] = float(zc.min()), float(zc.max())
    return lst


def espigueiro(m, e, i):
    """Um espigueiro de Lindoso (fotografias): pés de granito aos pares com as mesas (discos) por cima, a soleira de
    lajes, o corpo de esteios de granito com grades de ripas de madeira nos lados compridos, as empenas de granito
    (numa delas a porta de madeira em arco), o telhado de lajes de duas águas e a cruz e o pináculo nos topos."""
    c = np.array(e["c"])
    t = np.array([math.sin(math.radians(e["az"])), math.cos(math.radians(e["az"]))])
    f = Plano(c, t)
    L = float(np.clip(e["L"], 3.0, 12.0))
    Wt = float(np.clip(e["W"], 1.7, 2.6))                     # largura do telhado (a planta do OSM)
    wb = float(np.clip(Wt - 0.7, 1.0, 1.6)) / 2               # meia largura do corpo
    lb = L / 2 - 0.25                                         # meio comprimento do corpo
    cume = e["cume"]
    alt_t = (wb + 0.3) * math.tan(math.radians(27))           # altura do telhado (lajes a 27°)
    z_mesa = e["chao_max"] + 0.95                             # face de cima das mesas
    z_sol = z_mesa + 0.2                                      # soleira
    z_beir = max(cume - alt_t, z_sol + 1.35)
    cume = z_beir + alt_t
    G, GR, LJ, MD, SB = 0, 1, 2, 3, 4                         # granito, grade, lajes, madeira, sombra
    # pés e mesas aos pares
    npar = max(2, int(round(2 * lb / 1.7)) + 1)
    for s in np.linspace(-lb + 0.3, lb - 0.3, npar):
        for d in (-wb + 0.18, wb - 0.18):
            p = f.xy(s, d)
            zc = float(cota(*p))
            m.caixa_face(f, s - 0.15, s + 0.15, d - 0.15, d + 0.15, zc - 0.4, z_mesa - 0.2, G)
            m.cilindro(p, 0.38, z_mesa - 0.2, z_mesa - 0.08, G, k=8, r1=0.46)
            m.cilindro(p, 0.46, z_mesa - 0.08, z_mesa, G, k=8)
            m.caixa_face(f, s - 0.14, s + 0.14, d - 0.14, d + 0.14, z_mesa, z_sol - 0.18, G)  # calço
    # soleira (lajes do fundo) a todo o comprimento
    m.caixa_face(f, -lb - 0.08, lb + 0.08, -wb - 0.08, wb + 0.08, z_sol - 0.2, z_sol, G)
    # lados compridos: esteios e grades
    nst = max(2, int(math.ceil(2 * lb / 0.95)))
    ss = np.linspace(-lb, lb, nst + 1)
    lp = 0.17
    for lado in (-1, 1):
        d0, d1 = (wb - 0.16, wb) if lado > 0 else (-wb, -wb + 0.16)
        for k_, s in enumerate(ss):
            a_ = max(s - lp / 2, -lb)
            b_ = min(s + lp / 2, lb)
            m.caixa_face(f, a_, b_, d0, d1, z_sol, z_beir - 0.2, G)
        for a_, b_ in zip(ss[:-1], ss[1:]):
            # grade recuada 4 cm, com a moldura de cima e de baixo em granito
            dd = (wb - 0.14, wb - 0.04) if lado > 0 else (-wb + 0.04, -wb + 0.14)
            m.caixa_face(f, a_ + lp / 2, b_ - lp / 2, dd[0], dd[1], z_sol + 0.12, z_beir - 0.32, GR)
            m.caixa_face(f, a_ + lp / 2, b_ - lp / 2, d0, d1, z_sol, z_sol + 0.12, G)
            m.caixa_face(f, a_ + lp / 2, b_ - lp / 2, d0, d1, z_beir - 0.32, z_beir - 0.2, G)
    # interior escuro (vê-se pelas frestas)
    m.caixa_face(f, -lb + 0.15, lb - 0.15, -wb + 0.14, wb - 0.14, z_sol, z_beir - 0.2, SB)
    # padieira (viga de cima) a todo o comprimento
    m.caixa_face(f, -lb, lb, -wb, wb, z_beir - 0.2, z_beir, G)
    # empenas de granito nos topos, com a porta em arco numa delas
    porta = 1 if i % 2 else -1
    for topo in (-1, 1):
        s0, s1 = (lb - 0.2, lb) if topo > 0 else (-lb, -lb + 0.2)
        m.caixa_face(f, s0, s1, -wb, wb, z_sol, z_beir, G)
        sf = s1 if topo > 0 else s0
        pts = [(-wb - 0.02, z_beir), (wb + 0.02, z_beir), (0.0, cume - 0.05)]
        for sx in (s0, s1):
            m.face([(*f.xy(sx, d), z) for d, z in (pts if (sx == s1) == (topo > 0) else pts[::-1])], G)
        for (d1_, z1_), (d2_, z2_) in ((pts[1], pts[2]), (pts[2], pts[0])):
            m.face([(*f.xy(s0, d1_), z1_), (*f.xy(s1, d1_), z1_), (*f.xy(s1, d2_), z2_), (*f.xy(s0, d2_), z2_)], G)
        if topo == porta:
            fe = Plano(f.xy(sf, 0), (f.n if topo > 0 else -f.n), n=(t if topo > 0 else -t))
            wd = min(0.62, 2 * wb - 0.45)
            zb_ = z_sol + 0.08
            zt_ = min(z_beir - 0.12, zb_ + 1.15)
            m.face([fe.p(s_, 0.012, z_) for s_, z_ in contorno_vao(0.0, wd, zb_, zt_, True, 10)], MD)
            for s_ in np.linspace(-wd / 2 + 0.1, wd / 2 - 0.1, 4):
                m.face([fe.p(s_ - 0.008, 0.016, zb_ + 0.05), fe.p(s_ + 0.008, 0.016, zb_ + 0.05),
                        fe.p(s_ + 0.008, 0.016, zt_ - wd / 2 - 0.02), fe.p(s_ - 0.008, 0.016, zt_ - wd / 2 - 0.02)], SB)
    # telhado de lajes: duas águas com beiral de 0,3 m nos lados e 0,15 m nos topos, e o cume
    ov, ot, esp = 0.3, 0.15, 0.09
    for lado in (-1, 1):
        d_b = lado * (wb + ov)
        z_b = cume - (wb + ov) * math.tan(math.radians(27))
        a = [f.p(-lb - ot, 0.0, cume), f.p(lb + ot, 0.0, cume), f.p(lb + ot, d_b, z_b), f.p(-lb - ot, d_b, z_b)]
        b = [(p[0], p[1], p[2] + esp) for p in a]
        if lado < 0:
            a, b = a[::-1], b[::-1]
        m.face(b, LJ)
        m.face(a[::-1], LJ)
        for k_ in range(4):
            p, q = k_, (k_ + 1) % 4
            m.face([a[p], a[q], b[q], b[p]], LJ)
    m.caixa_face(f, -lb - ot - 0.02, lb + ot + 0.02, -0.08, 0.08, cume + esp - 0.02, cume + esp + 0.08, G)  # cumeeira
    # cruz num topo e pináculo no outro
    zc_ = cume + esp + 0.08
    sc_ = -porta * (lb + ot - 0.08)
    m.caixa_face(f, sc_ - 0.05, sc_ + 0.05, -0.05, 0.05, zc_, zc_ + 0.55, G)
    m.caixa_face(f, sc_ - 0.05, sc_ + 0.05, -0.17, 0.17, zc_ + 0.30, zc_ + 0.40, G)
    sp_ = porta * (lb + ot - 0.1)
    m.caixa_face(f, sp_ - 0.09, sp_ + 0.09, -0.09, 0.09, zc_, zc_ + 0.2, G)
    m.piramide([f.xy(sp_ - 0.09, -0.09), f.xy(sp_ + 0.09, -0.09), f.xy(sp_ + 0.09, 0.09), f.xy(sp_ - 0.09, 0.09)],
               zc_ + 0.2, zc_ + 0.48, G)
    return {"c": [round(float(c[0]), 2), round(float(c[1]), 2)], "L": round(L, 2), "W": round(Wt, 2),
            "az": round(float(e["az"]), 1), "cume": round(cume, 2), "fonte": e["fonte"]}


def espigueiros():
    lst = lista_espigueiros()
    m = Malha(["granito_esp", "grade_esp", "lajes_esp", "madeira", "sombra"])
    out = [espigueiro(m, e, i) for i, e in enumerate(lst)]
    m.objeto("espigueiros", "espigueiros", GRUPOS["Espigueiros"], uv=1.5, vcor=patine_esp)
    print("espigueiros", len(out), "(OSM", sum(1 for e in out if e["fonte"] == "OSM"), ")")
    MEDIDAS["espigueiros"] = {"total": len(out), "osm": sum(1 for e in out if e["fonte"] == "OSM"), "lista": out}
    return lst


# ---------------------------------------------------------------- terreno e envolvente

def caminhos():
    out_a, out_t = [], []
    for f in F["caminhos"]:
        if f["geometria"]["type"] != "LineString":
            continue
        cl = f["classe"]
        larg = {"residential": 5.0, "tertiary": 6.0, "secondary": 6.5, "unclassified": 5.0, "service": 3.5,
                "living_street": 4.0, "pedestrian": 3.0, "unknown": 3.0, "track": 2.5}.get(cl, 1.8)
        g = shapely.LineString(f["geometria"]["coordinates"]).buffer(larg / 2, cap_style="flat")
        (out_a if cl in ("residential", "tertiary", "secondary", "unclassified", "living_street", "service") else out_t).append(g)
    return (shapely.union_all(out_a) if out_a else shapely.Polygon()), (shapely.union_all(out_t) if out_t else shapely.Polygon())


def monumento_pl():
    """Planta de tudo o que é monumento (para tirar casas e árvores de cima dele)."""
    torre_pl = shapely.Polygon([W(-A - 1, -A - 1), W(A + 1, -A - 1), W(A + 1, A + 1), W(-A - 1, A + 1)])
    return shapely.union_all([POL_MUROS.buffer(0.6), torre_pl, shapely.Polygon(ZONA_CASTELO).buffer(0.0),
                              shapely.Polygon(TORRE_DA_PORTA).buffer(0.5)])


def construir_terreno(esp):
    passo = 0.5
    xs = np.arange(X0, X1 + 1e-6, passo)
    ys = np.arange(Y0, Y1 + 1e-6, passo)
    X, Y = np.meshgrid(xs, ys)
    Z = cota(X, Y)
    pts_all = shapely.points(X.ravel(), Y.ravel())
    zona = shapely.contains(shapely.Polygon(ZONA_CASTELO), pts_all).reshape(X.shape)
    mur = shapely.contains(POL_MUROS, pts_all).reshape(X.shape)
    bloco_pn = _bloco(PORTA_NORTE["c"], PORTA_NORTE["az"], PORTA_NORTE["s0"], PORTA_NORTE["s1"], PORTA_NORTE["meia"], 0.6)
    excl = shapely.union_all([shapely.Polygon(p) for p in (ALA_OESTE, ALA_SUL, CORREDOR, TORRE_DA_PORTA, PASSAGEM_SUL, bloco_pn)] +
                             [shapely.Polygon([W(-A - 0.5, -A - 0.5), W(A + 0.5, -A - 0.5), W(A + 0.5, A + 0.5), W(-A - 0.5, A + 0.5)])])
    exc = shapely.contains(excl.buffer(0.5), pts_all).reshape(X.shape)
    # plataformas dos baluartes (terraplenos que o MDT não tem como chão): o LiDAR das classes solo e muros
    zs = ndi.median_filter(superficie(X, Y), 3)
    plat = zona & ~mur & ~exc & (zs - Z > 0.6)
    plat = ndi.binary_opening(plat, iterations=1)
    Z = np.where(plat, zs, Z)
    # junto às faces dos muros, o MDT sobe para o topo (interpolação por baixo do muro): baixa-se o chão até
    # 1,25 m dos muros para o mínimo das células à volta que não são muro
    banda = ndi.binary_dilation(mur, iterations=3) & ~mur & ~plat
    zl = np.where(mur, 1e4, Z)
    zmin = ndi.minimum_filter(zl, size=5)
    Z = np.where(banda, np.minimum(Z, np.maximum(zmin, Z - 1.2)), Z)
    ny, nx = X.shape
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    idx = np.arange(ny * nx).reshape(ny, nx)
    Fq = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    mats = ["erva", "relva", "rocha", "terra", "asfalto", "calcada", "terreiro"]
    ob = malha_rapida("Terreno", V, Fq, mats[0], GRUPOS["Terreno"], "terreno")
    for mt in mats[1:]:
        ob.data.materials.append(MATS[mt])
    cx = V[Fq].mean(1)
    pts = shapely.points(cx[:, 0], cx[:, 1])
    mi_ = np.zeros(len(Fq), np.int32)
    ci = np.clip(((cx[:, 0] - OX0) / OP).astype(int), 0, ORTO.shape[1] - 1)
    ri = np.clip(((OY1 - cx[:, 1]) / OP).astype(int), 0, ORTO.shape[0] - 1)
    o = ndi.uniform_filter(ORTO, size=(7, 7, 1))[ri, ci]
    verde = (o[:, 1] > o[:, 0] * 1.03) & (o[:, 1] > o[:, 2] * 1.08)
    sat = o.max(1) - o.min(1)
    cinza = (sat < 0.07) & (o.mean(1) > 0.38)
    # afloramentos de granito: relevo local positivo do MDT ou pedra cinzenta no ortofotomapa
    rel = ndi.gaussian_filter(MDT - ndi.gaussian_filter(MDT, 3.0), 0.7)
    rl = amostra(rel, LX0, LY0, LP, cx[:, 0], cx[:, 1])
    mi_[verde] = 1
    mi_[(rl > 0.12) | (cinza & (rl > 0.0))] = 2
    asf, trilho = caminhos()
    if not trilho.is_empty:
        mi_[shapely.contains(trilho, pts)] = 3
    if not asf.is_empty:
        mi_[shapely.contains(asf, pts)] = 4
    # pátio do castelo: relvado (fotografias de drone), com os afloramentos
    patio = shapely.Polygon(PATIO)
    no_patio = shapely.contains(patio, pts)
    mi_[no_patio] = 1
    mi_[no_patio & (rl > 0.12)] = 2
    ob.data.polygons.foreach_set("material_index", mi_)
    uv_mundo(ob, 2.0)
    m = Malha(["base"])
    zb = float(Z.min()) - 1.5
    borda = [(X[0, i], Y[0, i], Z[0, i]) for i in range(nx)] + [(X[i, -1], Y[i, -1], Z[i, -1]) for i in range(1, ny)] + \
            [(X[-1, i], Y[-1, i], Z[-1, i]) for i in range(nx - 2, -1, -1)] + [(X[i, 0], Y[i, 0], Z[i, 0]) for i in range(ny - 2, 0, -1)]
    for a, b in zip(borda, borda[1:] + borda[:1]):
        m.face([(a[0], a[1], zb), (b[0], b[1], zb), b, a], 0)
    m.objeto("base", None, GRUPOS["Terreno"], uv=None)
    MEDIDAS["terreno"] = {"plataformas_m2": round(float(plat.sum()) * passo * passo)}


# o pátio, por dentro da cerca medieval (relvado)
PATIO = [(-12.8, 2.0), (-2.5, 9.0), (13.5, 7.5), (17.5, -6.0), (14.0, -19.5), (8.0, -25.5), (-6.5, -28.0), (-12.5, -10.0)]
def casas_lidar(mon, osm):
    """Plantas das casas que o OSM não tem (à volta do castelo faltam muitas): manchas da classe 6 do LiDAR (edifícios)
    com mais de 2,5 m acima do chão, partidas onde o telhado salta mais de 1,5 m (casas de alturas diferentes)."""
    from rasterio import features
    from rasterio.transform import Affine
    h = MDS - MDT
    ed = (CLS == 6) & (h > 2.5)
    ed = ndi.binary_opening(ed, iterations=1)
    salto = (ndi.maximum_filter(MDS, 3) - ndi.minimum_filter(MDS, 3)) > 1.5
    nucleo = ed & ~salto
    lab, k = ndi.label(nucleo)
    # as células de salto vão para a casa vizinha mais próxima
    _, (ii, jj) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    lab = np.where(ed, lab[ii, jj], 0)
    tr = Affine(LP, 0, LX0 - LP / 2, 0, LP, LY0 - LP / 2)
    out = []
    excl = shapely.union_all([mon.buffer(0.8), osm.buffer(0.6)])
    caixa = shapely.box(X0 + 0.3, Y0 + 0.3, X1 - 0.3, Y1 - 0.3)
    for geo, val in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=tr):
        p = shapely.geometry.shape(geo).buffer(0)
        p = p.intersection(caixa).difference(excl)
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
    cornija = float(np.percentile(zs, 25))
    if cume - chao < 2.5:
        return False
    cornija = min(max(cornija, chao + 2.6), cume - 0.4)
    base = np.array(p.exterior.coords)[:-1]
    if shapely.Polygon(base).exterior.is_ccw is False:
        base = base[::-1]
    rec = (cume - cornija) / math.tan(math.radians(28))
    top = p.buffer(-rec, join_style="mitre")
    if top.is_empty or top.geom_type != "Polygon" or rec < 0.3 or p.convex_hull.area > 1.25 * p.area or len(base) > 14:
        # plantas recortadas ou muito irregulares: volume de cobertura plana a meia altura do telhado
        zc = cornija + 0.5 * (cume - cornija)
        m.prisma(base, chao - 0.8, zc, 0, topo=False)
        m.face([(*q, zc) for q in base], 0)
        return True
    m.prisma(base, chao - 0.8, cornija, 0, topo=False)
    tb = np.array(top.exterior.coords)[:-1]
    if not shapely.Polygon(tb).exterior.is_ccw:
        tb = tb[::-1]
    kd = cKDTree(tb)
    _, j = kd.query(base)
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


def construir_envolvente(mon, esp_pl):
    """Casas à volta em volumes simples: plantas do OSM e, onde o OSM não as tem, do LiDAR; recortadas pela maqueta
    e pelo monumento; cornija e cume medidos no MDS, telhado de águas por recuo da planta. Um só cinzento."""
    caixa = shapely.box(X0 + 0.3, Y0 + 0.3, X1 - 0.3, Y1 - 0.3)
    m = Malha(["cinza_viz"])
    plantas = []
    tirar = shapely.union_all([mon, esp_pl])
    for f in F["edificios"]:
        g = shapely.geometry.shape(f["geometria"])
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
        if g.geom_type != "Polygon" or g.area < 6:
            continue
        g = g.buffer(0)
        if g.intersection(tirar).area > 0.4 * g.area:
            continue  # a torre, os muros e os espigueiros também estão no OSM como edifícios
        g = g.intersection(caixa).difference(tirar)
        for p in (g.geoms if g.geom_type in ("MultiPolygon", "GeometryCollection") else [g]):
            if p.geom_type == "Polygon" and p.area >= 6:
                plantas.append(p.simplify(0.3))
    osm = shapely.union_all(plantas) if plantas else shapely.Polygon()
    n_osm = len(plantas)
    plantas += casas_lidar(tirar, osm)
    n = sum(_casa(m, p) for p in plantas)
    print("casas", n, "(OSM", n_osm, ")")
    MEDIDAS["casas"] = {"total": n, "osm": n_osm}
    m.objeto("envolvente", "envolvente", GRUPOS["Envolvente"], uv=2.0, recalc=False)


def construir_arvores(mon):
    """Árvores: máximos locais da vegetação (classes 3 a 5) dentro da maqueta, fora do monumento; copas
    esquemáticas."""
    h = MDS - MDT
    veg = (CLS >= 3) & (CLS <= 5) & (h > 2.0)
    veg = ndi.binary_opening(veg, iterations=1)
    hs = ndi.gaussian_filter(np.where(veg, h, 0.0), 1.2)
    topo = (hs == ndi.maximum_filter(hs, size=13)) & veg & (hs > 3.0)
    lab, k = ndi.label(topo)
    V, Fc, C = [], [], []
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    v0 = np.array([x.co[:] for x in bm.verts])
    f0 = [[x.index for x in fa.verts] for fa in bm.faces]
    bm.free()
    mon_l = mon.buffer(1.0)
    n = 0
    for (r, c) in ndi.center_of_mass(topo, lab, range(1, k + 1)):
        r, c = int(round(r)), int(round(c))
        x, y = LX0 + c * LP, LY0 + r * LP
        if not (X0 + 2 < x < X1 - 2 and Y0 + 2 < y < Y1 - 2):
            continue
        if mon_l.contains(shapely.Point(x, y)):
            continue  # pontos de vegetação sobre o monumento (muros com ervas, a torre): não são árvores
        zt, zc = MDT[r, c], MDS[r, c]
        alt = zc - zt
        rad = float(np.clip(np.sqrt(veg[max(r - 10, 0):r + 11, max(c - 10, 0):c + 11].sum() * 0.25 / math.pi), 1.3, 5.5))
        for (dx, dy, dz, s) in ((0, 0, -0.35, 1.0), (0.4, 0.2, -0.5, 0.7), (-0.35, -0.3, -0.55, 0.7), (0.1, -0.4, -0.3, 0.6)):
            b = len(V)
            pts = (v0 + rng.normal(0, 0.06, v0.shape)) * np.array([rad * s, rad * s, alt * 0.30 * s]) + np.array([x + dx * rad, y + dy * rad, zc + dz * alt * 0.45])
            V.extend(map(tuple, pts))
            Fc.extend([[b + i for i in fa] for fa in f0])
            tt = np.clip((pts[:, 2] - zt) / alt, 0, 1)
            C.extend(map(tuple, np.array([0.22, 0.32, 0.14])[None, :] * (0.55 + 0.5 * tt)[:, None]))
        b = len(V)
        for j in range(6):
            a = 2 * math.pi * j / 6
            V.append((x + 0.22 * math.cos(a), y + 0.22 * math.sin(a), zt - 0.2))
            V.append((x + 0.16 * math.cos(a), y + 0.16 * math.sin(a), zc - alt * 0.45))
        for j in range(6):
            jj = (j + 1) % 6
            Fc.append([b + 2 * j, b + 2 * jj, b + 2 * jj + 1, b + 2 * j + 1])
        C.extend([(0.36, 0.30, 0.24)] * 12)
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
    at.data.foreach_set("color", np.concatenate([np.array(C), np.ones((len(C), 1))], 1).astype(np.float32).ravel())
    me.shade_smooth()
    ob = bpy.data.objects.new("arvores", me)
    bpy.context.scene.collection.objects.link(ob)
    ob["elemento"] = "vegetacao"
    ob.parent = GRUPOS["Vegetacao"]



# ---------------------------------------------------------------- construção

print("muros…")
muros()
print("torre…")
torre()
bandeira()
F_PORTA, Z_PORTA = torre_porta()
F_PN = porta_norte()
PG_O, PG_Z = postigo()
guaritas()
print("museu…")
museu()
print("espigueiros…")
ESP = espigueiros()
ESP_PL = shapely.union_all([shapely.Point(e["c"]).buffer(max(e["L"], 3.0) / 2 + 1.0) for e in ESP]) if ESP else shapely.Polygon()
print("terreno…")
construir_terreno(ESP)
print("envolvente…")
MON = monumento_pl()
construir_envolvente(MON, ESP_PL)
construir_arvores(shapely.union_all([MON, ESP_PL]))


def _anc(x, y, z):
    return [round(float(x), 1), round(float(z), 1), round(float(-y), 1)]


def _olhar(x, y, up=0.35):
    d = unit((x, y))
    return [round(float(d[0]), 2), up, round(float(-d[1]), 2)]


_le = MEDIDAS["espigueiros"]["lista"]
_ce = np.mean([e["c"] for e in _le], 0) if _le else np.array([-50.0, -110.0])
_esp1 = min(_le, key=lambda e: np.hypot(e["c"][0] - _ce[0], e["c"][1] - _ce[1])) if _le else {"c": list(_ce), "cume": 450.0}
MEDIDAS["ancoras"] = {
    "torre": _anc(0, 0, Z_PONTA + 1.2),
    "coroamento": _anc(*W(A + 0.6, -A - 0.6), Z_PONTA - 0.3),
    "cerca": _anc(-13.5, -14.0, 469.0),
    "porta_sul": _anc(*F_PORTA.xy(0, 1.0), Z_PORTA + 1.0),
    "porta_norte": _anc(*F_PN.xy(0, 0.8), MEDIDAS["porta_norte"]["topo"] + 1.0),
    "postigo": _anc(*PG_O, PG_Z + 2.6),
    "guaritas": _anc(*MEDIDAS["guaritas"][0][:2], MEDIDAS["guaritas"][0][2] + 3.4),
    "baluartes": _anc(32.0, 0.0, 466.0),
    "museu": _anc(-4.0, -24.0, MEDIDAS["museu"]["ala_sul"]["cume"] + 1.0),
    "espigueiros": _anc(_esp1["c"][0], _esp1["c"][1], _esp1["cume"] + 1.2),
}
MEDIDAS["olhar"] = {
    "coroamento": _olhar(*W(1, -1), 0.25),
    "cerca": _olhar(-1, 0.1, 0.35),
    "porta_sul": _olhar(*F_PORTA.n, 0.25),
    "porta_norte": _olhar(*F_PN.n, 0.2),
    "postigo": _olhar(*W(0, 1), 0.2),
    "guaritas": _olhar(0.7, -0.7, 0.3),
    "baluartes": _olhar(1, 0.2, 0.5),
    "museu": _olhar(0.3, 1, 0.6),
    "espigueiros": _olhar(0.2, -1, 0.45),
}
(AQUI / "dados" / "medidas.json").write_text(json.dumps(MEDIDAS, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in MEDIDAS.items() if k not in ("ancoras", "olhar", "espigueiros")}, ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
(AQUI / "saida").mkdir(exist_ok=True)
glb = AQUI / "saida" / "lindoso-bruto.glb"
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
    final = SAIDA / "lindoso.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "lindoso.blend"))

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
    AZ_SOL = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--sol=")), 200))
    d = Vector((math.sin(math.radians(AZ_SOL)), math.cos(math.radians(AZ_SOL)), 0)) * math.cos(math.radians(48))
    d.z = math.sin(math.radians(48))
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

    # capa: de sudoeste, do alto, com o castelo atrás e os espigueiros à frente
    vista("capa", (-95, -150), 512, (-12, -40), 458, lente=34, res=(1280, 800), destino=SAIDA / "capa.jpg")
    if "--capa" not in sys.argv:
        # f296: drone a sul, a olhar para norte
        vista("f296_sul", (2, -72), 496, (2, -8), 462, lente=30, res=(1030, 776))
        # 73c31f43: drone de norte-nordeste, a olhar para sul (espigueiros ao fundo)
        vista("73c3_norte", (22, 58), 505, (-8, -22), 458, lente=30, res=(1356, 904))
        # 5a85: do alto, de sudoeste
        vista("5a85_sudoeste", (-60, -60), 495, (0, -8), 463, lente=34, res=(1200, 800))
        # 26f57b11: a torre do pátio (face SSE com a janela)
        vista("26f5_torre", tuple(W(1.0, -16.0)), 463.4, tuple(W(0.0, 0.0)), 469.0, lente=22, res=(800, 1100), perto=0.3)
        # d0492f83: Porta Norte de frente, com a torre por trás
        _pn = F_PN.xy(0.0, 0.0)
        _ln = unit(_pn)
        vista("d049_porta_norte", tuple(_pn + _ln * 4.2), 462.4, tuple(_pn - _ln * 2.0), 467.2, lente=24, res=(780, 1040), perto=0.3)
        # image 8: cunhal SE com a guarita, de baixo
        g0 = MEDIDAS["guaritas"][0]
        vista("guarita_se", (g0[0] - 3.0, g0[1] - 12.0), 462.5, (g0[0] - 2.0, g0[1]), 468.0, lente=24, res=(1000, 700), perto=0.3)
        # torre da porta, de sul
        vista("porta_sul", tuple(F_PORTA.xy(0.0, 16.0)), 468.5, tuple(F_PORTA.xy(0.0, 0.0)), 464.5, lente=26, res=(1000, 750), perto=0.3)
        # espigueiros (33d8 e d8462)
        e0 = _esp1
        vista("33d8_espigueiro", (e0["c"][0] - 6.0, e0["c"][1] + 5.0), e0["cume"] - 2.2, tuple(e0["c"]), e0["cume"] - 1.4, lente=24, res=(900, 500), perto=0.3)
        vista("d846_espigueiros", (_ce[0] + 4, _ce[1] + 38), _esp1["cume"] + 14, (_ce[0], _ce[1] - 5), _esp1["cume"] - 3, lente=30, res=(1280, 960))
        # planta (ortográfica, norte para cima), para comparar com o ortofotomapa
        co.location = Vector(((X0 + X1) / 2, (Y0 + Y1) / 2, 900))
        co.rotation_euler = (0, 0, 0)
        cam.type = "ORTHO"
        cam.ortho_scale = Y1 - Y0
        sc.render.resolution_x, sc.render.resolution_y = int(1100 * (X1 - X0) / (Y1 - Y0)), 1100
        sc.render.filepath = str(pasta / "topo.jpg")
        if not SO or "topo" in SO:
            bpy.ops.render.render(write_still=True)
