"""Constrói a maqueta 3D da Fortaleza de Peniche (estado atual) em Blender e exporta-a para
public/maquetas/peniche/peniche.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow
    python pipeline/maquetas/peniche/construir.py            # GLB
    python pipeline/maquetas/peniche/construir.py --vistas   # GLB + capa + vistas de verificação

Dados: dados/fontes.json (recolher.py: OSM via Overture), dados/lidar.npz e dados/orto.jpg (lidar.py: MDT, MDS,
classes e cor da nuvem de pontos do LiDAR 2024 da DGT), dados/plataformas.json (plataformas.py: a massa
construída partida em plataformas) e src/data/maquetas/peniche.json, com a leitura das fontes.

A fortaleza assenta num promontório de calcário sobre o mar. Na nuvem de pontos da DGT as muralhas, os
terraplenos e os edifícios da antiga prisão vêm todos como "edifício" (classe 6), por isso o MDT passa por
baixo deles e não serve para os traçar. Aqui o relevo vem do MDT (rochas, praça, fossos) e a massa construída
das plataformas do MDS: cada uma é levantada com o topo medido e panos de alvenaria (ou paredes caiadas, nos
edifícios) até ao chão. Por cima vão os parapeitos com merlões, as guaritas, o Baluarte Redondo, a ponte de
arcos, a porta e os edifícios com janelas, pelas fotografias. O mar é um plano à cota 0,4 m, com a cor pela
fundura, espuma junto às rochas e rochedos soltos na linha de água.
Coordenadas da cena: x para este, y para norte, z para cima, em metros (1:1). O glTF converte para y para
cima (-z é o norte no visor).
"""
import json
import math
import sys
from pathlib import Path

import bpy  # noqa: I001 (o bpy tem de vir antes do bmesh)
import bmesh
import numpy as np
import shapely
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pipeline/, para o kit comum
from kit.geometria import unit, area, densificar, dentro, simplificar, mediana_movel, partir, _retangulo, _ang  # noqa: E402
from kit.texturas import ruido, fbm, textura_grao, imagem  # noqa: E402
from kit.cena import arvore_modelo  # noqa: E402
from kit.fotos import textura_foto, fontes_fotos  # noqa: E402
from kit.arquitetura import Quadro, _vao_desloc, altura_telhado, barra, cupula, faixa_inclinada, guarita, oitavado, porta_revelim, telhado_4aguas, telhado_generico  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "peniche.json").read_text())
PLAT = json.loads((AQUI / "dados" / "plataformas.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "peniche"

LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([0.0, 0.0])  # centro da grelha do LiDAR (igual ao lidar.py)
ZREF = 0.0  # as cotas da cena são as altitudes (m)
rng = np.random.default_rng(1557)


# ---------------------------------------------------------------- geografia

def P(x, y):
    """Metros a partir da origem -> coordenadas da cena."""
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


LID = np.load(AQUI / "dados" / "lidar.npz")
LPASSO = float(LID["passo"])
MDT = LID["mdt"] / 100.0 + float(LID["base"]) - ZREF
MDS = LID["mds"] / 100.0 + float(LID["base"]) - ZREF
CLS = LID["classe"]
VEG = (CLS >= 3) & (CLS <= 5)
LXS = np.linspace(-float(LID["meiox"]), float(LID["meiox"]), MDT.shape[1])
LYS = np.linspace(-float(LID["meioy"]), float(LID["meioy"]), MDT.shape[0])


def amostra(G, x, y):
    """Interpolação bilinear numa grelha do LiDAR."""
    c = (np.asarray(x, float) - LXS[0]) / LPASSO
    r = (np.asarray(y, float) - LYS[0]) / LPASSO
    c0 = np.clip(np.floor(c).astype(int), 0, G.shape[1] - 2)
    r0 = np.clip(np.floor(r).astype(int), 0, G.shape[0] - 2)
    fc, fr = np.clip(c - c0, 0, 1), np.clip(r - r0, 0, 1)
    return (G[r0, c0] * (1 - fc) * (1 - fr) + G[r0, c0 + 1] * fc * (1 - fr)
            + G[r0 + 1, c0] * (1 - fc) * fr + G[r0 + 1, c0 + 1] * fc * fr)


def cota(x, y):
    """Terreno: MDT LiDAR de 50 cm (solo sem edifícios nem vegetação)."""
    return amostra(MDT, x, y)


def celulas(poly=None, centro=None, raio=None, sem_veg=True, G=None):
    G = MDS if G is None else G
    if poly is not None:
        x0, y0 = poly.min(0) - 1
        x1, y1 = poly.max(0) + 1
    else:
        x0, y0 = np.asarray(centro) - raio - 1
        x1, y1 = np.asarray(centro) + raio + 1
    c0, c1 = np.searchsorted(LXS, [x0, x1])
    r0, r1 = np.searchsorted(LYS, [y0, y1])
    X, Y = np.meshgrid(LXS[c0:c1], LYS[r0:r1])
    ok = dentro(X, Y, poly) if poly is not None else np.hypot(X - centro[0], Y - centro[1]) <= raio
    if sem_veg:
        ok &= ~VEG[r0:r1, c0:c1]
    return G[r0:r1, c0:c1][ok]


def medir(poly=None, centro=None, raio=None, q=95, sem_veg=True):
    v = celulas(poly, centro, raio, sem_veg)
    return float(np.percentile(v, q)) if len(v) >= 3 else float("nan")


def distancia(px, py, linha, fechada=False):
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
        esc = 1.0 / max(0.5, float(nm @ n1))
        out.append(linha[i] + nm * (d[i] if np.ndim(d) else d) * esc)
    return np.array(out)


# ---------------------------------------------------------------- ruído


# ---------------------------------------------------------------- malhas

MATS = {}


def uv_mundo(ob, escala):
    """UV em metros do mundo: faces horizontais em planta, verticais ao longo da face (as fiadas não
    esticam nos panos oblíquos)."""
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
    ls = np.empty(len(me.polygons), int)
    me.polygons.foreach_get("loop_start", ls)
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

    def __init__(self):
        self.v, self.f, self.m = [], [], []

    def face(self, pts, mat=0):
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

    def caixa(self, c, u, largura, prof, z0, z1, mat=0, fundo=False):
        """Caixa centrada em c, com eixo u (unitário) no plano, largura ao longo de u."""
        u = unit(np.asarray(u, float))
        w = np.array([-u[1], u[0]])
        c = np.asarray(c, float)
        hl, hp = largura / 2, prof / 2
        q = [c - u * hl - w * hp, c + u * hl - w * hp, c + u * hl + w * hp, c - u * hl + w * hp]
        self.prisma(q, z0, z1, mat, topo=True, fundo=fundo)

    def cilindro(self, c, r, z0, z1, mat=0, k=12, topo=True, r1=None):
        """Cilindro (ou tronco de cone, com r1) vertical."""
        r1 = r if r1 is None else r1
        a = np.linspace(0, 2 * math.pi, k, endpoint=False)
        b0 = [(c[0] + r * math.cos(t), c[1] + r * math.sin(t), z0) for t in a]
        b1 = [(c[0] + r1 * math.cos(t), c[1] + r1 * math.sin(t), z1) for t in a]
        for i in range(k):
            j = (i + 1) % k
            self.face([b0[i], b0[j], b1[j], b1[i]], mat)
        if topo and r1 > 0.01:
            self.face(b1, mat)

    def esfera(self, c, r, mat=0, k=10, achata=1.0):
        """Bola (pináculos e remates) por anéis."""
        anel = []
        for i in range(1, k // 2):
            fi = math.pi * i / (k // 2)
            anel.append([(c[0] + r * math.sin(fi) * math.cos(t), c[1] + r * math.sin(fi) * math.sin(t), c[2] + r * achata * math.cos(fi))
                         for t in np.linspace(0, 2 * math.pi, k, endpoint=False)])
        topo, fundo = (c[0], c[1], c[2] + r * achata), (c[0], c[1], c[2] - r * achata)
        for j in range(k):
            jj = (j + 1) % k
            self.face([anel[0][j], anel[0][jj], topo][::-1], mat)
            self.face([anel[-1][jj], anel[-1][j], fundo][::-1], mat)
            for a, b in zip(anel[:-1], anel[1:]):
                self.face([a[j], b[j], b[jj], a[jj]][::-1], mat)

    def objeto(self, nome, elemento, mats, pai=None, uv=3.0, liso=False, recalc=True):
        me = bpy.data.meshes.new(nome)
        me.from_pydata(self.v, [], self.f)
        me.update()
        for m in mats:
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
        ob.parent = pai or RAIZ_OB
        if uv:
            uv_mundo(ob, uv)
        return ob


def malha_rapida(nome, V, Fq, mat, pai, elemento=None):
    """Malha grande (terreno) a partir de arrays: V (n×3) e quadriláteros Fq (m×4)."""
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


def material(nome, cor, rug=0.9, tex=None, vcor=False, normal=None, forca=1.0, uv_nome=None):
    mt = bpy.data.materials.new(nome)
    mt.use_nodes = True
    nt = mt.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    if normal is not None:
        normal.colorspace_settings.name = "Non-Color"
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = normal
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = forca
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
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


def normais(alt, forca):
    """Mapa de normais (espaço tangente, convenção do glTF) a partir de um mapa de alturas periódico."""
    gx = (np.roll(alt, -1, 1) - np.roll(alt, 1, 1)) * forca
    gy = (np.roll(alt, -1, 0) - np.roll(alt, 1, 0)) * forca
    nv = np.stack([-gx, -gy, np.ones_like(alt)], 2)
    nv /= np.linalg.norm(nv, axis=2, keepdims=True)
    return nv * 0.5 + 0.5


def textura_alvenaria(n=512, metros=4.0, semente=1661):
    """Alvenaria de calcário das muralhas (4 m por repetição), como nas fotografias: pedras de tamanho
    desigual em fiadas pouco regulares, cinzento-amareladas com manchas ferrugem e de líquenes, e
    escorrências escuras. Devolve a cor e o relevo (para o mapa de normais)."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    tex = np.zeros((n, n, 3))
    alt = np.zeros((n, n))
    y = 0
    while y < n:
        hf = int(r.uniform(0.22, 0.42) * m_px)
        x = -int(r.uniform(0, 0.5) * m_px)
        while x < n:
            wf = int(r.uniform(0.3, 0.8) * m_px)
            tom = r.choice(4, p=[0.35, 0.3, 0.2, 0.15])
            base = np.array([(0.62, 0.58, 0.49), (0.70, 0.65, 0.54), (0.55, 0.52, 0.45), (0.71, 0.60, 0.44)][tom])
            base = base * r.uniform(0.9, 1.08) + r.normal(0, 0.008, 3)
            x0, x1 = max(x, 0), min(x + wf, n)
            # as pedras não são retângulos perfeitos: a altura varia um pouco de uma para outra
            y0 = min(y + int(r.uniform(-0.03, 0.04) * m_px), n - 1)
            y1 = min(y + hf + int(r.uniform(-0.04, 0.03) * m_px), n)
            y0 = max(y0, 0)
            if x1 > x0 and y1 > y0:
                tex[y0:y1, x0:x1] = base
                yy, xx = np.mgrid[y0:y1, x0:x1]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                bomb = 1 - (((xx - cx) / max((x1 - x0) / 2, 1)) ** 2 + ((yy - cy) / max((y1 - y0) / 2, 1)) ** 2) * 0.5
                alt[y0:y1, x0:x1] = np.maximum(alt[y0:y1, x0:x1], 0.5 + 0.5 * bomb + r.uniform(-0.1, 0.1))
                tex[y0:y1, x0:min(x0 + 3, x1)] *= 0.8
                alt[y0:y1, x0:min(x0 + 3, x1)] = 0.0
            x += wf
        tex[y:min(y + 3, n)] *= 0.8
        alt[y:min(y + 3, n)] = 0.0
        y += hf
    yy, xx = np.mgrid[0:n, 0:n] / n
    grao = fbm(xx * 48, yy * 48, 1.5, semente + 1)[..., None]
    liq = fbm(xx * 36, yy * 36, 1.0, semente + 2)[..., None]
    escorre = (0.5 + 0.5 * np.sin(2 * math.pi * (xx * 9 + 0.3 * np.sin(2 * math.pi * yy * 2))))[..., None]
    tex = tex * (0.9 + 0.16 * grao)
    tex = np.where(liq > 0.70, tex * 0.5 + np.array([0.80, 0.62, 0.38]) * 0.5, tex)  # manchas ferrugem (fotografias)
    tex = np.where(liq < 0.26, tex * 0.78, tex)
    tex = tex * (0.93 + 0.08 * escorre)
    return np.clip(tex, 0, 1), ndi.gaussian_filter(alt, 1.2) + 0.15 * grao[..., 0]


def textura_cantaria(n=512, metros=3.0, semente=1713):
    """Cantaria de granito aparelhada (cunhais, cordão, portas): silhares retangulares em fiadas regulares."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    tex = np.zeros((n, n, 3))
    alt = np.ones((n, n))
    y = 0
    while y < n:
        hf = int(r.uniform(0.4, 0.5) * m_px)
        x = -int(r.uniform(0, 0.6) * m_px)
        while x < n:
            wf = int(r.uniform(0.7, 1.2) * m_px)
            base = np.array([0.74, 0.70, 0.61]) * r.uniform(0.92, 1.06) + r.normal(0, 0.006, 3)
            x0, x1 = max(x, 0), min(x + wf, n)
            y1 = min(y + hf, n)
            if x1 > x0:
                tex[y:y1, x0:x1] = base
                tex[y:y1, x0:min(x0 + 2, x1)] *= 0.8
                alt[y:y1, x0:min(x0 + 2, x1)] = 0.0
            x += wf
        tex[y:min(y + 2, n)] *= 0.8
        alt[y:min(y + 2, n)] = 0.0
        y += hf
    yy, xx = np.mgrid[0:n, 0:n] / n
    liq = fbm(xx * 30, yy * 30, 1.0, semente + 2)[..., None]
    tex = tex * (0.93 + 0.12 * fbm(xx * 40, yy * 40, 1.5, semente + 1)[..., None])
    tex = np.where(liq > 0.72, tex * 0.5 + 0.42, tex)
    tex = np.where(liq < 0.27, tex * 0.85, tex)
    return np.clip(tex, 0, 1), ndi.gaussian_filter(alt, 1.0)


def textura_telha(n=256, metros=2.0):
    """Telha de canudo (2 m por repetição): canais em ondas, com fiadas sobrepostas."""
    yy, xx = np.mgrid[0:n, 0:n] / n * metros
    onda = 0.5 + 0.5 * np.cos(2 * math.pi * xx / 0.22)
    fiada = (yy % 0.42) / 0.42
    v = 0.78 + 0.22 * onda - 0.18 * (fiada > 0.9)
    g = fbm(xx * 8, yy * 8, 1.0, 77)
    v = v * (0.9 + 0.2 * g)
    return np.repeat(v[..., None], 3, 2).clip(0, 1), onda - 0.4 * (fiada > 0.9)


CORES_PAREDE = [0, 0, 0, 0, 0, 0, 2, 3, 3, 4]  # cal (a maioria), ocre, granito à vista, rosa (fotografias)


def casa(m, poly, altura):
    """Volume simples: paredes até à cornija (cume no MDS menos a altura do telhado) e telhado de quatro
    águas de telha vermelha. Sem eco no LiDAR, usa a altura dada."""
    if area(poly) < 0:
        poly = poly[::-1]
    zt = cota_t(poly[:, 0], poly[:, 1])
    z0 = float(zt.min()) - 1.0
    cume = medir(desvio(poly, 0.4, True) if abs(area(poly)) > 12 else poly)
    medido = not math.isnan(cume) and cume > float(np.median(zt)) + 2.0
    z1 = max(cume - altura_telhado(poly), float(np.median(zt)) + 2.6) if medido else float(np.median(zt)) + altura
    m.prisma(poly, z0, z1, int(rng.choice(CORES_PAREDE)), topo=False)
    s = simplificar(poly)
    if len(s) == 4 and abs(area(s)) > 0.85 * abs(area(poly)):
        telhado_4aguas(m, s, z1, mat=1)
    else:
        telhado_generico(m, poly, z1, mat=1)
    return z1, medido


ICO = {k: arvore_modelo(k) for k in (0, 1, 2)}
MV, MF, MC = [], [], []


def _lobo(cx, cy, cz, rx, rz, cor, base, topo, sub, irregular):
    v0, f0 = ICO[sub]
    b = len(MV)
    pts = (v0 + rng.normal(0, irregular, v0.shape)) * np.array([rx, rx, rz]) + np.array([cx, cy, cz])
    MV.extend(map(tuple, pts))
    MF.extend([[b + i for i in fa] for fa in f0])
    t = np.clip((pts[:, 2] - base) / max(topo - base, 0.1), 0, 1)
    MC.extend(map(tuple, np.clip(np.array(cor)[None, :] * (0.6 + 0.5 * t ** 0.8)[:, None], 0, 1)))


def plantar(x, y, z, raio, alt, especie, tronco):
    """Árvore com posição, topo e raio de copa medidos no LiDAR; copa esquemática: folhosa (plátanos,
    carvalhos e tílias dos largos e das encostas, copa redonda em lobos) ou pinheiro (copa alta e escura)."""
    b = len(MV)
    rt = 0.12 + 0.045 * raio
    for k in range(6):
        a = 2 * math.pi * k / 6
        MV.append((x + rt * math.cos(a), y + rt * math.sin(a), z - 0.3))
        MV.append((x + 0.7 * rt * math.cos(a), y + 0.7 * rt * math.sin(a), z + tronco + alt * 0.3))
    for k in range(6):
        j = (k + 1) % 6
        MF.append([b + 2 * k, b + 2 * j, b + 2 * j + 1, b + 2 * k + 1])
    MC.extend([(0.36, 0.30, 0.24)] * 12)
    base, topo = z + tronco, z + tronco + alt
    sh = 0.88 + 0.24 * rng.random()
    if especie == "pinheiro":
        cor = tuple(np.clip(np.array((0.22, 0.31, 0.18)) * sh, 0, 1))
        _lobo(x, y, topo - alt * 0.4, raio * 0.85, alt * 0.4, cor, base, topo, 1, 0.12)
        for k in range(3):
            a = rng.random() * 2 * math.pi
            _lobo(x + raio * 0.45 * math.cos(a), y + raio * 0.45 * math.sin(a), topo - alt * 0.45, raio * 0.5, alt * 0.28, cor, base, topo, 1, 0.12)
        return
    cor = tuple(np.clip(np.array((0.33, 0.44, 0.20)) * sh, 0, 1))
    if raio < 1.8:
        _lobo(x, y, base + alt / 2, raio, alt / 2, cor, base, topo, 1, 0.07)
        return
    _lobo(x, y, topo - alt * 0.42, raio * 0.7, alt * 0.42, cor, base, topo, 2 if raio > 3 else 1, 0.07)
    a0 = rng.random() * 2 * math.pi
    for k in range(4):
        a = a0 + 2 * math.pi * k / 4 + rng.normal(0, 0.3)
        d = raio * (0.4 + 0.08 * rng.random())
        rl = raio * (0.5 + 0.08 * rng.random())
        c2 = tuple(np.clip(np.array(cor) * (0.92 + 0.14 * rng.random()), 0, 1))
        _lobo(x + d * math.cos(a), y + d * math.sin(a), base + alt * (0.36 + 0.12 * rng.random()), rl, min(rl, alt * 0.4), c2, base, topo, 1, 0.07)


CAL, CAN, TEL, SOM, MAD, FER, BRO, VID, LOU, LIS = range(10)
MAT_IGR = ["cal", "cantaria", "telha", "sombra", "madeira_verde", "ferro", "bronze", "vidro", "lousa", "cantaria_lisa"]


def cruz(Q, s, z, d, alt=1.2, mat=FER, esp=0.08):
    """Cruz latina de remate (ferro ou granito)."""
    Q.bloco(s - esp / 2, s + esp / 2, z, z + alt, d - esp / 2, d + esp / 2, mat)
    Q.bloco(s - alt * 0.3, s + alt * 0.3, z + alt * 0.62, z + alt * 0.62 + esp, d - esp / 2, d + esp / 2, mat)


def pinaculo(Q, s, z, d, esc=1.0, mat=CAN, tipo="bola"):
    """Pináculo de granito: plinto, dado, e bola (ou agulha piramidal)."""
    Q.bloco(s - 0.28 * esc, s + 0.28 * esc, z, z + 0.12 * esc, d - 0.28 * esc, d + 0.28 * esc, mat)
    Q.bloco(s - 0.2 * esc, s + 0.2 * esc, z + 0.12 * esc, z + 0.55 * esc, d - 0.2 * esc, d + 0.2 * esc, mat)
    Q.bloco(s - 0.25 * esc, s + 0.25 * esc, z + 0.55 * esc, z + 0.63 * esc, d - 0.25 * esc, d + 0.25 * esc, mat)
    if tipo == "bola":
        Q.esfera(s, z + 0.84 * esc, d, 0.21 * esc, mat)
        Q.m.cilindro(Q.p(s, 0, d)[:2], 0.03 * esc, z + 1.04 * esc, z + 1.25 * esc, mat, k=6, r1=0.005)
    else:
        piramide(Q.m, Q.p(s, 0, d)[:2], Q.t, 0.36 * esc, z + 0.63 * esc, z + 1.45 * esc, mat)
        Q.esfera(s, z + 1.5 * esc, d, 0.07 * esc, mat)


def piramide(m, c, u, lado, z0, z1, mat, lado1=0.0):
    """Pirâmide (ou tronco) de base quadrada alinhada com u."""
    u = unit(np.asarray(u, float))
    w = np.array([-u[1], u[0]])
    c = np.asarray(c, float)
    b = [c + (u * a + w * b_) * lado / 2 for a, b_ in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    t = [c + (u * a + w * b_) * lado1 / 2 for a, b_ in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for i in range(4):
        j = (i + 1) % 4
        if lado1 > 0.01:
            m.face([(*b[i], z0), (*b[j], z0), (*t[j], z1), (*t[i], z1)], mat)
        else:
            m.face([(*b[i], z0), (*b[j], z0), (*c, z1)], mat)
    if lado1 > 0.01:
        m.face([(*p, z1) for p in t], mat)


def corpo(Q, L, W, z0, ze, inc=30.0, mw=CAL, mr=TEL, beiral=0.32, cachorros=False, cunhais=True,
          cornija_frente=True, soco=True, d_ini=0.0):
    """Corpo de nave ou capela-mor: paredes com empena, cunhais e cornijas de granito, soco, telhado de
    duas águas com beiral e (opcional) cachorrada de modilhões sob as cornijas (românico)."""
    tg = math.tan(_ang(inc))
    zr = ze + W / 2 * tg
    a, b = d_ini - L, d_ini
    Q.placa([(-W / 2, z0 - 1.5), (W / 2, z0 - 1.5), (W / 2, ze), (0, zr), (-W / 2, ze)], a, b, mw)
    if soco:
        Q.bloco(-W / 2 - 0.07, W / 2 + 0.07, z0 - 1.5, z0 + 0.5, a - 0.07, b + 0.07, CAN)
    if cunhais:
        for sg in (-1, 1):
            for dd in (a, b):
                ss = sorted((sg * (W / 2 - 0.55), sg * (W / 2 + 0.08)))
                d_ = sorted((dd + (0.55 if dd == a else -0.55), dd + (-0.08 if dd == a else 0.08)))
                Q.bloco(ss[0], ss[1], z0 + 0.5, ze, d_[0], d_[1], CAN)
    # cornijas laterais em dois ressaltos
    for sg in (-1, 1):
        s1 = sorted((sg * (W / 2 - 0.05), sg * (W / 2 + 0.2)))
        Q.bloco(s1[0], s1[1], ze - 0.24, ze - 0.06, a - 0.12, b + 0.12, CAN)
        s2 = sorted((sg * (W / 2 - 0.05), sg * (W / 2 + 0.3)))
        Q.bloco(s2[0], s2[1], ze - 0.06, ze + 0.05, a - 0.18, b + 0.18, CAN)
        if cachorros:
            for dd in np.arange(a + 0.5, b - 0.3, 0.75):
                s3 = sorted((sg * W / 2, sg * (W / 2 + 0.24)))
                Q.bloco(s3[0], s3[1], ze - 0.52, ze - 0.24, dd - 0.1, dd + 0.1, CAN)
    # telhado de duas águas
    e = 0.16
    zb = ze + 0.05 - beiral * tg
    Q.placa([(-W / 2 - beiral, zb), (0, zr + 0.06), (W / 2 + beiral, zb), (W / 2 + beiral, zb + e),
             (0, zr + 0.06 + e / math.cos(_ang(inc))), (-W / 2 - beiral, zb + e)], a - 0.05, b - 0.05, mr)
    # cumeeira (fiada de telhões)
    Q.bloco(-0.12, 0.12, zr + 0.06 + e, zr + 0.06 + e + 0.12, a - 0.05, b - 0.05, mr)
    # cornijas das empenas (granito, por cima do telhado)
    for dd, sg in ((b, 1), (a, -1)):
        if dd == b and not cornija_frente:
            continue
        d0, d1 = (dd - 0.12, dd + 0.14) if sg > 0 else (dd - 0.14, dd + 0.12)
        faixa_inclinada(Q, [(-W / 2 - 0.18, ze - 0.05), (0, zr), (W / 2 + 0.18, ze - 0.05)], 0.38, d0, d1, CAN)
        if cachorros:
            for s_ in np.arange(-W / 2 + 0.5, W / 2 - 0.4, 0.7):
                zz = zr - abs(s_) * tg
                d2 = (dd, dd + 0.22) if sg > 0 else (dd - 0.22, dd)
                Q.bloco(s_ - 0.09, s_ + 0.09, zz - 0.32, zz - 0.05, d2[0], d2[1], CAN)
    return zr


def janela(Q, s0, z0, w, h, arco=True, mat_vidro=VID, moldura=0.18, d=0.0, grade=False):
    """Janela com moldura de cantaria saliente (arco pleno ou verga reta) e caixilho escuro."""
    hn = h - (w / 2 if arco else 0)
    Q.vao(w, z0, hn, d + 0.012, mat_vidro, reto=not arco)
    for sg in (-1, 1):
        ss = sorted((s0 + sg * w / 2, s0 + sg * (w / 2 + moldura)))
        Q.bloco(ss[0], ss[1], z0, z0 + hn, d, d + 0.07, CAN)
    Q.bloco(s0 - w / 2 - moldura - 0.06, s0 + w / 2 + moldura + 0.06, z0 - 0.14, z0, d, d + 0.12, CAN)  # peitoril
    if arco:
        Q.arco_sz(s0, w / 2, w / 2 + moldura, z0 + hn, d, d + 0.07, CAN, k=10, aduelas=True)
    else:
        Q.bloco(s0 - w / 2 - moldura, s0 + w / 2 + moldura, z0 + hn, z0 + hn + moldura + 0.04, d, d + 0.07, CAN)
    if grade:
        for k in range(1, 4):
            ss = s0 - w / 2 + w * k / 4
            Q.bloco(ss - 0.015, ss + 0.015, z0, z0 + hn, d + 0.02, d + 0.04, FER)


def porta_igreja(Q, s0, z0, w, hn, moldura=0.3, arco=True, d=0.0, folhas=MAD):
    """Porta com moldura de cantaria (arco pleno ou verga reta) e folhas de madeira verde almofadadas."""
    _vao_desloc(Q, s0, w, z0, hn, d + 0.012, folhas, reto=not arco)
    for k in (-1, 1):  # almofadas das folhas
        for zz in (z0 + 0.3, z0 + hn * 0.55):
            sa = s0 + k * w / 4
            Q.bloco(sa - w / 4 + 0.12, sa + w / 4 - 0.12, zz, zz + hn * 0.35, d + 0.012, d + 0.04, folhas)
    for sg in (-1, 1):
        ss = sorted((s0 + sg * w / 2, s0 + sg * (w / 2 + moldura)))
        Q.bloco(ss[0], ss[1], z0, z0 + hn, d, d + 0.1, CAN)
    if arco:
        Q.arco_sz(s0, w / 2, w / 2 + moldura, z0 + hn, d, d + 0.1, CAN, k=12, aduelas=True)
    else:
        Q.bloco(s0 - w / 2 - moldura, s0 + w / 2 + moldura, z0 + hn, z0 + hn + moldura, d, d + 0.1, CAN)


def sino(m, c, z, r=0.32, mat=BRO):
    """Sino de bronze (sino, cabeça e jugo)."""
    m.cilindro(c, r, z, z + r * 0.35, mat, k=12, r1=r * 0.8)
    m.cilindro(c, r * 0.8, z + r * 0.35, z + r * 1.2, mat, k=12, r1=r * 0.5)
    m.esfera((c[0], c[1], z + r * 1.2), r * 0.5, mat, k=10, achata=0.6)


def varrer(m, linha, perfil, mat, fechar=True):
    """Varre um perfil ao longo de uma polilinha (o exterior fica à direita). O perfil é uma lista de
    (afastamento para fora, cota), cada um um número ou um array com um valor por vértice; os mats
    podem ser um índice ou uma lista com um índice por troço do perfil."""
    n = len(linha)
    cols = []
    for off, z in perfil:
        o = np.broadcast_to(np.asarray(off, float), (n,))
        zz = np.broadcast_to(np.asarray(z, float), (n,))
        xy = desvio(linha, -o)
        cols.append(np.column_stack([xy, zz]))
    mats = mat if isinstance(mat, (list, tuple)) else [mat] * (len(perfil) - 1)
    for k in range(len(cols) - 1):
        A, B = cols[k], cols[k + 1]
        for i in range(n - 1):
            m.face([A[i], A[i + 1], B[i + 1], B[i]], mats[k])
    if fechar:
        m.face([c[0] for c in cols], mats[0])
        m.face([c[-1] for c in cols][::-1], mats[0])
    return cols


# ================================================================ Peniche

NIVEL = 0.4  # nível do mar na maqueta (m): as rochas e o fundo do fosso abaixo dele ficam debaixo de água
CONTORNO = shapely.box(-200.0, -140.0, 200.0, 150.0)
CONT_XY = np.array(CONTORNO.exterior.coords)[:-1]
if area(CONT_XY) < 0:
    CONT_XY = CONT_XY[::-1]
XG, YG = np.meshgrid(LXS, LYS)
ALTURA = MDS - MDT

# massa construída: células de classe 6 mais de 1 m acima do MDT; o MDS suavizado (mediana 5×5) dá o topo
_mb = (CLS == 6) & (ALTURA > 1.0)
S5 = ndi.median_filter(np.where(_mb, MDS, np.nan), size=5, mode="nearest")
S5 = np.where(np.isnan(S5), MDS, S5)
SMDS = ndi.median_filter(MDS, size=3)
S9 = ndi.gaussian_filter(ndi.median_filter(S5, size=9), 1.0)  # topo dos terraplenos sem o entulho e a erva
GULLY = shapely.Polygon([(61.5, 45.0), (64.5, 78.5), (70.0, 79.5), (97.0, 77.0), (99.0, 43.0), (76.0, 38.0)])  # fosso da ponte: rocha


def topo_s(x, y):
    return amostra(S5, x, y)


def chao_s(x, y):
    """O que está à volta de um ponto (MDS suavizado): o chão, a rocha ou outra plataforma."""
    return amostra(SMDS, x, y)


# Túnel por onde a água do mar entra no fosso da ponte (fotografias: a boca em arco no canto sul do fosso, com
# água dentro, e a do lado da baía, ao nível do mar). Eixo e bocas pelos entalhes que o MDT tem nas duas margens
# (a 21 m na baía, x 72, e a 43 m no fosso, x 82,5); o traçado de dentro é estimativa (reta entre as bocas).
TUNEL = dict(eixo=np.array([(72.0, 20.3), (72.0, 24.5), (82.5, 39.5), (82.5, 44.2)]),
             # (boca, normal para fora, comprimento do canal, topo do paramento ou None = 1,6 m acima do fecho)
             bocas=[(np.array([72.0, 20.3]), np.array([0.0, -1.0]), 3.6, 9.0),   # baía: paramento até ao parapeito
                    (np.array([82.5, 44.2]), np.array([0.0, 1.0]), 4.0, None)],  # fosso, na face do pano em talude
             largura=3.8, nascenca=NIVEL + 0.9, fundo=NIVEL - 1.4)
TUNEL["fecho"] = TUNEL["nascenca"] + TUNEL["largura"] / 2
TUNEL["paramento"] = TUNEL["fecho"] + 1.6  # topo do paramento das bocas


def boca_perto(x, y):
    """Pontos à frente ou à volta de uma boca do túnel, onde os panos e as escarpas param acima do paramento."""
    k = np.zeros(np.shape(x), bool)
    for b, n, _, topo in TUNEL["bocas"]:
        if topo is not None:  # esta boca tem paramento próprio até cima: os panos não se cortam
            continue
        t = np.array([n[1], -n[0]])
        rs = (np.asarray(x) - b[0]) * t[0] + (np.asarray(y) - b[1]) * t[1]
        rd = (np.asarray(x) - b[0]) * n[0] + (np.asarray(y) - b[1]) * n[1]
        k |= (np.abs(rs) < TUNEL["largura"] / 2 + 3.4) & (rd > -4.0) & (rd < 3.0)
    return k


def canais_tunel(X, Y):
    """Células dos canais de água entre cada boca do túnel e o mar (ou o fosso)."""
    k = np.zeros(X.shape, bool)
    for b, n, comp, _ in TUNEL["bocas"]:
        t = np.array([n[1], -n[0]])
        rel_s = (X - b[0]) * t[0] + (Y - b[1]) * t[1]
        rel_d = (X - b[0]) * n[0] + (Y - b[1]) * n[1]
        k |= (np.abs(rel_s) < TUNEL["largura"] / 2 + 0.3) & (rel_d > -0.2) & (rel_d < comp)
    return k


# mar: células abaixo do nível, em manchas grandes (o mar aberto, o porto e o fosso da ponte, que enche com a maré)
_ag = MDT < NIVEL
_lab, _k = ndi.label(_ag)
_tam = ndi.sum(np.ones_like(_lab), _lab, np.arange(1, _k + 1)) * LPASSO ** 2
AGUA = np.isin(_lab, np.nonzero(_tam > 150)[0] + 1) | canais_tunel(XG, YG)
_maior = np.argmax(_tam) + 1
MAR_ABERTO = _lab == _maior
D_AGUA = ndi.distance_transform_edt(AGUA) * LPASSO        # distância à costa, dentro de água
D_MAR = ndi.distance_transform_edt(~MAR_ABERTO) * LPASSO  # distância ao mar aberto, em terra


def perto_do_mar(x, y):
    return float(amostra(D_MAR, x, y))


# ---------------------------------------------------------------- texturas próprias

def textura_reboco(cor, n=512, metros=6.0, semente=7, gasto=0.5):
    """Reboco pintado (amarelo-ocre ou cal) gasto pelo salitre, como nas fotografias: manchas mais claras
    e mais escuras do pigmento e, onde a pintura caiu, a pedra cinzenta por baixo."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    f1 = fbm(xx * 10, yy * 10, 1.0, semente)
    f2 = fbm(xx * 40, yy * 40, 1.0, semente + 5)
    f3 = fbm(xx * 6, yy * 6, 1.0, semente + 9)
    c = np.array(cor)[None, None, :] * (0.9 + 0.18 * f2[..., None])
    c = np.where((f1 > 0.66)[..., None], c * 0.86 + 0.06, c)                  # pintura mais velha
    caiu = (f3 > 1 - 0.3 * gasto) & (f2 > 0.45)
    c = np.where(caiu[..., None], np.array([0.66, 0.63, 0.56]) * (0.9 + 0.2 * f2[..., None]), c)
    esc = 0.5 + 0.5 * np.sin(2 * math.pi * (xx * 14 + 0.4 * f1))                 # escorrências
    c = c * (0.96 + 0.05 * esc[..., None] * (yy[..., None] > 0.0))
    return np.clip(c, 0, 1)


def textura_ondas(n=256):
    """Ondulação miúda do mar (mapa de alturas periódico para o mapa de normais)."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    a = np.zeros((n, n))
    r = np.random.default_rng(5)
    for _ in range(14):
        kx, ky = r.integers(-6, 7, 2)
        if kx == 0 and ky == 0:
            continue
        a += np.sin(2 * math.pi * (kx * xx + ky * yy) + r.random() * 6.28) / math.hypot(kx, ky)
    return a / np.abs(a).max()


def textura_lajes(n=256, metros=3.0):
    """Lajeado de calcário (adarves, terraços dos edifícios)."""
    r = np.random.default_rng(3)
    tex = np.ones((n, n, 3)) * np.array([0.78, 0.76, 0.71])
    m_px = n / metros
    y = 0
    while y < n:
        h = int(r.uniform(0.5, 0.9) * m_px)
        x = -int(r.uniform(0, 0.6) * m_px)
        while x < n:
            w = int(r.uniform(0.6, 1.2) * m_px)
            tex[y:y + h, max(x, 0):min(x + w, n)] *= r.uniform(0.9, 1.05)
            tex[y:y + h, max(x, 0):max(x, 0) + 1] *= 0.75
            x += w
        tex[y:y + 1] *= 0.75
        y += h
    yy, xx = np.mgrid[0:n, 0:n] / n
    return np.clip(tex * (0.92 + 0.12 * fbm(xx * 20, yy * 20, 1.0, 4))[..., None], 0, 1)


# ---------------------------------------------------------------- UV mistas

def uv_misto(ob, escala, planta=()):
    """UV em metros (como uv_mundo); as faces dos materiais em `planta` levam a UV da ortoimagem (a mesma
    textura do chão), para os adarves, os terraplenos e os pátios mostrarem o lajeado e a relva medidos."""
    uv_mundo(ob, escala)
    if not planta:
        return
    me = ob.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    vi = np.empty(len(me.loops), int)
    me.loops.foreach_get("vertex_index", vi)
    lt = np.empty(len(me.polygons), int)
    me.polygons.foreach_get("loop_total", lt)
    mi = np.empty(len(me.polygons), int)
    me.polygons.foreach_get("material_index", mi)
    sel = np.repeat(np.isin(mi, planta), lt)
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    p = co[vi]
    uv[sel, 0] = (p[sel, 0] - LXS[0]) / (LXS[-1] - LXS[0])
    uv[sel, 1] = (p[sel, 1] - LYS[0]) / (LYS[-1] - LYS[0])
    me.uv_layers.active.data.foreach_set("uv", uv.ravel())


# ---------------------------------------------------------------- plataformas (muralhas e edifícios)

# Zona da antiga prisão (hoje Museu Nacional Resistência e Liberdade): as plataformas que caem aqui são
# edifícios (paredes caiadas, janelas, terraços), as outras são terraplenos e panos de muralha.
PRISAO = shapely.Polygon([(-46, -34), (-46, 28), (-30, 40), (-17, 67), (8, 62), (33, 47), (34, 14), (28, -36)])
FORTIM = dict(c=np.array([-10.5, -63.3]), r=11.8, corda=-57.3, frente=-50.8, topo=17.05, chao=14.3)
AMARELO_EDIF = shapely.box(17.0, 18.0, 36.0, 47.0)  # a ponta sudeste do pavilhão grande, amarela (fotografias)
ALV, CHAO_, CAL_, TERR, OCRE_, CANT_ = range(6)
MAT_PLAT = ["alvenaria", "chao", "cal", "terraco", "ocre", "cantaria"]
ESCARPAS = []   # escarpas de alvenaria por baixo dos parapeitos: o terreno à frente é rebaixado ao pé
BORDAS = []     # (linha, normal para fora, topo, queda, tipo) de cada troço das plataformas, para parapeitos
EDIFICIOS = []  # (poligono, topo, chão, amarelo)
PLAT_SHP = []


def plataforma(p, m):
    poly = shapely.geometry.polygon.orient(shapely.Polygon(p["contorno"], p["buracos"]).buffer(0), 1.0)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    c = poly.centroid
    if np.hypot(c.x - FORTIM["c"][0], c.y - FORTIM["c"][1]) < 10:
        return None  # o Fortim Redondo é feito à parte
    edif = poly.intersection(PRISAO).area > 0.6 * poly.area
    if not edif:
        # o tabuleiro da ponte também é classe 6: tira-se, a ponte é feita à parte
        ponte_shp = shapely.LineString([PONTE["a"], PONTE["b"]]).buffer(PONTE["largura"] / 2 + 0.8, cap_style="flat")
        if poly.intersects(ponte_shp):
            pd_ = poly.difference(ponte_shp)
            if pd_.is_empty:
                return None
            if pd_.geom_type != "Polygon":
                pd_ = max(pd_.geoms, key=lambda g: g.area)
            if pd_.area < 5:
                return None
            poly = shapely.geometry.polygon.orient(pd_, 1.0)
        # panos direitos: o contorno da grelha de 50 cm vem aos dentes; simplificado, cada pano fica um plano
        ps = poly.simplify(0.75, preserve_topology=True).buffer(0)
        if ps.geom_type == "Polygon" and ps.area > 0.9 * poly.area:
            poly = shapely.geometry.polygon.orient(ps, 1.0)
    amarelo = False
    tipo = "edificio" if edif else "muralha"
    PLAT_SHP.append(poly)
    mat_lado = (OCRE_ if amarelo else CAL_) if edif else ALV
    mat_topo = TERR if edif else CHAO_
    aneis = [np.array(poly.exterior.coords)[:-1]] + [np.array(r.coords)[:-1] for r in poly.interiors]
    bpts, bz = [], []
    for q in aneis:
        q = densificar(q, 0.7, True)
        dentro_ = desvio(q, 1.1, True)  # para dentro (o anel exterior é anti-horário, os buracos horários)
        ok = shapely.contains_xy(poly, dentro_[:, 0], dentro_[:, 1])
        dentro_[~ok] = q[~ok]
        zt = topo_s(dentro_[:, 0], dentro_[:, 1])
        if edif:
            zt = np.full(len(q), float(np.median(zt)))
        elif len(zt) > 9:
            # coroamento regular: mediana ao longo do pano (c. 5 m), sem os dentes do MDS
            zt = ndi.median_filter(zt, size=7, mode="wrap")
        bpts.append(q)
        bz.append(zt)
        # panos: do topo ao chão de fora, em talude nas muralhas (1/10 da altura)
        fora = desvio(q, -0.9, True)
        zg = np.minimum(cota_t(fora[:, 0], fora[:, 1]), zt - 0.3) - 0.8
        if not edif and len(zg) > 9:
            zg = ndi.minimum_filter(ndi.median_filter(zg, size=5, mode="wrap"), size=3, mode="wrap")
        bat = np.clip(0.10 * (zt - zg), 0, 1.6) if not edif else np.zeros(len(q))
        base = desvio(q, -bat, True)
        n = len(q)
        if not edif:
            # cordão de pedra (meia-cana) a 1,2 m do coroamento, nos panos com mais de 3,5 m (fotografias)
            alto = (zt - zg) > 3.5
            zc = zt - 1.2
            frac = np.clip(1.2 / np.maximum(zt - zg, 0.1), 0, 1)
            i0 = 0
            alto_c = np.append(alto, alto[0])
            k = 0
            while k < n:
                if not alto[k]:
                    k += 1
                    continue
                j = k
                while j < n and alto[j]:
                    j += 1
                idx_ = list(range(k, min(j + 1, n)))
                if len(idx_) >= 3:
                    lin_ = q[idx_]
                    off = bat[idx_] * frac[idx_]
                    zz = zc[idx_]
                    perf = [(off, zz - 0.16), (off + 0.16, zz - 0.1), (off + 0.21, zz), (off + 0.16, zz + 0.1), (off, zz + 0.16)]
                    varrer(m, lin_, perf, CANT_, fechar=False)
                k = j
        if not edif:  # sobre as bocas do túnel o pano para no topo do paramento
            zg = np.where(boca_perto(base[:, 0], base[:, 1]), np.maximum(zg, TUNEL["paramento"] - 0.2), zg)
        for i in range(n):
            j = (i + 1) % n
            mt_ = mat_lado
            if edif and AMARELO_EDIF.contains(shapely.Point(*((q[i] + q[j]) / 2))):
                mt_ = OCRE_
            m.face([(*q[i], zt[i]), (*base[i], zg[i]), (*base[j], zg[j]), (*q[j], zt[j])], mt_)
        # troços para os parapeitos: queda medida 2 m para fora (no MDS suavizado: chão, rocha ou outra plataforma)
        nr = np.array([unit(np.array([q[(i + 1) % n][1] - q[i - 1][1], -(q[(i + 1) % n][0] - q[i - 1][0])])) for i in range(n)])
        fora2 = q + nr * 2.2
        queda = zt - np.minimum(chao_s(fora2[:, 0], fora2[:, 1]), cota_t(fora2[:, 0], fora2[:, 1]) + 0.3)
        BORDAS.append(dict(linha=q, nr=nr, topo=zt, queda=queda, tipo=tipo, amarelo=amarelo))
    # topo: triangulação de Delaunay do contorno denso e de uma grelha de 1 m por dentro
    from scipy.spatial import Delaunay
    inner = poly.buffer(-0.5)
    gx, gy = np.meshgrid(np.arange(math.floor(poly.bounds[0]), poly.bounds[2], 0.7),
                         np.arange(math.floor(poly.bounds[1]), poly.bounds[3], 0.7))
    gx, gy = gx.ravel(), gy.ravel()
    k = shapely.contains_xy(inner, gx, gy)
    ip = np.column_stack([gx[k], gy[k]])
    iz = amostra(S9, ip[:, 0], ip[:, 1]) if not edif else np.full(len(ip), float(np.median(np.concatenate(bz))))
    pts = np.vstack(bpts + [ip])
    zz = np.concatenate(bz + [iz])
    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(1)
    ok = shapely.contains_xy(poly, cen[:, 0], cen[:, 1])
    lado = np.max(np.linalg.norm(pts[tri] - pts[np.roll(tri, 1, 1)], axis=2), axis=1)
    tri = tri[ok & (lado < 3.2)]
    for t in tri:
        a, b, c_ = pts[t]
        if (b - a)[0] * (c_ - a)[1] - (b - a)[1] * (c_ - a)[0] < 0:
            t = t[::-1]
        zr = zz[t].max() - zz[t].min()
        m.face([(*pts[i], zz[i]) for i in t], mat_topo if zr < 1.4 else mat_lado)  # degraus por dentro: pano
    if edif:
        EDIFICIOS.append((poly, float(np.median(np.concatenate(bz))), float(cota_t(c.x, c.y)), amarelo))
    return poly


def parapeitos(mq):
    """Parapeitos, merlões e muros finos traçados no relevo local do MDS (plataformas.py). Em cada troço,
    de 0,5 em 0,5 m: a base é a cota do MDS menos o relevo (o chão ou o adarve ao lado), o parapeito é o
    percentil 35 da cota numa janela de 3,5 m e, onde o percentil 85 passa mais de 0,35 m acima dele, há
    merlões (postos a cada 3,2 m, com a altura medida). Junto ao mar aberto vão pintados de amarelo-ocre
    (a cortina da costa, como nas fotografias); no resto ficam em pedra."""
    n_merl = n_lin = 0
    comp = 0.0
    guaritas = []
    edif = shapely.union_all([p.buffer(1.0) for p, *_ in EDIFICIOS]) if EDIFICIOS else shapely.Point(1e6, 1e6)
    linhas_ = [(io_, shapely.LineString(mu["linha"])) for io_, mu in enumerate(MUROS) if np.median(mu["relevo"]) >= 0.6]
    n_dup = 0
    # Um parapeito anda de nível: onde o topo do traço cai mais de 2,5 m abaixo do nível do parapeito (mediana em
    # 20 m), ou numa ponta que sobe ou desce mais de 0,6 m por metro, o "relevo" do MDS é a aresta da arriba, não
    # um muro (12 nov. 2026: troços amarelos a descer pela rocha). Esses bocados saem e o resto parte-se em peças
    # de 4 m ou mais.
    from shapely.ops import substring
    pecas_ = []
    n_ingr = 0
    for io_, mu in enumerate(MUROS):
        mu = dict(mu, origem=io_)
        cota_ = np.array(mu["cota"])
        rel_ = np.clip(np.array(mu["relevo"]), 0, None)
        if len(cota_) < 6:
            pecas_.append(mu)
            continue
        top_ = ndi.percentile_filter(cota_, 20, size=9, mode="nearest")
        ref_ = ndi.median_filter(top_, size=41, mode="nearest")  # o nível do parapeito em 20 m
        def declive_(t):  # quanto o topo sobe ou desce por metro, em 4 m
            d = np.zeros(len(t))
            d[4:-4] = np.abs(t[8:] - t[:-8]) / 4.0
            d[:4], d[-4:] = d[4], d[-5]
            return d
        # o topo que se desenha (base + 0,6 a 2,4 m) também conta: na ponta de um traço sobre a aresta da arriba a
        # base cai e o parapeito descia a pique até à água
        bd_ = ndi.median_filter(cota_ - rel_, size=9, mode="nearest")
        td_ = bd_ + np.clip(top_ - bd_, 0.6, 2.4)
        dz_ = np.maximum(declive_(top_), declive_(td_))
        # saem os mergulhos (2,5 m abaixo do nível, com 1 m de margem) e, em cada ponta, tudo até ao primeiro
        # bocado de 4 m seguidos sem mergulho e sem subir ou descer mais de 0,6 m por metro (os degraus a meio
        # de um parapeito ficam); dos bocados que restam, saem os de menos de 4 m e os que ficam 2,5 m ou mais
        # abaixo do nível do traço (o pé da arriba)
        ingr = ndi.binary_dilation((top_ < ref_ - 2.5) | (td_ < ref_ - 2.5), iterations=2)
        bom_ = np.convolve(((dz_ <= 0.6) & ~ingr).astype(int), np.ones(8, int), "valid") == 8
        if bom_.any():
            j_ = np.nonzero(bom_)[0]
            ingr[:j_[0]] = True
            ingr[j_[-1] + 8:] = True
        else:
            ingr[:] = True
        lab_, nl_ = ndi.label(~ingr)
        med_ = np.median(top_)
        for q_ in range(1, nl_ + 1):
            idx_ = np.nonzero(lab_ == q_)[0]
            if len(idx_) < 8 or np.median(top_[idx_]) < med_ - 2.5:
                ingr[idx_] = True
        ln_ = shapely.LineString(mu["linha"])
        if not ingr.any():
            pecas_.append(mu)
            continue
        n_ingr += 1
        lab_, nl_ = ndi.label(~ingr)
        for q_ in range(1, nl_ + 1):
            idx_ = np.nonzero(lab_ == q_)[0]
            if len(idx_) < 8:
                continue
            i0_, i1_ = idx_[0], idx_[-1]
            sub = substring(ln_, i0_ * 0.5, min(i1_ * 0.5, ln_.length))
            if sub.geom_type != "LineString" or sub.length < 3.5:
                continue
            pecas_.append(dict(mu, linha=np.array(sub.coords).tolist(), cota=cota_[i0_:i1_ + 1].tolist(),
                               relevo=np.array(mu["relevo"])[i0_:i1_ + 1].tolist()))
    for mu in pecas_:
        ln = shapely.LineString(mu["linha"])
        if np.median(mu["relevo"]) < 0.6 or ln.intersection(edif).length > 0.5 * ln.length:
            continue
        # junto ao mar, o relevo do MDS dá às vezes dois traços para o mesmo parapeito (a crista e o pé): o mais
        # curto, se tiver metade ou mais a menos de 3,5 m de um mais comprido, sai (dois panos paralelos faziam
        # uma vala entre eles, e a muralha parecia partir-se na rocha)
        if perto_do_mar(*np.array(ln.centroid.coords[0])) < 30 and any(
                io_ != mu["origem"] and o.length > ln.length and ln.intersection(o.buffer(3.5)).length >= 0.5 * ln.length
                for io_, o in linhas_):
            n_dup += 1
            continue
        cf = ln.centroid
        ponte_shp = shapely.LineString([PONTE["a"], PONTE["b"]]).buffer(PONTE["largura"] / 2 + 1.5)
        if ln.intersection(ponte_shp).length > 0.3 * ln.length:
            continue  # as guardas da ponte, feitas com ela
        if cf.distance(shapely.Point(*FORTIM["c"])) < FORTIM["r"] - 1.0 and ln.length < 40:
            continue
        if cf.distance(shapely.Point(*FORTIM["c"])) < FORTIM["r"] + 6 and cf.y > FORTIM["corda"] - 2:
            continue  # o portal e a sineira do fortim, feitos à parte
        cota_ = np.array(mu["cota"])
        rel_ = np.clip(np.array(mu["relevo"]), 0, None)
        k = len(cota_)
        if k < 6:
            continue
        base = ndi.median_filter(cota_ - rel_, size=9, mode="nearest")
        p35 = ndi.percentile_filter(cota_, 20, size=9, mode="nearest")
        p85 = ndi.percentile_filter(cota_, 85, size=9, mode="nearest")
        top = base + np.clip(p35 - base, 0.6, 2.4)
        merl = (p85 - top) > 0.35
        lin = np.array(mu["linha"])
        sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(lin, axis=0), axis=1))])
        ss = np.arange(k) * 0.5
        lin = densificar(lin, 1.0)
        sv = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(lin, axis=0), axis=1))])
        zt = np.interp(sv, ss, top)
        nr = np.array([unit(np.array([lin[min(i + 1, len(lin) - 1)][1] - lin[max(i - 1, 0)][1],
                                      -(lin[min(i + 1, len(lin) - 1)][0] - lin[max(i - 1, 0)][0])])) for i in range(len(lin))])
        za = cota_t(*(lin + nr * 1.3).T)
        zb_ = cota_t(*(lin - nr * 1.3).T)
        zbase = np.interp(sv, ss, base)
        zlow = np.maximum(np.minimum(np.minimum(za, zb_), zt - 0.8) - 0.5, zbase - 9.0)
        zpin = np.minimum(np.maximum(zbase - 0.25, zlow + 0.05), zt - 0.3)  # a pintura começa no chão de dentro
        amarelo = np.mean([perto_do_mar(*p) for p in lin[::3]]) < 25 and lin[:, 0].mean() < 80
        # escarpa de alvenaria por baixo do parapeito, do lado de fora (o mais baixo): da cota do chão de
        # dentro até ao pé, medido 2 a 3,6 m para fora (no máximo 9 m de pano); por baixo fica a rocha
        e = 0.42
        so = 1.0 if np.median(za - zb_) < 0 else -1.0
        nout = nr * so
        zf = np.min([cota_t(*(lin + nout * o).T) for o in (2.0, 2.8, 3.6, 4.5)], axis=0)
        zf = ndi.minimum_filter(ndi.median_filter(zf, size=7, mode="nearest"), size=3, mode="nearest")
        zf = np.maximum(np.maximum(zf, zbase - 12.0), NIVEL + 0.3)
        # pano contínuo: os intervalos curtos (até c. 4 m) entre troços altos também levam escarpa; sem isto
        # ficavam dentes de rocha entre panos e a muralha parecia desfazer-se na arriba
        alto = ndi.binary_closing((zbase - zf) > 1.2, structure=np.ones(9, bool), border_value=0) | ((zbase - zf) > 1.2)
        if alto.sum() >= 3:
            bat = 0.12 * (zbase - zf)
            top_ = lin + nout * e
            pe_ = lin + nout * (e + bat)[:, None]
            zt_e = np.minimum(zlow, zbase)
            # junto à água o pano entra no mar (sem fresta escura entre o pé e a água)
            zpe_ = np.where(zf < NIVEL + 1.0, NIVEL - 1.0, zf - 0.3)
            zpe_ = np.where(boca_perto(pe_[:, 0], pe_[:, 1]), np.maximum(zpe_, TUNEL["paramento"] - 0.2), zpe_)
            for i in range(len(lin) - 1):
                if alto[i] and alto[i + 1]:
                    f_ = [(*top_[i], zt_e[i]), (*top_[i + 1], zt_e[i + 1]), (*pe_[i + 1], zpe_[i + 1]), (*pe_[i], zpe_[i])]
                    mq.face(f_ if so < 0 else f_[::-1], 0)
            ESCARPAS.append(dict(linha=lin, nout=nout, pe=zf - 0.3, alto=alto))
        mw, mc = (1, 3) if amarelo else (0, 2)
        e = 0.42
        perfil = [(e, zlow), (e, zpin), (e, zt), (e - 0.07, zt + 0.1), (-e + 0.07, zt + 0.1), (-e, zt), (-e, zpin), (-e, zlow)]
        varrer(mq, lin, perfil, [0, mw, mc, mc, mc, mw, 0], fechar=True)
        n_lin += 1
        comp += float(sv[-1])
        # merlões
        if merl.mean() > 0.3:
            for s0 in np.arange(0.5, sv[-1] - 2.0, 3.2):
                if not merl[min(int((s0 + 1.0) / 0.5), k - 1)]:
                    continue
                a_ = np.array([np.interp(s0, sv, lin[:, 0]), np.interp(s0, sv, lin[:, 1])])
                b_ = np.array([np.interp(s0 + 2.0, sv, lin[:, 0]), np.interp(s0 + 2.0, sv, lin[:, 1])])
                if np.linalg.norm(b_ - a_) < 1.5:
                    continue
                z0 = float(np.interp(s0 + 1.0, sv, zt)) + 0.1
                z1 = max(float(np.interp(s0 + 1.0, ss, p85)), z0 + 0.5)
                t_ = unit(b_ - a_)
                mq.caixa((a_ + b_) / 2, t_, float(np.linalg.norm(b_ - a_)), 2 * e, z0 - 0.1, z1, mw)
                mq.caixa((a_ + b_) / 2, t_, float(np.linalg.norm(b_ - a_)) + 0.05, 2 * e + 0.05, z1, z1 + 0.08, mc)
                n_merl += 1
        # guaritas nos ângulos dos panos altos
        ls = shapely.LineString(lin).simplify(1.2)
        q = np.array(ls.coords)
        for i in range(1, len(q) - 1):
            t1, t2 = unit(q[i] - q[i - 1]), unit(q[i + 1] - q[i])
            if np.linalg.norm(q[i] - q[i - 1]) < 4 or np.linalg.norm(q[i + 1] - q[i]) < 4:
                continue
            ang = math.degrees(math.atan2(t1[0] * t2[1] - t1[1] * t2[0], t1 @ t2))
            if abs(ang) < 45:
                continue
            bis = unit(np.array([t1[1], -t1[0]]) + np.array([t2[1], -t2[0]]))
            if ang < 0:
                bis = -bis
            # o lado de fora é o mais baixo
            z_ = float(np.interp(np.argmin(np.linalg.norm(lin - q[i], axis=1)), np.arange(len(zt)), zt))
            zf = float(chao_s(*(q[i] + bis * 3.0)))
            zd = float(chao_s(*(q[i] - bis * 3.0)))
            if zd < zf:
                bis, zf = -bis, zd
            if z_ - zf > 5.0:
                guaritas.append((q[i], bis, z_ - 0.95, abs(ang)))
    MEDIDAS["parapeitos"] = {"troços": n_lin, "comprimento": comp, "duplicados_tirados": n_dup, "com_troços_a_descer": n_ingr}
    return n_merl, guaritas


def escolher_guaritas(cands, dmin=26.0):
    """Ângulos mais agudos primeiro, com 22 m pelo menos entre guaritas (simplificação a 0,7 m dá vários
    vértices no mesmo canto)."""
    out = []
    for p, bis, z, ang in sorted(cands, key=lambda c: -c[3]):
        if all(np.linalg.norm(p - o[0]) > dmin for o in out):
            out.append((p, bis, z, ang))
    return out


# ---------------------------------------------------------------- edifícios da prisão

PAINEL = 0.3  # espessura do paramento das fachadas da prisão, para os vãos ficarem fundos


def edificio(m, poly, zt, amarelo):
    """Paredes caiadas (ou amarelas) já estão nas plataformas: aqui vão as janelas por pisos, com grade
    nos pavilhões prisionais, o soco, a platibanda com capeamento e os cunhais. As janelas são vãos fundos:
    cada fachada leva um paramento de 0,3 m furado nos vãos, com o caixilho escuro lá atrás (12 nov. 2026:
    as janelas pintadas na parede não pareciam buracos)."""
    e = PAINEL
    q = np.array(poly.simplify(1.3).exterior.coords)[:-1]
    n = len(q)
    # a parede vem do contorno em grelha de 50 cm, que sai até c. 0,6 m da fachada simplificada: as fachadas
    # (paramento, vãos e platibanda) ficam à frente do ponto mais saliente, para nenhum dente tapar um vão
    bd = densificar(np.array(poly.exterior.coords)[:-1], 0.25, True)
    D = 0.0
    for i in range(n):
        a, b = q[i], q[(i + 1) % n]
        t = unit(b - a)
        nr_ = np.array([t[1], -t[0]])
        rel = bd - (a + b) / 2
        su, dd = rel @ t, rel @ nr_
        k_ = (np.abs(su) < np.linalg.norm(b - a) / 2) & (np.abs(dd) < 1.5)
        if k_.any():
            D = max(D, float(dd[k_].max()))
    D = min(D, 1.0)
    e_ = e + D
    # platibanda e cornija
    perfil = [(e_ + 0.02, zt - 0.5), (e_ + 0.18, zt - 0.5), (e_ + 0.18, zt - 0.3), (e_ + 0.02, zt - 0.3), (e_ + 0.02, zt + 0.75),
              (-0.22, zt + 0.75), (-0.22, zt - 0.05)]
    varrer(m, np.vstack([q, q[:1]]), perfil, [CAL, CAL, CAL, CAL, CAL, CAL], fechar=False)
    for i in range(n):
        a, b = q[i], q[(i + 1) % n]
        amarelo = AMARELO_EDIF.contains(shapely.Point(*((a + b) / 2)))
        L = float(np.linalg.norm(b - a))
        t = unit(b - a)
        nr = np.array([t[1], -t[0]])
        mid = (a + b) / 2
        g = float(np.median(cota_t(*(np.linspace(a, b, 5) + nr * 1.2).T)))
        g = min(g, zt - 2.5)
        Q = Quadro(m, mid + nr * D, nr)
        mp = OCR if amarelo else CAL
        L = L + 2 * D  # nos cantos, o paramento fecha com o da fachada vizinha
        Q.bloco(-L / 2 - e, L / 2 + e, g - 0.6, g + 0.6, -0.1, e + 0.06, LIS)  # soco
        if amarelo:  # cunhais e cornija brancos (fotografias)
            Q.bloco(L / 2 - 0.5, L / 2 + e + 0.02, g, zt - 0.3, -0.02, e + 0.08, CAL)
            Q.bloco(-L / 2 - e - 0.02, -L / 2 + 0.5, g, zt - 0.3, -0.02, e + 0.08, CAL)
            Q.bloco(-L / 2, L / 2, zt - 0.95, zt - 0.6, -0.02, e + 0.1, CAL)
        vaos = []
        if L >= 3.2:
            pisos = np.arange(g + 1.1, zt - 2.0, 3.1)
            passo = 3.2 if amarelo else 2.6
            k = int((L - 1.6) // passo)
            ss = (np.arange(max(k, 0)) - (k - 1) / 2) * passo
            for pz in pisos:
                for s_ in ss:
                    if amarelo:
                        vaos.append((s_, pz, 1.0, 2.0, True))
                    else:
                        # fotografia do pátio: vãos de c. 1 × 1,3 m com caixilho claro e grade
                        vaos.append((s_, pz + 0.25, 0.95, 1.3, False))
        # paramento furado: grelha cortada pelos vãos (cada vão em retângulo; o arco leva tímpano por cima)
        z0p, z1p = g + 0.6, zt - 0.5 if not amarelo else zt - 0.95
        sb = sorted({-L / 2 - e, L / 2 + e} | {v[0] + sg * v[2] / 2 for v in vaos for sg in (-1, 1)})
        zb = sorted({z0p, z1p} | {v[1] + dz for v in vaos for dz in (0.0, v[3])})
        for s0, s1 in zip(sb[:-1], sb[1:]):
            for za, zb_ in zip(zb[:-1], zb[1:]):
                if zb_ <= z0p or za >= z1p:
                    continue
                sc, zc = (s0 + s1) / 2, (za + zb_) / 2
                if any(abs(sc - v[0]) < v[2] / 2 and v[1] < zc < v[1] + v[3] for v in vaos):
                    continue
                Q.bloco(s0, s1, max(za, z0p), min(zb_, z1p), 0.0, e, mp)
        for s_, z_, w_, h_, arco in vaos:
            hn = h_ - (w_ / 2 if arco else 0)
            # fundo do vão (caixilho escuro) e ombreiras por dentro
            # (em placa fina, que tem as duas faces: o Q.vao faz uma face só, virada para dentro nestas
            # fachadas, e o visor não a desenhava; via-se a parede caiada através do vão)
            w2 = w_ / 2
            pv = [(s_ - w2, z_), (s_ + w2, z_), (s_ + w2, z_ + hn)]
            if arco:
                pv += [(s_ + w2 * math.cos(math.pi * k_ / 14), z_ + hn + w2 * math.sin(math.pi * k_ / 14)) for k_ in range(1, 14)]
            pv += [(s_ - w2, z_ + hn)]
            Q.placa(pv, 0.0, 0.012, VID if arco else SOM)
            if arco:  # tímpano do arco: o paramento enche o retângulo por cima do arco
                r_ = w_ / 2
                arc = [(s_ + r_ * math.cos(math.pi * k_ / 10), z_ + hn + r_ * math.sin(math.pi * k_ / 10)) for k_ in range(11)]
                for (sa, za), (sb2, zb2) in zip(arc[:-1], arc[1:]):
                    Q.placa([(sa, za), (sb2, zb2), (sb2, z_ + h_), (sa, z_ + h_)], 0.0, e, mp)
            # moldura de cantaria à face do paramento
            for sg in (-1, 1):
                s2 = sorted((s_ + sg * w_ / 2, s_ + sg * (w_ / 2 + 0.15)))
                Q.bloco(s2[0], s2[1], z_, z_ + hn, e, e + 0.07, CAN)
            Q.bloco(s_ - w_ / 2 - 0.21, s_ + w_ / 2 + 0.21, z_ - 0.14, z_, 0.0, e + 0.12, CAN)  # peitoril
            if arco:
                Q.arco_sz(s_, w_ / 2, w_ / 2 + 0.16, z_ + hn, e, e + 0.07, CAN, k=10, aduelas=True)
            else:
                Q.bloco(s_ - w_ / 2 - 0.14, s_ + w_ / 2 + 0.14, z_ + hn, z_ + hn + 0.18, e, e + 0.07, CAN)
                for k_ in range(1, 4):  # grade a meio do vão
                    sx = s_ - w_ / 2 + w_ * k_ / 4
                    Q.bloco(sx - 0.015, sx + 0.015, z_, z_ + hn, e * 0.5, e * 0.5 + 0.03, FER)
    return n


def torre_branca(m, c, z0, ztopo):
    """Torre branca da prisão (fotografias): fuste quadrado caiado que afina para cima, com cornija de
    pedra, lanternim com vãos e remate em pirâmide."""
    u = np.array([1.0, 0.0])
    h = ztopo - z0
    zf = z0 + h * 0.66
    piramide(m, c, u, 3.4, z0, zf, CAL, lado1=2.5)
    for k_, zz in enumerate(np.arange(z0 + 1.5, zf - 1.0, 2.6)):
        Qf = Quadro(m, c + np.array([0, -1.45 + 0.17 * k_ / 3]), (0, -1))
        Qf.vao(0.4, zz, 0.8, 0.02, SOM, reto=True)
    m.caixa(c, u, 2.9, 2.9, zf, zf + 0.25, CAN)
    m.caixa(c, u, 2.6, 2.6, zf + 0.25, zf + 0.4, CAL)
    zl = zf + 0.4
    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        m.caixa(c + np.array([a, b]) * 0.95, u, 0.5, 0.5, zl, zl + 1.6, CAL)
    m.caixa(c, u, 1.4, 1.4, zl, zl + 1.6, SOM)
    m.caixa(c, u, 2.8, 2.8, zl + 1.6, zl + 1.85, CAN)
    piramide(m, c, u, 2.6, zl + 1.85, ztopo, CAL)
    m.esfera((c[0], c[1], ztopo + 0.15), 0.18, CAN)


# ---------------------------------------------------------------- Fortim Redondo

def fortim(m):
    """Fortim Redondo (1557–1558 a c. 1578; fotografias e Catarina Oliveira, MNRL 2020). Lido no MDS: um
    círculo de 11,8 m de raio, cortado a norte por uma frente reta (y = −50,8) onde abre o portal; o
    terrapleno (17,05 m) só ocupa a parte sul, até y = −57,3, e entre ele e a frente fica um pátio ao nível
    da praça (14,3 m). Pano em talude pintado de amarelo-ocre com cordão de pedra; no terrapleno, parapeito
    grosso com merlões e canhoneiras; na frente, parapeito liso. Portal de pedra do século XVIII com
    frontão triangular, cartela e armas, e a sineira com varanda no bordo do terrapleno, por trás do portal
    (pico do MDS a 23,1 m; o pináculo, fino, não aparece)."""
    C, R, yc, TOP = FORTIM["c"], FORTIM["r"], FORTIM["corda"], FORTIM["topo"]
    yf, CH = FORTIM["frente"], FORTIM["chao"]
    # o círculo do lado do mar (de 160° a 380°) e, para a praça, dois panos retos até aos cantos da frente
    angs = np.linspace(math.radians(160), math.radians(380), 60)
    arco = np.array([C + R * np.array([math.cos(a), math.sin(a)]) for a in angs])
    FW, FE = np.array([C[0] - 6.0, yf]), np.array([C[0] + 5.5, yf])
    lado_e = np.array([arco[-1] + (FE - arco[-1]) * t for t in np.linspace(0, 1, 6)[1:]])
    frente = np.array([FE + (FW - FE) * t for t in np.linspace(0, 1, 9)[1:]])
    lado_o = np.array([FW + (arco[0] - FW) * t for t in np.linspace(0, 1, 6)[1:-1]])
    linha = np.vstack([arco, lado_e, frente, lado_o])  # anti-horário
    n = len(linha)
    terr = linha[:, 1] < yc + 0.4           # troços do terrapleno (parapeito grosso com merlões)
    fora = desvio(linha, -2.5, True)
    zg = np.minimum(cota_t(fora[:, 0], fora[:, 1]), CH - 0.2) - 0.6
    zc = TOP - 0.25  # cordão
    bat = np.clip(0.12 * (zc - zg), 0.2, 1.6)
    base = desvio(linha, -bat, True)
    for i in range(n):
        j = (i + 1) % n
        m.face([(*linha[i], zc - 0.15), (*base[i], zg[i]), (*base[j], zg[j]), (*linha[j], zc - 0.15)], OGA)
    rc = 0.2
    perf = [(0.0, zc - 0.15), (rc * 0.8, zc - 0.1), (rc, zc), (rc * 0.8, zc + 0.1), (0.0, zc + 0.15)]
    varrer(m, np.vstack([linha, linha[:1]]), perf, CAN, fechar=False)
    # parapeito: grosso e boleado no terrapleno, liso e fino na frente do pátio (fotografias)
    hp = 1.45
    TF = 17.6
    ztp = np.where(terr, TOP + hp, TF)
    esp = np.where(terr, 1.2, 0.65)
    zin = np.where(terr, TOP - 0.05, CH)
    L2 = np.vstack([linha, linha[:1]])
    ztp, esp, zin = (np.append(a, a[0]) for a in (ztp, esp, zin))
    perf = [(0.0, zc + 0.15), (0.0, ztp - 0.2), (-0.12, ztp), (-esp + 0.12, ztp), (-esp, ztp - 0.2), (-esp, zin)]
    varrer(m, L2, perf, [OGA, OGA, OGA, OGA, OGA], fechar=False)
    seg = np.linalg.norm(np.diff(arco, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    for s0 in np.arange(1.0, s[-1] - 3.0, 4.4):
        a_ = np.array([np.interp(s0, s, arco[:, 0]), np.interp(s0, s, arco[:, 1])])
        b_ = np.array([np.interp(s0 + 3.0, s, arco[:, 0]), np.interp(s0 + 3.0, s, arco[:, 1])])
        if max(a_[1], b_[1]) > yc - 0.5:
            continue
        t_ = unit(b_ - a_)
        no = np.array([t_[1], -t_[0]])
        m.caixa((a_ + b_) / 2 - no * 0.6, t_, np.linalg.norm(b_ - a_), 1.2, TOP + hp - 0.05, TOP + hp + 0.85, OGA)
        m.caixa((a_ + b_) / 2 - no * 0.6, t_, np.linalg.norm(b_ - a_) - 0.3, 0.9, TOP + hp + 0.85, TOP + hp + 1.0, OGA)
    # terrapleno lajeado (com a cor da ortoimagem), até ao bordo sobre o pátio
    circ = shapely.Polygon(linha).buffer(-1.2)
    tp = circ.intersection(shapely.box(-1e3, -1e3, 1e3, yc))
    if tp.geom_type == "Polygon":
        q = np.array(tp.exterior.coords)[:-1]
        if area(q) < 0:
            q = q[::-1]
        m.face([(*p, TOP) for p in q], CHAO_P)
    # pano do terrapleno virado ao pátio, com parapeito baixo e uma porta para as casernas
    _cr = shapely.LineString([(-1e3, yc), (1e3, yc)]).intersection(shapely.Polygon(linha))
    xw, xe = _cr.bounds[0] + 1.0, _cr.bounds[2] - 1.0
    Qc = Quadro(m, np.array([C[0], yc]), (0, 1))
    Qc.bloco(xw - C[0], xe - C[0], CH - 0.3, TOP + 1.0, -1.2, 0.0, OGA)
    Qc.bloco(xw - C[0], xe - C[0], TOP + 1.0, TOP + 1.12, -1.25, 0.05, CAN)
    for sp in (-6.0, 5.0):
        Qd = Quadro(m, np.array([C[0] + sp, yc]), (0, 1))
        Qd.bloco(-0.85, 0.85, CH, CH + 2.75, 0.0, 0.12, CAN)
        Qd.vao(1.2, CH, 2.0, 0.13, SOM)
    # chão do pátio
    pt = shapely.Polygon(linha).buffer(-0.7).intersection(shapely.box(-1e3, yc, 1e3, 1e3))
    if pt.geom_type == "Polygon":
        q = np.array(pt.exterior.coords)[:-1]
        if area(q) < 0:
            q = q[::-1]
        m.face([(*p, CH + 0.02) for p in q], CHAO_P)
    # portal na frente reta, com frontão e cartela (fotografias)
    meio = np.array([C[0], yf])
    nf = np.array([0.0, 1.0])
    g = float(cota_t(*(meio + nf * 3)))
    g = min(max(g, CH - 0.4), CH)
    Q = Quadro(m, meio + nf * 0.02, nf)
    Q.bloco(-2.3, 2.3, g - 0.3, g + 0.18, 0.0, 0.9, CAN)            # degrau
    Q.vao(1.6, g + 0.18, 2.5, 0.01, SOM, reto=True)
    Q.vao(1.6, g + 0.18, 2.5, -0.64, SOM, reto=True)
    for sg in (-1, 1):                                                # ombreiras e pilastras
        s0, s1 = sorted((sg * 0.8, sg * 1.2))
        Q.bloco(s0, s1, g + 0.18, g + 2.9, 0.0, 0.16, CAN)
        s0, s1 = sorted((sg * 1.2, sg * 1.7))
        Q.bloco(s0, s1, g + 0.18, g + 3.25, 0.0, 0.3, CAN)
    for k_ in range(1, 7):                                            # grade de ferro, aberta
        Q.bloco(-0.8 + 0.23 * k_ - 0.02, -0.8 + 0.23 * k_ + 0.02, g + 0.18, g + 2.7, 0.05, 0.08, FER)
    Q.bloco(-0.8, 0.8, g + 2.7, g + 2.9, 0.0, 0.16, CAN)              # verga
    Q.bloco(-1.9, 1.9, g + 3.25, g + 3.6, 0.0, 0.42, CAN)             # entablamento
    Q.bloco(-1.1, 1.1, g + 3.6, g + 4.6, 0.0, 0.2, CAN)               # cartela com a inscrição
    Q.bloco(-0.95, 0.95, g + 3.75, g + 4.45, 0.2, 0.24, LIS)
    zf = g + 3.6
    Q.placa([(-2.0, zf), (2.0, zf), (0.0, zf + 1.65)], 0.0, 0.25, CAN)   # frontão
    for sg in (-1, 1):
        Q.placa([(sg * 2.1, zf - 0.02), (sg * 1.9, zf - 0.02), (0.0, zf + 1.6), (0.0, zf + 1.85)], 0.0, 0.42, CAN)
    Q.escudo(0.0, zf + 0.25, 0.55, 0.62, 0.25, 0.36, CAN, coroa=True)
    pinaculo(Q, 0.0, zf + 1.8, 0.2, esc=0.9, tipo="bola")
    Q.bloco(-5.4, -3.6, g + 1.1, g + 2.0, 0.0, 0.08, LIS)              # lápide à esquerda (fotografia)
    # sineira: corpo amarelo, varanda, vão de arco alto com o sino, cornija e remate bolboso com pináculo
    cs = np.array([C[0], yc - 0.2])
    Qs = Quadro(m, cs, nf)
    zb = TOP + 1.12
    Qs.bloco(-1.45, 1.45, TOP - 0.5, zb + 0.9, -2.2, 0.0, OGA)
    Qs.bloco(-1.6, 1.6, zb + 0.9, zb + 1.05, -2.3, 0.4, CAN)          # varanda
    for k_ in range(8):
        Qs.bloco(-1.5 + 0.43 * k_ - 0.02, -1.5 + 0.43 * k_ + 0.02, zb + 1.05, zb + 1.8, 0.32, 0.36, FER)
    Qs.bloco(-1.6, 1.6, zb + 1.8, zb + 1.85, 0.3, 0.38, FER)
    wv, za = 1.2, zb + 1.0
    hn = 2.0
    top = [(-1.35, za), (-wv / 2, za), (-wv / 2, za + hn)]
    top += [(-wv / 2 * math.cos(math.pi * k / 12), za + hn + wv / 2 * math.sin(math.pi * k / 12)) for k in range(1, 12)]
    top += [(wv / 2, za + hn), (wv / 2, za), (1.35, za), (1.35, za + 3.3), (-1.35, za + 3.3)]
    for d0, d1 in ((-0.2, 0.0), (-2.2, -2.0)):
        Qs.placa(top, d0, d1, OGA)
    for sg in (-1, 1):
        s0, s1 = sorted((sg * 1.35, sg * 1.15))
        Qs.bloco(s0, s1, za, za + 3.3, -2.0, -0.2, OGA)
    Qs.arco_sz(0.0, wv / 2, wv / 2 + 0.17, za + hn, 0.0, 0.07, CAN, k=12, aduelas=True)
    for sg in (-1, 1):
        s0, s1 = sorted((sg * wv / 2, sg * (wv / 2 + 0.17)))
        Qs.bloco(s0, s1, za, za + hn, 0.0, 0.07, CAN)
    Qs.bloco(-0.75, 0.75, za - 0.08, za + 0.04, -0.2, 0.1, CAN)        # peitoril
    sino(m, Qs.p(0, 0, -1.1)[:2], za + 0.8, r=0.36)
    zt = za + 3.3
    Qs.bloco(-1.5, 1.5, zt, zt + 0.22, -2.35, 0.15, CAN)               # cornija
    # remate bolboso (perfil em S) com pináculo, como na fotografia
    rem = [(-1.3, zt + 0.22), (1.3, zt + 0.22), (1.2, zt + 0.6), (0.85, zt + 0.95), (0.55, zt + 1.15), (0.4, zt + 1.5),
           (0.25, zt + 1.75), (0.0, zt + 1.85), (-0.25, zt + 1.75), (-0.4, zt + 1.5), (-0.55, zt + 1.15), (-0.85, zt + 0.95),
           (-1.2, zt + 0.6)]
    for d0, d1 in ((-2.1, -0.1),):
        Qs.placa(rem, d0, d1, OGA)
    Qs.bloco(-0.2, 0.2, zt + 1.85, zt + 2.05, -1.3, -0.9, CAN)
    Qs.esfera(0.0, zt + 2.25, -1.1, 0.2, CAN)
    MEDIDAS["fortim"] = {"raio": R, "terrapleno": TOP, "parapeito": TOP + hp, "sineira": round(zt + 2.45, 1), "porta": g}
    return linha


# ---------------------------------------------------------------- ponte de arcos e porta

PONTE = dict(a=np.array([61.3, 49.4]), b=np.array([66.9, 79.0]), deck=10.75, largura=3.6)


def ponte_arcos(m):
    """Ponte de quatro arcos sobre o fosso, rebocada e pintada de amarelo (fotografias), da porta na
    muralha até ao campo a norte. O tabuleiro está à cota medida no LiDAR (10,75 m)."""
    A, B, zd, W = PONTE["a"], PONTE["b"], PONTE["deck"], PONTE["largura"]
    u = unit(B - A)
    L = float(np.linalg.norm(B - A))
    nq = np.array([-u[1], u[0]])
    Q = Quadro(m, A, nq)  # s ao longo da ponte (de A para B), d para o lado
    if (Q.t @ u) < 0:
        Q = Quadro(m, A, -nq)
    z0 = 1.5
    vaos = [4.6, 5.4, 5.4, 5.0]
    pilar = 2.0
    s = 1.8
    arcos = []
    for v in vaos:
        arcos.append((s, s + v))
        s += v + pilar
    esc = L - (s - pilar)
    sil = [(0.0, z0), (0.0, zd), (L, zd), (L, z0)]
    for a_, b_ in reversed(arcos):
        r_ = (b_ - a_) / 2
        zn = zd - 0.75 - r_
        sil += [(b_, z0), (b_, zn)]
        sil += [((a_ + b_) / 2 + r_ * math.cos(math.pi * k / 12), zn + r_ * math.sin(math.pi * k / 12)) for k in range(1, 12)]
        sil += [(a_, zn), (a_, z0)]
    Q.placa(sil, -W / 2, W / 2, OGA)
    # guardas com capeamento
    PAT = 2.5  # patamar de pedra junto à porta, com guardas de ferro (fotografia "Entrada no Forte de Peniche 02")
    for sg in (-1, 1):
        d0, d1 = sorted((sg * W / 2, sg * (W / 2 - 0.35)))
        Q.placa([(PAT, zd), (L, zd), (L, zd + 1.0), (PAT, zd + 1.0)], d0, d1, OGA)
        d0, d1 = sorted((sg * (W / 2 + 0.05), sg * (W / 2 - 0.4)))
        Q.placa([(PAT, zd + 1.0), (L, zd + 1.0), (L, zd + 1.1), (PAT, zd + 1.1)], d0, d1, OCR)
        d0, d1 = sorted((sg * W / 2, sg * (W / 2 - 0.4)))
        Q.placa([(0.4, zd), (PAT, zd), (PAT, zd + 0.3), (0.4, zd + 0.3)], d0, d1, LIS)  # lancil
        dm = sg * (W / 2 - 0.2)
        for sp in (0.5, 1.5, PAT - 0.06):
            Q.placa([(sp - 0.03, zd + 0.3), (sp + 0.03, zd + 0.3), (sp + 0.03, zd + 1.0), (sp - 0.03, zd + 1.0)], dm - 0.03, dm + 0.03, FER)
        for hz, e in ((0.62, 0.04), (0.96, 0.06)):
            Q.placa([(0.45, zd + hz), (PAT, zd + hz), (PAT, zd + hz + e), (0.45, zd + hz + e)], dm - 0.025, dm + 0.025, FER)
        # nichos de arco nos pilares, dos dois lados (fotografias)
        for a_, b_ in arcos[:-1]:
            sp = b_ + pilar / 2
            pts = [(sp - 0.3, zd - 4.6), (sp + 0.3, zd - 4.6), (sp + 0.3, zd - 3.4)]
            pts += [(sp + 0.3 * math.cos(math.pi * k / 8), zd - 3.4 + 0.3 * math.sin(math.pi * k / 8)) for k in range(1, 8)]
            pts += [(sp - 0.3, zd - 3.4)]
            dd = sg * (W / 2 + 0.01)
            Q.m.face([Q.p(a, b, dd) for a, b in (pts if sg > 0 else pts[::-1])], SOM)
    # lajeado do tabuleiro
    Q.placa([(0, zd), (L, zd), (L, zd + 0.04), (0, zd + 0.04)], -W / 2 + 0.35, W / 2 - 0.35, LIS)
    MEDIDAS["ponte"] = {"comprimento": L, "tabuleiro": zd, "arcos": len(vaos), "vaos": vaos}
    return L


def porta_ponte(m, mg):
    """Porta na muralha onde chega a ponte (fotografia Commons "Entrada no Forte de Peniche 02"; revisão do perito,
    13 nov. 2026): um frontispício de cantaria clara contínuo do chão à guarita. Pilastras com meios frontões
    para fora, entablamento, painel com as armas reais coroadas sob um frontão curvo, a guarita como caixa
    quadrada de cantos chanfrados assente no painel, com janela envidraçada de caixilho escuro, cornija, cúpula
    e pináculo, e uma bola de pedra de cada lado; porta de madeira escura ferrada, fechada, em arco de volta
    perfeita. Materiais: 0 cantaria clara, 1 sombra, 2 madeira escura, 3 ferro, 4 vidro."""
    CC, SO, MD, FE, VI = range(5)
    A, z = PONTE["a"], PONTE["deck"]
    u = unit(PONTE["b"] - A)
    F0 = A - u * 0.15
    Q = Quadro(m, F0, u)
    # topo da muralha por cima da porta: LiDAR, mas pelo menos 6,6 m acima da soleira (na fotografia a muralha
    # chega às bolas de pedra, c. 7 m acima da soleira, com o vão de 2,3 m como escala)
    ZT = max(float(topo_s(*(A - u * 1.5))), z + 6.6)
    w, zn = 1.15, z + 2.35  # vão de 2,3 m, fecho a 3,5 m
    D, DF = 0.3, -1.6  # face do frontispício e fundo (metido na muralha, que é em talude)
    # pano com o vão, aduelas, fecho e pedras das ombreiras
    Q.pano_vao(-2.45, 2.45, z - 0.5, z + 4.2, DF, D, CC, w, z, zn)
    Q.arco_sz(0.0, w, w + 0.42, zn, D, D + 0.05, CC, k=13, aduelas=True)
    Q.bloco(-0.2, 0.2, zn + w - 0.02, z + 4.2, D, D + 0.1, CC)
    for sg in (-1, 1):
        for k, z0 in enumerate(np.arange(z, zn - 0.1, 0.59)):
            a, b = sorted((sg * w, sg * (w + (0.42 if k % 2 == 0 else 0.3))))
            Q.bloco(a, b, z0 + 0.015, min(z0 + 0.575, zn), D, D + 0.05, CC)
    # porta de madeira fechada, recuada 0,4 m, com tábuas, juntas e ferragens
    Q.tunel(w, z, zn, D, 0.42, CC, None)
    arco = [(-w, z), (w, z), (w, zn)] + [(w * math.cos(math.pi * k / 16), zn + w * math.sin(math.pi * k / 16)) for k in range(1, 16)] + [(-w, zn)]
    Q.placa(arco, D - 0.55, D - 0.4, MD)
    dfo = D - 0.4
    for st in np.arange(-w + 0.29, w - 0.1, 0.29):
        hz = zn + math.sqrt(max(w * w - st * st, 0.0))
        Q.bloco(st - 0.012, st + 0.012, z, hz - 0.02, dfo, dfo + 0.008, SO if abs(st) > 0.05 else FE)
    for hz in (0.45, 1.25, 2.05, 2.85):
        hw = w if z + hz < zn else math.sqrt(max(w * w - (z + hz - zn) ** 2, 0.0))
        Q.bloco(-hw + 0.04, hw - 0.04, z + hz, z + hz + 0.07, dfo, dfo + 0.02, FE)
    for sg in (-1, 1):
        Q.bloco(sg * 0.12 - 0.06, sg * 0.12 + 0.06, z + 1.1, z + 1.35, dfo, dfo + 0.03, FE)  # fechaduras
    # pilastras com base e capitel
    for sg in (-1, 1):
        a, b = sorted((sg * 1.6, sg * 2.3))
        Q.bloco(a, b, z - 0.3, z + 3.95, D, D + 0.22, CC)
        Q.bloco(a - 0.05, b + 0.05, z - 0.3, z + 0.35, D, D + 0.28, CC)
        Q.bloco(a - 0.06, b + 0.06, z + 3.95, z + 4.2, D, D + 0.3, CC)
    # entablamento
    Q.bloco(-2.4, 2.4, z + 4.2, z + 4.45, D, D + 0.3, CC)
    Q.bloco(-2.55, 2.55, z + 4.45, z + 4.65, D, D + 0.42, CC)
    zb = z + 4.65
    # meios frontões virados para fora, com o pedestal e a bola de pedra no canto de dentro
    for sg in (-1, 1):
        Q.placa([(sg * 1.5, zb), (sg * 2.45, zb), (sg * 1.5, zb + 0.95)], DF, D + 0.2, CC)
        faixa_inclinada(Q, [(sg * 2.6, zb), (sg * 1.5, zb + 1.03)] if sg > 0 else [(sg * 1.5, zb + 1.03), (sg * 2.6, zb)], 0.16, D, D + 0.4, CC)
        a, b = sorted((sg * 1.3, sg * 1.75))
        zp = ZT - 0.45
        Q.bloco(a, b, zb, zp, DF, D + 0.12, CC)
        Q.bloco(a - 0.05, b + 0.05, zp, zp + 0.12, DF, D + 0.18, CC)
        Q.esfera(sg * 1.52, zp + 0.36, D - 0.1, 0.24, CC)
    # painel com as armas reais coroadas e frontão curvo por cima
    Q.bloco(-1.25, 1.25, zb, z + 5.75, DF, D + 0.18, CC)
    Q.escudo(0.0, zb + 0.1, 0.78, 0.74, D + 0.18, D + 0.36, CC)
    Q.bloco(-1.08, -0.52, zb + 0.15, zb + 0.95, D + 0.18, D + 0.22, CC)  # lápide, à direita das armas (fotografia)
    zf, f = z + 5.75, 0.3
    R = (1.35 ** 2 + f ** 2) / (2 * f)
    zc = zf + f - R
    t0 = math.asin(1.35 / R)
    fora = [(R * math.sin(t), zc + R * math.cos(t)) for t in np.linspace(-t0, t0, 13)]
    dentro = [(s_ * (R - 0.18) / R, zc + (zz - zc) * (R - 0.18) / R) for s_, zz in fora]
    Q.placa([(-1.4, zf - 0.05), (1.4, zf - 0.05)] + fora[::-1], DF, D + 0.1, CC)  # tímpano
    Q.placa(fora + dentro[::-1], D + 0.1, D + 0.36, CC)
    # guarita: base (com dois frisos), corpo de cantos chanfrados, janela, cornija, cúpula e pináculo
    L0, L1, ch = -1.55, D + 0.15, 0.24
    plan_sd = [(-1.0, L0 + ch), (-1.0 + ch, L0), (1.0 - ch, L0), (1.0, L0 + ch),
               (1.0, L1 - ch), (1.0 - ch, L1), (-1.0 + ch, L1), (-1.0, L1 - ch)]
    cs = (L0 + L1) / 2

    def planta(f_=1.0):
        return [Q.p(s_ * f_, 0, cs + (d_ - cs) * f_)[:2] for s_, d_ in plan_sd]

    zg0 = z + 5.95
    m.prisma(planta(), zf - 0.05, zg0, CC)
    m.prisma(planta(1.06), zg0, zg0 + 0.12, CC)
    m.prisma(planta(), zg0 + 0.12, ZT, CC)
    m.prisma(planta(1.08), ZT, ZT + 0.14, CC)
    zt1 = ZT + 2.0
    m.prisma(planta(), ZT + 0.14, zt1, CC)
    for sg in (-1, 1):  # pilastras dos cunhais da frente
        a, b = sorted((sg * 0.62, sg * 0.98))
        Q.bloco(a, b, ZT + 0.14, zt1, L1 - 0.05, L1 + 0.05, CC)
    # janela envidraçada de caixilho escuro (3 × 4 vidros)
    Q.bloco(-0.55, 0.55, ZT + 0.5, ZT + 1.55, L1, L1 + 0.07, CC)
    Q.bloco(-0.42, 0.42, ZT + 0.62, ZT + 1.42, L1, L1 + 0.075, VI)
    for st in (-0.42, -0.21, 0.0, 0.21, 0.42):
        Q.bloco(st - 0.022, st + 0.022, ZT + 0.62, ZT + 1.42, L1 + 0.075, L1 + 0.09, FE)
    for hz in (0.62, 0.89, 1.15, 1.42):
        Q.bloco(-0.44, 0.44, ZT + hz - 0.022, ZT + hz + 0.022, L1 + 0.075, L1 + 0.09, FE)
    m.prisma(planta(1.1), zt1, zt1 + 0.12, CC)
    m.prisma(planta(1.16), zt1 + 0.12, zt1 + 0.26, CC)
    prev, zprev = planta(1.0), zt1 + 0.26
    for k in range(1, 6):
        fi = math.pi / 2 * k / 5
        cur, zz = planta(max(math.cos(fi), 0.08)), zt1 + 0.26 + 0.75 * math.sin(fi)
        for i in range(len(prev)):
            j = (i + 1) % len(prev)
            m.face([(*prev[i], zprev), (*prev[j], zprev), (*cur[j], zz), (*cur[i], zz)], CC)
        prev, zprev = cur, zz
    m.face([(*q, zprev) for q in prev], CC)
    c_ = np.array(Q.p(0, 0, cs)[:2])
    m.cilindro(c_, 0.1, zprev - 0.02, zprev + 0.22, CC, k=8)
    m.esfera((c_[0], c_[1], zprev + 0.36), 0.15, CC, k=10)
    m.cilindro(c_, 0.03, zprev + 0.48, zprev + 0.62, CC, k=6)
    MEDIDAS["porta"] = {"soleira": round(z, 2), "vao": 2 * w, "fecho": round(zn + w - z, 2), "largura": 5.1,
                        "topo_muralha": round(ZT, 1), "guarita_topo": round(zprev + 0.62, 1)}
    return F0, ZT


def tunel_agua(m):
    """O túnel da água do mar entre a baía e o fosso: abóbada de berço em alvenaria (só a face de dentro, que se vê
    pelas bocas), a água lá dentro à cota do mar e, em cada boca, um paramento de alvenaria com o arco de aduelas
    (fotografias). Materiais: 0 cantaria, 1 alvenaria, 2 água escura."""
    w, zn, zf_ = TUNEL["largura"], TUNEL["nascenca"], TUNEL["fundo"]
    r = w / 2
    eixo = densificar(TUNEL["eixo"], 1.0)
    k = 12
    # perfil de dentro: (afastamento lateral, cota), do fundo de um lado ao fundo do outro, pela abóbada
    perf = [(-r, zf_), (-r, zn)] + [(-r * math.cos(math.pi * i / k), zn + r * math.sin(math.pi * i / k)) for i in range(1, k)] \
        + [(r, zn), (r, zf_)]
    n_ = len(eixo)
    tang = np.array([unit(eixo[min(i + 1, n_ - 1)] - eixo[max(i - 1, 0)]) for i in range(n_)])
    lado = np.column_stack([-tang[:, 1], tang[:, 0]])
    anel = [[(*(eixo[i] + lado[i] * o), z) for o, z in perf] for i in range(n_)]
    for i in range(n_ - 1):
        c_ax = np.array([*(eixo[i] + eixo[i + 1]) / 2, zn])
        for j in range(len(perf) - 1):
            f_ = [anel[i][j], anel[i + 1][j], anel[i + 1][j + 1], anel[i][j + 1]]
            P_ = np.array(f_)
            nn = np.cross(P_[1] - P_[0], P_[3] - P_[0])
            if nn @ (c_ax - P_.mean(0)) < 0:
                f_ = f_[::-1]
            m.face(f_, 1)
    # a água dentro do túnel
    for i in range(n_ - 1):
        q_ = [eixo[i] - lado[i] * r, eixo[i + 1] - lado[i + 1] * r, eixo[i + 1] + lado[i + 1] * r, eixo[i] + lado[i] * r]
        if area(np.array(q_)) < 0:
            q_ = q_[::-1]
        m.face([(*p, NIVEL - 0.03) for p in q_], 2)
    # bocas: paramento com o vão em arco (furado), aduelas e pés-direitos de cantaria
    for b, n, comp, topo in TUNEL["bocas"]:
        Q = Quadro(m, b, n)
        largo, z_alto = r + 2.2, (topo or TUNEL["paramento"])
        # paramento à volta do arco, na face d = 0, virado para fora (normal n)
        arco = [(r * math.cos(math.pi * i / k), zn + r * math.sin(math.pi * i / k)) for i in range(k + 1)]  # de s=+r a s=-r
        pecas = [[(r, zf_), (largo, zf_), (largo, z_alto), (r + 0.0, zn)],
                 [(-largo, zf_), (-r, zf_), (-r, zn), (-largo, z_alto)]]
        for (sa, za), (sb, zb) in zip(arco[:-1], arco[1:]):
            pecas.append([(sa, za), (sb, zb), (sb, z_alto), (sa, z_alto)])
        pecas.append([(r, zn), (largo, z_alto), (r, z_alto)])
        pecas.append([(-largo, z_alto), (-r, zn), (-r, z_alto)])
        for pc in pecas:
            P_ = [Q.p(s_, z_, 0.0) for s_, z_ in pc]
            A_ = np.array(P_)
            nn = np.cross(A_[1] - A_[0], A_[2] - A_[0])
            if nn[:2] @ n < 0:
                P_ = P_[::-1]
            m.face(P_, 1)
        # o paramento entra 1,5 m pelo terreno dos lados e por cima (sem frestas para o escuro de dentro)
        for sg in (-1, 1):
            Q.bloco(sg * largo, sg * (largo + 1.5), zf_, z_alto, -2.5, 0.0, 1)
        Q.bloco(-largo - 1.5, largo + 1.5, z_alto, z_alto + 0.8, -2.5, 0.0, 1)
        # arco de aduelas e pés-direitos
        Q.arco_sz(0.0, r, r + 0.5, zn, 0.0, 0.18, 0, k=13, aduelas=True)
        for sg in (-1, 1):
            s0, s1 = sorted((sg * r, sg * (r + 0.5)))
            Q.bloco(s0, s1, zf_ + 0.6, zn, 0.0, 0.18, 0)
    MEDIDAS["tunel"] = {"comprimento": float(np.sum(np.linalg.norm(np.diff(TUNEL["eixo"], axis=0), axis=1))),
                        "largura": w, "fecho": TUNEL["fecho"], "agua": NIVEL}


def cruzeiro(m, c):
    """Cruz de pedra sobre plinto em degraus, no terraço junto ao pavilhão amarelo (fotografia)."""
    g = float(cota_t(*c))
    z = g
    for k_, (l_, h_) in enumerate(((2.2, 0.3), (1.6, 0.3), (1.0, 0.7))):
        m.caixa(c, (1, 0), l_, l_, z - (0.4 if k_ == 0 else 0), z + h_, CAN)
        z += h_
    Q = Quadro(m, c, (0, -1))
    cruz(Q, 0.0, g + 1.3, 0.0, alt=3.2, mat=CAN, esp=0.32)


# ---------------------------------------------------------------- palmeiras, rochedos e mar

def palmeira(x, y, z, alt):
    """Palmeira das Canárias (fotografias): tronco grosso e escamoso e coroa de folhas pinadas em arco."""
    b = len(MV)
    k = 8
    for j, t in enumerate(np.linspace(0, 1, 6)):
        r = 0.42 - 0.12 * t
        for i in range(k):
            a = 2 * math.pi * i / k
            MV.append((x + r * math.cos(a), y + r * math.sin(a), z - 0.3 + t * alt))
            MC.append(tuple(np.array((0.46, 0.38, 0.29)) * (0.85 + 0.25 * ((i + j) % 2))))
    for j in range(5):
        for i in range(k):
            i2 = (i + 1) % k
            MF.append([b + j * k + i, b + j * k + i2, b + (j + 1) * k + i2, b + (j + 1) * k + i])
    zt = z - 0.3 + alt
    _lobo(x, y, zt, 0.6, 0.5, (0.42, 0.40, 0.22), zt - 0.5, zt + 0.5, 1, 0.05)
    nf = 22
    for i in range(nf):
        a = 2 * math.pi * i / nf + rng.normal(0, 0.1)
        sobe = 0.45 if i % 2 == 0 else -0.1
        comp = 4.2 + rng.normal(0, 0.3)
        pts = []
        for t in np.linspace(0, 1, 7):
            d = comp * t
            h = sobe * comp * t - 1.5 * (t ** 2) * comp * 0.45
            pts.append(np.array([x + d * math.cos(a), y + d * math.sin(a), zt + 0.2 + h]))
        dirl = np.array([-math.sin(a), math.cos(a), 0.0])
        b = len(MV)
        for j, p in enumerate(pts):
            w = 0.75 * math.sin(math.pi * min(0.95, 0.12 + j / 6))
            MV.append(tuple(p - dirl * w + [0, 0, -0.12 * w]))
            MV.append(tuple(p))
            MV.append(tuple(p + dirl * w + [0, 0, -0.12 * w]))
            cor = tuple(np.array((0.30, 0.45, 0.20)) * (0.8 + 0.3 * j / 6))
            MC.extend([cor, cor, cor])
        for j in range(len(pts) - 1):
            o = b + 3 * j
            MF.append([o, o + 1, o + 4, o + 3])
            MF.append([o + 1, o + 2, o + 5, o + 4])


RELVADOS = []
# relvados com lancil branco (fotografias: em frente do Fortim e ao longo da muralha da costa, até ao cruzeiro)
_costa = shapely.LineString([(12.0, -44.0), (14.0, -36.0), (24.0, -28.0), (45.0, -1.0)])
_faixa = shapely.Polygon(list(_costa.offset_curve(3.6).coords) + list(_costa.offset_curve(9.5).coords)[::-1]).buffer(0)
for _corte in (shapely.box(-1e3, -1e3, 1e3, -33.0), shapely.box(-1e3, -31.0, 1e3, -13.0), shapely.box(-1e3, -11.0, 1e3, 1e3)):
    _pp = _faixa.intersection(_corte)
    if _pp.area > 10:
        RELVADOS.append(_pp)
RELVADOS += [shapely.Polygon([(-28.0, -47.5), (-13.2, -47.5), (-13.2, -33.0), (-31.0, -33.0)]),
             shapely.Polygon([(-7.8, -47.5), (4.0, -47.5), (8.0, -42.0), (8.0, -33.0), (-7.8, -33.0)])]


def relvados(m):
    """Relva (malha que segue o terreno, 5 cm acima) e lancil branco de 0,2 × 0,12 m à volta."""
    from scipy.spatial import Delaunay
    area_ = 0.0
    for poly in RELVADOS:
        poly = poly.buffer(0)
        if poly.geom_type != "Polygon":
            continue
        area_ += poly.area
        borda = densificar(np.array(poly.exterior.coords)[:-1], 1.0, True)
        x0, y0, x1, y1 = poly.bounds
        gx, gy = np.meshgrid(np.arange(x0, x1, 1.0), np.arange(y0, y1, 1.0))
        gx, gy = gx.ravel(), gy.ravel()
        k = shapely.contains_xy(poly.buffer(-0.4), gx, gy)
        pts = np.vstack([borda, np.column_stack([gx[k], gy[k]])])
        zz = cota_t(pts[:, 0], pts[:, 1]) + 0.06
        tri = Delaunay(pts).simplices
        cen = pts[tri].mean(1)
        tri = tri[shapely.contains_xy(poly, cen[:, 0], cen[:, 1])]
        for t in tri:
            a, b, c_ = pts[t]
            if (b - a)[0] * (c_ - a)[1] - (b - a)[1] * (c_ - a)[0] < 0:
                t = t[::-1]
            m.face([(*pts[i], zz[i]) for i in t], 1)
        anel = np.vstack([borda, borda[:1]])
        if area(borda) < 0:
            anel = anel[::-1]
        zb = cota_t(anel[:, 0], anel[:, 1])
        varrer(m, anel, [(0.0, zb - 0.1), (0.0, zb + 0.12), (0.2, zb + 0.12), (0.2, zb - 0.1)], 0, fechar=False)
    MEDIDAS["relvados"] = round(area_)


def rochedo(x, y, z, r, achata, semente):
    """Penedo de calcário: icosaedro deformado, mais claro em cima e escuro e molhado abaixo da linha de água."""
    v0, f0 = ICO[2]
    g = np.random.default_rng(semente)
    b = len(MV)
    esc = np.array([r * g.uniform(0.8, 1.3), r * g.uniform(0.7, 1.1), r * achata])
    rot = g.uniform(0, 2 * math.pi)
    cr, sr = math.cos(rot), math.sin(rot)
    v = (v0 + g.normal(0, 0.07, v0.shape) + 0.12 * np.sin(v0 * g.uniform(2, 4, 3))) * esc
    v = np.column_stack([v[:, 0] * cr - v[:, 1] * sr, v[:, 0] * sr + v[:, 1] * cr, v[:, 2]])
    v[:, 2] = np.where(v[:, 2] > 0, v[:, 2] * g.uniform(0.6, 1.0), v[:, 2])
    v += [x, y, z]
    MV.extend(map(tuple, v))
    MF.extend([[b + i for i in fa] for fa in f0])
    tom = np.array((0.66, 0.62, 0.53)) * g.uniform(0.85, 1.08)
    for p in v:
        if p[2] < NIVEL + 0.15:
            MC.append(tuple(tom * 0.55 * np.array([0.9, 0.95, 0.82])))   # faixa molhada e das algas
        else:
            MC.append(tuple(np.clip(tom * (0.85 + 0.05 * (p[2] - z)), 0, 1)))


def construir_mar():
    """Plano do mar à cota NIVEL sobre a área com água (e 3 m para dentro da terra, por baixo das rochas),
    com a cor pela fundura (turquesa no baixio, azul-petróleo no largo) e espuma branca junto à costa."""
    passo = 1.0
    xs = np.arange(LXS[0] + 1, LXS[-1] - 1, passo)
    ys = np.arange(LYS[0] + 1, LYS[-1] - 1, passo)
    xs = xs[(xs >= CONT_XY[:, 0].min()) & (xs <= CONT_XY[:, 0].max())]
    ys = ys[(ys >= CONT_XY[:, 1].min()) & (ys <= CONT_XY[:, 1].max())]
    X, Y = np.meshgrid(xs, ys)
    agua_d = ndi.binary_dilation(AGUA, iterations=6)
    usa = amostra(agua_d.astype(float), X, Y) > 0.01
    prof = NIVEL - amostra(ZTER[0], X, Y)
    dcosta = amostra(D_AGUA, X, Y)
    idx = -np.ones(X.shape, int)
    idx[usa] = np.arange(usa.sum())
    V = np.stack([X[usa], Y[usa], np.full(usa.sum(), NIVEL)], 1)
    I, J = np.nonzero(usa[:-1, :-1] & usa[1:, :-1] & usa[:-1, 1:] & usa[1:, 1:])
    Fq = np.stack([idx[I, J], idx[I, J + 1], idx[I + 1, J + 1], idx[I + 1, J]], 1)
    p = np.clip(prof[usa], 0, 12)
    t = np.clip(p / 6.0, 0, 1)[:, None]
    raso, fundo = np.array([0.36, 0.72, 0.66]), np.array([0.07, 0.30, 0.38])
    cor = raso * (1 - t) + fundo * t
    ruido_ = fbm(X[usa] / 3, Y[usa] / 3, 1.0, 12)
    esp = np.clip((1.6 - dcosta[usa]) / 1.6, 0, 1) * (0.6 + 0.6 * ruido_)
    esp = np.clip(esp + np.clip(0.4 - p, 0, 0.4), 0, 1)[:, None]
    cor = cor * (1 - esp) + np.array([0.88, 0.93, 0.92]) * esp
    me = bpy.data.meshes.new("mar")
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(Fq.size)
    me.loops.foreach_set("vertex_index", Fq.astype(np.int32).ravel())
    me.polygons.add(len(Fq))
    me.polygons.foreach_set("loop_start", (np.arange(len(Fq)) * 4).astype(np.int32))
    me.update(calc_edges=True)
    ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", np.hstack([cor ** 2.2, np.ones((len(cor), 1))]).astype(np.float32).ravel())
    me.color_attributes.active_color = ca
    me.materials.append(MATS["mar"])
    ob = bpy.data.objects.new("Terreno_mar", me)
    bpy.context.scene.collection.objects.link(ob)
    ob["elemento"] = "mar"
    ob.parent = GRUPOS["Terreno"]
    # simplifica (as zonas fundas e lisas ficam com poucos triângulos)
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = min(1.0, 70000 / (2 * len(Fq)))
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear()
    ob.data = me2
    me2.materials.clear()
    me2.materials.append(MATS["mar"])
    uv_mundo(ob, 9.0)
    MEDIDAS["mar"] = {"nivel": NIVEL, "triangulos": len(me2.polygons)}


# ---------------------------------------------------------------- terreno

def textura_terreno():
    from PIL import Image
    img = np.asarray(Image.open(AQUI / "dados" / "orto.jpg"), float) / 255.0
    H = MDS - MDT
    tapa = ((CLS == 6) & (H > 1.5) & ~_dentro_plat) | (VEG & (H > 1.8)) | ((CLS == 0) & ~AGUA)
    tc = (tapa[:-1, :-1] | tapa[1:, :-1] | tapa[:-1, 1:] | tapa[1:, 1:])[::-1]
    tc = ndi.binary_dilation(tc, iterations=1)
    # As sombras do voo (ao pé dos muros, dos edifícios e das palmeiras) também se tapam: a maqueta faz as suas
    # sombras (auditoria de novembro de 2026: sombras carimbadas no chão). A água fica como está.
    lum = img.mean(-1)
    azul = img[..., 2] - (img[..., 0] + img[..., 1]) / 2
    sombra = ((lum < 0.72 * ndi.median_filter(lum, size=31)) & (lum < 0.42)) | ((azul > 0.0) & (lum < 0.40))
    sombra = ndi.binary_dilation(ndi.binary_opening(sombra, iterations=1), iterations=2)
    sombra &= ~(AGUA[:-1, :-1] | AGUA[1:, 1:])[::-1][:sombra.shape[0], :sombra.shape[1]]
    tc |= sombra[:tc.shape[0], :tc.shape[1]]
    bom = (~tc).astype(float)
    out = img.copy()
    falta = tc.copy()
    for sg in (1.5, 3, 6, 12, 24, 48):
        num = np.stack([ndi.gaussian_filter(img[..., k] * bom, sg) for k in range(3)], -1)
        den = ndi.gaussian_filter(bom, sg)[..., None]
        ok = (den[..., 0] > 0.15) & falta
        out[ok] = (num / np.maximum(den, 1e-6))[ok]
        falta &= ~ok
    out = np.stack([ndi.gaussian_filter(out[..., k], 0.6) for k in range(3)], -1)
    # debaixo de água: areia e rocha esverdeadas (só se vê junto à costa, pela espuma)
    ag = (AGUA[:-1, :-1] & AGUA[1:, 1:])[::-1]
    out[ag] = np.array([0.45, 0.47, 0.40])
    verde = np.clip((out[..., 1] - (out[..., 0] + out[..., 2]) / 2) * 8, 0, 1)[..., None]
    out = out * (1 - 0.3 * verde) + (out * np.array([0.9, 1.1, 0.82])) * 0.3 * verde
    cinza = out.mean(-1, keepdims=True)
    out = cinza + (out - cinza) * 1.35
    out = out * np.array([1.0, 0.985, 0.95])
    return np.clip(out * 0.86, 0, 1)


def construir_terreno():
    X, Y = XG, YG
    Z = MDT.copy()
    # fundo do mar: desce com a distância à costa (o MDT sobre a água é só uma interpolação)
    prof = np.clip(0.6 + 0.22 * D_AGUA, 0.6, 14.0)
    Z = np.where(AGUA, np.minimum(Z, NIVEL - prof), Z)
    # debaixo da ponte: o fundo do fosso entre as duas margens (o MDT tem o tabuleiro como chão)
    A, B = PONTE["a"], PONTE["b"]
    u = unit(B - A)
    nq = np.array([-u[1], u[0]])
    rel = np.stack([X - A[0], Y - A[1]], -1)
    s_, d_ = rel @ u, rel @ nq
    L = np.linalg.norm(B - A)
    caixa = (s_ > 1.0) & (s_ < L - 1.0) & (np.abs(d_) < 2.6)
    za = amostra(MDT, X + nq[0] * 3.6, Y + nq[1] * 3.6)
    zb = amostra(MDT, X - nq[0] * 3.6, Y - nq[1] * 3.6)
    Z = np.where(caixa, np.minimum(Z, (za + zb) / 2 - 0.4), Z)
    # debaixo dos arcos: o chão do fosso desce, para os vãos ficarem abertos como na fotografia
    sob = (s_ > 0.5) & (s_ < L - 0.5) & (np.abs(d_) < PONTE["largura"] / 2 + 0.3)
    rampa = np.clip((np.minimum(s_, L - s_) - 0.3) / 1.2, 0, 1)
    Z = np.where(sob, np.minimum(Z, 4.6 - 1.4 * rampa + 6.0 * (1 - rampa)), Z)
    # pátio do Fortim, entre o terrapleno e a frente: plano, à cota da praça
    C_f, R_f = FORTIM["c"], FORTIM["r"]
    patio = (((X - C_f[0]) ** 2 + (Y - C_f[1]) ** 2 < (R_f + 0.3) ** 2) | ((np.abs(X - C_f[0] + 0.25) < 7.0) & (Y < FORTIM["frente"] + 0.3))) \
        & (Y > FORTIM["corda"] - 0.5) & (Y < FORTIM["frente"] + 0.3)
    Z = np.where(patio, np.minimum(Z, FORTIM["chao"] - 0.05), Z)
    # em frente das escarpas de alvenaria da costa, o terreno desce ao pé do pano (o MDT faz uma rampa)
    Z0 = Z.copy()
    for es in ESCARPAS:
        lin, nout, pe, alto = es["linha"], es["nout"], es["pe"], es["alto"]
        for i in range(len(lin) - 1):
            if not (alto[i] and alto[i + 1]):
                continue
            quad = shapely.Polygon([lin[i] - nout[i] * 0.3, lin[i + 1] - nout[i + 1] * 0.3,
                                    lin[i + 1] + nout[i + 1] * 4.6, lin[i] + nout[i] * 4.6])
            if not quad.is_valid or quad.area < 0.05:
                continue
            quad = quad.buffer(0.45)  # sobrepõe os vizinhos: nos ângulos não ficam células por rebaixar
            x0, y0, x1, y1 = quad.bounds
            c0, c1 = np.searchsorted(LXS, [x0, x1])
            r0, r1 = np.searchsorted(LYS, [y0, y1])
            sx, sy = X[r0:r1 + 1, c0:c1 + 1], Y[r0:r1 + 1, c0:c1 + 1]
            if sx.size == 0:
                continue
            k_ = shapely.contains_xy(quad, sx, sy)
            zp = 0.5 * (pe[i] + pe[i + 1])
            blk = Z[r0:r1 + 1, c0:c1 + 1]
            blk[k_] = np.minimum(blk[k_], zp)
    # túnel: os canais das bocas descem abaixo da água; por cima das bocas, até 2,5 m para dentro, o terreno fica
    # pelo menos 0,5 m acima do fecho da abóbada (o MDT tem aí o entalhe da boca, que furaria a abóbada)
    Z = np.where(canais_tunel(X, Y), np.minimum(Z, TUNEL["fundo"] - 0.3), Z)
    for b, n, _, _ in TUNEL["bocas"]:
        t = np.array([n[1], -n[0]])
        rs = (X - b[0]) * t[0] + (Y - b[1]) * t[1]
        rd = (X - b[0]) * n[0] + (Y - b[1]) * n[1]
        teto = (np.abs(rs) < TUNEL["largura"] / 2 + 1.2) & (rd <= -0.2) & (rd > -2.5)
        Z = np.where(teto, np.maximum(Z, TUNEL["fecho"] + 0.9), Z)
    # sem agulhas de rocha à frente das escarpas: até 8 m delas, uma abertura morfológica (2,5 m) tira os picos
    # estreitos que o MDT deixa entre o pano e a arriba (o terreno só desce, nunca sobe)
    # e sem covas: onde duas escarpas se viram uma para a outra, o rebaixamento ao pé de uma abria um poço atrás
    # da outra; um fecho morfológico (4,5 m) enche essas covas sem subir acima do MDT (os declives para o mar ficam)
    if ESCARPAS:
        pe_lin = shapely.union_all([shapely.LineString(es["linha"]) for es in ESCARPAS])
        zona = shapely.contains_xy(pe_lin.buffer(8.0), X, Y) & ~AGUA
        Z = np.where(zona, np.minimum(Z0, ndi.grey_closing(Z, size=9)), Z)
        Z = np.where(zona, np.minimum(Z, ndi.grey_opening(Z, size=5)), Z)
        MEDIDAS["escarpas"] = {"troços": len(ESCARPAS), "celulas_limpas": int(zona.sum())}
    ZTER[:] = [Z]
    dentro_ = shapely.contains_xy(CONTORNO, X + LPASSO / 2, Y + LPASSO / 2)[:-1, :-1]
    usados = np.zeros(X.shape, bool)
    usados[:-1, :-1] |= dentro_
    usados[1:, :-1] |= dentro_
    usados[:-1, 1:] |= dentro_
    usados[1:, 1:] |= dentro_
    idx = -np.ones(X.shape, int)
    idx[usados] = np.arange(usados.sum())
    V = np.stack([X[usados], Y[usados], Z[usados]], 1)
    I, J = np.nonzero(dentro_)
    Fq = np.stack([idx[I, J], idx[I, J + 1], idx[I + 1, J + 1], idx[I + 1, J]], 1)
    ob = malha_rapida("Terreno", V, Fq, "terreno", GRUPOS["Terreno"], "terreno")
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = min(1.0, 320000 / (2 * len(Fq)))
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear()
    # as faces íngremes das arribas (mais de 50°) passam para um objeto de rocha, com textura de estratos
    co = np.empty(len(me2.vertices) * 3)
    me2.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    nrm = np.empty(len(me2.polygons) * 3)
    me2.polygons.foreach_get("normal", nrm)
    nrm = nrm.reshape(-1, 3)
    faces = [list(p.vertices) for p in me2.polygons]
    cen = np.array([co[f].mean(0) for f in faces])
    em_plat = shapely.contains_xy(PLAT_UNIAO, cen[:, 0], cen[:, 1]) if PLAT_UNIAO is not None else np.zeros(len(faces), bool)
    ingreme = ((nrm[:, 2] < 0.64) | shapely.contains_xy(GULLY, cen[:, 0], cen[:, 1])) & ~em_plat
    for nome, sel, mat in (("Terreno", ~ingreme, "terreno"), ("Terreno_rocha", ingreme, "rocha")):
        fs = [faces[i] for i in np.nonzero(sel)[0]]
        usa = np.unique(np.concatenate(fs))
        mapa = -np.ones(len(co), int)
        mapa[usa] = np.arange(len(usa))
        me = bpy.data.meshes.new(nome)
        me.from_pydata(co[usa].tolist(), [], [[int(mapa[v]) for v in f] for f in fs])
        me.update()
        me.materials.append(MATS[mat])
        me.shade_smooth()
        if nome == "Terreno":
            ob.data = me
            o2 = ob
        else:
            o2 = bpy.data.objects.new(nome, me)
            bpy.context.scene.collection.objects.link(o2)
            o2["elemento"] = "terreno"
            o2.parent = GRUPOS["Terreno"]
        if mat == "terreno":
            me.uv_layers.new(name="UVMap")
            c2 = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", c2)
            c2 = c2.reshape(-1, 3)
            vi = np.empty(len(me.loops), int)
            me.loops.foreach_get("vertex_index", vi)
            uu = (c2[vi, 0] - LXS[0]) / (LXS[-1] - LXS[0])
            vv = (c2[vi, 1] - LYS[0]) / (LYS[-1] - LYS[0])
            me.uv_layers.active.data.foreach_set("uv", np.stack([uu, vv], 1).ravel().astype(np.float32))
        else:
            uv_mundo(o2, 8.0)
    MEDIDAS["terreno"] = {"triangulos": len(faces), "rocha": int(ingreme.sum()), "area_ha": round(CONTORNO.area / 1e4, 1)}
    # base de maqueta de museu (saia e fundo), com o mar à cota do nível
    bxy = densificar(CONT_XY, 1.0, True)
    bz = np.maximum(amostra(Z, bxy[:, 0], bxy[:, 1]), NIVEL) - 0.05
    fundo = -16.0
    kk = len(bxy)
    anel = np.column_stack([bxy, bz])
    verts = anel.tolist() + [[p[0], p[1], fundo] for p in anel]
    fs = [[i, (i + 1) % kk, kk + (i + 1) % kk, kk + i] for i in range(kk)]
    fs.append(list(range(2 * kk - 1, kk - 1, -1)))
    mb = bpy.data.meshes.new("base")
    mb.from_pydata(verts, [], fs)
    mb.update()
    mb.materials.append(MATS["base"])
    obb = bpy.data.objects.new("Base", mb)
    bpy.context.scene.collection.objects.link(obb)
    obb.parent = GRUPOS["Terreno"]


ZTER = []


def cota_t(x, y):
    """Terreno da maqueta (o MDT, com o fundo do mar e do fosso já rebaixados)."""
    return amostra(ZTER[0] if ZTER else MDT, x, y)


# ================================================================ cena

bpy.ops.wm.read_factory_settings(use_empty=True)
RAIZ_OB = bpy.data.objects.new("Fortaleza_Peniche", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = LAT0, LON0
RAIZ_OB["centro_m"] = [float(CENTRO[0]), float(CENTRO[1])]
RAIZ_OB["cota_zero"] = ZREF
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
for g in ["Terreno", "Muralhas", "Guaritas", "Portas", "Edificios", "Fortim", "Ponte", "Vila", "Vegetacao", "Rochedos"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

# Pedra fotografada (kit.fotos, fotos CC0 da Poly Haven; créditos em medidas.json "texturas_foto"), com o tom
# das fotografias da fortaleza: alvenaria de calcário cinzento-amarelado (4 m por imagem) e cantaria clara
# (3 m).
TEX_ALV, ALT_ALV = textura_foto("old_stone_wall", n=1024, metros=4.0, cor=(0.64, 0.61, 0.54), contraste=0.9, metros_foto=4.0,
                                saturacao=0.6, liquen=0.05, liquen_cor=(0.80, 0.74, 0.45), semente=51)
material("alvenaria", (1, 1, 1), 0.93, imagem("alvenaria", TEX_ALV), normal=imagem("alvenaria_normal", normais(ALT_ALV, 2.5)))
TEX_CANT, ALT_CANT = textura_foto("stone_brick_wall_001", n=1024, metros=3.0, cor=(0.74, 0.71, 0.64), contraste=0.85,
                                  metros_foto=3.0, semente=52)
material("cantaria", (1, 1, 1), 0.88, imagem("cantaria", TEX_CANT), normal=imagem("cantaria_normal", normais(ALT_CANT, 3.0)), forca=0.7)
# cantaria clara do frontispício da porta da ponte (#d9d3c4 na fotografia; revisão do perito, 13 nov. 2026)
TEX_CANT_C, _ = textura_foto("stone_brick_wall_001", n=1024, metros=3.0, cor=(0.85, 0.83, 0.77), contraste=0.6,
                             metros_foto=3.0, semente=53)
material("cantaria_clara", (1, 1, 1), 0.85, imagem("cantaria_clara", TEX_CANT_C), normal=imagem("cantaria_normal_c", normais(ALT_CANT, 3.0)), forca=0.5)
material("madeira_escura", (0.23, 0.165, 0.125), 0.75)  # #3a2a20
material("cantaria_lisa", (0.76, 0.73, 0.66), 0.88)
material("sombra", (0.07, 0.065, 0.06), 1.0)
material("cal", (1, 1, 1), 0.85, imagem("cal", textura_reboco((0.95, 0.94, 0.91), semente=11, gasto=0.08)))
material("ocre", (1, 1, 1), 0.85, imagem("ocre", textura_reboco((0.91, 0.70, 0.28), semente=21, gasto=0.15)))
material("ocre_gasto", (1, 1, 1), 0.88, imagem("ocre_gasto", textura_reboco((0.89, 0.68, 0.27), semente=31, gasto=0.6)))
material("terraco", (1, 1, 1), 0.9, imagem("terraco", textura_lajes() * np.array([0.92, 0.92, 0.93])))
material("chao", (1, 1, 1), 0.95)
_yy, _xx = np.mgrid[0:256, 0:256] / 256
_f = fbm(_xx * 8, _yy * 8, 1.0, 77)
material("relva", (1, 1, 1), 0.95, imagem("relva", np.stack([0.24 + 0.06 * _f, 0.38 + 0.08 * _f, 0.15 + 0.03 * _f], -1)))
# Arribas: calcário fotografado (cliff_side, Poly Haven CC0), em blocos partidos por camadas, recolorido para o
# cinzento das rochas de Peniche nas fotografias (12 nov. 2026: a antiga textura de estratos desenhada à mão
# fazia um ondulado amarelo que não existe). 8 m por imagem, como a UV das arribas.
TEX_R, ALT_R = textura_foto("cliff_side", n=1024, metros=8.0, cor=(0.56, 0.55, 0.52), contraste=1.15, metros_foto=4.0,
                            saturacao=0.25, variacao=0.12, liquen=0.04, liquen_cor=(0.70, 0.62, 0.45), semente=61)
material("rocha", (1, 1, 1), 0.93, imagem("rocha", TEX_R), normal=imagem("rocha_normal", normais(ALT_R, 3.0)), forca=1.0)
TEX_TELHA, ALT_TELHA = textura_foto("roof_09", n=512, metros=2.0, cor=(0.50, 0.50, 0.50), metros_foto=2.0, saturacao=0.3, variacao=0.06)
material("telha", (1, 1, 1), 0.8, imagem("telha", np.clip(TEX_TELHA * np.array([1.44, 0.84, 0.56]), 0, 1)),
         normal=imagem("telha_normal", normais(ALT_TELHA, 2.0)), forca=0.6)
material("rosa", (0.90, 0.80, 0.72), 0.85)
material("lousa", (0.42, 0.42, 0.43), 0.8)
material("madeira_verde", (0.16, 0.33, 0.26), 0.7)
material("ferro", (0.14, 0.15, 0.16), 0.5)
material("bronze", (0.24, 0.30, 0.25), 0.5)
material("vidro", (0.18, 0.22, 0.28), 0.25)
# casas da vila: um só cinzento claro, paredes e telhados, para o destaque ficar na fortaleza
material("cinza_viz", (0.74, 0.74, 0.73), 0.9)
material("base", (0.33, 0.26, 0.20), 0.95)
material("agua_tunel", (0.05, 0.16, 0.17), 0.15)  # água à sombra, dentro do túnel
material("terreno", (1, 1, 1), 0.95)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)
material("pedra_vc", (1, 1, 1), 0.92, None, vcor=True)
material("mar", (1, 1, 1), 0.12, None, vcor=True, normal=imagem("ondas_normal", normais(textura_ondas(), 1.2)), forca=0.35)
img_chao = imagem("chao_orto", np.zeros((4, 4, 3)))  # substituída depois do terreno

CAL, CAN, TEL, SOM, MAD, FER, BRO, VID, LOU, LIS, OCR, OGA, CHAO_P = range(13)
MAT_P = ["cal", "cantaria", "telha", "sombra", "madeira_verde", "ferro", "bronze", "vidro", "lousa", "cantaria_lisa",
         "ocre", "ocre_gasto", "chao"]
MEDIDAS = {}

print("plataformas…")
m = Malha()
MUROS = PLAT["muros"]
for p in PLAT["plataformas"]:
    plataforma(p, m)
PLAT_UNIAO = shapely.union_all(PLAT_SHP + [shapely.Point(*FORTIM["c"]).buffer(FORTIM["r"] + 0.5)]) if PLAT_SHP else None
_dentro_plat = shapely.contains_xy(PLAT_UNIAO, XG, YG)
print("parapeitos e escarpas…")
mq = Malha()
n_merl, cands = parapeitos(mq)
print("terreno…")
construir_terreno()
ob = m.objeto("Muralhas", "muralhas", MAT_PLAT, GRUPOS["Muralhas"], uv=None, recalc=False)
uv_misto(ob, 4.0, planta=(CHAO_,))

print("guaritas…")
mq.objeto("Parapeitos", "muralhas", ["alvenaria", "ocre_gasto", "cantaria", "ocre"], GRUPOS["Muralhas"], uv=3.0)
GUAR = escolher_guaritas(cands)
mg = Malha()
for p_, bis, z_, _ in GUAR:
    guarita(mg, p_ + bis * 0.35, bis, z_ + 0.9, z_ + 1.1)
MEDIDAS["muralhas"] = {"plataformas": len(PLAT_SHP), "merloes": n_merl, "guaritas": len(GUAR) + 1}

print("edifícios…")
me_ = Malha()
for poly, zt, g, am in EDIFICIOS:
    edificio(me_, poly, zt, am)
# a torre branca no topo do pavilhão a noroeste (pico do MDS a 36,0 m)
torre_branca(me_, np.array([-9.5, 60.0]), 27.9, 36.0)
cruzeiro(me_, np.array([31.0, 15.5]))
me_.objeto("Edificios_prisao", "prisao", MAT_P, GRUPOS["Edificios"], uv=2.5)

print("fortim…")
mf = Malha()
fortim(mf)
obf = mf.objeto("Fortim_Redondo", "fortim", MAT_P, GRUPOS["Fortim"], uv=2.5)
uv_misto(obf, 2.5, planta=(CHAO_P,))

print("ponte e porta…")
mp = Malha()
L_ponte = ponte_arcos(mp)
mp.objeto("Ponte", "ponte", MAT_P, GRUPOS["Ponte"], uv=2.5)
mpo = Malha()
F0_porta, ztop_porta = porta_ponte(mpo, mg)
mpo.objeto("Porta_da_ponte", "porta", ["cantaria_clara", "sombra", "madeira_escura", "ferro", "vidro"], GRUPOS["Portas"], uv=2.5)
mt_ = Malha()
tunel_agua(mt_)
mt_.objeto("Tunel_da_agua", "tunel", ["cantaria", "alvenaria", "agua_tunel"], GRUPOS["Portas"], uv=2.5, recalc=False)
mg.objeto("Guaritas", "guaritas", ["cantaria_lisa", "cantaria", "sombra"], GRUPOS["Guaritas"], uv=2.0)

print("mar…")
construir_mar()
img_t = imagem("chao", textura_terreno()[::-1])
for nm_ in ("terreno", "chao"):
    mt = MATS[nm_]
    ti = mt.node_tree.nodes.new("ShaderNodeTexImage")
    ti.image = img_t
    ti.extension = "EXTEND"
    mt.node_tree.links.new(ti.outputs["Color"], mt.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
# o terrapleno do fortim usa a mesma textura do chão (material "chao")

print("relvados…")
mr_ = Malha()
relvados(mr_)
obr = mr_.objeto("Relvados", "vegetacao", ["cal", "relva"], GRUPOS["Vegetacao"], uv=4.0)

print("rochedos…")
MV, MF, MC = [], [], []
costa = AGUA & ~ndi.binary_erosion(AGUA)
cr, cc_ = np.nonzero(costa)
sel = rng.choice(len(cr), size=min(len(cr), 900), replace=False)
n_roc = 0
for i in sel:
    x, y = LXS[cc_[i]], LYS[cr[i]]
    if not CONTORNO.contains(shapely.Point(x, y).buffer(3)) or (PLAT_UNIAO is not None and PLAT_UNIAO.distance(shapely.Point(x, y)) < 2.0):
        continue
    if any(np.hypot(x - b[0], y - b[1]) < 7.0 for b, *_ in TUNEL["bocas"]):
        continue  # as bocas do túnel ficam desimpedidas
    if abs(x - 85) < 14 and abs(y - 58) < 18:
        r_ = rng.uniform(0.5, 1.3)  # no fosso da ponte, pedras miúdas
    else:
        r_ = rng.uniform(0.7, 2.4)
    rochedo(x + rng.normal(0, 1.0), y + rng.normal(0, 1.0), NIVEL - 0.1 * r_, r_, rng.uniform(0.45, 0.8), int(rng.integers(1e6)))
    n_roc += 1
# penedos soltos no baixio (até 9 m da costa), como na fotografia aérea
ar, ac = np.nonzero(AGUA & (D_AGUA > 2.5) & (D_AGUA < 9))
for i in rng.choice(len(ar), size=min(len(ar), 120), replace=False):
    x, y = LXS[ac[i]], LYS[ar[i]]
    if not CONTORNO.contains(shapely.Point(x, y).buffer(3)) or any(np.hypot(x - b[0], y - b[1]) < 9.0 for b, *_ in TUNEL["bocas"]):
        continue
    r_ = rng.uniform(0.6, 2.0)
    rochedo(x, y, NIVEL - 0.35 * r_, r_, rng.uniform(0.4, 0.7), int(rng.integers(1e6)))
    n_roc += 1
me = bpy.data.meshes.new("rochedos")
me.from_pydata(MV, [], MF)
me.update()
cl = np.array(MC) ** 2.2
ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
ca.data.foreach_set("color", np.hstack([cl, np.ones((len(cl), 1))]).astype(np.float32).ravel())
me.color_attributes.active_color = ca
me.materials.append(MATS["pedra_vc"])
ob = bpy.data.objects.new("Rochedos", me)
bpy.context.scene.collection.objects.link(ob)
ob["elemento"] = "mar"
ob.parent = GRUPOS["Rochedos"]
MEDIDAS["rochedos"] = n_roc

print("cidade…")
EVITAR = []
m = Malha()
pegadas = []
FORT_SHP = shapely.Polygon([(-182, -52), (-182, 96), (-100, 102), (-45, 128), (40, 128), (112, 122), (162, 66),
                            (162, -22), (124, -22), (60, -12), (0, -98), (-60, -62), (-150, -52)])
for f in F["edificios"]:
    if f["geometria"]["type"] != "Polygon":
        continue
    poly = geom(f)
    if area(poly) < 0:
        poly = poly[::-1]
    g_ = shapely.Polygon(poly)
    if not CONTORNO.contains(g_.buffer(-0.2)) or FORT_SHP.intersects(g_) or (PLAT_UNIAO is not None and PLAT_UNIAO.intersects(g_)):
        continue
    h = f.get("altura") or ((f.get("pisos") or 2) * 3.0 + 0.6)
    for pp in partir(poly):
        casa(m, pp, h)
    pegadas.append(poly)
n_osm = len(pegadas)
for c_ in json.loads((AQUI / "dados" / "casas.json").read_text()):
    poly = np.array(c_["planta"], float)
    if area(poly) < 0:
        poly = poly[::-1]
    g_ = shapely.Polygon(poly)
    if not CONTORNO.contains(g_) or FORT_SHP.intersects(g_):
        continue
    for pp in partir(poly):
        casa(m, pp, 6.0)
    pegadas.append(poly)
m.objeto("Casario", "vila", ["cinza_viz"] * 5, GRUPOS["Vila"], uv=3.0)  # as 5 cores antigas, todas a cinzento
MEDIDAS["vila"] = {"edificios": len(pegadas), "do_osm": n_osm}

print("vegetação…")
MV, MF, MC = [], [], []
pg_shp = shapely.union_all([shapely.Polygon(p).buffer(1.0) for p in pegadas])
n_arv = n_palm = 0
PALMEIRAS = [(36.5, 8.0), (35.5, 13.5), (-33.5, 51.5), (28.0, 65.0)]
for x, y, topo, _, r in LID["arvores"]:
    x, y = float(x), float(y)
    if not CONTORNO.contains(shapely.Point(x, y).buffer(2.0)) or pg_shp.contains(shapely.Point(x, y)):
        continue
    z = float(cota_t(x, y))
    h = float(topo) - ZREF - z
    if h < 2.5:
        continue
    if any(math.hypot(x - a, y - b) < 1.0 for a, b in PALMEIRAS):
        palmeira(x, y, z, max(h - 1.2, 3.5))
        n_palm += 1
        continue
    r = float(np.clip(r, 1.0, 5.5))
    alt = float(np.clip(h * 0.62, 1.6, 2.2 * r))
    plantar(x, y, z, r, alt, "folhosa", tronco=max(h - alt, 0.8))
    n_arv += 1
me = bpy.data.meshes.new("vegetacao")
me.from_pydata(MV, [], MF)
me.update()
me.shade_smooth()
cl = np.array(MC) ** 2.2
ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
ca.data.foreach_set("color", np.hstack([cl, np.ones((len(cl), 1))]).astype(np.float32).ravel())
me.color_attributes.active_color = ca
me.materials.append(MATS["arvore"])
ob = bpy.data.objects.new("Vegetacao", me)
bpy.context.scene.collection.objects.link(ob)
ob["elemento"] = "vegetacao"
ob.parent = GRUPOS["Vegetacao"]
MEDIDAS["vegetacao"] = {"arvores": n_arv, "palmeiras": n_palm}


def _arred(v):
    if isinstance(v, (float, np.floating)):
        return round(float(v), 1)
    if isinstance(v, (list, tuple)):
        return [_arred(x) for x in v]
    if isinstance(v, dict):
        return {k: _arred(x) for k, x in v.items()}
    if isinstance(v, np.integer):
        return int(v)
    return v


def ancora(x, y, z):
    """Ponto de uma placa no visor (x este, y cota, z sul)."""
    return [round(float(x), 1), round(float(z), 1), round(-float(y), 1)]


C_ = FORTIM["c"]
MEDIDAS["ancoras"] = {
    "fortim": ancora(C_[0], FORTIM["corda"] - 1.7, 27.5),
    "ponte": ancora(*((PONTE["a"] + PONTE["b"]) / 2), PONTE["deck"] + 6),
    "porta": ancora(*(F0_porta + unit(PONTE["b"] - PONTE["a"]) * 1.5), ztop_porta + 6),
    "prisao": ancora(5.0, 45.0, 34.0),
    "torre": ancora(-9.5, 60.0, 40.0),
    "muralhas": ancora(-95.0, 22.0, 30.0),
    "guaritas": ancora(*(GUAR[0][0] if GUAR else (0, 0)), (GUAR[0][2] if GUAR else 15) + 7),
    "mar": ancora(40.0, -70.0, 6.0),
    "tunel": ancora(82.5, 45.0, TUNEL["fecho"] + 5.0),
    "vila": ancora(60.0, 125.0, 24.0),
}
MEDIDAS["texturas_foto"] = fontes_fotos()  # créditos das fotos de pedra e telha
(AQUI / "dados" / "medidas.json").write_text(json.dumps(_arred(MEDIDAS), ensure_ascii=False, indent=1))
print(json.dumps(_arred(MEDIDAS), ensure_ascii=False))


# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
(AQUI / "saida").mkdir(exist_ok=True)
glb = AQUI / "saida" / "peniche-bruto.glb"
props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
opcoes = dict(
    filepath=str(glb), export_format="GLB", export_extras=True, export_yup=True, export_apply=True,
    export_image_format="JPEG", export_jpeg_quality=82,
)
if "export_vertex_color" in props:
    opcoes["export_vertex_color"] = "ACTIVE"
if "export_active_vertex_color_when_no_material" in props:
    opcoes["export_active_vertex_color_when_no_material"] = True
if "--sem-glb" not in sys.argv:
    bpy.ops.export_scene.gltf(**{k: v for k, v in opcoes.items() if k in props})
    print("GLB:", glb, round(glb.stat().st_size / 1e6, 2), "MB")
    import subprocess  # noqa: E402
    final = SAIDA / "peniche.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "peniche.blend"))

# ---------------------------------------------------------------- capa e vistas de verificação

if "--vistas" in sys.argv or "--capa" in sys.argv:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--amostras=")), 24))
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "AgX"
    w = bpy.data.worlds.new("mundo")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.78, 0.82, 0.88, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7
    sol = bpy.data.lights.new("sol", "SUN")
    sol.energy = 3.6
    sol.angle = math.radians(3)
    so = bpy.data.objects.new("sol", sol)
    sc.collection.objects.link(so)
    from mathutils import Vector
    d = Vector((-math.sin(math.radians(45)), math.cos(math.radians(45)), 0)) * math.cos(math.radians(50))
    d.z = math.sin(math.radians(50))
    so.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    cam = bpy.data.cameras.new("cam")
    co = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    pasta = AQUI / "vistas"
    pasta.mkdir(exist_ok=True)
    SO = next((a_.split("=")[1].split(",") for a_ in sys.argv if a_.startswith("--so=")), None)
    SUF = next((a_.split("=")[1] for a_ in sys.argv if a_.startswith("--sufixo=")), "")

    def vista(nome, azimute, elev, dist, lente=50, orto=None, res=(1200, 800), destino=None, alvo_xy=(0, 0), alvo_z=None):
        if SO and nome not in SO:
            return
        a, e = math.radians(azimute), math.radians(elev)
        al = Vector((alvo_xy[0], alvo_xy[1], float(cota_t(*alvo_xy)) + 2 if alvo_z is None else alvo_z))
        co.location = al + Vector((math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))) * dist
        co.rotation_euler = (al - co.location).to_track_quat("-Z", "Y").to_euler()
        if orto:
            cam.type = "ORTHO"
            cam.ortho_scale = orto
        else:
            cam.type = "PERSP"
            cam.lens = lente
        cam.clip_end = 6000
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.render.filepath = str(destino or (pasta / f"{nome}{SUF}.jpg"))
        bpy.ops.render.render(write_still=True)
        print("vista", nome)

    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 84
    # capa como a 1.ª fotografia aérea (de sudoeste, sobre o mar)
    vista("capa", 205, 30, 430, lente=50, res=(1280, 800), destino=SAIDA / "capa.jpg", alvo_xy=(5, 5), alvo_z=12)
    if "--capa" not in sys.argv:
        vista("topo", 180, 89.9, 800, orto=420, res=(1200, 900))
        vista("aereo_ne", 40, 28, 230, lente=40, alvo_xy=(15, 15), alvo_z=14)       # como a fotografia da ponte vista de cima
        vista("fortim", 2, 10, 42, lente=35, alvo_xy=(-11, -58), alvo_z=18)         # como a fotografia da sineira
        vista("fortim_mar", 200, 18, 75, lente=35, alvo_xy=(-11, -63), alvo_z=12)
        vista("ponte", 118, 9, 26, lente=30, alvo_xy=(65, 62), alvo_z=7)           # do fosso, como a fotografia grande
        vista("fosso", 100, 16, 30, lente=30, alvo_xy=(68, 62), alvo_z=5)  # de dentro do fosso, com a água
        vista("porta", 11, 8, 10, lente=28, alvo_xy=(61.3, 49.4), alvo_z=14.5)  # da ponte, como "Entrada no Forte de Peniche 02"
        vista("costa", 70, 22, 75, lente=35, alvo_xy=(5, -25), alvo_z=14)           # como a fotografia dos merlões amarelos
        vista("prisao", 150, 32, 130, lente=40, alvo_xy=(0, 20), alvo_z=18)
        vista("leste", 95, 30, 170, lente=40, alvo_xy=(105, 20), alvo_z=8)
        vista("oeste", 250, 30, 230, lente=40, alvo_xy=(-80, 20), alvo_z=12)
