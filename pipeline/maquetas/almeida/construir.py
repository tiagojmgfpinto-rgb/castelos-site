"""Constrói a maqueta 3D da Praça-forte de Almeida (estado atual) em Blender e exporta-a para
public/maquetas/almeida/almeida.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow
    python pipeline/maquetas/almeida/construir.py            # GLB
    python pipeline/maquetas/almeida/construir.py --vistas   # GLB + capa + vistas de verificação

Dados: dados/fontes.json (recolher.py: OSM via Overture), dados/lidar.npz e dados/orto.jpg (lidar.py:
MDT, MDS, classes e cor da nuvem de pontos do LiDAR 2024 da DGT), dados/escarpas.json (tracar.py: as
escarpas de alvenaria traçadas no MDT) e src/data/maquetas/almeida.json, com a leitura das fontes.

Uma praça abaluartada é sobretudo terra revestida de pedra. O relevo da maqueta é o próprio MDT de 50 cm
(terraplenos, parapeitos de terra, fossos, estrada coberta e esplanada), com a cor do chão tirada da
ortoimagem da nuvem de pontos; as escarpas de alvenaria são panos próprios, com talude, cordão, parapeito,
cunhais de cantaria e guaritas nos ângulos salientes dos baluartes, como nas fotografias. As portas, as
igrejas, a Torre do Relógio e o Quartel levam o detalhe das fotografias; o casario fica em volumes simples.
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
from kit.geometria import unit, area, densificar, dentro, simplificar, mediana_movel, angulos, partir, _retangulo, _ang, _arred  # noqa: E402
from kit.texturas import ruido, fbm, textura_grao, imagem  # noqa: E402
from kit.cena import arvore_modelo  # noqa: E402
from kit.arquitetura import Quadro as QuadroKit, _vao_desloc, altura_telhado, barra, cupula, faixa_inclinada, guardas_corredor, guarita, oitavado, porta_revelim, porta_tunel, telhado_4aguas, telhado_generico  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "almeida.json").read_text())
ESC = json.loads((AQUI / "dados" / "escarpas.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "almeida"

LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([0.0, 0.0])  # centro da grelha do LiDAR (igual ao lidar.py e ao tracar.py)
ZREF = 0.0  # as cotas da cena são as altitudes (m)
rng = np.random.default_rng(1641)


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
    """Alvenaria de granito das escarpas (4 m por repetição), como nas fotografias: pedras de tamanho
    desigual em fiadas pouco regulares, cinzento-acastanhadas, com manchas brancas de líquenes e
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
            base = np.array([(0.58, 0.56, 0.51), (0.64, 0.61, 0.55), (0.52, 0.50, 0.47), (0.68, 0.64, 0.56)][tom])
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
    tex = np.where(liq > 0.70, tex * 0.4 + np.array([0.82, 0.82, 0.78]) * 0.6, tex)  # líquenes brancos
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
            base = np.array([0.66, 0.64, 0.59]) * r.uniform(0.92, 1.06) + r.normal(0, 0.006, 3)
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


# ---------------------------------------------------------------- contorno da maqueta

def contorno_maqueta():
    """Limite da maqueta: a praça e as obras exteriores com a esplanada à volta (30 m para fora das escarpas
    e das linhas de muralha do OSM, alisado), dentro da área do LiDAR. Em Almeida o OSM não tem o polígono da
    praça: faz as vezes dele o anel da contraescarpa do fosso principal (retaining_wall)."""
    gs, praca = [], None
    for f in F["infraestrutura"]:
        if f["classe"] not in ("city_wall", "retaining_wall"):
            continue
        g = geom(f)
        if f["geometria"]["type"] == "Polygon":
            gs.append(shapely.Polygon(g))
        elif f["geometria"]["type"] == "LineString":
            gs.append(shapely.LineString(g))
            if f["classe"] == "retaining_wall" and len(g) > 50:
                praca = shapely.Polygon(g)
    for e in ESC:
        gs.append(shapely.LineString(e["linha"]))
    u = shapely.union_all([g.buffer(30) for g in gs if g.distance(praca) < 140] + [praca])
    u = u.buffer(18).buffer(-26).buffer(8)
    caixa = shapely.box(LXS[0] + 2, LYS[0] + 2, LXS[-1] - 2, LYS[-1] - 2)
    u = u.intersection(caixa)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    u = shapely.Polygon(u.exterior).simplify(0.8)
    # base redonda (pedido do Tiago, 10 nov.): o círculo que leva todas as obras exteriores, centrado na praça,
    # dentro da área do LiDAR
    c = praca.centroid
    r = max(c.distance(shapely.Point(q)) for q in u.exterior.coords) + 6
    r = min(r, c.x - LXS[0] - 3, LXS[-1] - c.x - 3, c.y - LYS[0] - 3, LYS[-1] - c.y - 3)
    print(f"  base redonda: centro ({c.x:.1f}, {c.y:.1f}), raio {r:.1f} m")
    return c.buffer(r, quad_segs=64), praca


CONTORNO, PRACA = contorno_maqueta()
CONT_XY = np.array(CONTORNO.exterior.coords)[:-1]
if area(CONT_XY) < 0:
    CONT_XY = CONT_XY[::-1]


# ---------------------------------------------------------------- escarpas (panos de alvenaria)

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


PARAPEITO = 1.1  # altura do parapeito de alvenaria acima do cordão (fotografias)
ESC_DADOS = []   # pontos densificados de todas as escarpas, para o terreno, as portas e o casario
GUARITAS = []


def _pontos_escarpas():
    """Todos os traçados densificados a 0,5 m, com o id da peça (para medir o espaço entre faces)."""
    P_, I_ = [], []
    for e in ESC:
        q = densificar(np.array(e["linha"], float), 0.5)
        P_.append(q)
        I_ += [e["id"]] * len(q)
    return cKDTree(np.vstack(P_)), np.array(I_)


ARV_ESC_TODAS = None


def espaco_livre(lin, eid, dmax=6.0):
    """Distância, para dentro de cada vértice, até ao traçado de outra escarpa (inf se não houver até 6 m):
    nos muros estreitos, como as calçadas das portas, as duas faces não se podem cruzar."""
    global ARV_ESC_TODAS
    if ARV_ESC_TODAS is None:
        ARV_ESC_TODAS = _pontos_escarpas()
    arv, ids = ARV_ESC_TODAS
    n = len(lin)
    dl = np.full(n, np.inf)
    for i in range(n):
        t1 = lin[min(i + 1, n - 1)] - lin[max(i - 1, 0)]
        t1 = t1 / max(np.linalg.norm(t1), 1e-6)
        nr = np.array([t1[1], -t1[0]])
        for d in np.arange(0.5, dmax + 0.01, 0.3):
            dd, k = arv.query(lin[i] - nr * d, k=4)
            if any(dist < 0.8 and ids[kk] != eid for dist, kk in zip(dd, k)):
                dl[i] = d
                break
    return dl


def degraus(lin, topo, pe, a, fim, salto=2.5, folga=0.8):
    """Onde o topo salta mais de 2,5 m entre dois vértices (um muro baixo que encosta a um pano alto,
    como um muro baixo encostado a uma frontaria), a mudança faz-se a 0,8 m do vértice alto, em vez de rampa."""
    L, T, P, A, Fm = [lin[0]], [topo[0]], [pe[0]], [a[0]], [fim[0]]
    for i in range(1, len(lin)):
        seg = float(np.linalg.norm(lin[i] - lin[i - 1]))
        if abs(topo[i] - topo[i - 1]) > salto and seg > 2 * folga:
            alto, baixo = (i - 1, i) if topo[i - 1] > topo[i] else (i, i - 1)
            q = lin[alto] + (lin[baixo] - lin[alto]) * folga / seg
            L.append(q)
            T.append(topo[baixo]); P.append(pe[baixo]); A.append(a[baixo]); Fm.append(fim[baixo])
            if alto == i:  # o ponto novo fica antes do vértice alto: corrige a ordem
                L[-1] = lin[i] + (lin[i - 1] - lin[i]) * folga / seg
        L.append(lin[i]); T.append(topo[i]); P.append(pe[i]); A.append(a[i]); Fm.append(fim[i])
    return np.array(L), np.array(T), np.array(P), np.array(A), np.array(Fm)


def _vao(lin, topo, pe, a, fim, zbot, d_, v, alvo, sole):
    """Abre um vão num pano, no cruzamento do traçado com o eixo da porta a cerca de `alvo` m da face."""
    n_ = unit(np.array(d_["normal"], float))
    t_ = np.array([-n_[1], n_[0]])
    p0 = np.array(d_["ponto"])
    r, omb = v["r"], v["omb"]
    s_ = (lin - p0) @ t_
    dd = (lin - p0) @ n_
    cruza = [i for i in range(len(lin) - 1)
             if (s_[i] + r + 0.5) * (s_[i + 1] - r - 0.5) < 0 or (s_[i] - r - 0.5) * (s_[i + 1] + r + 0.5) < 0
             if min(abs(dd[i] - alvo), abs(dd[i + 1] - alvo)) < 8 and abs(s_[i] - s_[i + 1]) > 1e-6]
    cruza = [i for i in cruza if abs(dd[i] + (dd[i + 1] - dd[i]) * (0 - s_[i]) / (s_[i + 1] - s_[i]) - alvo) < 4.0]
    if not cruza:
        return lin, topo, pe, a, fim, zbot
    i = cruza[0]
    # amostras novas: bordas exatas (com um par para a ombreira vertical) e de 0,1 em 0,1 m no vão
    ss = sorted(set(np.round(np.concatenate([np.arange(-r, r + 1e-6, 0.1), [-r - 0.02, r + 0.02, -r - 0.5, r + 0.5]]), 3)))
    fs = [(x - s_[i]) / (s_[i + 1] - s_[i]) for x in ss]
    fs = sorted(f for f in fs if 0 < f < 1)
    novos = [lin[i] + (lin[i + 1] - lin[i]) * f for f in fs]
    interp = lambda v_: [v_[i] + (v_[i + 1] - v_[i]) * f for f in fs]
    lin = np.vstack([lin[:i + 1], novos, lin[i + 1:]])
    topo = np.concatenate([topo[:i + 1], interp(topo), topo[i + 1:]])
    pe = np.concatenate([pe[:i + 1], interp(pe), pe[i + 1:]])
    a = np.concatenate([a[:i + 1], interp(a), a[i + 1:]])
    fim = np.concatenate([fim[:i + 1], interp(fim), fim[i + 1:]])
    zbot = np.concatenate([zbot[:i + 1], np.full(len(fs), -np.inf), zbot[i + 1:]])
    s_ = (lin - p0) @ t_
    dentro = (np.abs(s_) <= r + 1e-3) & (np.abs(np.arange(len(lin)) - (i + 1 + len(fs) / 2)) <= len(fs) / 2 + 1)
    zb = sole + omb + (0.0 if v.get("reto") else np.sqrt(np.clip(r * r - s_ ** 2, 0, None)))
    zbot = np.where(dentro, np.maximum(zbot, zb), zbot)
    # o pé à frente do vão é a soleira
    perto_ = np.abs(s_) <= r + 0.6
    pe = np.where(perto_ & (np.abs(np.arange(len(lin)) - (i + 1 + len(fs) / 2)) <= len(fs) / 2 + 1), sole, pe)
    return lin, topo, pe, a, fim, zbot


def vaos(lin, topo, pe, a, fim):
    """Vãos das portas abertos no próprio pano (sem o cortar): o traçado é densificado a 0,1 m na largura do
    vão e, aí, a face começa no intradorso do arco (zbot) em vez de no chão; o resto do pano (por cima do
    arco, o cordão e o parapeito) segue inteiro. Devolve os arrays densificados e zbot (-inf fora dos vãos)."""
    zbot = np.full(len(lin), -np.inf)
    for d_ in PORTAS.values():
        v = d_.get("vao")
        if not v or d_.get("sem_escarpa"):
            continue
        # a face de fora e, nas passagens com saída, a escarpa de dentro onde o túnel sai (Porta Nova)
        alvos = [(0.0, d_["soleira"] - ZREF)]
        if v.get("saida"):
            alvos.append((-v["tunel"] - 3.0, d_["soleira"] - ZREF + v.get("sobe", 0.0)))
        for alvo, sole in alvos:
            lin, topo, pe, a, fim, zbot = _vao(lin, topo, pe, a, fim, zbot, d_, v, alvo, sole)
    return lin, topo, pe, a, fim, zbot


def escarpa(e, m, mq):
    """Um pano de escarpa: face em talude até ao cordão, parapeito de alvenaria por cima e o maciço por
    trás (escondido pelo terrapleno). Os valores medidos (topo, pé, aresta) vêm do tracar.py."""
    lin, topo, pe, a, fim = degraus(np.array(e["linha"], float), np.array(e["topo"]) - ZREF, np.array(e["pe"]) - ZREF,
                                    np.clip(np.array(e["aresta"]), -2.0, 1.0), np.array(e["fim"]))
    lin, topo, pe, a, fim, zbot = vaos(lin, topo, pe, a, fim)
    h = np.maximum(topo - pe, 0.5)
    # talude: o MDT de 50 cm alarga a faixa da escarpa; fica entre 1/10 e 1/4 da altura
    bt = np.clip((fim - a) / h, 0.10, 0.25)
    hm = float(np.median(h))
    alto = hm > 5.0
    # Nas cortinas e baluartes altos o cordão fica a meia altura do pano, com o parapeito de alvenaria vertical por
    # cima (fotografias das portas de São Francisco e de Santo António); nos muros mais baixos, 1,1 m abaixo do topo.
    zc = topo - np.where(h > 7.0, np.clip(0.46 * h, PARAPEITO, 5.2), PARAPEITO if alto else 0.0)
    zf = np.maximum(pe - 1.2, zbot)  # nos vãos das portas a face começa no arco
    tem_cordao = hm > 3.4
    # muros estreitos: a face não passa do meio do muro e o maciço chega só ao meio (mais 0,3 m)
    dl = espaco_livre(lin, e["id"])
    estreito = np.isfinite(dl)
    a = np.where(estreito, np.maximum(a, -(dl / 2 - 0.6)), a)
    T = np.where(estreito, np.clip(a + dl / 2 + 0.3, 1.0, 3.0), 3.0)
    if tem_cordao:
        rc = 0.17
        perfil = [
            (a + bt * (zc - rc - zf), zf),
            (a, zc - rc),
            (a + rc * 0.7, zc - rc * 0.7), (a + rc, zc), (a + rc * 0.7, zc + rc * 0.7),  # cordão (meia-cana)
            (a, zc + rc),
        ]
        mats = [0, 1, 1, 1, 1]
        if alto:
            perfil += [(a - 0.1, zc + rc), (a - 0.1, topo - 0.12), (a - 0.02, topo - 0.12), (a - 0.02, topo), (a - 0.9, topo)]
            mats += [0, 0, 1, 1, 1]
        else:
            perfil += [(a - 0.9, topo)]
            mats += [1]
    else:
        perfil = [(a + bt * (topo - zf), zf), (a, topo), (a - 0.7, topo)]
        mats = [0, 1]
    # nos muros estreitos o maciço fica abaixo do chão da calçada, que é o terreno medido (lajeado contínuo)
    baixa = np.where(T < 2.9, 0.5, 0.15)
    perfil += [(a - 0.9, topo - baixa), (a - T, topo - baixa), (a - T, zf)]
    # nos muros estreitos (calçadas sobre o fosso) o alto é lajeado, não terrapleno com erva
    mats += [1, 1, 0] if float(np.median(T)) < 2.5 else [2, 2, 0]
    varrer(m, lin, perfil, mats)
    # dados densificados para o terreno: afastamento da aresta, talude, topo, pé (por 0,5 m)
    seg = np.linalg.norm(np.diff(lin, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1] + 1e-6, 0.4)
    xy = np.column_stack([np.interp(ss, s, lin[:, 0]), np.interp(ss, s, lin[:, 1])])
    k = np.clip(np.searchsorted(s, ss, side="right") - 1, 0, len(seg) - 1)
    t = (lin[k + 1] - lin[k]) / seg[k][:, None]
    nr = np.column_stack([t[:, 1], -t[:, 0]])
    ESC_DADOS.append(dict(xy=xy, nr=nr, a=np.interp(ss, s, a), bt=np.interp(ss, s, bt), topo=np.interp(ss, s, topo),
                          pe=np.interp(ss, s, pe), zc=np.interp(ss, s, zc), id=e["id"], alto=alto, h=np.interp(ss, s, h)))
    # cunhais de cantaria nos ângulos salientes, em silhares alternados nas duas faces (fotografias)
    ang = angulos(lin)
    for i in np.nonzero(ang > 20)[0]:
        t1, t2 = unit(lin[i] - lin[i - 1]), unit(lin[i + 1] - lin[i])
        n1, n2 = np.array([t1[1], -t1[0]]), np.array([t2[1], -t2[0]])
        bis = unit(n1 + n2)
        esc_ = 1.0 / max(0.5, float(bis @ n1))
        for j, z0 in enumerate(np.arange(zf[i] + 1.2, zc[i] - 0.45, 0.5)):
            off = a[i] + bt[i] * (zc[i] - z0 - 0.25) + 0.04
            c = lin[i] + bis * off * esc_
            for face_, (tt, nn) in enumerate(((t1, n1), (t2, n2))):
                comp = 0.95 if (j + face_) % 2 == 0 else 0.5
                cc = c + (-tt if face_ == 0 else tt) * comp / 2 + nn * 0.02
                mq.caixa(cc, tt, comp, 0.08, z0 + 0.02, z0 + 0.48, 0)
        # guaritas nos ângulos salientes dos baluartes e dos revelins (fotografias)
        if alto and ang[i] > 38 and seg[i - 1] > 6 and seg[i] > 6 and h[i] > 4.5 \
                and all(np.linalg.norm(g[0] - lin[i]) > 6 for g in GUARITAS):
            GUARITAS.append((lin[i] + bis * (a[i] + 0.35) * esc_, bis, float(zc[i]), float(topo[i])))
    return lin




# ---------------------------------------------------------------- terreno

def textura_terreno():
    """Cor do chão: a ortoimagem da nuvem de pontos (cor do ponto mais alto de cada célula de 50 cm),
    sem os telhados nem as copas, que se preenchem com a cor do chão à volta (convolução normalizada a
    várias escalas), um pouco suavizada para a escala de maqueta. Norte em cima, como o orto.jpg."""
    from PIL import Image
    img = np.asarray(Image.open(AQUI / "dados" / "orto.jpg"), float) / 255.0
    ky, kx = img.shape[:2]
    H = MDS - MDT
    tapa = ((CLS == 6) & (H > 1.5)) | (VEG & (H > 1.8)) | (CLS == 0)
    # nós da grelha (linha 0 = sul) -> células da imagem (linha 0 = norte)
    tc = (tapa[:-1, :-1] | tapa[1:, :-1] | tapa[:-1, 1:] | tapa[1:, 1:])[::-1]
    tc = ndi.binary_dilation(tc, iterations=1)
    # As sombras do voo (faixas escuras e azuladas ao pé das muralhas e das árvores) também se tapam: na
    # maqueta a luz é a do visor e as sombras carimbadas no chão saíam azuis (auditoria de novembro de 2026).
    lum = img.mean(-1)
    azul = img[..., 2] - (img[..., 0] + img[..., 1]) / 2
    sombra = ((lum < 0.72 * ndi.median_filter(lum, size=31)) & (lum < 0.42)) | ((azul > 0.0) & (lum < 0.40))
    sombra = ndi.binary_dilation(ndi.binary_opening(sombra, iterations=1), iterations=2)
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
    # relva um pouco mais verde e viva, como na maqueta de museu; o resto quase como está
    verde = np.clip((out[..., 1] - (out[..., 0] + out[..., 2]) / 2) * 8, 0, 1)[..., None]
    out = out * (1 - 0.25 * verde) + (out * np.array([0.92, 1.08, 0.86])) * 0.25 * verde
    cinza = out.mean(-1, keepdims=True)
    out = cinza + (out - cinza) * 1.3  # um pouco mais de cor: a ortoimagem do LiDAR é desmaiada
    return np.clip(out * 0.94, 0, 1)


def construir_terreno():
    nx, ny = len(LXS), len(LYS)
    X, Y = np.meshgrid(LXS, LYS)
    Z = MDT.copy()
    # Debaixo e à frente das escarpas o MDT desce em rampa (as faces são quase verticais e a grelha é de
    # 50 cm): baixa-se ao plano da face, para o pano ficar à vista e o terreno não o atravessar.
    pts = np.vstack([d["xy"] for d in ESC_DADOS])
    cat = lambda k: np.concatenate([np.broadcast_to(d[k], (len(d["xy"]),) + np.shape(d[k])[1:]) for d in ESC_DADOS])
    NR, A_, BT, TP, PE, ZC = cat("nr"), cat("a"), cat("bt"), cat("topo"), cat("pe"), cat("zc")
    arv = cKDTree(pts)
    sel = np.nonzero(arv.query(np.column_stack([X.ravel(), Y.ravel()]), distance_upper_bound=7.0)[0] < 7.0)[0]
    q = np.column_stack([X.ravel()[sel], Y.ravel()[sel]])
    _, k = arv.query(q)
    rel = q - pts[k]
    d = np.einsum("ij,ij->i", rel, NR[k])
    tan = np.abs(rel[:, 0] * -NR[k, 1] + rel[:, 1] * NR[k, 0])
    # Debaixo do maciço (3 m para dentro da aresta) e à frente da face o terreno desce ao pé: assim nenhum
    # triângulo do terreno vai do topo ao pé por fora da face. O maciço tapa o que fica por baixo dele.
    pe_face = A_[k] + BT[k] * (ZC[k] - PE[k])
    # à frente do pé deixa-se o chão subir devagar (40 %), para apagar o resto da face que o MDT espalha
    lim = np.where(d < A_[k] - 2.55, TP[k] - 0.2, PE[k] - 0.12 + 0.4 * np.maximum(d - pe_face - 0.45, 0))
    ok = (((tan < 0.5) & (d > A_[k] - 6.0)) | ((tan < 2.0) & (d > A_[k]))) & (d < pe_face + 3.0)
    # a calçada à frente das portas com corredor (em Almeida não há) fica à cota medida: não se rebaixa
    for d_ in PORTAS.values():
        if d_.get("corredor"):
            n_ = unit(np.array(d_["normal"], float))
            r_ = q - np.array(d_["ponto"])
            caixa_ = (np.abs(r_ @ np.array([-n_[1], n_[0]])) < 4.5) & (r_ @ n_ > 0.3) & (r_ @ n_ < 14.0)
            ok &= ~caixa_
    zr = Z.ravel()
    zz = zr[sel]
    zz[ok] = np.minimum(zz[ok], lim[ok])
    # calçadas das portas com corredor: lajeado plano à cota da soleira, entre os muros
    for d_ in PORTAS.values():
        if d_.get("corredor") and "soleira" in d_:
            n_ = unit(np.array(d_["normal"], float))
            r_ = q - np.array(d_["ponto"])
            caixa_ = (np.abs(r_ @ np.array([-n_[1], n_[0]])) < 4.5) & (r_ @ n_ > 0.0) & (r_ @ n_ < 14.0)
            sole_ = d_["soleira"] - ZREF
            plano = caixa_ & (np.abs(zz - sole_) < 1.6)
            zz[plano] = sole_ - 0.03
    zr[sel] = zz
    Z = zr.reshape(Z.shape)
    # Passagens das portas: dentro da faixa do túnel o terreno não pode ficar entre o chão e o alto da abóbada
    # (taparia o vão); onde o terrapleno passa por cima da abóbada fica como está.
    for poly, p0, n_, r_, sole_, sobe_, L_, omb_ in passagens():
        x0_, y0_ = poly.min(0) - 1
        x1_, y1_ = poly.max(0) + 1
        c0_, c1_ = np.searchsorted(LXS, [x0_, x1_])
        r0_, r1_ = np.searchsorted(LYS, [y0_, y1_])
        Xs, Ys = X[r0_:r1_, c0_:c1_], Y[r0_:r1_, c0_:c1_]
        dentro_p = dentro(Xs, Ys, poly)
        dd_ = -((Xs - p0[0]) * n_[0] + (Ys - p0[1]) * n_[1])
        chao_ = sole_ - ZREF + sobe_ * np.clip(dd_ / L_, 0, 1)
        reto_ = any(d_["vao"].get("reto") and np.allclose(d_["ponto"], p0) for d_ in PORTAS.values())
        vtop = chao_ + omb_ + (0.6 if reto_ else r_ + 0.6)
        Zs = Z[r0_:r1_, c0_:c1_]
        # no troço coberto pela abóbada (da frontaria até ao fim do túnel) o terreno não pode ficar dentro do
        # túnel (o teste de raios batia no talude a 3,5 m da porta): fica por cima do extradorso. Um campo de
        # alturas não faz túneis, por isso o salto do chão para o terrapleno tem de ficar na frontaria.
        Ff = face_porta(next(d_ for d_ in PORTAS.values() if np.allclose(d_["ponto"], p0)))
        ddF = -((Xs - Ff[0]) * n_[0] + (Ys - Ff[1]) * n_[1])
        coberto_ = dentro_p & (ddF > 0.35) & (dd_ < L_ - 0.5) & (Zs < vtop + 0.6)
        Zs[coberto_] = vtop[coberto_] + 0.6
        # e o degrau entre o chão e o terrapleno, logo atrás da frontaria, sai da malha (fica tapado por cima
        # pela escarpa e pela frontaria); senão tapava o vão
        t_ = np.array([n_[1], -n_[0]])
        w_ = r_ + 0.2
        BURACOS.append(np.array([Ff + t_ * w_ + n_ * 0.6, Ff - t_ * w_ + n_ * 0.6, Ff - t_ * w_ - n_ * 1.6, Ff + t_ * w_ - n_ * 1.6]))
        baixa_ = dentro_p & (Zs < vtop)
        Zs[baixa_] = np.minimum(Zs[baixa_], chao_[baixa_] - 0.05)
    # Casas da guarda: o MDT por baixo delas é interpolado (rampa do fundo do fosso até ao terrapleno) e
    # sairia pelas paredes; desce até ao pé do edifício (o mesmo zchao de casa_guarda)
    for pid, d_ in PORTAS.items():
        f = next((x for x in F["edificios"] if "casa" in d_ and x["id"].startswith(d_["casa"])), None)
        if f is None:
            continue
        pg = shapely.Polygon(geom(f)).buffer(0)
        zch = chao_casa(pg, d_)
        bb = np.array(pg.buffer(0.8).exterior.coords)
        x0_, y0_ = bb.min(0) - 1
        x1_, y1_ = bb.max(0) + 1
        c0_, c1_ = np.searchsorted(LXS, [x0_, x1_])
        r0_, r1_ = np.searchsorted(LYS, [y0_, y1_])
        Zs = Z[r0_:r1_, c0_:c1_]
        dn = dentro(X[r0_:r1_, c0_:c1_], Y[r0_:r1_, c0_:c1_], bb)
        Zs[dn] = np.minimum(Zs[dn], zch + 0.3)
        # à frente da fachada, até 5 m para cada lado: fundo do fosso (os remendos tapam o resto)
        fr, n_ = frente_casa(pg, d_)
        bb = np.array(pg.buffer(5.0).exterior.coords)
        x0_, y0_ = bb.min(0) - 1
        x1_, y1_ = bb.max(0) + 1
        c0_, c1_ = np.searchsorted(LXS, [x0_, x1_])
        r0_, r1_ = np.searchsorted(LYS, [y0_, y1_])
        Xs, Ys, Zs = X[r0_:r1_, c0_:c1_], Y[r0_:r1_, c0_:c1_], Z[r0_:r1_, c0_:c1_]
        dn = dentro(Xs, Ys, bb) & ((Xs - fr[0][0]) * n_[0] + (Ys - fr[0][1]) * n_[1] > -0.5)
        Zs[dn] = np.minimum(Zs[dn], zch + 0.3)
    ZTER[:] = [Z]

    dentro_ = shapely.contains_xy(CONTORNO, X + LPASSO / 2, Y + LPASSO / 2)[:-1, :-1]
    for b_ in BURACOS:
        dentro_ &= ~shapely.contains_xy(shapely.Polygon(b_), X[:-1, :-1] + LPASSO / 2, Y[:-1, :-1] + LPASSO / 2)
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
    print("  terreno bruto:", len(V), "vértices", len(Fq), "quadriláteros")
    # simplificação (colapso de arestas por erro quadrático): fica com c. 300 mil triângulos
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = min(1.0, 300000 / (2 * len(Fq)))
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear()
    ob.data = me2
    me2.materials.clear()
    me2.materials.append(MATS["terreno"])
    me2.shade_smooth()
    # UV em planta sobre a textura do chão (a ortoimagem cobre a grelha toda)
    me2.uv_layers.new(name="UVMap")
    co = np.empty(len(me2.vertices) * 3)
    me2.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    vi = np.empty(len(me2.loops), int)
    me2.loops.foreach_get("vertex_index", vi)
    u = (co[vi, 0] - LXS[0]) / (LXS[-1] - LXS[0])
    v = (co[vi, 1] - LYS[0]) / (LYS[-1] - LYS[0])
    me2.uv_layers.active.data.foreach_set("uv", np.stack([u, v], 1).ravel().astype(np.float32))
    MEDIDAS["terreno"] = {"vertices": int(len(me2.vertices)), "triangulos": int(len(me2.polygons)),
                          "area_ha": round(CONTORNO.area / 1e4, 1)}
    print("  terreno simplificado:", len(me2.polygons), "triângulos")

    # base de maqueta de museu (saia e fundo) pelo contorno
    bxy = densificar(CONT_XY, 1.0, True)
    x_, y_ = bxy[:, 0], bxy[:, 1]
    if 0.5 * float(np.sum(x_ * np.roll(y_, -1) - np.roll(x_, -1) * y_)) > 0:  # anel no sentido horário: saia para fora
        bxy = bxy[::-1].copy()
    bz = amostra(Z, bxy[:, 0], bxy[:, 1]) - 0.05
    fundo = float(bz.min()) - 10.0
    kk = len(bxy)
    anel = np.column_stack([bxy, bz])
    verts = anel.tolist() + [[p[0], p[1], fundo] for p in anel] * 2
    fs = [[i, (i + 1) % kk, kk + (i + 1) % kk, kk + i] for i in range(kk)]
    fs.append(list(range(2 * kk, 3 * kk)))  # fundo com vértices próprios (normais não se misturam com as da saia)
    mb = bpy.data.meshes.new("base")
    mb.from_pydata(verts, [], fs)
    mb.update()
    mb.materials.append(MATS["base"])
    obb = bpy.data.objects.new("Base", mb)
    bpy.context.scene.collection.objects.link(obb)
    obb.parent = GRUPOS["Terreno"]


ZTER = []
BURACOS = []  # quadrículas do terreno que saem (degrau atrás das frontarias das portas)


def cota_t(x, y):
    """Terreno da maqueta (o MDT já rebaixado à frente das escarpas)."""
    return amostra(ZTER[0], x, y)


# ---------------------------------------------------------------- casario (volumes simples)


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
    m.prisma(poly, z0, z1, 0, topo=False)
    s = simplificar(poly)
    if len(s) == 4 and abs(area(s)) > 0.85 * abs(area(poly)):
        telhado_4aguas(m, s, z1, mat=1)
    else:
        telhado_generico(m, poly, z1, mat=1)
    return z1, medido


# ---------------------------------------------------------------- vegetação


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


# ---------------------------------------------------------------- portas
# As portas de Almeida são duplas (Wikipédia PT, "Praça-forte de Almeida"): a de fora no revelim, a de dentro na
# cortina, com ponte de alvenaria sobre o fosso. Posições: edifícios "Porta de … (interna/externa)" do OSM, as
# pontes da classe 26 do LiDAR (pontes.py) e as faces das escarpas. As frontarias seguem as fotografias de
# referência (ver LEIAME): o vão é aberto no próprio pano (vaos(), sem o cortar) ou na fachada da casa da guarda,
# e o túnel atravessa o terrapleno ou a casa da guarda.


class Quadro(QuadroKit):
    """Referencial local de uma fachada (como o do kit, com talude): s ao longo dela (para a direita de quem olha
    de fora), z para cima e d para fora da parede, a partir de um ponto da face. Com bt > 0, d recua bt por metro
    acima de zb (faces em talude)."""

    def __init__(self, m, origem, normal):
        self.m = m
        self.o = np.asarray(origem, float)
        self.n = unit(np.asarray(normal, float))
        self.t = np.array([self.n[1], -self.n[0]])
        self.bt, self.zb = 0.0, 0.0

    def p(self, s, z, d):
        q = self.o + self.t * s + self.n * (d - self.bt * (z - self.zb))
        return (q[0], q[1], z)

    def bloco(self, s0, s1, z0, z1, d0, d1, mat):
        c = self.o + self.t * (s0 + s1) / 2 + self.n * (d0 + d1) / 2
        self.m.caixa(c, self.t, abs(s1 - s0), abs(d1 - d0), z0, z1, mat, fundo=True)

    def placa(self, sz, d0, d1, mat):
        sz = [tuple(q) for q in sz]
        if area(np.array(sz)) < 0:
            sz = sz[::-1]
        frente = [self.p(s, z, d1) for s, z in sz]
        tras = [self.p(s, z, d0) for s, z in sz]
        self.m.face(frente[::-1], mat)
        self.m.face(tras, mat)
        k = len(sz)
        for i in range(k):
            j = (i + 1) % k
            self.m.face([frente[i], frente[j], tras[j], tras[i]], mat)

    def arco_sz(self, s0, r0, r1, zn, d0, d1, mat, a0=0.0, a1=math.pi, k=12, aduelas=False):
        for i in range(k):
            t0 = a0 + (a1 - a0) * i / k + (0.012 if aduelas else 0)
            t1 = a0 + (a1 - a0) * (i + 1) / k - (0.012 if aduelas else 0)
            q = [(s0 + r0 * math.cos(t0), zn + r0 * math.sin(t0)), (s0 + r1 * math.cos(t0), zn + r1 * math.sin(t0)),
                 (s0 + r1 * math.cos(t1), zn + r1 * math.sin(t1)), (s0 + r0 * math.cos(t1), zn + r0 * math.sin(t1))]
            self.placa(q, d0, d1, mat)

    def esfera(self, s, z, d, r, mat):
        self.m.esfera(self.p(s, z, d), r, mat, k=10)

    def escudo(self, s0, z0, larg, alt, d0, d1, mat, coroa=True):
        """Escudo português (base redonda) com coroa real por cima, em relevo."""
        w = larg / 2
        pts = [(s0 - w, z0 + alt), (s0 - w, z0 + alt * 0.4)]
        pts += [(s0 - w * math.cos(math.pi / 2 * k / 6), z0 + alt * 0.4 - alt * 0.4 * math.sin(math.pi / 2 * k / 6)) for k in range(1, 7)]
        pts += [(s0 + w * math.sin(math.pi / 2 * k / 6), z0 + alt * 0.4 - alt * 0.4 * math.cos(math.pi / 2 * k / 6)) for k in range(1, 6)]
        pts += [(s0 + w, z0 + alt * 0.4), (s0 + w, z0 + alt)]
        self.placa(pts, d0, d1, mat)
        self.placa([(s0 - w * 0.55, z0 + alt * 0.25), (s0 + w * 0.55, z0 + alt * 0.25), (s0 + w * 0.55, z0 + alt * 0.85),
                    (s0 - w * 0.55, z0 + alt * 0.85)], d1, d1 + 0.05, mat)
        if coroa:
            zc = z0 + alt
            self.placa([(s0 - w * 1.05, zc), (s0 + w * 1.05, zc), (s0 + w * 1.15, zc + alt * 0.28), (s0 + w * 0.5, zc + alt * 0.2),
                        (s0, zc + alt * 0.36), (s0 - w * 0.5, zc + alt * 0.2), (s0 - w * 1.15, zc + alt * 0.28)], d0, d1, mat)
            self.esfera(s0, zc + alt * 0.45, (d0 + d1) / 2, alt * 0.09, mat)


C_, S_, MV_, ALV_, CAL_, REL_, VID_, FER_ = range(8)
MAT_PORTA = ["cantaria", "sombra", "madeira_verde", "alvenaria", "cal", "relva", "vidro", "ferro"]


def guarita_redonda(m, c, z0, frente, r=0.72, h=2.1, mat=C_, misula=True):
    """Guarita redonda de cantaria (fotografias das portas e de um ângulo dos baluartes): mísula em anéis, corpo
    cilíndrico com uma fresta para fora e a porta para dentro, cornija, cúpula em meia laranja e remate em bola."""
    c = np.asarray(c, float)
    u = unit(np.asarray(frente, float))
    w = np.array([-u[1], u[0]])
    if misula:
        for k_, (rr, a_, b_) in enumerate(((1.08, -0.3, 0.0), (0.92, -0.6, -0.3), (0.74, -0.9, -0.6), (0.5, -1.2, -0.9))):
            m.cilindro(c, r * rr * 0.92, z0 + a_, z0 + b_, mat, k=16, r1=r * rr)
        m.cilindro(c, 0.08, z0 - 1.45, z0 - 1.2, mat, k=8, r1=r * 0.46)
    m.cilindro(c, r + 0.1, z0, z0 + 0.22, mat, k=16)  # base moldurada
    m.cilindro(c, r, z0 + 0.22, z0 + h, mat, k=16)
    for d_, larg, a_, b_ in ((u, 0.11, z0 + 0.9, z0 + 1.6), (w, 0.11, z0 + 0.9, z0 + 1.6), (-w, 0.11, z0 + 0.9, z0 + 1.6),
                             (-u, 0.48, z0 + 0.22, z0 + 1.75)):
        m.caixa(c + d_ * (r + 0.004), np.array([-d_[1], d_[0]]), larg, 0.02, a_, b_, S_)
    m.cilindro(c, r + 0.14, z0 + h, z0 + h + 0.12, mat, k=16)  # cornija
    m.cilindro(c, r + 0.06, z0 + h + 0.12, z0 + h + 0.22, mat, k=16)
    zb = z0 + h + 0.22
    prev_r, prev_z = r + 0.02, zb
    for k_ in range(1, 6):
        fi = math.pi / 2 * k_ / 5
        rr, zz = (r + 0.02) * math.cos(fi), zb + (r * 0.95) * math.sin(fi)
        m.cilindro(c, prev_r, prev_z, zz, mat, k=16, r1=max(rr, 0.02), topo=(k_ == 5))
        prev_r, prev_z = rr, zz
    m.cilindro(c, 0.1, prev_z - 0.02, prev_z + 0.18, mat, k=8)
    m.esfera((c[0], c[1], prev_z + 0.36), 0.2, mat, k=10)
    return prev_z + 0.56


def tunel(Q, r, sole, omb, L, sobe=0.0, mat=ALV_):
    """Passagem de abóbada de volta perfeita (raio r, nascença a omb do chão) que entra L metros por trás da face;
    o chão sobe `sobe` metros até ao fim (a saída do lado de dentro). Paredes e abóbada de alvenaria e lajeado no
    chão; o fim, se for cego, fica escuro."""
    bt0, Q.bt = Q.bt, 0.0
    n = max(2, int(math.ceil(L / 2.0)))
    for i in range(n):
        d0, d1 = -L * i / n, -L * (i + 1) / n
        z0 = sole + sobe * i / n
        z1 = sole + sobe * (i + 1) / n
        zm = (z0 + z1) / 2
        for sg in (-1, 1):
            a_, b_ = sorted((sg * r, sg * (r + 0.5)))
            Q.bloco(a_, b_, zm - 0.4, zm + omb, d1, d0 + (0.02 if i == 0 else 0.0), mat)
        Q.arco_sz(0.0, r, r + 0.5, zm + omb, d1, d0 - (bt0 * (omb + r + 0.5) + 0.1 if i == 0 else 0.0), mat, k=12)
        Q.bloco(-r - 0.05, r + 0.05, zm - 0.35, zm + 0.01, d1, d0 + (0.3 if i == 0 else 0.0), C_)
    Q.bt = bt0


def arco_cantaria(Q, r, sole, omb, aduela=0.5, n_ad=13, d0=-0.02, d1=0.08, ombreira=(0.55, 0.38)):
    """Arco de volta perfeita de cantaria: aduelas e ombreiras de silhares alternados, salientes da face."""
    zn = sole + omb
    # anel contínuo por trás das aduelas: as juntas não deixam ver o escuro do vão (as "raias" de Melgaço)
    Q.arco_sz(0.0, r + 0.01, r + aduela - 0.02, zn, d0 - 0.04, d0 + 0.02, C_, k=n_ad * 2)
    for k_ in range(n_ad):
        t0 = math.pi * k_ / n_ad + 0.004
        t1 = math.pi * (k_ + 1) / n_ad - 0.004
        ra = r + aduela * (1.15 if k_ == n_ad // 2 else 1.0)
        Q.placa([(r * math.cos(t0), zn + r * math.sin(t0)), (ra * math.cos(t0), zn + ra * math.sin(t0)),
                 (ra * math.cos(t1), zn + ra * math.sin(t1)), (r * math.cos(t1), zn + r * math.sin(t1))], d0, d1, C_)
    for k_, zz in enumerate(np.arange(sole, zn - 0.12, 0.48)):
        e_ = ombreira[k_ % 2]
        for sg in (-1, 1):
            a_, b_ = sorted((sg * r, sg * (r + e_)))
            Q.placa([(a_, zz + 0.015), (b_, zz + 0.015), (b_, min(zz + 0.465, zn)), (a_, min(zz + 0.465, zn))], d0, d1, C_)


def pano_com_arco(Q, s0, s1, z0, z1, r, omb, d0, d1, mat=C_):
    """Placa (s, z) entre s0 e s1 e z0 e z1, com o recorte do arco do vão (centrado em s = 0)."""
    pts = [(s0, z0), (-r, z0), (-r, z0 + omb)]
    pts += [(-r * math.cos(math.pi * k_ / 14), z0 + omb + r * math.sin(math.pi * k_ / 14)) for k_ in range(1, 14)]
    pts += [(r, z0 + omb), (r, z0), (s1, z0), (s1, z1), (s0, z1)]
    # a placa tem uma reentrância: parte-se em duas metades simples
    esq = [(s0, z0), (-r, z0), (-r, z0 + omb)] + [(-r * math.cos(math.pi * k_ / 14), z0 + omb + r * math.sin(math.pi * k_ / 14)) for k_ in range(1, 8)] + [(0.0, z1), (s0, z1)]
    dir_ = [(0.0, z1)] + [(-r * math.cos(math.pi * k_ / 14), z0 + omb + r * math.sin(math.pi * k_ / 14)) for k_ in range(7, 14)] + [(r, z0 + omb), (r, z0), (s1, z0), (s1, z1)]
    Q.placa(esq, d0, d1, mat)
    Q.placa(dir_, d0, d1, mat)
    return pts


def pilastra(Q, s0, s1, z0, z1, d0, d1, capitel=True, almofadas=False):
    Q.bloco(s0 - 0.07, s1 + 0.07, z0 - 0.2, z0 + 0.45, d0, d1 + 0.08, C_)  # plinto
    if almofadas:
        for k_, zz in enumerate(np.arange(z0 + 0.45, z1 - 0.1, 0.5)):
            ss = 0.0 if k_ % 2 == 0 else 0.07
            Q.bloco(s0 + ss, s1 - ss, zz + 0.03, min(zz + 0.47, z1), d0, d1, C_)
        Q.bloco(s0 + 0.08, s1 - 0.08, z0 + 0.45, z1, d0, d1 - 0.06, C_)
    else:
        Q.bloco(s0, s1, z0 + 0.45, z1, d0, d1, C_)
    if capitel:
        Q.bloco(s0 - 0.06, s1 + 0.06, z1, z1 + 0.22, d0, d1 + 0.07, C_)


def porta_sao_francisco_dentro(Q, sole):
    """Portas de São Francisco (da Cruz), porta de dentro, na cortina (fotografia da N332): frontispício de
    cantaria com dois pares de pilastras toscanas, arco de volta perfeita com aduelas, as armas reais por cima do
    arco, entablamento e frontão curvo (segmento) a toda a largura; a guarita redonda ao meio, no alto. O cordão
    da muralha passa à altura da nascença do arco. Ponte de alvenaria à frente (classe 26 do LiDAR)."""
    v = PORTAS["sf_dentro"]["vao"]
    r, omb = v["r"], v["omb"]
    tunel(Q, r, sole, omb, v["tunel"], v.get("sobe", 0.0))
    topo = Q.topo
    H = topo - sole
    W = 3.9
    ze = sole + max(6.2, 0.68 * H)
    dt = -Q.bt * (ze + 2 - sole) - 0.6  # o fundo do frontispício entra no pano em talude
    Q.bt, bt0 = 0.0, Q.bt
    pano_com_arco(Q, -W + 0.9, W - 0.9, sole, ze, r, omb, dt, 0.22)
    arco_cantaria(Q, r, sole, omb, aduela=0.55, d0=0.2, d1=0.32)
    for sg in (-1, 1):  # dois pares de pilastras
        for a0, a1 in ((r + 0.75, r + 1.35), (W - 0.6, W)):
            s0, s1 = sorted((sg * a0, sg * a1))
            Q.bloco(s0, s1, sole, ze - 0.22, dt, 0.3, C_)
            Q.almofadas(s0, s1, sole + 0.6, ze - 0.5, 0.3, 0.46, C_, fiada=0.5)  # colunas de tambores almofadados (foto)
            Q.bloco(s0 - 0.06, s1 + 0.06, ze - 0.5, ze - 0.22, 0.3, 0.5, C_)  # capitel
            Q.bloco(s0 - 0.06, s1 + 0.06, sole, sole + 0.6, 0.3, 0.5, C_)  # plinto
        a_, b_ = sorted((sg * (r + 1.35), sg * (W - 0.6)))
        Q.bloco(a_, b_, sole, ze, dt, 0.26, C_)
    Q.escudo(0.0, sole + omb + r + 0.75, 1.0, 1.15, 0.22, 0.38, C_)
    Q.bloco(-W - 0.1, W + 0.1, ze, ze + 0.4, dt, 0.5, C_)   # arquitrave e friso
    Q.bloco(-W - 0.25, W + 0.25, ze + 0.4, ze + 0.62, dt, 0.68, C_)  # cornija
    # frontão curvo (segmento de arco) sobre a cornija, com o tímpano
    R = 5.6
    zcen = ze + 0.62 - math.sqrt(R * R - W * W)
    a0 = math.acos((W + 0.2) / R)
    Q.arco_sz(0.0, R - 0.35, R, zcen, dt, 0.6, C_, a0=a0, a1=math.pi - a0, k=12)
    tp = [((R - 0.35) * math.cos(t), zcen + (R - 0.35) * math.sin(t)) for t in np.linspace(a0, math.pi - a0, 13)]
    Q.placa(tp, dt, 0.3, C_)
    zt = zcen + R
    zg = guarita_redonda(Q.m, Q.p(0.0, 0.0, -0.6)[:2], zt - 0.15, -Q.n * -1, r=0.72, h=1.9, misula=False)
    for sg in (-1, 1):  # pináculos em bola nos extremos do frontão
        Q.bloco(sg * W - 0.25, sg * W + 0.25, ze + 0.62, ze + 1.2, -0.2, 0.4, C_)
        Q.esfera(sg * W, ze + 1.5, 0.1, 0.28, C_)
    Q.bt = bt0
    return zg - sole


def porta_sao_francisco_fora(Q, sole):
    """Portas de São Francisco, porta de fora, no revelim (fotografias da Estrada das Portas de São Francisco):
    frontaria de cantaria baixa com o arco ao meio e pilastras, entablamento, e por cima as armas reais coroadas
    entre dois pináculos em bola, com a guarita redonda ao meio; à frente, a calçada entre muretes de granito com
    dois obeliscos piramidais nos extremos."""
    v = PORTAS["sf_fora"]["vao"]
    r, omb = v["r"], v["omb"]
    tunel(Q, r, sole, omb, v["tunel"], v.get("sobe", 0.0))
    W = 2.9
    ze = sole + 4.9
    dt = -Q.bt * (ze + 3 - sole) - 0.6
    Q.bt, bt0 = 0.0, Q.bt
    pano_com_arco(Q, -W, W, sole, ze, r, omb, dt, 0.18)
    arco_cantaria(Q, r, sole, omb, aduela=0.5, d0=0.16, d1=0.28)
    for sg in (-1, 1):
        s0, s1 = sorted((sg * (r + 0.55), sg * (r + 1.15)))
        pilastra(Q, s0, s1, sole, ze - 0.22, dt, 0.36)
    Q.bloco(-W - 0.1, W + 0.1, ze, ze + 0.35, dt, 0.42, C_)
    Q.bloco(-W - 0.2, W + 0.2, ze + 0.35, ze + 0.55, dt, 0.55, C_)
    # ático com as armas reais e pináculos
    za = ze + 0.55
    Q.bloco(-1.3, 1.3, za, za + 1.9, dt, 0.25, C_)
    Q.escudo(0.0, za + 0.25, 1.0, 1.1, 0.25, 0.4, C_)
    for sg in (-1, 1):
        Q.bloco(sg * 2.2 - 0.3, sg * 2.2 + 0.3, za, za + 0.75, -0.2, 0.35, C_)
        Q.bloco(sg * 2.2 - 0.18, sg * 2.2 + 0.18, za + 0.75, za + 0.95, -0.08, 0.23, C_)
        Q.esfera(sg * 2.2, za + 1.28, 0.08, 0.33, C_)
    zg = guarita_redonda(Q.m, Q.p(0.0, 0.0, -0.75)[:2], za + 1.9, Q.n, r=0.68, h=1.8, misula=False)
    # calçada entre muretes, com obeliscos piramidais nos extremos de fora
    Lc = v.get("calcada", 0.0)
    if Lc > 0:
        for sg in (-1, 1):
            a_, b_ = sorted((sg * 2.6, sg * 3.15))
            Q.bloco(a_, b_, sole - 0.6, sole + 0.95, 0.4, Lc, C_)
            Q.bloco(a_ - 0.04, b_ + 0.04, sole + 0.95, sole + 1.08, 0.4, Lc, C_)
            sm = (a_ + b_) / 2
            Q.bloco(sm - 0.5, sm + 0.5, sole - 0.2, sole + 1.6, Lc - 1.0, Lc, C_)
            c_ = Q.p(sm, 0, Lc - 0.5)[:2]
            piramide(Q.m, c_, Q.t, 0.7, sole + 1.6, sole + 4.0, C_, lado1=0.12)
        Q.bloco(-2.6, 2.6, sole - 0.3, sole + 0.01, 0.3, Lc, C_)  # lajeado
    Q.bt = bt0
    return zg - sole


def porta_santo_antonio_dentro(Q, sole):
    """Portas de Santo António, porta de dentro, na cortina (fotografia de 2022): frontispício de cantaria com
    pilastras dóricas, arco de volta perfeita com aduelas, as armas reais por cima do arco, entablamento e frontão
    triangular interrompido ao meio pela guarita redonda; o cordão da muralha corta a frontaria à altura da
    nascença; à direita, um postigo gradeado. Ponte de alvenaria à frente (classe 26 do LiDAR)."""
    v = PORTAS["sa_dentro"]["vao"]
    r, omb = v["r"], v["omb"]
    tunel(Q, r, sole, omb, v["tunel"], v.get("sobe", 0.0))
    topo = Q.topo
    H = topo - sole
    W = 2.9
    ze = sole + max(6.0, 0.62 * H)
    dt = -Q.bt * (ze + 3 - sole) - 0.6
    Q.bt, bt0 = 0.0, Q.bt
    pano_com_arco(Q, -W + 0.75, W - 0.75, sole, ze, r, omb, dt, 0.2)
    arco_cantaria(Q, r, sole, omb, aduela=0.5, d0=0.18, d1=0.3)
    for sg in (-1, 1):
        s0, s1 = sorted((sg * (W - 0.75), sg * W))
        pilastra(Q, s0, s1, sole, ze - 0.22, dt, 0.4, almofadas=True)
    Q.escudo(0.0, sole + omb + r + 0.7, 0.95, 1.1, 0.2, 0.34, C_)
    Q.bloco(-W - 0.1, W + 0.1, ze, ze + 0.38, dt, 0.46, C_)
    Q.bloco(-W - 0.28, W + 0.28, ze + 0.38, ze + 0.6, dt, 0.66, C_)
    zf = ze + 0.6
    hf = 1.9
    for sg in (-1, 1):  # meios frontões triangulares, interrompidos ao meio
        Q.placa([(sg * (W + 0.28), zf), (sg * 0.95, zf), (sg * 0.95, zf + hf * (W + 0.28 - 0.95) / (W + 0.28))], dt, 0.5, C_)
        Q.placa([(sg * (W + 0.36), zf - 0.02), (sg * (W + 0.18), zf - 0.02), (sg * 0.95, zf + hf * (W + 0.28 - 0.95) / (W + 0.28)),
                 (sg * 0.95, zf + hf * (W + 0.28 - 0.95) / (W + 0.28) + 0.25)], 0.4, 0.7, C_)
    zg = guarita_redonda(Q.m, Q.p(0.0, 0.0, -0.3)[:2], zf + 0.1, Q.n, r=0.7, h=1.9, misula=False)
    # postigo gradeado à direita (fotografia)
    Q.bloco(W + 3.4, W + 4.0, sole + omb + 0.4, sole + omb + 1.1, 0.0, 0.06, S_)
    for k_ in range(4):
        ss = W + 3.45 + k_ * 0.16
        Q.bloco(ss, ss + 0.03, sole + omb + 0.4, sole + omb + 1.1, 0.06, 0.1, FER_)
    Q.bt = bt0
    return zg - sole


def porta_santo_antonio_fora(Q, sole):
    """Portas de Santo António, porta de fora, na casa da guarda do revelim (fotografia, só referência; ver
    LEIAME): frontaria lisa de cantaria com o arco ao meio, as armas reais por cima do arco, remate em empena de
    pedra e a guarita redonda ao meio, no alto. A casa da guarda é alta (MDS)."""
    v = PORTAS["sa_fora"]["vao"]
    r, omb = v["r"], v["omb"]
    tunel(Q, r, sole, omb, v["tunel"], v.get("sobe", 0.0))
    W = 3.2
    ze = sole + 5.4
    pano_com_arco(Q, -W, W, sole, ze, r, omb, -0.6, 0.2)
    arco_cantaria(Q, r, sole, omb, aduela=0.48, d0=0.18, d1=0.3)
    Q.escudo(0.0, sole + omb + r + 0.6, 0.9, 1.0, 0.2, 0.34, C_)
    Q.bloco(-W - 0.1, W + 0.1, ze, ze + 0.35, -0.6, 0.36, C_)
    Q.placa([(-W, ze + 0.35), (W, ze + 0.35), (0.0, ze + 1.7)], -0.6, 0.25, C_)  # empena de pedra
    zg = guarita_redonda(Q.m, Q.p(0.0, 0.0, -0.6)[:2], ze + 1.45, Q.n, r=0.66, h=1.8, misula=False)
    return zg - sole


def porta_nova(Q, sole):
    """Porta Nova (fotografia de 2023): passagem moderna de verga reta, aberta na cortina para a estrada, com
    ombreiras de silhares de granito, aro de aço corten e túnel forrado; sem frontispício."""
    v = PORTAS["porta_nova"]["vao"]
    r, omb = v["r"], v["omb"]
    bt0, Q.bt = Q.bt, 0.0
    L = v["tunel"]
    for sg in (-1, 1):
        a_, b_ = sorted((sg * r, sg * (r + 0.45)))
        Q.bloco(a_, b_, sole - 0.3, sole + omb, -L, 0.05, ALV_)
        a_, b_ = sorted((sg * r, sg * (r + 0.12)))
        Q.bloco(a_, b_, sole, sole + omb, -0.05, 0.12, FER_)  # aro de aço
    Q.bloco(-r - 0.45, r + 0.45, sole + omb, sole + omb + 0.5, -L, 0.05, ALV_)
    Q.bloco(-r - 0.12, r + 0.12, sole + omb - 0.12, sole + omb + 0.02, -0.05, 0.12, FER_)
    Q.bloco(-r - 0.05, r + 0.05, sole - 0.3, sole + 0.01, -L, 0.6, C_)
    Q.bt = bt0
    return omb


PORTAS = {
    # ponto na face e normal (para fora); vao: r = meia largura, omb = nascença do arco, tunel = comprimento da
    # passagem por trás da face, sobe = quanto sobe o chão até à saída. "casa": edifício do OSM da casa da guarda.
    "sa_dentro": dict(ponto=(-177.2, -44.0), normal=(-0.988, -0.154), estilo=porta_santo_antonio_dentro, elemento="portas_santo_antonio",
                      fixo=True, soleira=749.37, ponte="ponte_santo_antonio", largura=3.4,
                      vao=dict(r=1.65, omb=2.9, tunel=20.0, sobe=0.0)),
    "sa_fora": dict(ponto=(-326.6, -19.7), normal=(-0.6536, 0.7568), estilo=porta_santo_antonio_fora, elemento="portas_santo_antonio",
                    fixo=True, sem_escarpa=True, soleira=745.4, ponte="ponte_santo_antonio_revelim", casa="304b8aa6", chamine=(6.5, 5.0), cobertura=0.13, largura=3.2,
                    vao=dict(r=1.5, omb=2.6, tunel=22.0, sobe=0.4, saida=True)),
    "sf_dentro": dict(ponto=(123.8, -175.8), normal=(0.57, -0.8217), estilo=porta_sao_francisco_dentro, elemento="portas_sao_francisco",
                      fixo=True, sem_escarpa=True, soleira=742.29, ponte="ponte_sao_francisco", casa="b2fed706", cobertura=0.27, largura=3.4,
                      vao=dict(r=1.6, omb=2.8, tunel=25.0, sobe=0.9, saida=True)),
    "sf_fora": dict(ponto=(151.8, -263.7), normal=(-0.27, -0.963), estilo=porta_sao_francisco_fora, elemento="portas_sao_francisco",
                    fixo=True, soleira=741.4, casa="e754914d", largura=3.0,
                    vao=dict(r=1.45, omb=2.55, tunel=16.5, sobe=-0.1, saida=True, calcada=13.0)),
    # Porta Nova: a estrada (OSM "Porta Nova") passa por baixo do terrapleno, de 738,6 por fora a 740,2 por dentro (MDT)
    "porta_nova": dict(ponto=(169.4, 171.6), normal=(0.47, 0.883), estilo=porta_nova, elemento="porta_nova",
                       fixo=True, largura=4.6, soleira=738.6, vao=dict(r=2.3, omb=4.0, tunel=13.8, sobe=1.4, reto=True, saida=True)),
}
PONTOS_PORTA = {}
PROPRIOS_PORTAS = {d["casa"] for d in PORTAS.values() if "casa" in d} | {"28e37bfa"}  # 28e37bfa: terraço sobre o túnel de Santo António


def passagens():
    """Faixas das passagens (para o terreno e para afastar casas e árvores): retângulo da face até ao fim do túnel,
    com o chão em rampa. Devolve [(polígono, ponto, normal, r, sole, sobe, L, omb)]."""
    out = []
    for pid, d_ in PORTAS.items():
        v = d_["vao"]
        n_ = unit(np.array(d_["normal"], float))
        t_ = np.array([n_[1], -n_[0]])
        p0 = np.array(d_["ponto"], float)
        sole = d_.get("soleira", None)
        if sole is None:
            sole = float(cota(*(p0 + n_ * 4.0)))
            d_["soleira"] = sole
        w = v["r"] + 0.55
        L = v["tunel"]
        poly = np.array([p0 + t_ * w + n_ * 0.6, p0 - t_ * w + n_ * 0.6, p0 - t_ * w - n_ * L, p0 + t_ * w - n_ * L])
        out.append((poly, p0, n_, v["r"], sole, v.get("sobe", 0.0), L, v["omb"]))
    return out


def gravar_pontes(m):
    """Pontes de alvenaria sobre os fossos, medidas na classe 26 do LiDAR (pontes.py): tabuleiro lajeado, guardas
    de granito de c. 0,9 m (o topo das guardas também se mede) e arcos de volta perfeita sobre pegões até ao
    fundo do fosso. O número de arcos é uma estimativa (vãos de c. 6 a 7 m)."""
    PJ = json.loads((AQUI / "dados" / "pontes.json").read_text())
    for nome, p in PJ.items():
        c = np.array(p["centro"], float)
        ax = unit(np.array(p["eixo"], float))
        L, W, zt = p["comprimento"], max(p["largura"], 4.0), p["tabuleiro"] - ZREF
        Qp = Quadro(m, c - ax * L / 2, ax)  # d ao longo da ponte (0 a L), s na largura
        guardas = max(p["guardas"] - p["tabuleiro"], 0.85)
        Qp.bloco(-W / 2, W / 2, zt - 0.7, zt, 0.0, L, C_)
        Qp.bloco(-W / 2 + 0.5, W / 2 - 0.5, zt - 0.02, zt + 0.02, 0.3, L - 0.3, C_)
        for sg in (-1, 1):
            a_, b_ = sorted((sg * W / 2, sg * (W / 2 - 0.45)))
            Qp.bloco(a_, b_, zt, zt + guardas - 0.12, 0.0, L, C_)
            Qp.bloco(a_ - 0.04, b_ + 0.04, zt + guardas - 0.12, zt + guardas, 0.0, L, C_)
        fundo = [float(cota(*(c - ax * L / 2 + ax * d))) for d in np.linspace(1, L - 1, 12)]
        zfundo = float(np.percentile(fundo, 30)) - ZREF
        nv = max(1, int(round(L / 6.5)))
        vao_ = L / nv
        pg = min(1.4, vao_ * 0.22)  # pegões
        r_ = (vao_ - pg) / 2
        zn = min(zt - 0.7 - r_ - 0.25, zt - 1.2)
        r_ = min(r_, zt - 0.95 - zn)
        for i in range(nv + 1):
            d_ = i * vao_
            a_, b_ = max(0.0, d_ - pg / 2), min(L, d_ + pg / 2)
            if b_ - a_ > 0.2:
                Qp.bloco(-W / 2 + 0.05, W / 2 - 0.05, zfundo - 0.6, zn + 0.1, a_, b_, C_)
        for i in range(nv):
            dc = (i + 0.5) * vao_
            # tímpanos: pano cheio entre o tabuleiro e o arco, dos dois lados; o vão por baixo fica aberto
            for sg in (-1, 1):
                ss0, ss1 = sorted((sg * W / 2, sg * (W / 2 - 0.6)))
                pts = [(dc - vao_ / 2, zn), (dc - r_, zn)]
                pts += [(dc - r_ * math.cos(math.pi * k_ / 12), zn + r_ * math.sin(math.pi * k_ / 12)) for k_ in range(1, 12)]
                pts += [(dc + r_, zn), (dc + vao_ / 2, zn), (dc + vao_ / 2, zt - 0.7), (dc - vao_ / 2, zt - 0.7)]
                # placa no plano (d, z): usa-se um Quadro de lado
                Ql = Quadro(m, Qp.o + Qp.t * (ss0 + ss1) / 2, Qp.t)
                Ql.placa([(-dd, zz) for dd, zz in pts], -(ss1 - ss0) / 2, (ss1 - ss0) / 2, C_)
            # intradorso da abóbada
            for k_ in range(12):
                t0, t1 = math.pi * k_ / 12, math.pi * (k_ + 1) / 12
                q0 = (dc - r_ * math.cos(t0), zn + r_ * math.sin(t0))
                q1 = (dc - r_ * math.cos(t1), zn + r_ * math.sin(t1))
                pa = [Qp.p(s_, q0[1], q0[0]) for s_ in (-W / 2 + 0.6, W / 2 - 0.6)]
                pb = [Qp.p(s_, q1[1], q1[0]) for s_ in (-W / 2 + 0.6, W / 2 - 0.6)]
                m.face([pa[0], pa[1], pb[1], pb[0]], ALV_)
            # enchimento entre o extradorso e o tabuleiro, por cima do intradorso
            Qp.bloco(-W / 2 + 0.6, W / 2 - 0.6, zn + r_ + 0.05, zt - 0.7, dc - r_ * 0.7, dc + r_ * 0.7, ALV_)
        EVITAR.append(np.array([Qp.p(-W / 2 - 1, 0, -1)[:2], Qp.p(W / 2 + 1, 0, -1)[:2], Qp.p(W / 2 + 1, 0, L + 1)[:2], Qp.p(-W / 2 - 1, 0, L + 1)[:2]]))
        MEDIDAS[nome] = {"comprimento": L, "largura": W, "tabuleiro": p["tabuleiro"], "fundo_fosso": zfundo + ZREF, "arcos": nv}
        print(f"  {nome}: {L:.1f} m, {nv} arcos, tabuleiro {p['tabuleiro']:.1f}, fundo {zfundo + ZREF:.1f}")


def chao_casa(poly, d_):
    """Pé das paredes da casa da guarda: o ponto mais baixo do MDT à volta dela (fundo do fosso), menos 0,5 m."""
    b = np.array(poly.buffer(1.0).exterior.coords)
    return min(float(cota(*b.T).min()) - 0.5, d_["soleira"] - 1.0) - ZREF


def frente_casa(poly, d_):
    """Os dois cantos da fachada da casa da guarda virada ao fosso (os mais avançados segundo a normal)."""
    n_ = unit(np.array(d_["normal"], float))
    b = np.array(poly.exterior.coords)[:-1]
    k = np.argsort(-((b - np.array(d_["ponto"])) @ n_))[:2]
    return b[k], n_


def remendos(m, d_, poly, zchao, pts, TP):
    """Entre a ponta da escarpa e o canto da casa da guarda fica um vão de 2–5 m sem muralha no traçado: um pano
    de alvenaria liga-os, do fundo do fosso ao topo da escarpa."""
    fr, n_ = frente_casa(poly, d_)
    for c in fr:
        dist = np.hypot(*(pts - c).T)
        k = int(np.argmin(dist))
        if dist[k] > 9.0 or dist[k] < 0.4:
            continue
        a = pts[k]
        u = unit(c - a)
        w = np.array([-u[1], u[0]])
        if w @ n_ < 0:
            w = -w
        b = [a + w * 0.3, c + w * 0.3, c - w * 1.6, a - w * 1.6]
        zt = float(TP[k])
        m.prisma(b, zchao, zt - 0.3, ALV_, topo=False)
        m.prisma(b, zt - 0.3, zt, C_)


def casa_guarda(m, d_, pid, pts=None, TP=None):
    """Casa da guarda (edifício do OSM sobre a passagem): dois corpos de alvenaria de cada lado da passagem, com a
    altura do MDS, e a laje de pedra por cima da abóbada; terraço de cantaria."""
    f = next((x for x in F["edificios"] if x["id"].startswith(d_["casa"])), None)
    if f is None:
        return
    poly = shapely.Polygon(geom(f)).buffer(0)
    n_ = unit(np.array(d_["normal"], float))
    t_ = np.array([n_[1], -n_[0]])
    p0 = np.array(d_["ponto"], float)
    v = d_["vao"]
    # só o que fica por trás da face (o frontispício vai à frente)
    tras = shapely.Polygon([p0 + t_ * 60 - n_ * 0.3, p0 - t_ * 60 - n_ * 0.3, p0 - t_ * 60 - n_ * 80, p0 + t_ * 60 - n_ * 80])
    faixa = shapely.Polygon([p0 + t_ * (v["r"] + 0.5) + n_ * 2, p0 - t_ * (v["r"] + 0.5) + n_ * 2,
                             p0 - t_ * (v["r"] + 0.5) - n_ * 80, p0 + t_ * (v["r"] + 0.5) - n_ * 80])
    corpo_ = poly.intersection(tras)
    zt = medir(np.array(poly.exterior.coords)[:-1], q=90) - ZREF
    zchao = chao_casa(poly, d_)
    zv = d_["soleira"] - ZREF + v.get("sobe", 0.0) + v["omb"] + v["r"] + 0.55
    zt = max(zt, zv + 0.8)
    # cobertura à prova de bomba em duas águas muito baixas, cumeeira no eixo da passagem (foto 19: a fachada
    # desce do meio para os lados); cada metade é plana
    s_ = (np.array(corpo_.exterior.coords) - p0) @ t_ if corpo_.geom_type == "Polygon" else np.array([1.0])
    meia = float(np.abs(s_).max())
    sobe = min(3.4, d_.get("cobertura", 0.15) * meia)
    zf = lambda b: zt + sobe * (1 - np.abs((b - p0) @ t_) / meia)
    meios = [shapely.Polygon([p0 + n_ * 90, p0 - n_ * 90, p0 - n_ * 90 + t_ * sgn * 90, p0 + n_ * 90 + t_ * sgn * 90]) for sgn in (1, -1)]
    lados = corpo_.difference(faixa)
    meio = corpo_.intersection(faixa)
    for peca, z0f in ((lados, lambda b: zchao), (meio, lambda b: zv)):
        for hp in meios:
            pp = peca.intersection(hp)
            for g in getattr(pp, "geoms", [pp]):
                if g.geom_type != "Polygon" or g.area < 1:
                    continue
                b = np.array(g.exterior.coords)[:-1]
                if area(b) < 0:
                    b = b[::-1]
                zz = zf(b)
                m.prisma(b, z0f(b), zz - 0.25, ALV_, topo=False, fundo=peca is meio)
                m.prisma(b, zz - 0.25, zz, C_)
    # cordão de meia-cana a toda a largura da fachada (o plano da frente do corpo, 0,3 m atrás do ponto da porta),
    # à altura do alto do arco (fotografias das duas portas)
    if corpo_.geom_type == "Polygon":
        ss_ = (np.array(corpo_.exterior.coords) - p0) @ t_
        zcd = d_["soleira"] - ZREF + v["omb"] + v["r"] + 0.3
        cc = p0 + t_ * (ss_.min() + ss_.max()) / 2 - n_ * 0.3 + n_ * 0.15
        m.caixa(cc, t_, float(ss_.max() - ss_.min()), 0.3, zcd - 0.17, zcd + 0.17, C_)
    if d_.get("chamine"):
        c = p0 - n_ * d_["chamine"][1] + t_ * d_["chamine"][0]
        zc_ = float(zf(c[None])[0])
        m.caixa(c, t_, 1.1, 1.1, zc_ - 0.5, zc_ + 1.6, ALV_)
        m.caixa(c, t_, 1.35, 1.35, zc_ + 1.6, zc_ + 1.85, C_)
    zt = zt + sobe
    if pts is not None:
        remendos(m, d_, poly, zchao, pts, TP)
    EVITAR.append(np.array(poly.exterior.coords)[:-1])
    MEDIDAS.setdefault("casas_guarda", {})[pid] = {"topo": zt + ZREF, "area": round(poly.area, 1)}


def face_porta(d_):
    """Ponto da frontaria ao nível da soleira: o pé da escarpa (face em talude) em frente do ponto da porta, ou o
    próprio ponto nas portas sem escarpa. Devolve também o índice da amostra da escarpa (ou None)."""
    p0, n = np.array(d_["ponto"], float), unit(np.array(d_["normal"], float))
    if d_.get("sem_escarpa"):
        return p0
    pts = np.vstack([d["xy"] for d in ESC_DADOS])
    cat = lambda k: np.concatenate([np.broadcast_to(d[k], (len(d["xy"]),) + np.shape(d[k])[1:]) for d in ESC_DADOS])
    NR, A_, BT, ZC = cat("nr"), cat("a"), cat("bt"), cat("zc")
    dist = np.hypot(*(pts - p0).T)
    ok = (dist < 7.0) & (NR @ n > 0.85)
    if not ok.any():
        ok = dist < 9.0
    if not ok.any():
        return p0
    k = int(np.nonzero(ok)[0][np.argmin(dist[ok])])
    tq = np.array([-n[1], n[0]])
    q = pts[k] + tq * float((p0 - pts[k]) @ tq)
    return q + n * (A_[k] + BT[k] * (ZC[k] - (d_["soleira"] - ZREF)) + 0.02)


def portas():
    pts = np.vstack([d["xy"] for d in ESC_DADOS])
    cat = lambda k: np.concatenate([np.broadcast_to(d[k], (len(d["xy"]),) + np.shape(d[k])[1:]) for d in ESC_DADOS])
    NR, A_, BT, TP, PE, ZC = cat("nr"), cat("a"), cat("bt"), cat("topo"), cat("pe"), cat("zc")
    mp = Malha()
    gravar_pontes(mp)
    mp.objeto("Pontes", "pontes", MAT_PORTA, GRUPOS["Portas"], uv=2.5)
    for pid, d_ in PORTAS.items():
        p0, n0 = np.array(d_["ponto"], float), unit(np.array(d_["normal"], float))
        sole = d_["soleira"] - ZREF
        m = Malha()
        if d_.get("sem_escarpa"):
            F0, n = p0, n0
            Q = Quadro(m, F0, n)
            Q.topo = sole + 8.0
            Q.zc = sole + 5.0
        else:
            dist = np.hypot(*(pts - p0).T)
            ok = (dist < 7.0) & (NR @ n0 > 0.85)
            if not ok.any():
                ok = dist < 9.0
            if not ok.any():
                print("  porta sem escarpa:", pid)
                continue
            k = int(np.nonzero(ok)[0][np.argmin(dist[ok])])
            n = n0
            tq = np.array([-n[1], n[0]])
            q = pts[k] + tq * float((p0 - pts[k]) @ tq)
            F0 = q + n * (A_[k] + BT[k] * (ZC[k] - sole) + 0.02)
            Q = Quadro(m, F0, n)
            Q.bt, Q.zb = float(BT[k]), sole
            Q.topo = float(TP[k])
            Q.zc = float(ZC[k])
        alt = d_["estilo"](Q, sole)
        if "casa" in d_:
            casa_guarda(m, d_, pid, pts, TP)
        if d_["vao"].get("saida"):  # boca do lado de dentro, à saída da passagem
            v = d_["vao"]
            Qd = Quadro(m, F0 - n * v["tunel"], -n)
            zs = sole + v.get("sobe", 0.0)
            if v.get("reto"):
                for sg in (-1, 1):
                    a_, b_ = sorted((sg * v["r"], sg * (v["r"] + 2.5)))
                    Qd.bloco(a_, b_, zs - 0.3, zs + v["omb"] + 1.2, -0.4, 0.05, ALV_)
                Qd.bloco(-v["r"] - 2.5, v["r"] + 2.5, zs + v["omb"], zs + v["omb"] + 1.2, -0.4, 0.05, ALV_)
                Qd.bloco(-v["r"] - 2.6, v["r"] + 2.6, zs + v["omb"] + 1.2, zs + v["omb"] + 1.4, -0.4, 0.12, C_)
            else:
                pano_com_arco(Qd, -v["r"] - 1.2, v["r"] + 1.2, zs, zs + v["omb"] + v["r"] + 1.0, v["r"], v["omb"], -0.4, 0.05, ALV_)
                arco_cantaria(Qd, v["r"], zs, v["omb"], aduela=0.45, d0=0.03, d1=0.12)
        else:  # passagem cega (Santo António de dentro: o túnel dobra para o pátio, sem prova do traçado): fundo escuro
            v = d_["vao"]
            Qf = Quadro(m, F0 - n * v["tunel"], n)
            Qf.bloco(-v["r"] - 0.3, v["r"] + 0.3, sole - 0.2, sole + v["omb"] + v["r"] + 0.3, -0.3, 0.0, S_)
        m.objeto(pid.title(), d_["elemento"], MAT_PORTA, GRUPOS["Portas"], uv=2.5)
        PONTOS_PORTA[pid] = (F0, n, sole)
        tt = np.array([-n[1], n[0]])
        lg = d_.get("largura", 3.0)
        EVITAR.append(np.array([F0 + tt * (lg + 1) + n * 0.3, F0 + tt * (lg + 1) + n * 14, F0 - tt * (lg + 1) + n * 14, F0 - tt * (lg + 1) + n * 0.3]))
        EVITAR.append(np.array([F0 + tt * (lg / 2 + 1), F0 - tt * (lg / 2 + 1), F0 - tt * (lg / 2 + 1) - n * (d_["vao"]["tunel"] + 3),
                                F0 + tt * (lg / 2 + 1) - n * (d_["vao"]["tunel"] + 3)]))
        MEDIDAS[pid] = {"soleira": sole + ZREF, "topo_muralha": float(Q.topo + ZREF), "altura_frontaria": float(alt),
                        "vao": d_["vao"], "face": [round(float(F0[0]), 2), round(float(F0[1]), 2)]}
        print(f"  {pid}: soleira {sole + ZREF:.1f}, topo {Q.topo + ZREF:.1f}, frontaria {alt:.1f} m")


# ---------------------------------------------------------------- igrejas, torre do relógio, quartel e castelo
# Medidas tiradas do MDS de 50 cm (cumeeiras, torres, pegadas) e pormenores das fotografias cedidas.
# Cada igreja tem um referencial Quadro na fachada principal: s ao longo da fachada, d para fora; o
# corpo da igreja fica em d < 0.

CAL, CAN, TEL, SOM, MAD, FER, BRO, VID, LOU, LIS = range(10)
MAT_IGR = ["cal", "cantaria", "telha", "sombra", "madeira_verde", "ferro", "bronze", "vidro", "lousa", "cantaria_lisa", "ocre",
           "vermelho", "alvenaria", "madeira", "telha_quartel"]
OCR, VER, ALVI, MADC, TELQ = 10, 11, 12, 13, 14


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


def portal_arquivoltas(Q, s0, z0, w, hn, n=3, passo=0.24, prof=0.5):
    """Portal românico (Santa Maria dos Anjos): arquivoltas de volta perfeita em degrau, cada uma sobre
    colunelos com capitel e base, tímpano liso e porta de madeira; num corpo ligeiramente saliente."""
    R = w / 2 + (n + 1) * passo
    # corpo saliente do portal, com remate em cornija
    for sg in (-1, 1):
        ss = sorted((s0 + sg * R, s0 + sg * (R + 0.45)))
        Q.bloco(ss[0], ss[1], z0, z0 + hn + R + 0.5, 0, prof, CAN)
    Q.bloco(s0 - R - 0.45, s0 + R + 0.45, z0 + hn + R, z0 + hn + R + 0.5, 0, prof, CAN)
    Q.bloco(s0 - R - 0.55, s0 + R + 0.55, z0 + hn + R + 0.5, z0 + hn + R + 0.68, 0, prof + 0.1, CAN)
    for i in range(n + 1):
        r0 = w / 2 + i * passo
        di = prof * (i + 1) / (n + 1)
        for sg in (-1, 1):
            ss = sorted((s0 + sg * r0, s0 + sg * (r0 + passo)))
            Q.bloco(ss[0], ss[1], z0, z0 + hn, 0, di, CAN)
        Q.arco_sz(s0, r0, r0 + passo, z0 + hn, 0, di, CAN, k=14, aduelas=True)
        # tímpano: ocupa por dentro da primeira arquivolta
        if i == 0:
            continue
        # o espaço entre o degrau i e o i-1, em planta, recebe um colunelo
        for sg in (-1, 1):
            sc = s0 + sg * (r0 + 0.02)
            c = Q.p(sc, 0, prof * i / (n + 1) + 0.1)[:2]
            Q.m.cilindro(c, 0.095, z0 + 0.35, z0 + hn - 0.3, CAN, k=8)
            Q.bloco(sc - 0.15, sc + 0.15, z0, z0 + 0.35, prof * i / (n + 1) - 0.05, prof * i / (n + 1) + 0.25, CAN)
            Q.bloco(sc - 0.17, sc + 0.17, z0 + hn - 0.3, z0 + hn, prof * i / (n + 1) - 0.07, prof * i / (n + 1) + 0.27, CAN)
    # porta e tímpano
    _vao_desloc(Q, s0, w, z0, hn - 0.15, 0.012, MAD, reto=True)
    Q.bloco(s0 - w / 2, s0 + w / 2, z0 + hn - 0.2, z0 + hn, 0.0, 0.08, CAN)  # lintel
    tp = [(s0 + w / 2 * math.cos(math.pi * k / 12), z0 + hn + w / 2 * math.sin(math.pi * k / 12)) for k in range(13)]
    Q.placa(tp, 0.0, 0.05, CAN)
    Q.bloco(s0 - R - 0.8, s0 + R + 0.8, z0 - 0.18, z0, -0.1, prof + 0.9, CAN)  # degrau


def rosacea(Q, s0, zc, r, d=0.0, raios=8):
    """Rosácea (Misericórdia): anel moldurado de granito, vitral escuro e raios de pedra."""
    k = 24
    Q.m.face([Q.p(s0 + r * math.cos(2 * math.pi * i / k), zc + r * math.sin(2 * math.pi * i / k), d + 0.012) for i in range(k)], VID)
    Q.arco_sz(s0, r, r + 0.22, zc, d, d + 0.12, CAN, a0=0, a1=2 * math.pi, k=k)
    Q.arco_sz(s0, r + 0.22, r + 0.32, zc, d, d + 0.06, CAN, a0=0, a1=2 * math.pi, k=k)
    Q.arco_sz(s0, 0.0, r * 0.18, zc, d, d + 0.08, CAN, a0=0, a1=2 * math.pi, k=10)
    for i in range(raios):
        t = 2 * math.pi * i / raios
        c, s_ = math.cos(t), math.sin(t)
        e = 0.035
        p0, p1 = r * 0.18, r
        Q.placa([(s0 + c * p0 - s_ * e, zc + s_ * p0 + c * e), (s0 + c * p1 - s_ * e, zc + s_ * p1 + c * e),
                 (s0 + c * p1 + s_ * e, zc + s_ * p1 - c * e), (s0 + c * p0 + s_ * e, zc + s_ * p0 - c * e)], d, d + 0.06, CAN)


def sino(m, c, z, r=0.32, mat=BRO):
    """Sino de bronze (sino, cabeça e jugo)."""
    m.cilindro(c, r, z, z + r * 0.35, mat, k=12, r1=r * 0.8)
    m.cilindro(c, r * 0.8, z + r * 0.35, z + r * 1.2, mat, k=12, r1=r * 0.5)
    m.esfera((c[0], c[1], z + r * 1.2), r * 0.5, mat, k=10, achata=0.6)


def torre(m, c, u, lado, z0, zcorpo, zsino, estilo, mw=CAL, faces_janela=(), zjanela=None, oculo=None):
    """Torre sineira: corpo caiado com cunhais de granito, cornija, sineira aberta com um arco por face e
    sino, cornija de remate e cobertura (agulha de pedra, agulha oitavada com pináculos, ou cúpula)."""
    u = unit(np.asarray(u, float))
    w = np.array([-u[1], u[0]])
    c = np.asarray(c, float)
    h = lado / 2
    m.caixa(c, u, lado, lado, z0 - 1.5, zcorpo, mw)
    m.caixa(c, u, lado + 0.14, lado + 0.14, z0 - 1.5, z0 + 0.9, CAN)  # soco
    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):  # cunhais
        cc = c + (u * a + w * b) * (h - 0.25)
        m.caixa(cc, u, 0.66, 0.66, z0 + 0.9, zsino, CAN)
    faces = [(u, w), (-u, -w), (w, -u), (-w, u)]
    QS = [Quadro(m, c + n_ * h, n_) for n_, _ in faces]
    for i, Qf in enumerate(QS):
        if i in faces_janela and zjanela is not None:
            janela(Qf, 0.0, zjanela, 0.5, 1.0, arco=False, mat_vidro=SOM)
    if oculo is not None:  # óculo (mostrador) numa face
        rosacea(QS[oculo[0]], 0.0, oculo[1], 0.42, raios=0)
    # cornija entre o corpo e a sineira
    m.caixa(c, u, lado + 0.3, lado + 0.3, zcorpo - 0.25, zcorpo, CAN)
    m.caixa(c, u, lado + 0.44, lado + 0.44, zcorpo, zcorpo + 0.16, CAN)
    # sineira oca: quatro pilares de canto, peitoris, arcos e piso
    za = zcorpo + 0.16
    wv = lado - 1.5  # largura do vão
    hn = (zsino - za) - wv / 2 - 0.55
    m.caixa(c, u, lado - 0.2, lado - 0.2, za - 0.2, za + 0.02, SOM)
    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        cc = c + (u * a + w * b) * (h - 0.375)
        m.caixa(cc, u, 0.75, 0.75, za, zsino, mw)
    for Qf in QS:
        Qf.bloco(-wv / 2, wv / 2, za, za + 0.75, -0.6, 0.0, mw)  # peitoril
        Qf.bloco(-wv / 2 - 0.05, wv / 2 + 0.05, za + 0.75, za + 0.85, -0.65, 0.05, CAN)
        top = [(wv / 2, za + hn), (wv / 2, zsino), (-wv / 2, zsino), (-wv / 2, za + hn)]
        top += [(wv / 2 * math.cos(math.pi - math.pi * k / 12), za + hn + wv / 2 * math.sin(math.pi - math.pi * k / 12)) for k in range(1, 12)]
        Qf.placa(top, -0.6, 0.0, mw)
        Qf.arco_sz(0.0, wv / 2, wv / 2 + 0.2, za + hn, 0.0, 0.07, CAN, k=12, aduelas=True)
        for sg in (-1, 1):
            ss = sorted((sg * wv / 2, sg * (wv / 2 + 0.2)))
            Qf.bloco(ss[0], ss[1], za + 0.85, za + hn, 0.0, 0.07, CAN)
        Qf.bloco(-wv / 2 - 0.25, wv / 2 + 0.25, za + hn - 0.14, za + hn, 0.0, 0.1, CAN)  # impostas
    m.caixa(c, u, lado - 1.4, lado - 1.4, zsino - 0.3, zsino, mw)  # tecto da sineira
    sino(m, c + u * 0.45, za + 0.95, r=0.4)
    sino(m, c - u * 0.5, za + 1.15, r=0.3)
    Qs = Quadro(m, c, u)
    Qs.bloco(-0.04, 0.04, za + 1.6, zsino - 0.3, -h + 0.4, h - 0.4, MAD)  # jugo de madeira
    # cornija de remate
    m.caixa(c, u, lado + 0.3, lado + 0.3, zsino, zsino + 0.22, CAN)
    m.caixa(c, u, lado + 0.5, lado + 0.5, zsino + 0.22, zsino + 0.4, CAN)
    zt = zsino + 0.4
    if estilo == "agulha_pedra":  # Misericórdia: cachorrada de remate e agulha piramidal alta
        for Qf in QS:
            for s_ in np.linspace(-h + 0.25, h - 0.25, 6):
                Qf.bloco(s_ - 0.12, s_ + 0.12, zsino - 0.45, zsino, 0.0, 0.26, CAN)
            Qf.bloco(-h - 0.25, h + 0.25, zt, zt + 0.35, 0.0, 0.25, CAN)  # platibanda
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cc = c + (u * a + w * b) * (h + 0.05)
            Qc = Quadro(m, cc, u)
            pinaculo(Qc, 0, zt, 0, esc=0.7, tipo="agulha")
        piramide(m, c, u, lado - 0.3, zt, zt + 0.25, CAN, lado1=lado - 0.5)
        piramide(m, c, u, lado - 0.5, zt + 0.25, zt + 5.0, CAN)
        m.esfera((c[0], c[1], zt + 5.05), 0.14, CAN)
        return zt + 5.2
    if estilo == "ameias":  # Matriz de Almeida (fotografia): platibanda com ameias e pináculos nos cantos
        for Qf in QS:
            Qf.bloco(-h - 0.15, h + 0.15, zt, zt + 0.45, -0.45, 0.15, CAN)
            for s_ in np.linspace(-h + 0.35, h - 0.35, 4):
                Qf.bloco(s_ - 0.32, s_ + 0.32, zt + 0.45, zt + 1.25, -0.45, 0.15, CAN)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cc = c + (u * a + w * b) * (h + 0.0)
            pinaculo(Quadro(m, cc, u), 0, zt + 1.25, 0, esc=1.0, tipo="bola")
        return zt + 2.3
    if estilo == "agulha_pinaculos":  # Santa Maria dos Anjos: pináculos nos cantos e agulha oitavada curta
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cc = c + (u * a + w * b) * (h - 0.1)
            pinaculo(Quadro(m, cc, u), 0, zt, 0, esc=0.95, tipo="bola")
        oitavado(m, c, lado * 0.32, lado * 0.26, zt, zt + 0.5, CAN)
        oitavado(m, c, lado * 0.24, 0.0, zt + 0.5, zt + 2.6, CAN)
        m.esfera((c[0], c[1], zt + 2.75), 0.18, CAN)
        cruz(Qs, 0, zt + 2.9, 0, alt=1.2, mat=FER, esp=0.06)
        return zt + 4.1
    if estilo == "lanterna":  # Torre do Relógio (Street View): balaustrada, pináculos, lanterna pequena e cata-vento
        for Qf in QS:
            Qf.bloco(-h - 0.2, h + 0.2, zt, zt + 0.12, -0.2, 0.2, CAN)
            for s_ in np.arange(-h + 0.3, h - 0.2, 0.32):
                Qf.m.cilindro(Qf.p(s_, 0, 0.0)[:2], 0.07, zt + 0.12, zt + 0.72, CAN, k=6)
            Qf.bloco(-h - 0.2, h + 0.2, zt + 0.72, zt + 0.86, -0.2, 0.2, CAN)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cc = c + (u * a + w * b) * (h + 0.05)
            pinaculo(Quadro(m, cc, u), 0, zt + 0.86, 0, esc=1.1, tipo="bola")
        # pedestal em degraus e lanterna quadrada de cantaria com um arco por face, capelo e cata-vento
        m.caixa(c, u, lado * 0.46, lado * 0.46, zt, zt + 0.55, mw)
        m.caixa(c, u, lado * 0.52, lado * 0.52, zt + 0.55, zt + 0.7, CAN)
        m.caixa(c, u, lado * 0.36, lado * 0.36, zt + 0.7, zt + 1.0, CAN)
        lt = lado * 0.3
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            m.caixa(c + (u * a + w * b) * (lt / 2 - 0.1), u, 0.22, 0.22, zt + 1.0, zt + 2.0, CAN)
        m.caixa(c, u, lt * 0.55, lt * 0.55, zt + 1.0, zt + 1.9, SOM)
        m.caixa(c, u, lt + 0.12, lt + 0.12, zt + 2.0, zt + 2.15, CAN)
        cupula(m, c, lt * 0.5, zt + 2.15, CAN, k=12, aneis=4, alt=0.55)
        m.esfera((c[0], c[1], zt + 2.8), 0.12, CAN)
        cruz(Qs, 0, zt + 2.9, 0, alt=1.1, mat=FER, esp=0.06)
        return zt + 4.0
    if estilo == "cupula":  # Santo Estêvão: balaustrada, pináculos, tambor oitavado, cúpula, lanterna e cruz
        for Qf in QS:
            Qf.bloco(-h - 0.2, h + 0.2, zt, zt + 0.12, -0.2, 0.2, CAN)
            for s_ in np.arange(-h + 0.3, h - 0.2, 0.32):
                Qf.m.cilindro(Qf.p(s_, 0, 0.0)[:2], 0.07, zt + 0.12, zt + 0.72, CAN, k=6)
            Qf.bloco(-h - 0.2, h + 0.2, zt + 0.72, zt + 0.86, -0.2, 0.2, CAN)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cc = c + (u * a + w * b) * (h + 0.05)
            pinaculo(Quadro(m, cc, u), 0, zt + 0.86, 0, esc=1.1, tipo="bola")
        r = lado * 0.36
        oitavado(m, c, r, r, zt, zt + 1.5, CAL)
        oitavado(m, c, r + 0.1, r + 0.1, zt + 1.5, zt + 1.7, CAN)
        cupula(m, c, r + 0.05, zt + 1.7, CAN, alt=r * 1.05)
        zl = zt + 1.7 + r * 1.05
        oitavado(m, c, 0.32, 0.32, zl - 0.05, zl + 0.65, CAN)
        cupula(m, c, 0.36, zl + 0.65, CAN, k=8, aneis=3, alt=0.3)
        m.esfera((c[0], c[1], zl + 1.05), 0.12, CAN)
        cruz(Qs, 0, zl + 1.15, 0, alt=1.3, mat=FER, esp=0.07)
        return zl + 2.45
    if estilo == "coroa_ferro":  # Santo Estêvão (fotografia): pináculos de chama, cupulim e coroa de ferro
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cc = c + (u * a + w * b) * (h - 0.05)
            pinaculo(Quadro(m, cc, u), 0, zt, 0, esc=1.0, tipo="agulha")
        m.caixa(c, u, lado * 0.55, lado * 0.55, zt, zt + 0.55, CAN)
        m.caixa(c, u, lado * 0.62, lado * 0.62, zt + 0.55, zt + 0.68, CAN)
        cupula(m, c, lado * 0.27, zt + 0.68, CAN, k=12, aneis=4, alt=0.6)
        zf = zt + 1.25
        anel = [np.array([c[0] + 0.55 * math.cos(t_), c[1] + 0.55 * math.sin(t_)]) for t_ in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        topo_ = np.array([c[0], c[1], zf + 2.0])
        for p_ in anel:  # oito barras curvas que se juntam em cima
            p1 = np.array([*(c + (p_ - c) * 1.25), zf + 0.8])
            barra(m, [*p_, zf], p1, 0.03, FER)
            barra(m, p1, topo_, 0.03, FER)
        for zz, rr in ((zf, 0.55), (zf + 0.8, 0.69)):
            pts_ = [np.array([c[0] + rr * math.cos(t_), c[1] + rr * math.sin(t_), zz]) for t_ in np.linspace(0, 2 * math.pi, 9)]
            for p0_, p1_ in zip(pts_[:-1], pts_[1:]):
                barra(m, p0_, p1_, 0.025, FER)
        m.esfera(tuple(topo_ + [0, 0, 0.12]), 0.13, FER)
        barra(m, topo_ + [0, 0, 0.2], topo_ + [0, 0, 1.3], 0.03, FER)
        barra(m, topo_ + [*(w * -0.3), 0.95], topo_ + [*(w * 0.3), 0.95], 0.025, FER)
        return zf + 3.3
    return zt


def _rel(F0, n, p):
    """Coordenadas (s, d) de um ponto do mapa no Quadro da fachada (F0, n)."""
    t = np.array([n[1], -n[0]])
    q = np.asarray(p, float) - F0
    return float(q @ t), float(q @ n)


def igreja_matriz(m):
    """Igreja Matriz de Nossa Senhora das Candeias (fotografia de D. Perez, Commons; câmara a poente, GPS):
    fachada de cantaria virada a poente, empena com cornija, pináculos e cruz, portal de arco pleno e janela
    por cima; torre sineira no canto sudoeste, quase à face da fachada, com sineira, relógio e ameias.
    Medidas do MDS: nave de 24 x 9,5 m, beiral a 747,9 e cumeeira a 749,9 m; cabeceira mais larga (11,5 m)
    e mais baixa (cumeeira 749,1 m); torre de 5 m de lado com o remate a 758,7 m (pináculos a 760,8 m)."""
    n = unit(np.array([-0.995, 0.0999]))
    F0 = np.array([144.3, -43.6])
    z0 = 741.2 - ZREF
    Q = Quadro(m, F0, n)
    zr = corpo(Q, 24.0, 9.5, z0, z0 + 6.7, inc=23, mw=CAN, mr=TEL)
    sc, _ = _rel(F0, n, (172.0, -47.3))
    Qc = Quadro(m, F0 + Q.t * sc, n)
    corpo(Qc, 11.5, 11.5, z0, z0 + 5.8, inc=20, mw=CAN, mr=TEL, d_ini=-23.5, cornija_frente=False)
    porta_igreja(Q, 0.0, z0, 1.7, 2.7, moldura=0.32, arco=True, folhas=MADC)  # porta de madeira castanha (fotografia)
    janela(Q, 0.0, z0 + 4.4, 0.9, 1.2, arco=False, mat_vidro=VID)
    for sg in (-1, 1):
        pinaculo(Q, sg * 4.55, z0 + 6.75, 0.0, esc=1.1, tipo="bola")
    cruz(Q, 0.0, zr + 0.35, 0.0, alt=1.5, mat=CAN, esp=0.16)
    for k in range(3):  # janelas altas da nave, a sul
        Qs = Quadro(m, F0 - n * (6.0 + 6.5 * k) - Q.t * 4.75, -Q.t)
        janela(Qs, 0.0, z0 + 3.8, 0.8, 1.6, arco=True)
    ts, td = _rel(F0, n, (146.0, -50.4))
    c = F0 + Q.t * ts + n * td
    torre(m, c, n, 5.0, z0, z0 + 11.6, z0 + 15.8, "ameias", mw=CAN, oculo=(0, z0 + 10.2))
    return np.array([F0 + Q.t * 4.9 + n * 0.5, F0 + Q.t * -8.8 + n * 0.5, F0 + Q.t * -8.8 - n * 36, F0 + Q.t * 4.9 - n * 36])


def igreja_misericordia(m):
    """Igreja da Misericórdia (fotografias de Hovallef e D. Perez, Commons, com GPS na Praça da República):
    fachada barroca de granito encaixada no quarteirão, virada a nordeste. A nave (10 m) é a empena que o
    MDS mostra a meio do lado nordeste do quarteirão, cumeeira perpendicular à rua a c. 752,4 m. Portal de
    verga reta com frontão curvo e brasão, duas janelas de sacada com grade, pilastras, pináculos e cruz."""
    n = unit(np.array([0.584, 0.812]))
    F0 = np.array([79.46, -128.62])
    z0 = float(cota(*(F0 + n * 1.5))) - ZREF
    W = 10.0
    Q = Quadro(m, F0, n)
    zr = corpo(Q, 20.0, W, z0, z0 + 5.9, inc=22, mw=CAL, mr=TEL, cunhais=False)
    tg = math.tan(_ang(22))
    # pano de cantaria da fachada (à frente da parede), com pilastras
    Q.placa([(-W / 2, z0 - 0.8), (W / 2, z0 - 0.8), (W / 2, z0 + 5.9), (0, zr), (-W / 2, z0 + 5.9)], 0.0, 0.12, CAN)
    for s_ in (-W / 2 + 0.3, -1.9, 1.9, W / 2 - 0.3):
        Q.bloco(s_ - 0.3, s_ + 0.3, z0 - 0.3, z0 + 5.7, 0.12, 0.3, CAN)
        Q.bloco(s_ - 0.38, s_ + 0.38, z0 + 5.7, z0 + 5.9, 0.12, 0.36, CAN)
    Q.bloco(-W / 2 - 0.1, W / 2 + 0.1, z0 + 5.75, z0 + 6.0, 0.0, 0.4, CAN)  # entablamento
    Q.bloco(-W / 2 - 0.1, W / 2 + 0.1, z0 - 0.8, z0 + 0.15, 0.0, 0.6, CAN)  # degraus
    # portal: verga reta, frontão curvo interrompido e brasão
    _vao_desloc(Q, 0.0, 1.7, z0 + 0.15, 2.9, 0.13, VER, reto=True)
    for sg in (-1, 1):
        ss = sorted((sg * 0.85, sg * 1.25))
        Q.bloco(ss[0], ss[1], z0 + 0.15, z0 + 3.05, 0.12, 0.34, CAN)
    Q.bloco(-1.35, 1.35, z0 + 3.05, z0 + 3.35, 0.12, 0.4, CAN)
    Q.arco_sz(0.0, 1.05, 1.3, z0 + 3.35, 0.12, 0.42, CAN, a0=0.0, a1=math.pi * 0.42, k=5)
    Q.arco_sz(0.0, 1.05, 1.3, z0 + 3.35, 0.12, 0.42, CAN, a0=math.pi * 0.58, a1=math.pi, k=5)
    Q.escudo(0.0, z0 + 3.55, 0.75, 0.85, 0.12, 0.3, CAN)
    # janelas de sacada
    for sg in (-1, 1):
        sj = sg * 3.15
        janela(Q, sj, z0 + 3.4, 1.0, 1.7, arco=False, d=0.12, mat_vidro=SOM)
        Q.bloco(sj - 0.75, sj + 0.75, z0 + 3.25, z0 + 3.4, 0.12, 0.62, CAN)
        for zz in (z0 + 3.45, z0 + 4.25):
            barra(m, (*Q.p(sj - 0.7, zz, 0.58)[:2], zz), (*Q.p(sj + 0.7, zz, 0.58)[:2], zz), 0.025, FER)
        for s_ in np.linspace(sj - 0.7, sj + 0.7, 8):
            barra(m, Q.p(s_, z0 + 3.4, 0.58), Q.p(s_, z0 + 4.25, 0.58), 0.015, FER)
        pinaculo(Q, sg * (W / 2 - 0.2), z0 + 6.0, 0.15, esc=1.0, tipo="bola")
    rosacea(Q, 0.0, z0 + 6.9, 0.45, d=0.12, raios=0)
    cruz(Q, 0.0, zr + 0.3, 0.1, alt=1.4, mat=CAN, esp=0.15)
    return np.array([F0 + Q.t * 5.3 + n * 1.0, F0 - Q.t * 5.3 + n * 1.0, F0 - Q.t * 5.3 - n * 20.5, F0 + Q.t * 5.3 - n * 20.5])


def torre_relogio(m):
    """Torre do Relógio (fotografias de Hovallef e D. Perez, Commons): torre quadrada rebocada a ocre com
    cunhais e cornijas de granito, porta e janelas, mostrador do relógio, sineira com um arco por face e
    remate com balaustrada, pináculos nos cantos e uma lanterna pequena com cata-vento (Street View, 10 nov.). Planta do OSM e do MDS (5,5 m de lado,
    alinhada com os eixos); chão a 762,7 m, balaustrada a c. 781 m e lanterna a c. 786 m (MDS)."""
    c = np.array([-56.2, 37.6])
    u = np.array([1.0, 0.0])
    z0 = 762.7 - ZREF
    torre(m, c, u, 5.5, z0, z0 + 12.9, z0 + 16.7, "lanterna", mw=OCR, faces_janela=(0, 1, 2, 3), zjanela=z0 + 5.6,
          oculo=(0, z0 + 10.6))
    Qe = Quadro(m, c + u * 2.75, u)
    porta_igreja(Qe, 0.0, z0, 1.2, 2.3, moldura=0.25, arco=True, folhas=MAD)
    for Qf in [Quadro(m, c + v * 2.75, v) for v in (u, -u, np.array([0.0, 1.0]), np.array([0.0, -1.0]))]:
        Qf.bloco(-2.9, 2.9, z0 + 8.3, z0 + 8.6, -0.2, 0.12, CAN)  # cornija entre andares
    return np.array([c + (-3.3, -3.3), c + (3.3, -3.3), c + (3.3, 3.3), c + (-3.3, 3.3)])


def quartel(m):
    """Quartel das Esquadras (fotografia de D. Perez, Commons, e Street View de 10 nov. dos dois lados): corpo
    comprido de dois pisos, paredes caiadas, soco, cunhais e cornija de granito, telhado de quatro águas de telha
    velha, escura no ortofotomapa, com chaminés; no rés-do-chão dos DOIS lados compridos (Terreiro do Poço do
    Rancho a noroeste e Rua da Muralha a sudeste) uma arcaria de arcos plenos sobre pilares de granito, com uma
    porta verde ao fundo de cada arco, e por cima uma faixa de duas fiadas de granito; janelas pequenas no
    andar. Planta do OSM (105,5 x 14,6 m); beiral a c. 752,2 m e cumeeira a 755,8 m (MDS). O número de arcos
    (22 por lado), as janelas e as chaminés são contados por estimativa."""
    A, B, C, D = (np.array(p) for p in ((92.4, -135.2), (100.6, -147.3), (12.9, -206.0), (4.7, -193.9)))
    z0 = 743.9 - ZREF
    ze = z0 + 8.3
    base = np.array([D, C, B, A])
    m.prisma(base, z0 - 1.0, ze, CAL, topo=False)
    m.prisma(np.array([p + unit(p - base.mean(0)) * 0.1 for p in base]), z0 - 1.0, z0 + 0.3, CAN, topo=False)
    for p in base:  # cunhais
        k = unit(p - base.mean(0))
        m.caixa(p - k * 0.25, B - A if np.allclose(p, A) or np.allclose(p, B) else A - D, 0.7, 0.7, z0 + 0.3, ze, CAN)
    m.prisma(np.array([p + unit(p - base.mean(0)) * 0.3 for p in base]), ze - 0.3, ze, CAN)
    telhado_4aguas(m, base, ze, inclinacao=26, mat=TELQ, beiral=0.35)
    L = float(np.hypot(*(A - D)))
    W = float(np.hypot(*(B - A)))
    ax = unit(A - D)
    nw = np.array([-ax[1], ax[0]])
    if nw @ (D - C) < 0:
        nw = -nw
    passo, nar = 4.2, 22
    s0 = (L - passo * nar) / 2
    hn = 2.3  # impostas a c. 2,3 m e arcos plenos de 2,8 m de vão (Street View)
    for lado_, o_ in ((-nw, C), (nw, D)):
        Qa = Quadro(m, o_, lado_)
        sg = 1 if Qa.t @ ax > 0 else -1
        for k in range(nar):
            sc = sg * (s0 + passo * (k + 0.5))
            _vao_desloc(Qa, sc, 2.8, z0, hn, 0.012, CAL)  # fundo caiado da galeria
            porta_igreja(Qa, sc, z0 + 0.15, 0.95, 2.0, moldura=0.18, arco=False, d=0.02)
            Qa.arco_sz(sc, 1.4, 1.75, z0 + hn, 0.0, 0.12, CAN, k=10, aduelas=True)
        for k in range(nar + 1):
            sp = sg * (s0 + passo * k)
            Qa.bloco(sp - 0.7, sp + 0.7, z0, z0 + hn, 0.0, 0.16, CAN)
            Qa.bloco(sp - 0.78, sp + 0.78, z0 + hn - 0.15, z0 + hn, 0.0, 0.2, CAN)  # impostas
        sa, sb = sorted((sg * (s0 - 0.7), sg * (L - s0 + 0.7)))
        Qa.bloco(sa, sb, z0 + 3.9, z0 + 4.8, 0.0, 0.18, CAN)  # faixa de duas fiadas sobre a arcaria
        for k in range(nar):  # janelas pequenas do andar, uma por arco
            janela(Qa, sg * (s0 + passo * (k + 0.5)), z0 + 5.6, 0.75, 0.85, arco=False)
    # chaminés de granito só na água noroeste (Street View do Terreiro; a fotografia do lado sudeste não as mostra)
    for k in range(11):
        q = (D + C) / 2 + ax * (s0 + passo * (2 * k + 1)) + nw * (W * 0.22)
        m.caixa(q, ax, 0.7, 0.7, ze + W * 0.25 * math.tan(math.radians(26)), ze + 0.5 * W * math.tan(math.radians(26)) + 0.4, CAN)
    m.caixa((A + B) / 2 - ax * 1.2, ax, 0.9, 0.9, ze + 1.0, ze + 0.5 * W * math.tan(math.radians(26)) + 0.8, CAN)  # a do topo
    # topo nordeste (fotografia de D. Perez): brasão no alto e escadaria de granito em V encostada à empena,
    # dois lanços que sobem do meio para os cantos até ao nível da faixa sobre a arcaria
    Qn = Quadro(m, (A + B) / 2, ax)
    Qn.escudo(0.0, z0 + 5.9, 1.0, 1.2, 0.0, 0.2, CAN)
    n_deg, alto = 20, 4.4
    larg = (W / 2 - 0.4) / n_deg
    for sgn in (-1, 1):
        for i in range(n_deg):
            sa_, sb_ = sorted((sgn * (0.6 + i * larg), sgn * (0.6 + (i + 1) * larg)))
            Qn.bloco(sa_, sb_, z0 - 1.0, z0 + (i + 1) * alto / n_deg, 0.0, 2.2, CAN)
        for i in range(n_deg):  # guarda de cantaria por fora, em degraus
            sa_, sb_ = sorted((sgn * (0.6 + i * larg), sgn * (0.6 + (i + 1) * larg)))
            Qn.bloco(sa_, sb_, z0 + (i + 1) * alto / n_deg, z0 + (i + 1) * alto / n_deg + 0.9, 1.95, 2.2, CAN)
    Qn.bloco(-0.6, 0.6, z0 - 1.0, z0 + 0.15, 0.0, 3.4, CAN)  # degraus de arranque ao meio
    Qn.bloco(-2.2, 2.2, z0 - 1.0, z0 + 0.15, 2.2, 3.4, CAN)
    return base


def castelo(m):
    """Restos do castelo medieval no meio da praça (OSM, ortoimagem e relevo do MDT): as bases de três torres
    redondas nos cantos norte, este e oeste, que o relevo mostra como anéis de pedra a 2–3 m acima do fosso
    seco; o canto sul não tem torre visível. Diâmetros (8–11 m) e alturas tirados do MDT."""
    for c, r, zb, zt in (((-105.0, 27.0), 5.4, 756.4, 759.7), ((-66.0, -5.0), 4.6, 756.6, 759.8),
                         ((-124.0, -28.0), 4.6, 756.2, 759.2)):
        m.cilindro(c, r, zb - 0.8 - ZREF, zt - 0.4 - ZREF, ALVI, k=20)
        m.cilindro(c, r - 0.25, zt - 0.4 - ZREF, zt - ZREF, ALVI, k=20, topo=False)
        m.cilindro(c, r - 1.9, zt - ZREF - 1.4, zt - ZREF - 0.6, ALVI, k=16)  # miolo arrasado, mais baixo
    return np.array([(-130, -34), (-60, -34), (-60, 33), (-130, 33)]) * 0  # sem EVITAR


IGREJAS = [("Igreja_Matriz", "igreja_matriz", igreja_matriz), ("Igreja_Misericordia", "igreja_misericordia", igreja_misericordia),
           ("Torre_do_Relogio", "torre_relogio", torre_relogio), ("Quartel_das_Esquadras", "quartel_esquadras", quartel),
           ("Castelo", "castelo", castelo)]
# vistas de conferência dos monumentos (nome, azimute da câmara, elevação, distância, alvo, cota do alvo), do lado
# de onde foram tiradas as fotografias de referência
VISTAS_PERTO = [("m_matriz", 300, 14, 42, (150.0, -45.0), 747.0), ("m_misericordia", 45, 10, 30, (79.5, -128.6), 748.5),
                ("m_relogio", 66, 8, 48, (-56.2, 37.6), 773.0), ("m_quartel", 75, 12, 80, (70.0, -160.0), 747.0),
                ("m_castelo", 160, 35, 130, (-95.0, -3.0), 758.0),
                # das posições das fotografias de comparação (ver LEIAME)
                ("c_sf_dentro", 145, 4, 30, (123.8, -175.8), 746.0), ("c_sa_fora", 319, 6, 20, (-326.6, -19.7), 748.5),
                ("c_matriz", 292, 9, 24, (146.0, -44.5), 747.5), ("c_relogio", 63, 14, 34, (-56.2, 37.6), 773.0),
                ("c_quartel", 72, 2.5, 78, (66.0, -162.0), 747.5)]


def igrejas():
    for nome, el, fn in IGREJAS:
        m = Malha()
        planta = fn(m)
        m.objeto(nome, el, MAT_IGR, GRUPOS["Igrejas"], uv=2.5)
        if np.abs(planta).sum() > 0:
            EVITAR.append(np.array(planta, float))
        MEDIDAS[el] = {"faces": len(m.f)}
        print(f"  {nome}: {len(m.f)} faces")


# ================================================================ cena

bpy.ops.wm.read_factory_settings(use_empty=True)
RAIZ_OB = bpy.data.objects.new("Praca_Forte_Almeida", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = LAT0, LON0
RAIZ_OB["centro_m"] = [float(CENTRO[0]), float(CENTRO[1])]
RAIZ_OB["cota_zero"] = ZREF
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
for g in ["Terreno", "Muralhas", "Guaritas", "Portas", "Igrejas", "Estatua", "Vila", "Vegetacao"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

TEX_ALV, ALT_ALV = textura_alvenaria()
NRM_ALV = imagem("alvenaria_normal", normais(ALT_ALV, 2.5))
material("alvenaria", (1, 1, 1), 0.93, imagem("alvenaria", TEX_ALV), normal=NRM_ALV)
TEX_CANT, ALT_CANT = textura_cantaria()
NRM_CANT = imagem("cantaria_normal", normais(ALT_CANT, 3.0))
material("cantaria", (1, 1, 1), 0.88, imagem("cantaria", TEX_CANT), normal=NRM_CANT, forca=0.7)
material("cantaria_lisa", (0.70, 0.68, 0.63), 0.88)
material("relva", (0.47, 0.52, 0.30), 0.97, imagem("grao_relva", textura_grao()))
material("sombra", (0.08, 0.075, 0.07), 1.0)
material("cal", (0.93, 0.92, 0.88), 0.85)
material("ocre", (0.90, 0.82, 0.60), 0.85)  # reboco creme-amarelado desbotado (Torre do Relógio, fotografia)
material("rosa", (0.88, 0.76, 0.70), 0.85)
TEX_TELHA, ALT_TELHA = textura_telha()
material("telha", (1, 1, 1), 0.8, imagem("telha", TEX_TELHA * np.array([0.78, 0.40, 0.25])),
         normal=imagem("telha_normal", normais(ALT_TELHA, 2.0)), forca=0.6)
material("telha_velha", (1, 1, 1), 0.85, imagem("telha_v", TEX_TELHA * np.array([0.56, 0.40, 0.33])),
         normal=imagem("telha_v_normal", normais(ALT_TELHA, 2.0)), forca=0.6)
material("telha_quartel", (1, 1, 1), 0.88, imagem("telha_q", TEX_TELHA * np.array([0.47, 0.38, 0.33])),
         normal=imagem("telha_q_normal", normais(ALT_TELHA, 2.0)), forca=0.6)
material("lousa", (0.42, 0.42, 0.43), 0.8)
material("madeira_verde", (0.16, 0.33, 0.26), 0.7)
material("madeira", (0.40, 0.26, 0.16), 0.8)
material("vermelho", (0.50, 0.07, 0.06), 0.6)
material("ferro", (0.16, 0.17, 0.18), 0.5)
material("bronze", (0.24, 0.30, 0.25), 0.5)
material("vidro", (0.20, 0.24, 0.32), 0.25)
material("base", (0.33, 0.26, 0.20), 0.95)
material("terreno", (1, 1, 1), 0.95)
# casario em volumes simples num cinzento neutro claro (como no castillos.site), para a fortaleza sobressair
material("casa_parede", (0.80, 0.80, 0.79), 0.9)
material("casa_telhado", (0.70, 0.70, 0.69), 0.9)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)

MEDIDAS = {}

print("escarpas…")
m = Malha()
mq = Malha()
LINHAS_ESC = []
def cortar(e):
    """Corta as peças no vão das portas fixas: um vértice que cai dentro da passagem (até 1,9 m para cada
    lado do eixo e atrás da face) passa para a borda da passagem, ao longo do segmento que vem de fora;
    se os dois vizinhos também estão dentro, sai."""
    lin = np.array(e["linha"], float)
    chaves = ("topo", "pe", "aresta", "fim")
    for d_ in PORTAS.values():
        if not d_.get("fixo"):
            continue
        n_ = unit(np.array(d_["normal"], float))
        t_ = np.array([-n_[1], n_[0]])
        p0 = np.array(d_["ponto"])
        s = (lin - p0) @ t_
        dd = (lin - p0) @ n_
        dentro_ = (np.abs(s) < 1.9) & (dd > -2.0) & (dd < -0.3)
        if not dentro_.any():
            continue
        nova, vals = [], {k: [] for k in chaves}
        for i_ in range(len(lin)):
            if not dentro_[i_]:
                nova.append(lin[i_])
                for k in chaves:
                    vals[k].append(e[k][i_])
                continue
            for j_ in (i_ - 1, i_ + 1):  # vizinho de fora: corta o segmento na borda
                if 0 <= j_ < len(lin) and not dentro_[j_]:
                    alvo = 1.9 * np.sign(s[j_])
                    f = (alvo - s[j_]) / (s[i_] - s[j_]) if s[i_] != s[j_] else 0.0
                    q = lin[j_] + (lin[i_] - lin[j_]) * float(np.clip(f, 0, 1))
                    if j_ == i_ + 1:
                        nova.append(q)
                    else:
                        nova.append(q)
                    for k in chaves:
                        vals[k].append(e[k][i_])
        if len(nova) < 2:
            return e
        lin = np.array(nova)
        e = {**e, "linha": lin.tolist(), **vals}
    return e


def aparar_passagens(e):
    """Tira as pontas de um pano que caem dentro da passagem de uma porta, atrás da face (o retorno do muro do
    revelim de São Francisco junto à casa da guarda atravessava o túnel a 6 m da porta: teste de raios)."""
    ch = ("linha", "topo", "pe", "aresta", "fim")
    for d_ in PORTAS.values():
        n_ = unit(np.array(d_["normal"], float))
        t_ = np.array([-n_[1], n_[0]])
        p0 = np.array(d_["ponto"])
        r, L = d_["vao"]["r"], d_["vao"]["tunel"]
        dentro_ = lambda q: abs((np.asarray(q) - p0) @ t_) < r + 0.8 and -(L + 0.5) < (np.asarray(q) - p0) @ n_ < -1.5
        # retorno curto que sai do canto da porta para trás (e016, São Francisco de fora): sai inteiro
        perto_ = lambda q: np.hypot(*(np.asarray(q) - p0)) < 2.5
        atras_ = lambda q: (np.asarray(q) - p0) @ n_ < -1.5 and np.hypot(*(np.asarray(q) - p0)) < 8.0
        if len(e["linha"]) > 2 and perto_(e["linha"][1]) and atras_(e["linha"][0]):
            for k in ch:
                e[k] = e[k][1:]
        if len(e["linha"]) > 2 and perto_(e["linha"][-2]) and atras_(e["linha"][-1]):
            for k in ch:
                e[k] = e[k][:-1]
        while len(e["linha"]) > 2 and dentro_(e["linha"][0]):
            for k in ch:
                e[k] = e[k][1:]
        while len(e["linha"]) > 2 and dentro_(e["linha"][-1]):
            for k in ch:
                e[k] = e[k][:-1]
    return e


for e in ESC:
    LINHAS_ESC.append(escarpa(aparar_passagens(e), m, mq))
m.objeto("Escarpas", "escarpas", ["alvenaria", "cantaria", "relva"], GRUPOS["Muralhas"], uv=4.0)
mq.objeto("Cunhais", "escarpas", ["cantaria"], GRUPOS["Muralhas"], uv=3.0)
alturas = np.concatenate([d["h"] for d in ESC_DADOS])
MEDIDAS["escarpas"] = {"panos": len(ESC), "comprimento": float(sum(e["comprimento"] for e in ESC)),
                       "altura_p50": float(np.median(alturas)), "altura_p90": float(np.percentile(alturas, 90)),
                       "altura_max": float(alturas.max())}

print("guaritas…", len(GUARITAS))
m = Malha()
for c, bis, zc, topo in GUARITAS:
    # guarita redonda sobre mísula no ângulo, com o piso à altura do adarve (fotografia de um ângulo de baluarte)
    guarita_redonda(m, c, topo - 1.15, bis, r=0.72, h=2.0)
m.objeto("Guaritas", "guaritas", MAT_PORTA, GRUPOS["Guaritas"], uv=2.0)
MEDIDAS["guaritas"] = len(GUARITAS)

print("terreno…")
construir_terreno()
img_t = imagem("chao", textura_terreno()[::-1])  # o Blender guarda a linha 0 em baixo (sul)
mt = MATS["terreno"]
ti = mt.node_tree.nodes.new("ShaderNodeTexImage")
ti.image = img_t
ti.extension = "EXTEND"
mt.node_tree.links.new(ti.outputs["Color"], mt.node_tree.nodes["Principled BSDF"].inputs["Base Color"])

PROPRIOS = set()  # plantas do OSM que são modeladas à parte (igrejas, capelas)
PROPRIOS |= PROPRIOS_PORTAS
PROPRIOS |= {"1fe3360b", "6293e779", "a12c03fe"}  # Matriz, Torre do Relógio, Quartel das Esquadras
EVITAR = []       # plantas onde não vão casas nem árvores
if "portas" in globals():
    print("portas…")
    portas()
if "igrejas" in globals():
    print("igrejas, torre do relógio, quartel e castelo…")
    igrejas()

print("casario…")
m = Malha()
n_med = 0
pegadas = []
linhas_alto = [d["xy"] for d in ESC_DADOS]
arv_esc = cKDTree(np.vstack(linhas_alto))


RECUSAS = {"fora": 0, "evitar": 0, "escarpa": 0}


def livre(poly, folga=1.5, so_fora=False):
    c = poly.mean(0)
    if not CONTORNO.contains(shapely.Polygon(poly).buffer(-0.2)):
        RECUSAS["fora"] += 1
        return False
    if so_fora:
        return True
    for z in EVITAR:
        if dentro(c[0], c[1], z) or distancia(poly[:, 0], poly[:, 1], z, True)[0].min() < folga:
            RECUSAS["evitar"] += 1
            return False
    # casas encostadas às muralhas são normais; só se recusam as que o traçado atravessa
    if arv_esc.query(densificar(poly, 0.5, True))[0].min() < 0.25:
        RECUSAS["escarpa"] += 1
        return False
    return True


for f in F["edificios"]:
    if f["geometria"]["type"] != "Polygon" or f["id"][:8] in PROPRIOS:
        continue
    poly = geom(f)
    if area(poly) < 0:
        poly = poly[::-1]
    if not livre(poly, so_fora=True):
        continue
    h = f.get("altura") or ((f.get("pisos") or 2) * 3.0 + 0.6)
    # os quarteirões do OSM partem-se em casas; cada casa é recusada à parte junto às igrejas e portas
    partes = [pp for pp in partir(poly) if livre(pp, folga=0.4)]
    for pp in partes:
        n_med += casa(m, pp, h)[1]
    if partes:
        pegadas.append(poly)
n_osm = len(pegadas)
for c_ in json.loads((AQUI / "dados" / "casas.json").read_text()):
    poly = np.array(c_["planta"], float)
    if area(poly) < 0:
        poly = poly[::-1]
    if not livre(poly):
        continue
    for pp in partir(poly):
        n_med += casa(m, pp, 6.0)[1]
    pegadas.append(poly)
m.objeto("Casario", "vila", ["casa_parede", "casa_telhado"], GRUPOS["Vila"], uv=3.0)
MEDIDAS["vila"] = {"edificios": len(pegadas), "do_osm": n_osm, "do_lidar": len(pegadas) - n_osm, "medidos": n_med, "recusas": RECUSAS}

print("vegetação…")
pg_shp = shapely.union_all([shapely.Polygon(p).buffer(1.0) for p in pegadas] + [shapely.Polygon(z).buffer(1.5) for z in EVITAR])
n_arv = 0
# a cota do pé vem da malha do terreno já simplificada (com a base redonda, a simplificação tira mais vértices
# longe da praça e o MDT chegava a ficar 2–5 m acima da malha: árvores a flutuar)
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
_ter = bpy.data.objects["Terreno"]
_bvh = BVHTree.FromObject(_ter, bpy.context.evaluated_depsgraph_get())
_mw = _ter.matrix_world
_mwi = _mw.inverted()


def cota_malha(x, y, z_):
    hit = _bvh.ray_cast(_mwi @ Vector((x, y, z_ + 200.0)), (_mwi.to_3x3() @ Vector((0, 0, -1))).normalized())
    return float((_mw @ hit[0]).z) if hit[0] is not None else z_


for x, y, topo, _, r in LID["arvores"]:
    x, y = float(x), float(y)
    if not CONTORNO.contains(shapely.Point(x, y).buffer(2.0)):
        continue
    if arv_esc.query([x, y])[0] < 2.5 or pg_shp.contains(shapely.Point(x, y)):
        continue
    z = float(cota_t(x, y))
    h = float(topo) - ZREF - z
    if h < 2.5:
        continue
    z = min(z, cota_malha(x, y, z)) - 0.1
    r = float(np.clip(r, 1.0, 6.5))
    especie = "pinheiro" if (h > 13 and r < 3.6 and rng.random() < 0.6) else "folhosa"
    alt = float(np.clip(h * 0.62, 1.6, 2.2 * r))
    plantar(x, y, z, r, alt, especie, tronco=max(h - alt, 0.8))
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
MEDIDAS["vegetacao"] = {"arvores": n_arv}


def ancora(x, y, acima=2.0):
    """Ponto de uma placa no visor (x este, y cota, z sul)."""
    return [round(float(x), 1), round(float(cota_t(x, y)) + acima, 1), round(-float(y), 1)]


MEDIDAS.setdefault("ancoras", {})
# pontos das placas (x este, y cota, z sul): portas à frente da frontaria, monumentos por cima do telhado ou da
# torre, muralhas no ângulo saliente do baluarte de São Francisco (a sul), guaritas na guarita mais próxima dele
for pid_, (F0_, n_, sole_) in PONTOS_PORTA.items():
    MEDIDAS["ancoras"][pid_] = [round(float(F0_[0] + n_[0] * 2), 1), round(float(sole_ + ZREF + 11), 1), round(-float(F0_[1] + n_[1] * 2), 1)]
_pts = np.vstack([d["xy"] for d in ESC_DADOS])
_e = _pts[np.argmin(np.hypot(_pts[:, 0] - 40, _pts[:, 1] + 330))]
MEDIDAS["ancoras"]["escarpas"] = ancora(_e[0], _e[1], 12.0)
if GUARITAS:
    g_ = min(GUARITAS, key=lambda g: np.hypot(g[0][0] - _e[0], g[0][1] - _e[1]))[0]
    MEDIDAS["ancoras"]["guaritas"] = [round(float(g_[0]), 1), round(float(cota_t(*g_)) + 6, 1), round(-float(g_[1]), 1)]
for el_, (x_, y_, h_) in {"igreja_matriz": (146.0, -50.4, 22.0), "igreja_misericordia": (73.0, -137.0, 13.0),
                          "torre_relogio": (-56.2, 37.6, 27.0), "quartel_esquadras": (52.7, -170.6, 16.0),
                          "castelo": (-95.0, -3.0, 8.0)}.items():
    MEDIDAS["ancoras"][el_] = ancora(x_, y_, h_)
MEDIDAS["ancoras"]["vila"] = ancora(10, -60, 14.0)
(AQUI / "saida").mkdir(exist_ok=True)
(AQUI / "dados" / "medidas.json").write_text(json.dumps(_arred(MEDIDAS), ensure_ascii=False, indent=1))
print(json.dumps(_arred(MEDIDAS), ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
glb = AQUI / "saida" / "almeida-bruto.glb"
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
    final = SAIDA / "almeida.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "almeida.blend"))

# ---------------------------------------------------------------- capa e vistas de verificação

if "--vistas" in sys.argv:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--amostras=")), 24))
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "AgX"
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
        sc.render.filepath = str(destino or (pasta / f"{nome}.jpg"))
        bpy.ops.render.render(write_still=True)
        print("vista", nome)

    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 84
    VISTAS_EXTRA = list(globals().get("VISTAS_EXTRA", []))
    VISTAS_PERTO = list(globals().get("VISTAS_PERTO", []))
    SUF = next((a_.split("=")[1] for a_ in sys.argv if a_.startswith("--sufixo=")), "")
    for nm_, az_, el_, di_, xy_, zz_ in VISTAS_PERTO:
        VISTAS_EXTRA.append(((nm_ + SUF, az_, el_, di_), dict(lente=35, alvo_xy=xy_, alvo_z=zz_)))
    for pid_, (F0_, n_, sole_) in PONTOS_PORTA.items():  # uma vista de frente, um pouco de lado, para cada porta
        az_ = math.degrees(math.atan2(n_[0], n_[1])) + 25
        VISTAS_EXTRA.append(((f"p_{pid_}{SUF}", az_, 22, 40), dict(lente=35, alvo_xy=tuple(F0_), alvo_z=sole_ + 4)))
    vista("capa", 200, 33, 1500, lente=50, res=(1280, 800), destino=SAIDA / "capa.jpg", alvo_xy=(-10, -30), alvo_z=745)
    if "--capa" not in sys.argv:
        vista("topo", 180, 89.9, 1500, orto=960, res=(1000, 960))
        vista("aereo_sul", 190, 30, 820, lente=50, alvo_xy=(0, -120))
        vista("aereo_leste", 95, 30, 900, lente=50, alvo_xy=(0, 0))
        vista("poente", 270, 28, 1000)
        vista("norte", 10, 30, 950, alvo_xy=(0, 100))
        vista("baluarte", 200, 12, 90, lente=35, alvo_xy=(-150, -255))
        for v_ in VISTAS_EXTRA:
            vista(*v_[0], **v_[1])
