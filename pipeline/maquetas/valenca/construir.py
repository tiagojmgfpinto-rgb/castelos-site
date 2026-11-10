"""Constrói a maqueta 3D da Fortaleza de Valença (estado atual) em Blender e exporta-a para
public/maquetas/valenca/valenca.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow
    python pipeline/maquetas/valenca/construir.py            # GLB
    python pipeline/maquetas/valenca/construir.py --vistas   # GLB + capa + vistas de verificação

Dados: dados/fontes.json (recolher.py: OSM via Overture), dados/lidar.npz e dados/orto.jpg (lidar.py:
MDT, MDS, classes e cor da nuvem de pontos do LiDAR 2025 da DGT), dados/escarpas.json (tracar.py: as
escarpas de alvenaria traçadas no MDT) e src/data/maquetas/valenca.json, com a leitura das fontes.

Uma praça abaluartada é sobretudo terra revestida de pedra. O relevo da maqueta é o próprio MDT de 50 cm
(terraplenos, parapeitos de terra, fossos, estrada coberta e esplanada), com a cor do chão tirada da
ortoimagem da nuvem de pontos; as escarpas de alvenaria são panos próprios, com talude, cordão, parapeito,
cunhais de cantaria e guaritas nos ângulos salientes dos baluartes, como nas fotografias. As portas, as
igrejas e a estátua de São Teotónio levam o detalhe das fotografias; o casario fica em volumes simples.
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
from kit.arquitetura import Quadro, _vao_desloc, altura_telhado, barra, cupula, faixa_inclinada, guardas_corredor, guarita, oitavado, porta_revelim, porta_tunel, telhado_4aguas, telhado_generico  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "valenca.json").read_text())
ESC = json.loads((AQUI / "dados" / "escarpas.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "valenca"

LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([-40.0, 10.0])  # centro da grelha do LiDAR (igual ao lidar.py e ao tracar.py)
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
    """Limite da maqueta: a praça e as obras exteriores com a esplanada à volta (35 m para fora das
    linhas de muralha do OSM, alisado), dentro da área do LiDAR."""
    gs = []
    for f in F["infraestrutura"]:
        if f["classe"] != "city_wall":
            continue
        g = geom(f)
        gs.append(shapely.Polygon(g) if f["geometria"]["type"] == "Polygon" else shapely.LineString(g))
    for e in ESC:
        gs.append(shapely.LineString(e["linha"]))
    praca = next(shapely.Polygon(geom(f)) for f in F["infraestrutura"] if f["classe"] == "city_wall" and f["geometria"]["type"] == "Polygon")
    # só as escarpas da faixa das fortificações contam para o limite
    u = shapely.union_all([g.buffer(32) for g in gs if g.distance(praca) < 120])
    u = u.buffer(18).buffer(-26).buffer(8)
    caixa = shapely.box(LXS[0] + 2, LYS[0] + 2, LXS[-1] - 2, LYS[-1] - 2)
    u = u.intersection(caixa)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    u = shapely.Polygon(u.exterior).simplify(0.8)
    return u, praca


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
    como a calçada da Porta da Coroada), a mudança faz-se a 0,8 m do vértice alto, em vez de rampa."""
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


def escarpa(e, m, mq):
    """Um pano de escarpa: face em talude até ao cordão, parapeito de alvenaria por cima e o maciço por
    trás (escondido pelo terrapleno). Os valores medidos (topo, pé, aresta) vêm do tracar.py."""
    lin, topo, pe, a, fim = degraus(np.array(e["linha"], float), np.array(e["topo"]) - ZREF, np.array(e["pe"]) - ZREF,
                                    np.clip(np.array(e["aresta"]), -2.0, 1.0), np.array(e["fim"]))
    h = np.maximum(topo - pe, 0.5)
    # talude: o MDT de 50 cm alarga a faixa da escarpa; fica entre 1/10 e 1/4 da altura
    bt = np.clip((fim - a) / h, 0.10, 0.25)
    hm = float(np.median(h))
    alto = hm > 5.0
    zc = topo - (PARAPEITO if alto else 0.0)
    zf = pe - 1.2
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
        if alto and ang[i] > 45 and seg[i - 1] > 8 and seg[i] > 8 and h[i] > 4.5 \
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
    # a calçada à frente das portas com corredor (Coroada, Meio) fica à cota medida: não se rebaixa
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
    bz = amostra(Z, bxy[:, 0], bxy[:, 1]) - 0.05
    fundo = float(bz.min()) - 10.0
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
    m.prisma(poly, z0, z1, int(rng.choice(CORES_PAREDE)), topo=False)
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


def porta_coroada(Q, sole):
    """Porta da Coroada (fotografia): frontispício barroco de cantaria com dois pares de pilastras de
    almofadado, arco de volta perfeita com aduelas e fecho com escudo coroado, entablamento, ático com as
    armas reais entre volutas, remate curvo com cruz e pináculos em bola nos extremos."""
    c, s_ = 0, 1  # cantaria, sombra
    z = sole
    Q.bloco(-4.9, 4.9, z - 0.6, z + 6.7, -1.6, 0.30, c)  # pano de cantaria da frontaria
    for sg in (-1, 1):
        for a0, a1 in ((2.0, 2.8), (3.3, 4.1)):
            s0, s1 = sorted((sg * a0, sg * a1))
            Q.bloco(s0 - 0.08, s1 + 0.08, z - 0.4, z + 0.55, 0.28, 0.78, c)  # plinto
            Q.almofadas(s0, s1, z + 0.55, z + 6.15, 0.30, 0.72, c)
            Q.bloco(s0 - 0.06, s1 + 0.06, z + 6.15, z + 6.45, 0.28, 0.78, c)  # capitel
    # arco de volta perfeita com aduelas salientes e fecho com o escudo coroado
    w, zn = 1.65, z + 3.45
    Q.vao(2 * w, z, zn - z, 0.31, s_)
    Q.arco_sz(0.0, w, w + 0.6, zn, 0.30, 0.42, c, k=13, aduelas=True)
    for k, zz in enumerate(np.arange(z, zn - 0.2, 0.5)):  # ombreiras em silhares alternados
        e_ = 0.6 if k % 2 == 0 else 0.42
        for sg in (-1, 1):
            s0, s1 = sorted((sg * w, sg * (w + e_)))
            Q.bloco(s0, s1, zz + 0.02, zz + 0.48, 0.30, 0.40, c)
    Q.placa([(-0.32, zn + w - 0.15), (0.32, zn + w - 0.15), (0.42, zn + w + 0.75), (-0.42, zn + w + 0.75)], 0.30, 0.55, c)
    Q.escudo(0.0, zn + w + 0.05, 0.56, 0.62, 0.55, 0.66, c)
    # entablamento: arquitrave, friso e cornija saliente
    Q.bloco(-4.25, 4.25, z + 6.45, z + 6.85, 0.30, 0.82, c)
    Q.bloco(-4.5, 4.5, z + 6.85, z + 7.2, 0.30, 1.0, c)
    # ático com as armas reais
    Q.bloco(-1.85, 1.85, z + 7.2, z + 9.9, -0.4, 0.55, c)
    for sg in (-1, 1):
        Q.bloco(sg * 1.85 - 0.25, sg * 1.85 + 0.25, z + 7.2, z + 9.9, 0.5, 0.68, c)
    Q.escudo(0.0, z + 7.75, 1.25, 1.35, 0.55, 0.75, c)
    # remate curvo (segmento de arco) com cruz
    Q.arco_sz(0.0, 4.1, 4.45, z + 6.0, -0.4, 0.75, c, a0=math.radians(62), a1=math.radians(118), k=8)
    zt = z + 6.0 + 4.45
    Q.bloco(-0.09, 0.09, zt - 0.1, zt + 0.95, 0.1, 0.28, c)
    Q.bloco(-0.38, 0.38, zt + 0.48, zt + 0.66, 0.1, 0.28, c)
    # volutas: meio frontão partido de cada lado, a subir dos extremos para o ático, com o enrolamento
    for sg in (-1, 1):
        a_, b_ = sg * 4.3, sg * 2.05
        pts = [(a_, z + 7.2), (b_, z + 8.5), (b_, z + 8.95), (a_, z + 7.65)]
        Q.placa(pts, -0.2, 0.62, c)
        circ = [(b_ + 0.38 * math.cos(t), z + 8.95 + 0.38 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
        Q.placa(circ, -0.2, 0.7, c)
        circ2 = [(b_ + 0.18 * math.cos(t), z + 8.95 + 0.18 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 10, endpoint=False)]
        Q.placa(circ2, 0.7, 0.76, c)
        # pináculo em bola sobre plinto, nos extremos
        Q.bloco(sg * 4.1 - 0.32, sg * 4.1 + 0.32, z + 7.2, z + 7.95, 0.0, 0.62, c)
        Q.bloco(sg * 4.1 - 0.12, sg * 4.1 + 0.12, z + 7.95, z + 8.15, 0.2, 0.42, c)
        Q.esfera(sg * 4.1, z + 8.5, 0.31, 0.36, c)
    return 10.4


def porta_frontao(Q, sole):
    """Porta de praça com frontão (estimativa, pelo tipo das portas do século XVIII em Valença): pilastras
    toscanas, arco de volta perfeita com aduelas, entablamento, frontão triangular com o escudo real no
    tímpano e pináculos em bola."""
    c, s_ = 0, 1
    z = sole
    Q.bloco(-3.8, 3.8, z - 0.5, z + 5.6, -1.5, 0.28, c)
    w, zn = 1.5, z + 3.0
    Q.vao(2 * w, z, zn - z, 0.29, s_)
    Q.arco_sz(0.0, w, w + 0.5, zn, 0.28, 0.4, c, k=13, aduelas=True)
    Q.placa([(-0.25, zn + w - 0.1), (0.25, zn + w - 0.1), (0.32, zn + w + 0.55), (-0.32, zn + w + 0.55)], 0.28, 0.5, c)
    for sg in (-1, 1):
        s0, s1 = sorted((sg * 2.35, sg * 3.05))
        Q.bloco(s0 - 0.06, s1 + 0.06, z - 0.3, z + 0.4, 0.28, 0.68, c)
        Q.bloco(s0, s1, z + 0.4, z + 4.9, 0.28, 0.6, c)
        Q.bloco(s0 - 0.06, s1 + 0.06, z + 4.9, z + 5.15, 0.28, 0.68, c)
    Q.bloco(-3.3, 3.3, z + 5.15, z + 5.55, 0.28, 0.7, c)
    Q.bloco(-3.5, 3.5, z + 5.55, z + 5.85, 0.28, 0.88, c)
    zf = z + 5.85
    Q.placa([(-3.3, zf), (3.3, zf), (0.0, zf + 1.6)], -0.3, 0.55, c)
    for sg in (-1, 1):  # cornijas inclinadas do frontão
        Q.placa([(sg * 3.5, zf - 0.02), (sg * 3.3, zf - 0.02), (0.0, zf + 1.6), (0.0, zf + 1.85)], 0.3, 0.75, c)
        Q.bloco(sg * 3.3 - 0.25, sg * 3.3 + 0.25, zf, zf + 0.5, 0.1, 0.6, c)
        Q.esfera(sg * 3.3, zf + 0.82, 0.35, 0.3, c)
    Q.escudo(0.0, zf + 0.15, 0.62, 0.7, 0.55, 0.66, c, coroa=False)
    Q.bloco(-0.08, 0.08, zf + 1.85, zf + 2.6, 0.2, 0.36, c)
    Q.bloco(-0.3, 0.3, zf + 2.25, zf + 2.4, 0.2, 0.36, c)
    return 8.5


def porta_meio_armas(Q, sole):
    """Porta do Meio, face sul (fotografia): moldura de cantaria com pilastras de almofadado, arco de volta
    perfeita com aduelas, entablamento liso; por cima, no parapeito, as armas reais coroadas entre
    enrolamentos e, aos lados, duas bolas sobre plintos no alto da muralha."""
    c, s_ = 0, 1
    z = sole
    topo = getattr(Q, "topo", z + 6.0)
    w, zn = 1.45, z + 2.7
    Q.bloco(-2.9, 2.9, z - 0.4, z + 4.9, -1.5, 0.22, c)  # pano de cantaria da moldura
    Q.vao(2 * w, z, zn - z, 0.23, s_)
    Q.arco_sz(0.0, w, w + 0.45, zn, 0.22, 0.36, c, k=13, aduelas=True)
    Q.placa([(-0.24, zn + w - 0.05), (0.24, zn + w - 0.05), (0.3, zn + w + 0.5), (-0.3, zn + w + 0.5)], 0.22, 0.44, c)
    for sg in (-1, 1):
        s0, s1 = sorted((sg * 1.95, sg * 2.75))
        Q.almofadas(s0, s1, z, z + 4.2, 0.22, 0.52, c, fiada=0.5)
        Q.bloco(s0 - 0.08, s1 + 0.08, z + 4.2, z + 4.4, 0.22, 0.58, c)  # capitel
        s2 = sorted((sg * (w + 0.45), sg * 1.95))
        Q.bloco(s2[0], s2[1], z, zn, 0.22, 0.34, c)  # pés-direitos
    Q.bloco(-2.95, 2.95, z + 4.4, z + 4.95, 0.22, 0.6, c)  # entablamento
    Q.bloco(-3.1, 3.1, z + 4.95, z + 5.15, 0.22, 0.72, c)
    zt = max(topo, z + 5.6)
    # armas reais no parapeito, entre enrolamentos
    Q.bloco(-1.2, 1.2, z + 5.15, zt + 0.2, -0.3, 0.3, c)
    Q.escudo(0.0, zt - 0.1, 1.0, 1.15, 0.3, 0.45, c, coroa=True)
    for sg in (-1, 1):
        Q.arco_sz(sg * 1.25, 0.15, 0.4, zt + 0.1, -0.1, 0.35, c, a0=0, a1=2 * math.pi, k=10)
        Q.placa([(sg * 0.9, zt - 0.2), (sg * 1.6, zt - 0.2), (sg * 1.6, zt + 0.05), (sg * 0.9, zt + 0.5)], -0.1, 0.35, c)
        # bolas sobre plintos no alto da muralha
        Q.bloco(sg * 3.0 - 0.35, sg * 3.0 + 0.35, zt, zt + 0.7, -0.5, 0.2, c)
        Q.bloco(sg * 3.0 - 0.42, sg * 3.0 + 0.42, zt + 0.7, zt + 0.82, -0.57, 0.27, c)
        Q.esfera(sg * 3.0, zt + 1.22, -0.15, 0.4, c)
    return zt + 1.6 - z


PORTAS = {
    # ponto (OSM), normal aproximada da face (para fora), estilo e se leva ponte
    # a Porta da Coroada abre a meio do pano sul, no topo da calçada entre muros que vai ao revelim
    "porta_coroada": dict(ponto=(-3.0, -325.6), normal=(0.0, -1.0), estilo=porta_coroada, elemento="porta_coroada",
                          fixo=True, corredor=True, soleira=73.4),  # a calçada tem 3 a 4 m de largura, à cota 73,4
    "porta_revelim": dict(ponto=(-1.3, -379.0), normal=(0.66, 0.75), estilo=porta_revelim, elemento="porta_revelim"),
    "porta_revelim_fora": dict(ponto=(-12.8, -390.9), normal=(-0.66, -0.75), estilo=porta_tunel, elemento="porta_revelim"),
    "porta_meio": dict(ponto=(78.0, -62.9), normal=(0.38, 0.92), estilo=porta_frontao, elemento="porta_meio"),
    # face sul da Porta do Meio (fotografia): sai para a calçada estreita (3 a 8 m) que atravessa o fosso, à cota 71,8
    "porta_meio_sul": dict(ponto=(73.5, -77.7), normal=(-0.13, -0.99), estilo=porta_meio_armas, elemento="porta_meio",
                           fixo=True, corredor=True, soleira=71.8),
    "portas_sol": dict(ponto=(152.1, -33.5), normal=(0.87, -0.5), estilo=porta_frontao, elemento="portas_sol"),
    "portas_sol_dentro": dict(ponto=(135.7, -24.8), normal=(-0.86, 0.5), estilo=porta_tunel, elemento="portas_sol"),
    "porta_gaviarra": dict(ponto=(71.5, 317.8), normal=(0.84, 0.55), estilo=porta_frontao, elemento="porta_gaviarra"),
    "porta_gaviarra_dentro": dict(ponto=(51.2, 307.8), normal=(-0.71, -0.71), estilo=porta_tunel, elemento="porta_gaviarra"),
}
PONTOS_PORTA = {}


def ponte(m, F0, n, t, sole, largura=4.4):
    """Ponte de cantaria sobre o fosso, da soleira para fora até onde o terreno volta à cota da soleira
    (fotografia da Porta da Coroada: tabuleiro lajeado e guardas baixas de granito)."""
    L = 0.0
    for dd in np.arange(1.0, 40.0, 0.5):
        L = dd
        if dd > 3 and cota_t(*(F0 + n * dd)) > sole - 0.4:
            break
    if L < 4:
        return 0.0
    Qp = Quadro(m, F0, n)
    Qp.bloco(-largura / 2, largura / 2, sole - 0.75, sole, 0.0, L, 0)
    for sg in (-1, 1):
        s0, s1 = sorted((sg * largura / 2, sg * (largura / 2 - 0.45)))
        Qp.bloco(s0, s1, sole, sole + 1.0, 0.0, L, 0)
        Qp.bloco(s0 - 0.04, s1 + 0.04, sole + 1.0, sole + 1.12, 0.0, L, 0)
    for dd in np.arange(4.0, L - 1.0, 6.0):  # pegões até ao fundo do fosso
        zb = float(cota_t(*(F0 + n * dd))) - 0.8
        Qp.bloco(-largura / 2 + 0.2, largura / 2 - 0.2, zb, sole - 0.75, dd - 0.8, dd + 0.8, 0)
    return L


def portas():
    pts = np.vstack([d["xy"] for d in ESC_DADOS])
    cat = lambda k: np.concatenate([np.broadcast_to(d[k], (len(d["xy"]),) + np.shape(d[k])[1:]) for d in ESC_DADOS])
    NR, A_, BT, TP, PE, ZC = cat("nr"), cat("a"), cat("bt"), cat("topo"), cat("pe"), cat("zc")
    for pid, d_ in PORTAS.items():
        p0, n0 = P(0, 0) * 0 + np.array(d_["ponto"]), unit(np.array(d_["normal"], float))
        dist = np.hypot(*(pts - p0).T)
        ok = (dist < 7.0) & (NR @ n0 > (0.97 if d_.get("fixo") else 0.6))
        if not ok.any():
            print("  porta sem escarpa:", pid)
            continue
        k = int(np.nonzero(ok)[0][np.argmin(dist[ok])])
        n = n0 if d_.get("fixo") else NR[k]
        q = pts[k] + np.array([-n[1], n[0]]) * float((p0 - pts[k]) @ np.array([-n[1], n[0]])) if d_.get("fixo") else pts[k]
        sole = float(np.median([cota_t(*(q + n * (A_[k] + BT[k] * (ZC[k] - PE[k]) + dd))) for dd in (1.5, 2.5, 3.5)]))
        sole = min(max(sole, PE[k]), ZC[k] - 2.5)
        if "soleira" in d_:
            sole = d_["soleira"] - ZREF
        F0 = q + n * (A_[k] + BT[k] * (ZC[k] - sole) + 0.02)
        m = Malha()
        Q = Quadro(m, F0, n)
        Q.topo = float(TP[k])
        alt = d_["estilo"](Q, sole)
        L = ponte(m, F0, n, Q.t, sole) if d_.get("ponte") else 0.0
        if d_.get("corredor"):
            guardas_corredor(m, F0, n, pts, NR, A_, TP)
        m.objeto(pid.title(), d_["elemento"], ["cantaria", "sombra", "madeira_verde"], GRUPOS["Portas"], uv=2.5)
        PONTOS_PORTA[pid] = (F0, n, sole)
        tt = np.array([-n[1], n[0]])  # passagem livre de casas e árvores à frente da porta
        EVITAR.append(np.array([F0 + tt * 3 + n * 0.3, F0 + tt * 3 + n * 9, F0 - tt * 3 + n * 9, F0 - tt * 3 + n * 0.3]))
        MEDIDAS[pid] = {"soleira": sole + ZREF, "topo_muralha": float(TP[k] + ZREF), "altura_frontaria": alt, "ponte": L}
        print(f"  {pid}: soleira {sole:.1f}, topo da muralha {TP[k]:.1f}, ponte {L:.1f} m")


# ---------------------------------------------------------------- igrejas e estátua
# Medidas tiradas do MDS de 50 cm (cumeeiras, torres, pegadas) e pormenores das fotografias cedidas.
# Cada igreja tem um referencial Quadro na fachada principal: s ao longo da fachada, d para fora; o
# corpo da igreja fica em d < 0.

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
    if estilo == "agulha_pinaculos":  # Santa Maria dos Anjos: pináculos nos cantos e agulha oitavada curta
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cc = c + (u * a + w * b) * (h - 0.1)
            pinaculo(Quadro(m, cc, u), 0, zt, 0, esc=0.95, tipo="bola")
        oitavado(m, c, lado * 0.32, lado * 0.26, zt, zt + 0.5, CAN)
        oitavado(m, c, lado * 0.24, 0.0, zt + 0.5, zt + 2.6, CAN)
        m.esfera((c[0], c[1], zt + 2.75), 0.18, CAN)
        cruz(Qs, 0, zt + 2.9, 0, alt=1.2, mat=FER, esp=0.06)
        return zt + 4.1
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


def igreja_anjos(m):
    """Igreja de Santa Maria dos Anjos (românica, granito): fachada a poente com portal de arquivoltas,
    torre a sudoeste caiada com cunhais, cachorrada sob as cornijas, capela-mor mais baixa a nascente."""
    g = 64.9 - ZREF
    Q = Quadro(m, (17.2, 263.0), (-1.0, 0.0))
    corpo(Q, 17.8, 8.8, g, 72.75 - ZREF, inc=30, mw=CAN, cachorros=True)
    Qc = Quadro(m, (35.0, 263.0), (-1.0, 0.0))
    corpo(Qc, 6.2, 6.6, g, 70.5 - ZREF, inc=30, mw=CAN, cachorros=True, cornija_frente=False)
    portal_arquivoltas(Q, 0.8, g, 1.7, 2.5, n=3)
    janela(Q, 2.4, g + 5.6, 0.5, 0.7, arco=False, mat_vidro=SOM, grade=True)  # fresta com grade (fotografia)
    cruz(Q, 0, 72.75 - ZREF + 4.4 * math.tan(_ang(30)) + 0.35, 0.0, alt=1.3, mat=CAN, esp=0.16)
    # frestas na nave (lado sul)
    Qs = Quadro(m, (26.0, 258.6), (0.0, -1.0))
    for s_ in (-4.0, 0.5, 5.0):
        janela(Qs, s_, g + 4.2, 0.35, 1.3, arco=True, mat_vidro=SOM, moldura=0.14)
    torre(m, (18.2, 256.3), (-1.0, 0.0), 4.4, g, g + 11.3, g + 15.3, "agulha_pinaculos", faces_janela=(0, 2), zjanela=g + 8.0)
    return [[15.6, 253.7], [41.6, 253.7], [41.6, 268.0], [15.6, 268.0]]


def igreja_misericordia(m):
    """Igreja da Misericórdia (caiada, cunhais de granito): fachada a poente com rosácea e porta de arco,
    torre a noroeste com cachorrada e agulha de pedra, portal lateral a sul com o relevo da Misericórdia
    num nicho, sacristia mais baixa a nascente (fotografias)."""
    g = 63.9 - ZREF
    ze = 72.6 - ZREF
    Q = Quadro(m, (3.0, 272.0), (-1.0, 0.0))
    corpo(Q, 14.0, 6.8, g, ze, inc=30, cachorros=True)
    Qa = Quadro(m, (17.0, 272.0), (-1.0, 0.0))
    corpo(Qa, 6.5, 6.0, g, 70.0 - ZREF, inc=30, cornija_frente=False)
    porta_igreja(Q, 0.0, g + 0.5, 1.6, 2.5, arco=True)
    Q.bloco(-1.5, 1.5, g, g + 0.5, 0.0, 1.2, CAN)  # degraus
    Q.bloco(-1.2, 1.2, g + 0.25, g + 0.5, 1.2, 1.6, CAN)
    rosacea(Q, 0.0, g + 6.7, 0.85)
    cruz(Q, 0, ze + 3.4 * math.tan(_ang(30)) + 0.35, 0.0, alt=1.2, mat=CAN, esp=0.14)
    # portal lateral (sul): retábulo de granito com relevo num nicho, pináculos e cruz
    Ql = Quadro(m, (10.0, 268.6), (0.0, -1.0))
    porta_igreja(Ql, 0.0, g + 0.3, 1.5, 2.9, arco=False, moldura=0.28)
    for sg in (-1, 1):
        Ql.bloco(sg * 1.25 - 0.22, sg * 1.25 + 0.22, g + 0.3, g + 3.8, 0.0, 0.22, CAN)  # pilastras
    Ql.bloco(-1.6, 1.6, g + 3.8, g + 4.15, 0.0, 0.3, CAN)  # entablamento
    Ql.bloco(-0.85, 0.85, g + 4.15, g + 6.3, 0.0, 0.16, CAN)  # moldura do nicho
    _vao_desloc(Ql, 0.0, 1.05, g + 4.35, 1.1, 0.165, SOM)
    Ql.placa([(-0.4, g + 4.4), (0.4, g + 4.4), (0.3, g + 5.3), (0.0, g + 5.6), (-0.3, g + 5.3)], 0.12, 0.3, LIS)  # relevo
    Ql.esfera(0.0, g + 5.75, 0.24, 0.15, LIS)
    Ql.arco_sz(0.0, 0.85, 1.1, g + 6.3, 0.0, 0.2, CAN, k=12)  # frontão curvo
    Ql.bloco(-1.1, 1.1, g + 6.15, g + 6.32, 0.0, 0.24, CAN)
    for sg in (-1, 1):
        pinaculo(Ql, sg * 1.35, g + 4.15, 0.16, esc=0.8, tipo="agulha")
    cruz(Ql, 0.0, g + 7.4, 0.1, alt=1.1, mat=CAN, esp=0.13)
    janela(Ql, -4.4, g + 2.4, 0.7, 1.4, arco=False, mat_vidro=VID, grade=True)
    torre(m, (4.8, 277.2), (-1.0, 0.0), 3.6, g, g + 9.6, g + 12.6, "agulha_pedra", faces_janela=(0,), zjanela=g + 6.4)
    return [[1.8, 268.0], [24.0, 268.0], [24.0, 279.6], [1.8, 279.6]]


def igreja_estevao(m):
    """Igreja Matriz de Santo Estêvão (fotografias da fachada e da Rua de São Francisco): fachada a poente
    caiada, com quatro pilastras de granito, entablamento, frontão triangular sobre o tramo central com
    cruz, pináculos de chama, portal de verga reta com frontão, três janelas de arco por cima, janelas
    nos tramos laterais e escadaria; torre a sul caiada com cunhais, óculo, sineira, pináculos e coroa de
    ferro; nave alta com corpos laterais mais baixos (pegada do OSM, cumeeiras do MDS)."""
    g = 69.3 - ZREF
    ze = 80.3 - ZREF
    W = 13.4
    F = 0.6  # espessura da fachada, que se ergue à frente da empena da nave
    Q = Quadro(m, (83.0, 155.2), (-1.0, 0.0))
    corpo(Q, 21.4, W, g, ze, inc=31, cornija_frente=False, d_ini=-F)
    Qc = Quadro(m, (105.0, 155.0), (-1.0, 0.0))
    corpo(Qc, 7.0, 9.6, g, 79.6 - ZREF, inc=30, cornija_frente=False)
    Qn = Quadro(m, (88.0, 165.5), (-1.0, 0.0))  # corpos a norte (sacristia e capelas)
    corpo(Qn, 21.0, 6.0, g, 75.2 - ZREF, inc=26)
    Qs_ = Quadro(m, (97.5, 144.6), (-1.0, 0.0))  # corpo a sul, a nascente da torre
    corpo(Qs_, 9.0, 5.6, g, 75.0 - ZREF, inc=26)
    # fachada: pano caiado com o frontão sobre o tramo central (metade da largura)
    wp = W / 4 + 0.35
    hp = wp * 0.62
    Q.placa([(-W / 2, g - 1.5), (W / 2, g - 1.5), (W / 2, ze), (wp, ze), (0.0, ze + hp), (-wp, ze)], -F, 0.0, CAL)
    Q.bloco(-W / 2 - 0.08, W / 2 + 0.08, g - 1.5, g + 0.55, -F, 0.12, CAN)  # soco
    for s_ in (-W / 2 + 0.4, -W / 4, W / 4, W / 2 - 0.4):  # pilastras gigantes
        Q.bloco(s_ - 0.42, s_ + 0.42, g + 0.55, ze - 1.0, 0.0, 0.22, CAN)
        Q.bloco(s_ - 0.52, s_ + 0.52, g + 0.2, g + 0.75, 0.0, 0.32, CAN)  # base
        Q.bloco(s_ - 0.52, s_ + 0.52, ze - 1.25, ze - 1.0, 0.0, 0.3, CAN)  # capitel
        Q.bloco(s_ - 0.47, s_ + 0.47, ze - 1.4, ze - 1.25, 0.0, 0.26, CAN)
    Q.bloco(-W / 2 - 0.1, W / 2 + 0.1, ze - 1.0, ze - 0.35, 0.0, 0.26, CAN)  # arquitrave e friso
    Q.bloco(-W / 2 - 0.3, W / 2 + 0.3, ze - 0.35, ze, 0.0, 0.48, CAN)  # cornija
    faixa_inclinada(Q, [(-wp - 0.3, ze - 0.05), (0, ze + hp + 0.05), (wp + 0.3, ze - 0.05)], 0.4, -0.05, 0.48, CAN)
    janela(Q, 0.0, ze + 0.45, 0.6, 0.6, arco=False, mat_vidro=SOM, moldura=0.14)  # postigo do tímpano
    cruz(Q, 0.0, ze + hp + 0.4, 0.1, alt=1.5, mat=CAN, esp=0.17)
    Q.bloco(-0.3, 0.3, ze + hp + 0.1, ze + hp + 0.45, -0.1, 0.3, CAN)
    for s_ in (-W / 2 + 0.4, -wp, wp, W / 2 - 0.4):  # pináculos de chama
        pinaculo(Q, s_, ze, 0.2, esc=1.15, tipo="agulha")
    # tramo central: portal com frontão e três janelas de arco
    porta_igreja(Q, 0.0, g + 0.7, 1.8, 3.3, arco=False, moldura=0.32)
    Q.bloco(-1.5, 1.5, g + 4.32, g + 4.55, 0.0, 0.38, CAN)
    faixa_inclinada(Q, [(-1.6, g + 4.55), (0.0, g + 5.3), (1.6, g + 4.55)], 0.2, 0.0, 0.38, CAN)
    janela(Q, 0.0, g + 6.0, 1.15, 2.3, arco=True, mat_vidro=VID, moldura=0.26, grade=True)
    for sg in (-1, 1):
        janela(Q, sg * 1.95, g + 6.3, 0.75, 1.65, arco=True, mat_vidro=VID, moldura=0.2, grade=True)
        # tramos laterais: janela baixa e janela quadrada alta, de molduras de granito
        janela(Q, sg * 4.95, g + 2.3, 0.75, 1.1, arco=False, mat_vidro=VID, moldura=0.2, grade=True)
        janela(Q, sg * 4.95, g + 6.6, 0.85, 0.95, arco=False, mat_vidro=VID, moldura=0.2, grade=True)
    # escadaria e adro
    for k_ in range(5):
        Q.bloco(-W / 2 - 0.4, W / 2 + 0.4, g - 1.0, g - 0.15 + 0.17 * (k_ + 1), 0.0, 3.0 - 0.38 * k_, CAN)
    # janelas altas da nave (sul), acima dos corpos laterais
    Qs = Quadro(m, (94.5, 148.5), (0.0, -1.0))
    for s_ in (-9.0, -4.0, 5.5):
        janela(Qs, s_, g + 7.4, 0.9, 1.6, arco=True, mat_vidro=VID)
    torre(m, (93.2, 145.5), (-1.0, 0.0), 4.4, g, g + 12.2, g + 16.0, "coroa_ferro", faces_janela=(2,), zjanela=g + 6.0,
          oculo=(0, g + 10.6))
    return [[81.0, 141.0], [113.0, 141.0], [113.0, 172.0], [81.0, 172.0]]


def capela_bom_jesus(m):
    """Capela do Bom Jesus (fotografia): caiada, com largas pilastras de granito nos cunhais, cornijas e
    empena de granito com bolas sobre plintos nos cantos e cruz lavrada no vértice; portal de arco com
    pilastras, entablamento e frontão interrompido de enrolamentos, que sobe numa janela emoldurada com
    remate curvo; janelas de granito nas fachadas lateral e principal; capela-mor mais baixa a noroeste."""
    g = 73.3 - ZREF
    ze = 81.0 - ZREF
    W = 8.5
    u = unit(np.array([0.91, -0.41]))
    F = np.array([16.4, -207.6])
    Q = Quadro(m, F, u)
    zr = corpo(Q, 13.0, W, g, ze, inc=30, mr=LOU, cornija_frente=False)
    Qc = Quadro(m, F - u * 13.0, u)
    corpo(Qc, 5.5, 6.5, g, 79.1 - ZREF, inc=30, mr=LOU, cornija_frente=False)
    tg = math.tan(_ang(30))
    # pilastras largas nos cunhais e cornija da empena
    for sg in (-1, 1):
        ss = sorted((sg * (W / 2 - 0.85), sg * (W / 2 + 0.12)))
        Q.bloco(ss[0], ss[1], g, ze - 0.3, 0.0, 0.25, CAN)
        Q.bloco(ss[0] - 0.05, ss[1] + 0.05, ze - 0.55, ze - 0.3, 0.0, 0.32, CAN)
    Q.bloco(-W / 2 - 0.25, W / 2 + 0.25, ze - 0.3, ze + 0.1, 0.0, 0.45, CAN)
    faixa_inclinada(Q, [(-W / 2 - 0.25, ze + 0.0), (0, zr + 0.15), (W / 2 + 0.25, ze + 0.0)], 0.45, -0.15, 0.45, CAN)
    for sg in (-1, 1):  # bolas sobre plintos nos cantos da empena
        Q.bloco(sg * (W / 2 - 0.15) - 0.32, sg * (W / 2 - 0.15) + 0.32, ze + 0.1, ze + 0.75, 0.0, 0.5, CAN)
        Q.esfera(sg * (W / 2 - 0.15), ze + 1.12, 0.25, 0.36, CAN)
    # cruz lavrada sobre plinto com enrolamentos, no vértice
    zv = zr + 0.55
    Q.bloco(-0.45, 0.45, zv, zv + 0.55, -0.1, 0.4, CAN)
    for sg in (-1, 1):
        Q.arco_sz(sg * 0.45, 0.08, 0.25, zv + 0.2, 0.0, 0.3, CAN, a0=0, a1=2 * math.pi, k=8)
    Q.esfera(0.0, zv + 0.75, 0.15, 0.22, CAN)
    cruz(Q, 0.0, zv + 0.9, 0.15, alt=1.5, mat=CAN, esp=0.16)
    for k_ in range(4):  # remates da cruz em flor
        a_ = math.pi / 2 * k_
        Q.esfera(0.0 + 0.32 * math.cos(a_) * (k_ % 2 == 0), zv + 0.9 + 0.93 + 0.32 * math.sin(a_), 0.15, 0.07, CAN)
    # frontispício: portal de arco com pilastras, entablamento, frontão de enrolamentos e janela
    porta_igreja(Q, 0.0, g + 0.35, 1.6, 2.7, arco=True, moldura=0.3)
    for sg in (-1, 1):
        ss = sorted((sg * 1.25, sg * 1.7))
        Q.bloco(ss[0], ss[1], g + 0.35, g + 3.95, 0.0, 0.3, CAN)  # pilastras
        Q.bloco(ss[0] - 0.06, ss[1] + 0.06, g, g + 0.6, 0.0, 0.38, CAN)  # plintos
    Q.bloco(-1.9, 1.9, g + 3.95, g + 4.45, 0.0, 0.42, CAN)  # entablamento
    for sg in (-1, 1):  # frontão interrompido de enrolamentos
        Q.placa([(sg * 1.9, g + 4.45), (sg * 0.9, g + 4.45), (sg * 0.9, g + 4.9), (sg * 1.9, g + 4.75)], 0.0, 0.4, CAN)
        Q.arco_sz(sg * 0.95, 0.1, 0.3, g + 4.95, 0.0, 0.42, CAN, a0=0, a1=2 * math.pi, k=10)
    janela(Q, 0.0, g + 5.2, 0.85, 1.35, arco=False, mat_vidro=VID, moldura=0.24, grade=True)
    for sg in (-1, 1):  # orelhas e enrolamentos laterais da janela
        Q.placa([(sg * 0.67, g + 5.2), (sg * 1.05, g + 5.35), (sg * 1.0, g + 6.3), (sg * 0.67, g + 6.55)], 0.0, 0.2, CAN)
    Q.arco_sz(0.0, 0.55, 0.78, g + 6.75, 0.0, 0.3, CAN, k=9)  # remate curvo
    Q.bloco(-0.8, 0.8, g + 6.62, g + 6.8, 0.0, 0.32, CAN)
    for sg in (-1, 1):
        janela(Q, sg * 2.65, g + 1.6, 0.6, 1.3, arco=False, mat_vidro=VID, moldura=0.18, grade=True)
    Q.bloco(-2.6, 2.6, g - 0.4, g + 0.35, 0.0, 1.6, CAN)  # adro
    # janelas altas da fachada lateral (sudoeste e nordeste)
    for sg in (-1, 1):
        Ql = Quadro(m, F - u * 6.5 + np.array([-u[1], u[0]]) * sg * W / 2 * -1 * 0 + np.array([u[1], -u[0]]) * sg * W / 2,
                    np.array([u[1], -u[0]]) * sg)
        for s_ in (-3.0, 2.5):
            janela(Ql, s_, g + 3.2, 0.8, 1.9, arco=False, mat_vidro=VID, moldura=0.22, grade=True)
    return [[float(v) for v in F + u * a_ + np.array([-u[1], u[0]]) * b_] for a_, b_ in ((1.8, -5.6), (1.8, 5.6), (-19.6, 5.6), (-19.6, -5.6))]


def capela_encontro(m):
    """Capela do Senhor do Encontro: pequena, caiada, cunhais e cornijas de granito, porta e postigo,
    pináculos e cruz na empena."""
    g = 74.5 - ZREF
    ze = 78.9 - ZREF
    u = unit(np.array([0.92, -0.40]))
    F = np.array([27.5, -284.5]) - u * 5.5
    Q = Quadro(m, F, -u)
    zr = corpo(Q, 11.0, 6.0, g, ze, inc=30)
    porta_igreja(Q, 0.0, g + 0.3, 1.3, 2.3, arco=True, moldura=0.25)
    janela(Q, 0.0, g + 3.6, 0.5, 0.5, arco=False, mat_vidro=SOM, moldura=0.15)
    for sg in (-1, 1):
        pinaculo(Q, sg * 3.0, ze + 0.05, 0.1, esc=0.8, tipo="bola")
    cruz(Q, 0.0, zr + 0.4, 0.0, alt=1.0, mat=CAN, esp=0.13)
    Q.bloco(-1.3, 1.3, g - 0.4, g + 0.3, 0.0, 1.0, CAN)
    return [[float(v) for v in F + (-u) * a_ + np.array([u[1], -u[0]]) * b_] for a_, b_ in ((1.2, -3.6), (1.2, 3.6), (-11.6, 3.6), (-11.6, -3.6))]


def estatua_teotonio(m):
    """Estátua de São Teotónio (bronze sobre pedestal de granito): figura de pé, de hábito e capa, com o
    báculo na mão esquerda."""
    c = np.array([31.9, -215.0])
    g = float(cota_t(*c))
    u = np.array([0.0, -1.0])
    m.caixa(c, u, 3.2, 3.2, g - 0.6, g + 0.35, CAN)
    m.caixa(c, u, 2.6, 2.6, g + 0.35, g + 0.7, CAN)
    m.caixa(c, u, 1.7, 1.7, g + 0.7, g + 0.95, CAN)
    m.caixa(c, u, 1.4, 1.4, g + 0.95, g + 3.0, CAN)
    m.caixa(c, u, 1.7, 1.7, g + 3.0, g + 3.25, CAN)
    m.caixa(c, u, 1.5, 1.5, g + 3.25, g + 3.4, CAN)
    z = g + 3.4
    m.cilindro(c, 0.42, z, z + 1.0, BRO, k=10, r1=0.33)       # hábito
    m.cilindro(c, 0.33, z + 1.0, z + 1.65, BRO, k=10, r1=0.3)
    m.cilindro(c, 0.38, z + 1.15, z + 1.7, BRO, k=10, r1=0.26)  # capa
    m.esfera((c[0], c[1], z + 1.92), 0.15, BRO, k=10, achata=1.15)
    m.cilindro(c, 0.16, z + 2.0, z + 2.3, BRO, k=8, r1=0.06)    # mitra
    for sg in (-1, 1):  # braços
        cc = c + np.array([sg * 0.36, -0.08])
        m.cilindro(cc, 0.09, z + 1.05, z + 1.6, BRO, k=6)
    b = c + np.array([-0.48, -0.1])
    m.cilindro(b, 0.03, z, z + 2.5, BRO, k=6)                  # báculo
    m.esfera((b[0] + 0.08, b[1], z + 2.55), 0.1, BRO, k=8)
    return [[c[0] - 1.8, c[1] - 1.8], [c[0] + 1.8, c[1] - 1.8], [c[0] + 1.8, c[1] + 1.8], [c[0] - 1.8, c[1] + 1.8]]


IGREJAS = [
    ("Igreja_Santa_Maria_dos_Anjos", "igreja_anjos", igreja_anjos),
    ("Igreja_Misericordia", "igreja_misericordia", igreja_misericordia),
    ("Igreja_Santo_Estevao", "igreja_estevao", igreja_estevao),
    ("Capela_Bom_Jesus", "capela_bom_jesus", capela_bom_jesus),
    ("Capela_Senhor_do_Encontro", "capela_encontro", capela_encontro),
]


def igrejas():
    for nome, el, fn in IGREJAS:
        m = Malha()
        planta = fn(m)
        m.objeto(nome, el, MAT_IGR, GRUPOS["Igrejas"], uv=2.5)
        EVITAR.append(np.array(planta, float))
        MEDIDAS[el] = {"faces": len(m.f)}
        print(f"  {nome}: {len(m.f)} faces")
    m = Malha()
    planta = estatua_teotonio(m)
    m.objeto("Estatua_Sao_Teotonio", "estatua", MAT_IGR, GRUPOS["Estatua"], uv=2.0)
    EVITAR.append(np.array(planta, float))


# ================================================================ cena

bpy.ops.wm.read_factory_settings(use_empty=True)
RAIZ_OB = bpy.data.objects.new("Fortaleza_Valenca", None)
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
material("ocre", (0.86, 0.72, 0.42), 0.85)
material("rosa", (0.88, 0.76, 0.70), 0.85)
TEX_TELHA, ALT_TELHA = textura_telha()
material("telha", (1, 1, 1), 0.8, imagem("telha", TEX_TELHA * np.array([0.78, 0.40, 0.25])),
         normal=imagem("telha_normal", normais(ALT_TELHA, 2.0)), forca=0.6)
material("telha_velha", (1, 1, 1), 0.85, imagem("telha_v", TEX_TELHA * np.array([0.56, 0.40, 0.33])),
         normal=imagem("telha_v_normal", normais(ALT_TELHA, 2.0)), forca=0.6)
material("lousa", (0.42, 0.42, 0.43), 0.8)
material("madeira_verde", (0.16, 0.33, 0.26), 0.7)
material("madeira", (0.40, 0.26, 0.16), 0.8)
material("ferro", (0.16, 0.17, 0.18), 0.5)
material("bronze", (0.24, 0.30, 0.25), 0.5)
material("vidro", (0.20, 0.24, 0.32), 0.25)
material("base", (0.33, 0.26, 0.20), 0.95)
material("terreno", (1, 1, 1), 0.95)
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


for e in ESC:
    LINHAS_ESC.append(escarpa(cortar(e), m, mq))
m.objeto("Escarpas", "escarpas", ["alvenaria", "cantaria", "relva"], GRUPOS["Muralhas"], uv=4.0)
mq.objeto("Cunhais", "escarpas", ["cantaria"], GRUPOS["Muralhas"], uv=3.0)
alturas = np.concatenate([d["h"] for d in ESC_DADOS])
MEDIDAS["escarpas"] = {"panos": len(ESC), "comprimento": float(sum(e["comprimento"] for e in ESC)),
                       "altura_p50": float(np.median(alturas)), "altura_p90": float(np.percentile(alturas, 90)),
                       "altura_max": float(alturas.max())}

print("guaritas…", len(GUARITAS))
m = Malha()
for c, bis, zc, topo in GUARITAS:
    guarita(m, c, bis, zc, topo)
m.objeto("Guaritas", "guaritas", ["cantaria_lisa", "cantaria", "sombra"], GRUPOS["Guaritas"], uv=2.0)
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
# 6951d651: o OSM desenha como "edifício" o patamar lajeado por cima da Porta da Coroada (ortoimagem); partido
# em casas, dava telhados deitados no terrapleno e uma casa alta encostada à porta (auditoria de novembro de 2026)
PROPRIOS.add("6951d651")
EVITAR = []       # plantas onde não vão casas nem árvores
if "portas" in globals():
    print("portas…")
    portas()
if "igrejas" in globals():
    print("igrejas e estátua…")
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
m.objeto("Casario", "vila", ["cal", "telha", "ocre", "cantaria", "rosa"], GRUPOS["Vila"], uv=3.0)
MEDIDAS["vila"] = {"edificios": len(pegadas), "do_osm": n_osm, "do_lidar": len(pegadas) - n_osm, "medidos": n_med, "recusas": RECUSAS}

print("vegetação…")
pg_shp = shapely.union_all([shapely.Polygon(p).buffer(1.0) for p in pegadas] + [shapely.Polygon(z).buffer(1.5) for z in EVITAR])
n_arv = 0
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
# pontos das placas (x este, y cota, z sul): portas à frente da frontaria, igrejas no alto da torre ou da
# empena, muralhas num baluarte da Coroada, guaritas na guarita mais próxima dele, vila no centro
for pid_ in ("porta_coroada", "porta_revelim", "porta_meio", "portas_sol", "porta_gaviarra"):
    if pid_ in PONTOS_PORTA:
        F0_, n_, sole_ = PONTOS_PORTA[pid_]
        MEDIDAS["ancoras"][pid_] = [round(float(F0_[0] + n_[0] * 2), 1), round(float(sole_ + ZREF + 11), 1), round(-float(F0_[1] + n_[1] * 2), 1)]
for el_, (x_, y_, z_) in {"igreja_anjos": (18.2, 256.3, 85.5), "igreja_misericordia": (4.8, 277.2, 82.0),
                         "igreja_estevao": (93.2, 145.5, 92.0), "capela_bom_jesus": (16.0, -207.6, 88.0),
                         "capela_encontro": (27.5, -284.5, 82.0), "estatua": (31.9, -215.0, 80.5)}.items():
    MEDIDAS["ancoras"][el_] = [x_, z_, -y_]
MEDIDAS["ancoras"]["escarpas"] = ancora(-40, -300, 14.0)
if GUARITAS:
    g_ = min(GUARITAS, key=lambda g: np.hypot(g[0][0] - 60, g[0][1] + 395))[0]
    MEDIDAS["ancoras"]["guaritas"] = [round(float(g_[0]), 1), round(float(cota_t(*g_)) + 9, 1), round(-float(g_[1]), 1)]
MEDIDAS["ancoras"]["vila"] = ancora(55, 200, 16.0)
(AQUI / "saida").mkdir(exist_ok=True)
(AQUI / "dados" / "medidas.json").write_text(json.dumps(_arred(MEDIDAS), ensure_ascii=False, indent=1))
print(json.dumps(_arred(MEDIDAS), ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
glb = AQUI / "saida" / "valenca-bruto.glb"
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
    final = SAIDA / "valenca.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "valenca.blend"))

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
    SUF = next((a_.split("=")[1] for a_ in sys.argv if a_.startswith("--sufixo=")), "")
    for nm_, az_, el_, di_, xy_, zz_ in [  # vistas das igrejas, do lado das fotografias
        ("i_anjos", 245, 24, 34, (22, 259), 71), ("i_miser", 285, 10, 30, (6, 273), 69),
        ("i_miser_sul", 170, 32, 30, (11, 269), 68), ("i_estevao", 250, 14, 55, (92, 155), 77),
        ("i_bomjesus", 120, 16, 40, (12, -206), 78), ("i_encontro", 150, 30, 30, (26, -284), 77),
        ("i_estatua", 150, 24, 16, (31.9, -215), 77), ("i_largo", 220, 35, 90, (20, 265), 70),
        ("m_meio_n", 20, 40, 75, (75, -70), 72), ("m_meio_s", 200, 40, 75, (75, -70), 72),
        ("m_meio_alto", 0, 80, 120, (75, -70), 72), ("f_estevao", 285, 9, 38, (83, 153), 76),
        ("f_estevao_ne", 30, 20, 55, (97, 160), 76), ("f_bomjesus", 125, 9, 26, (16.4, -207.6), 78),
        ("f_meio_sul", 195, 12, 30, (73.5, -79), 74)]:
        VISTAS_EXTRA.append(((nm_ + SUF, az_, el_, di_), dict(lente=35, alvo_xy=xy_, alvo_z=zz_)))
    VISTAS_EXTRA.append((("o_calcada_meio" + SUF, 180, 89.9, 80), dict(orto=26, alvo_xy=(70, -86), alvo_z=72, res=(800, 800))))
    for pid_, (F0_, n_, sole_) in PONTOS_PORTA.items():  # uma vista de frente, um pouco de lado, para cada porta
        az_ = math.degrees(math.atan2(n_[0], n_[1])) + 25
        VISTAS_EXTRA.append(((f"p_{pid_}{SUF}", az_, 22, 40), dict(lente=35, alvo_xy=tuple(F0_), alvo_z=sole_ + 4)))
    vista("capa", 160, 34, 1330, lente=50, res=(1280, 800), destino=SAIDA / "capa.jpg", alvo_xy=(0, -150), alvo_z=60)
    if "--capa" not in sys.argv:
        vista("topo", 180, 89.9, 1500, orto=900, res=(800, 1300))
        vista("aereo_sul", 185, 32, 700, lente=50, alvo_xy=(30, -200))   # como a 1.ª fotografia aérea
        vista("aereo_sudeste", 150, 38, 900, lente=50, alvo_xy=(30, 0))  # como a 2.ª fotografia aérea
        vista("poente", 270, 28, 900)
        vista("norte", 10, 30, 800, alvo_xy=(30, 150))
        vista("baluarte", 200, 12, 70, lente=35, alvo_xy=(-30, -335))
        for v_ in VISTAS_EXTRA:
            vista(*v_[0], **v_[1])
