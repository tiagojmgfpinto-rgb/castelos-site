"""Constrói a maqueta 3D da Fortaleza de Monção (estado atual) em Blender e exporta-a para
public/maquetas/moncao/moncao.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow
    python pipeline/maquetas/moncao/construir.py            # GLB
    python pipeline/maquetas/moncao/construir.py --vistas   # GLB + capa + vistas de verificação

Dados: dados/fontes.json (recolher.py: OSM via Overture), dados/lidar.npz e dados/orto.jpg (lidar.py:
MDT, MDS, classes e cor da nuvem de pontos do LiDAR 2025 da DGT), dados/ortofoto.jpg (ortofoto.py:
ortofotomapa OrtoSat 2023 da DGT a 25 cm, acertado com o LiDAR), dados/escarpas.json (tracar.py: os
panos de muralha traçados no relevo) e src/data/maquetas/moncao.json, com a leitura das fontes.

Monção é uma vila medieval cercada no século XVII por uma fortificação abaluartada, num esporão sobre o
rio Minho. O relevo da maqueta é o MDT de 50 cm, com a cor do chão tirada do ortofotomapa da DGT; os panos de muralha são peças próprias (talude, cordão, parapeito, cunhais e guaritas), com a
alvenaria de granito cinzento e ferrugento das fotografias. As portas do Rosal e de Salvaterra, a ponte
da estrada nacional sobre a muralha e as igrejas Matriz e da Misericórdia levam o detalhe das
fotografias; o casario fica em volumes simples. O rio Minho é um plano de água à cota medida.
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
from kit.texturas import ruido, fbm, imagem  # noqa: E402
from kit.cena import Obra, arvore_modelo  # noqa: E402
from kit.arquitetura import _vao_desloc, altura_telhado, barra, cupula, faixa_inclinada, guardas_corredor, guarita, oitavado, porta_tunel, telhado_4aguas, telhado_generico  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "moncao.json").read_text())
ESC = json.loads((AQUI / "dados" / "escarpas.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "moncao"

LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([0.0, 0.0])  # centro da grelha do LiDAR (igual ao lidar.py e ao tracar.py)
ZREF = 0.0  # as cotas da cena são as altitudes (m)
rng = np.random.default_rng(1656)


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


# tendas da feira junto ao rio (a nascente da vila): estão no ortofotomapa de 2023 e no LiDAR de 2025, mas são
# provisórias; não entram nem como casas, nem como terrapleno, nem na cor do chão (o Tiago confirmou)
FEIRA = shapely.Polygon([(222, 82), (332, 82), (332, 148), (222, 148)])


def _terraplenos():
    """Em Monção os terraplenos de vários baluartes (o da Porta do Rosal, o da frente norte) vêm na classe 6
    ou 1 da nuvem de pontos e o MDT da DGT passa por baixo deles, à cota da rua. Onde a superfície sem
    árvores (dados/superficie.npz) fica mais de 0,5 m acima do MDT em manchas largas (mais de 80 m², com
    mais de 2,5 m de largura), fora das casas do OSM e do LiDAR, o terreno passa a ser essa superfície."""
    S = np.load(AQUI / "dados" / "superficie.npz")
    sup = S["sup"].astype(float)
    tem = sup > -32000
    sup = np.where(tem, sup / 100.0 - ZREF, MDT)
    sup = np.where(VEG & ~tem, MDT, sup)
    pl = []
    for f in F["edificios"]:
        g = geom(f)
        if g is not None and np.ndim(g) == 2 and len(g) >= 3:
            q = shapely.Polygon(g)
            if q.is_valid:
                pl.append(q.buffer(0.8))
    pl += [shapely.Polygon(c["planta"]).buffer(0.8) for c in json.loads((AQUI / "dados" / "casas.json").read_text())]
    pl.append(FEIRA)
    X, Y = np.meshgrid(LXS, LYS)
    casa = shapely.contains_xy(shapely.union_all(pl), X, Y)
    alto = (sup - MDT > 0.5) & ~casa
    alto &= ~((X > -86) & (X < -70) & (Y > 36) & (Y < 46))  # degraus da Misericórdia, no largo
    alto = ndi.binary_opening(alto, structure=np.ones((5, 5)))
    lab, n_ = ndi.label(alto)
    tam = ndi.sum(alto, lab, range(1, n_ + 1)) * LPASSO ** 2
    alto = np.isin(lab, 1 + np.nonzero(tam > 80)[0])
    # preenche as falhas (sem pontos) dentro das manchas com a vizinhança
    sup = ndi.grey_closing(np.where(alto, sup, -1e3), size=3)
    # suaviza (os pontos mais altos de cada célula fazem degraus e dentes no terrapleno)
    peso = ndi.gaussian_filter(alto.astype(float), 1.5)
    sup = ndi.gaussian_filter(np.where(alto, sup, 0.0), 1.5) / np.maximum(peso, 1e-6)
    return np.where(alto, np.maximum(sup, MDT), MDT), alto


MDT, TERRAPLENO = _terraplenos()


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


def material(nome, cor, rug=0.9, tex=None, vcor=False, normal=None, forca=1.0, uv_nome=None, escala=None):
    mt = bpy.data.materials.new(nome)
    mt.use_nodes = True
    nt = mt.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    mapa = None
    if escala is not None:  # escala própria do material (sai no glTF como KHR_texture_transform)
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
    """Mapa de normais (espaço tangente, convenção do glTF) a partir de um mapa de alturas periódico."""
    gx = (np.roll(alt, -1, 1) - np.roll(alt, 1, 1)) * forca
    gy = (np.roll(alt, -1, 0) - np.roll(alt, 1, 0)) * forca
    nv = np.stack([-gx, -gy, np.ones_like(alt)], 2)
    nv /= np.linalg.norm(nv, axis=2, keepdims=True)
    return nv * 0.5 + 0.5


def textura_alvenaria(n=1024, metros=8.0, semente=1661):
    """Alvenaria de granito dos panos (8 m por repetição), medida no Street View da Porta do Rosal: fiadas de
    0,28 a 0,5 m, pedras de 0,45 a 1,1 m de comprido (e algumas pequenas), mal fiadas, bege, cinzentas e cor
    de salmão, com manchas de ferrugem (o granito da região tem óxidos de ferro), juntas escuras e fundas,
    cantos arredondados, musgo e líquenes brancos. Os blocos são células de Voronoi alongadas, periódicas.
    Devolve a cor e o relevo (para o mapa de normais)."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    pts, cores = [], []
    y = 0.0
    # cinzentos rosados do granito de Monção; a ferrugem vem depois, em manchas largas
    # cores medidas nas fotografias da Porta do Rosal (Street View): granito cor de salmão e ferrugem,
    # bege e cinzento, algumas pedras castanho-avermelhadas escuras
    # (revisto pelas cores medidas na fotografia da muralha ao sol: bege rosado, mediana c. 182/156/129)
    paleta = np.array([(0.66, 0.50, 0.42), (0.60, 0.45, 0.38), (0.68, 0.60, 0.52), (0.60, 0.55, 0.50),
                       (0.55, 0.52, 0.49), (0.47, 0.44, 0.42), (0.52, 0.38, 0.33), (0.74, 0.65, 0.56)])
    prob = np.array([0.12, 0.08, 0.22, 0.20, 0.16, 0.08, 0.05, 0.09])
    while y < n - 0.15 * m_px:
        hf = r.uniform(0.28, 0.50) * m_px  # fiadas medidas na Porta do Rosal (c. 30 px/m no Street View)
        if y + hf > n - 0.25 * m_px:
            hf = n - y
        x = r.uniform(0, 0.6) * m_px
        x0 = x
        while x < x0 + n:
            wf = (r.uniform(0.22, 0.38) if r.random() < 0.18 else r.uniform(0.45, 1.1)) * m_px
            pts.append(((x + wf / 2) % n, y + hf / 2 + r.uniform(-0.12, 0.12) * hf))
            cores.append(paleta[r.choice(8, p=prob)] * r.uniform(0.92, 1.08))
            x += wf
        y += hf
    pts = np.array(pts)
    cores = np.array(cores)
    k = len(pts)
    viz = np.vstack([pts + np.array([dx, dy]) * n for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    arv = cKDTree(viz * np.array([1.0, 2.1]))  # métrica alongada: blocos deitados
    yy, xx = np.mgrid[0:n, 0:n]
    # contornos irregulares: deforma as coordenadas antes de procurar a pedra mais próxima
    wx = (fbm(xx, yy, 0.25 * m_px, semente + 7) - 0.5) * 0.12 * m_px
    wy = (fbm(xx, yy, 0.25 * m_px, semente + 8) - 0.5) * 0.08 * m_px
    d, idx = arv.query(np.stack([(xx + wx).ravel(), (yy + wy).ravel() * 2.1], 1), k=2)
    junta = (d[:, 1] - d[:, 0]).reshape(n, n)
    cel = (idx[:, 0] % k).reshape(n, n)
    cores = cores * 0.72 + np.array([0.60, 0.54, 0.47]) * 0.28  # menos saturado: o granito é baço
    tex = cores[cel]
    manchas = fbm(xx, yy, 9.0, semente + 13)[..., None]  # cada pedra é manchada
    tex = tex * (0.8 + 0.4 * manchas)
    # cada pedra é um pouco abaulada: mais clara no meio, escura na junta
    # juntas fundas (2 a 4 cm), cantos das pedras arredondados (a distância à junta é suavizada)
    junta = ndi.gaussian_filter(junta, 1.1, mode="wrap")
    larg_j = (0.016 + 0.02 * fbm(xx, yy, 30.0, semente + 11)) * m_px  # juntas de largura variável
    bomb = np.clip((junta - larg_j) / (0.06 * m_px), 0, 1)
    alt = np.sqrt(bomb) * (junta > larg_j)
    tex = tex * (0.74 + 0.26 * alt[..., None])
    sombra_j = np.clip(1.0 - (junta - larg_j * 0.6) / (larg_j * 0.6 + 1e-6), 0, 1)[..., None]
    arg = np.array([0.22, 0.19, 0.17]) * (0.8 + 0.4 * fbm(xx, yy, 6.0, semente + 14))[..., None]
    tex = tex * (1.0 - sombra_j) + arg * sombra_j
    # grão do granito: pintas escuras (biotite) e claras (feldspato)
    rg = np.random.default_rng(semente + 12)
    pint = ndi.gaussian_filter(rg.random((n, n)), 0.7, mode="wrap")
    pint = (pint - pint.mean()) / pint.std()
    tex = tex * (1 + 0.07 * pint[..., None])
    tex = np.where((pint < -2.3)[..., None], tex * 0.75, tex)
    tex = np.where((pint > 2.1)[..., None], tex * 0.75 + 0.22, tex)
    # óxidos de ferro: manchas largas que apanham várias pedras, cada uma com a sua intensidade
    ferr = np.clip((fbm(xx, yy, 70.0, semente + 9) - 0.5) * 3.2, 0, 1)[..., None]
    absorve = r.uniform(0.2, 1.0, k)[cel][..., None]
    tex = tex * (1 - 0.25 * ferr * absorve) + np.array([0.62, 0.43, 0.31]) * 0.25 * ferr * absorve
    u, v = xx / n, yy / n
    grao = fbm(u * 48, v * 48, 1.5, semente + 1)[..., None]
    liq = fbm(u * 36, v * 36, 1.0, semente + 2)[..., None]
    musgo = fbm(u * 18, v * 18, 1.0, semente + 3)[..., None]
    escorre = (0.5 + 0.5 * np.sin(2 * math.pi * (u * 9 + 0.3 * np.sin(2 * math.pi * v * 2))))[..., None]
    tex = tex * (0.86 + 0.24 * grao)
    tex = np.where(liq > 0.84, tex * 0.5 + np.array([0.78, 0.78, 0.74]) * 0.5, tex)
    tex = np.where((musgo > 0.74) & (bomb[..., None] < 0.5), tex * 0.6 + np.array([0.26, 0.30, 0.17]) * 0.4, tex)
    tex = tex * (0.93 + 0.08 * escorre)
    return np.clip(tex, 0, 1), ndi.gaussian_filter(alt, 1.0) + 0.12 * grao[..., 0]


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
            base = np.array([0.66, 0.62, 0.55]) * r.uniform(0.9, 1.06) + r.normal(0, 0.008, 3)
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
    tex = np.where(liq > 0.84, tex * 0.5 + 0.42, tex)
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


def textura_grao(n=256):
    yy, xx = np.mgrid[0:n, 0:n] / n
    g = fbm(xx * 30, yy * 30, 2.0, 31)
    return ((0.84 + 0.22 * g)[..., None] * np.array([0.40, 0.47, 0.25])).clip(0, 1)


# ---------------------------------------------------------------- contorno da maqueta

def contorno_maqueta():
    """Limite da maqueta: a vila amuralhada com as obras exteriores (o invólucro dos panos traçados e das
    linhas de muralha do OSM, 30 m para fora) e, a norte, a encosta e a margem do rio Minho até ao limite
    das folhas do LiDAR."""
    gs = []
    for f in F["infraestrutura"]:
        if f["classe"] == "city_wall" and f["geometria"]["type"] == "LineString":
            gs.append(shapely.LineString(geom(f)))
    for e in ESC:
        gs.append(shapely.LineString(e["linha"]))
    u = shapely.union_all(gs).convex_hull.buffer(30, join_style="round")
    u = u.union(shapely.box(-470, 170, 395, 300))
    u = u.buffer(-12).buffer(12)
    caixa = shapely.box(LXS[0] + 2, LYS[0] + 2, LXS[-1] - 2, LYS[-1] - 2)
    u = u.intersection(caixa)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    return shapely.Polygon(u.exterior).simplify(0.8), None


CONTORNO, PRACA = contorno_maqueta()

# ---------------------------------------------------------------- rio Minho
# Sobre a água não há pontos do LiDAR (classe 0) e o MDT fica à cota da superfície, entre 2,7 e 3,1 m.
NIVEL = 3.0
_ag = (CLS == 0) & (MDT < NIVEL + 0.45)
_lab, _k = ndi.label(_ag)
_ar = ndi.sum(np.ones_like(MDT), _lab, np.arange(1, _k + 1)) * LPASSO * LPASSO
AGUA = np.isin(_lab, 1 + np.nonzero(_ar > 400)[0])
AGUA = ndi.binary_closing(AGUA, iterations=3) & (MDT < NIVEL + 0.6)
D_AGUA = ndi.distance_transform_edt(AGUA) * LPASSO  # distância à margem, dentro de água
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


def vaos(lin, topo, pe, a, fim):
    """Vãos das portas abertos no próprio pano (sem o cortar): o traçado é densificado a 0,1 m na largura do
    vão e, aí, a face começa no intradorso do arco (zbot) em vez de no chão; o resto do pano (por cima do
    arco, o cordão e o parapeito) segue inteiro. Devolve os arrays densificados e zbot (-inf fora dos vãos)."""
    zbot = np.full(len(lin), -np.inf)
    for d_ in PORTAS.values():
        v = d_.get("vao")
        if not v:
            continue
        n_ = unit(np.array(d_["normal"], float))
        t_ = np.array([-n_[1], n_[0]])
        p0 = np.array(d_["ponto"])
        r, omb, sole = v["r"], v["omb"], d_["soleira"] - ZREF
        s_ = (lin - p0) @ t_
        dd = (lin - p0) @ n_
        cruza = [i for i in range(len(lin) - 1)
                 if (s_[i] + r + 0.5) * (s_[i + 1] - r - 0.5) < 0 or (s_[i] - r - 0.5) * (s_[i + 1] + r + 0.5) < 0
                 if min(abs(dd[i]), abs(dd[i + 1])) < 8 and abs(s_[i] - s_[i + 1]) > 1e-6]
        cruza = [i for i in cruza if abs(dd[i] + (dd[i + 1] - dd[i]) * (0 - s_[i]) / (s_[i + 1] - s_[i])) < 4.0]
        if not cruza:
            continue
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
        zb = sole + omb + np.sqrt(np.clip(r * r - s_ ** 2, 0, None))
        zbot = np.where(dentro, np.maximum(zbot, zb), zbot)
        # o pé à frente do vão é a soleira
        perto_ = np.abs(s_) <= r + 0.6
        pe = np.where(perto_ & (np.abs(np.arange(len(lin)) - (i + 1 + len(fs) / 2)) <= len(fs) / 2 + 1), sole, pe)
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
    zc = topo - (PARAPEITO if alto else 0.0)
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


def textura_ondas(n=256):
    """Ondulação miúda da água (mapa de alturas periódico para o mapa de normais)."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    a = np.zeros((n, n))
    r = np.random.default_rng(5)
    for _ in range(14):
        kx, ky = r.integers(-6, 7, 2)
        if kx == 0 and ky == 0:
            continue
        a += np.sin(2 * math.pi * (kx * xx + ky * yy) + r.random() * 6.28) / math.hypot(kx, ky)
    return a / np.abs(a).max()


def construir_rio():
    """Plano de água do Minho à cota NIVEL sobre a área sem pontos do LiDAR (e 3 m por baixo das margens),
    com a cor pela fundura: verde-garrafa escuro no canal e castanho-esverdeado junto às margens, como
    nas fotografias aéreas, e um fio claro de corrente na margem."""
    passo = 1.0
    xs = np.arange(LXS[0] + 1, LXS[-1] - 1, passo)
    ys = np.arange(LYS[0] + 1, LYS[-1] - 1, passo)
    X, Y = np.meshgrid(xs, ys)
    agua_d = ndi.binary_dilation(AGUA, iterations=6)
    usa = (amostra(agua_d.astype(float), X, Y) > 0.01) & shapely.contains_xy(CONTORNO.buffer(-0.3), X, Y)
    if usa.sum() < 10:
        return
    dcosta = amostra(D_AGUA, X, Y)
    idx = -np.ones(X.shape, int)
    idx[usa] = np.arange(usa.sum())
    V = np.stack([X[usa], Y[usa], np.full(usa.sum(), NIVEL)], 1)
    I, J = np.nonzero(usa[:-1, :-1] & usa[1:, :-1] & usa[:-1, 1:] & usa[1:, 1:])
    Fq = np.stack([idx[I, J], idx[I, J + 1], idx[I + 1, J + 1], idx[I + 1, J]], 1)
    t = np.clip(dcosta[usa] / 25.0, 0, 1)[:, None]
    raso, fundo = np.array([0.30, 0.34, 0.27]), np.array([0.06, 0.15, 0.18])
    cor = raso * (1 - t) + fundo * t
    ruido_ = fbm(X[usa] / 6, Y[usa] / 6, 1.0, 12)[:, None]
    cor = cor * (0.92 + 0.16 * ruido_)
    esp = np.clip((1.2 - dcosta[usa]) / 1.2, 0, 1)[:, None] * 0.35
    cor = cor * (1 - esp) + np.array([0.62, 0.66, 0.62]) * esp
    me = bpy.data.meshes.new("rio")
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
    me.materials.append(MATS["rio"])
    ob = bpy.data.objects.new("Terreno_rio", me)
    bpy.context.scene.collection.objects.link(ob)
    ob["elemento"] = "rio"
    ob.parent = GRUPOS["Terreno"]
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = min(1.0, 40000 / (2 * len(Fq)))
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear()
    ob.data = me2
    me2.materials.clear()
    me2.materials.append(MATS["rio"])
    uv_mundo(ob, 9.0)
    MEDIDAS["rio"] = {"nivel": NIVEL, "area_ha": round(float(usa.sum()) * passo * passo / 1e4, 2), "triangulos": len(me2.polygons)}

# ---------------------------------------------------------------- terreno

TOMBO = (0.36, -0.18)  # deslocamento dos telhados no OrtoSat 2023 (linhas para sul, colunas para este) por metro de altura


def textura_terreno(k=2):
    """Cor do chão a 25 cm: o ortofotomapa da DGT (OrtoSat 2023, dados/ortofoto.jpg do ortofoto.py, acertado
    com o LiDAR), a camada que o WebSIG da Câmara de Monção mostra. Os telhados (que na maqueta são peças
    próprias) preenchem-se com a cor do chão à volta (convolução normalizada a várias escalas); as sombras
    fortes dos prédios levantam-se um pouco, porque a maqueta faz as suas. Norte em cima."""
    from PIL import Image
    img = np.asarray(Image.open(AQUI / "dados" / "ortofoto.jpg"), float) / 255.0
    H, W = img.shape[:2]
    assert (H, W) == ((len(LYS) - 1) * k, (len(LXS) - 1) * k), "ortofoto.jpg fora da grelha (corre ortofoto.py)"
    Hm = MDS - MDT
    # só os telhados (as copas ficam: o chão debaixo das árvores é sombra de mata, como na imagem)
    casa = (CLS == 6) & (Hm > 1.5)
    tapa = casa[:-1, :-1][::-1]  # células da imagem (linha 0 = norte), a 50 cm
    h_ = np.where(casa, Hm, 0.0)[:-1, :-1][::-1]
    tc = ndi.zoom(tapa.astype(np.uint8), k, order=0).astype(bool)
    # o satélite vê os prédios de lado: o telhado aparece deslocado para sul-sudoeste c. 0,36 m por metro de
    # altura para sul e 0,18 m para oeste (medido pela sobreposição dos telhados da imagem com os do LiDAR);
    # tapa-se o caminho da base ao telhado (fachada e telhado tombados), conforme a altura de cada célula
    rr, cc = np.nonzero(tapa)
    hh = h_[rr, cc]
    for t_ in np.linspace(0.1, 1.0, 10):
        r2 = ((rr + 0.5) * k + t_ * TOMBO[0] * hh * k / LPASSO).astype(int).clip(0, H - 1)
        c2 = ((cc + 0.5) * k + t_ * TOMBO[1] * hh * k / LPASSO).astype(int).clip(0, W - 1)
        for dr in range(k):
            for dc in range(k):
                tc[(r2 - k // 2 + dr).clip(0, H - 1), (c2 - k // 2 + dc).clip(0, W - 1)] = True
    sem_pontos = ((CLS == 0) & ~AGUA)[:-1, :-1][::-1]
    tc |= ndi.zoom(sem_pontos.astype(np.uint8), k, order=0).astype(bool)
    tc = ndi.binary_closing(tc, iterations=1) | tc
    # as tendas brancas da feira: o branco (e a sua sombra) passa a ser o chão à volta
    Xp, Yp = np.meshgrid(LXS[0] + (np.arange(W) + 0.5) / k * LPASSO, LYS[-1] - (np.arange(H) + 0.5) / k * LPASSO)
    na_feira = shapely.contains_xy(FEIRA, Xp, Yp)
    # (fica só a mata das bordas: verde-escuro)
    mata = ((img[..., 1] - (img[..., 0] + img[..., 2]) / 2) > 0.02) & (img.mean(-1) < 0.33)
    mata = ndi.binary_opening(mata, iterations=3)
    borda = shapely.contains_xy(FEIRA.buffer(-6.0), Xp, Yp)
    tenda = na_feira & ~(mata & ~borda)
    tc |= tenda
    # chão aberto à volta do recinto (nem mata, nem casas, nem tendas): a cor do terreiro
    perto = shapely.contains_xy(FEIRA.buffer(35.0), Xp, Yp) & ~na_feira & ~tc & ~mata
    cor_feira = np.median(img[perto], 0)
    # sombras de árvores e muros no chão aberto (escuras face à vizinhança e sem ser mata): tapam-se também,
    # porque a maqueta faz as suas sombras (auditoria de novembro de 2026: sombras carimbadas no chão)
    lum_ = img.mean(-1)
    sombra = (lum_ < 0.72 * ndi.uniform_filter(lum_, 30 * k + 1)) & (lum_ < 0.36) & ~mata
    tc |= ndi.binary_dilation(ndi.binary_opening(sombra, iterations=1), iterations=k)
    bom = (~tc).astype(float)
    out = img.copy()
    falta = tc.copy()
    for sg in (2, 4, 8, 16, 32):
        num = np.stack([ndi.gaussian_filter(img[..., c] * bom, sg) for c in range(3)], -1)
        den = ndi.gaussian_filter(bom, sg)[..., None]
        ok = (den[..., 0] > 0.15) & falta
        out[ok] = (num / np.maximum(den, 1e-6))[ok]
        falta &= ~ok
    # no miolo da vila o satélite não vê as ruas (os telhados tombados tapam-nas): fica a cor média do chão à
    # volta, com o grão dos cubos de granito (só se vê nas frestas entre as casas da maqueta)
    for sg in (64, 128):
        num = np.stack([ndi.gaussian_filter(img[..., c] * bom, sg) for c in range(3)], -1)
        den = ndi.gaussian_filter(bom, sg)[..., None]
        ok = (den[..., 0] > 0.003) & falta
        out[ok] = (num / np.maximum(den, 1e-6))[ok]
        falta &= ~ok
    # o recinto da feira sem as tendas: terreiro de terra e erva rala, com manchas
    yy_, xx_ = np.nonzero(tenda)
    d_ = np.clip(ndi.distance_transform_edt(tenda)[tenda] / 24.0, 0, 1)[:, None]  # a cor da borda esbate-se
    out[tenda] = out[tenda] * (1 - d_) + cor_feira * d_
    out[tenda] *= (1 + 0.18 * (fbm(xx_, yy_, 40.0, 58) - 0.5) + 0.08 * (fbm(xx_, yy_, 6.0, 59) - 0.5))[:, None]
    g_ = ndi.gaussian_filter(np.random.default_rng(56).random((H, W)), 0.7)
    out[tc] *= (1 + 0.08 * (g_[tc] - g_.mean()) / g_.std())[:, None]
    # sombras dos prédios e dos muros (a imagem é de manhã): levantam-se até metade do caminho para a luz
    # média à volta, sem mexer na cor
    lum = out.mean(-1)
    viz = ndi.gaussian_filter(lum, 10)
    somb = np.clip((viz - lum - 0.05) * 8, 0, 1) * np.clip((0.32 - lum) * 6, 0, 1)
    somb = ndi.gaussian_filter(somb, 1.0)[..., None]
    out = out * (1 + somb * np.clip(viz / np.maximum(lum, 0.03) - 1, 0, 1.5)[..., None] * 0.5)
    # debaixo de água: cascalho e lodo do leito (só se vê junto às margens)
    ag = ndi.zoom((AGUA[:-1, :-1] & AGUA[1:, 1:])[::-1].astype(np.uint8), k, order=0).astype(bool)
    out[ag] = np.array([0.36, 0.36, 0.30])
    out = np.clip(out, 0, 1)
    return pavimentos(out, img, tc, k)


# encosta norte, entre a vila e o rio (socalcos e o passadiço de madeira; ortofotomapa da DGT)
ENCOSTA_N = shapely.Polygon([(-104, 192), (-50, 168), (-13, 180), (5, 170), (57, 152), (85, 143), (90, 150),
                             (90, 250), (-104, 250)])


# larguras (m) de cada classe de caminho do OSM
LARG_CAM = {"secondary": 8.0, "tertiary": 7.0, "unclassified": 5.5, "residential": 5.0, "service": 4.0,
            "pedestrian": 5.0, "footway": 2.2, "path": 1.6, "track": 3.0, "steps": 2.5, "cycleway": 2.0, "unknown": 3.0}


def pavimentos(out, img, tc, k=2):
    """Limpa as ruas e estradas: o ortofotomapa já tem o pavimento verdadeiro (asfalto, cubos, a calçada da
    praça Deu-la-Deu, os jardins), mas também os carros estacionados e as sombras. Em cada troço do OSM
    (com a largura da classe, e só onde a imagem é de facto chão de rua) mede-se a cor do pavimento (mediana
    dos pontos ao longo do eixo); o que se afasta muito dela (carros, sombras, marcas) passa a essa cor, com
    o grão fino da imagem à volta. O resto fica como no ortofotomapa."""
    from PIL import Image, ImageDraw
    H, W = out.shape[:2]
    px = lambda x, y: ((x - LXS[0]) / (LXS[-1] - LXS[0]) * W, (LYS[-1] - y) / (LYS[-1] - LYS[0]) * H)
    ppm = W / (LXS[-1] - LXS[0])
    cor = Image.new("RGB", (W, H), (0, 0, 0))
    msk = Image.new("L", (W, H), 0)
    dc, dm = ImageDraw.Draw(cor), ImageDraw.Draw(msk)
    lum_o = img.mean(-1)
    n_tr = 0
    for f in sorted(F["caminhos"], key=lambda f_: LARG_CAM.get(f_["classe"], 3.0)):  # as largas por cima
        g = f["geometria"]
        if g["type"] != "LineString" or f["subtipo"] != "road":
            continue
        cl = f["classe"]
        if cl in ("path", "track", "steps"):
            continue  # caminhos de terra e escadas ficam como no orto
        xy = geom(f)
        ln_ = shapely.LineString(xy)
        if ln_.intersection(ENCOSTA_N).length > 0.5 * ln_.length:
            continue  # o passadiço e os socalcos da encosta norte
        q_ = np.array([ln_.interpolate(d_).coords[0] for d_ in np.arange(0, ln_.length, 0.25)] or [xy[0]])
        cc_, rr_ = px(q_[:, 0], q_[:, 1])
        cc_, rr_ = cc_.astype(int).clip(0, W - 1), rr_.astype(int).clip(0, H - 1)
        livre_ = ~tc[rr_, cc_]
        if livre_.sum() < 8:
            continue
        v_ = img[rr_, cc_][livre_]
        l_ = v_.mean(1)
        v_ = v_[(l_ >= np.percentile(l_, 25)) & (l_ <= np.percentile(l_, 85))]  # sem sombras nem carros claros
        med = np.median(v_, 0)
        larg = LARG_CAM.get(cl, 3.0) * ppm * 0.9
        pts = [px(*q) for q in xy]
        dc.line(pts, fill=tuple(int(c * 255) for c in med), width=max(2, int(round(larg))), joint="curve")
        dm.line(pts, fill=255, width=max(2, int(round(larg))), joint="curve")
        n_tr += 1
    rua = (np.asarray(msk) > 0) & ~tc
    med = np.asarray(cor, float) / 255.0
    # só onde a imagem parece pavimento (cinzento ou bege pouco saturado), não relva nem canteiros
    sat = img.max(-1) - img.min(-1)
    verde = img[..., 1] - (img[..., 0] + img[..., 2]) / 2
    chao = ~((verde > 0.035) & (sat > 0.06))
    dif = np.abs(img - med).max(-1)
    lum_m = med.mean(-1)
    # carros e sombras: longe da cor do pavimento; o resto do pavimento fica com o seu pormenor
    estranho = rua & ((dif > 0.10) | (lum_o < lum_m - 0.07))
    estranho = ndi.binary_opening(estranho, iterations=1)
    estranho = ndi.binary_dilation(estranho, iterations=2) & rua & (chao | (dif > 0.10))
    grao = np.clip(lum_o - ndi.gaussian_filter(lum_o, 2.0), -0.04, 0.04)
    rg = np.random.default_rng(2023)
    ru = ndi.gaussian_filter(rg.random((H, W)), 0.7)
    ru = (ru - ru.mean()) / ru.std()
    limpo = med * (1 + 0.035 * ru[..., None] + 0.12 * (fbm(*np.mgrid[0:H, 0:W][::-1], 14.0 * k, 77)[..., None] - 0.5))
    res = out.copy()
    res[estranho] = np.clip(limpo + grao[..., None] * 0.5, 0, 1)[estranho]
    MEDIDAS["chao"] = {"fonte": "DGT OrtoSat 2023 (WMS ortosat2023), 25 cm, acertado ao LiDAR 2025",
                       "trocos_limpos": n_tr, "rua_limpa_pct": round(100 * float(estranho.sum()) / max(1, rua.sum()), 1)}
    # bordas suaves para não ficar serrilhado
    sm = ndi.gaussian_filter(estranho.astype(float), 0.8)[..., None]
    res = out * (1 - sm) + res * sm
    return np.clip(res, 0, 1)


def construir_terreno():
    nx, ny = len(LXS), len(LYS)
    X, Y = np.meshgrid(LXS, LYS)
    Z = MDT.copy()
    # leito do rio: desce com a distância à margem (o MDT sobre a água é só a superfície)
    Z = np.where(AGUA, np.minimum(Z, NIVEL - np.clip(0.4 + 0.12 * D_AGUA, 0.4, 5.0)), Z)
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


# ---------------------------------------------------------------- portas

class Quadro:
    """Referencial local de uma fachada: s ao longo dela (para a direita de quem olha de fora), z para
    cima e d para fora da parede, a partir de um ponto da face."""

    def __init__(self, m, origem, normal):
        self.m = m
        self.o = np.asarray(origem, float)
        self.n = unit(np.asarray(normal, float))
        self.t = np.array([self.n[1], -self.n[0]])  # olhando de fora (contra a normal), a direita
        self.bt, self.zb = 0.0, 0.0  # talude da face: d recua bt por metro acima de zb

    def p(self, s, z, d):
        q = self.o + self.t * s + self.n * (d - self.bt * (z - self.zb))
        return (q[0], q[1], z)

    def caixa_t(self, s0, s1, z0, z1, d0, d1, mat):
        """Bloco encostado à face em talude (as faces da frente e de trás seguem o talude)."""
        self.placa([(s0, z0), (s1, z0), (s1, z1), (s0, z1)], d0, d1, mat)

    def bloco(self, s0, s1, z0, z1, d0, d1, mat):
        c = self.o + self.t * (s0 + s1) / 2 + self.n * (d0 + d1) / 2
        self.m.caixa(c, self.t, abs(s1 - s0), abs(d1 - d0), z0, z1, mat, fundo=True)

    def placa(self, sz, d0, d1, mat):
        """Extrude um polígono do plano (s, z) entre as profundidades d0 e d1."""
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
        """Coroa de arco (entre os raios r0 e r1) centrada em (s0, zn), extrudida entre d0 e d1;
        com aduelas=True, partida em aduelas com juntas finas."""
        for i in range(k):
            t0 = a0 + (a1 - a0) * i / k + (0.012 if aduelas else 0)
            t1 = a0 + (a1 - a0) * (i + 1) / k - (0.012 if aduelas else 0)
            q = [(s0 + r0 * math.cos(t0), zn + r0 * math.sin(t0)), (s0 + r1 * math.cos(t0), zn + r1 * math.sin(t0)),
                 (s0 + r1 * math.cos(t1), zn + r1 * math.sin(t1)), (s0 + r0 * math.cos(t1), zn + r0 * math.sin(t1))]
            self.placa(q, d0, d1, mat)

    def vao(self, largura, z0, nascenca, d, mat, reto=False):
        """Vão escuro (arco pleno, ou verga reta) no plano d."""
        w = largura / 2
        pts = [(-w, z0), (w, z0), (w, z0 + nascenca)]
        if not reto:
            pts += [(w * math.cos(math.pi * k / 14), z0 + nascenca + w * math.sin(math.pi * k / 14)) for k in range(1, 14)]
        pts += [(-w, z0 + nascenca)]
        self.m.face([self.p(s, z, d) for s, z in pts], mat)

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
        self.placa([(s0 - w * 0.55, z0 + alt * 0.25), (s0 + w * 0.55, z0 + alt * 0.25), (s0 + w * 0.55, z0 + alt * 0.85), (s0 - w * 0.55, z0 + alt * 0.85)],
                   d1, d1 + 0.05, mat)  # escudete das quinas
        if coroa:
            zc = z0 + alt
            self.placa([(s0 - w * 1.05, zc), (s0 + w * 1.05, zc), (s0 + w * 1.15, zc + alt * 0.28), (s0 + w * 0.5, zc + alt * 0.2),
                        (s0, zc + alt * 0.36), (s0 - w * 0.5, zc + alt * 0.2), (s0 - w * 1.15, zc + alt * 0.28)], d0, d1, mat)
            self.esfera(s0, zc + alt * 0.45, (d0 + d1) / 2, alt * 0.09, mat)

    def almofadas(self, s0, s1, z0, z1, d0, d1, mat, fiada=0.55):
        """Pilastra de almofadado rústico: blocos salientes com juntas fundas (fotografia da Porta da Coroada)."""
        self.bloco(s0 + 0.04, s1 - 0.04, z0, z1, d0, d1 - 0.06, mat)
        for k, z in enumerate(np.arange(z0, z1 - 0.1, fiada)):
            ss = (0.0 if k % 2 == 0 else 0.06)
            self.bloco(s0 + ss, s1 - ss, z + 0.04, min(z + fiada - 0.04, z1), d1 - 0.08, d1, mat)


def tunel(Q, r, sole, omb, L, bt_=None):
    """Túnel de abóbada de volta perfeita (raio r, nascença a omb do chão) que entra L metros na muralha a
    partir da face: paredes e abóbada de alvenaria escurecida (a luz não entra) e lajeado no chão. A boca
    segue o talude da face."""
    zn = sole + omb
    bt0, Q.bt = Q.bt, 0.0
    d0 = -L
    d1 = -bt0 * omb  # a boca recua com o talude: o túnel começa onde a face está à altura da nascença
    for sg in (-1, 1):
        a_, b_ = sorted((sg * r, sg * (r + 0.4)))
        Q.bloco(a_, b_, sole - 0.1, zn, d0, d1 + 0.02, 1)
    # a abóbada começa por trás da face à altura do fecho (senão o extradorso escuro sai do pano em talude)
    Q.arco_sz(0.0, r, r + 0.4, zn, d0, -bt0 * (omb + r + 0.4) - 0.1, 1, k=12)
    Q.bloco(-r - 0.05, r + 0.05, sole - 0.3, sole + 0.01, d0, 0.3, 0)
    Q.bt = bt0


def arco_granito(Q, r, sole, omb, aduela=0.5, n_ad=13, fiada=0.48, ombreira=(0.62, 0.40), prof=0.7, saliencia=0.06,
                 impostas=False, chave=False):
    """Arco de volta perfeita de granito sobre a face em talude: aduelas, ombreiras de silhares alternados
    e intradorso de cantaria nos primeiros 0,7 m do túnel. Com impostas=True, cornija de imposta na
    nascença; com chave=True, aduela de fecho saliente (Salvaterra)."""
    zn = sole + omb
    for k_ in range(n_ad):
        t0 = math.pi * k_ / n_ad + 0.01
        t1 = math.pi * (k_ + 1) / n_ad - 0.01
        ra = r + aduela * (1.2 if (chave and k_ == n_ad // 2) else 1.0)
        q = [(r * math.cos(t0), zn + r * math.sin(t0)), (ra * math.cos(t0), zn + ra * math.sin(t0)),
             (ra * math.cos(t1), zn + ra * math.sin(t1)), (r * math.cos(t1), zn + r * math.sin(t1))]
        Q.placa(q, -0.02, saliencia + (0.06 if (chave and k_ == n_ad // 2) else 0.0), 0)
    # intradorso em cantaria (boca do túnel)
    bt0, Q.bt = Q.bt, 0.0
    d1 = -bt0 * omb
    Q.arco_sz(0.0, r - 0.01, r + 0.02, zn, d1 - prof, d1 + 0.02, 0, k=12)
    for sg in (-1, 1):
        a_, b_ = sorted((sg * (r - 0.01), sg * (r + 0.02)))
        Q.bloco(a_, b_, sole, zn, d1 - prof, 0.02, 0)
    Q.bt = bt0
    for k_, zz in enumerate(np.arange(sole, zn - 0.12, fiada)):
        e_ = ombreira[k_ % 2]
        for sg in (-1, 1):
            a_, b_ = sorted((sg * (r - 0.0), sg * (r + e_)))
            Q.caixa_t(a_, b_, zz + 0.015, min(zz + fiada - 0.015, zn), -0.02, saliencia, 0)
    if impostas:
        for sg in (-1, 1):
            a_, b_ = sorted((sg * (r - 0.08), sg * (r + 0.75)))
            Q.caixa_t(a_, b_, zn - 0.2, zn, -0.02, saliencia + 0.1, 0)


def porta_rosal(Q, sole):
    """Porta do Rosal, por fora (Street View, 2021): arco de volta perfeita de granito, de c. 2,1 m de vão e
    3,5 m de altura, aberto num pano alto e liso de alvenaria (o vão é aberto no próprio pano, que segue
    inteiro por cima até ao cordão e ao parapeito); aduelas grandes, ombreiras de silhares; o túnel
    atravessa o terrapleno até à porta de dentro. Por cima, a meio do pano, a guarita sobre mísula, assente
    no cordão. À frente, o lajeado e a grelha de escoamento."""
    topo = getattr(Q, "topo", sole + 9.0)
    zc = getattr(Q, "zc", topo - PARAPEITO)
    v = PORTAS["porta_rosal"]["vao"]
    r, omb = v["r"], v["omb"]
    tunel(Q, r, sole, omb, v["tunel"])
    arco_granito(Q, r, sole, omb, aduela=0.5, n_ad=13, ombreira=(0.55, 0.38))
    Q.bt, bt0 = 0.0, Q.bt
    Q.bloco(-r, r, sole - 0.02, sole + 0.02, 0.0, 0.7, 1)  # grelha de escoamento
    Q.bloco(-2.6, 2.6, sole - 0.25, sole + 0.01, 0.0, 3.0, 0)  # lajeado à frente do arco
    Q.bt = bt0
    GUAR_EXTRA.append((np.array(Q.p(0.0, zc, 0.3)[:2]), Q.n.copy(), float(zc), float(topo)))
    return zc - sole


def porta_rosal_dentro(Q, sole):
    """Porta do Rosal, por dentro (Street View, 2021): o arco da passagem, de granito, ao fundo de um recanto
    entre dois contrafortes grossos de alvenaria (com o alto em talude e musgo); entre os contrafortes e os
    muros das escadas, duas portas de lojas abertas no terrapleno (à esquerda envidraçada, à direita de
    madeira verde) com moldura de cantaria; no alto do pano, nos dois cantos, dois arcos de abóbada a que se
    chega pelas escadas de pedra que sobem por trás dos muros laterais; gradeamento de ferro no alto. O
    pano é o da própria muralha, com o vão aberto."""
    c, s_, mv, alv, rel, vid, fer = 0, 1, 2, 3, 5, 6, 7
    topo = getattr(Q, "topo", sole + 9.0)
    v = PORTAS["porta_rosal_dentro"]["vao"]
    r, omb = v["r"], v["omb"]
    tunel(Q, r, sole, omb, v["tunel"])
    arco_granito(Q, r, sole, omb, aduela=0.42, n_ad=11, ombreira=(0.45, 0.32), saliencia=0.05)
    bt0, Q.bt = Q.bt, 0.0
    for sg in (-1, 1):
        a_, b_ = sorted((sg * (r + 0.55), sg * (r + 1.55)))
        Q.bloco(a_, b_, sole - 0.3, sole + 3.6, -0.3, 1.25, alv)              # contrafortes
        q = [(a_, sole + 3.6), (b_, sole + 3.6), (b_, sole + 4.5), (a_, sole + 4.5)]
        Q.placa([(a_, sole + 3.6), (b_, sole + 3.6), (b_, sole + 4.6)] if sg < 0 else
                [(a_, sole + 3.6), (b_, sole + 3.6), (a_, sole + 4.6)], -0.3, 1.25, alv)  # remate em talude
        a2, b2 = sorted((sg * 6.4, sg * 7.4))
        db_ = -0.3 - bt0 * (sole + 5.0 - Q.zb)  # encostados ao pano em talude
        Q.bloco(a2, b2, sole - 0.3, sole + 4.8, db_, 2.6, alv)               # muros das escadas
        Q.bloco(a2 - 0.05, b2 + 0.05, sole + 4.8, sole + 4.95, db_, 2.65, c)    # capeamento de granito
    # portas das lojas, com moldura de cantaria e lintel
    for s0, mat in ((-(r + 2.6), vid), (r + 2.6, mv)):
        Q.bloco(s0 - 0.55, s0 + 0.55, sole, sole + 2.15, -0.06, 0.0, mat)
        for sg in (-1, 1):
            a_, b_ = sorted((s0 + sg * 0.55, s0 + sg * 0.78))
            Q.bloco(a_, b_, sole, sole + 2.38, -0.02, 0.06, c)
        Q.bloco(s0 - 0.78, s0 + 0.78, sole + 2.15, sole + 2.45, -0.02, 0.06, c)
    # arcos de abóbada altos nos cantos, com aduelas de granito
    zs = sole + 5.1
    for sg in (-1, 1):
        o_ = Q.o
        Q.o = Q.o + Q.t * sg * 6.6  # por cima dos muros das escadas, como no Street View
        Q.bt = bt0  # os arcos ficam na face do pano em talude
        Q.vao(2.2, zs, 0.6, 0.012, s_)
        Q.arco_sz(0.0, 1.1, 1.5, zs + 0.6, 0.0, 0.07, c, k=10, aduelas=True)
        for sg2 in (-1, 1):
            a_, b_ = sorted((sg2 * 1.1, sg2 * 1.5))
            Q.bloco(a_, b_, zs, zs + 0.6, 0.0, 0.07, c)
        Q.o = o_
        Q.bt = 0.0
        # escadas por trás dos muros laterais: do chão até ao patamar dos arcos
        n = 16
        for k_ in range(n):
            d0_ = 2.5 - k_ * 0.16
            a_, b_ = sorted((sg * 7.4, sg * 8.6))
            Q.bloco(a_, b_, sole - 0.2, sole + (k_ + 1) * (zs - 0.3 - sole) / n, max(d0_ - 0.16, 0.05), 2.6, c)  # o último degrau fica sob o patamar
        a_, b_ = sorted((sg * 5.2, sg * 8.6))
        Q.bloco(a_, b_, zs - 0.25, zs, -0.3 - bt0 * (zs - Q.zb), 2.6, alv)  # patamar, à frente dos arcos
    for s0 in np.arange(-8.4, 8.41, 1.2):
        barra(Q.m, Q.p(s0, topo, 0.3), Q.p(s0, topo + 1.05, 0.3), 0.025, fer)
    barra(Q.m, Q.p(-8.4, topo + 1.05, 0.3), Q.p(8.4, topo + 1.05, 0.3), 0.03, fer)
    barra(Q.m, Q.p(-8.4, topo + 0.55, 0.3), Q.p(8.4, topo + 0.55, 0.3), 0.02, fer)
    Q.bt = bt0
    return topo - sole


def porta_salvaterra(Q, sole):
    """Porta de Salvaterra (fotografia de 2026): frontispício de cantaria de granito à frente do pano, com
    duas pilastras de silhares almofadados, arco de volta perfeita (c. 2,6 m de vão, 3,9 m de altura) com
    impostas e aduela de fecho, entablamento à altura do cordão da muralha e, por cima, um frontão
    curvo interrompido com um bloco central (tarja) mais alto, que passa um pouco o parapeito. O pano em
    talude continua dos dois lados, com o cordão e o parapeito; a guarita fica no ângulo à esquerda."""
    c, alv = 0, 3
    topo = getattr(Q, "topo", sole + 6.0)
    zc = getattr(Q, "zc", topo - PARAPEITO)
    v = PORTAS["porta_salvaterra"]["vao"]
    r, omb = v["r"], v["omb"]
    tunel(Q, r, sole, omb, v["tunel"])
    W = 2.55          # meia largura do frontispício
    pil = 1.0         # largura das pilastras
    ze = sole + 4.7   # base do entablamento (à altura do cordão)
    bt0 = Q.bt
    Q.bt = 0.0
    base_d = bt0 * 0.0
    prof = bt0 * (ze - sole) + 0.35   # o frontispício é vertical: à altura do entablamento sai 0,35 m da face
    # pano do frontispício entre as pilastras, com o recorte do arco
    pts = [(-(W - pil), sole), (-r, sole), (-r, sole + omb)]
    pts += [(-r * math.cos(math.pi * k_ / 14), sole + omb + r * math.sin(math.pi * k_ / 14)) for k_ in range(1, 14)]
    pts += [(r, sole + omb), (r, sole), (W - pil, sole), (W - pil, ze), (-(W - pil), ze)]
    Q.placa(pts, -0.4, prof - 0.1, c)
    # pilastras de silhares almofadados (fiadas alternadas)
    for sg in (-1, 1):
        a_, b_ = sorted((sg * (W - pil), sg * W))
        Q.bloco(a_, b_, sole - 0.2, sole + 0.45, -0.4, prof + 0.12, c)  # plinto
        for k_, zz in enumerate(np.arange(sole + 0.45, ze - 0.1, 0.52)):
            ex = 0.05 if k_ % 2 else 0.0
            Q.bloco(a_ - (ex if sg < 0 else 0), b_ + (ex if sg > 0 else 0), zz + 0.02, min(zz + 0.5, ze), -0.4, prof + 0.04, c)
    # arco: aduelas com fecho, impostas
    Q.o, o_ = Q.o + Q.n * (prof - 0.1), Q.o
    arco_granito(Q, r, sole, omb, aduela=0.5, n_ad=11, ombreira=(0.35, 0.35), prof=prof + 0.6, saliencia=0.05,
                 impostas=True, chave=True)
    Q.o = o_
    # entablamento: arquitrave, friso e cornija
    Q.bloco(-W - 0.05, W + 0.05, ze, ze + 0.22, -0.4, prof + 0.06, c)
    Q.bloco(-W - 0.05, W + 0.05, ze + 0.22, ze + 0.42, -0.4, prof + 0.02, c)
    Q.bloco(-W - 0.2, W + 0.2, ze + 0.42, ze + 0.58, -0.4, prof + 0.2, c)
    # ático: pedestais por cima das pilastras e frontão curvo interrompido
    za = ze + 0.58
    for sg in (-1, 1):
        a_, b_ = sorted((sg * (W - 0.8), sg * W))
        Q.bloco(a_, b_, za, za + 0.5, -0.3, prof, c)
        # meio frontão: arco de círculo que sobe da pilastra para o meio, com cornija
        pts = []
        for k_ in range(9):
            f = k_ / 8
            ss = sg * (W - f * (W - 1.0))
            zz = za + 0.5 + 0.75 * math.sin(f * math.pi / 2)
            pts.append((ss, zz))
        inner = [(s0_, z0_ - 0.32) for s0_, z0_ in pts[::-1]]
        Q.placa(pts + inner, -0.3, prof + 0.1, c)
    # bloco central (tarja) com almofada
    Q.bloco(-0.95, 0.95, za, za + 1.35, -0.3, prof + 0.05, c)
    Q.bloco(-0.55, 0.55, za + 0.35, za + 1.0, prof + 0.05, prof + 0.12, c)
    Q.bloco(-1.05, 1.05, za + 1.35, za + 1.5, -0.3, prof + 0.12, c)
    Q.bt = bt0
    return za + 1.5 - sole


GUAR_EXTRA = []
MAT_PORTA = ["cantaria", "sombra", "madeira_verde", "alvenaria", "cal", "relva", "vidro", "ferro"]

PORTAS = {
    # ponto e normal (para fora) medidos na nuvem de pontos; o vão é aberto no próprio pano (sem o cortar):
    # r = meia largura, omb = altura da nascença, tunel = comprimento da passagem
    "porta_rosal": dict(ponto=(-353.4, 83.0), normal=(-0.66, -0.75), estilo=porta_rosal, elemento="porta_rosal", fixo=True,
                        soleira=30.85, vao=dict(r=1.05, omb=2.4, tunel=21.0)),
    "porta_rosal_dentro": dict(ponto=(-341.2, 100.5), normal=(0.31, 0.95), estilo=porta_rosal_dentro, elemento="porta_rosal",
                               fixo=True, largura=8.5, soleira=32.0, vao=dict(r=0.95, omb=2.25, tunel=3.0)),
    "porta_salvaterra": dict(ponto=(-232.6, 202.5), normal=(-0.99, -0.12), estilo=porta_salvaterra, elemento="porta_salvaterra",
                             fixo=True, largura=2.6, soleira=33.0, vao=dict(r=1.3, omb=2.65, tunel=3.0)),
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
        if not ok.any() and d_.get("fixo"):  # o vão já está aberto: procura o pano ao lado, mais longe
            ok = (dist < 12.0) & (NR @ n0 > 0.85)
            if not ok.any():  # vão entre dois panos de outra orientação: usa o mais perto para as cotas
                ok = dist < 12.0
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
        Q.zc = float(ZC[k])
        alt = d_["estilo"](Q, sole)
        L = ponte(m, F0, n, Q.t, sole) if d_.get("ponte") else 0.0
        if d_.get("corredor"):
            guardas_corredor(m, F0, n, pts, NR, A_, TP)
        m.objeto(pid.title(), d_["elemento"], MAT_PORTA, GRUPOS["Portas"], uv=2.5)
        PONTOS_PORTA[pid] = (F0, n, sole)
        tt = np.array([-n[1], n[0]])  # passagem livre de casas e árvores à frente da porta
        lg = d_.get("largura", 3.0)
        EVITAR.append(np.array([F0 + tt * (lg + 1) + n * 0.3, F0 + tt * (lg + 1) + n * 14, F0 - tt * (lg + 1) + n * 14, F0 - tt * (lg + 1) + n * 0.3]))
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
          cornija_frente=True, soco=True, d_ini=0.0, cachorros_empena=None):
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
        if cachorros if cachorros_empena is None else cachorros_empena:
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


def portal_arquivoltas(Q, s0, z0, w, hn, n=3, passo=0.24, prof=0.5, saliente=True, timpano=True):
    """Portal românico (Santa Maria dos Anjos): arquivoltas de volta perfeita em degrau, cada uma sobre
    colunelos com capitel e base, tímpano liso e porta de madeira; num corpo ligeiramente saliente
    (saliente=False: aberto no pano, só com o arco de resguardo por fora, como na fotografia da Matriz)."""
    R = w / 2 + (n + 1) * passo
    if saliente:
        # corpo saliente do portal, com remate em cornija
        for sg in (-1, 1):
            ss = sorted((s0 + sg * R, s0 + sg * (R + 0.45)))
            Q.bloco(ss[0], ss[1], z0, z0 + hn + R + 0.5, 0, prof, CAN)
        Q.bloco(s0 - R - 0.45, s0 + R + 0.45, z0 + hn + R, z0 + hn + R + 0.5, 0, prof, CAN)
        Q.bloco(s0 - R - 0.55, s0 + R + 0.55, z0 + hn + R + 0.5, z0 + hn + R + 0.68, 0, prof + 0.1, CAN)
    else:
        # arco de resguardo com dentículos e ombreiras lisas até ao chão
        Q.arco_sz(s0, R, R + 0.22, z0 + hn, prof - 0.1, prof + 0.06, CAN, k=18, aduelas=True)
        for k_ in range(15):
            a_ = math.pi * (k_ + 0.5) / 15
            sc, zc_ = s0 + (R + 0.11) * math.cos(a_), z0 + hn + (R + 0.11) * math.sin(a_)
            Q.bloco(sc - 0.06, sc + 0.06, zc_ - 0.06, zc_ + 0.06, prof + 0.06, prof + 0.13, CAN)
        for sg in (-1, 1):
            ss = sorted((s0 + sg * R, s0 + sg * (R + 0.22)))
            Q.bloco(ss[0], ss[1], z0, z0 + hn, 0, prof + 0.06, CAN)
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
    if timpano:  # porta reta com lintel e tímpano liso
        _vao_desloc(Q, s0, w, z0, hn - 0.15, 0.012, MAD, reto=True)
        Q.bloco(s0 - w / 2, s0 + w / 2, z0 + hn - 0.2, z0 + hn, 0.0, 0.08, CAN)  # lintel
        tp = [(s0 + w / 2 * math.cos(math.pi * k / 12), z0 + hn + w / 2 * math.sin(math.pi * k / 12)) for k in range(13)]
        Q.placa(tp, 0.0, 0.05, CAN)
    else:  # porta de madeira em arco, até ao intradorso (Matriz)
        _vao_desloc(Q, s0, w, z0, hn, 0.012, MAD, reto=False)
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


def ponte_en202(m):
    """Passagem da Estrada Nacional 202 através da muralha (fotografia): a estrada corre no fundo de um
    corte de c. 25 m entre dois panos; de cada lado um encontro de cantaria sai do pano para a estrada e,
    entre eles, uma ponte pedonal de aço com guardas de prumos e, sobre a guarda sul, três silhuetas de
    torres ameadas em ferro. Cotas do LiDAR: estrada a 27,6 m, tabuleiro a c. 33,9 m (o MDS mostra a
    faixa de 7 m sobre a estrada em y = −130)."""
    y0, larg = -129.0, 3.2
    xa, xb = -74.5, -49.6      # faces dos panos (traçado)
    zr = float(cota_t(-62, y0)) - 0.3
    zt = 33.9
    u = np.array([1.0, 0.0])
    for x0, x1 in ((xa - 1.0, xa + 4.2), (xb - 4.2, xb + 1.0)):
        c = np.array([(x0 + x1) / 2, y0])
        m.caixa(c, u, x1 - x0, larg + 2.2, zr, zt - 0.35, CAN)
        m.caixa(c, u, x1 - x0 + 0.3, larg + 2.5, zt - 0.35, zt, CAN)  # cornija
        m.caixa(c, u, x1 - x0 + 0.2, larg + 2.4, zr, zr + 0.6, CAN)    # soco
    # tabuleiro: viga-caixão de aço
    L = (xb - 4.2) - (xa + 4.2)
    c = np.array([(xa + xb) / 2, y0])
    m.caixa(c, u, L + 0.4, larg, zt - 0.95, zt - 0.08, FER)
    m.caixa(c, u, L + 0.4, larg - 0.4, zt - 0.08, zt, LIS)  # pavimento
    x_ini, x_fim = xa - 1.0, xb + 1.0
    for sg in (-1, 1):
        yy = y0 + sg * (larg / 2 - 0.06)
        for x in np.arange(x_ini, x_fim + 0.01, 1.25):
            barra(m, (x, yy, zt), (x, yy, zt + 1.1), 0.03, FER)
        barra(m, (x_ini, yy, zt + 1.1), (x_fim, yy, zt + 1.1), 0.04, FER)
        barra(m, (x_ini, yy, zt + 0.15), (x_fim, yy, zt + 0.15), 0.03, FER)
    # silhuetas de torres ameadas sobre a guarda sul (três, como na fotografia)
    ys = y0 - (larg / 2 - 0.06)
    for xc in (-66.0, -62.0, -58.0):
        z0 = zt + 1.1
        w, hh = 0.9, 1.7
        pts = [(xc - w, z0), (xc - w, z0 + hh), (xc - w * 0.55, z0 + hh), (xc - w * 0.55, z0 + hh - 0.3),
               (xc - w * 0.2, z0 + hh - 0.3), (xc - w * 0.2, z0 + hh + 0.5), (xc + w * 0.2, z0 + hh + 0.5),
               (xc + w * 0.2, z0 + hh - 0.3), (xc + w * 0.55, z0 + hh - 0.3), (xc + w * 0.55, z0 + hh), (xc + w, z0 + hh), (xc + w, z0)]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            barra(m, (p0[0], ys, p0[1]), (p1[0], ys, p1[1]), 0.04, FER)
    MEDIDAS["ponte_en202"] = {"estrada": zr + 0.3, "tabuleiro": zt, "vao": round(L, 1)}
    return [[xa - 1.5, y0 - 4], [xb + 1.5, y0 - 4], [xb + 1.5, y0 + 4], [xa - 1.5, y0 + 4]]


def oculo_lobado(Q, s0, zc, r, d=0.0, lobos=4):
    """Óculo polilobado (quadrifólio) com moldura de granito e vidro escuro, como os que ladeiam o portal
    da Matriz (fotografia)."""
    k = 48
    pts = []
    for i in range(k):
        t = 2 * math.pi * i / k
        rr = r * (0.78 + 0.22 * abs(math.cos(lobos / 2 * t)))
        pts.append((s0 + rr * math.cos(t), zc + rr * math.sin(t)))
    Q.m.face([Q.p(s, z, d + 0.012) for s, z in pts], VID)
    for i in range(k):
        j = (i + 1) % k
        (s1, z1), (s2, z2) = pts[i], pts[j]
        f1 = ((s1 - s0) * 1.28 + s0, (z1 - zc) * 1.28 + zc)
        f2 = ((s2 - s0) * 1.28 + s0, (z2 - zc) * 1.28 + zc)
        Q.placa([(s1, z1), (s2, z2), f2, f1], d, d + 0.12, CAN)
    for sg in (-1, 1):  # grade em cruz
        Q.bloco(s0 - 0.02, s0 + 0.02, zc - r * 0.8, zc + r * 0.8, d + 0.02, d + 0.05, FER)
        Q.bloco(s0 - r * 0.8, s0 + r * 0.8, zc - 0.02, zc + 0.02, d + 0.02, d + 0.05, FER)


def relogio(Q, s0, zc, r, d=0.0):
    """Mostrador de relógio (fotografia da torre da Matriz): disco claro com aro de granito e ponteiros."""
    k = 24
    Q.m.face([Q.p(s0 + r * math.cos(2 * math.pi * i / k), zc + r * math.sin(2 * math.pi * i / k), d + 0.03) for i in range(k)], LIS)
    Q.arco_sz(s0, r, r + 0.12, zc, d, d + 0.08, CAN, a0=0, a1=2 * math.pi, k=k)
    for i in range(12):
        t = 2 * math.pi * i / 12
        Q.bloco(s0 + r * 0.8 * math.cos(t) - 0.02, s0 + r * 0.8 * math.cos(t) + 0.02,
                zc + r * 0.8 * math.sin(t) - 0.02, zc + r * 0.8 * math.sin(t) + 0.02, d + 0.03, d + 0.06, FER)
    Q.placa([(s0 - 0.025, zc), (s0 + 0.025, zc), (s0 + 0.025, zc + r * 0.75), (s0 - 0.025, zc + r * 0.75)], d + 0.04, d + 0.07, FER)
    Q.placa([(s0, zc - 0.025), (s0 + r * 0.55, zc + r * 0.2 - 0.025), (s0 + r * 0.55, zc + r * 0.2 + 0.025), (s0, zc + 0.025)], d + 0.04, d + 0.07, FER)


def torre_matriz(m, c, u, lado, g):
    """Torre sineira da Matriz (fotografia): toda de cantaria de granito; corpo em três andares separados
    por cornijas, fresta e mostrador do relógio no segundo, sineira com um arco de volta perfeita por face
    e sino de bronze, cornija de remate muito saliente, pináculos em bola nos cantos, coruchéu em bola ao
    meio e cata-vento de ferro. Altura no MDS: 18,6 m acima do adro."""
    u = unit(np.asarray(u, float))
    w = np.array([-u[1], u[0]])
    c = np.asarray(c, float)
    h = lado / 2
    z1, z2, z3, z4 = g + 6.8, g + 10.0, g + 14.2, g + 14.7  # cornija de remate a 49,3 m no MDS
    m.caixa(c, u, lado, lado, g - 1.5, z2 + 0.14, CAN)  # corpo maciço até à sineira, que é aberta
    m.caixa(c, u, lado + 0.16, lado + 0.16, g - 1.5, g + 0.7, CAN)  # soco
    for zz in (z1, z2):  # cornijas entre andares
        m.caixa(c, u, lado + 0.22, lado + 0.22, zz - 0.18, zz, CAN)
        m.caixa(c, u, lado + 0.34, lado + 0.34, zz, zz + 0.14, CAN)
    faces = [u, -u, w, -w]
    QS = [Quadro(m, c + n_ * h, n_) for n_ in faces]
    for i, Qf in enumerate(QS):
        if i == 0:  # face da frente (poente): relógio e, por baixo à direita, uma janela pequena
            relogio(Qf, -0.25, g + 9.1, 0.55)
            janela(Qf, 1.2, g + 7.6, 0.34, 0.5, arco=False, mat_vidro=SOM, moldura=0.1)
        elif i == 3:  # a sul: fresta
            janela(Qf, 0.0, g + 7.6, 0.28, 0.7, arco=False, mat_vidro=SOM, moldura=0.1)
        # pilastras de canto no andar da sineira
        for sg in (-1, 1):
            a, b = sorted((sg * (h - 0.45), sg * h))
            Qf.bloco(a, b, z2 + 0.14, z3, 0.0, 0.07, CAN)
        # sineira: arco de volta perfeita, alto e estreito (fotografias), com impostas
        wv = 1.75
        hn = (z3 - z2) - wv / 2 - 1.1
        za = z2 + 0.14
        zn = za + 0.6 + hn
        # paredes da sineira com o arco vazado (vê-se o sino): dois pilares, peitoril e tímpano em arco
        for sg in (-1, 1):
            a, b = sorted((sg * wv / 2, sg * (h + 0.001)))
            Qf.bloco(a, b, za, z3, -0.5, 0.0, CAN)
        Qf.bloco(-wv / 2, wv / 2, za, za + 0.6, -0.5, 0.0, CAN)
        ta = [(wv / 2 * math.cos(math.pi * k / 12), zn + wv / 2 * math.sin(math.pi * k / 12)) for k in range(13)]
        Qf.placa(ta + [(-wv / 2, z3), (wv / 2, z3)], -0.5, 0.0, CAN)
        Qf.arco_sz(0.0, wv / 2, wv / 2 + 0.28, za + 0.6 + hn, -0.02, 0.08, CAN, k=12, aduelas=True)
        for sg in (-1, 1):
            a, b = sorted((sg * wv / 2, sg * (wv / 2 + 0.28)))
            Qf.bloco(a, b, za + 0.6, za + 0.6 + hn, -0.02, 0.08, CAN)
        for sg in (-1, 1):  # impostas, só sobre os pilares (o vão fica livre)
            a, b = sorted((sg * (wv / 2 - 0.08), sg * (wv / 2 + 0.38)))
            Qf.bloco(a, b, zn - 0.16, zn, -0.5, 0.12, CAN)
        Qf.bloco(-wv / 2 - 0.1, wv / 2 + 0.1, za + 0.45, za + 0.62, 0.0, 0.14, CAN)  # peitoril
    sino(m, c + u * 0.0 + w * 0.0, z2 + 1.25, r=0.45)
    Qs = Quadro(m, c, u)
    Qs.bloco(-0.05, 0.05, z2 + 2.1, z3 - 0.5, -h + 0.5, h - 0.5, MAD)  # jugo
    # cornija de remate em três ressaltos
    m.caixa(c, u, lado + 0.25, lado + 0.25, z3, z3 + 0.2, CAN)
    m.caixa(c, u, lado + 0.5, lado + 0.5, z3 + 0.2, z3 + 0.38, CAN)
    m.caixa(c, u, lado + 0.7, lado + 0.7, z3 + 0.38, z4, CAN)
    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        cc = c + (u * a + w * b) * (h + 0.05)
        pinaculo(Quadro(m, cc, u), 0, z4, 0, esc=1.25, tipo="bola")
    # remate ao meio (fotografias): base em degraus, pirâmide baixa com bola e cata-vento de ferro
    m.caixa(c, u, 1.6, 1.6, z4, z4 + 0.35, CAN)
    m.caixa(c, u, 1.15, 1.15, z4 + 0.35, z4 + 0.75, CAN)
    piramide(m, c, u, 1.0, z4 + 0.75, z4 + 1.45, CAN, lado1=0.3)
    m.esfera((c[0], c[1], z4 + 1.62), 0.2, CAN, k=10)
    barra(m, (c[0], c[1], z4 + 1.75), (c[0], c[1], z4 + 3.4), 0.03, FER)
    barra(m, (*(c - w * 0.3), z4 + 2.9), (*(c + w * 0.3), z4 + 2.9), 0.025, FER)
    barra(m, (*(c - u * 0.45), z4 + 2.55), (*(c + u * 0.45), z4 + 2.55), 0.025, FER)  # seta do cata-vento
    Qs.placa([(0.3, z4 + 2.45), (0.55, z4 + 2.55), (0.3, z4 + 2.65)], -0.012, 0.012, FER)
    return z4 + 3.3


def igreja_matriz(m):
    """Igreja Matriz de Monção, de Santa Maria dos Anjos (fotografia da fachada): românica, toda de
    granito aparelhado; fachada a poente em empena com cruz, portal de quatro arquivoltas sobre colunelos
    e porta verde, um óculo polilobado de cada lado do portal, óculo com grade ao alto; torre sineira a
    sul, encostada à fachada; nave com cachorrada e capela-mor mais baixa a nascente. Pegada do OSM,
    fachada, cumeeiras e torre no MDS (adro a 34,6 m, cumeeira da nave a c. 46 m, torre a 53,3 m)."""
    g = 34.6
    n = np.array([-0.946, 0.325])          # para fora da fachada (poente)
    t = np.array([n[1], -n[0]])            # ao longo da fachada, para norte
    p0 = np.array([7.5, 108.0])            # ponta sul da fachada no MDS
    W = 11.0
    cf = p0 + t * (4.9 + W / 2)
    Q = Quadro(m, cf, n)
    ze = 41.6
    # empena lisa, sem modilhões (fotografia); a cachorrada fica só nas cornijas laterais
    zr = corpo(Q, 22.0, W, g, ze, inc=37, mw=CAN, cachorros=True, cachorros_empena=False)
    Qc = Quadro(m, cf - n * 22.0, n)
    corpo(Qc, 8.5, 7.6, g, 40.2, inc=33, mw=CAN, cachorros=True, cornija_frente=False, cachorros_empena=False)
    portal_arquivoltas(Q, 0.0, g, 1.6, 2.55, n=4, passo=0.22, prof=0.62, saliente=False, timpano=False)
    for sg in (-1, 1):
        oculo_lobado(Q, sg * 2.75, g + 2.55, 0.42)
    # rosácea grande (c. 2,2 m com a moldura, na fotografia), com anéis moldurados
    rosacea(Q, 0.0, g + 6.9, 0.8, raios=6)
    Q.arco_sz(0.0, 1.12, 1.24, g + 6.9, 0.0, 0.1, CAN, a0=0, a1=2 * math.pi, k=28)
    cruz(Q, 0.0, zr + 0.25, 0.0, alt=1.5, mat=CAN, esp=0.18)
    # frestas da nave (lado norte, para o largo)
    Qn = Quadro(m, cf + t * (W / 2) - n * 11.0, t)
    for s_ in (-6.0, 0.0, 6.0):
        janela(Qn, s_, g + 4.0, 0.3, 1.2, arco=True, mat_vidro=SOM, moldura=0.12)
    # capela a norte (São Sebastião): corpo mais baixo encostado à nave
    Qa = Quadro(m, cf + t * (W / 2 + 3.4) - n * 2.0, n)
    corpo(Qa, 9.0, 6.8, g, 39.6, inc=30, mw=CAN, cornija_frente=True)
    porta_igreja(Qa, 0.0, g, 1.3, 2.2, arco=False, moldura=0.25)
    cruz(Qa, 0.0, 39.6 + 3.4 * math.tan(_ang(30)) + 0.3, 0.0, alt=0.9, mat=CAN, esp=0.13)
    # torre a sul, alinhada com a fachada
    ct = p0 + t * 2.35 - n * 1.75  # no MDS a torre sai c. 0,6 m da fachada
    topo = torre_matriz(m, ct, n, 4.7, g)
    MEDIDAS["igreja_matriz"] = {"adro": g, "cumeeira": round(zr, 1), "torre": round(topo, 1)}
    return [list(p0 + n * 7.0 - t * 1.0), list(p0 + t * 21.5 + n * 7.0), list(p0 + t * 21.5 - n * 32.0), list(p0 - t * 1.0 - n * 32.0)]


def igreja_misericordia(m):
    """Igreja da Misericórdia (Street View): a fachada é a NORTE, sobre o largo (no MDS o frontão mais alto,
    47,7 m, fica no topo norte e a capela-mor, mais baixa, a sul). Fachada caiada entre cunhais de granito:
    portal de arco pleno com porta verde, em moldura de pilastras e entablamento; por cima, nicho com a
    imagem entre volutas; à direita um óculo gradeado; friso e cornija de granito; frontão triangular caiado
    com o relógio, cruz no vértice e bolas nos cantos. À direita, a casa da Misericórdia (dois pisos, janelas
    de sacada com frontão e varandas de ferro) e a sineira de um sino sobre o encontro das duas. Medidas no
    MDS: largo a 33,3 m, cornija a 43,0 m, cumeeira da nave a 45,2 m."""
    g = 33.3
    n = np.array([-0.37, 0.93])
    cf = np.array([-79.0, 39.2])
    W, L = 11.0, 22.0
    Q = Quadro(m, cf, n)
    t = Q.t  # para a esquerda de quem olha a fachada (para nascente)
    ze = 43.0
    zr = corpo(Q, L, W, g, ze, inc=22, cornija_frente=False, cunhais=False)
    corpo(Q, 10.0, 8.2, g, 41.6, inc=24, d_ini=-L + 0.3, cornija_frente=True)  # capela-mor, a sul
    # cunhais e soco de granito
    for s0_, larg in ((W / 2, 0.9), (-W / 2, 1.0)):
        a_, b_ = sorted((s0_, s0_ - np.sign(s0_) * larg))
        Q.bloco(a_, b_, g - 1.0, ze - 0.62, 0.0, 0.16, CAN)
    Q.bloco(-W / 2 - 0.05, W / 2 + 0.05, g - 1.0, g + 0.55, 0.0, 0.2, CAN)
    # friso e cornija
    Q.bloco(-W / 2 - 0.05, W / 2 + 0.05, ze - 0.62, ze - 0.12, 0.0, 0.2, CAN)
    Q.bloco(-W / 2 - 0.2, W / 2 + 0.2, ze - 0.12, ze + 0.18, 0.0, 0.42, CAN)
    # frontão triangular caiado, com cornija inclinada de granito, até 47,6 m
    zt = 47.4
    Q.placa([(-W / 2, ze + 0.18), (W / 2, ze + 0.18), (0.0, zt)], -0.6, 0.02, CAL)
    faixa_inclinada(Q, [(-W / 2 - 0.25, ze + 0.12), (0.0, zt + 0.05), (W / 2 + 0.25, ze + 0.12)], 0.4, -0.6, 0.4, CAN)
    relogio(Q, 0.0, ze + 2.5, 0.55, d=0.02)
    Q.arco_sz(0.0, 0.67, 0.85, ze + 2.5, 0.0, 0.12, CAN, a0=0, a1=2 * math.pi, k=20)
    for sg in (-1, 1):
        pinaculo(Q, sg * (W / 2 - 0.35), ze + 0.18, 0.1, esc=1.0, tipo="bola")
    Q.bloco(-0.35, 0.35, zt, zt + 0.45, -0.25, 0.25, CAN)
    cruz(Q, 0.0, zt + 0.45, 0.0, alt=1.5, mat=CAN, esp=0.16)
    # portal: pilastras, entablamento, arco pleno com porta verde e degrau
    sp = 0.9
    porta_igreja(Q, sp, g + 0.15, 2.2, 2.7, moldura=0.28, arco=True)
    for sg in (-1, 1):
        a_, b_ = sorted((sp + sg * 1.45, sp + sg * 1.95))
        Q.bloco(a_, b_, g, g + 5.3, 0.0, 0.18, CAN)
        Q.bloco(a_ - 0.06, b_ + 0.06, g, g + 0.5, 0.0, 0.24, CAN)
    Q.bloco(sp - 2.05, sp + 2.05, g + 5.3, g + 5.75, 0.0, 0.26, CAN)
    Q.bloco(sp - 2.15, sp + 2.15, g + 5.75, g + 5.9, 0.0, 0.34, CAN)
    Q.bloco(sp - 1.7, sp + 1.7, g - 0.2, g + 0.15, 0.0, 0.9, CAN)  # degrau
    # nicho com a imagem, entre volutas, com remate
    zn0 = g + 6.2
    Q.bloco(sp - 0.75, sp + 0.75, zn0 - 0.2, zn0, 0.0, 0.35, CAN)  # peanha
    _vao_desloc(Q, sp, 1.0, zn0, 1.7, 0.01, SOM)
    Q.arco_sz(sp, 0.5, 0.72, zn0 + 1.7, 0.0, 0.14, CAN, k=10)
    for sg in (-1, 1):
        a_, b_ = sorted((sp + sg * 0.5, sp + sg * 0.72))
        Q.bloco(a_, b_, zn0, zn0 + 1.7, 0.0, 0.14, CAN)
        # voluta: enrolamento em espiral achatada
        Q.arco_sz(sp + sg * 1.05, 0.12, 0.3, zn0 + 0.35, 0.0, 0.16, CAN, a0=0, a1=2 * math.pi, k=10)
        pts = [(sp + sg * (0.72 + 0.6 * f), zn0 + 2.3 - 1.7 * f - 0.25 * math.sin(math.pi * f)) for f in np.linspace(0, 1, 7)]
        Q.placa(pts + [(x_, z_ - 0.2) for x_, z_ in pts[::-1]], 0.0, 0.14, CAN)
    Q.m.cilindro(Q.p(sp, 0, -0.15)[:2], 0.2, zn0 + 0.02, zn0 + 1.05, LIS, k=10, r1=0.14)  # a imagem
    Q.esfera(sp, zn0 + 1.22, -0.15, 0.13, LIS)
    Q.bloco(sp - 0.95, sp + 0.95, zn0 + 2.45, zn0 + 2.65, 0.0, 0.25, CAN)
    Q.placa([(sp - 0.85, zn0 + 2.65), (sp + 0.85, zn0 + 2.65), (sp, zn0 + 3.05)], 0.0, 0.2, CAN)
    # óculo gradeado à direita
    so = -2.1
    zo = g + 8.4
    Q.m.face([Q.p(so + 0.55 * math.cos(2 * math.pi * i / 20), zo + 0.55 * math.sin(2 * math.pi * i / 20), 0.012) for i in range(20)], VID)
    Q.arco_sz(so, 0.55, 0.85, zo, 0.0, 0.14, CAN, a0=0, a1=2 * math.pi, k=20)
    for k_ in range(-2, 3):
        Q.bloco(so + k_ * 0.18 - 0.012, so + k_ * 0.18 + 0.012, zo - 0.5, zo + 0.5, 0.02, 0.04, FER)
        Q.bloco(so - 0.5, so + 0.5, zo + k_ * 0.18 - 0.012, zo + k_ * 0.18 + 0.012, 0.02, 0.04, FER)
    # casa da Misericórdia, à direita (poente): dois pisos caiados, cornija de granito, telhado de quatro águas
    s_a, s_b, P_ = -W / 2, -W / 2 - 15.0, 10.0
    za = 41.6
    Qa = Quadro(m, cf + t * ((s_a + s_b) / 2) - n * 0.25, n)
    Wa = abs(s_b - s_a)
    corpo(Qa, P_, Wa, g, za, inc=28, cornija_frente=False, cunhais=True)
    Qa.bloco(-Wa / 2 - 0.05, Wa / 2 + 0.05, za - 0.35, za + 0.05, 0.0, 0.3, CAN)
    Qa.bloco(-Wa / 2, Wa / 2, g + 3.65, g + 3.85, 0.0, 0.12, CAN)  # friso entre pisos
    for k_, sx in enumerate((-5.0, -0.8, 3.4)):
        sx_ = sx + 0.4
        # rés-do-chão: porta verde ou janela, com moldura
        if k_ == 1:
            porta_igreja(Qa, sx_, g, 1.3, 2.5, moldura=0.18, arco=False)
        else:
            janela(Qa, sx_, g + 1.0, 1.0, 1.4, arco=False, moldura=0.16)
        # andar: janela de sacada com frontão e varanda de ferro
        janela(Qa, sx_, g + 4.3, 1.2, 2.1, arco=False, moldura=0.18)
        Qa.placa([(sx_ - 0.95, g + 6.62), (sx_ + 0.95, g + 6.62), (sx_, g + 7.15)], 0.0, 0.16, CAN)
        Qa.bloco(sx_ - 0.95, sx_ + 0.95, g + 4.15, g + 4.3, 0.0, 0.55, CAN)
        for b_ in np.arange(sx_ - 0.85, sx_ + 0.86, 0.17):
            Qa.bloco(b_ - 0.012, b_ + 0.012, g + 4.3, g + 5.25, 0.47, 0.5, FER)
        Qa.bloco(sx_ - 0.9, sx_ + 0.9, g + 5.25, g + 5.3, 0.45, 0.53, FER)
    # sineira sobre o encontro: duas pilastras, arco com o sino, cornija, bolas e cruz
    ss_ = -W / 2 - 1.2
    zs = za + 0.05
    for sg in (-1, 1):
        a_, b_ = sorted((ss_ + sg * 0.55, ss_ + sg * 0.95))
        Q.bloco(a_, b_, zs, zs + 1.9, -0.6, 0.1, CAN)
    Q.arco_sz(ss_, 0.55, 0.95, zs + 1.3, -0.6, 0.1, CAN, k=8)
    Q.bloco(ss_ - 1.05, ss_ + 1.05, zs + 1.9, zs + 2.15, -0.7, 0.2, CAN)
    sino(m, Q.p(ss_, 0, -0.25)[:2], zs + 0.75, r=0.3)
    for sg in (-1, 1):
        Q.esfera(ss_ + sg * 0.8, zs + 2.4, -0.25, 0.17, CAN)
    Q.bloco(ss_ - 0.18, ss_ + 0.18, zs + 2.15, zs + 2.5, -0.43, -0.07, CAN)
    cruz(Q, ss_, zs + 2.5, -0.25, alt=0.9, mat=CAN, esp=0.12)
    MEDIDAS["igreja_misericordia"] = {"largo": g, "cumeeira": round(zr, 1), "frontao": zt}
    q0 = cf + t * (W / 2 + 0.5) + n * 3.0
    q1 = cf + t * (s_b - 0.5) + n * 3.0
    return [list(q0), list(q1), list(q1 - n * (P_ + 3.5)), list(cf + t * (s_a - 0.3) - n * (P_ + 3.5)),
            list(cf + t * (s_a - 0.3) - n * (L + 11.0)), list(cf + t * (W / 2 + 0.5) - n * (L + 11.0))]


IGREJAS = [
    ("Igreja_Matriz", "igreja_matriz", igreja_matriz),
    ("Igreja_Misericordia", "igreja_misericordia", igreja_misericordia),
    ("Ponte_EN202", "ponte_en202", ponte_en202),
]


def igrejas():
    for nome, el, fn in IGREJAS:
        m = Malha()
        planta = fn(m)
        m.objeto(nome, el, MAT_IGR, GRUPOS["Igrejas"] if el != "ponte_en202" else GRUPOS["Portas"], uv=2.5)
        EVITAR.append(np.array(planta, float))
        MEDIDAS.setdefault(el, {})["faces"] = len(m.f)
        print(f"  {nome}: {len(m.f)} faces")


# ================================================================ cena

bpy.ops.wm.read_factory_settings(use_empty=True)
RAIZ_OB = bpy.data.objects.new("Fortaleza_Moncao", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = LAT0, LON0
RAIZ_OB["centro_m"] = [float(CENTRO[0]), float(CENTRO[1])]
RAIZ_OB["cota_zero"] = ZREF
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
for g in ["Terreno", "Muralhas", "Guaritas", "Portas", "Igrejas", "Vila", "Vegetacao"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

TEX_ALV, ALT_ALV = textura_alvenaria()
NRM_ALV = imagem("alvenaria_normal", normais(ALT_ALV, 2.5))
material("alvenaria", (1, 1, 1), 0.93, imagem("alvenaria", TEX_ALV), normal=NRM_ALV, escala=0.5)  # textura de 8 m nos UV de 4 m
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
material("rio", (1, 1, 1), 0.1, None, vcor=True, normal=imagem("ondas_normal", normais(textura_ondas(), 1.2)), forca=0.3)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)

MEDIDAS = {}

print("escarpas…")
m = Malha()
mq = Malha()
LINHAS_ESC = []
def cortar(e):
    """Abre as peças no vão das portas fixas: os segmentos que atravessam a passagem (largura da porta
    para cada lado do eixo, até 9 m atrás e 2 m à frente do ponto da porta) são partidos nas bordas e o
    troço de dentro sai; devolve a lista de troços que sobram."""
    chaves = ("topo", "pe", "aresta", "fim")
    pecas = [e]
    for d_ in PORTAS.values():
        if not d_.get("fixo"):
            continue
        n_ = unit(np.array(d_["normal"], float))
        t_ = np.array([-n_[1], n_[0]])
        p0 = np.array(d_["ponto"])
        lg = d_.get("largura", 1.9)
        novas = []
        for p in pecas:
            lin = np.array(p["linha"], float)
            vals = {k: np.array(p[k], float) for k in chaves}
            s = (lin - p0) @ t_
            dd = (lin - p0) @ n_
            # densifica: insere os pontos onde cada segmento cruza s = ±lg
            L, V = [lin[0]], {k: [vals[k][0]] for k in chaves}
            for i_ in range(1, len(lin)):
                fs = sorted((b_ - s[i_ - 1]) / (s[i_] - s[i_ - 1]) for b_ in (-lg, lg)
                            if (s[i_ - 1] - b_) * (s[i_] - b_) < 0)
                for f in fs:
                    if True:
                        L.append(lin[i_ - 1] + (lin[i_] - lin[i_ - 1]) * f)
                        for k in chaves:
                            V[k].append(vals[k][i_ - 1] + (vals[k][i_] - vals[k][i_ - 1]) * f)
                L.append(lin[i_])
                for k in chaves:
                    V[k].append(vals[k][i_])
            L = np.array(L)
            s = (L - p0) @ t_
            dd = (L - p0) @ n_
            dentro_ = (np.abs(s) < lg - 1e-6) & (dd > -9.0) & (dd < 2.0)
            # um segmento é de dentro se o seu ponto médio cai na passagem
            meio = (L[1:] + L[:-1]) / 2
            sm, dm = (meio - p0) @ t_, (meio - p0) @ n_
            seg_dentro = (np.abs(sm) < lg) & (dm > -9.0) & (dm < 2.0)
            if not seg_dentro.any():
                novas.append(p)
                continue
            ini = 0
            for i_ in range(len(seg_dentro) + 1):
                if i_ == len(seg_dentro) or seg_dentro[i_]:
                    fim_ = i_  # troço de vértices ini..fim_
                    if fim_ - ini >= 1:
                        q = {**p, "linha": L[ini:fim_ + 1].tolist(),
                             **{k: list(map(float, V[k][ini:fim_ + 1])) for k in chaves}}
                        q["id"] = p["id"] + ("" if ini == 0 else f"_{ini}")
                        novas.append(q)
                    ini = i_ + 1
        pecas = novas
    return pecas


for e in ESC:
    LINHAS_ESC.append(escarpa(e, m, mq))
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
# passadiço de madeira em ziguezague na encosta norte, entre a vila e o rio (ortofotomapa da DGT no WebSIG
# da Câmara; traçado dos caminhos pedonais do OSM): tabuleiro de 1,8 m sobre o terreno, com guardas de madeira
m = Malha()
n_pass = 0.0
for f in F["caminhos"]:
    if f["classe"] not in ("footway", "path", "steps") or f["geometria"]["type"] != "LineString":
        continue
    ln_ = shapely.LineString(geom(f))
    if ln_.length < 5 or ln_.intersection(ENCOSTA_N).length < 0.6 * ln_.length:
        continue
    xs_ = np.array(ln_.coords)[:, 0]
    if xs_.min() < -125 or xs_.max() > 105:  # os caminhos de terra ao longo do rio, que seguem para fora, não são passadiço
        continue
    q_ = densificar(np.array(ln_.coords), 1.0)
    z_ = mediana_movel(np.array([float(cota_t(*p_)) for p_ in q_]), 5) + 0.35
    for i in range(len(q_) - 1):
        a_, b_ = q_[i], q_[i + 1]
        u_ = unit(b_ - a_)
        w_ = np.array([-u_[1], u_[0]]) * 0.9
        za, zb = z_[i], z_[i + 1]
        m.face([(*(a_ - w_), za), (*(b_ - w_), zb), (*(b_ + w_), zb), (*(a_ + w_), za)], 0)        # tabuleiro
        for sg in (-1, 1):
            m.face([(*(a_ + sg * w_), za), (*(b_ + sg * w_), zb), (*(b_ + sg * w_), zb - 0.25), (*(a_ + sg * w_), za - 0.25)], 0)
            barra(m, (*(a_ + sg * w_), za), (*(a_ + sg * w_), za + 1.0), 0.05, 0)                    # prumos
            barra(m, (*(a_ + sg * w_), za + 1.0), (*(b_ + sg * w_), zb + 1.0), 0.04, 0)              # corrimão
            barra(m, (*(a_ + sg * w_), za + 0.5), (*(b_ + sg * w_), zb + 0.5), 0.03, 0)
    n_pass += ln_.length
m.objeto("Passadico", "caminhos", ["madeira"], GRUPOS["Vila"], uv=2.0)
MEDIDAS["passadico"] = round(n_pass, 1)

_tt = textura_terreno()
if "--vistas" in sys.argv:
    from PIL import Image as _Im
    _Im.fromarray((_tt * 255).astype(np.uint8)).save(AQUI / "saida" / "chao.jpg", quality=85)
img_t = imagem("chao", _tt[::-1])  # o Blender guarda a linha 0 em baixo (sul)
mt = MATS["terreno"]
ti = mt.node_tree.nodes.new("ShaderNodeTexImage")
ti.image = img_t
ti.extension = "EXTEND"
mt.node_tree.links.new(ti.outputs["Color"], mt.node_tree.nodes["Principled BSDF"].inputs["Base Color"])

PROPRIOS = set()  # plantas do OSM que são modeladas à parte (igrejas, capelas)
EVITAR = []       # plantas onde não vão casas nem árvores
if "portas" in globals():
    print("portas…")
    portas()
if GUAR_EXTRA:
    m = Malha()
    for c, bis, zc, topo in GUAR_EXTRA:
        guarita(m, c, bis, zc, topo)
    m.objeto("Guarita_Rosal", "porta_rosal", ["cantaria_lisa", "cantaria", "sombra"], GRUPOS["Guaritas"], uv=2.0)
print("rio…")
construir_rio()
print("igrejas e ponte…")
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
    if FEIRA.contains(shapely.Point(poly.mean(0))):
        continue  # tendas da feira
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
OBRA = Obra()  # nenhuma copa entra em muros, casas ou igrejas (auditoria de novembro de 2026: 31 entravam)
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
    aj = OBRA.ajustar(x, y, z, h, r, alt, 2.2, tronco_min=0.8)
    if aj is None:
        continue
    r, alt, tronco_ = aj
    plantar(x, y, z, r, alt, especie, tronco=tronco_)
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
MEDIDAS["vegetacao"] = {"arvores": n_arv, "encolhidas": OBRA.encolhidas, "retiradas": OBRA.retiradas}


def ancora(x, y, acima=2.0):
    """Ponto de uma placa no visor (x este, y cota, z sul)."""
    return [round(float(x), 1), round(float(cota_t(x, y)) + acima, 1), round(-float(y), 1)]


MEDIDAS.setdefault("ancoras", {})
# pontos das placas (x este, y cota, z sul): portas à frente da frontaria, igrejas no alto da torre ou da
# empena, muralhas num baluarte da Coroada, guaritas na guarita mais próxima dele, vila no centro
for pid_ in ("porta_rosal", "porta_salvaterra"):
    if pid_ in PONTOS_PORTA:
        F0_, n_, sole_ = PONTOS_PORTA[pid_]
        MEDIDAS["ancoras"][pid_] = [round(float(F0_[0] + n_[0] * 2), 1), round(float(sole_ + ZREF + 12), 1), round(-float(F0_[1] + n_[1] * 2), 1)]
for el_, (x_, y_, z_) in {"igreja_matriz": (10.5, 109.5, 56.0), "igreja_misericordia": (-78.5, 37.5, 50.0),
                         "ponte_en202": (-62.0, -129.0, 40.0)}.items():
    MEDIDAS["ancoras"][el_] = [x_, z_, -y_]
MEDIDAS["ancoras"]["escarpas"] = ancora(200, -230, 14.0)
if GUARITAS:
    g_ = min(GUARITAS, key=lambda g: np.hypot(g[0][0] - 330, g[0][1] + 230))[0]
    MEDIDAS["ancoras"]["guaritas"] = [round(float(g_[0]), 1), round(float(cota_t(*g_)) + 9, 1), round(-float(g_[1]), 1)]
MEDIDAS["ancoras"]["rio"] = [150.0, 10.0, -275.0]
MEDIDAS["ancoras"]["vila"] = ancora(-60, 60, 18.0)
(AQUI / "saida").mkdir(exist_ok=True)
(AQUI / "dados" / "medidas.json").write_text(json.dumps(_arred(MEDIDAS), ensure_ascii=False, indent=1))
print(json.dumps(_arred(MEDIDAS), ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
glb = AQUI / "saida" / "moncao-bruto.glb"
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
    final = SAIDA / "moncao.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "moncao.blend"))

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

    def vista(nome, azimute, elev, dist, lente=50, orto=None, res=(1200, 800), destino=None, alvo_xy=(0, 0), alvo_z=None,
              perto=0.1):
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
        cam.clip_start = perto  # >0,1 para ver através das casas entre a câmara e o monumento
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.render.filepath = str(destino or (pasta / f"{nome}.jpg"))
        bpy.ops.render.render(write_still=True)
        print("vista", nome)

    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 84
    VISTAS_EXTRA = list(globals().get("VISTAS_EXTRA", []))
    SUF = next((a_.split("=")[1] for a_ in sys.argv if a_.startswith("--sufixo=")), "")
    for nm_, az_, el_, di_, xy_, zz_ in [  # vistas das igrejas e da ponte, do lado das fotografias
        ("i_matriz_alto", 250, 35, 60, (20, 112), 40),
        ("i_miser", 338, 10, 36, (-82, 37), 38), ("f_passadico", 340, 30, 90, (-10, 185), 25), ("i_miser_alto", 330, 35, 60, (-78, 30), 38),
        ("f_ponte", 180, 6, 40, (-62, -129), 31), ("f_ponte_n", 0, 10, 45, (-62, -129), 31),
        ("f_rosal", 212, 8, 26, (-353, 83), 34), ("f_rosal_dentro", 30, 30, 32, (-343, 97), 36), ("f_rosal_alto", 30, 50, 70, (-345, 92), 36)]:
        VISTAS_EXTRA.append(((nm_ + SUF, az_, el_, di_), dict(lente=35, alvo_xy=xy_, alvo_z=zz_)))
    # a Matriz abre para uma rua estreita (c. 8 m, ortofotomapa): vista de perto e grande angular, como a fotografia
    # (a câmara fica dentro das casas do outro lado da rua: perto=4 corta-as, como se a fotografia fosse da rua)
    VISTAS_EXTRA.append((("i_matriz" + SUF, 289, 7, 11.0), dict(lente=15, alvo_xy=(9.5, 116.5), alvo_z=40.5, perto=4.5)))
    for pid_, (F0_, n_, sole_) in PONTOS_PORTA.items():  # uma vista de frente, um pouco de lado, para cada porta
        az_ = math.degrees(math.atan2(n_[0], n_[1])) + 25
        VISTAS_EXTRA.append(((f"p_{pid_}{SUF}", az_, 22, 40), dict(lente=35, alvo_xy=tuple(F0_), alvo_z=sole_ + 4)))
    vista("capa", 18, 30, 1250, lente=50, res=(1280, 800), destino=SAIDA / "capa.jpg", alvo_xy=(-40, 30), alvo_z=30)  # do lado do rio
    if "--capa" not in sys.argv:
        vista("topo", 180, 89.9, 1500, orto=960, res=(1300, 800))
        vista("aereo_sul", 185, 40, 800, lente=50, alvo_xy=(-40, 20))      # como a 1.ª fotografia aérea
        vista("aereo_norte", 350, 38, 800, lente=50, alvo_xy=(-60, 60))    # do lado do rio, como a 2.ª
        vista("poente", 270, 28, 800, alvo_xy=(-150, 50))
        vista("nascente", 90, 28, 800, alvo_xy=(150, -50))
        for v_ in VISTAS_EXTRA:
            vista(*v_[0], **v_[1])
