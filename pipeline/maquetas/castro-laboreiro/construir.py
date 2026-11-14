"""Constrói a maqueta 3D do Castelo de Castro Laboreiro (estado atual) em Blender e exporta-a para
public/maquetas/castro-laboreiro/castro-laboreiro.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely
    python pipeline/maquetas/castro-laboreiro/construir.py            # GLB
    python pipeline/maquetas/castro-laboreiro/construir.py --vistas   # GLB + capa + vistas de verificação

Dados: dados/fontes.json (recolher.py: OSM via Overture), dados/lidar.npz e dados/orto.jpg (lidar.py:
MDT, MDS, classes e cor da nuvem de pontos do LiDAR 2025 da DGT) e src/data/maquetas/castro-laboreiro.json,
com a leitura das fontes e o estado de cada elemento. A cerca está traçada no próprio LiDAR (o OSM tem só
um contorno grosseiro): os panos seguem a crista medida no MDS e as cotas e alturas medem-se no LiDAR. O que
ele não mede (espessuras, vãos, aduelas, o marco geodésico) vem da ficha do SIPA e das fotografias.
O castelo está em ruína: as muralhas não têm remate (SIPA), por isso não levam ameias nem parapeito.
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
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pipeline/, para o kit comum
from kit.geometria import unit, area, densificar, dentro, distancia, desvio, mediana_movel, arco, arco_quebrado, _arred  # noqa: E402
from kit.texturas import imagem  # noqa: E402
from kit.fotos import textura_foto, fontes_fotos  # noqa: E402
from kit.cena import arvore_modelo  # noqa: E402
from kit.arquitetura import piramide  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "castro-laboreiro.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "castro-laboreiro"

LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
# A maqueta é só o monte do castelo, com pouca envolvente (pedido do Tiago): círculo de 92 m à volta
# do centro do recinto, que apanha as duas pontas da cerca (c. 136 m de norte a sul) com folga.
CENTRO = np.array([0.0, 0.0])
RAIO = 92.0
ZREF = 1000.0  # cota que fica em z = 0
rng = np.random.default_rng(1144)
DESVIO_OSM = np.array([0.0, 0.0])


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
# Debaixo do arvoredo cerrado da encosta poente o MDT é interpolado entre poucos pontos de solo: tem picos
# e cortinas verticais que não existem (rugosidade dupla da do chão aberto). Aí suaviza-se o MDT (mediana
# 3 × 3 e gaussiana de 1 m), tanto mais quanto mais fechado o coberto; nas lajes e no recinto fica intacto.
from scipy import ndimage as _ndi  # noqa: E402

COBERTO = _ndi.uniform_filter(VEG.astype(float), 9)
_peso = np.clip((COBERTO - 0.2) / 0.45, 0, 1)
MDT = MDT + _peso * (_ndi.gaussian_filter(_ndi.median_filter(MDT, 3), 2.0) - MDT)


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


# ---------------------------------------------------------------- ruído e ortoimagem


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


def uv_mundo(ob, escala, caixa=False):
    """UV em metros do mundo. Com caixa=True (rocha), projeção pelo eixo dominante da normal, para o grão
    não esticar em riscas nas faces a pique."""
    me = ob.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    uv = me.uv_layers.active.data
    for poly in me.polygons:
        n = poly.normal
        ax = max(range(3), key=lambda k: abs(n[k]))
        for li in poly.loop_indices:
            co = ob.matrix_world @ me.vertices[me.loops[li].vertex_index].co
            if caixa:
                uv[li].uv = ((co.x, co.y) if ax == 2 else (co.y, co.z) if ax == 0 else (co.x, co.z))
                uv[li].uv = (uv[li].uv[0] / escala, uv[li].uv[1] / escala)
            elif abs(n.z) > 0.7:
                uv[li].uv = (co.x / escala, co.y / escala)
            else:  # faces verticais: u ao longo da face, para as fiadas não esticarem nos panos oblíquos
                h = math.hypot(n.x, n.y) or 1.0
                uv[li].uv = ((co.x * -n.y + co.y * n.x) / h / escala, co.z / escala)


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


# ---------------------------------------------------------------- materiais e vãos


def textura_grao(n=256):
    yy, xx = np.mgrid[0:n, 0:n] / n
    g = fbm(xx * 30, yy * 30, 2.0, 31)
    return np.repeat((0.84 + 0.22 * g)[..., None], 3, 2).clip(0, 1)


def material(nome, cor, rug=0.9, tex=None, vcor=False, normal=None, forca=1.0):
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


def aduelas(m, centro, u, normal, largura, soleira, nascenca, mat, quebrado=False, esp=0.5, saliencia=0.06, k=11):
    """Moldura de cantaria de um vão: k aduelas do arco (pleno ou quebrado) e ombreiras em fiadas
    alternadas, um pouco salientes da face (normal para fora)."""
    w = largura / 2
    zn = soleira + nascenca

    def ponto(r_extra, t):  # t de 0 (arranque direito) a 1 (arranque esquerdo)
        if quebrado:  # dois arcos de raio igual à largura, centrados nos arranques opostos
            r = largura + r_extra
            am = math.acos(w / r)  # os dois arcos encontram-se no eixo do vão
            if t <= 0.5:
                a = am * (t / 0.5)
                return (w - r * math.cos(a), zn + r * math.sin(a))
            a = am * ((1 - t) / 0.5)
            return (-w + r * math.cos(a), zn + r * math.sin(a))
        a = math.pi * (1 - t)
        return ((w + r_extra) * math.cos(a), zn + (w + r_extra) * math.sin(a))
    d = normal * saliencia
    for i in range(k):
        t0, t1 = i / k + 0.004, (i + 1) / k - 0.004  # junta fina entre aduelas
        a0, a1 = ponto(0, t0), ponto(0, t1)
        b0, b1 = ponto(esp, t0), ponto(esp, t1)
        P_ = lambda q: (*(centro + u * q[0] + d), q[1])
        m.face([P_(a0), P_(a1), P_(b1), P_(b0)][::-1], mat)
        for (q0, q1) in ((a0, a1), (b1, b0)):  # espessura da saliência
            m.face([(*(centro + u * q0[0]), q0[1]), (*(centro + u * q1[0]), q1[1]), P_(q1), P_(q0)], mat)
    zs = np.linspace(soleira, zn, max(2, int(round(nascenca / 0.45))) + 1)
    for j in range(len(zs) - 1):  # ombreiras: silhares alternadamente compridos e curtos
        e_ = esp if j % 2 == 0 else esp * 0.6
        for sg in (-1, 1):
            cc = centro + u * sg * (w + e_ / 2) + normal * (saliencia / 2)
            m.caixa(cc, u, e_, saliencia, zs[j] + 0.02, zs[j + 1] - 0.02, mat)


# ---------------------------------------------------------------- texturas


def textura_pedra(n=512):
    """Cantaria isódoma de granito (3 m por repetição): fiadas regulares de silhares, cinzentos com
    líquenes, como nas fotografias das muralhas de Castro Laboreiro."""
    r = np.random.default_rng(1290)
    tex = np.zeros((n, n, 3))
    alt = np.ones((n, n))  # relevo para o mapa de normais: juntas fundas, silhares abaulados
    m_px = n / 3.0
    y = 0
    while y < n:
        hf = int(r.uniform(0.38, 0.50) * m_px)
        x = -int(r.uniform(0, 0.6) * m_px)
        while x < n:
            wf = int(r.uniform(0.55, 1.05) * m_px)
            tom = r.choice(3, p=[0.5, 0.3, 0.2])
            base = np.array([(0.62, 0.62, 0.60), (0.66, 0.65, 0.61), (0.56, 0.56, 0.54)][tom]) * r.uniform(0.92, 1.06) + r.normal(0, 0.006, 3)
            x0, x1 = max(x, 0), min(x + wf, n)
            y1 = min(y + hf, n)
            if x1 > x0:
                tex[y:y1, x0:x1] = base
                tex[y:y1, x0:min(x0 + 3, x1)] *= 0.78
                alt[y:y1, x0:min(x0 + 4, x1)] = 0.0
                alt[y:y1, x0:x1] += r.uniform(-0.12, 0.12)  # faces dos silhares desencontradas
            x += wf
        tex[y:min(y + 3, n)] *= 0.76
        alt[y:min(y + 4, n)] = 0.0
        y += hf
    yy, xx = np.mgrid[0:n, 0:n] / n
    grao = fbm(xx * 40, yy * 40, 1.5, 21)[..., None]
    fase = r.uniform(0, 2 * math.pi, 6)
    manchas = sum(np.sin(2 * math.pi * (fx * xx + fy * yy) + ph) for (fx, fy), ph in zip(((3, 1), (5, 2), (2, 3), (7, 1), (4, 5), (1, 2)), fase))
    manchas = (0.5 + manchas / 6)[..., None]
    # líquenes: manchas claras e escuras miúdas, que a pedra da serra tem por toda a parte
    liq = fbm(xx * 24, yy * 24, 1.0, 41)[..., None]
    tex = tex * (0.92 + 0.12 * grao) * (0.90 + 0.12 * manchas)
    tex = tex * np.where(liq > 0.68, 1.12, 1.0) * np.where(liq < 0.30, 0.86, 1.0)
    PEDRA_ALT[:] = [ndi.gaussian_filter(alt, 1.6) + 0.25 * grao[..., 0]]
    return np.clip(tex, 0, 1)


PEDRA_ALT = []


def normais(alt, forca):
    """Mapa de normais (espaço tangente, convenção do glTF) a partir de um mapa de alturas periódico."""
    gx = (np.roll(alt, -1, 1) - np.roll(alt, 1, 1)) * forca
    gy = (np.roll(alt, -1, 0) - np.roll(alt, 1, 0)) * forca
    nv = np.stack([-gx, -gy, np.ones_like(alt)], 2)
    nv /= np.linalg.norm(nv, axis=2, keepdims=True)
    return nv * 0.5 + 0.5


def textura_granito(n=512):
    """Rocha granítica (4 m por repetição): grão grosso de feldspato e mica, líquenes e fissuras."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    g1 = fbm(xx * 64, yy * 64, 1.0, 51)
    g2 = fbm(xx * 10, yy * 10, 1.0, 52)
    liq = fbm(xx * 16, yy * 16, 1.0, 53)
    v = 0.86 + 0.18 * (g1 - 0.5) + 0.16 * (g2 - 0.5)
    v = v * np.where(liq > 0.66, 1.10, 1.0) * np.where(liq < 0.32, 0.84, 1.0)
    pint = np.random.default_rng(54).random((n, n))
    v = v * np.where(pint > 0.985, 0.6, 1.0) * np.where(pint < 0.01, 1.15, 1.0)  # cristais escuros e claros
    GRANITO_ALT[:] = [0.6 * g2 + 0.4 * g1]
    return np.repeat(v[..., None], 3, 2).clip(0, 1)


GRANITO_ALT = []


# ---------------------------------------------------------------- planta da cerca (traçada no LiDAR)

from scipy import ndimage as ndi  # noqa: E402

# relevo local do MDS (o que sobressai numa janela de 6 m): as muralhas e os penedos
LOC = MDS - ndi.grey_opening(MDS, size=(13, 13))
ARV_LIDAR = LID["arvores"]

# Panos de muralha, linha do meio de cada um (metros a partir da origem), desenhada sobre o relevo do
# MDS de 50 cm, no sentido em que o exterior fica à direita; o perfil() encosta-a depois à crista medida.
# Espessuras: no castelo, 1,7 m (c. 1,5 varas na legenda de Duarte de Armas, SIPA); na cerca, 2 m,
# estimados pelas fotografias.
PANOS = {
    # Cerca da vila (recinto sul): pano nascente, do reduto até ao castelo; troço ovalado a sul e
    # retilíneo a norte (SIPA)
    "cerca_nascente": dict(pts=[(-4.5, -68.5), (-1.5, -68.3), (0.0, -66.0), (3.0, -60.0), (7.0, -53.0), (12.0, -45.0),
                                (17.5, -35.0), (21.5, -25.0), (23.3, -15.0), (23.5, 0.0), (22.5, 15.0), (21.5, 22.5)],
                           esp=2.0, elemento="cerca"),
    # Frente poente da cerca: muito irregular, aproveita os afloramentos como defesa (SIPA). Traçado do
    # OSM, com troços baixos entre os penedos.
    "cerca_poente": dict(pts=[(-16.4, 21.4), (-17.1, 12.2), (-22.3, 4.9), (-23.4, 2.3), (-25.6, -2.9), (-27.8, -8.0),
                              (-30.0, -13.2), (-26.0, -21.1), (-20.6, -23.7), (-15.1, -26.3), (-15.6, -29.0), (-16.5, -34.4),
                              (-17.0, -43.0), (-15.5, -51.0), (-15.8, -61.0)],
                         esp=1.5, elemento="cerca", baixo=True),
    # Castelo (recinto norte, mais alto): planta retangular, com o ângulo nordeste curvo e a reentrância
    # do antigo cubelo (SIPA)
    "muro_recintos": dict(pts=[(-16.4, 21.4), (-13.0, 21.5), (-2.0, 21.5), (10.0, 21.6), (14.0, 21.7), (20.0, 22.3)],
                          esp=1.7, elemento="castelo"),
    "castelo_nascente": dict(pts=[(20.0, 22.3), (17.6, 26.0), (17.3, 33.0), (19.0, 36.0), (18.6, 45.0), (17.6, 55.0),
                                  (17.0, 59.0), (14.0, 62.0), (9.0, 63.5), (3.0, 64.5)],
                             esp=1.7, elemento="castelo"),
    "castelo_norte": dict(pts=[(3.0, 64.5), (-2.0, 64.8), (-7.0, 65.1), (-13.0, 65.2)], esp=1.7, elemento="castelo"),
    "castelo_poente": dict(pts=[(-13.0, 65.2), (-13.3, 55.0), (-13.6, 45.0), (-14.6, 35.0), (-15.6, 28.0), (-16.4, 21.4)],
                           esp=1.6, elemento="castelo", baixo=True),
}
# Reduto na ponta sul: plataforma retangular de muralha, cheia, à cota do topo do pano nascente
# (MDS: 1021,3 m), saliente para sul e poente; é a "frente sul estreita e retilínea" do SIPA.
REDUTO = np.array([(-16.3, -70.4), (-4.3, -70.4), (-4.3, -61.4), (-16.3, -61.4)])
# Vestígios da construção quadrangular junto ao muro norte da cerca, com acesso a norte (SIPA: talvez a
# capela de Nossa Senhora dos Remédios ou de Santa Bárbara), medidos no MDS
CAPELA = np.array([(7.0, 9.3), (12.6, 9.3), (12.6, 14.7), (7.0, 14.7)])
MARCO = np.array([3.0, 44.5])  # marco geodésico no ponto mais alto (pico estreito de 2 m no MDS)
# Portas (posição no OSM ou no SIPA e nas fotografias; tipo de arco pelo SIPA)
# Revisão do perito (13 nov. 2026): os vãos atravessam o pano de lado a lado (antes eram faces pretas), com
# intradorso de pedra, chão de saibro e o terreno aberto num corredor diante de cada boca (corredor: largura;
# fora/dentro: patamar à cota da soleira; rampa: inclinação depois do patamar). Aduelas grandes da pedra do pano
# (Sol 8, Recintos 11, Traição 9), sem impostas; Porta do Sol com 1,15 m de vão, pés-direitos de 1,8 m e arco
# redondo e baixo do lado de dentro (Commons: Castle of Castro Laboreiro 13, 14, 18; Castelo de Castro Laboreiro).
# Folhas de madeira: Sol entreaberta (verde-água), Traição aberta (verde-escuro), Fonte fechada (madeira velha).
PORTAS = {
    "porta_sol": dict(pano="cerca_nascente", ponto=(6.8, -54.9), arco="quebrado", arco_dentro="pleno", largura=1.15,
                      nascenca=1.8, aduelas=8, corredor=1.6, fora=1.5, dentro=1.5, rampa=0.18,
                      folhas=((0.31, 0.50, 0.48), 40)),
    "porta_fonte": dict(pano="cerca_nascente", depois_de=("porta_sol", 7.0), arco="reto", largura=1.1, nascenca=1.9,
                        corredor=1.4, fora=1.0, dentro=1.0, rampa=0.2, folhas=((0.46, 0.43, 0.38), 0)),
    "porta_recintos": dict(pano="muro_recintos", ponto=(-2.0, 22.1), arco="pleno", largura=1.4, nascenca=1.6, aduelas=11,
                           corredor=1.8, fora=1.0, dentro=1.0, rampa=0.2),
    "porta_traicao": dict(pano="castelo_norte", ponto=(-7.4, 65.5), arco="pleno", largura=1.6, nascenca=1.8, aduelas=9,
                          misulas=True, corredor=2.2, fora=0.6, dentro=4.0, rampa=0.25, folhas=((0.18, 0.31, 0.27), 75)),
}


def P(x, y):  # noqa: F811 (aqui a origem é o centro da maqueta)
    return np.array([x, y], dtype=float) - CENTRO


def ajustar_crista(linha, maximo=2.5):
    """Encosta a linha à crista do pano no relevo local do MDS: em cada vértice procura, na normal e até
    `maximo` metros, o ponto mais saliente; só desloca quando a crista se destaca (> 0,5 m) do que está
    sob a linha. As muralhas aparecem na nuvem de pontos ora como solo ora como vegetação, por isso aqui
    não se separam as classes. Os deslocamentos são filtrados ao longo do pano; as pontas ficam."""
    linha = densificar(linha, 1.0)
    nrm = desvio(linha, 1.0) - linha
    ds = np.arange(-maximo, maximo + 0.01, 0.25)
    xy = linha[:, None, :] + nrm[:, None, :] * ds[None, :, None]
    z = amostra(LOC, xy[..., 0], xy[..., 1]) - 0.12 * np.abs(ds)[None, :]
    k = np.argmax(z, axis=1)
    z0 = z[:, len(ds) // 2]
    d = np.where(z[np.arange(len(linha)), k] - z0 > 0.5, ds[k], 0.0)
    d[:2] = d[-2:] = 0.0
    d = mediana_movel(d, 7)
    d = np.convolve(np.pad(d, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    return linha + nrm * d[:, None], d


def perfil(linha, esp, baixo=False):
    """Mede um pano no LiDAR, vértice a vértice: topo no MDS (percentil 92 das células num raio de 1 m,
    filtrado ao longo do pano), pé exterior e chão interior no MDT. Devolve a linha densificada, topo,
    pé exterior e chão interior (z da cena)."""
    linha, desl = ajustar_crista(linha)
    if np.abs(desl).max() > 0.5:
        print(f"  linha ajustada ao MDS: deslocamento máx. {np.abs(desl).max():.1f} m, mediano {np.median(np.abs(desl)):.1f} m")
    topo = np.array([medir(centro=p, raio=1.0, q=92, sem_veg=False) for p in linha])
    topo = mediana_movel(topo, 5)
    fora = np.min([cota(*desvio(linha, -(esp / 2 + d)).T) for d in (1.0, 1.5, 2.0)], axis=0)
    dentro_ = np.median([cota(*desvio(linha, esp / 2 + d).T) for d in (1.0, 1.5, 2.0)], axis=0)
    chao = np.maximum(fora, dentro_)
    if baixo:  # troços baixos entre os penedos: o que o LiDAR mede acima disto é rocha
        topo = np.clip(topo, dentro_ + 0.9, dentro_ + 2.4)
    topo = np.clip(topo, chao + 0.8, chao + 7.0)  # copas encostadas ao pano não sobem o topo
    topo = mediana_movel(topo, 3)
    if baixo:  # o chão de dentro salta de penedo em penedo: o coroamento destes troços corre liso
        topo = np.convolve(np.pad(topo, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    return linha, topo, fora, dentro_


def medir(poly=None, centro=None, raio=None, q=95, sem_veg=True):  # noqa: F811
    v = celulas(poly, centro, raio, sem_veg)
    return float(np.percentile(v, q)) if len(v) >= 3 else float("nan")


def degraus(topo, salto=0.45):
    """Coroamento em degraus: onde o pano sobe ou desce, o topo fica em patamares horizontais com
    espelhos verticais, como nas fotografias (o adarve sobe aos lanços junto à Porta do Sol)."""
    k = min(9, len(topo) // 2 * 2 - 1)
    t_ = np.convolve(np.pad(topo, k // 2, mode="edge"), np.ones(k) / k, mode="valid") if k >= 3 else topo
    out = np.empty_like(topo)
    nivel, run = t_[0], 0
    for i, t in enumerate(t_):
        run += 1
        if abs(t - nivel) >= salto and run >= 3:  # patamares de 3 m pelo menos: nada de dentes
            nivel, run = t, 0
        out[i] = nivel
    return out


def parede(m, linha, esp, topo, fora, dentro_, mat=0, mat_topo=3, fechada=False, escalonar=True):
    """Pano de muralha arruinado ao longo da linha central (o exterior fica à direita): faces de cantaria
    até ao topo medido, sem parapeito nem ameias (SIPA: sem remate superior), e coroamento de lajes em
    patamares. Os panos descem 1,5 m abaixo do terreno para não ficarem no ar."""
    ext = desvio(linha, -esp / 2, fechada)
    inn = desvio(linha, esp / 2, fechada)
    n = len(linha)
    ze = np.minimum(fora, topo - 0.5) - 1.5
    zi = np.minimum(dentro_, topo - 0.5) - 1.5
    a = degraus(topo) if escalonar else topo
    # sequência de (vértice, cota do topo); num degrau o vértice repete-se com as duas cotas
    # sem escalonar, o coroamento corre em rampa de vértice a vértice (revisão do perito, P1-14: antes cada
    # vértice fazia um degrau e o topo dos troços baixos saía em escadinha)
    seq = []
    for i in range(n):
        if escalonar and seq and a[i] != seq[-1][1]:
            seq.append((i, seq[-1][1]))
        seq.append((i, a[i]))
    if fechada:
        if escalonar and a[0] != seq[-1][1]:
            seq.append((0, seq[-1][1]))
        seq.append((0, a[0]))
    for (i, ai), (j, aj) in zip(seq[:-1], seq[1:]):
        if i == j:  # espelho do degrau: só a face vertical do coroamento
            m.face([(*ext[i], ai), (*ext[i], aj), (*inn[i], aj), (*inn[i], ai)], mat)
            continue
        m.face([(*ext[i], ze[i]), (*ext[j], ze[j]), (*ext[j], aj), (*ext[i], ai)], mat)
        m.face([(*inn[j], zi[j]), (*inn[i], zi[i]), (*inn[i], ai), (*inn[j], aj)], mat)
        m.face([(*ext[i], ai), (*ext[j], aj), (*inn[j], aj), (*inn[i], ai)], mat_topo)
        m.face([(*ext[j], ze[j]), (*ext[i], ze[i]), (*inn[i], zi[i]), (*inn[j], zi[j])], mat)
    if not fechada:
        for i in (0, n - 1):
            m.face([(*ext[i], ze[i]), (*inn[i], zi[i]), (*inn[i], a[i]), (*ext[i], a[i])], mat)
    return ext, inn, a


def vao_reto(m, centro, u, normal, largura, soleira, altura, mat_escuro, mat_cant, escuro=True):
    """Porta de verga reta: vão escuro (se escuro), ombreiras de silhares alternados e verga inteiriça."""
    w = largura / 2
    if escuro:
        m.face([(*(centro - u * w + normal * 0.03), soleira), (*(centro + u * w + normal * 0.03), soleira),
            (*(centro + u * w + normal * 0.03), soleira + altura), (*(centro - u * w + normal * 0.03), soleira + altura)], mat_escuro)
    zs = np.linspace(soleira, soleira + altura, 5)
    for j in range(4):
        e_ = 0.45 if j % 2 == 0 else 0.3
        for sg in (-1, 1):
            m.caixa(centro + u * sg * (w + e_ / 2) + normal * 0.03, u, e_, 0.06, zs[j] + 0.02, zs[j + 1] - 0.02, mat_cant)
    m.caixa(centro + normal * 0.04, u, largura + 0.9, 0.08, soleira + altura, soleira + altura + 0.45, mat_cant)


def no_pano(pano, ponto):
    """Índice do vértice do pano (já medido) mais próximo de um ponto, tangente e normal exterior."""
    lin = PERFIS[pano][:, :2]
    i = int(np.argmin(np.hypot(*(lin - ponto).T)))
    i = min(max(i, 1), len(lin) - 2)
    t = unit(lin[i + 1] - lin[i - 1])
    return i, lin[i], t, np.array([t[1], -t[0]])


# ---------------------------------------------------------------- terreno e afloramentos


def rebaixar(X, Y, Z):
    """Debaixo das muralhas, do reduto e dos vestígios, o MDT apanha às vezes o topo dos panos (pontos
    classificados como solo): aí o terreno desce ao chão vizinho, para os volumes não ficarem enterrados."""
    from scipy.spatial import cKDTree
    Z = Z.copy()
    for s_, (lin, esp, topo, fora, dentro_) in MEDIDOS.items():
        baixo_ = np.minimum(fora, dentro_) - 0.2
        arv = cKDTree(lin)
        x0, y0 = lin.min(0) - esp
        x1, y1 = lin.max(0) + esp
        sel = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        d, _, _ = distancia(X[sel], Y[sel], lin)
        _, k = arv.query(np.stack([X[sel], Y[sel]], 1))
        perto = d < esp / 2 + 0.3
        if s_ in PENEDO:  # onde o muro se interrompe no penedo, a rocha fica como está
            perto &= ~PENEDO[s_][k]
        zz = Z[sel]
        zz[perto] = np.minimum(zz[perto], baixo_[k[perto]])
        Z[sel] = zz
    for poly, folga in ((CAPELA, 0.2),):
        dentro_r = dentro(X, Y, desvio(poly, -folga, True))
        if dentro_r.any():
            Z[dentro_r] = np.minimum(Z[dentro_r], float(np.percentile(Z[dentro_r], 10)) - 0.5)
    return Z


def rocha_ao_pe(X, Y, Z):
    """À frente dos panos assentes, o terreno sobe (só sobe) ao pé da face de fora e desce dali em rocha
    (TALUDE_ROCHA), com bordos contínuos; nos troços de penedo sobe até ao topo medido. Longe das portas."""
    Z = Z.copy()
    for s_, pe in PE_ROCHA.items():
        lin, esp, topo = MEDIDOS[s_][:3]
        pe = np.where(PENEDO[s_], topo, pe)
        ln = shapely.LineString(lin)
        ss = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(lin, axis=0), axis=1))])
        lado = ln.buffer(-10.0, single_sided=True)  # o lado de fora (à direita da linha)
        x0, y0, x1, y1 = lado.bounds
        sel = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        xs, ys = X[sel], Y[sel]
        pts = shapely.points(xs, ys)
        d = shapely.distance(ln, pts)
        alvo = np.interp(shapely.line_locate_point(ln, pts), ss, pe) - TALUDE_ROCHA * np.clip(d - (esp / 2 + 0.4), 0, None)
        ok = shapely.contains_xy(lado.buffer(0.05), xs, ys)
        for pid, (i, p, t, n_, sole) in PONTOS_PORTA.items():
            if PORTAS[pid]["pano"] == s_:
                ok &= np.hypot(xs - p[0], ys - p[1]) > 3.0
        zz = Z[sel]
        zz[ok] = np.maximum(zz[ok], alvo[ok])
        Z[sel] = zz
    return Z


def construir_terreno():
    passo = 0.5
    n = int(2 * RAIO / passo) + 1
    xs = np.linspace(-RAIO, RAIO, n)
    X, Y = np.meshgrid(xs, xs)
    Z0 = cota(X, Y)
    Z = abrir_portas(X, Y, rocha_ao_pe(X, Y, rebaixar(X, Y, Z0)))
    subida = Z > Z0 + 0.3  # rocha posta à frente dos panos assentes: pinta-se como afloramento
    gy, gx = np.gradient(Z0, passo)
    decl = np.degrees(np.arctan(np.hypot(gx, gy)))
    curv = ndi.laplace(ndi.gaussian_filter(Z0, 1.0))  # > 0 nas fissuras e nos pés dos penedos

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

    # Cores por vértice (sRGB), guiadas pela ortoimagem da nuvem de pontos: erva seca dourada no
    # recinto e nas encostas, mato verde-escuro, e o granito cinzento-claro onde ele aflora (ortoimagem
    # clara e pouco saturada, ou declive forte), com as fissuras mais escuras.
    xv, yv, zv = V[:, 0], V[:, 1], V[:, 2]
    dv = decl[usados]
    cv = curv[usados]
    orto = ortofoto(xv, yv)
    lum = orto.mean(1)
    sat = orto.max(1) - orto.min(1)
    verde = np.clip((orto[:, 1] - (orto[:, 0] + orto[:, 2]) / 2) * 9 + 0.25, 0, 1)
    ci = np.clip(np.round((xv + LMEIO) / LPASSO).astype(int), 0, VEG.shape[1] - 1)
    ri = np.clip(np.round((yv + LMEIO) / LPASSO).astype(int), 0, VEG.shape[0] - 1)
    sob = np.zeros(len(xv))
    for dr in (-2, 0, 2):
        for dc in (-2, 0, 2):
            sob += VEG[np.clip(ri + dr, 0, VEG.shape[0] - 1), np.clip(ci + dc, 0, VEG.shape[1] - 1)]
    sob = np.clip(sob / 9 * 1.5, 0, 1)
    verde = np.maximum(verde, sob)
    seca = np.array([0.76, 0.66, 0.43])
    erva = np.array([0.50, 0.53, 0.32])
    mato = np.array([0.33, 0.38, 0.24])
    folhada = np.array([0.40, 0.37, 0.25])  # chão do souto: folha de castanheiro e de sobreiro
    cob = amostra(COBERTO, xv, yv)
    cor = seca * (1 - verde[:, None]) + erva * verde[:, None]
    cor = cor * (1 - sob[:, None] * 0.6) + mato * sob[:, None] * 0.6
    fl = np.clip((cob - 0.35) / 0.4, 0, 1)[:, None] * 0.55
    cor = cor * (1 - fl) + folhada * fl
    # o ruído leva a cota, para as faces a pique não ficarem às riscas verticais
    cor = cor * (0.93 + 0.14 * fbm(xv + 0.6 * zv, yv - 0.4 * zv, 5.0, 3))[:, None]
    # Rocha: onde a ortoimagem é clara e pouco saturada (as lajes e os penedos), e nas faces quase a
    # pique fora do arvoredo; debaixo das árvores o declive sozinho não faz rocha (ali é souto).
    rocha = np.clip((lum - 0.52) * 5, 0, 1) * np.clip((0.17 - sat) * 9, 0, 1) * (1 - sob)
    rocha = np.maximum(rocha, np.clip((dv - 52) / 10, 0, 1) * np.clip(1 - cob * 2.5, 0, 1))
    rocha = np.maximum(rocha, subida[usados].astype(float))
    rocha = ndi_suave(rocha, usados, X.shape)
    gran = np.array([0.70, 0.69, 0.65]) * (0.88 + 0.2 * fbm(xv + 0.7 * zv, yv - 0.5 * zv, 3.0, 5))[:, None]
    gran = gran * (1 - np.clip(cv * 1.6, -0.12, 0.30))[:, None]  # fissuras escuras, lombas claras
    gran = gran * (0.9 + 0.1 * np.clip(lum / 0.75, 0, 1.2))[:, None]
    cor = cor * (1 - rocha[:, None]) + gran * rocha[:, None]

    rocha_f = rocha[faces].mean(1) > 0.5
    cor_lin = np.clip(cor, 0, 1) ** 2.2
    obs = {}
    for nome, sel, el, mat in (("Terreno", ~rocha_f, "terreno", "terreno"), ("Afloramentos", rocha_f, "afloramentos", "granito")):
        fs = faces[sel]
        usa, inv = np.unique(fs.ravel(), return_inverse=True)
        me = bpy.data.meshes.new(nome.lower())
        me.from_pydata(V[usa].tolist(), [], inv.reshape(-1, 4).tolist())
        me.update()
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        ca.data.foreach_set("color", np.hstack([cor_lin[usa], np.ones((len(usa), 1))]).astype(np.float32).ravel())
        me.color_attributes.active_color = ca
        me.shade_smooth()
        ob = novo_objeto(nome, me, el, mat=mat, pai=GRUPOS["Terreno"] if nome == "Terreno" else GRUPOS["Afloramentos"])
        uv_mundo(ob, 8.0 if nome == "Terreno" else 4.0, caixa=nome != "Terreno")
        obs[nome] = ob
    MEDIDAS["terreno"] = {"vertices": int(len(V)), "faces_rocha": int(rocha_f.sum()), "faces": int(len(faces))}

    # borda circular e base de maqueta de museu (saia e fundo)
    ang = np.linspace(0, 2 * math.pi, 361)[:-1]
    bxy = np.stack([np.cos(ang), np.sin(ang)], 1) * (RAIO - passo * 0.6)
    bz = cota(bxy[:, 0], bxy[:, 1])
    fundo = float(bz.min()) - 14.0
    k = len(bxy)
    anel = np.stack([bxy[:, 0], bxy[:, 1], bz], 1)
    verts = anel.tolist() + [[p[0], p[1], fundo] for p in anel]
    fs = [[i, (i + 1) % k, k + (i + 1) % k, k + i] for i in range(k)]
    fs.append(list(range(2 * k - 1, k - 1, -1)))
    mb = bpy.data.meshes.new("base")
    mb.from_pydata(verts, [], fs)
    mb.update()
    novo_objeto("Base", mb, None, mat="base", pai=GRUPOS["Terreno"])
    return Z


def ndi_suave(v, usados, forma):
    """Suaviza um valor por vértice na grelha do terreno (para a orla da rocha não ficar serrilhada)."""
    g = np.zeros(forma)
    g[usados] = v
    w = ndi.gaussian_filter(usados.astype(float), 0.8)
    g = ndi.gaussian_filter(g, 0.8) / np.maximum(w, 1e-6)
    return g[usados]


# ---------------------------------------------------------------- cena


bpy.ops.wm.read_factory_settings(use_empty=True)
RAIZ_OB = bpy.data.objects.new("Castelo_Castro_Laboreiro", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = LAT0, LON0
RAIZ_OB["centro_m"] = [float(CENTRO[0]), float(CENTRO[1])]
RAIZ_OB["cota_zero"] = ZREF
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
for g in ["Terreno", "Afloramentos", "Castelo", "Cerca", "Portas", "Reduto", "Vestigios", "Caminhos", "Vegetacao"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

# Pedra fotografada (kit.fotos, fotos CC0 da Poly Haven; créditos em medidas.json "texturas_foto"), com o tom das
# fotografias de Castro Laboreiro: silhares de granito cinzento-escuro em fiadas de c. 0,37 m, salpicados de
# líquen (3 m por imagem); afloramentos de granito com líquen (4 m por imagem, UV em caixa), quase sem cor: o tom
# vem da cor por vértice, como antes.
TEX_PEDRA, _ALT_PEDRA = textura_foto("stone_brick_wall_001", n=1024, metros=3.0, cor=(0.57, 0.57, 0.55), contraste=1.3,
                                     saturacao=0.6, variacao=0.12, metros_foto=3.0, liquen=0.12, semente=111)
NRM_PEDRA = imagem("pedra_normal", normais(_ALT_PEDRA, 3.0))
material("pedra", (1, 1, 1), 0.92, imagem("pedra", TEX_PEDRA), normal=NRM_PEDRA)
material("sombra", (0.10, 0.09, 0.08), 1.0)
material("cantaria", (0.72, 0.71, 0.67), 0.9, normal=NRM_PEDRA, forca=0.5)
material("coroamento", (1, 1, 1), 0.97, imagem("coroa", TEX_PEDRA * np.array([0.86, 0.86, 0.80])), normal=NRM_PEDRA)  # topo dos panos: silhares à vista
material("erva_seca", (1, 1, 1), 0.97, imagem("erva", textura_grao() * np.array([0.72, 0.63, 0.42])))
material("cal", (0.95, 0.95, 0.93), 0.8)
material("terreno", (1, 1, 1), 0.97, imagem("grao", textura_grao()), vcor=True)
TEX_GRANITO, _ALT_GRANITO = textura_foto("rock_01", n=1024, metros=4.0, cor=(0.86, 0.86, 0.86), contraste=0.9,
                                         saturacao=0.15, liquen=0.10, liquen_cor=(0.95, 0.97, 0.92), semente=112)
material("granito", (1, 1, 1), 0.9, imagem("granito", TEX_GRANITO), vcor=True,
         normal=imagem("granito_normal", normais(_ALT_GRANITO, 6.0)), forca=0.8)
material("base", (0.33, 0.26, 0.20), 0.95)
material("trilho", (0.74, 0.66, 0.52), 0.97)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)
material("ferro", (0.24, 0.25, 0.26), 0.55)
for pid_, d_ in PORTAS.items():  # tábuas das folhas, na cor de cada porta (fotografias)
    if "folhas" in d_:
        material("madeira_" + pid_, d_["folhas"][0], 0.85)

MEDIDAS = {}  # o que se mediu no LiDAR, para conferir e citar no castro-laboreiro.json
PERFIS = {}
MEDIDOS = {}

# Panos assentes nos penedos e a altura máxima da face de fora acima da rocha (frente poente da cerca e pano
# poente do castelo: troços baixos e finos entre penedos; pano norte do castelo: em cima de penedos enormes)
ASSENTES = {"cerca_poente": 2.8, "castelo_poente": 3.0, "castelo_norte": 4.5}
PE_ROCHA = {}   # cota do pé da face de fora (o topo da rocha) em cada vértice
PENEDO = {}     # vértices onde o penedo chega ao topo do pano (o muro interrompe-se)
TALUDE_ROCHA = 2.5  # declive da rocha que desce do pé do pano até encontrar o MDT (c. 68°)

print("perfis das muralhas…")
for s_, d_ in PANOS.items():
    lin = np.array([P(*p) for p in d_["pts"]])
    lin, topo, fora, dentro_ = perfil(lin, d_["esp"], d_.get("baixo", False))
    if s_ in ASSENTES:
        # assente nos penedos (revisão do perito, P1-4 e P1-14): a face de fora acaba na rocha, que o MDT
        # mostra debaixo do pano ou, onde o MDT é interpolado, no máximo ASSENTES[s_] m abaixo do topo
        mdt_l = cota(*lin.T)
        pe = np.minimum(np.maximum(mdt_l - 0.4, topo - ASSENTES[s_]), topo - 0.6)
        PE_ROCHA[s_] = pe
        fora = np.maximum(fora, pe + 1.2)  # o pano desce 1,5 m abaixo de fora: fica 0,3 m metido na rocha
        PENEDO[s_] = (topo - mdt_l) < 0.7  # o penedo chega ao topo do pano: aí não há muro
    PERFIS[s_] = np.stack([lin[:, 0], lin[:, 1], topo, fora, dentro_], 1)
    MEDIDOS[s_] = (lin, d_["esp"], topo, fora, dentro_)
    alt_e, alt_i = topo - (PE_ROCHA[s_] if s_ in PE_ROCHA else fora), topo - dentro_
    comp = float(np.sum(np.linalg.norm(np.diff(lin, axis=0), axis=1)))
    MEDIDAS[s_] = {"comprimento": comp, "topo": [float(topo.min() + ZREF), float(topo.max() + ZREF)],
                   "altura_exterior": [float(np.percentile(alt_e, 10)), float(np.percentile(alt_e, 90))],
                   "altura_interior": [float(np.percentile(alt_i, 10)), float(np.percentile(alt_i, 90))]}

# Portas: posição no pano medido (antes do terreno, que abre o corredor de cada uma, e dos panos, que se cortam no vão)
PONTOS_PORTA = {}
for pid, d_ in PORTAS.items():
    pano = d_["pano"]
    lin, esp, topo, fora, dentro_ = MEDIDOS[pano]
    if "ponto" in d_:
        i, p, t, n_ = no_pano(pano, P(*d_["ponto"]))
    else:  # a uma distância medida ao longo do pano a partir de outra porta
        ref, dist_ = d_["depois_de"]
        i0 = PONTOS_PORTA[ref][0]
        ss = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(lin, axis=0), axis=1))])
        i = int(np.argmin(np.abs(ss - (ss[i0] + dist_))))
        i, p, t, n_ = no_pano(pano, lin[i])
    sole = float(min(cota(*(p + n_ * (esp / 2 + 0.7))), cota(*(p - n_ * (esp / 2 + 0.7)))))
    PONTOS_PORTA[pid] = (i, p, t, n_, sole)


def abrir_portas(X, Y, Z):
    """Corredor diante de cada porta, dos dois lados do pano: patamar à cota da soleira e depois rampa até ao
    terreno; os lados descem em talude. Só rebaixa (a rocha e o chão que tapavam os vãos saem do caminho)."""
    Z = Z.copy()
    for pid, (i, p, t, n_, sole) in PONTOS_PORTA.items():
        d_ = PORTAS[pid]
        esp = MEDIDOS[d_["pano"]][1]
        S = (X - p[0]) * t[0] + (Y - p[1]) * t[1]
        D = (X - p[0]) * n_[0] + (Y - p[1]) * n_[1]
        pat = np.where(D > 0, esp / 2 + d_["fora"], esp / 2 + d_["dentro"])
        alem = np.maximum(np.abs(D) - pat, 0.0)
        lado = np.maximum(np.abs(S) - d_["corredor"] / 2, 0.0)
        sel = (lado < 1.5) & (alem < 15.0)
        lim = sole - 0.03 + d_["rampa"] * alem + 2.0 * lado
        Z[sel] = np.minimum(Z[sel], lim[sel])
    return Z


def cortar_vaos(lin, vals, cortes):
    """Troços da linha entre os vãos: cortes são intervalos (de, até) em metros ao longo da linha; cada troço
    acaba exatamente na ombreira, com os valores interpolados."""
    ss = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(lin, axis=0), axis=1))])

    def em(s_):
        return [np.array([np.interp(s_, ss, lin[:, 0]), np.interp(s_, ss, lin[:, 1])])] + [np.interp(s_, ss, v) for v in vals]
    limites = [0.0] + [x for c in sorted(cortes) for x in c] + [ss[-1]]
    trocos = []
    for a, b in zip(limites[0::2], limites[1::2]):
        pts = [em(a)] + [[lin[k]] + [v[k] for v in vals] for k in range(len(lin)) if a + 0.05 < ss[k] < b - 0.05] + [em(b)]
        trocos.append([np.array([q[j] for q in pts]) for j in range(len(vals) + 1)])
    return trocos


print("reduto…")
# Reduto: na ponta sul, uma cinta de muralha de 1,5 m em volta de um penedo (fotografia aérea; no MDT o
# miolo é uma lomba de rocha que sobe de 1019,5 m a poente a 1021,4 m a nascente). Por fora as faces
# descem ao pé medido, até 10 m; por dentro a cinta sobe 0,7 m acima do penedo e o miolo é o próprio
# terreno, com a rocha e o mato que lá estão.
# Revisão do perito (P1-5): no real é um penedo enorme com a muralha encostada só às faces sul e poente, até à
# altura do topo da rocha (±0,3 m), sem parapeito; a rocha fica à vista no miolo e a nascente, e não há muro do
# lado da cerca. Aqui fica só esse L (de noroeste a sueste, pelo canto sudoeste).
ESP_R = 1.5
b = REDUTO if area(REDUTO) > 0 else REDUTO[::-1]
_lr = desvio(b, ESP_R / 2, True)
anel_r = densificar(np.array([_lr[3], _lr[0], _lr[1]]), 1.0)
fora_r = np.min([cota(*desvio(anel_r, -(ESP_R / 2 + d)).T) for d in (0.8, 1.5)], axis=0)
dentro_r = np.max([cota(*desvio(anel_r, ESP_R / 2 + d).T) for d in (0.5, 1.0)], axis=0)
topo_mds = np.array([medir(centro=q, raio=0.8, q=90, sem_veg=False) for q in anel_r])
topo_rr = np.clip(np.nan_to_num(topo_mds, nan=-1e9), dentro_r - 0.3, dentro_r + 0.3)
topo_rr = mediana_movel(topo_rr, 3)
MEDIDOS["reduto"] = (anel_r, ESP_R, topo_rr, fora_r, dentro_r)
m = Malha()
parede(m, anel_r, ESP_R, topo_rr, fora_r, dentro_r, mat=0, mat_topo=1, escalonar=False)
# cunhais de silhares maiores nas esquinas de fora (sudoeste, e as pontas a sueste e a noroeste)
for k in (0, 1, 3):
    c_ = b[k]
    u_ = unit(b[(k + 1) % 4] - b[k])
    z_top = float(topo_rr[int(np.argmin(np.hypot(*(anel_r - c_).T)))])
    z_pe = float(cota(*(c_ + (c_ - b.mean(0)) * 0.08))) - 1.0
    zs = np.arange(z_pe, z_top - 0.3, 0.55)
    for j, z_ in enumerate(zs):
        uu = u_ if j % 2 == 0 else np.array([-u_[1], u_[0]])
        m.caixa(c_ + uu * 0.45 - (b.mean(0) - c_) / np.linalg.norm(b.mean(0) - c_) * 0.02, uu, 0.95, 0.06,
                z_ + 0.02, min(z_ + 0.53, z_top), 2)
m.objeto("Reduto", "reduto", ["pedra", "coroamento", "cantaria"], GRUPOS["Reduto"])
pe_min = float(np.min(fora_r))
topo_r = float(np.max(topo_rr))
chao_in = float(np.median(cota(*densificar(np.array([P(-3.0, -66.0), P(-2.0, -62.0)]), 0.5).T)))
MEDIDAS["reduto"] = {"topo": topo_r + ZREF, "planta": [float(np.linalg.norm(b[1] - b[0])), float(np.linalg.norm(b[2] - b[1]))],
                     "pe_exterior": pe_min + ZREF, "altura_exterior_max": topo_r - pe_min,
                     "miolo": [float(dentro_r.min() + ZREF), float(dentro_r.max() + ZREF)],
                     "acima_do_recinto": topo_r - chao_in}

print("terreno e afloramentos…")
ZTER = construir_terreno()
_XT = np.linspace(-RAIO, RAIO, ZTER.shape[0])


def cota_t(x, y):
    """Cota do terreno da maqueta (o MDT já rebaixado junto às muralhas), por interpolação bilinear."""
    p_ = _XT[1] - _XT[0]
    c = np.clip((np.asarray(x, float) + RAIO) / p_, 0, len(_XT) - 1.001)
    r = np.clip((np.asarray(y, float) + RAIO) / p_, 0, len(_XT) - 1.001)
    c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
    fc, fr = c - c0, r - r0
    return (ZTER[r0, c0] * (1 - fc) * (1 - fr) + ZTER[r0, c0 + 1] * fc * (1 - fr)
            + ZTER[r0 + 1, c0] * (1 - fc) * fr + ZTER[r0 + 1, c0 + 1] * fc * fr)


def chao_sob(x, y, raio):
    """Cota mais baixa do terreno debaixo de uma copa (centro e anel a 70 % do raio): nas encostas a pique
    a copa assenta no lado de baixo e não fica a flutuar (auditoria de novembro de 2026)."""
    a_ = np.linspace(0, 2 * math.pi, 9)[:-1]
    xs_ = np.r_[x, x + 0.7 * raio * np.cos(a_)]
    ys_ = np.r_[y, y + 0.7 * raio * np.sin(a_)]
    return float(cota_t(xs_, ys_).min())

print("muralhas…")
TOPO_VAO = {}  # cota do coroamento por cima de cada vão (a mais baixa das duas pontas que lá chegam)
for s_, d_ in PANOS.items():
    lin, esp, topo, fora, dentro_ = MEDIDOS[s_]
    m = Malha()
    ss = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(lin, axis=0), axis=1))])
    vaos = [(pid, ss[PONTOS_PORTA[pid][0]], PORTAS[pid]["largura"] / 2) for pid in PORTAS if PORTAS[pid]["pano"] == s_]
    vaos.sort(key=lambda v: v[1])
    cortes = [(c - h, c + h) for _, c, h in vaos]
    # onde o penedo chega ao topo do pano (3 vértices seguidos ou mais, longe das portas), o muro interrompe-se
    pen = PENEDO.get(s_, np.zeros(len(lin), bool)).copy()
    for _, c, h in vaos:
        pen &= np.abs(ss - c) > h + 2.5
    lab, nlab = ndi.label(pen)
    n_pen = 0
    for k in range(1, nlab + 1):
        ii = np.nonzero(lab == k)[0]
        if len(ii) >= 3 and ii[0] > 0 and ii[-1] < len(lin) - 1:
            cortes.append((ss[ii[0]], ss[ii[-1]]))
            n_pen += 1
    if s_ in ASSENTES:
        MEDIDAS[s_]["interrupcoes_penedo"] = n_pen
    trocos = cortar_vaos(lin, [topo, fora, dentro_], cortes)
    # os troços baixos entre os penedos acompanham o chão; os panos altos sobem em patamares
    ln_ = shapely.LineString(lin)
    pontas = []
    for tr in trocos:
        a_ = parede(m, *tr[:1], esp, *tr[1:], mat=0, mat_topo=1, escalonar=not d_.get("baixo", False))[2]
        s_a, s_b = (ln_.project(shapely.Point(*tr[0][q])) for q in (0, -1))
        pontas.append((s_a, s_b, float(a_[0]), float(a_[-1])))
    for pid, c, h in vaos:
        antes = min(pontas, key=lambda q: abs(q[1] - (c - h)))
        depois = min(pontas, key=lambda q: abs(q[0] - (c + h)))
        TOPO_VAO[pid] = min(antes[3], depois[2])
    grupo = GRUPOS["Castelo"] if d_["elemento"] == "castelo" else GRUPOS["Cerca"]
    m.objeto(s_.title(), d_["elemento"], ["pedra", "coroamento"], grupo)

print("portas…")


def contorno(tipo, w, zs, k=12, extra=0.0):
    """Intradorso de um vão no plano da face (s, z), do arranque esquerdo (s = -w/2) ao direito, em k troços:
    arco quebrado (dois arcos de raio igual ao vão, centrados nos arranques opostos), pleno, ou verga reta."""
    h = w / 2
    out = []
    for j in range(k + 1):
        t_ = j / k
        if tipo == "quebrado":
            r = w + extra
            am = math.acos(h / r)
            a_ = am * (t_ / 0.5) if t_ <= 0.5 else am * ((1 - t_) / 0.5)
            out.append((h - r * math.cos(a_), zs + r * math.sin(a_)) if t_ <= 0.5 else (-h + r * math.cos(a_), zs + r * math.sin(a_)))
        elif tipo == "pleno":
            a_ = math.pi * (1 - t_)
            out.append(((h + extra) * math.cos(a_), zs + (h + extra) * math.sin(a_)))
        else:
            out.append((-h + w * t_, zs))
    return out


def folhas(m, p, t, n_, esp, w, sole, zs, tipo, abertura, mat_m, mat_f):
    """Duas folhas de tábuas (6 cm) com quatro ferragens, nas ombreiras junto à face de dentro do vão, com o
    contorno do arco dessa face; abertura em graus (0 = fechadas), sempre para dentro."""
    d_h = -esp / 2 + 0.12
    topo_ = contorno(tipo, w, zs, 16)
    for sg in (-1, 1):
        # contorno da folha em (x a partir da dobradiça, z): do pé da dobradiça ao eixo, pelo arco
        arco_ = [(w / 2 - sg * s_, z) for s_, z in topo_ if sg * s_ >= -1e-6]
        cont = [(0.0, sole + 0.03), (w / 2 - 0.01, sole + 0.03)] + sorted(arco_, key=lambda q: -q[0])
        cont = [(min(x, w / 2 - 0.01), z - 0.02) if i > 1 else (x, z) for i, (x, z) in enumerate(cont)]
        th = math.radians(abertura)
        dobr = p + t * sg * (w / 2) + n_ * d_h
        dir_ = -sg * t * math.cos(th) - n_ * math.sin(th)
        nrm = np.array([-dir_[1], dir_[0]])
        for e0, e1 in ((-0.03, 0.03),):
            A = [(*(dobr + dir_ * x + nrm * e0), z) for x, z in cont]
            B = [(*(dobr + dir_ * x + nrm * e1), z) for x, z in cont]
            m.face(A, mat_m)
            m.face(B[::-1], mat_m)
            for i in range(len(cont)):
                j = (i + 1) % len(cont)
                m.face([A[i], A[j], B[j], B[i]], mat_m)
        for f_ in np.linspace(0.12, 0.85, 4):
            z_ = sole + (zs - sole) * f_ * 1.15
            m.caixa(dobr + dir_ * (w / 4), dir_, w / 2 * 0.86, 0.08, z_, z_ + 0.07, mat_f)


for pid, d_ in PORTAS.items():
    m = Malha()
    pano = d_["pano"]
    lin, esp, topo, fora, dentro_ = MEDIDOS[pano]
    i, p, t, n_, sole = PONTOS_PORTA[pid]
    w, zn = d_["largura"], d_["nascenca"]
    zs = sole + zn
    tipo_f, tipo_d = d_["arco"], d_.get("arco_dentro", d_["arco"])
    z_top = TOPO_VAO[pid]
    Q = lambda s_, d, z: (*(p + t * s_ + n_ * d), z)
    e = esp / 2
    # por cima do vão: a alvenaria entre as duas pontas do pano, com o intradorso do arco de face a face
    cf, cd = contorno(tipo_f, w, zs), contorno(tipo_d, w, zs)
    topo_arco = max(z for _, z in cf + cd)
    z_top = max(z_top, topo_arco + 0.5)
    m.face([Q(s_, e, z) for s_, z in cf] + [Q(w / 2, e, z_top), Q(-w / 2, e, z_top)], 0)
    m.face([Q(s_, -e, z) for s_, z in cd] + [Q(w / 2, -e, z_top), Q(-w / 2, -e, z_top)], 0)
    for j in range(len(cf) - 1):
        m.face([Q(*cf[j][:1], e, cf[j][1]), Q(*cf[j + 1][:1], e, cf[j + 1][1]), Q(*cd[j + 1][:1], -e, cd[j + 1][1]), Q(*cd[j][:1], -e, cd[j][1])], 0)
    m.face([Q(-w / 2, e, z_top), Q(w / 2, e, z_top), Q(w / 2, -e, z_top), Q(-w / 2, -e, z_top)], 1)
    for sg in (-1, 1):
        m.face([Q(sg * w / 2, e, zs), Q(sg * w / 2, -e, zs), Q(sg * w / 2, -e, z_top), Q(sg * w / 2, e, z_top)], 0)
    # chão de saibro no vão, com meio metro para cada lado
    q = [p + t * (-w / 2) + n_ * (e + 0.5), p + t * (w / 2) + n_ * (e + 0.5), p + t * (w / 2) - n_ * (e + 0.5), p + t * (-w / 2) - n_ * (e + 0.5)]
    m.prisma(q if area(np.array(q)) > 0 else q[::-1], sole - 0.5, sole + 0.02, 3)
    # molduras: aduelas grandes da pedra do pano, metidas nas fiadas (sem impostas); verga reta inteiriça
    for sg, tipo in ((1, tipo_f), (-1, tipo_d)):
        c_ = p + sg * n_ * e
        if tipo == "reto":
            vao_reto(m, c_, t, sg * n_, w, sole, zn, 0, 0, escuro=False)
        else:
            aduelas(m, c_, t, sg * n_, w, sole, zn, 0, quebrado=tipo == "quebrado", esp=0.6, saliencia=0.03,
                    k=d_.get("aduelas", 11) if sg == 1 or tipo_d == tipo_f else 7)
    if d_.get("misulas"):
        # duas mísulas grossas no topo do pano, juntas e quase por cima do arco, saindo c. 0,5 m (SIPA, fotos)
        z_m = max(zs + w / 2 + 0.6, z_top - 0.75)
        for lado in (-1, 1):
            c_ = p + n_ * (e + 0.25) + t * lado * 0.65
            m.caixa(c_, t, 0.4, 0.5, z_m, z_m + 0.32, 0)
            m.caixa(c_ - n_ * 0.08, t, 0.4, 0.34, z_m - 0.25, z_m, 0)
    if "folhas" in d_:
        _, abertura = d_["folhas"]
        folhas(m, p, t, n_, esp, w, sole, zs, tipo_d, abertura, 4, 5)
    MEDIDAS[pid] = {"soleira": sole + ZREF, "topo_muralha": float(topo[i] + ZREF),
                    "acima_da_soleira": float(topo[i] - sole), "vao": w, "fecho": float(topo_arco - sole),
                    "atravessa": float(esp)}
    mats_ = ["pedra", "coroamento", "cantaria", "trilho", "madeira_" + pid if "folhas" in d_ else "ferro", "ferro"]
    m.objeto(pid.title(), pid, mats_, GRUPOS["Portas"])  # um objeto por porta (placa e clique)

print("escada da porta norte…")
# Escada metálica com guarda que sobe ao portal norte (SIPA: o acesso pela porta norte "acaba numa escada
# metálica com guarda"). Lanço reto, perpendicular ao pano, da soleira até onde o MDT desce 2,5 m ou mais;
# degraus de 0,2 m; largura, perfis e guardas estimados.


def viga(m, a, b, larg, alt, t, mat):
    """Perfil de secção retangular entre dois pontos 3D (a face de cima passa por a e b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    o = np.array([t[0], t[1], 0.0]) * larg / 2
    h = np.array([0, 0, alt])
    q = [a - o, b - o, b + o, a + o]
    m.face([tuple(x) for x in q][::-1], mat)
    m.face([tuple(x - h) for x in q], mat)
    for k in range(4):
        x0, x1 = q[k], q[(k + 1) % 4]
        m.face([tuple(x0), tuple(x0 - h), tuple(x1 - h), tuple(x1)], mat)


i, p, t, n_, sole = PONTOS_PORTA["porta_traicao"]
esp_n = MEDIDOS["castelo_norte"][1]
ini = p + n_ * (esp_n / 2)
L = 6.0
for d in np.arange(0.5, 7.0, 0.25):
    if cota(*(ini + n_ * d)) <= sole - 2.5:
        L = float(d)
        break
fundo_e = float(cota(*(ini + n_ * L)))
ndeg = max(3, int(math.ceil((sole - fundo_e) / 0.2)))
passo_e, esp_e = L / ndeg, (sole - fundo_e) / ndeg
m = Malha()
LARG_E = 1.1
for k in range(ndeg):
    z_ = sole - (k + 1) * esp_e
    m.caixa(ini + n_ * passo_e * (k + 0.5), t, LARG_E, passo_e * 0.92, z_ - 0.04, z_, 0)
for sg in (-1, 1):
    lado = t * sg * (LARG_E / 2 + 0.04)
    a3 = np.array([*(ini + lado), sole])
    b3 = np.array([*(ini + n_ * L + lado), fundo_e])
    viga(m, a3, b3, 0.06, 0.25, t, 0)  # banzo
    viga(m, a3 + [0, 0, 1.0], b3 + [0, 0, 1.0], 0.05, 0.05, t, 0)  # corrimão
    viga(m, a3 + [0, 0, 0.5], b3 + [0, 0, 0.5], 0.03, 0.03, t, 0)  # travessa
    for f_ in np.linspace(0, 1, max(2, int(L / 1.2) + 1)):
        c_ = ini + n_ * L * f_ + lado
        z_ = sole + (fundo_e - sole) * f_
        m.caixa(c_, t, 0.05, 0.05, z_ - 0.2, z_ + 1.0, 0)  # prumos
m.objeto("Escada_Norte", "escada_norte", ["ferro"], GRUPOS["Portas"], uv=None)
MEDIDAS["escada_norte"] = {"lanco": L, "desnivel": sole - fundo_e, "degraus": ndeg}

print("vestígios da capela e marco geodésico…")
# Construção quadrangular junto ao muro norte da cerca (SIPA): só as primeiras fiadas, com a entrada a
# norte. Planta e altura das paredes medidas no MDS; espessura de 1 m.
m = Malha()
b = CAPELA if area(CAPELA) > 0 else CAPELA[::-1]
anel_c = densificar(desvio(b, 0.5, True), 0.5, True)
topo_c = float(np.nanpercentile([medir(centro=q, raio=0.8, q=90, sem_veg=False) for q in anel_c], 50))
chao_c = float(np.median(cota(*densificar(desvio(b, -1.0, True), 1.0, True).T)))
topo_c = min(max(topo_c, chao_c + 0.6), chao_c + 1.8)
ESP_C = 1.0
inn_c = desvio(b, ESP_C, True)
xs_porta = (9.2, 10.2)  # vão a norte, onde o MDS mostra o corte na parede
for k in range(4):
    a, c, ia, ic = b[k], b[(k + 1) % 4], inn_c[k], inn_c[(k + 1) % 4]
    if k == 2:  # parede norte (de nascente para poente): corta-se o vão da entrada
        for x0_, x1_ in ((max(a[0], c[0]), xs_porta[1]), (xs_porta[0], min(a[0], c[0]))):
            sa = (a[0] - x0_) / (a[0] - c[0])
            sc = (a[0] - x1_) / (a[0] - c[0])
            q = [a + (c - a) * sa, a + (c - a) * sc, ia + (ic - ia) * sc, ia + (ic - ia) * sa]
            m.prisma(q, chao_c - 0.6, topo_c, 0)
    else:
        m.prisma([a, c, ic, ia], chao_c - 0.6, topo_c, 0)
m.objeto("Capela", "capela", ["pedra"], GRUPOS["Vestigios"])
MEDIDAS["capela"] = {"planta": [float(np.linalg.norm(b[1] - b[0])), float(np.linalg.norm(b[2] - b[1]))],
                     "topo_paredes": topo_c + ZREF, "altura": topo_c - chao_c}
# Marco geodésico: pilar cilíndrico branco sobre base quadrada, no topo do penedo mais alto (o MDS tem
# ali um pico estreito com 2 m). Proporções do modelo corrente dos vértices geodésicos.
m = Malha()
z0_m = float(cota(*MARCO))
z1_m = float(np.max(celulas(centro=MARCO, raio=0.8, sem_veg=False)))
h_m = max(z1_m - z0_m, 1.5)
m.caixa(MARCO, np.array([1.0, 0.0]), 1.1, 1.1, z0_m - 0.4, z0_m + 0.35, 0)
m.prisma([MARCO + 0.32 * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)],
         z0_m + 0.35, z0_m + h_m - 0.12, 0)
m.prisma([MARCO + 0.38 * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)],
         z0_m + h_m - 0.12, z0_m + h_m, 0)
m.objeto("Marco_Geodesico", "marco_geodesico", ["cal"], GRUPOS["Vestigios"], uv=None)
MEDIDAS["marco_geodesico"] = {"chao": z0_m + ZREF, "topo": z1_m + ZREF, "altura": h_m}

print("caminhos…")
# Trilhos de terra do OSM: o acesso pela Porta do Sol, os carreiros do recinto e a escadaria da porta norte.
m = Malha()
paredes = [v[0] for v in MEDIDOS.values()]
for f in F["caminhos"]:
    if f["subtipo"] != "road" or f["geometria"]["type"] != "LineString" or f["classe"] not in ("path", "footway", "steps", "track"):
        continue
    lw = 1.6 if f["classe"] == "steps" else 1.2
    pts = densificar(geom(f), 0.5)
    ok = np.hypot(*pts.T) < RAIO - 1.5
    for lin_ in paredes:  # não sobem às muralhas (passam nas portas, que ficam por baixo do coroamento)
        ok &= distancia(pts[:, 0], pts[:, 1], lin_)[0] > 1.2
    ok &= ~dentro(pts[:, 0], pts[:, 1], REDUTO)
    ok &= distancia(pts[:, 0], pts[:, 1], np.array([ini - n_ * 0.5, ini + n_ * (L + 0.8)]))[0] > LARG_E / 2 + 0.3  # a escada
    troco = []
    for i, (p, k) in enumerate(zip(pts, ok)):
        if k:
            troco.append(p)
        if (not k or i == len(pts) - 1) and len(troco) > 1:
            t = np.array(troco)
            L_ = desvio(t, lw / 2)
            R_ = desvio(t, -lw / 2)
            zl = np.maximum(cota(L_[:, 0], L_[:, 1]), cota(*t.T)) + 0.12  # por cima da rocha rugosa
            zr = np.maximum(cota(R_[:, 0], R_[:, 1]), cota(*t.T)) + 0.12
            for j in range(len(t) - 1):
                m.face([(*R_[j], zr[j]), (*R_[j + 1], zr[j + 1]), (*L_[j + 1], zl[j + 1]), (*L_[j], zl[j])], 0)
        if not k:
            troco = []
m.objeto("Caminhos", "caminhos", ["trilho"], GRUPOS["Caminhos"], uv=None, liso=True)

print("vegetação…")


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


def plantar(x, y, z, raio, alt, especie, tronco=1.0):
    """Árvore com a posição, o topo e o raio de copa medidos; a forma da copa é esquemática, por espécie:
    castanheiro (copa redonda e cheia, verde-vivo, tronco cinzento grosso) ou sobreiro (copa larga e
    achatada, irregular, verde-escuro acinzentado, tronco de cortiça)."""
    b = len(mv)
    sobreiro = especie == "sobreiro"
    rt = (0.16 + 0.05 * raio) if sobreiro else (0.14 + 0.045 * raio)
    inclina = rng.normal(0, 0.25 if sobreiro else 0.1, 2)  # o sobreiro tem o tronco torto
    for k in range(6):
        a = 2 * math.pi * k / 6
        mv.append((x + rt * math.cos(a), y + rt * math.sin(a), z - 0.3))
        mv.append((x + inclina[0] + 0.7 * rt * math.cos(a), y + inclina[1] + 0.7 * rt * math.sin(a), z + tronco + alt * 0.35))
    for k in range(6):
        j = (k + 1) % 6
        mf.append([b + 2 * k, b + 2 * j, b + 2 * j + 1, b + 2 * k + 1])
    mc.extend([(0.42, 0.31, 0.22) if sobreiro else (0.36, 0.32, 0.27)] * 12)
    x, y = x + inclina[0], y + inclina[1]
    base, topo = z + tronco, z + tronco + alt
    sh = 0.88 + 0.24 * rng.random()
    if sobreiro:
        cor = tuple(np.clip(np.array((0.27, 0.33, 0.22)) * sh, 0, 1))
        irregular, n, abre, achata = 0.13, (2 if raio < 1.8 else 5), 0.52, 0.32
    else:
        cor = tuple(np.clip(np.array((0.31, 0.43, 0.17)) * sh, 0, 1))
        irregular, n, abre, achata = 0.07, (1 if raio < 1.8 else 4), 0.40, 0.40
    if n == 1:
        _lobo(x, y, base + alt / 2, raio, alt / 2, cor, base, topo, False, irregular)
        return
    # lobo principal, um pouco abaixo do topo medido, e lobos menores à volta sem passar do raio medido
    _lobo(x, y, topo - alt * 0.42, raio * (0.6 if sobreiro else 0.7), alt * 0.42, cor, base, topo, raio > 3, irregular)
    a0 = rng.random() * 2 * math.pi
    for k in range(n):
        a = a0 + 2 * math.pi * k / n + rng.normal(0, 0.3)
        d = raio * (abre + 0.08 * rng.random())
        rl = raio * (0.48 + 0.08 * rng.random())
        zl = base + alt * ((0.5 if sobreiro else 0.36) + 0.12 * rng.random())
        c2 = tuple(np.clip(np.array(cor) * (0.92 + 0.14 * rng.random()), 0, 1))
        _lobo(x + d * math.cos(a), y + d * math.sin(a), zl, rl, min(rl, alt * achata), c2, base, topo, False, irregular)


ico0, ico0f = arvore_modelo(0)
ico1, ico1f = arvore_modelo(1)
ico2, ico2f = arvore_modelo(2)
mv, mf, mc = [], [], []
evitar = [CAPELA]  # no miolo do reduto há mato (fotografia aérea)

# Copas medidas no LiDAR (lidar.py): posição, topo e raio de cada árvore; a forma é esquemática. Os panos
# aparecem por vezes na nuvem de pontos como vegetação alta: os "topos" a menos de 2,5 m de uma
# muralha não são árvores.
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
_ov, _of = [], []
for _ob in bpy.data.objects:
    if _ob.type != "MESH" or _ob.name in ("Terreno", "Afloramentos", "Base", "Vegetacao"):
        continue
    _b = len(_ov)
    _ov.extend([tuple(_ob.matrix_world @ v.co) for v in _ob.data.vertices])
    _of.extend([[_b + i for i in p_.vertices] for p_ in _ob.data.polygons])
OBRA = BVHTree.FromPolygons(_ov, _of) if _of else None


def folga_obra(pts):
    if OBRA is None:
        return 99.0
    d = [OBRA.find_nearest(Vector(p_))[3] for p_ in pts]
    return min([x for x in d if x is not None] or [99.0])


n_arv = n_sob = 0
n_afastadas = {"encolhidas": 0, "retiradas": 0}
for x, y, topo, _, r in ARV_LIDAR:
    if np.hypot(x, y) > RAIO - 2:
        continue
    if any(distancia(np.array([x]), np.array([y]), lin_)[0][0] < 2.5 for lin_ in paredes):
        continue
    if any(dentro(x, y, z_) for z_ in evitar) or np.hypot(x - MARCO[0], y - MARCO[1]) < 2.5:
        continue
    z = float(cota(x, y))
    h = float(topo) - ZREF - z
    if h < 2.0:
        continue
    r = float(np.clip(r, 1.0, 5.5))
    # Castanheiros e sobreiros (o arvoredo do monte, segundo o Tiago): o souto cerrado da encosta poente é
    # sobretudo de castanheiros; nas lajes e no mato aberto, de sobreiros.
    p_sob = 0.2 + 0.6 * float(np.clip(1 - amostra(COBERTO, x, y) * 1.4, 0, 1))
    especie = "sobreiro" if rng.random() < p_sob else "castanheiro"
    alt = float(np.clip(h * (0.5 if especie == "sobreiro" else 0.65), 1.4, (1.5 if especie == "sobreiro" else 2.4) * r))
    # chão real da maqueta (rebaixado junto aos panos) em vez do MDT; o topo medido mantém-se
    z_ = min(z, float(cota_t(x, y)))
    h += z - z_
    z = z_
    tronco_ = max(h - alt, 0.6)
    # nenhuma copa entra nos panos (auditoria de novembro de 2026: 12 tocavam)
    if folga_obra([(x, y, z + tronco_ + alt * f) for f in (0.15, 0.5, 0.85)]) < r * 1.1 + 0.15:
        f_ = folga_obra([(x, y, z + tronco_ + alt * f) for f in (0.15, 0.5, 0.85)])
        if f_ < 1.2:
            n_afastadas["retiradas"] += 1
            continue
        r = (f_ - 0.15) / 1.1
        alt = min(alt, (1.5 if especie == "sobreiro" else 2.4) * r)
        tronco_ = max(h - alt, 0.6)
        if folga_obra([(x, y, z + tronco_ + alt * f) for f in (0.15, 0.5, 0.85)]) < r * 1.1 + 0.05:
            n_afastadas["retiradas"] += 1
            continue
        n_afastadas["encolhidas"] += 1
    plantar(float(x), float(y), z, r, alt, especie, tronco=tronco_)
    n_arv += 1
    n_sob += especie == "sobreiro"

# Mato baixo (giestas, urze, tojo das fotografias): máximos da altura da vegetação entre 0,5 e 2,5 m nas
# células de classe vegetação, cada um com uma moita de copa redonda à altura medida.
H = MDS - MDT
mato = VEG & (H > 0.5) & (H < 2.5)
hs = ndi.gaussian_filter(np.where(mato, H, 0.0), 1.0)
pico = (hs == ndi.maximum_filter(hs, size=5)) & mato & (hs > 0.45)
rr, cc = np.nonzero(pico)
n_mato = 0
for r_, c_ in zip(rr, cc):
    x, y = float(LXS[c_]), float(LXS[r_])
    if np.hypot(x, y) > RAIO - 1.5:
        continue
    if any(distancia(np.array([x]), np.array([y]), lin_)[0][0] < 1.6 for lin_ in paredes) or any(dentro(x, y, z_) for z_ in evitar) \
            or np.hypot(x - MARCO[0], y - MARCO[1]) < 2.0:  # o marco geodésico sai como vegetação na nuvem
        continue
    h = float(np.clip(H[r_, c_], 0.5, 2.4))
    raio_ = 0.45 + 0.35 * h + 0.25 * rng.random()
    z0_ = float(cota_t(x, y))
    z = max(chao_sob(x, y, raio_), z0_ - 0.6)  # assenta no lado de baixo da encosta, sem se enterrar
    if folga_obra([(x, y, z + h * 0.3), (x, y, z + h * 0.7)]) < raio_ + 0.1:
        continue  # moita encostada a um pano
    cor = np.array((0.36, 0.42, 0.22) if rng.random() < 0.7 else (0.46, 0.46, 0.26)) * (0.9 + 0.2 * rng.random())
    b_ = len(mv)
    jit = rng.normal(0, 0.08, ico0.shape)
    pts = (ico0 + jit) * np.array([raio_, raio_, h / 2]) + np.array([x, y, z + h / 2 - 0.1])
    mv.extend(map(tuple, pts))
    mf.extend([[b_ + i for i in fa] for fa in ico0f])
    luz = 0.65 + 0.45 * np.clip((pts[:, 2] - z) / max(h, 0.1), 0, 1)
    mc.extend(map(tuple, np.clip(cor[None, :] * luz[:, None], 0, 1)))
    n_mato += 1
MEDIDAS["vegetacao"] = {"arvores": n_arv, "sobreiros": int(n_sob), "castanheiros": n_arv - int(n_sob), "moitas": n_mato,
                        **n_afastadas}

me = bpy.data.meshes.new("vegetacao")
me.from_pydata(mv, [], mf)
me.update()
me.shade_smooth()
cl = np.array(mc) ** 2.2
ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
ca.data.foreach_set("color", np.hstack([cl, np.ones((len(cl), 1))]).astype(np.float32).ravel())
me.color_attributes.active_color = ca
novo_objeto("Vegetacao", me, "vegetacao", mat="arvore", pai=GRUPOS["Vegetacao"])


# Pontos das placas no visor (x este, y cota, z sul), para o castro-laboreiro.json
def ancora(x, y, acima=2.0):
    return [round(float(x), 1), round(float(cota(x, y)) + acima, 1), round(-float(y), 1)]


MEDIDAS["ancoras"] = {
    "castelo": ancora(*P(2.0, 50.0), 5.0),
    "cerca": ancora(*P(0.0, -20.0), 4.0),
    "afloramentos": ancora(*P(40.0, 10.0), 2.0),
}
for pid, (i, p, t, n_, sole) in PONTOS_PORTA.items():
    MEDIDAS["ancoras"][pid] = [round(float(p[0]), 1), round(float(MEDIDOS[PORTAS[pid]["pano"]][2][i]) + 1.5, 1), round(-float(p[1]), 1)]

(AQUI / "saida").mkdir(exist_ok=True)
np.savez(AQUI / "saida" / "perfis.npz", **PERFIS)
MEDIDAS["texturas_foto"] = fontes_fotos()  # créditos das fotos de pedra e rocha
(AQUI / "dados" / "medidas.json").write_text(json.dumps(_arred(MEDIDAS), ensure_ascii=False, indent=1))
print(json.dumps(_arred(MEDIDAS), ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
glb = AQUI / "saida" / "castro-laboreiro-bruto.glb"
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

final = SAIDA / "castro-laboreiro.glb"
subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "castro-laboreiro.blend"))

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
    from mathutils import Vector
    d = Vector((-math.sin(math.radians(45)), math.cos(math.radians(45)), 0)) * math.cos(math.radians(50))
    d.z = math.sin(math.radians(50))
    so.rotation_euler = d.to_track_quat("Z", "Y").to_euler()  # sol a noroeste, como no relevo do atlas
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
        al = Vector((alvo_xy[0], alvo_xy[1], float(cota(*alvo_xy)) + 2 if alvo_z is None else alvo_z))
        co.location = al + Vector((math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))) * dist
        co.rotation_euler = (al - co.location).to_track_quat("-Z", "Y").to_euler()
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

    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 84
    C = P(0.0, -2.0)
    vista("capa", 200, 34, 300, lente=50, res=(1280, 800), destino=SAIDA / "capa.jpg", alvo_xy=tuple(C), alvo_z=float(cota(*C)) - 4)
    if "--capa" not in sys.argv:
        vista("topo", 180, 89.9, 400, orto=190, res=(1100, 1100))
        vista("aereo", 215, 48, 190, lente=50, alvo_xy=tuple(P(2, -5)))  # como a fotografia aérea do Tiago
        vista("norte", 0, 30, 260)
        vista("nascente", 90, 28, 260)
        vista("poente", 270, 28, 260)
        vista("sul", 180, 28, 260)
        for pid, (i, p, t, n_, sole) in PONTOS_PORTA.items():
            az = math.degrees(math.atan2(n_[0], n_[1]))  # de fora
            vista(pid, az + 15, 8, 14, lente=35, alvo_xy=tuple(p), alvo_z=sole + 1.6)
            vista(pid + "_dentro", az + 195, 8, 14, lente=35, alvo_xy=tuple(p), alvo_z=sole + 1.6)
        vista("reduto", 210, 25, 45, lente=35, alvo_xy=tuple(REDUTO.mean(0)))
        vista("pano_nascente", 250, 18, 45, lente=28, alvo_xy=tuple(P(3, -50)))  # como a primeira fotografia
        vista("castelo", 170, 30, 90, lente=35, alvo_xy=tuple(P(2, 42)))
        vista("capela", 200, 40, 22, lente=35, alvo_xy=tuple(CAPELA.mean(0)))
        vista("marco", 160, 15, 14, lente=35, alvo_xy=tuple(MARCO))
