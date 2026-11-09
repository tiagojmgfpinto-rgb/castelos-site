"""Constrói a maqueta 3D da Torre de Menagem de Braga (estado atual) em Blender e exporta-a para
public/maquetas/braga/braga.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow
    python pipeline/maquetas/braga/construir.py            # GLB
    python pipeline/maquetas/braga/construir.py --vistas   # GLB + capa + vistas de verificação

Dados: dados/lidar.npz, dados/torre.npz e dados/orto.jpg (lidar.py: LiDAR 2025 e ortofotomapa 2021 da DGT),
dados/fontes.json (recolher.py: plantas e ruas do OSM via Overture).

A maqueta é só a torre, com a praça à volta e os quarteirões vizinhos em volumes simples. A torre é medida
no LiDAR (planta, rotação, cotas da praça, do parapeito, das pontas dos merlões e do telhado, e a escada
em dois lanços) e desenhada com o detalhe das fotografias: balcões com matacães sobre mísulas nos quatro
cunhais, merlões piramidais, janela geminada a sul, janela em arco a norte, seteiras, gárgulas, a porta
alta com o escudo de D. Dinis, a escada de pedra adossada com portão de ferro, e a cantaria de granito
cinzento-bege com silhares rosados em fiadas de c. 0,38 m.
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
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pipeline/, para o kit comum
from kit.geometria import normais, quad, arco_pts  # noqa: E402
from kit.texturas import ruido, fbm, textura_telha, imagem  # noqa: E402
from kit.cena import uv_mundo  # noqa: E402
from kit.materiais import MATS, malha_rapida, material  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "braga.json").read_text())
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "braga"
LID = np.load(AQUI / "dados" / "lidar.npz")
LPASSO = float(LID["passo"])
MEIO_L = float(LID["meio"])
MDT = LID["mdt"] / 100.0 + float(LID["base"])
MDS = LID["mds"] / 100.0 + float(LID["base"])
CLS = LID["classe"]
LXS = np.linspace(-MEIO_L, MEIO_L, MDT.shape[1])
PT = np.load(AQUI / "dados" / "torre.npz")["pontos"]

MEIO = 22.0  # a maqueta: quadrado de 44 m centrado na torre (envolvente mínima, pedido do Tiago)
rng = np.random.default_rng(1300)

# ---------------------------------------------------------------- referencial da torre (medido no LiDAR)
# O retângulo mínimo dos pontos do topo dá as faces a 71,7° e 161,7° de azimute: a torre está rodada c. 18°
# em relação ao norte. u = ao longo do azimute 71,7° (ENE), v = azimute 341,7° (NNO).
AZ = math.radians(71.7)
EU = np.array([math.sin(AZ), math.cos(AZ)])
EV = np.array([-math.cos(AZ), math.sin(AZ)])
A, B = 5.2, 5.0          # meias medidas da torre em u e v (10,4 × 10,0 m entre faces, LiDAR)
CHAO = 191.35            # praça junto à torre (MDT: 191,2 a poente, 191,6 a nascente)
Z_AMEIA = 215.65         # soleira das ameias (parapeito entre merlões), LiDAR
Z_MERLAO = 216.95        # ponta dos merlões do pano, LiDAR (216,6 a 217,0)
Z_BALCAO = 216.30        # topo do parapeito dos balcões dos cunhais (fotografias, LiDAR)
Z_BALC_MER = 217.00      # ponta dos merlões dos balcões (o ponto mais alto da nuvem: 217,0)
Z_BALC_BASE = 215.10     # base do parapeito dos balcões, sobre as mísulas
Z_MISULA = 213.50        # pé das mísulas (fotografia de frente da face sul: c. 1,6 m por baixo do parapeito)
Z_MISULA_CUNHAL = 212.85 # a mísula do cunhal desce mais uma fiada
Z_BEIRAL = 214.60        # beiral do telhado de quatro águas por dentro do parapeito (LiDAR)
Z_CUME = 216.15          # cume do telhado (LiDAR)
ESP_PAR = 0.80           # espessura do parapeito (o telhado começa a 4,4 m do centro, LiDAR)
LB = 2.45                # comprimento de cada balcão ao longo da face, a partir do cunhal (fotografia de frente)
PB = 0.55                # saliência dos balcões (LiDAR: 0,5 a 0,7 m; fotografia de frente: c. 0,5 m)
EB = 0.30                # espessura do parapeito dos balcões


def W(u, v):
    """Ponto do referencial da torre (u, v) para a cena (x este, y norte)."""
    return EU * u + EV * v


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def amostra(G, x, y):
    fx = (np.atleast_1d(np.asarray(x, float)) + MEIO_L) / LPASSO
    fy = (np.atleast_1d(np.asarray(y, float)) + MEIO_L) / LPASSO
    r = ndi.map_coordinates(G, [fy, fx], order=1, mode="nearest")
    return r if np.ndim(x) else float(r[0])


def cota(x, y):
    return amostra(MDT, x, y)


# ---------------------------------------------------------------- ruído


# ---------------------------------------------------------------- malhas


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

    def caixa_uv(self, u0, u1, v0, v1, z0, z1, mat=0, topo=True, fundo=False):
        """Caixa alinhada com a torre (limites em u e v)."""
        q = [W(u0, v0), W(u1, v0), W(u1, v1), W(u0, v1)]
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

    def objeto(self, nome, elemento, mats, pai=None, uv=3.0, liso=False, vcor=None):
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
        if vcor is not None:  # cor por vértice (patine da pedra), multiplicada pela textura
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            cv = vcor(co.reshape(-1, 3))
            at = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
            at.data.foreach_set("color", np.concatenate([cv, np.ones((len(cv), 1))], 1).astype(np.float32).ravel())
        return ob


# ---------------------------------------------------------------- materiais e texturas


def textura_cantaria_torre(n=1024, metros=6.0, semente=1300, fiada=(0.30, 0.48), comp=(0.55, 1.6), liquen=0.80, tinta=(1.0, 1.0, 1.0), sat=1.0):
    """Cantaria de granito da torre (6 m por repetição), medida nas fotografias: aparelho em fiadas
    horizontais de 0,30 a 0,48 m (média 0,38 m, pela periodicidade das juntas na fotografia da face sul),
    silhares de 0,55 a 1,6 m, bem esquadriados, com juntas finas e as arestas gastas.
    Granito cinzento do Minho, de grão médio: a cor de fundo é um cinzento ligeiramente quente (mediana
    149/137/117 nas fotografias ao sol, que inclui as manchas de óxido); a diferença entre silhares é pequena
    (cinzento mais claro ou mais escuro, alguns tingidos de ocre pelo óxido de ferro, raros rosados) e o
    grão vê-se de perto: pintas escuras de biotite e claras de feldspato. Escorrências verticais escuras e
    líquenes claros. Devolve a cor e o relevo."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    paleta = np.array([
        (0.585, 0.555, 0.505),  # cinzento do granito, quente (a maioria)
        (0.645, 0.615, 0.560),  # cinzento claro
        (0.480, 0.460, 0.425),  # cinzento escuro, gasto e sujo
        (0.610, 0.555, 0.470),  # tingido de ocre (óxido de ferro)
        (0.570, 0.500, 0.415),  # ocre-acastanhado
        (0.600, 0.540, 0.500),  # rosado ténue
    ])
    prob = np.array([0.42, 0.16, 0.18, 0.15, 0.05, 0.04])
    cel = np.zeros((n, n), int)
    cores = []
    y = 0.0
    k = 0
    while y < n - 1:
        hf = r.uniform(*fiada) * m_px
        if y + hf > n - 0.26 * m_px:
            hf = n - y
        y0, y1 = int(round(y)), int(round(y + hf))
        x = r.uniform(0, 1.0) * m_px
        x_ini = x
        while x < x_ini + n:
            wf = r.uniform(*comp) * m_px
            if r.random() < 0.08:
                wf = r.uniform(0.35, 0.5) * m_px
            xs = (np.arange(int(round(x)), int(round(x + wf)))) % n
            cel[y0:y1, xs] = k
            cores.append(paleta[r.choice(len(paleta), p=prob)] * r.uniform(0.85, 1.12))
            k += 1
            x += wf
        y += hf
    cores = np.array(cores)
    yy, xx = np.mgrid[0:n, 0:n]
    wx = ((fbm(xx, yy, 0.2 * m_px, semente + 3) - 0.5) * 0.04 * m_px).astype(int)
    wy = ((fbm(xx, yy, 0.2 * m_px, semente + 4) - 0.5) * 0.03 * m_px).astype(int)
    celw = cel[(yy + wy) % n, (xx + wx) % n]
    tex = cores[celw]
    borda = np.zeros((n, n), bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        borda |= celw != np.roll(np.roll(celw, dy, 0), dx, 1)
    dist = ndi.distance_transform_edt(~borda) / m_px  # metros até à junta
    junta = dist < 0.005
    gasto = np.clip(dist / 0.04, 0, 1)  # arestas arredondadas, gastas
    alt = np.sqrt(gasto) * (~junta)
    tex = tex * (0.88 + 0.12 * alt[..., None])
    # juntas: argamassa cinzenta, pouco contrastada (no granito as juntas quase não se notam ao longe)
    tex = np.where(junta[..., None], tex * 0.62 + np.array([0.40, 0.39, 0.37]) * 0.38, tex)
    # manchas suaves dentro de cada silhar
    tex = tex * (0.92 + 0.16 * fbm(xx, yy, 0.30 * m_px, semente + 5))[..., None]
    # grão do granito: pintas escuras (biotite) e claras (feldspato, quartzo), de 1 a 2 píxeis (6 a 12 mm)
    g1 = ndi.gaussian_filter(r.random((n, n)), 0.6, mode="wrap")
    g1 = (g1 - g1.mean()) / g1.std()
    g2 = r.random((n, n))
    tex = tex * (1 + 0.05 * g1[..., None])
    tex = np.where((g2 < 0.045)[..., None], tex * 0.62, tex)
    tex = np.where((g2 > 0.93)[..., None], tex * 0.80 + 0.17, tex)
    # óxido: manchas ocre que passam de um silhar a outro
    ox = np.clip((fbm(xx, yy, 0.6 * m_px, semente + 8) - 0.58) * 3.0, 0, 1)[..., None]
    tex = tex * (1 - 0.30 * ox) + np.array([0.60, 0.50, 0.38]) * 0.30 * ox
    # escorrências verticais escuras (água da chuva), esticadas na vertical
    esc = fbm(xx * 6.0, yy * 0.25, 0.9 * m_px, semente + 9)
    esc = np.clip((esc - 0.55) * 2.2, 0, 1)[..., None]
    tex = tex * (1 - 0.20 * esc)
    # líquenes claros, de vez em quando
    liq = fbm(xx, yy, 0.25 * m_px, semente + 6)
    tex = np.where((liq > liquen)[..., None], tex * 0.60 + np.array([0.78, 0.78, 0.73]) * 0.40, tex)
    # saturação: o visor (three.js, sem a transformação de cor do Cycles) mostra as cores tal como estão na
    # textura; o granito da torre visto ao natural é cinzento, com o tom quente só nas manchas de óxido
    lum = (tex @ np.array([0.30, 0.59, 0.11]))[..., None]
    tex = lum + sat * (tex - lum)
    return np.clip(tex * np.array(tinta), 0, 1), ndi.gaussian_filter(alt, 0.8) + 0.05 * g1


def textura_vitral(n=512, metros=2.0):
    """Vidraça de chumbo em losangos (janelas geminadas da face sul, fotografias)."""
    yy, xx = np.mgrid[0:n, 0:n] / n * metros
    p = 0.11  # losangos de c. 8 × 8 cm (fotografias), nos UV de 2 m
    a = np.abs(((xx + yy) / p) % 1 - 0.5)
    b = np.abs(((xx - yy) / p) % 1 - 0.5)
    chumbo = (a > 0.44) | (b > 0.44)
    vidro = np.array([0.30, 0.36, 0.42]) * (0.8 + 0.4 * fbm(xx * 40, yy * 40, 1.0, 5))[..., None]
    return np.where(chumbo[..., None], np.array([0.62, 0.62, 0.60]), vidro).clip(0, 1)


def textura_cubos(n=1024, metros=8.0, semente=91):
    """Calçada de cubos de granito da praça (cubos de c. 10 cm, juntas de areia) e faixas de lajes."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    c = 0.105 * m_px
    gi = (xx // c).astype(int) + 1000 * ((yy + (xx // (3 * c)).astype(int) % 2 * c * 0.5) // c).astype(int)
    tom = r.uniform(0.85, 1.12, gi.max() + 1)[gi]
    fx, fy = (xx % c) / c, ((yy + (xx // (3 * c)).astype(int) % 2 * c * 0.5) % c) / c
    junta = (fx < 0.12) | (fy < 0.12)
    base = np.array([0.60, 0.58, 0.55])
    tex = base * tom[..., None] * (0.9 + 0.2 * fbm(xx, yy, 1.5 * m_px, semente + 1))[..., None]
    tex = np.where(junta[..., None], np.array([0.40, 0.38, 0.35]), tex)
    return np.clip(tex, 0, 1), (~junta).astype(float)


# ---------------------------------------------------------------- cena

RAIZ_OB = bpy.data.objects.new("Torre_Menagem_Braga", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = CFG["origem"]["lat"], CFG["origem"]["lon"]
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
for g in ["Terreno", "Torre", "Envolvente", "Vegetacao"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

TEX_T, ALT_T = textura_cantaria_torre(tinta=(1.08, 1.07, 1.05), sat=0.45)
material("cantaria", (1, 1, 1), 0.9, imagem("cantaria", TEX_T), normal=imagem("cantaria_normal", normais(ALT_T, 3.0)),
         forca=0.8, escala=0.5, vcor=True)
# a escada: silhares maiores (fiadas de c. 0,5 m), granito mais claro e cinzento, com mais líquenes (fotografias)
TEX_E, ALT_E = textura_cantaria_torre(semente=1400, fiada=(0.42, 0.58), comp=(0.7, 1.5), liquen=0.70, tinta=(1.05, 1.05, 1.04), sat=0.45)
material("cantaria_escada", (1, 1, 1), 0.9, imagem("cantaria_escada", TEX_E), normal=imagem("cantaria_escada_normal", normais(ALT_E, 3.0)),
         forca=0.8, escala=0.5, vcor=True)
material("cantaria_lisa", (0.60, 0.59, 0.57), 0.85)
material("sombra", (0.07, 0.065, 0.06), 1.0)
material("vitral", (1, 1, 1), 0.3, imagem("vitral", textura_vitral()))
material("madeira", (0.24, 0.16, 0.10), 0.8)
material("madeira_azul", (0.16, 0.24, 0.42), 0.6)
material("ferro", (0.13, 0.13, 0.14), 0.45)
TEX_TELHA, ALT_TELHA = textura_telha()
material("telha", (1, 1, 1), 0.8, imagem("telha", TEX_TELHA * np.array([0.74, 0.42, 0.28])),
         normal=imagem("telha_normal", normais(ALT_TELHA, 2.0)), forca=0.6)
TEX_CUB, ALT_CUB = textura_cubos()
material("cubos", (1, 1, 1), 0.92, imagem("cubos", TEX_CUB), normal=imagem("cubos_normal", normais(ALT_CUB, 1.5)), forca=0.5)
material("lajes", (0.66, 0.64, 0.60), 0.9)
material("granito_padrao", (0.66, 0.65, 0.61), 0.8)
material("asfalto", (0.30, 0.30, 0.31), 0.95)
# casas vizinhas: um só cinzento claro, sem cor nem textura, para o destaque ficar todo na torre
material("cinza_viz", (0.74, 0.74, 0.73), 0.9)
material("base", (0.33, 0.26, 0.20), 0.95)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)

MEDIDAS = {}


def patine(co):
    """Cor por vértice da cantaria: a pedra escurece e esverdeia para o alto da torre (líquenes e musgo,
    fotografias), e é um pouco mais escura junto ao chão (salpicos e humidade)."""
    z = co[:, 2]
    t = np.clip((z - 203.0) / 12.0, 0, 1) ** 1.3
    base = np.stack([1.0 - 0.20 * t, 1.0 - 0.15 * t, 1.0 - 0.17 * t], 1)
    pe = np.clip(1 - (z - CHAO) / 1.5, 0, 1)[:, None] * 0.12
    return np.clip(base * (1 - pe), 0, 1)


# ---------------------------------------------------------------- faces da torre

class Face:
    """Uma face da torre: n = normal para fora (em uv), t = eixo ao longo da face, da esquerda para a direita
    de quem está à frente dela; meia = meia largura da face; off = distância do centro à face."""

    def __init__(self, nome, n, off, meia):
        self.nome = nome
        self.n = np.array(n, float)
        self.t = np.array([self.n[1], -self.n[0]])  # -n × z
        self.off = off
        self.meia = meia

    def p(self, s, d, z):
        """Ponto da face: s ao longo da face, d para fora (negativo = para dentro da parede)."""
        uv = self.n * (self.off + d) + self.t * s
        q = W(*uv)
        return (q[0], q[1], z)

    def xy(self, s, d):
        uv = self.n * (self.off + d) + self.t * s
        return W(*uv)


FACES = {
    "ENE": Face("ENE", (1, 0), A, B),
    "NNO": Face("NNO", (0, 1), B, A),
    "OSO": Face("OSO", (-1, 0), A, B),
    "SSE": Face("SSE", (0, -1), B, A),
}


def parede(m, f, s0, s1, z0, z1, vaos, mat=0, passo_z=1.5):
    """Plano da face entre s0..s1 e z0..z1, com vãos retangulares ou em arco abertos.
    vaos: lista de (sa, sb, za, zb, arco) — arco=True: o topo zb é o fecho do arco de volta perfeita."""
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
                quad(m, f, a, b, za + (zb - za) * i / nz, za + (zb - za) * (i + 1) / nz, mat=mat)
    # tímpanos dos vãos em arco: entre o arco e os cantos do retângulo
    for sa, sb, za, zb, arco in vaos:
        if not arco:
            continue
        r = (sb - sa) / 2
        zc = zb - r
        pts = arco_pts((sa + sb) / 2, zc, r, 12)
        meio = len(pts) // 2
        # metade direita (do arranque ao fecho) com o canto de cima à direita; idem à esquerda
        m.face([f.p(s, 0, z) for s, z in pts[:meio + 1]] + [f.p(sb, 0, zb)], mat)
        m.face([f.p(s, 0, z) for s, z in pts[meio:]] + [f.p(sa, 0, zb)], mat)


def vao(m, f, sc, w, za, zb, prof, arco=True, mat_fundo="sombra", mats=None, k=12, d0=0.0):
    """Ombreiras, soleira e intradorso do vão (para dentro da parede até prof) e o fundo."""
    mi = mats.index
    r = w / 2
    sa, sb = sc - r, sc + r
    zc = zb - r if arco else zb
    contorno = [(sa, za), (sb, za)]
    if arco:
        contorno += arco_pts(sc, zc, r, k)
    else:
        contorno += [(sb, zb), (sa, zb)]
    # faces laterais (do plano da face até ao fundo)
    pts = contorno + [contorno[0]]
    for (s1, z1), (s2, z2) in zip(pts[:-1], pts[1:]):
        m.face([f.p(s1, d0, z1), f.p(s2, d0, z2), f.p(s2, d0 - prof, z2), f.p(s1, d0 - prof, z1)], mi("cantaria"))
    m.face([f.p(s, d0 - prof, z) for s, z in contorno], mi(mat_fundo))
    return contorno


# ---------------------------------------------------------------- torre

MAT_T = ["cantaria", "cantaria_lisa", "sombra", "vitral", "madeira", "madeira_azul", "ferro", "telha"]
mi = MAT_T.index

# vãos por face (medidos nas fotografias, à escala da largura da face e da altura da torre)
# SSE: janela geminada (duas luzes de 0,52 × 1,35 m em arco de volta perfeita, mainel de 0,30 m) a 211,5-212,85,
# um pouco a poente do eixo (0,47 da largura na fotografia de frente: 87,5 px/m na horizontal; a fotografia foi
# endireitada e a escala vertical, tirada da altura medida no LiDAR, é 103 px/m), e duas seteiras por baixo.
GEM_S = -0.31
VAOS = {
    "SSE": [(GEM_S - 0.67, GEM_S - 0.15, 211.50, 212.85, True), (GEM_S + 0.15, GEM_S + 0.67, 211.50, 212.85, True),
            (-0.40, -0.26, 207.70, 208.90, False), (-0.20, -0.06, 203.30, 204.60, False)],
    # NNO: uma janela em arco (1,1 m) a 211,9-213,3, um pouco a nascente do eixo (fotografia de 2006)
    "NNO": [(0.60 - 0.55, 0.60 + 0.55, 211.90, 213.30, True)],
    # ENE: a porta alta, no topo da escada, em arco de volta perfeita (Wikipédia: porta em arco no lado
    # nordeste, com o escudo de D. Dinis por cima); janela em arco ao nível do telhado (estimativa)
    "ENE": [(-1.65, -0.65, 196.60, 198.95, True), (-0.50, 0.50, 211.90, 213.30, True)],
    # OSO: janela em arco ao nível do telhado (Wikipédia: janelas em arco em todas as faces; estimativa)
    "OSO": [(-0.50, 0.50, 211.90, 213.30, True)],
}


def torre():
    m = Malha()
    mb, mj, mp = Malha(), Malha(), Malha()  # balcões, janela geminada e porta: elementos próprios no visor
    zb = CHAO - 0.6  # a torre entra no terreno
    # soco ligeiramente escalonado junto ao chão (Wikipédia; fotografias da face sul)
    for sal, z1 in ((0.20, CHAO + 0.45), (0.10, CHAO + 0.95)):
        m.caixa_uv(-A - sal, A + sal, -B - sal, B + sal, zb, z1, mi("cantaria"))
    for nome, f in FACES.items():
        L = f.meia
        # pano da face: até às ameias no meio, até à base dos balcões nos cunhais
        parede(m, f, -L, -L + LB, CHAO + 0.9, Z_BALC_BASE, [v_ for v_ in VAOS[nome] if v_[1] <= -L + LB], mi("cantaria"))
        parede(m, f, -L + LB, L - LB, CHAO + 0.9, Z_AMEIA, [v_ for v_ in VAOS[nome] if -L + LB < v_[0] and v_[1] < L - LB], mi("cantaria"))
        parede(m, f, L - LB, L, CHAO + 0.9, Z_BALC_BASE, [v_ for v_ in VAOS[nome] if v_[0] >= L - LB], mi("cantaria"))
        # face de dentro do parapeito e topo do parapeito (soleira das ameias)
        s0, s1 = -L + LB, L - LB
        quad(m, f, s1, s0, Z_BEIRAL - 0.1, Z_AMEIA, d=-ESP_PAR, mat=mi("cantaria"))
        m.face([f.p(s0, 0, Z_AMEIA), f.p(s1, 0, Z_AMEIA), f.p(s1, -ESP_PAR, Z_AMEIA), f.p(s0, -ESP_PAR, Z_AMEIA)], mi("cantaria"))
        # degrau entre a soleira das ameias e o chão dos balcões, por dentro, nas pontas do troço do meio
        for s_d in (s0, s1):
            m.face([f.p(s_d, -ESP_PAR, Z_BALC_BASE), f.p(s_d, 0, Z_BALC_BASE), f.p(s_d, 0, Z_AMEIA), f.p(s_d, -ESP_PAR, Z_AMEIA)], mi("cantaria"))
        # merlões do pano: três, de 0,68 m, com remate piramidal (fotografias: merlões chanfrados em ponta)
        n_m = 3
        wm = 0.68
        ameia = ((s1 - s0) - n_m * wm) / (n_m + 1)
        for i in range(n_m):
            a = s0 + ameia * (i + 1) + wm * i
            merlao(m, f, a, a + wm, 0.0, -ESP_PAR, Z_AMEIA, Z_MERLAO - 0.42, Z_MERLAO)
        # seteiras nos merlões de fora da face norte (luz passa por elas na fotografia de 2006)
        if nome == "NNO":
            for i in (0, 2):
                a = s0 + ameia * (i + 1) + wm * i + wm / 2
                m.face([f.p(a - 0.04, 0.005, Z_AMEIA + 0.25), f.p(a + 0.04, 0.005, Z_AMEIA + 0.25),
                        f.p(a + 0.04, 0.005, Z_AMEIA + 0.62), f.p(a - 0.04, 0.005, Z_AMEIA + 0.62)], mi("sombra"))
        # gárgula a meio da face, por baixo da soleira, um pouco à direita do eixo (fotografias da face sul)
        gargula(m, f, 0.6, Z_AMEIA - 0.45)
        # vãos
        for sa, sb, za, zb_, arco in VAOS[nome]:
            w = sb - sa
            sc = (sa + sb) / 2
            if nome == "ENE" and za < 200:
                porta(mp, f, sc, w, za, zb_)
            elif nome == "SSE" and zb_ - za > 1.0 and w < 0.6:  # luzes da geminada
                vao(mj, f, sc, w, za, zb_, 0.32, arco, "vitral", MAT_T)
            elif w < 0.2:  # seteiras
                vao(m, f, sc, w, za, zb_, 0.6, False, "sombra", MAT_T)
                # pedras da moldura: lintel e soleira um pouco salientes
                m.caixa_uv(*_box_face(f, sc - 0.28, sc + 0.28, 0.0, 0.04), zb_, zb_ + 0.22, mi("cantaria_lisa"))
                m.caixa_uv(*_box_face(f, sc - 0.22, sc + 0.22, 0.0, 0.04), za - 0.18, za, mi("cantaria_lisa"))
            else:
                fundo = "madeira_azul" if nome == "NNO" else "vitral"
                vao(m, f, sc, w, za, zb_, 0.35, arco, fundo, MAT_T)
                aduelas(m, f, sc, w, za, zb_, 0.28, 0.035)
                m.caixa_uv(*_box_face(f, sa - 0.15, sb + 0.15, -0.02, 0.10), za - 0.14, za, mi("cantaria_lisa"))
        if nome == "SSE":
            janela_geminada(mj, f)
    # balcões dos cunhais
    for su in (-1, 1):
        for sv in (-1, 1):
            balcao(mb, su, sv)
    # telhado de quatro águas por dentro do parapeito (LiDAR: beiral 214,6, cume 216,15)
    ri, rv = A - ESP_PAR, B - ESP_PAR
    telhado(m, ri, rv, Z_BEIRAL, Z_CUME)
    ob = m.objeto("torre", "torre", MAT_T, GRUPOS["Torre"], uv=2.0, vcor=patine)
    mb.objeto("balcoes", "balcoes", MAT_T, GRUPOS["Torre"], uv=2.0, vcor=patine)
    mj.objeto("janela_geminada", "janela_geminada", MAT_T, GRUPOS["Torre"], uv=2.0, vcor=patine)
    mp.objeto("porta", "porta", MAT_T, GRUPOS["Torre"], uv=2.0, vcor=patine)
    return ob


def _box_face(f, s0, s1, d0, d1):
    """Limites (u0, u1, v0, v1) de uma caixa encostada a uma face, de s0 a s1 e de d0 a d1 para fora."""
    a = f.n * (f.off + d0) + f.t * s0
    b = f.n * (f.off + d1) + f.t * s1
    return min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])


def merlao(m, f, sa, sb, d_fora, d_dentro, z0, z1, zp):
    """Merlão com remate piramidal, entre sa..sb, de d_dentro a d_fora (em relação à face)."""
    q = [W(*(f.n * (f.off + d_fora) + f.t * sa)), W(*(f.n * (f.off + d_fora) + f.t * sb)),
         W(*(f.n * (f.off + d_dentro) + f.t * sb)), W(*(f.n * (f.off + d_dentro) + f.t * sa))]
    m.prisma(q, z0, z1, mi("cantaria"), topo=False)
    # pequena aba na base do remate (o chanfro das fotografias) e a pirâmide
    m.piramide(q, z1, zp, mi("cantaria"))


def gargula(m, f, s, z):
    """Gárgula de pedra que despeja a água do telhado, saliente c. 0,75 m e um pouco inclinada."""
    for k in range(4):
        d0, d1 = 0.18 * k, 0.18 * (k + 1) + 0.02
        dz = -0.05 * k
        w = 0.24 - 0.025 * k
        q = [f.xy(s - w / 2, d0), f.xy(s + w / 2, d0), f.xy(s + w / 2, d1), f.xy(s - w / 2, d1)]
        m.prisma(q, z + dz - 0.22, z + dz, mi("cantaria"), fundo=True)
    # boca
    m.face([f.p(s - 0.05, 0.75, z - 0.30), f.p(s + 0.05, 0.75, z - 0.30), f.p(s + 0.05, 0.75, z - 0.22),
            f.p(s - 0.05, 0.75, z - 0.22)], mi("sombra"))


def aduelas(m, f, sc, w, za, zb, larg, sal, n=9):
    """Aduelas do arco e ombreiras de cantaria, ligeiramente salientes da face."""
    r = w / 2
    zc = zb - r
    R = r + larg
    for i in range(n):
        a0, a1 = math.pi * i / n, math.pi * (i + 1) / n
        g = 0.012
        pts = [(sc + r * math.cos(a0 + g), zc + r * math.sin(a0 + g)), (sc + R * math.cos(a0 + g), zc + R * math.sin(a0 + g)),
               (sc + R * math.cos(a1 - g), zc + R * math.sin(a1 - g)), (sc + r * math.cos(a1 - g), zc + r * math.sin(a1 - g))]
        m.face([f.p(s, sal, z) for s, z in pts], mi("cantaria_lisa"))
        for (s1, z1), (s2, z2) in zip(pts, pts[1:] + pts[:1]):
            m.face([f.p(s1, 0, z1), f.p(s2, 0, z2), f.p(s2, sal, z2), f.p(s1, sal, z1)], mi("cantaria_lisa"))
    for s0, s1 in ((sc - r - larg, sc - r), (sc + r, sc + r + larg)):
        hz = zc - za
        k = max(1, int(round(hz / 0.42)))
        for j in range(k):
            z0_, z1_ = za + hz * j / k + 0.008, za + hz * (j + 1) / k - 0.008
            m.caixa_uv(*_box_face(f, s0, s1, 0.0, sal), z0_, z1_, mi("cantaria_lisa"))


def janela_geminada(m, f):
    """Janela geminada da face sul: duas luzes em arco de volta perfeita com mainel, vidraça de chumbo em
    losangos, moldura em arco de descarga por cima das duas e um peitoril saliente (fotografias)."""
    sc = GEM_S
    # mainel entre as luzes (pedra inteira, à face)
    # arco de descarga: anel de aduelas finas por cima das duas luzes
    r = 0.80
    zc = 212.19
    n = 11
    for i in range(n):
        a0, a1 = math.pi * i / n, math.pi * (i + 1) / n
        pts = [(sc + r * math.cos(a0), zc + r * math.sin(a0)), (sc + (r + 0.16) * math.cos(a0), zc + (r + 0.16) * math.sin(a0)),
               (sc + (r + 0.16) * math.cos(a1), zc + (r + 0.16) * math.sin(a1)), (sc + r * math.cos(a1), zc + r * math.sin(a1))]
        m.face([f.p(s, 0.025, z) for s, z in pts], mi("cantaria_lisa"))
    # ombreiras e arquinhos das luzes, um pouco salientes
    for c in (sc - 0.41, sc + 0.41):
        aduelas(m, f, c, 0.52, 211.50, 212.85, 0.10, 0.02, n=5)
    # peitoril saliente de c. 2,1 m (fotografias)
    m.caixa_uv(*_box_face(f, sc - 1.05, sc + 1.05, -0.02, 0.12), 211.30, 211.50, mi("cantaria_lisa"))


def porta(m, f, sc, w, za, zb):
    """Porta alta da face nascente, em arco de volta perfeita com aduelas, porta de madeira recuada, e o
    escudo de D. Dinis por cima (Wikipédia)."""
    vao(m, f, sc, w, za, zb, 0.45, True, "madeira", MAT_T)
    aduelas(m, f, sc, w, za, zb, 0.36, 0.04, n=11)
    brasao(m, f, sc, zb + 0.75)


def brasao(m, f, sc, z0, larg=0.62, alt=0.78):
    """Escudo real (de D. Dinis): campo com as cinco quinas em cruz e bordadura com castelos, dentro de
    uma pedra lisa saliente. Desenho esquemático, à escala das fotografias."""
    m.caixa_uv(*_box_face(f, sc - larg / 2 - 0.14, sc + larg / 2 + 0.14, 0.0, 0.05), z0 - 0.14, z0 + alt + 0.14, mi("cantaria_lisa"))
    # escudo de ponta redonda
    pts = [(sc - larg / 2, z0 + alt), (sc + larg / 2, z0 + alt)]
    pts += [(sc + larg / 2 * math.cos(a), z0 + larg / 2 + larg / 2 * math.sin(a)) for a in np.linspace(0, -math.pi, 9)]
    d = 0.09
    m.face([f.p(s, d, z) for s, z in pts], mi("cantaria_lisa"))
    for (s1, z1), (s2, z2) in zip(pts, pts[1:] + pts[:1]):
        m.face([f.p(s1, 0.05, z1), f.p(s2, 0.05, z2), f.p(s2, d, z2), f.p(s1, d, z1)], mi("cantaria_lisa"))
    zc = z0 + alt * 0.52
    q = 0.085
    for ds, dz in ((0, 0), (-0.13, 0), (0.13, 0), (0, 0.15), (0, -0.15)):  # as cinco quinas
        m.caixa_uv(*_box_face(f, sc + ds - q / 2, sc + ds + q / 2, d, d + 0.025), zc + dz - q * 0.6, zc + dz + q * 0.6, mi("cantaria_lisa"))
    for a in np.linspace(math.pi * 0.95, math.pi * 0.05, 7):  # castelos da bordadura (sete, esquemáticos)
        s_, z_ = sc + 0.25 * math.cos(a) * 1.1, zc + 0.30 * math.sin(a) * 0.95 - 0.02
        if z_ < z0 + 0.12:
            continue
        m.caixa_uv(*_box_face(f, s_ - 0.03, s_ + 0.03, d, d + 0.02), z_ - 0.035, z_ + 0.035, mi("cantaria_lisa"))


def balcao(m, su, sv):
    """Balcão com matacães num cunhal: parapeito saliente (PB) sobre três mísulas de quatro fiadas em cada
    face, com os matacães abertos entre elas, e merlões piramidais por cima (fotografias)."""
    # as duas faces do cunhal
    fu = FACES["ENE"] if su > 0 else FACES["OSO"]
    fv = FACES["NNO"] if sv > 0 else FACES["SSE"]
    for f, outra_sinal in ((fu, sv), (fv, su)):
        L = f.meia
        # lado do cunhal em s: +L ou -L conforme o t da face
        ext = None
        for s_ in (L, -L):
            uvp = f.n * f.off + f.t * s_
            if (np.sign(uvp[0]) == su) and (np.sign(uvp[1]) == sv):
                ext = s_
        sg = np.sign(ext)
        s_ini = ext - sg * LB  # ponta de dentro do balcão
        s_fim = ext + sg * PB  # o parapeito dá a volta ao cunhal (sai PB da outra face)
        a_, b_ = sorted((s_ini, s_fim))
        # parapeito (face de fora, de dentro, topo e fundo)
        quad(m, f, a_, b_, Z_BALC_BASE, Z_BALCAO, d=PB, mat=mi("cantaria"))
        quad(m, f, b_, a_, Z_BALC_BASE, Z_BALCAO, d=PB - EB, mat=mi("cantaria"))
        m.face([f.p(a_, PB - EB, Z_BALCAO), f.p(b_, PB - EB, Z_BALCAO), f.p(b_, PB, Z_BALCAO), f.p(a_, PB, Z_BALCAO)], mi("cantaria"))
        m.face([f.p(a_, PB, Z_BALC_BASE), f.p(b_, PB, Z_BALC_BASE), f.p(b_, PB - EB, Z_BALC_BASE), f.p(a_, PB - EB, Z_BALC_BASE)], mi("cantaria"))
        # topo do pano por baixo do balcão (até ao parapeito de dentro do telhado)
        si_, sf_ = sorted((s_ini, ext))
        m.face([f.p(si_, -ESP_PAR, Z_BALC_BASE), f.p(sf_, -ESP_PAR, Z_BALC_BASE), f.p(sf_, 0, Z_BALC_BASE), f.p(si_, 0, Z_BALC_BASE)], mi("cantaria"))
        # retorno do parapeito na ponta de dentro (fecha o balcão contra a face)
        m.face([f.p(s_ini, 0, Z_BALC_BASE), f.p(s_ini, PB, Z_BALC_BASE), f.p(s_ini, PB, Z_BALCAO), f.p(s_ini, 0, Z_BALCAO)], mi("cantaria"))
        m.face([f.p(s_ini, 0, Z_BALCAO), f.p(s_ini, PB, Z_BALCAO), f.p(s_ini + sg * EB, PB - EB, Z_BALCAO), f.p(s_ini + sg * EB, 0, Z_BALCAO)], mi("cantaria"))
        m.face([f.p(s_ini + sg * EB, 0, Z_BALC_BASE), f.p(s_ini + sg * EB, PB - EB, Z_BALC_BASE), f.p(s_ini + sg * EB, PB - EB, Z_BALCAO), f.p(s_ini + sg * EB, 0, Z_BALCAO)], mi("cantaria"))
        m.face([f.p(s_ini, 0, Z_BALC_BASE), f.p(s_ini + sg * EB, 0, Z_BALC_BASE), f.p(s_ini + sg * EB, PB - EB, Z_BALC_BASE), f.p(s_ini, PB - EB, Z_BALC_BASE)], mi("cantaria"))
        # parapeito de dentro do balcão, por cima da torre (até ao telhado)
        quad(m, f, sf_, si_, Z_BEIRAL - 0.1, Z_BALC_BASE, d=-ESP_PAR, mat=mi("cantaria"))
        # mísulas: três por face (no cunhal, a meio e na ponta de dentro), quatro fiadas arredondadas
        # a do cunhal só uma vez (nas faces ENE/OSO), mais comprida, encostada ao cunhal
        if f is fu:
            misula(m, f, ext - sg * 0.22, z_pe=Z_MISULA_CUNHAL, fiadas=5)
        for s_m in (ext - sg * (LB / 2 + 0.05), s_ini + sg * 0.22):
            misula(m, f, s_m)
        # merlões (fotografia de frente): no cunhal (quadrado), dois pelo meio e um na ponta de dentro
        wm = 0.50
        for s_c in (ext + sg * PB - sg * 1.20, ext + sg * PB - sg * 2.10, s_ini + sg * (wm / 2)):
            merlao(m, f, s_c - wm / 2, s_c + wm / 2, PB, PB - EB, Z_BALCAO, Z_BALC_MER - 0.36, Z_BALC_MER)
    # merlão do cunhal: quadrado, no canto de fora
    c_u, c_v = su * (A + PB), sv * (B + PB)
    q = [W(c_u, c_v), W(c_u - su * 0.6, c_v), W(c_u - su * 0.6, c_v - sv * 0.6), W(c_u, c_v - sv * 0.6)]
    m.prisma(q, Z_BALCAO, Z_BALC_MER - 0.36, mi("cantaria"), topo=False)
    m.piramide(q, Z_BALC_MER - 0.36, Z_BALC_MER, mi("cantaria"))
    # laje do cunhal (fora das duas faces), por cima da mísula do cunhal
    m.caixa_uv(*sorted((su * A, su * (A + PB))), *sorted((sv * B, sv * (B + PB))), Z_BALC_BASE - 0.12, Z_BALC_BASE + 0.01, mi("cantaria"))
    # chão do balcão por cima da torre e da saliência, só por cima das mísulas (os matacães ficam abertos)
    # (os vãos entre mísulas veem-se de baixo e do alto)


def misula(m, f, s, larg=0.40, z_pe=Z_MISULA, fiadas=4):
    """Mísula de quatro (ou cinco) fiadas, cada uma mais saliente que a de baixo, de frente arredondada,
    com uma laje por cima que leva o parapeito do balcão."""
    alt = (Z_BALC_BASE - z_pe) / fiadas
    for k in range(fiadas):
        z0, z1 = z_pe + alt * k + 0.01, z_pe + alt * (k + 1) - 0.01
        sal = 0.16 + (PB - 0.16) * k / (fiadas - 1)
        # perfil arredondado em planta: meio octógono à frente
        pts = [f.xy(s - larg / 2, 0)]
        for a in np.linspace(math.pi, 0, 7):
            pts.append(f.xy(s + larg / 2 * math.cos(a), sal - 0.12 + 0.12 * math.sin(a)))
        pts.append(f.xy(s + larg / 2, 0))
        # arredondado também na vertical: a fiada estreita um pouco em baixo
        m.prisma(pts, z0, z1, mi("cantaria"), fundo=True)
    # laje sobre a mísula, do pano ao parapeito
    m.caixa_uv(*_box_face(f, s - larg / 2, s + larg / 2, 0.0, PB), Z_BALC_BASE - 0.12, Z_BALC_BASE + 0.01, mi("cantaria"))


def telhado(m, ri, rv, z0, z1):
    """Telhado de quatro águas (cume ao longo de u, o lado maior)."""
    c = ri - rv
    p = [W(-ri, -rv), W(ri, -rv), W(ri, rv), W(-ri, rv)]
    k0, k1 = W(-c, 0), W(c, 0)
    t = mi("telha")
    m.face([(*p[0], z0), (*p[1], z0), (*k1, z1), (*k0, z1)], t)
    m.face([(*p[2], z0), (*p[3], z0), (*k0, z1), (*k1, z1)], t)
    m.face([(*p[1], z0), (*p[2], z0), (*k1, z1)], t)
    m.face([(*p[3], z0), (*p[0], z0), (*k0, z1)], t)
    m.face([(*p[3], z0), (*p[2], z0), (*p[1], z0), (*p[0], z0)], t)


# ---------------------------------------------------------------- escada

# Medida no LiDAR (pontos de 2025): 1.º lanço encostado à face sul, de u 0,7 (praça) a 4,9 (194,55); patamar
# no cunhal sudeste; 2.º lanço encostado à face nascente, de v -4,4 a -2,0; patamar da porta a 196,6 até
# v -0,3. O guarda-corpo de pedra do lado de fora sobe c. 0,9 m acima dos degraus (topo medido: 194,0 a meio do
# 1.º lanço, 195,4 no patamar, 197,55 no alto).
E_V0, E_V1, E_V2 = -B, -B - 0.90, -B - 1.20   # 1.º lanço: piso de v -5,0 a -5,9, guarda até -6,2
E_U0, E_U1 = A, A + 0.90                      # 2.º lanço: piso de u 5,2 a 6,1, guarda até 6,4
E_U2 = A + 1.20
Z_PAT1, Z_PAT2 = 194.55, 196.60
ES_U_INI, ES_U_FIM = 0.40, 4.90  # o bloco começa a c. 5,3 m do cunhal poente (fotografia de frente)
ES_V_INI, ES_V_FIM, ES_V_TOPO = -4.40, -2.00, -0.30
GUARDA = 0.95


MAT_E = MAT_T + ["cantaria_escada"]


def escada():
    m = Malha()
    c = MAT_E.index("cantaria_escada")
    z_pe = CHAO - 0.4
    # 1.º lanço: 17 degraus
    n1 = 18
    du = (ES_U_FIM - ES_U_INI) / n1
    dz = (Z_PAT1 - CHAO) / n1
    for i in range(n1):
        u0 = ES_U_INI + du * i
        m.caixa_uv(u0, u0 + du + 0.02, E_V1, E_V0, z_pe, CHAO + dz * (i + 1), c)
        # focinho do degrau, um pouco saliente
        m.caixa_uv(u0 - 0.03, u0 + 0.05, E_V1, E_V0 - 0.01, CHAO + dz * (i + 1) - 0.06, CHAO + dz * (i + 1), mi("cantaria_lisa"))
    # patamar do cunhal
    m.caixa_uv(ES_U_FIM, E_U1, E_V1, ES_V_INI, z_pe, Z_PAT1, c)
    # 2.º lanço: 10 degraus
    n2 = 10
    dv = (ES_V_FIM - ES_V_INI) / n2
    dz2 = (Z_PAT2 - Z_PAT1) / n2
    for i in range(n2):
        v0 = ES_V_INI + dv * i
        m.caixa_uv(E_U0, E_U1, v0, v0 + dv + 0.02, z_pe, Z_PAT1 + dz2 * (i + 1), c)
        m.caixa_uv(E_U0 + 0.01, E_U1, v0 - 0.03, v0 + 0.05, Z_PAT1 + dz2 * (i + 1) - 0.06, Z_PAT1 + dz2 * (i + 1), mi("cantaria_lisa"))
    # patamar da porta
    m.caixa_uv(E_U0, E_U1, ES_V_FIM, ES_V_TOPO, z_pe, Z_PAT2, c)
    # guarda-corpo de pedra do 1.º lanço (topo inclinado), do patamar e do 2.º lanço
    g0 = CHAO + 1.40  # pilar do portão, mais alto (fotografia da escada)
    q = [W(ES_U_INI, E_V2), W(ES_U_FIM, E_V2), W(ES_U_FIM, E_V1), W(ES_U_INI, E_V1)]
    m.prisma(q, z_pe, [z_pe, z_pe, z_pe, z_pe], c, topo=False)
    zt = [g0, Z_PAT1 + 0.9, Z_PAT1 + 0.9, g0]
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        m.face([(*a, z_pe), (*b, z_pe), (*b, zt[(i + 1) % 4]), (*a, zt[i])], c)
    m.face([(*q[i], zt[i]) for i in range(4)], c)
    m.caixa_uv(ES_U_FIM, E_U2, E_V2, E_V1, z_pe, Z_PAT1 + 0.9, c)
    m.caixa_uv(E_U1, E_U2, E_V2, ES_V_INI, z_pe, Z_PAT1 + 0.9, c)
    q = [W(E_U1, ES_V_INI), W(E_U2, ES_V_INI), W(E_U2, ES_V_FIM), W(E_U1, ES_V_FIM)]
    zt = [Z_PAT1 + 0.9, Z_PAT1 + 0.9, Z_PAT2 + GUARDA, Z_PAT2 + GUARDA]
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        m.face([(*a, z_pe), (*b, z_pe), (*b, zt[(i + 1) % 4]), (*a, zt[i])], c)
    m.face([(*q[i], zt[i]) for i in range(4)], c)
    m.caixa_uv(E_U1, E_U2, ES_V_FIM, ES_V_TOPO + 0.30, z_pe, Z_PAT2 + GUARDA, c)
    m.caixa_uv(E_U0, E_U2, ES_V_TOPO, ES_V_TOPO + 0.30, z_pe, Z_PAT2 + GUARDA, c)
    # capeamento do guarda-corpo (lajes um pouco salientes)
    m.caixa_uv(ES_U_FIM - 0.02, E_U2 + 0.03, E_V2 - 0.03, E_V1, Z_PAT1 + 0.9, Z_PAT1 + 0.98, mi("cantaria_lisa"))
    # portão de ferro ao pé da escada, entre a torre e o pilar (fotografia)
    u_p = ES_U_INI - 0.05
    for k in range(9):
        v_ = E_V0 - 0.06 - (E_V0 - E_V1 - 0.12) * k / 8
        p_ = W(u_p, v_)
        m.cilindro(p_, 0.012, CHAO + 0.05, CHAO + 1.30, mi("ferro"), k=5)
    for z_ in (CHAO + 0.12, CHAO + 0.65, CHAO + 1.22):
        m.caixa_uv(u_p - 0.015, u_p + 0.015, E_V1 + 0.04, E_V0 - 0.04, z_ - 0.025, z_ + 0.025, mi("ferro"))
    # soleira de pedra do portão
    m.caixa_uv(ES_U_INI - 0.25, ES_U_INI + 0.02, E_V1, E_V0, CHAO - 0.2, CHAO + 0.06, mi("cantaria_lisa"))
    return m.objeto("escada", "escada", MAT_E, GRUPOS["Torre"], uv=2.0, vcor=patine)


# ---------------------------------------------------------------- terreno, praça e envolvente

def caminhos():
    """Ruas do OSM com faixa de rodagem (asfalto); o resto é calçada de cubos e lajes."""
    out = []
    for f in F["caminhos"]:
        if f["subtipo"] != "road" or f["geometria"]["type"] != "LineString":
            continue
        cl = f["classe"]
        larg = {"primary": 9.0, "secondary": 8.0}.get(cl)
        if larg is None:
            continue
        out.append(shapely.LineString(f["geometria"]["coordinates"]).buffer(larg / 2, cap_style="flat"))
    return shapely.union_all(out) if out else shapely.Polygon()


def construir_terreno(plantas):
    passo = 0.5
    xs = np.arange(-MEIO, MEIO + 1e-6, passo)
    X, Y = np.meshgrid(xs, xs)
    Z = cota(X, Y)
    # o MDT por baixo das casas é interpolado: alisa-se um pouco
    n = len(xs)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    idx = np.arange(n * n).reshape(n, n)
    Fq = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    ob = malha_rapida("praca", V, Fq, "cubos", GRUPOS["Terreno"], "praca")
    ob.data.materials.append(MATS["asfalto"])
    ruas = caminhos()
    cx = V[Fq].mean(1)
    pts = shapely.points(cx[:, 0], cx[:, 1])
    # o centro de Braga à volta da torre é pedonal: calçada de cubos (fotografias); o asfalto fica só nas
    # vias principais, se as houver dentro da maqueta
    mi_ = shapely.contains(ruas, pts).astype(np.int32) if not ruas.is_empty else np.zeros(len(Fq), np.int32)
    ob.data.polygons.foreach_set("material_index", mi_)
    uv_mundo(ob, 2.0)
    # lados da maqueta
    m = Malha()
    zb = float(Z.min()) - 1.5
    borda = [(X[0, i], Y[0, i], Z[0, i]) for i in range(n)] + [(X[i, -1], Y[i, -1], Z[i, -1]) for i in range(1, n)] + \
            [(X[-1, i], Y[-1, i], Z[-1, i]) for i in range(n - 2, -1, -1)] + [(X[i, 0], Y[i, 0], Z[i, 0]) for i in range(n - 2, 0, -1)]
    for a, b in zip(borda, borda[1:] + borda[:1]):
        m.face([(a[0], a[1], zb), (b[0], b[1], zb), b, a], 0)
    m.objeto("base", None, ["base"], GRUPOS["Terreno"], uv=None)
    return Z


def construir_envolvente():
    """Casas vizinhas em volumes simples: planta do OSM recortada pelo quadrado da maqueta, cornija e cume
    medidos no MDS, telhado de águas até ao cume por recuo da planta."""
    quad_ = shapely.box(-MEIO + 0.3, -MEIO + 0.3, MEIO - 0.3, MEIO - 0.3)
    torre_pl = shapely.Polygon([W(-A - 1.5, -B - 1.5), W(E_U2 + 0.3, -B - 1.5), W(E_U2 + 0.3, B + 1.5), W(-A - 1.5, B + 1.5)])
    m = Malha()
    mats = ["cinza_viz"]
    n = 0
    for f in F["edificios"]:
        g = shapely.geometry.shape(f["geometria"])
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
        if g.geom_type != "Polygon" or g.area < 6:
            continue
        if g.intersects(torre_pl) and g.intersection(torre_pl).area > 0.3 * g.area:
            continue  # a própria torre (o OSM tem-na como edifício)
        g = g.buffer(0).intersection(quad_).difference(torre_pl)
        if g.is_empty:
            continue
        for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            if p.geom_type != "Polygon" or p.area < 6:
                continue
            p = p.simplify(0.3)
            interior = p.buffer(-0.6)
            if interior.is_empty:
                continue
            # cotas do MDS dentro da planta
            minx, miny, maxx, maxy = interior.bounds
            gx, gy = np.meshgrid(np.arange(minx, maxx, 0.5), np.arange(miny, maxy, 0.5))
            dentro = shapely.contains_xy(interior, gx.ravel(), gy.ravel())
            if dentro.sum() < 4:
                continue
            zs = amostra(MDS, gx.ravel()[dentro], gy.ravel()[dentro])
            zt = amostra(MDT, gx.ravel()[dentro], gy.ravel()[dentro])
            chao = float(np.percentile(zt, 10))
            cume = float(np.percentile(zs, 92))
            cornija = float(np.percentile(zs, 25))
            if cume - chao < 2.5:
                continue
            cornija = min(max(cornija, chao + 2.6), cume - 0.4)
            base = np.array(p.exterior.coords)[:-1]
            if shapely.Polygon(base).exterior.is_ccw is False:
                base = base[::-1]
            cor = 0
            m.prisma(base, chao - 0.5, cornija, cor, topo=False)
            # telhado: recuo da planta até ao cume (águas a c. 28°)
            rec = (cume - cornija) / math.tan(math.radians(28))
            top = p.buffer(-rec, join_style="mitre")
            if top.is_empty or top.geom_type != "Polygon" or rec < 0.3:
                m.face([(*q, cornija) for q in base], 0)
            else:
                # águas: liga cada vértice ao ponto mais próximo do contorno recuado
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
            n += 1
    print("casas", n)
    return m.objeto("envolvente", "envolvente", mats, GRUPOS["Envolvente"], uv=2.0)


def construir_arvores():
    """Árvores: máximos locais da vegetação (classes 3 a 5) dentro da maqueta, copas esquemáticas."""
    h = MDS - MDT
    veg = (CLS >= 3) & (CLS <= 5) & (h > 2.0)
    veg = ndi.binary_opening(veg, iterations=1)
    hs = ndi.gaussian_filter(np.where(veg, h, 0.0), 1.2)
    topo = (hs == ndi.maximum_filter(hs, size=17)) & veg & (hs > 3.0)  # uma copa por árvore (8 m)
    lab, k = ndi.label(topo)
    V, Fc, C = [], [], []
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    v0 = np.array([x.co[:] for x in bm.verts])
    f0 = [[x.index for x in fa.verts] for fa in bm.faces]
    bm.free()
    n = 0
    for (r, c) in ndi.center_of_mass(topo, lab, range(1, k + 1)):
        r, c = int(round(r)), int(round(c))
        x, y = LXS[c], LXS[r]
        if abs(x) > MEIO - 2 or abs(y) > MEIO - 2:
            continue
        zt, zc = MDT[r, c], MDS[r, c]
        alt = zc - zt
        if abs(np.array([x, y]) @ EU) < A + 2.5 and abs(np.array([x, y]) @ EV) < B + 2.5:
            continue  # pontos do alto da torre classificados como vegetação: não é árvore
        rad = float(np.clip(np.sqrt(veg[max(r - 12, 0):r + 13, max(c - 12, 0):c + 13].sum() * 0.25 / math.pi), 1.5, 6.0))
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
            V.append((x + 0.25 * math.cos(a), y + 0.25 * math.sin(a), zt - 0.2))
            V.append((x + 0.18 * math.cos(a), y + 0.18 * math.sin(a), zc - alt * 0.45))
        for j in range(6):
            jj = (j + 1) % 6
            Fc.append([b + 2 * j, b + 2 * jj, b + 2 * jj + 1, b + 2 * j + 1])
        C.extend([(0.36, 0.30, 0.24)] * 12)
        n += 1
    print("árvores", n)
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


# Padrão em coluna junto à face norte (fotografia de 2006 da face norte; LiDAR): pedestal baixo quadrado,
# fuste cilíndrico de granito liso, bloco cúbico em cima e cruz de ferro de braços abertos. Fica 1,1 m a poente
# do cunhal noroeste e 1,4 m à frente da face norte (cinco pontos do LiDAR a 193,4 m, 2,1 m acima do chão, que
# são o topo do bloco). As proporções vêm da fotografia, à escala dessa altura: c. 2,5 m até ao alto da cruz.
PAD_U, PAD_V = -6.1, 6.4


def padrao():
    m = Malha()
    mats = ["granito_padrao", "ferro"]
    c = W(PAD_U, PAD_V)
    z0 = float(cota(c[0], c[1]))
    u, v = PAD_U, PAD_V
    z = z0 - 0.15
    m.caixa_uv(u - 0.36, u + 0.36, v - 0.36, v + 0.36, z, z0 + 0.13, 0)          # pedestal
    m.caixa_uv(u - 0.20, u + 0.20, v - 0.20, v + 0.20, z0 + 0.13, z0 + 0.19, 0)  # plinto do fuste
    m.cilindro(c, 0.135, z0 + 0.19, z0 + 0.25, 0, k=16, r1=0.115)               # base moldada
    m.cilindro(c, 0.110, z0 + 0.25, z0 + 1.63, 0, k=16, r1=0.100)               # fuste, um pouco afunilado
    m.cilindro(c, 0.100, z0 + 1.63, z0 + 1.69, 0, k=16, r1=0.135)               # colarinho
    m.caixa_uv(u - 0.17, u + 0.17, v - 0.17, v + 0.17, z0 + 1.69, z0 + 2.06, 0)  # bloco cúbico
    m.caixa_uv(u - 0.19, u + 0.19, v - 0.19, v + 0.19, z0 + 2.06, z0 + 2.11, 0)  # laje de remate
    # cruz de ferro de braços abertos, no plano paralelo à face norte (de frente para quem vem do largo)
    zc0, zc1 = z0 + 2.11, z0 + 2.47
    m.caixa_uv(u - 0.018, u + 0.018, v - 0.012, v + 0.012, zc0, zc1, 1)
    za = zc0 + 0.23
    m.caixa_uv(u - 0.11, u + 0.11, v - 0.012, v + 0.012, za - 0.018, za + 0.018, 1)
    for su in (-1, 1):  # pontas alargadas dos braços
        m.caixa_uv(u + su * 0.11 - 0.015, u + su * 0.11 + 0.015, v - 0.012, v + 0.012, za - 0.045, za + 0.045, 1)
    m.caixa_uv(u - 0.045, u + 0.045, v - 0.012, v + 0.012, zc1 - 0.03, zc1, 1)
    m.caixa_uv(u - 0.045, u + 0.045, v - 0.012, v + 0.012, zc0, zc0 + 0.03, 1)
    return m.objeto("padrao", "padrao", mats, GRUPOS["Torre"], uv=1.0)


# ---------------------------------------------------------------- construção

print("terreno…")
construir_terreno(None)
print("torre…")
torre()
escada()
padrao()
print("envolvente…")
construir_envolvente()
construir_arvores()

# medidas de conferência: o topo da nuvem de pontos dentro da torre
uvp = np.stack([PT[:, :2] @ EU, PT[:, :2] @ EV], 1)
dentro = (np.abs(uvp[:, 0]) < A + 0.8) & (np.abs(uvp[:, 1]) < B + 0.8)
zt = PT[dentro, 2]
MEDIDAS["ponto_mais_alto"] = round(float(zt.max()), 2)
_pp = np.array([W(-7.0, 0), W(7.0, 0), W(0, -7.5), W(0, 7.0)])
MEDIDAS["praca"] = round(float(np.median(cota(_pp[:, 0], _pp[:, 1]))), 2)
MEDIDAS["altura_ate_ameias"] = round(Z_AMEIA - CHAO, 2)
MEDIDAS["altura_ate_pontas"] = round(Z_BALC_MER - CHAO, 2)
MEDIDAS["planta"] = [2 * A, 2 * B]
MEDIDAS["azimute_faces"] = [71.7, 161.7]
def _anc(u, v, z):
    q = W(u, v)
    return [round(float(q[0]), 1), round(float(z), 1), round(float(-q[1]), 1)]


def _olhar(u, v, up=0.35):
    q = W(u, v)
    return [round(float(q[0]), 2), up, round(float(-q[1]), 2)]


MEDIDAS["ancoras"] = {"torre": _anc(0, 0, 219.5), "balcoes": _anc(-A - 0.3, -B - 0.3, 217.6),
                      "janela_geminada": _anc(GEM_S, -B - 0.3, 213.6), "porta": _anc(A + 0.6, -1.15, 200.8),
                      "escada": _anc(3.0, -B - 0.7, 196.0),
                      "padrao": _anc(PAD_U, PAD_V, CHAO + 2.9)}
MEDIDAS["olhar"] = {"janela_geminada": _olhar(0, -1), "porta": _olhar(1, -0.25), "escada": _olhar(0.3, -1, 0.5),
                    "balcoes": _olhar(-0.7, -0.7, 0.3), "padrao": _olhar(-0.2, 1, 0.25)}
(AQUI / "dados" / "medidas.json").write_text(json.dumps(MEDIDAS, ensure_ascii=False, indent=1))
print(MEDIDAS)

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
(AQUI / "saida").mkdir(exist_ok=True)
glb = AQUI / "saida" / "braga-bruto.glb"
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
    final = SAIDA / "braga.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "braga.blend"))

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
    d = Vector((math.sin(math.radians(AZ_SOL)), math.cos(math.radians(AZ_SOL)), 0)) * math.cos(math.radians(52))
    d.z = math.sin(math.radians(52))
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

    def vista(nome, cam_uv, cam_z, alvo_uv, alvo_z, lente=35, res=(1000, 1000), destino=None, perto=0.1, orto=None,
              sem_casas=False):
        """Câmara no referencial da torre (u, v), para repetir os pontos de vista das fotografias."""
        if SO and nome not in SO:
            return
        for ob_ in bpy.data.objects:
            if ob_.get("elemento") in ("envolvente", "vegetacao"):
                ob_.hide_render = sem_casas
        p = W(*cam_uv)
        a = W(*alvo_uv)
        co.location = Vector((p[0], p[1], cam_z))
        alvo = Vector((a[0], a[1], alvo_z))
        co.rotation_euler = (alvo - co.location).to_track_quat("-Z", "Y").to_euler()
        if orto:
            cam.type = "ORTHO"
            cam.ortho_scale = orto
        else:
            cam.type = "PERSP"
            cam.lens = lente
        cam.clip_start = perto
        cam.clip_end = 2000
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.render.filepath = str(destino or (pasta / f"{nome}.jpg"))
        bpy.ops.render.render(write_still=True)
        print("vista", nome)

    def livre(dir_uv, dmax, dmin=9.0, folga=1.2):
        """Ponto mais afastado da torre, na direção dir_uv (u, v) e até dmax, com o chão livre (sem casas
        nem árvores baixas) desde dmin: para as câmaras à altura de quem está na rua."""
        d_ = unit(dir_uv)
        ult = dmin
        for t in np.arange(dmin, dmax, 0.5):
            ok = True
            for lat in (-folga, 0, folga):
                q = W(*(d_ * t + np.array([-d_[1], d_[0]]) * lat))
                ci_ = (int(round((q[1] + MEIO_L) / LPASSO)), int(round((q[0] + MEIO_L) / LPASSO)))
                if abs(q[0]) > MEIO - 1 or abs(q[1]) > MEIO - 1 or (CLS[ci_] == 6 and MDS[ci_] - MDT[ci_] > 1.8):
                    ok = False
            if not ok:
                break
            ult = t
        print("câmara livre", dir_uv, round(ult, 1))
        return tuple(d_ * ult)

    # capa: do lado da praça (sudeste), do alto, como o visor abre
    vista("capa", (26, -48), 246, (0.8, -1.0), 203.5, lente=42, res=(1280, 800), destino=SAIDA / "capa.jpg")
    if "--capa" not in sys.argv:
        # fotografia 3 (face sul de frente, ao fundo da rua)
        vista("f3_sul", livre((-0.02, -1), 45), CHAO + 1.6, (-0.3, 0), 204, lente=30, res=(700, 1050))
        # fotografia 1 (face sul de perto, a olhar para cima, um pouco de nascente)
        vista("f1_sul_perto", (2.5, -14), CHAO + 1.6, (-1.0, 0), 209, lente=20, perto=0.5)
        # fotografia 2 (da praça a sudeste, grande angular, com a escada à direita)
        vista("f2_sudeste", livre((0.25, -1), 16), CHAO + 1.6, (0.5, -2), 201, lente=14, res=(1234, 823), perto=0.5)
        # fotografia 4 (face norte, do largo)
        vista("f4_norte", livre((0.05, 1), 20), CHAO + 1.6, (0.8, 0), 203, lente=24, res=(700, 820), perto=0.5)
        # fotografia 5 (a escada, do pé do portão)
        vista("f5_escada", (-0.6, -5.7), CHAO + 1.5, (5.5, -5.3), CHAO + 2.6, lente=24, perto=0.1)
        vista("nascente", (40, -6), 226, (0, -1), 204, lente=35, res=(900, 1000), perto=1)
        vista("poente", (-40, 4), 226, (0, 0), 204, lente=35, res=(900, 1000), perto=1)
        vista("norte_alto", (6, 40), 226, (0, 0), 204, lente=35, res=(900, 1000), perto=1)
        vista("balcao", (-9, -12), 214, (-5.2, -5.0), 214.5, lente=40, perto=1)
        vista("porta", (8.6, -3.5), 198.6, (5.2, -1.4), 198.8, lente=22, perto=0.3)
        vista("aereo", (-25, -45), 245, (0, 0), 204, lente=40, res=(1200, 900))
        vista("geminada", (GEM_S + 2.5, -16), 207, (GEM_S, -5.0), 212.6, lente=85, perto=1)
        vista("padrao", (PAD_U - 1.5, PAD_V + 5.5), CHAO + 1.7, (PAD_U, PAD_V), CHAO + 1.6, lente=35, perto=0.2)
        vista("canto", (-9, -9), 222, (-5.2, -5.0), 215.5, lente=40, perto=1)
        # fotografia 3 de longe, com teleobjetiva (a perspetiva quase não fecha a torre); corta as casas da rua
        vista("f3_longe", (-0.3, -85), CHAO + 1.6, (-0.3, 0), 204.5, lente=70, res=(700, 1050), sem_casas=True)
        # planta, com o norte para cima
        co.location = Vector((0, 0, 300))
        co.rotation_euler = (0, 0, 0)
        cam.type = "ORTHO"
        cam.ortho_scale = 34
        sc.render.resolution_x, sc.render.resolution_y = 1000, 1000
        sc.render.filepath = str(pasta / "topo.jpg")
        if not SO or "topo" in SO:
            bpy.ops.render.render(write_still=True)
