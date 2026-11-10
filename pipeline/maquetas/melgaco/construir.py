"""Constrói a maqueta 3D do Castelo de Melgaço e da muralha da vila (estado atual) em Blender e exporta-a para
public/maquetas/melgaco/melgaco.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow
    python pipeline/maquetas/melgaco/construir.py            # GLB
    python pipeline/maquetas/melgaco/construir.py --vistas   # GLB + capa + vistas de verificação

Dados (em dados/): lidar.npz e superficie.npz (lidar.py: LiDAR 2025 da DGT, folha 189572), muralhas.json
(tracar.py: estações medidas na cerca e na muralha da vila), orto.jpg (ortofoto.py: ortofotomapa 2021 da DGT),
fontes.json (recolher.py: plantas, ruas e elementos do OSM via Overture).

O que se modela, e de onde vem:
- a torre de menagem, rodada 26,7° (as faces olham a ENE, NNO, OSO e SSE), com a planta e as cotas do LiDAR e o
  coroamento das fotografias: parapeito saliente sobre uma cachorrada corrida com matacães, seis merlões de
  remate piramidal por face, telhado de quatro águas por dentro, porta alta em arco na face NNO com a escada
  metálica encostada, e as seteiras;
- a cerca oval do castelo, estação a estação (a cada grau): faces de dentro e de fora, adarve com os degraus,
  parapeito a norte, as escadas em V encostadas à face de dentro com a porta da traição ao pé de cada V, a
  torre pentagonal a sudeste, a plataforma do cunhal noroeste e a cisterna;
- a entrada a sul, aberta na cerca, e a barbacã em U com as duas arcadas no muro sul;
- a muralha da vila, do cunhal noroeste do castelo para poente e depois para sul, até à Porta de Baixo (arco de
  volta perfeita com duas arquivoltas, impostas e as mísulas dos matacães), e a Porta da Rua do Carvalho;
- as casas à volta em volumes simples de um só cinzento, as ruas, o relvado e as lajes de granito do recinto,
  e as árvores do LiDAR (nenhuma sobre o monumento).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pipeline/, para o kit comum
from kit.geometria import normais, amostra, quad, arco_pts, _sd, _anc  # noqa: E402
from kit.texturas import ruido, fbm, textura_cantaria, textura_cubos, textura_relva, textura_rocha, textura_telha, imagem  # noqa: E402
from kit.cena import uv_mundo  # noqa: E402
from kit.materiais import MATS, malha_rapida, material  # noqa: E402
from kit.arquitetura import aduelas, contorno_vao, parede, passagem, vao  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "melgaco.json").read_text())
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
MUR = json.loads((AQUI / "dados" / "muralhas.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "melgaco"
LID = np.load(AQUI / "dados" / "lidar.npz")
LP, LX0, LY0 = float(LID["passo"]), float(LID["x0"]), float(LID["y0"])
MDT = LID["mdt"] / 100.0 + float(LID["base"])
MDS = LID["mds"] / 100.0 + float(LID["base"])
CLS = LID["classe"]
SUPF = np.load(AQUI / "dados" / "superficie.npz")
SP, SX0, SY0 = float(SUPF["passo"]), float(SUPF["x0"]), float(SUPF["y0"])
SUP = SUPF["z"] / 100.0 + float(SUPF["base"])
ORTO = np.asarray(Image.open(AQUI / "dados" / "orto.jpg").convert("RGB"), float) / 255.0  # 25 cm, norte em cima
OX0, OY1, OP = -140.0, 60.0, 0.25

X0, X1, Y0, Y1 = -112.0, 46.0, -40.0, 46.0  # a maqueta: 158 × 86 m, do castelo à Porta de Baixo
rng = np.random.default_rng(1170)


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def cota(x, y):
    """Chão (MDT de 50 cm)."""
    return amostra(MDT, LX0, LY0, LP, x, y)


def superficie(x, y):
    """O ponto mais alto das classes solo, edifícios e muros (sem vegetação), em células de 25 cm."""
    return amostra(SUP, SX0, SY0, SP, x, y, ordem=0)


# ---------------------------------------------------------------- referencial da torre (medido no LiDAR)
# O retângulo mínimo dos pontos do topo dá as faces a 63,3° e 153,3° de azimute: u = ao longo do azimute 63,3°
# (ENE), v = azimute 333,3° (NNO). O centro da torre é a origem da cena.
ROT = math.radians(26.7)
EU = np.array([math.cos(ROT), math.sin(ROT)])
EV = np.array([-math.sin(ROT), math.cos(ROT)])
A = 4.40                 # meia largura do corpo da torre (8,8 m entre faces; o parapeito tem 9,6 m)
SAL = 0.40               # saliência do parapeito sobre a cachorrada (LiDAR: topo de 9,6 × 9,5 m)
CHAO_T = {"NNO": 191.1, "OSO": 190.3, "ENE": 190.5, "SSE": 190.2}  # chão junto a cada face (LiDAR)
Z_SOCO = 191.40          # topo do soco saliente (fotografia de baixo da face SSE)
Z_MISULA = 205.85        # pé da cachorrada (três fiadas de mísulas; fotografia)
Z_PAR = 207.00           # fundo do parapeito saliente, por cima das mísulas
Z_AMEIA = 208.55         # soleira das ameias (LiDAR)
Z_MERLAO = 209.35        # topo do corpo dos merlões (LiDAR: c. 209,5)
Z_PONTA = 210.25         # ponta do remate piramidal dos merlões (LiDAR: até 210,3)
Z_ADARVE_T = 207.55      # chão do adarve da torre, por dentro do parapeito
Z_BEIRAL = 207.70        # beiral do telhado de quatro águas (LiDAR)
Z_CUME = 208.90          # cume do telhado (LiDAR)
ESP_PAR = 0.80           # espessura do parapeito (o telhado começa a 4,0 m do centro; LiDAR)


def W(u, v):
    """Ponto do referencial da torre (u, v) para a cena (x este, y norte)."""
    return EU * u + EV * v


# ---------------------------------------------------------------- ruído


# ---------------------------------------------------------------- malhas


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


# ---------------------------------------------------------------- materiais e texturas


# ---------------------------------------------------------------- cena

for _ob in list(bpy.data.objects):  # a cena de arranque do bpy traz um cubo, uma luz e uma câmara
    bpy.data.objects.remove(_ob)
RAIZ_OB = bpy.data.objects.new("Castelo_Melgaco", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = CFG["origem"]["lat"], CFG["origem"]["lon"]
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
# "Muralhas_grupo" é o que o visor enquadra ao abrir (com os elementos com placa); o terreno chama-se "Terreno"
for g in ["Terreno", "Torre", "Muralhas", "Muralha_vila", "Envolvente", "Vegetacao"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

# torre: cantaria regular, silhares bem esquadriados em fiadas de c. 0,4 m, granito cinzento-dourado com líquenes
# (fotografias: a torre é de um tom muito igual, cinzento-dourado; as juntas notam-se como linhas finas e os
# líquenes brancos salpicam a pedra)
PAL_T = np.array([(0.640, 0.605, 0.530), (0.665, 0.630, 0.555), (0.590, 0.560, 0.500), (0.680, 0.610, 0.500),
                  (0.610, 0.595, 0.560)])
TEX, ALT = textura_cantaria(fiada=(0.34, 0.50), comp=(0.55, 1.4), liquen=0.76, paleta=PAL_T, prob=[0.50, 0.17, 0.13, 0.12, 0.08],
                            var=(0.94, 1.06), liq_forca=0.40, tinta=(1.0, 1.0, 0.98))
material("cantaria_torre", (1, 1, 1), 0.9, imagem("cantaria_torre", TEX), normal=imagem("cantaria_torre_n", normais(ALT, 3.0)),
         forca=0.8, escala=0.5, vcor=True)
# cerca e muralha: alvenaria de granito em fiadas irregulares, pedras mais pequenas e menos esquadriadas,
# mais cinzenta e com mais líquenes (fotografias do recinto e da barbacã)
PAL_M = np.array([(0.585, 0.570, 0.535), (0.650, 0.635, 0.600), (0.480, 0.465, 0.440), (0.610, 0.565, 0.490),
                  (0.540, 0.530, 0.510)])
TEX, ALT = textura_cantaria(semente=1263, fiada=(0.20, 0.42), comp=(0.28, 0.95), liquen=0.70, paleta=PAL_M,
                            prob=[0.38, 0.18, 0.18, 0.12, 0.14], junta_m=0.012, irregular=0.05, tinta=(1.03, 1.03, 1.0),
                            var=(0.92, 1.07), liq_forca=0.40)
material("muro", (1, 1, 1), 0.92, imagem("muro", TEX), normal=imagem("muro_n", normais(ALT, 3.5)), forca=1.0, escala=0.5, vcor=True)
TEX, ALT = textura_cantaria(semente=1264, fiada=(0.20, 0.45), comp=(0.25, 0.80), liquen=0.60, paleta=PAL_M,
                            prob=[0.38, 0.18, 0.18, 0.12, 0.14], junta_m=0.015, irregular=0.08, tinta=(1.0, 1.02, 0.95),
                            var=(0.92, 1.07), liq_forca=0.40)
material("muro_topo", (1, 1, 1), 0.95, imagem("muro_topo", TEX), normal=imagem("muro_topo_n", normais(ALT, 3.5)), forca=1.0, escala=0.5, vcor=True)
# Porta de Baixo: cantaria regular de fiadas de c. 0,4 m (fotografias), granito cinzento com líquenes e musgo
TEX, ALT = textura_cantaria(semente=1387, fiada=(0.32, 0.48), comp=(0.45, 1.2), liquen=0.68, paleta=PAL_M,
                            prob=[0.38, 0.18, 0.18, 0.12, 0.14], junta_m=0.008, irregular=0.02, tinta=(1.0, 1.02, 0.98),
                            var=(0.92, 1.07), liq_forca=0.40)
material("cantaria_porta", (1, 1, 1), 0.9, imagem("cantaria_porta", TEX), normal=imagem("cantaria_porta_n", normais(ALT, 3.0)),
         forca=0.8, escala=0.5, vcor=True)
material("cantaria_lisa", (0.63, 0.61, 0.56), 0.85)
material("intradorso", (0.34, 0.33, 0.31), 0.95)
material("sombra", (0.06, 0.055, 0.05), 1.0)
material("madeira", (0.22, 0.15, 0.10), 0.8)
material("ferro", (0.16, 0.16, 0.17), 0.5)
TEX, ALT = textura_telha()
material("telha", (1, 1, 1), 0.8, imagem("telha", TEX * np.array([0.78, 0.45, 0.30])), normal=imagem("telha_n", normais(ALT, 2.0)), forca=0.6)
TEX, ALT = textura_cubos()
material("calcada", (1, 1, 1), 0.92, imagem("calcada", TEX), normal=imagem("calcada_n", normais(ALT, 1.5)), forca=0.5)
TEX, ALT = textura_relva()
material("relva", (1, 1, 1), 0.95, imagem("relva", TEX), normal=imagem("relva_n", normais(ALT, 1.0)), forca=0.4)
TEX, ALT = textura_rocha()
material("rocha", (1, 1, 1), 0.85, imagem("rocha", TEX), normal=imagem("rocha_n", normais(ALT, 2.0)), forca=0.6)
material("terreiro", (0.62, 0.60, 0.55), 0.95)   # pátios, quintais e lajeados à volta (neutro)
material("verde_viz", (0.50, 0.56, 0.42), 0.95)  # hortas e jardins à volta (verde apagado)
material("asfalto", (0.33, 0.33, 0.34), 0.95)
# casas vizinhas: um só cinzento claro, sem cor nem textura, para o destaque ficar todo no castelo
material("cinza_viz", (0.74, 0.74, 0.73), 0.9)
material("base", (0.33, 0.26, 0.20), 0.95)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)

MEDIDAS = {}


def patine_torre(co):
    """A cantaria da torre escurece para o alto (líquenes e musgo) e junto ao chão (salpicos)."""
    z = co[:, 2]
    t = np.clip((z - 198.0) / 12.0, 0, 1) ** 1.3
    base = np.stack([1.0 - 0.10 * t, 1.0 - 0.07 * t, 1.0 - 0.09 * t], 1)
    pe = np.clip(1 - (z - 190.3) / 1.6, 0, 1)[:, None] * 0.15
    return np.clip(base * (1 - pe), 0, 1)


def patine_muro(co):
    """Os muros têm manchas grandes mais claras e mais escuras (pedra reposta e pedra velha, musgo) e são mais
    escuros ao pé."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x + 500, y + 500 + z * 0.7, 7.0, 31)
    v = 0.88 + 0.24 * g
    pe = np.clip(1 - (z - cota(x, y)) / 1.4, 0, 1) * 0.12
    c = np.stack([v, v * (1.0 + 0.02 * (g - 0.5)), v * 0.99], 1)
    return np.clip(c * (1 - pe)[:, None], 0, 1)


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


FACES = {"ENE": face_torre((1, 0), A), "NNO": face_torre((0, 1), A), "OSO": face_torre((-1, 0), A), "SSE": face_torre((0, -1), A)}


# ---------------------------------------------------------------- torre de menagem

MAT_T = ["cantaria_torre", "cantaria_lisa", "sombra", "madeira", "ferro", "telha"]

# vãos por face (s ao longo da face: na NNO s = u, na SSE s = -u, na OSO s = v, na ENE s = -v)
# NNO: a porta alta em arco de volta perfeita, ao cimo da escada metálica (patamar do LiDAR a 195,6 m, de u -1,3
# a 0,2; fotografia do recinto a olhar para SSE); por cima, uma fresta (estimativa).
PORTA_S, PORTA_W, PORTA_Z, PORTA_ALT = -0.55, 1.00, 195.75, 2.20
VAOS = {
    "NNO": [(PORTA_S - PORTA_W / 2, PORTA_S + PORTA_W / 2, PORTA_Z, PORTA_Z + PORTA_ALT, True),
            (PORTA_S - 0.05, PORTA_S + 0.05, 201.6, 202.6, False)],
    # SSE: duas seteiras, uma alta a meio e outra a meia altura, um pouco a poente (fotografia de baixo)
    "SSE": [(-0.05, 0.05, 203.0, 204.0, False), (0.85, 0.95, 199.8, 200.8, False)],
    # OSO e ENE: uma seteira alta em cada (fotografias de longe; estimativa)
    "OSO": [(-0.05, 0.05, 204.0, 205.0, False)],
    "ENE": [(-0.05, 0.05, 201.0, 202.0, False)],
}
# merlões: seis por face, contando os dos cunhais (0,9 m), os do meio com 0,75 m e ameias de 0,96 m (LiDAR)
MERLOES = [(-2.94, -2.19), (-1.23, -0.48), (0.48, 1.23), (2.19, 2.94)]
MISULAS = np.linspace(-3.70, 3.70, 8)  # oito mísulas por face, mais a do cunhal (fotografia de baixo da SSE)


def torre():
    m = Malha(MAT_T)
    mc = Malha(MAT_T)  # coroamento (mísulas, matacães, parapeito, merlões): elemento próprio no visor
    mp = Malha(MAT_T)  # porta
    zb = 189.4
    R = A + SAL
    # soco saliente junto ao chão (fotografia de baixo: c. 1 m acima do chão a SSE)
    m.caixa_uv(-A - 0.12, A + 0.12, -A - 0.12, A + 0.12, zb, Z_SOCO, "cantaria_torre")
    m.caixa_uv(-A - 0.12, A + 0.12, -A - 0.12, A + 0.12, Z_SOCO - 0.06, Z_SOCO, "cantaria_lisa", topo=False)
    for nome, f in FACES.items():
        parede(m, f, -A, A, Z_SOCO, Z_PAR, VAOS[nome], "cantaria_torre")
        for sa, sb, za, zb_, arco in VAOS[nome]:
            w = sb - sa
            sc = (sa + sb) / 2
            if arco:
                vao(mp, f, sc, w, za, zb_, 0.55, True, "cantaria_torre", "madeira")
                aduelas(mp, f, sc, w, za, zb_, 0.34, 0.03, n=11)
                mp.caixa_face(f, sa - 0.2, sb + 0.2, -0.02, 0.10, za - 0.12, za, "cantaria_lisa")
            else:
                vao(m, f, sc, w, za, zb_, 0.7, False, "cantaria_torre", "sombra")  # seteira: só a fresta escura
        # parapeito saliente: face de fora (de cunhal a cunhal), fundo com os matacães, face de dentro e soleira
        quad(mc, f, -R, R, Z_PAR, Z_AMEIA, d=SAL, mat="cantaria_torre")
        quad(mc, f, R - ESP_PAR, -(R - ESP_PAR), Z_ADARVE_T, Z_AMEIA, d=SAL - ESP_PAR, mat="cantaria_torre")
        mc.face([f.p(-R, SAL, Z_AMEIA), f.p(R, SAL, Z_AMEIA), f.p(R - ESP_PAR, SAL - ESP_PAR, Z_AMEIA),
                 f.p(-(R - ESP_PAR), SAL - ESP_PAR, Z_AMEIA)], "cantaria_torre")
        # fundo do parapeito entre a face e a saliência: matacães (aberturas escuras) entre as mísulas
        ms = list(MISULAS)
        bordas = [-A] + [x for s_ in ms for x in (s_ - 0.18, s_ + 0.18)] + [A]
        for i in range(0, len(bordas) - 1):
            a_, b_ = bordas[i], bordas[i + 1]
            mt = "sombra" if i % 2 == 0 else "cantaria_torre"
            mc.face([f.p(a_, 0.0, Z_PAR), f.p(b_, 0.0, Z_PAR), f.p(b_, SAL, Z_PAR), f.p(a_, SAL, Z_PAR)], mt)
        # um friso de pedra (a laje do parapeito) à frente dos matacães
        mc.caixa_face(f, -R, R, SAL - 0.02, SAL + 0.03, Z_PAR - 0.02, Z_PAR + 0.16, "cantaria_lisa")
        # mísulas: três fiadas, cada uma mais saliente, de frente arredondada
        for s_ in ms:
            misula(mc, f, s_)
        # merlões do meio, com remate piramidal
        for a_, b_ in MERLOES:
            merlao(mc, f, a_, b_, SAL, SAL - ESP_PAR)
    # cunhais: mísula de canto (em diagonal), fundo do parapeito e merlão do cunhal (0,9 × 0,9 m)
    for su in (-1, 1):
        for sv in (-1, 1):
            for k in range(3):
                z0, z1 = Z_MISULA + 0.383 * k + 0.01, Z_MISULA + 0.383 * (k + 1) - 0.01
                s = 0.12 + (SAL - 0.12) * k / 2
                mc.caixa_uv(*sorted((su * (A - 0.25), su * (A + s))), *sorted((sv * (A - 0.25), sv * (A + s))), z0, z1, "cantaria_torre", fundo=True)
            mc.caixa_uv(*sorted((su * A, su * R)), *sorted((sv * A, sv * R)), Z_PAR - 0.01, Z_PAR + 0.01, "cantaria_torre", fundo=True)
            q = [W(su * R, sv * R), W(su * (R - 0.9), sv * R), W(su * (R - 0.9), sv * (R - 0.9)), W(su * R, sv * (R - 0.9))]
            mc.prisma(q, Z_AMEIA, Z_MERLAO, "cantaria_torre", topo=False)
            mc.piramide(q, Z_MERLAO, Z_PONTA, "cantaria_torre")
    # adarve e telhado de quatro águas por dentro do parapeito (LiDAR: beiral 207,7, cume 208,9)
    ri = R - ESP_PAR
    m.caixa_uv(-ri, ri, -ri, ri, Z_ADARVE_T - 0.2, Z_ADARVE_T, "cantaria_lisa", topo=True)
    q = [W(-ri + 0.05, -ri + 0.05), W(ri - 0.05, -ri + 0.05), W(ri - 0.05, ri - 0.05), W(-ri + 0.05, ri - 0.05)]
    m.prisma(q, Z_ADARVE_T, Z_BEIRAL, "cantaria_torre", topo=False)
    m.piramide(q, Z_BEIRAL, Z_CUME, "telha")
    m.objeto("torre", "torre", GRUPOS["Torre"], uv=2.0, vcor=patine_torre)
    mc.objeto("coroamento", "coroamento", GRUPOS["Torre"], uv=2.0, vcor=patine_torre)
    mp.objeto("porta_torre", "porta_torre", GRUPOS["Torre"], uv=2.0, vcor=patine_torre)


def misula(m, f, s, larg=0.36):
    """Mísula de três fiadas por baixo do parapeito, cada uma mais saliente que a de baixo (0,13 / 0,27 / 0,40 m)."""
    alt = (Z_PAR - Z_MISULA) / 3
    for k in range(3):
        z0, z1 = Z_MISULA + alt * k + 0.01, Z_MISULA + alt * (k + 1) - 0.01
        sal = 0.13 + (SAL - 0.13) * k / 2
        pts = [f.xy(s - larg / 2, -0.02)]
        for a in np.linspace(math.pi, 0, 7):
            pts.append(f.xy(s + larg / 2 * math.cos(a), sal - 0.10 + 0.10 * math.sin(a)))
        pts.append(f.xy(s + larg / 2, -0.02))
        m.prisma(pts, z0, z1, "cantaria_torre", fundo=True)


def merlao(m, f, sa, sb, d_fora, d_dentro):
    q = [f.xy(sa, d_fora), f.xy(sb, d_fora), f.xy(sb, d_dentro), f.xy(sa, d_dentro)]
    m.prisma(q, Z_AMEIA, Z_MERLAO, "cantaria_torre", topo=False)
    m.piramide(q, Z_MERLAO, Z_PONTA, "cantaria_torre")


def escada_metalica():
    """Escada metálica moderna encostada à face NNO (LiDAR: sobe de u 5,3, a 191,5 m, até ao patamar de u 0,2 a
    -1,35, a 195,65 m; fotografia do recinto): degraus de chapa entre duas vigas, guarda do lado de fora e no
    patamar, prumos por baixo do patamar."""
    m = Malha(["ferro"])
    f = FACES["NNO"]
    s_pe, s_top, s_fim = 5.35, 0.25, -1.40
    z_pe, z_top = 191.25, 195.65
    d0, d1 = 0.06, 0.92
    n = 22
    run = (s_pe - s_top) / n
    rise = (z_top - z_pe) / n
    for i in range(n):
        s_a = s_pe - run * (i + 1)
        z = z_pe + rise * (i + 1)
        m.caixa_face(f, s_a, s_a + run + 0.02, d0 + 0.03, d1 - 0.03, z - 0.035, z, 0)
    # vigas laterais inclinadas
    for d_ in (d0, d1):
        for dd in (0.0, 0.012):
            m.face([f.p(s_pe + 0.1, d_ + dd, z_pe - 0.3), f.p(s_top, d_ + dd, z_top - 0.3), f.p(s_top, d_ + dd, z_top + 0.02),
                    f.p(s_pe + 0.1, d_ + dd, z_pe + 0.02)], 0)
    # patamar
    m.caixa_face(f, s_fim, s_top, 0.0, d1, z_top - 0.06, z_top, 0)
    # prumos do patamar e a meio da escada
    for s_ in (s_fim + 0.08, s_top - 0.08):
        m.barra(f.p(s_, d1 - 0.05, CHAO_T["NNO"] - 0.1), f.p(s_, d1 - 0.05, z_top - 0.06), 0.04, 0)
    s_m = (s_pe + s_top) / 2
    m.barra(f.p(s_m, d1 - 0.05, CHAO_T["NNO"] - 0.1), f.p(s_m, d1 - 0.05, (z_pe + z_top) / 2 - 0.3), 0.04, 0)
    # guarda de fora: prumos, corrimão e travessa a meia altura, ao longo da escada e do patamar
    pts = [(s_pe, z_pe), (s_top, z_top), (s_fim, z_top)]
    for (sa, za), (sb, zb) in zip(pts[:-1], pts[1:]):
        L = abs(sb - sa)
        k = max(1, int(round(L / 1.1)))
        for j in range(k + 1):
            s_ = sa + (sb - sa) * j / k
            z_ = za + (zb - za) * j / k
            m.barra(f.p(s_, d1, z_), f.p(s_, d1, z_ + 1.0), 0.02, 0, k=5)
        for h in (0.5, 1.0):
            m.barra(f.p(sa, d1, za + h), f.p(sb, d1, zb + h), 0.022, 0, k=5)
    # guarda na ponta do patamar (contra a face) e corrimão encostado à parede
    m.barra(f.p(s_fim, d1, z_top + 1.0), f.p(s_fim, 0.05, z_top + 1.0), 0.022, 0, k=5)
    m.barra(f.p(s_fim, 0.05, z_top), f.p(s_fim, 0.05, z_top + 1.0), 0.02, 0, k=5)
    m.barra(f.p(s_pe, 0.08, z_pe + 0.95), f.p(s_top, 0.08, z_top + 0.95), 0.02, 0, k=5)
    m.objeto("escada_metalica", "porta_torre", GRUPOS["Torre"], uv=None)


# ---------------------------------------------------------------- cerca do castelo

CENTRO = np.array([-0.56, -1.09])  # centro aproximado do recinto (tracar.py)


def _circ_interp(az, vals):
    """Preenche os None por interpolação circular e reamostra a cada grau."""
    a = np.asarray(az, float)
    v = np.array([np.nan if x is None else x for x in vals], float)
    ok = ~np.isnan(v)
    aa = np.concatenate([a[ok] - 360, a[ok], a[ok] + 360])
    vv = np.concatenate([v[ok], v[ok], v[ok]])
    return np.interp(np.arange(0, 360, 1.0), aa, vv)


def _circ_med(v, k):
    return ndi.median_filter(v, size=k, mode="wrap")


def _circ_suave(v, s):
    return ndi.gaussian_filter1d(v, s, mode="wrap")


def estacoes_cerca():
    est = MUR["cerca"]
    az = [i * 2.0 for i in range(len(est))]
    get = lambda k: [None if e is None else e[k] for e in est]  # noqa: E731
    rin = _circ_interp(az, get("rin"))
    rout = _circ_interp(az, get("rout"))
    # nas escadas em V o traçado mede os degraus em vez do adarve: tiram-se essas estações (LiDAR: degraus de
    # 191 a 195,5 m numa faixa de c. 1 m encostada à face de dentro)
    escada = [(38.5, 47.5), (55.5, 70.5), (336.5, 343.5), (349.5, 357.5)]
    adv = [None if (v is None or any(a0 < a_ < a1 for a0, a1 in escada)) else v for a_, v in zip(az, get("adarve"))]
    adarve = _circ_interp(az, adv)
    parap = _circ_interp(az, get("parapeito"))
    # espessura: limitada a 2,1–3,1 m (onde a torre SE, a plataforma NO e a barbacã se colam ao muro, o traçado
    # dá espessuras falsas)
    esp = _circ_med(np.clip(rout - rin, 2.1, 3.1), 9)
    rin = _circ_suave(_circ_med(rin, 5), 1.2)
    esp = _circ_suave(esp, 1.5)
    # adarve: a mediana tira as escadas em V encostadas à face de dentro (que contaminam az 42, 60, 66, 342, 354)
    adarve = _circ_med(adarve, 5)
    ap = np.clip(_circ_med(parap - adarve, 9), 0, None)
    graus = np.arange(360)
    ap[(graus > 30) & (graus < 330)] = 0.0  # parapeito só a norte (az 354–0 no LiDAR; prolonga-se um pouco)
    ap = np.where(ap > 0.25, np.maximum(ap, 0.55), 0.0)
    ap = _circ_suave(ap, 2.0)
    ap[ap < 0.08] = 0.0
    return graus.astype(float), rin, rin + esp, adarve, ap


AZ_C, RIN_C, ROUT_C, ADARVE_C, PARAP_C = estacoes_cerca()


def pt_cerca(az, r):
    a = math.radians(az)
    return CENTRO + r * np.array([math.sin(a), math.cos(a)])


def rin_az(az):
    return float(np.interp(az % 360, np.append(AZ_C, 360), np.append(RIN_C, RIN_C[0])))


def rout_az(az):
    return float(np.interp(az % 360, np.append(AZ_C, 360), np.append(ROUT_C, ROUT_C[0])))


def adarve_az(az):
    return float(np.interp(az % 360, np.append(AZ_C, 360), np.append(ADARVE_C, ADARVE_C[0])))


# cortes na cerca: a entrada a sul (passagem aberta, LiDAR) e as duas portas da traição ao pé das escadas em V
ENTRADA = (184.1, 186.9)
POSTIGOS = {"ne": (53.5, 50.8, 56.2), "n": (346.5, 344.3, 348.7)}


def cerca():
    m = Malha(["muro", "muro_topo", "intradorso", "calcada", "sombra"])
    cortes = [ENTRADA] + [(a, b) for _, a, b in POSTIGOS.values()]

    def cortado(a):
        return any(c0 < a < c1 for c0, c1 in cortes)

    passo = 1.0
    segs = []
    a = 0.0
    while a < 360 - 1e-6:
        b = a + passo
        # quebra nos limites dos cortes
        for c0, c1 in cortes:
            for c in (c0, c1):
                if a < c < b:
                    b = c
        segs.append((a, b))
        a = b
    for a, b in segs:
        am = (a + b) / 2
        if cortado(am):
            continue
        z_top = adarve_az(am)       # topo plano em cada segmento: as escadas do adarve ficam em degraus
        ri0, ri1 = rin_az(a), rin_az(b)
        ro0, ro1 = rout_az(a), rout_az(b)
        pi0, pi1 = pt_cerca(a, ri0), pt_cerca(b, ri1)
        po0, po1 = pt_cerca(a, ro0), pt_cerca(b, ro1)
        zi = min(cota(*pt_cerca(am, (ri0 + ri1) / 2 - 1.0)), cota(*pt_cerca(am, (ri0 + ri1) / 2 - 0.3))) - 0.4
        zo = min(cota(*pt_cerca(am, (ro0 + ro1) / 2 + 1.0)), cota(*pt_cerca(am, (ro0 + ro1) / 2 + 0.3))) - 0.4
        ap = float(np.interp(am, np.append(AZ_C, 360), np.append(PARAP_C, PARAP_C[0])))
        # face de dentro e de fora
        m.face([(*pi1, zi), (*pi0, zi), (*pi0, z_top), (*pi1, z_top)], "muro")
        z_po = z_top + ap
        m.face([(*po0, zo), (*po1, zo), (*po1, z_po), (*po0, z_po)], "muro")
        if ap > 0:
            # parapeito de 0,6 m de espessura do lado de fora
            pp0, pp1 = pt_cerca(a, ro0 - 0.6), pt_cerca(b, ro1 - 0.6)
            m.face([(*pi0, z_top), (*pi1, z_top), (*pp1, z_top), (*pp0, z_top)], "muro_topo")
            m.face([(*pp0, z_top), (*pp1, z_top), (*pp1, z_po), (*pp0, z_po)], "muro")
            m.face([(*pp0, z_po), (*pp1, z_po), (*po1, z_po), (*po0, z_po)], "muro_topo")
        else:
            m.face([(*pi0, z_top), (*pi1, z_top), (*po1, z_top), (*po0, z_top)], "muro_topo")
        # espelhos dos degraus (e as faces dos cortes): fecham o segmento nas duas pontas
        for az_, pi_, po_, ri_ in ((a, pi0, po0, ri0), (b, pi1, po1, ri1)):
            viz = am - passo if az_ == a else am + passo
            z_viz = -1e9 if cortado(viz % 360) else adarve_az(viz)
            if z_viz < z_top + ap - 0.02:
                zb_ = max(z_viz, min(zi, zo)) if z_viz > -1e8 else min(zi, zo)
                m.face([(*pi_, zb_), (*po_, zb_), (*po_, z_top + (ap if ap > 0 else 0)), (*pi_, z_top)], "muro")
    m.objeto("cerca", "cerca", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)


def postigos():
    """Portas da traição ao pé das escadas em V: arco de volta perfeita de c. 1,2 × 2,3 m, de lado a lado do muro
    (fotografias do recinto e de cima; a de norte é provável, não certa)."""
    m = Malha(["muro", "muro_topo", "intradorso", "calcada", "cantaria_lisa"])
    for nome, (az, a0, a1) in POSTIGOS.items():
        # plano da face de fora, centrado no azimute do postigo
        po = pt_cerca(az, rout_az(az))
        pi = pt_cerca(az, rin_az(az))
        nrm = unit(po - pi)
        t = np.array([nrm[1], -nrm[0]])
        f = Plano(po, t, nrm)
        prof = float(np.linalg.norm(po - pi))
        # largura do bloco: a corda do corte, à face de fora
        wb = float(np.linalg.norm(pt_cerca(a1, rout_az(a1)) - pt_cerca(a0, rout_az(a0)))) / 2
        wbi = float(np.linalg.norm(pt_cerca(a1, rin_az(a1)) - pt_cerca(a0, rin_az(a0)))) / 2
        z_top = adarve_az(az)
        chao_d = cota(*pt_cerca(az, rin_az(az) - 0.8))
        chao_f = cota(*pt_cerca(az, rout_az(az) + 0.8))
        zb = min(chao_d, chao_f) - 0.4
        ap = float(np.interp(az, np.append(AZ_C, 360), np.append(PARAP_C, PARAP_C[0])))
        passagem(m, f, -max(wb, wbi), max(wb, wbi), prof, zb, z_top, 1.20, 2.30, chao_f, chao_d)
        if ap > 0:
            m.caixa_face(f, -wb, wb, -0.6, 0.0, z_top, z_top + ap, "muro")
        # aduelas da face de dentro (fotografias: arco de pedras bem aparelhadas)
        fi = Plano(pi, -t, -nrm)
        aduelas(m, fi, 0.0, 1.20, chao_d, chao_d + 2.30, 0.30, 0.03, n=9, mat="cantaria_lisa")
        MEDIDAS[f"postigo_{nome}"] = {"az": az, "chao_dentro": round(chao_d, 2), "chao_fora": round(chao_f, 2)}
    m.objeto("postigos", "escadas", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)


# escadas em V encostadas à face de dentro: pé no meio, dois lanços a subir para os lados até ao adarve (LiDAR)
ESCADAS_V = [
    ("ne", 53.5, 39.0, 71.0),
    ("n", 346.5, 337.0, 358.0),
]


def escadas():
    m = Malha(["muro", "muro_topo"])
    larg = 1.0
    for nome, pe, a_esq, a_dir in ESCADAS_V:
        for topo_az in (a_esq, a_dir):
            sg = np.sign(topo_az - pe)
            a_ini = pe + sg * 2.6  # o primeiro degrau começa ao lado do postigo
            z0 = cota(*pt_cerca(pe, rin_az(pe) - 1.5))
            z1 = adarve_az(topo_az)
            n = max(6, int(round((z1 - z0) / 0.28)))
            for i in range(n):
                aa = a_ini + (topo_az - a_ini) * i / n
                ab = a_ini + (topo_az - a_ini) * (i + 1) / n
                if sg < 0:
                    aa, ab = ab, aa
                z = z0 + (z1 - z0) * (i + 1) / n
                q = [pt_cerca(aa, rin_az(aa) + 0.05), pt_cerca(ab, rin_az(ab) + 0.05),
                     pt_cerca(ab, rin_az(ab) - larg), pt_cerca(aa, rin_az(aa) - larg)]
                zb = min(cota(*q[2]), cota(*q[3])) - 0.3
                m.prisma(q, zb, z, "muro", mat_topo="muro_topo")
            MEDIDAS[f"escada_{nome}_{int(topo_az)}"] = {"pe": round(z0, 2), "topo": round(z1, 2), "degraus": n}
    m.objeto("escadas", "escadas", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)


# torre pentagonal a sudeste (LiDAR: topo a 196,3 m, parapeitos a 197,3 m nos lados de fora) e plataforma do
# cunhal noroeste (topo a 195,5 m, parapeito a 196,6 m a poente)
TORRE_SE = [(25.0, -17.7), (29.5, -18.5), (31.7, -15.8), (31.6, -10.6), (23.8, -10.6), (21.8, -16.2)]  # anti-horário
TORRE_SE_PARAP = [((25.0, -17.7), (29.5, -18.5)), ((29.5, -18.5), (31.7, -15.8)), ((31.7, -15.8), (31.6, -10.6)),
                  ((31.6, -10.6), (27.8, -10.6))]  # parapeito a sul, sueste, nascente e na ponta nascente do lado norte
PLAT_NO = [(-29.4, 16.6), (-20.5, 16.6), (-20.5, 22.4), (-29.4, 22.4)]
JUNTA_NO = [(-31.6, 22.4), (-26.0, 22.0), (-24.0, 25.6), (-29.6, 27.4), (-31.6, 26.8)]


def _parapeito_lado(m, a, b, z0, z1, esp, mat="muro", mat_topo="muro_topo"):
    """Parapeito ao longo do lado a→b (por dentro do polígono, à esquerda de a→b se for anti-horário)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = unit(b - a)
    n_in = np.array([-t[1], t[0]])
    q = [a, b, b + n_in * esp, a + n_in * esp]
    m.prisma(q, z0, z1, mat, mat_topo=mat_topo)


def torres_cerca():
    m = Malha(["muro", "muro_topo"])
    # torre SE
    pol = shapely.Polygon(TORRE_SE)
    base = np.array(pol.exterior.coords)[:-1]
    if not pol.exterior.is_ccw:
        base = base[::-1]
    zb = float(min(cota(*p) for p in base)) - 0.5
    m.prisma(base, zb, 196.30, "muro", mat_topo="muro_topo")
    # parapeito contínuo, com os cantos em meia-esquadria (sem cunhas abertas entre os lados)
    cad = [np.asarray(TORRE_SE_PARAP[0][0], float)] + [np.asarray(b_, float) for _, b_ in TORRE_SE_PARAP]
    esp = 0.9
    dentro = []
    for i, v in enumerate(cad):
        ns = []
        if i > 0:
            t = unit(v - cad[i - 1]); ns.append(np.array([-t[1], t[0]]))
        if i < len(cad) - 1:
            t = unit(cad[i + 1] - v); ns.append(np.array([-t[1], t[0]]))
        nm = unit(sum(ns))
        dentro.append(v + nm * esp / max(0.3, float(np.dot(nm, ns[0]))))
    m.prisma(cad + dentro[::-1], 196.30, 197.30, "muro", mat_topo="muro_topo")
    MEDIDAS["torre_se"] = {"topo": 196.3, "parapeito": 197.3, "chao_fora": round(float(cota(31.0, -14.0)), 2)}
    # plataforma NO e junção com a muralha da vila
    for pl, z in ((PLAT_NO, 195.50), (JUNTA_NO, 195.80)):
        p = shapely.Polygon(pl)
        b = np.array(p.exterior.coords)[:-1]
        if not p.exterior.is_ccw:
            b = b[::-1]
        zb = float(min(cota(*q) for q in b)) - 0.5
        m.prisma(b, zb, z, "muro", mat_topo="muro_topo")
    # parapeito a poente da plataforma (LiDAR: 196,5–196,6)
    _parapeito_lado(m, (-29.4, 22.4), (-29.4, 16.6), 195.50, 196.55, 2.6)
    m.objeto("torres_cerca", "torres_cerca", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)


def cisterna():
    """Cisterna retangular a norte da torre (OSM storage_tank; LiDAR: bordos a 191,8 m e cumeeira a 192,2 m ao
    comprido, c. 0,8–1,2 m acima do recinto): caixa de granito com cobertura de lajes em duas águas baixas."""
    g = next(i for i in F["infraestrutura"] if i["classe"] == "storage_tank")
    p = shapely.geometry.shape(g["geometria"])
    b = np.array(p.exterior.coords)[:-1]
    if not shapely.Polygon(b).exterior.is_ccw:
        b = b[::-1]
    m = Malha(["muro", "muro_topo", "cantaria_lisa"])
    zb = float(min(cota(*q) for q in b)) - 0.3
    m.prisma(b, zb, 191.80, "muro", topo=False)
    # duas águas ao longo do lado maior
    lados = [np.linalg.norm(b[(i + 1) % 4] - b[i]) for i in range(4)]
    i0 = int(np.argmax(lados))
    p0, p1, p2, p3 = b[i0], b[(i0 + 1) % 4], b[(i0 + 2) % 4], b[(i0 + 3) % 4]
    c0, c1 = (p0 + p3) / 2, (p1 + p2) / 2
    m.face([(*p0, 191.80), (*p1, 191.80), (*c1, 192.20), (*c0, 192.20)], "muro_topo")
    m.face([(*p2, 191.80), (*p3, 191.80), (*c0, 192.20), (*c1, 192.20)], "muro_topo")
    m.face([(*p1, 191.80), (*p2, 191.80), (*c1, 192.20)], "muro")
    m.face([(*p3, 191.80), (*p0, 191.80), (*c0, 192.20)], "muro")
    m.objeto("cisterna", "cisterna", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)


# barbacã a sul (LiDAR): muro poente (x -10,5 a -5,5) e muro sul (y -32,6 a -34,8), pátio a 187,7 m, saída a nascente
BARB_O = [(-10.6, -28.4), (-5.6, -28.4), (-5.5, -32.6), (-10.6, -32.6)]
BARB_S = [(-10.8, -32.6), (1.2, -32.6), (1.3, -34.8), (-10.8, -34.8)]  # a sul encosta-se uma casa (fotografia do alto da torre)
Z_BARB = 190.85
ARCADAS = [(-4.35, 1.30, 2.05), (-2.25, 1.30, 2.05)]  # centro (x), largura e altura das duas arcadas (foto de cima)


def barbaca():
    m = Malha(["muro", "muro_topo", "intradorso", "calcada", "sombra"])
    for pl in (BARB_O, BARB_S):
        p = shapely.Polygon(pl)
        b = np.array(p.exterior.coords)[:-1]
        if not p.exterior.is_ccw:
            b = b[::-1]
        zb = float(min(cota(*q) for q in b)) - 0.5
        if pl is BARB_S:
            # a face norte (do pátio) leva as duas arcadas: faz-se à parte
            x0, x1, y_n, y_s = -10.8, 1.25, -32.6, -34.8
            f = Plano((0.0, y_n), (-1.0, 0.0), (0.0, 1.0))  # s = -x, normal para norte (para o pátio)
            vaos = [(-(xc + w / 2), -(xc - w / 2), 187.70, 187.70 + h, True) for xc, w, h in ARCADAS]
            parede(m, f, -x1, -x0, zb, Z_BARB, vaos, "muro")
            for xc, w, h in ARCADAS:
                vao(m, f, -xc, w, 187.70, 187.70 + h, 1.1, True, "intradorso", "sombra")
                aduelas(m, f, -xc, w, 187.70, 187.70 + h, 0.32, 0.03, n=9, mat="muro", ombreiras=False)
            # resto do muro como um prisma (base anti-horária) sem a face norte, que é a das arcadas
            q = [(x0, y_s), (x1, y_s), (x1, y_n), (x0, y_n)]
            for k_ in (0, 1, 3):
                pa_, pb_ = q[k_], q[(k_ + 1) % 4]
                m.face([(*pa_, zb), (*pb_, zb), (*pb_, Z_BARB), (*pa_, Z_BARB)], "muro")
            m.face([(*p_, Z_BARB) for p_ in q], "muro_topo")
        else:
            m.prisma(b, zb, Z_BARB, "muro", mat_topo="muro_topo")
    MEDIDAS["barbaca"] = {"topo": Z_BARB, "patio": round(float(cota(-2.5, -31.0)), 2), "entrada": round(float(cota(-3.0, -26.5)), 2)}
    m.objeto("barbaca", "porta_castelo", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)


# ---------------------------------------------------------------- muralha da vila

def estacoes_vila():
    est = MUR["vila"]
    s = np.array([e["s"] for e in est])
    x = np.array([e["x"] for e in est])
    y = np.array([e["y"] for e in est])
    larg = np.array([e["largura"] for e in est])
    topo = np.array([e["topo"] for e in est])
    topo = ndi.gaussian_filter1d(ndi.median_filter(topo, 5, mode="nearest"), 1.0, mode="nearest")
    larg = ndi.gaussian_filter1d(ndi.median_filter(larg, 5, mode="nearest"), 1.5, mode="nearest")
    # direções suavizadas (o eixo desenhado à mão tem cantos a cada 4 m)
    xs = ndi.gaussian_filter1d(x, 1.5, mode="nearest")
    ys = ndi.gaussian_filter1d(y, 1.5, mode="nearest")
    xs[0], ys[0] = x[0], y[0]
    return s, xs, ys, larg, topo


S_V, X_V, Y_V, L_V, T_V = estacoes_vila()
# Porta de Baixo (LiDAR: passagem a 177,4 m entre x -99 e -97, a y -21,5; topo do bloco a 184,0–184,6 m; fotografias
# de fora): face de fora a poente, eixo da passagem para ESE
PB_O = np.array([-97.75, -21.45])
PB_N = unit((-1.0, -0.08))           # normal da face de fora (para fora da vila)
PB_T = np.array([PB_N[1], -PB_N[0]])  # ao longo da face (para norte)
PB_S0, PB_S1, PB_PROF = -5.05, 4.75, 4.5  # bloco de 9,8 m, com o arco 4,75 m do cunhal norte (fotografia de frente)
PB_TOPO, PB_CHAO_F, PB_CHAO_D = 184.40, 177.40, 178.00
# Porta da Rua do Carvalho (OSM; sem fotografia nem vão no LiDAR): na muralha norte, a s 18,5–22,5 m
PC_S = (18.0, 22.0)


def muralha():
    """Muralha da vila: faixa contínua de estação em estação (uma por metro), com as duas faces até ao chão de cada
    lado e o topo a acompanhar a cota medida. Interrompe-se na Porta da Rua do Carvalho e acaba dentro do bloco da
    Porta de Baixo."""
    m = Malha(["muro", "muro_topo"])
    ok = np.ones(len(S_V), bool)
    for i in range(len(S_V)):
        q = np.array([X_V[i], Y_V[i]]) - PB_O
        if PB_S0 < q @ PB_T < PB_S1 and -PB_PROF < q @ PB_N < 0.5:
            ok[i] = False
    fora = np.where(~ok)[0]
    if len(fora):
        ok[fora[0]] = True  # o último troço entra no bloco da porta
    P = np.stack([X_V, Y_V], 1)
    tg = np.gradient(P, axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
    E = P + nr * (L_V / 2)[:, None]
    D = P - nr * (L_V / 2)[:, None]
    zE = np.minimum(cota(*(E + nr * 0.8).T), cota(*(E + nr * 0.25).T)) - 0.4
    zD = np.minimum(cota(*(D - nr * 0.8).T), cota(*(D - nr * 0.25).T)) - 0.4
    T = T_V
    for i in range(len(S_V) - 1):
        j = i + 1
        if not (ok[i] and ok[j]):
            continue
        if PC_S[0] < (S_V[i] + S_V[j]) / 2 < PC_S[1]:
            continue
        m.face([(*E[i], zE[i]), (*E[j], zE[j]), (*E[j], T[j]), (*E[i], T[i])], "muro")
        m.face([(*D[j], zD[j]), (*D[i], zD[i]), (*D[i], T[i]), (*D[j], T[j])], "muro")
        m.face([(*E[i], T[i]), (*E[j], T[j]), (*D[j], T[j]), (*D[i], T[i])], "muro_topo")
    # pontas (no castelo, dos dois lados da Porta da Rua do Carvalho e dentro da Porta de Baixo)
    for i in range(len(S_V)):
        if not ok[i]:
            continue
        ant = i > 0 and ok[i - 1] and not (PC_S[0] < (S_V[i - 1] + S_V[i]) / 2 < PC_S[1])
        seg = i < len(S_V) - 1 and ok[i + 1] and not (PC_S[0] < (S_V[i] + S_V[i + 1]) / 2 < PC_S[1])
        if ant != seg:
            zb = min(zE[i], zD[i])
            m.face([(*E[i], zb), (*D[i], zb), (*D[i], T[i]), (*E[i], T[i])], "muro")
    n_ok = np.where(ok)[0]
    m.objeto("muralha", "muralha", GRUPOS["Muralha_vila"], uv=2.0, vcor=patine_muro)
    MEDIDAS["muralha"] = {"topo_inicio": round(float(T_V[1]), 2), "topo_fim": round(float(T_V[n_ok[-1]]), 2),
                          "comprimento": round(float(S_V[n_ok[-1]] - S_V[0]), 1)}


def porta_carvalho():
    """Porta da Rua do Carvalho: arco de volta perfeita simples por baixo da muralha norte (só o OSM a dá; o
    LiDAR não vê a passagem, que fica coberta, e não há fotografia). Estimativa: 2,6 m de largura e 3,4 m de
    altura, à cota da rua (185,0 m dos dois lados)."""
    m = Malha(["muro", "muro_topo", "intradorso", "calcada", "cantaria_lisa"])
    i0 = int(np.argmin(np.abs(S_V - PC_S[0])))
    i1 = int(np.argmin(np.abs(S_V - PC_S[1])))
    pa, pb = np.array([X_V[i0], Y_V[i0]]), np.array([X_V[i1], Y_V[i1]])
    c = (pa + pb) / 2
    t = unit(pb - pa)
    nrm_dentro = np.array([-t[1], t[0]])  # o lado esquerdo do eixo, que vai de nascente para poente: sul (vila)
    larg = float(np.mean(L_V[i0:i1 + 1]))
    zt = float(np.mean(T_V[i0:i1 + 1]))
    # face de fora: a norte (para fora da vila), normal -nrm_dentro
    nf = -nrm_dentro
    f = Plano(c + nf * larg / 2, -t, nf)
    h = float(np.linalg.norm(pb - pa)) / 2
    chao = float(cota(*(c + nf * (larg / 2 + 1.0))))
    chao_d = float(cota(*(c - nf * (larg / 2 + 1.0))))
    passagem(m, f, -h, h, larg, min(chao, chao_d) - 0.4, zt, 2.6, 3.4, chao, chao_d)
    aduelas(m, f, 0.0, 2.6, chao, chao + 3.4, 0.36, 0.03, n=11, mat="cantaria_lisa")
    fi = Plano(c - nf * larg / 2, t, -nf)
    aduelas(m, fi, 0.0, 2.6, chao_d, chao_d + 3.4, 0.36, 0.03, n=11, mat="cantaria_lisa")
    m.objeto("porta_carvalho", "porta_carvalho", GRUPOS["Muralha_vila"], uv=2.0, vcor=patine_muro)
    MEDIDAS["porta_carvalho"] = {"chao_fora": round(chao, 2), "chao_dentro": round(chao_d, 2), "topo": round(zt, 2),
                                 "centro": [round(float(c[0]), 1), round(float(c[1]), 1)]}
    return c, zt


def porta_baixo():
    """Porta de Baixo (fotografias de fora; LiDAR): bloco de cantaria regular mais largo que a muralha, com a
    passagem coberta de c. 5 m. Por fora: arco de volta perfeita com duas arquivoltas (a de fora mais larga e
    recuada 0,3 m), impostas salientes no arranque e, no alto, as cinco mísulas dos matacães, que sobem um pouco
    acima do topo. Por dentro: arco simples."""
    m = Malha(["cantaria_porta", "muro_topo", "intradorso", "calcada", "cantaria_lisa"])
    f = Plano(PB_O, PB_T, PB_N)
    zb = min(cota(*f.xy(0, 1.0)), cota(*f.xy(0, -PB_PROF - 1.0))) - 0.6
    # fotografia de frente, à escala da altura do LiDAR (7,0 m do chão de fora ao topo: 40,7 px/m)
    W1, H1 = 2.30, 3.70   # vão
    W2, H2 = 2.90, 4.00   # arquivolta de fora, recuada; as aduelas chegam a 3,9 m de extradorso
    passagem(m, f, PB_S0, PB_S1, PB_PROF, zb, PB_TOPO, W1, H1, PB_CHAO_F, PB_CHAO_D, mat="cantaria_porta",
             degrau=(W2, H2, 0.30), arco_dentro=(2.30, 3.90))
    # aduelas da arquivolta de fora, ligeiramente salientes (extradorso de c. 3,5 m)
    aduelas(m, f, 0.0, W2, PB_CHAO_F, PB_CHAO_F + H2, 0.50, 0.06, n=15, mat="cantaria_lisa", ombreiras=False)
    # aduelas do vão, na face do degrau
    aduelas(m, f, 0.0, W1, PB_CHAO_F, PB_CHAO_F + H1, 0.30, 0.02, n=11, mat="cantaria_lisa", ombreiras=False, d0=-0.30)
    # impostas: friso saliente no arranque dos arcos, sobre as ombreiras
    z_imp = PB_CHAO_F + H1 - W1 / 2
    for sg in (-1, 1):
        m.caixa_face(f, *sorted((sg * (W1 / 2), sg * (W2 / 2 + 0.50))), -0.30, 0.10, z_imp - 0.20, z_imp, "cantaria_lisa")
    # mísulas dos matacães: cinco, de duas fiadas, em 2,3 m, um pouco a sul do eixo do arco (fotografias); sobem
    # c. 0,6 m acima do topo
    for k in range(5):
        sc = -0.15 - 1.0 + 0.5 * k
        # blocos de pedra em degrau (a fiada de cima sai mais), de frente direita
        for z0, z1, sal in ((PB_TOPO - 0.35, PB_TOPO + 0.12, 0.36), (PB_TOPO + 0.12, PB_TOPO + 0.60, 0.62)):
            m.caixa_face(f, sc - 0.15, sc + 0.15, -0.3, sal, z0, z1, "cantaria_porta", fundo=True)
    m.objeto("porta_baixo", "porta_baixo", GRUPOS["Muralha_vila"], uv=2.0, vcor=patine_muro)
    MEDIDAS["porta_baixo"] = {"chao_fora": PB_CHAO_F, "topo": PB_TOPO, "vao": [W1, H1], "arquivolta": [W2, H2],
                              "altura_fora": round(PB_TOPO - PB_CHAO_F, 2)}


# ---------------------------------------------------------------- terreno e envolvente

def poligono_recinto(r_menos=0.0):
    return shapely.Polygon([pt_cerca(a, rin_az(a) - r_menos) for a in AZ_C])


def poligono_castelo():
    return shapely.Polygon([pt_cerca(a, rout_az(a)) for a in AZ_C])


def poligono_muralha(folga=0.0):
    pts = []
    for i in range(len(S_V)):
        pts.append(shapely.Point(X_V[i], Y_V[i]).buffer(L_V[i] / 2 + folga, 8))
    return shapely.union_all(pts)


def caminhos():
    out_c, out_a = [], []
    for f in F["caminhos"]:
        if f["geometria"]["type"] != "LineString":
            continue
        cl = f["classe"]
        larg = {"residential": 5.0, "service": 4.0, "pedestrian": 4.0, "unknown": 4.0, "track": 3.0}.get(cl, 2.0)
        g = shapely.LineString(f["geometria"]["coordinates"]).buffer(larg / 2, cap_style="flat")
        (out_a if cl == "residential" and f.get("nome") and "Avenida" in (f.get("nome") or "") else out_c).append(g)
    return (shapely.union_all(out_c) if out_c else shapely.Polygon()), (shapely.union_all(out_a) if out_a else shapely.Polygon())


def construir_terreno():
    passo = 0.5
    xs = np.arange(X0, X1 + 1e-6, passo)
    ys = np.arange(Y0, Y1 + 1e-6, passo)
    X, Y = np.meshgrid(xs, ys)
    Z = cota(X, Y)
    # na Porta de Baixo o MDT tem pontos do muro classificados como solo: o chão da passagem e da rua à frente
    # segue a soleira (177,4 m fora, 178,0 m dentro)
    q = np.stack([X - PB_O[0], Y - PB_O[1]], -1)
    s_, d_ = q @ PB_T, q @ PB_N
    zona = (s_ > PB_S0 + 0.3) & (s_ < PB_S1 - 0.3) & (d_ > -PB_PROF - 1.0) & (d_ < 4.0)
    plano = PB_CHAO_F + (PB_CHAO_D - PB_CHAO_F) * np.clip(-d_ / PB_PROF, 0, 1)
    Z = np.where(zona & (np.abs(s_) < 3.0), np.minimum(Z, plano), Z)
    ny, nx = X.shape
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    idx = np.arange(ny * nx).reshape(ny, nx)
    Fq = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    mats = ["terreiro", "relva", "rocha", "calcada", "asfalto", "verde_viz"]
    ob = malha_rapida("Terreno", V, Fq, mats[0], GRUPOS["Terreno"], "terreno")
    for mt in mats[1:]:
        ob.data.materials.append(MATS[mt])
    cx = V[Fq].mean(1)
    pts = shapely.points(cx[:, 0], cx[:, 1])
    mi_ = np.zeros(len(Fq), np.int32)
    # fora: verde apagado onde o ortofotomapa é verde (hortas, jardins), lajeado neutro no resto
    ci = np.clip(((cx[:, 0] - OX0) / OP).astype(int), 0, ORTO.shape[1] - 1)
    ri = np.clip(((OY1 - cx[:, 1]) / OP).astype(int), 0, ORTO.shape[0] - 1)
    o = ndi.uniform_filter(ORTO, size=(9, 9, 1))[ri, ci]
    verde = (o[:, 1] > o[:, 0] * 1.04) & (o[:, 1] > o[:, 2] * 1.08)

    def manchas(mask, minimo=12):
        """Arredonda uma máscara por face (grelha do terreno) e tira as manchas pequenas: sem isto o relvado e
        a rocha saíam aos quadrados soltos de 50 cm (auditoria de novembro de 2026)."""
        g = mask.reshape(ny - 1, nx - 1).astype(float)
        g = ndi.gaussian_filter(g, 1.2) > 0.5
        lab_, k_ = ndi.label(g)
        if k_:
            tam = ndi.sum(g, lab_, range(1, k_ + 1))
            g &= np.isin(lab_, 1 + np.nonzero(tam >= minimo)[0])
        return g.ravel()

    verde = manchas(verde, 20)
    mi_[verde] = 5
    calc, asf = caminhos()
    if not calc.is_empty:
        mi_[shapely.contains(calc, pts)] = 3
    if not asf.is_empty:
        mi_[shapely.contains(asf, pts)] = 4
    # recinto: relvado, com os afloramentos de granito onde o chão sobressai (relevo local do MDT)
    rec = poligono_recinto(0.0)
    dentro = shapely.contains(rec, pts)
    rel = ndi.gaussian_filter(MDT - ndi.gaussian_filter(MDT, 4.0), 0.8)
    rl = amostra(rel, LX0, LY0, LP, cx[:, 0], cx[:, 1])
    mi_[dentro] = 1
    mi_[dentro & manchas(rl > 0.09)] = 2
    # carreiro de lajes da entrada (fotografia de cima) e pátio da barbacã
    carreiro = shapely.LineString([(-3.0, -29.5), (-3.0, -24.0), (-2.0, -19.5)]).buffer(0.9)
    patio = shapely.Polygon([(-5.5, -28.6), (2.6, -28.6), (2.6, -32.6), (-5.5, -32.6)])
    mi_[shapely.contains(carreiro, pts) | shapely.contains(patio, pts)] = 0
    ob.data.polygons.foreach_set("material_index", mi_)
    uv_mundo(ob, 2.0)
    # lados da maqueta
    m = Malha(["base"])
    zb = float(Z.min()) - 1.5
    borda = [(X[0, i], Y[0, i], Z[0, i]) for i in range(nx)] + [(X[i, -1], Y[i, -1], Z[i, -1]) for i in range(1, ny)] + \
            [(X[-1, i], Y[-1, i], Z[-1, i]) for i in range(nx - 2, -1, -1)] + [(X[i, 0], Y[i, 0], Z[i, 0]) for i in range(ny - 2, 0, -1)]
    for a, b in zip(borda, borda[1:] + borda[:1]):
        m.face([(a[0], a[1], zb), (b[0], b[1], zb), b, a], 0)
    m.objeto("base", None, GRUPOS["Terreno"], uv=None)


def monumento_pl():
    """Planta de tudo o que é monumento (para tirar as casas e as árvores de cima dele)."""
    torre_pl = shapely.Polygon([W(-A - 1, -A - 1), W(A + 1, -A - 1), W(A + 1, A + 1), W(-A - 1, A + 1)])
    pb = shapely.Polygon([Plano(PB_O, PB_T, PB_N).xy(s, d) for s, d in ((PB_S0, 0.6), (PB_S1, 0.6), (PB_S1, -PB_PROF), (PB_S0, -PB_PROF))])
    return shapely.union_all([poligono_castelo().buffer(0.5), torre_pl, poligono_muralha(0.3), pb,
                              shapely.Polygon(TORRE_SE).buffer(0.3), shapely.Polygon(PLAT_NO).buffer(0.3),
                              shapely.Polygon(JUNTA_NO).buffer(0.3), shapely.Polygon(BARB_O).buffer(0.3),
                              shapely.Polygon(BARB_S).buffer(0.3)])


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


def construir_envolvente(mon):
    """Casas à volta em volumes simples: plantas do OSM e, onde o OSM não as tem, do LiDAR; recortadas pela maqueta
    e pelo monumento; cornija e cume medidos no MDS, telhado de águas por recuo da planta. Um só cinzento."""
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
        if g.intersection(mon).area > 0.5 * g.area:
            continue  # a torre, a cerca e a muralha também estão no OSM como edifícios
        g = g.intersection(caixa).difference(mon)
        for p in (g.geoms if g.geom_type in ("MultiPolygon", "GeometryCollection") else [g]):
            if p.geom_type == "Polygon" and p.area >= 6:
                plantas.append(p.simplify(0.3))
    osm = shapely.union_all(plantas) if plantas else shapely.Polygon()
    n_osm = len(plantas)
    plantas += casas_lidar(mon, osm)
    n = sum(_casa(m, p) for p in plantas)
    print("casas", n, "(OSM", n_osm, ")")
    MEDIDAS["casas"] = {"total": n, "osm": n_osm}
    m.objeto("envolvente", "envolvente", GRUPOS["Envolvente"], uv=2.0, recalc=False)  # faces já orientadas para fora


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

print("terreno…")
construir_terreno()
print("torre…")
torre()
escada_metalica()
print("cerca…")
cerca()
postigos()
escadas()
torres_cerca()
cisterna()
barbaca()
print("muralha…")
muralha()
PC_C, PC_Z = porta_carvalho()
porta_baixo()
print("envolvente…")
MON = monumento_pl()
construir_envolvente(MON)
construir_arvores(MON)

# medidas de conferência
PT = np.load(AQUI / "dados" / "torre.npz")["pontos"]
uvp = np.stack([PT[:, :2] @ EU, PT[:, :2] @ EV], 1)
dentro = (np.abs(uvp[:, 0]) < A + 0.8) & (np.abs(uvp[:, 1]) < A + 0.8) & ~np.isin(PT[:, 3], (3, 4, 5))  # sem pontos soltos de 'vegetação'
MEDIDAS["torre"] = {"ponto_mais_alto": round(float(PT[dentro, 2].max()), 2), "planta_corpo": 2 * A, "planta_parapeito": 2 * (A + SAL),
                    "chao": CHAO_T, "altura_ate_ameias_sse": round(Z_AMEIA - CHAO_T["SSE"], 2),
                    "altura_ate_pontas_sse": round(Z_PONTA - CHAO_T["SSE"], 2), "azimute_faces": [63.3, 153.3]}
MEDIDAS["cerca"] = {"adarve_norte": round(adarve_az(0), 2), "adarve_sul": round(adarve_az(230), 2),
                    "diametro_ns": round(rout_az(0) + rout_az(180), 1), "diametro_eo": round(rout_az(90) + rout_az(270), 1),
                    "recinto_m2": round(poligono_recinto().area)}


def _olhar(x, y, up=0.35):
    d = unit((x, y))
    return [round(float(d[0]), 2), up, round(float(-d[1]), 2)]


_pb = Plano(PB_O, PB_T, PB_N)
_pne, _pn = pt_cerca(55, rin_az(55) - 1.0), pt_cerca(345, rin_az(345) - 1.0)
_porta = FACES["NNO"].xy(PORTA_S, 0.6)
MEDIDAS["ancoras"] = {
    "torre": _anc(0, 0, 212.5),
    "coroamento": _anc(*W(-A - 0.6, -A - 0.6), 210.6),
    "porta_torre": _anc(*_porta, 199.0),
    "cerca": _anc(*pt_cerca(300, rout_az(300) + 0.5), 196.0),
    "escadas": _anc(*_pne, 197.5),
    "torres_cerca": _anc(28.0, -14.5, 199.0),
    "cisterna": _anc(-6.2, 21.5, 193.6),
    "porta_castelo": _anc(-3.0, -31.0, 192.8),
    "muralha": _anc(X_V[40], Y_V[40], T_V[40] + 2.0),
    "porta_baixo": _anc(*_pb.xy(0, 0.3), PB_TOPO + 1.8),
    "porta_carvalho": _anc(PC_C[0], PC_C[1], PC_Z + 1.8),
}
MEDIDAS["olhar"] = {
    "coroamento": _olhar(*(W(-1, -1)), 0.2),
    "porta_torre": _olhar(*FACES["NNO"].n, 0.3),
    "escadas": _olhar(*(CENTRO - _pne), 0.3),
    "torres_cerca": _olhar(1, -0.6, 0.35),
    "porta_castelo": _olhar(0, 1, 0.8),
    "muralha": _olhar(0.2, 1, 0.35),
    "porta_baixo": _olhar(*PB_N, 0.2),
    "porta_carvalho": _olhar(0, 1, 0.3),
}
(AQUI / "dados" / "medidas.json").write_text(json.dumps(MEDIDAS, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in MEDIDAS.items() if k not in ("ancoras", "olhar")}, ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
(AQUI / "saida").mkdir(exist_ok=True)
glb = AQUI / "saida" / "melgaco-bruto.glb"
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
    final = SAIDA / "melgaco.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "melgaco.blend"))

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
    AZ_SOL = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--sol=")), 215))
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

    def vista(nome, cam_xy, cam_z, alvo_xy, alvo_z, lente=35, res=(1000, 1000), destino=None, perto=0.1, orto=None):
        if SO and nome not in SO:
            return
        co.location = Vector((cam_xy[0], cam_xy[1], cam_z))
        alvo = Vector((alvo_xy[0], alvo_xy[1], alvo_z))
        co.rotation_euler = (alvo - co.location).to_track_quat("-Z", "Y").to_euler()
        if orto:
            cam.type = "ORTHO"
            cam.ortho_scale = orto
        else:
            cam.type = "PERSP"
            cam.lens = lente
        cam.clip_start = perto
        cam.clip_end = 3000
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.render.filepath = str(destino or (pasta / f"{nome}.jpg"))
        bpy.ops.render.render(write_still=True)
        print("vista", nome)

    # capa: de sueste, do alto, com o castelo à frente e a muralha da vila atrás
    vista("capa", (75, -85), 258, (-22, 2), 188, lente=36, res=(1280, 800), destino=SAIDA / "capa.jpg")
    if "--capa" not in sys.argv:
        # m12: do alto, de sudoeste, com a torre no meio do recinto
        vista("m12_sudoeste", (-48, -62), 232, (2, 0), 194, lente=40, res=(1000, 615))
        # m11: drone a sul (um pouco a nascente), a olhar para norte
        vista("m11_sul", (14, -58), 238, (0, 2), 191, lente=30, res=(1110, 610))
        # m09: torre de baixo, do sueste do recinto (faces SSE à esquerda e ENE à direita)
        vista("m09_torre", tuple(W(5.5, -11.5)), 191.6, tuple(W(-0.5, 0)), 203.0, lente=17, res=(730, 1300), perto=0.3)
        # m03: do recinto a norte, a olhar para SSE (porta e escada metálica na face NNO)
        vista("m03_norte", (-4.0, 24.5), 197.6, (1.0, -2.0), 197.0, lente=24, res=(1264, 632), perto=0.3)
        # m02: de sul-sudoeste, do alto da muralha a sul, a olhar para NNE
        vista("m02_sul", (-14.0, -28.0), 197.0, (2.0, 4.0), 197.0, lente=24, res=(1000, 628), perto=0.3)
        # m05: do alto da torre a olhar para sul (entrada e barbacã)
        vista("m05_entrada", tuple(W(0.0, -4.55)), 209.1, (-3.0, -31.0), 188.0, lente=26, res=(730, 1300), perto=0.3)
        # m06: Porta de Baixo de fora (poente)
        vista("m06_porta_baixo", tuple(PB_O + PB_N * 9.5 + PB_T * 0.6), 178.9, tuple(PB_O), 180.6, lente=28, res=(800, 920), perto=0.3)
        # muralha da vila, de noroeste, do alto
        vista("vila_noroeste", (-112, 62), 222, (-62, 8), 184, lente=34, res=(1200, 800))
        vista("porta_carvalho", (PC_C[0] - 1, PC_C[1] + 8), 186.8, (PC_C[0], PC_C[1]), 187.0, lente=28, res=(900, 700), perto=0.3)
        vista("barbaca", (-2.0, -46.0), 199.0, (-3.0, -31.0), 189.0, lente=30, res=(1000, 750), perto=0.3)
        vista("torre_se", (45.0, -30.0), 203.0, (27.0, -14.0), 192.0, lente=30, res=(1000, 750), perto=0.3)
        # planta (ortográfica, norte para cima), para comparar com o ortofotomapa
        co.location = Vector(((X0 + X1) / 2, (Y0 + Y1) / 2, 400))
        co.rotation_euler = (0, 0, 0)
        cam.type = "ORTHO"
        cam.ortho_scale = X1 - X0
        sc.render.resolution_x, sc.render.resolution_y = 1580, 860
        sc.render.filepath = str(pasta / "topo.jpg")
        if not SO or "topo" in SO:
            bpy.ops.render.render(write_still=True)
