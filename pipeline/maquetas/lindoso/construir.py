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
from comum import (ALA_OESTE, ALA_SUL, CORREDOR, LAT0, LON0, MEIA_TORRE, PASSAGEM_SUL, PORTA_CERCA_SUL,  # noqa: E402
                   PORTA_NORTE, POSTIGO, ROT_TORRE, TORRE_DA_PORTA, ZONA_CASTELO, _bloco)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pipeline/, para o kit comum
from kit.geometria import normais, amostra, quad, arco_pts, _sd, _anc  # noqa: E402
from kit.texturas import ruido, fbm, textura_cantaria, textura_cubos, textura_rocha, textura_telha, textura_terra, imagem  # noqa: E402
from kit.fotos import textura_foto, fontes_fotos  # noqa: E402
from kit.cena import Obra, uv_mundo  # noqa: E402
from kit.materiais import MATS, malha_rapida, material  # noqa: E402
from kit.arquitetura import aduelas, contorno_vao, parede, passagem, vao  # noqa: E402

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


def cota(x, y):
    """Chão (MDT de 50 cm)."""
    return amostra(MDT, LX0, LY0, LP, x, y)


def superficie(x, y):
    """O ponto mais alto das classes solo, edifícios e muros (sem vegetação), em células de 25 cm."""
    return amostra(SUP, SX0, SY0, SP, x, y, ordem=0)


# ---------------------------------------------------------------- ruído
# ---------------------------------------------------------------- ruído


# ---------------------------------------------------------------- malhas
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


# ---------------------------------------------------------------- materiais e texturas


def textura_relva(n=512, metros=8.0, semente=7):
    """Relvado do recinto (fotografias: verde médio, com manchas amareladas e terra à vista nos carreiros)."""
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    g = fbm(xx, yy, 1.2 * m_px, semente)
    g2 = fbm(xx, yy, 0.15 * m_px, semente + 5)
    base = np.array([0.36, 0.46, 0.22])
    amarelo = np.array([0.47, 0.52, 0.27])
    t = np.clip((g - 0.45) * 2.5, 0, 1)[..., None]
    tex = base * (1 - t) + amarelo * t
    tex = tex * (0.85 + 0.3 * g2[..., None])
    pinta = np.random.default_rng(semente).random((n, n))
    tex = np.where((pinta > 0.9)[..., None], tex * 1.15, tex)
    return np.clip(tex, 0, 1), g2


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
    """Erva do monte dos espigueiros: verde vivo com manchas mais amareladas (Street View e fotografias de drone)."""
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    g = fbm(xx, yy, 1.5 * m_px, semente)
    g2 = fbm(xx, yy, 0.12 * m_px, semente + 5)
    palha = np.array([0.47, 0.50, 0.29])  # manchas amareladas
    verde = np.array([0.30, 0.45, 0.19])
    t = np.clip((g - 0.30) * 2.2, 0, 1)[..., None]
    tex = palha * (1 - t) + verde * t
    tex = tex * (0.82 + 0.36 * g2[..., None])
    return np.clip(tex, 0, 1), g2


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
# Pedra fotografada (kit.fotos, fotos CC0 da Poly Haven; créditos em medidas.json "texturas_foto"), com o tom das
# fotografias de Lindoso. Com uv=3,0 e escala=0,5 cada imagem cobre 6 m: torre e cerca medieval em silhares (2 fotos
# de 3 m, fiadas de c. 0,37 m); muralha abaluartada (escala 0,8: 3,75 m) em alvenaria miúda.
TEX, ALT = textura_foto("stone_brick_wall_001", metros=6.0, cor=(0.60, 0.585, 0.55), contraste=1.35, saturacao=0.8, variacao=0.15, metros_foto=3.0,
                        liquen=0.10, semente=91)
material("cantaria_torre", (1, 1, 1), 0.9, imagem("cantaria_torre", TEX), normal=imagem("cantaria_torre_n", normais(ALT, 3.0)),
         forca=0.8, escala=0.5, vcor=True)
TEX, ALT = textura_foto("stone_brick_wall_001", metros=6.0, cor=(0.59, 0.575, 0.54), contraste=1.35, saturacao=0.8, variacao=0.15, metros_foto=3.0,
                        liquen=0.10, semente=92, rodar=True)
material("muro_medieval", (1, 1, 1), 0.92, imagem("muro_medieval", TEX), normal=imagem("muro_medieval_n", normais(ALT, 3.2)),
         forca=0.9, escala=0.5, vcor=True)
PAL_M = np.array([(0.575, 0.565, 0.540), (0.650, 0.640, 0.610), (0.460, 0.450, 0.430), (0.600, 0.570, 0.510),
                  (0.530, 0.525, 0.510)])
TEX, ALT = textura_foto("old_stone_wall", metros=3.75, cor=(0.58, 0.565, 0.53), contraste=1.0, metros_foto=3.75,
                        liquen=0.12, semente=93)
material("alvenaria", (1, 1, 1), 0.95, imagem("alvenaria", TEX), normal=imagem("alvenaria_n", normais(ALT, 4.0)), forca=1.0, escala=0.8, vcor=True)
TEX, ALT = textura_foto("old_stone_wall", metros=6.0, cor=(0.58, 0.575, 0.54), contraste=0.95, metros_foto=3.0,
                        liquen=0.12, semente=94, rodar=True)
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
material("hera", (0.13, 0.24, 0.09), 0.9)
material("chapa", (0.30, 0.31, 0.32), 0.55)
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
    vértice, faces a prumo até abaixo do chão de fora. Dois objetos: a cerca medieval e a muralha abaluartada, que o
    tracar.py traça à parte (topo acima ou abaixo de 466 m), para o topo não fazer rampas de uma para a outra."""
    mm = {True: Malha(["muro_medieval", "alvenaria", "muro_topo", "erva"]), False: Malha(["muro_medieval", "alvenaria", "muro_topo", "erva"])}
    for reg in MUR:
        med = bool(reg.get("medieval", np.mean(reg["aneis"][0]["topo"]) > ALTO_MEDIEVAL))
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
            zt = ts[j]
            mm[med].face([(x, y, z) for (x, y), z in zip(cs, zt)], 2)
        for a in reg["aneis"]:
            c = np.array(a["contorno"])
            tp = np.array(a["topo"])
            ch = np.minimum(np.array(a["chao"]), cota(c[:, 0], c[:, 1])) - 0.6
            n = len(c)
            for i in range(n):
                j = (i + 1) % n
                mm[med].face([(*c[i], ch[i]), (*c[j], ch[j]), (*c[j], tp[j]), (*c[i], tp[i])], 0 if med else 1)
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
    """Porta Norte (fotografia e Street View, de dentro do revelim): vão em arco de 1,5 × 2,4 m aberto num rebaixo
    da cantaria onde a ponte levadiça encaixa quando sobe, entre duas pilastras e com um lintel por cima; os dois
    braços de madeira da levadiça saem por rasgos acima do arco e as correntes descem às pontas do tabuleiro, que
    está baixado sobre o fosso. Atravessa a cortina norte da muralha abaluartada, que segue dos dois lados."""
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
    soleira = chao_f + 0.25
    W_V, H_V = 1.5, 2.4                      # vão
    R0, R1, H_R, P_R = -0.95, 0.95, 2.65, 0.18  # rebaixo da levadiça (largura, altura, fundo)
    m = Malha(["alvenaria", "muro_topo", "intradorso", "calcada", "cantaria_lisa", "madeira", "ferro", "sombra"])
    fr = Plano(f.xy(0, -P_R), t, n=n)       # face do rebaixo
    passagem(m, fr, s0, s1, 2 * meia - P_R, chao_f - 1.0, z_topo, W_V, H_V, soleira, chao_d, mat="alvenaria",
             mat_topo="muro_topo")
    aduelas(m, fr, 0.0, W_V, soleira, soleira + H_V, 0.26, 0.04, n=11)
    # pilastras e lintel de cantaria à volta do rebaixo, rentes à face do muro
    for a_, b_ in ((s0, R0), (R1, s1)):
        m.caixa_face(f, a_, b_, -P_R, 0.0, chao_f - 1.0, z_topo, "alvenaria", topo=False)
    m.caixa_face(f, R0, R1, -P_R, 0.0, soleira + H_R, z_topo, "alvenaria", topo=False)
    m.caixa_face(f, R0, R1, -P_R, 0.0, chao_f - 1.0, soleira, "cantaria_lisa", topo=False)
    # remate do lintel: cornija saliente por cima da porta (a caixa de cima na fotografia)
    m.caixa_face(f, s0 - 0.08, s1 + 0.08, -0.1, 0.12, z_topo - 0.05, z_topo + 0.22, "muro_topo")
    # braços da levadiça: saem por dois rasgos do lintel e avançam 1,6 m; correntes às pontas do tabuleiro
    z_b = soleira + 3.25
    L_T = 2.5                                # tabuleiro (= altura do rebaixo menos a soleira)
    for sb in (-0.72, 0.72):
        m.caixa_face(f, sb - 0.11, sb + 0.11, -0.02, 0.005, z_b - 0.13, z_b + 0.13, "sombra", topo=False)
        m.caixa_face(f, sb - 0.08, sb + 0.08, -0.6, 1.6, z_b - 0.09, z_b + 0.09, "madeira", fundo=True)
        m.barra(f.p(sb, 1.52, z_b - 0.09), f.p(sb * 1.15, L_T - 0.08, soleira + 0.12), 0.018, "ferro", k=4)
    # grade de ferro no vão, recuada 0,3 m
    r = W_V / 2
    zc = soleira + H_V - r
    for sg in np.arange(-r + 0.1, r - 0.05, 0.14):
        ztop = zc + math.sqrt(max(r * r - sg * sg, 0)) - 0.03
        m.barra(fr.p(sg, -0.3, soleira + 0.02), fr.p(sg, -0.3, ztop), 0.012, "ferro", k=4)
    for z in (soleira + 0.15, soleira + 1.2, zc):
        m.barra(fr.p(-r + 0.02, -0.3, z), fr.p(r - 0.02, -0.3, z), 0.02, "ferro", k=4)
    # fosso por baixo do tabuleiro, com as paredes de cantaria, e o tabuleiro de pranchas baixado sobre ele
    F0, F1, FD = -1.0, 1.0, chao_f - 1.1
    m.face([f.p(F0, 0.0, FD), f.p(F1, 0.0, FD), f.p(F1, L_T + 0.1, FD), f.p(F0, L_T + 0.1, FD)][::-1], "sombra")
    for sf in (F0, F1):
        m.face([f.p(sf, 0.0, FD), f.p(sf, L_T + 0.1, FD), f.p(sf, L_T + 0.1, chao_f), f.p(sf, 0.0, chao_f)][:: 1 if sf < 0 else -1], "cantaria_lisa")
    m.face([f.p(F0, L_T + 0.1, FD), f.p(F1, L_T + 0.1, FD), f.p(F1, L_T + 0.1, chao_f), f.p(F0, L_T + 0.1, chao_f)][::-1], "cantaria_lisa")
    for k_ in range(9):
        a_ = -0.9 + k_ * 0.2
        m.caixa_face(f, a_ + 0.01, a_ + 0.19, 0.0, L_T, soleira - 0.08, soleira, "madeira", fundo=True)
    for d_ in (0.25, L_T - 0.25):
        m.caixa_face(f, -0.95, 0.95, d_ - 0.07, d_ + 0.07, soleira - 0.2, soleira - 0.08, "madeira", fundo=True)
    m.objeto("porta_norte", "porta_norte", GRUPOS["Muralhas"], uv=3.0, vcor=patine_muro)
    MEDIDAS["porta_norte"] = {"topo": round(z_topo, 2), "vao": [W_V, H_V], "chao_fora": chao_f, "chao_dentro": chao_d,
                              "levadica": {"tabuleiro": [1.8, L_T], "fosso_fundo": round(FD, 2)}}
    return f


def _tufo(m, c, r, achat, mat, k=7, seed=0):
    """Tufo de folhagem: esfera grosseira (anéis × segmentos) achatada segundo o vetor achat (3D, |achat| = fator)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(c, float)
    na = unit(achat)
    fa = float(np.linalg.norm(achat))
    aneis = []
    for i in range(1, 4):
        th = math.pi * i / 4
        an = []
        for j in range(k):
            ph = 2 * math.pi * (j + 0.5 * (i % 2)) / k
            v = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
            v = v - na * np.dot(v, na) * (1 - fa)
            an.append(c + v * r * (0.85 + 0.3 * rng.random()))
        aneis.append(an)
    topo = c + np.array([0, 0, 1.0]) * r * (1 - (1 - fa) * abs(na[2]))
    fundo = c - np.array([0, 0, 1.0]) * r * (1 - (1 - fa) * abs(na[2]))
    for j in range(k):
        jj = (j + 1) % k
        m.face([topo, aneis[0][j], aneis[0][jj]][::-1], mat)
        for i in range(2):
            m.face([aneis[i][j], aneis[i + 1][j], aneis[i + 1][jj], aneis[i][jj]], mat)
        m.face([fundo, aneis[2][jj], aneis[2][j]][::-1], mat)


def hera():
    """A hera que cobre a cortina a sudoeste da Porta Norte (Street View): tufos de folhagem verde-escura pela face de
    fora e por cima do muro, a descer mais junto à porta."""
    reg = next((r for r in MUR if r.get("hera")), None)
    if reg is None:
        return
    m = Malha(["hera"])
    c = np.array(PORTA_NORTE["c"])
    az = math.radians(PORTA_NORTE["az"])
    t = np.array([math.sin(az), math.cos(az)])
    n = np.array([-t[1], t[0]])
    z_t = float(np.median(reg["aneis"][0]["topo"]))
    f = Plano(c + n * 1.0, t, n=n)
    rng = np.random.default_rng(5)
    k = 0
    for sv in np.arange(-6.8, PORTA_NORTE["s0"] - 0.2, 0.45):
        desce = 1.0 + 2.2 * max(0.0, 1 - (PORTA_NORTE["s0"] - sv) / 5.0)   # mais densa junto à porta
        for zv in np.arange(z_t + 0.2, z_t - desce, -0.5):
            if rng.random() < 0.15:
                continue
            p_ = f.p(sv + rng.uniform(-0.2, 0.2), 0.05 + 0.15 * rng.random(), zv + rng.uniform(-0.2, 0.2))
            _tufo(m, p_, 0.32 + 0.22 * rng.random(), np.array([*n, 0.0]) * 0.55, 0, seed=k)
            k += 1
        for dv in (-0.6, -1.4):
            p_ = f.p(sv + rng.uniform(-0.2, 0.2), dv, z_t + 0.1)
            _tufo(m, p_, 0.35 + 0.2 * rng.random(), np.array([0, 0, 0.6]), 0, seed=k)
            k += 1
    m.objeto("hera_porta_norte", "vegetacao", GRUPOS["Vegetacao"], uv=1.0)


def face_muro(o, nrm):
    """Ponto da face de fora do muro traçado (POL_MUROS) na normal que passa por o."""
    d_ = np.arange(-2.5, 2.5, 0.02)
    q = np.asarray(o) + np.outer(d_, nrm)
    dentro = shapely.contains_xy(POL_MUROS, q[:, 0], q[:, 1])
    return q[np.nonzero(dentro)[0].max()] + np.asarray(nrm) * 0.01 if dentro.any() else np.asarray(o, float)


def postigo():
    """Porta da traição (Street View e fotografia): arco estreito de 0,9 × 2,2 m na face de fora do troço noroeste da
    cerca, a 1,5 m do cunhal oeste da torre, emparedado ao fundo, e a rampa metálica com guarda que sobe até ele da
    lice, encostada à face (o LiDAR mostra-a como uma faixa 0,3–0,6 m acima da lice)."""
    o = np.array(POSTIGO["o"])
    t = unit(POSTIGO["t"])
    nrm = np.array([t[1], -t[0]])  # para fora (noroeste)
    w, h = POSTIGO["vao"]
    o = face_muro(o, nrm)
    f = Plano(o, -t, n=nrm)                 # s cresce para a torre
    ch_l = float(cota(*(o + nrm * 1.8)))    # chão da lice
    ch = ch_l + 0.55                         # soleira, no topo da rampa
    m = Malha(["cantaria_lisa", "sombra", "madeira", "ferro", "chapa", "muro_medieval"])
    # o muro traçado é maciço: o vão desenha-se para fora da face (ombreiras de 0,3 m) com o fundo escuro rente a ela
    vao(m, f, 0.0, w, ch, ch + h, 0.3, arco=True, mat_lado="muro_medieval", mat_fundo="sombra", d0=0.31)
    aduelas(m, f, 0.0, w, ch, ch + h, 0.24, 0.04, n=7, d0=0.31)
    m.caixa_face(f, -w / 2 - 0.24, w / 2 + 0.24, 0.0, 0.31, ch - 0.1, ch, "cantaria_lisa", topo=True)
    # rampa: patamar à porta e um lanço que desce ao longo da face, para longe da torre
    L, Lp, Wr = POSTIGO["rampa"], 1.2, 1.2
    D0 = 0.34                                # a rampa fica à frente das ombreiras
    z_fim = float(cota(*(o + t * (L + Lp) + nrm * 1.8)))
    z_p = ch - 0.05
    m.caixa_face(f, -0.7, 0.7, D0, D0 + Wr, z_p - 0.06, z_p, "chapa", fundo=True)                 # patamar
    p0, p1 = -0.7, -0.7 - L
    m.face([f.p(p0, D0, z_p), f.p(p1, D0, z_fim), f.p(p1, D0 + Wr, z_fim), f.p(p0, D0 + Wr, z_p)][::-1], "chapa")
    m.face([f.p(p0, D0, z_p - 0.06), f.p(p1, D0, z_fim - 0.06), f.p(p1, D0 + Wr, z_fim - 0.06), f.p(p0, D0 + Wr, z_p - 0.06)], "chapa")
    m.face([f.p(p0, D0 + Wr, z_p - 0.06), f.p(p1, D0 + Wr, z_fim - 0.06), f.p(p1, D0 + Wr, z_fim), f.p(p0, D0 + Wr, z_p)][::-1], "chapa")
    # guarda: prumos, corrimão e duas travessas do lado de fora
    for sk in [0.7, p0] + list(np.linspace(p0, p1, int(L / 1.1) + 2)[1:]):
        zk = z_p if sk >= p0 else z_p + (z_fim - z_p) * (p0 - sk) / L
        m.barra(f.p(sk, D0 + Wr - 0.03, zk), f.p(sk, D0 + Wr - 0.03, zk + 1.0), 0.02, "ferro", k=4)
    for zz in (1.0, 0.55):
        m.barra(f.p(0.7, D0 + Wr - 0.03, z_p + zz), f.p(p0, D0 + Wr - 0.03, z_p + zz), 0.018, "ferro", k=4)
        m.barra(f.p(p0, D0 + Wr - 0.03, z_p + zz), f.p(p1, D0 + Wr - 0.03, z_fim + zz), 0.018, "ferro", k=4)
    m.objeto("postigo", "postigo", GRUPOS["Muralhas"], uv=3.0, vcor=patine_muro)
    MEDIDAS["postigo"] = {"centro": [round(float(o[0]), 2), round(float(o[1]), 2)], "chao": round(ch, 2), "vao": [w, h],
                          "rampa_m": L}
    return o, ch


def porta_cerca_sul():
    """Porta pequena na base da cerca sul, a c. 4,5 m do cunhal sudoeste, ao lado da ala sul do museu (fotografia de
    drone de sul e Street View): vão retangular de 0,9 × 1,7 m com ombreiras e padieira de granito, porta de
    madeira recuada."""
    o0 = np.array(PORTA_CERCA_SUL["o"])
    t = unit(PORTA_CERCA_SUL["t"])
    nrm = np.array([t[1], -t[0]])            # para fora (sul)
    o = face_muro(o0, nrm)
    f = Plano(o, t, n=nrm)
    w, h = PORTA_CERCA_SUL["vao"]
    ch = float(cota(*(o + nrm * 1.0))) + 0.05
    m = Malha(["cantaria_lisa", "sombra", "madeira", "muro_medieval"])
    # o muro traçado é maciço: o vão desenha-se para fora da face (0,3 m) com o fundo escuro rente a ela
    vao(m, f, 0.0, w, ch, ch + h, 0.3, arco=False, mat_lado="muro_medieval", mat_fundo="sombra", d0=0.31)
    for a_, b_ in ((-w / 2 - 0.3, -w / 2), (w / 2, w / 2 + 0.3)):
        m.caixa_face(f, a_, b_, 0.0, 0.33, ch, ch + h, "muro_medieval")                                # ombreiras
    m.caixa_face(f, -w / 2 - 0.35, w / 2 + 0.35, 0.0, 0.35, ch + h, ch + h + 0.42, "muro_medieval")   # padieira
    m.caixa_face(f, -w / 2 - 0.1, w / 2 + 0.1, 0.0, 0.45, ch - 0.15, ch, "cantaria_lisa")             # soleira
    m.objeto("porta_cerca_sul", "cerca", GRUPOS["Muralhas"], uv=3.0, vcor=patine_muro)
    MEDIDAS["porta_cerca_sul"] = {"centro": [round(float(o[0]), 2), round(float(o[1]), 2)], "chao": round(ch, 2), "vao": [w, h]}
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
        if min(e) > 2.9 or g.area > 40 or max(e) < 3.4:  # o que é mais curto ou mais largo são arrumos
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
        # o cume dos espigueiros fica a 3,3–3,8 m do chão; onde o MDS dá mais é uma árvore ou uma casa por cima
        if not (2.9 <= e["cume"] - e["chao_max"] <= 4.3):
            e["cume"] = e["chao_max"] + 3.55
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
    # junto aos muros, o LiDAR apanha o topo do muro: aí fica o mais baixo à volta (faz dentes de erva pelas faces acima)
    perto = ndi.binary_dilation(mur, iterations=3)
    zs = np.where(perto, ndi.minimum_filter(np.where(mur, 1e4, zs), size=5), zs)
    plat = zona & ~mur & ~exc & (zs - Z > 0.6)
    plat = ndi.binary_opening(plat, iterations=1)
    Z = np.where(plat, zs, Z)
    # junto às faces dos muros, o MDT sobe para o topo (interpolação por baixo do muro): baixa-se o chão até
    # 1,25 m dos muros para o mínimo das células à volta que não são muro
    banda = ndi.binary_dilation(mur, iterations=3) & ~mur & ~plat
    zl = np.where(mur, 1e4, Z)
    zmin = ndi.minimum_filter(zl, size=5)
    Z = np.where(banda, np.minimum(Z, np.maximum(zmin, Z - 1.2)), Z)
    # debaixo dos muros finos o MDT também sobe: as quadrículas que os atravessam espetavam erva pelas faces
    Z = np.where(mur, np.minimum(Z, ndi.minimum_filter(zl, size=9)), Z)
    # fosso da ponte levadiça, à frente da Porta Norte (o tabuleiro está por cima)
    fosso = shapely.Polygon([F_PN.xy(-1.0, 0.0), F_PN.xy(1.0, 0.0), F_PN.xy(1.0, 2.6), F_PN.xy(-1.0, 2.6)]).buffer(0.2)
    Z = np.where(shapely.contains(fosso, pts_all).reshape(X.shape), np.minimum(Z, MEDIDAS["porta_norte"]["levadica"]["fosso_fundo"]), Z)
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
    # só os afloramentos claros: com o limiar baixo o monte ficava salpicado de cinzento
    mi_[ndi.binary_opening(((rl > 0.22) | (cinza & (rl > 0.12))).reshape(ny - 1, nx - 1)).ravel()] = 2
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
    # onde há uma árvore por cima o MDS dá a copa: os arrumos pequenos ficam com 4 m e nenhum telhado passa de 4 m
    if p.area < 20:
        cume, cornija = min(cume, chao + 4.0), min(cornija, chao + 2.8)
    cume = min(cume, cornija + 4.0)
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
    # nenhuma copa entra nos muros nem passa da borda da maqueta (auditoria de novembro de 2026: 4 copas
    # tocavam nos muros e 2 pendiam fora do terreno)
    obra = Obra(excluir=("Terreno", "terreno", "base", "Base", "arvores"))
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
        rad = min(rad, (min(x - X0, X1 - x, y - Y0, Y1 - y) - 0.3) / 1.1)
        fo = obra.folga([(x, y, zc - alt * f) for f in (0.15, 0.35, 0.55)])
        if fo < rad * 1.1 + 0.15:
            rad = (fo - 0.15) / 1.1
        if rad < 1.0 or obra.folga([(x, y, zt + 0.5), (x, y, zc - alt * 0.45)]) < 0.35:
            continue
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
hera()
PG_O, PG_Z = postigo()
porta_cerca_sul()
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
    "postigo": _olhar(-0.57, 0.82, 0.2),
    "guaritas": _olhar(0.7, -0.7, 0.3),
    "baluartes": _olhar(1, 0.2, 0.5),
    "museu": _olhar(0.3, 1, 0.6),
    "espigueiros": _olhar(0.2, -1, 0.45),
}
MEDIDAS["texturas_foto"] = fontes_fotos()  # créditos das fotos de pedra
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
        vista("d049_porta_norte", tuple(_pn + _ln * 6.5), 462.8, tuple(_pn - _ln * 2.0), 467.2, lente=24, res=(780, 1040), perto=0.3)
        # Street View do Tiago: a Porta Norte de dentro do revelim, a rampa da porta da traição e a cerca sul
        vista("sv_porta_norte", tuple(F_PN.xy(0.0, 7.0)), 463.4, tuple(F_PN.xy(0.0, -8.0)), 466.0, lente=16, res=(1400, 660), perto=0.3)
        vista("sv_postigo", (-22.0, 13.0), 472.0, (-9.0, 0.5), 464.0, lente=22, res=(1400, 620), perto=0.3)
        vista("sv_cerca_sul", (-12.0, -50.0), 471.0, (-3.0, -31.5), 464.0, lente=22, res=(1400, 630), perto=0.3)
        vista("perto_postigo", (-12.4, 3.4), 464.6, (-7.6, 1.6), 463.6, lente=18, res=(1000, 700), perto=0.2)
        vista("perto_porta_sul", (-5.2, -34.6), 464.6, (-7.0, -31.8), 463.4, lente=18, res=(1000, 700), perto=0.2)
        vista("ar_oeste", (-48.0, -2.0), 485.0, (-4.0, -6.0), 463.0, lente=30, res=(1300, 800))
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
