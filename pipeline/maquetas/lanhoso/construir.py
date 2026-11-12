"""Constrói a maqueta 3D do Castelo de Lanhoso e do Santuário de Nossa Senhora do Pilar (estado atual) em Blender
e exporta-a para public/maquetas/lanhoso/lanhoso.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow rasterio
    python pipeline/maquetas/lanhoso/construir.py            # GLB
    python pipeline/maquetas/lanhoso/construir.py --vistas   # GLB + capa + vistas de verificação

Dados (em dados/): lidar.npz, superficie.npz e torre.npz (lidar.py: LiDAR 2025 da DGT, folha 187514),
orto.jpg (ortofoto.py: ortofotomapa 2021 da DGT), fontes.json (recolher.py: OSM via Overture).
As plantas da torre, da cerca, da porta e do santuário estão em comum.py (medidas no LiDAR, ver o LEIAME).

O que se modela, e de onde vem:
- a torre de menagem (10,45 × 10,3 m, faces a 73,25° e 343,25°), com as cotas do LiDAR e o coroamento das
  fotografias: oito merlões de remate piramidal alto por face, contando os dos cunhais, alguns com seteira em
  arco; o adarve de tábuas por dentro do parapeito e o telhado de quatro águas de telha; a porta alta em arco de
  volta perfeita na face NNO, a 387,7 m, com a escada de pedra encostada à face e o corrimão de ferro; seteiras;
- a porta do castelo a poente: dois torreões com merlões e, entre eles, recuado, o muro com o arco ogival e as
  portas de madeira verde, a escadaria de granito que sobe da encosta e o penedo encostado ao torreão sul;
- a cerca: a muralha poente com merlões, o canto noroeste, a muralha norte com parapeito, o baluarte nordeste, a
  muralha nascente com o banco por dentro, o terraço do pátio e as escadas para os adarves; o muro fino a sudeste;
- o santuário de Nossa Senhora do Pilar (1680) a sul: nave, capela-mor e sacristia com as cotas do LiDAR, a
  fachada das fotografias (frontão, cruz, pináculos, nicho com a imagem, janelas e porta), cornijas ameadas,
  telhados de telha, o adro e as escadas;
- o terreno do LiDAR com o grande afloramento de granito do Monte de Lanhoso, o terreiro, a relva e as árvores
  (nenhuma a atravessar o monumento); as casas à volta (poucas) em volumes simples de um só cinzento.
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
import comum as C  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pipeline/, para o kit comum
from kit.geometria import ccw, orientar, normais, amostra, quad, arco_pts, ogiva, redondo, reto, _sd, _anc  # noqa: E402
from kit.texturas import ruido, fbm, textura_cantaria, textura_cubos, textura_relva, textura_rocha, textura_telha, textura_terra, imagem  # noqa: E402
from kit.fotos import textura_foto, fontes_fotos  # noqa: E402
from kit.cena import uv_mundo  # noqa: E402
from kit.materiais import MATS, malha_rapida, material  # noqa: E402
from kit.arquitetura import aduelas, contorno_vao, corrimao, merlao, painel, parede, passagem, tunel, vao  # noqa: E402

_cfg = RAIZ / "src" / "data" / "maquetas" / "lanhoso.json"
CFG = json.loads(_cfg.read_text()) if _cfg.exists() else {"origem": {"lat": C.LAT0, "lon": C.LON0}, "variante": "estado-atual", "versao": "0.1"}
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "lanhoso"
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

# a maqueta: 150 × 120 m, do castelo ao adro do santuário (a sul acaba no limite da folha do LiDAR)
X0, X1, Y0, Y1 = -70.0, 80.0, -63.0, 57.0
rng = np.random.default_rng(1121)  # 1121: Tratado de Lanhoso

EU = np.array(C.EU)
EV = np.array(C.EV)
AU, AV = C.MEIA_U, C.MEIA_V


def W(u, v):
    """Ponto do referencial da torre (u, v) para a cena (x este, y norte)."""
    return EU * u + EV * v


def WL(pts):
    return [tuple(W(*p)) for p in pts]


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


# ---------------------------------------------------------------- utilidades de Lanhoso


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


# ---------------------------------------------------------------- materiais

for _ob in list(bpy.data.objects):  # a cena de arranque do bpy traz um cubo, uma luz e uma câmara
    bpy.data.objects.remove(_ob)
RAIZ_OB = bpy.data.objects.new("Castelo_Lanhoso", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = CFG["origem"]["lat"], CFG["origem"]["lon"]
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
# "Muralhas_grupo" é o que o visor enquadra ao abrir; o terreno chama-se "Terreno"
for g in ["Terreno", "Torre", "Muralhas", "Santuario", "Envolvente", "Vegetacao"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

# Granito de Lanhoso nas fotografias (drone de 2018 ao sol, porta ao sol, pátio com tempo encoberto): cinzento
# claro e frio, de grão médio, com muitos líquenes brancos e cinzentos e manchas verdes de musgo nas juntas baixas;
# silhares grandes em fiadas de 0,30 a 0,45 m na torre; a cerca e os torreões em silhares mais irregulares.
PAL_T = np.array([(0.600, 0.595, 0.570), (0.660, 0.655, 0.630), (0.530, 0.525, 0.505), (0.615, 0.590, 0.545),
                  (0.570, 0.570, 0.560)])
# Pedra fotografada (kit.fotos, fotos CC0 da Poly Haven; créditos em medidas.json "texturas_foto"), com o tom das
# fotografias de Lanhoso. Com uv=3,0 e escala=0,5 cada imagem cobre 6 m: torre em silhares (2 fotos de 3 m, fiadas
# de c. 0,37 m); cerca em alvenaria de pedras maiores e irregulares (1 foto de 6 m, fiadas de c. 0,4 m).
TEX, ALT = textura_foto("stone_brick_wall_001", metros=6.0, cor=(0.61, 0.60, 0.575), contraste=1.0, metros_foto=3.0,
                        liquen=0.10, semente=81)
material("cantaria_torre", (1, 1, 1), 0.9, imagem("cantaria_torre", TEX), normal=imagem("cantaria_torre_n", normais(ALT, 3.0)),
         forca=0.8, escala=0.5, vcor=True)
TEX, ALT = textura_foto("old_stone_wall", metros=6.0, cor=(0.60, 0.595, 0.57), contraste=0.95, metros_foto=6.0,
                        liquen=0.10, semente=82)
material("muro", (1, 1, 1), 0.92, imagem("muro", TEX), normal=imagem("muro_n", normais(ALT, 3.2)), forca=0.9, escala=0.5, vcor=True)
TEX, ALT = textura_foto("old_stone_wall", metros=6.0, cor=(0.59, 0.59, 0.56), contraste=0.95, metros_foto=6.0,
                        liquen=0.12, semente=83, rodar=True)
material("muro_topo", (1, 1, 1), 0.95, imagem("muro_topo", TEX), normal=imagem("muro_topo_n", normais(ALT, 3.5)), forca=1.0, escala=0.5, vcor=True)
# santuário: cantaria de granito mais regular e mais clara (fachada de 1680, fotografia de frente)
PAL_I = np.array([(0.640, 0.635, 0.610), (0.690, 0.685, 0.660), (0.585, 0.580, 0.560), (0.650, 0.630, 0.590)])
TEX, ALT = textura_foto("stone_brick_wall_001", metros=6.0, cor=(0.655, 0.65, 0.625), contraste=0.8, metros_foto=3.0,
                        liquen=0.05, semente=84)
material("cantaria_igreja", (1, 1, 1), 0.88, imagem("cantaria_igreja", TEX), normal=imagem("cantaria_igreja_n", normais(ALT, 2.0)),
         forca=0.5, escala=0.5, vcor=True)
material("cantaria_lisa", (0.47, 0.465, 0.445), 0.85)
material("intradorso", (0.36, 0.355, 0.34), 0.95)
material("sombra", (0.05, 0.05, 0.05), 1.0)
material("madeira", (0.30, 0.22, 0.15), 0.8)
material("madeira_verde", (0.14, 0.36, 0.32), 0.7)   # portas da porta do castelo (fotografias: verde-azulado)
material("madeira_igreja", (0.34, 0.20, 0.13), 0.7)  # porta do santuário (fotografia: castanha)
material("tabuado", (0.52, 0.48, 0.42), 0.85)       # adarve da torre (fotografia de drone: tábuas cinzentas)
material("ferro", (0.14, 0.14, 0.15), 0.5)
material("vidro", (0.16, 0.19, 0.21), 0.2)
material("imagem", (0.78, 0.76, 0.70), 0.6)          # imagem de Nossa Senhora no nicho
material("bandeira_verde", (0.0, 0.40, 0.18), 0.8)
material("bandeira_vermelha", (0.80, 0.08, 0.10), 0.8)
TEX, ALT = textura_foto("roof_09", n=512, metros=2.0, cor=(0.5, 0.5, 0.5), metros_foto=2.0, saturacao=0.3, variacao=0.06)
# telhado da torre: telha castanho-avermelhada (fotografia de drone); santuário: telha alaranjada
material("telha_torre", (1, 1, 1), 0.8, imagem("telha_torre", np.clip(TEX * np.array([0.84, 0.50, 0.38]), 0, 1)), normal=imagem("telha_n", normais(ALT, 2.0)), forca=0.6)
material("telha", (1, 1, 1), 0.8, imagem("telha", np.clip(TEX * np.array([1.40, 0.80, 0.52]), 0, 1)), normal=imagem("telha_n2", normais(ALT, 2.0)), forca=0.6)
TEX, ALT = textura_cubos(base=(0.62, 0.61, 0.58))
material("calcada", (1, 1, 1), 0.92, imagem("calcada", TEX), normal=imagem("calcada_n", normais(ALT, 1.5)), forca=0.5)
TEX, ALT = textura_relva()
material("relva", (1, 1, 1), 0.95, imagem("relva", TEX), normal=imagem("relva_n", normais(ALT, 1.0)), forca=0.4)
COR_RELVA = tuple(float(v) for v in (np.clip(TEX, 0, 1) ** 2.2).mean((0, 1)))
TEX, ALT = textura_erva()
material("erva", (1, 1, 1), 0.95, imagem("erva", TEX * np.array([0.86, 1.02, 0.78])), normal=imagem("erva_n", normais(ALT, 1.0)), forca=0.4)
# chão do monte: a cor vem da cor por vértice (erva verde e lajes de granito, com passagens suaves); a textura só
# dá o grão (luminância da erva, com metade do contraste, média 1)
COR_ERVA = tuple(float(v) for v in (np.clip(TEX * np.array([0.86, 1.02, 0.78]), 0, 1) ** 2.2).mean((0, 1)))   # cor média linear
_l = TEX.mean(2, keepdims=True)
_l = np.clip(0.9 + 0.35 * (_l / _l.mean() - 1.0), 0, 1)   # c. 0,9 (sRGB) = 0,79 linear; compensado na cor
material("chao", (1, 1, 1), 0.93, imagem("chao", np.repeat(_l, 3, 2)), normal=imagem("chao_n", normais(ALT, 1.0)), forca=0.35, vcor=True)
material("penedo", (1, 1, 1), 0.9, imagem("penedo", np.repeat(_l, 3, 2)), vcor=True)   # penedo da porta: granito e musgo (cor por vértice, grão fino)
TEX, ALT = textura_terra()
material("terra", (1, 1, 1), 0.95, imagem("terra", TEX), normal=imagem("terra_n", normais(ALT, 1.0)), forca=0.4)
TEX, ALT = textura_rocha(semente=17)
material("rocha", (1, 1, 1), 0.85, imagem("rocha", TEX), normal=imagem("rocha_n", normais(ALT, 2.0)), forca=0.6, vcor=True)
# lajes de granito do monte (ortofotomapa e fotografias aéreas: cinzento claro rosado, com escorrências)
material("rocha_t", (1, 1, 1), 0.85, imagem("rocha_t", np.clip(TEX * np.array([1.02, 0.98, 0.95]), 0, 1)), normal=imagem("rocha_tn", normais(ALT, 1.5)), forca=0.5)
material("terreiro", (0.46, 0.45, 0.43), 0.95)       # terreiro do santuário (fotografias: saibro e lajeado cinzentos)
material("asfalto", (0.36, 0.36, 0.37), 0.95)
material("cinza_viz", (0.74, 0.74, 0.73), 0.9)       # casas vizinhas: um só cinzento claro
material("base", (0.33, 0.26, 0.20), 0.95)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)

MEDIDAS = {}


def patine_torre(co):
    """A cantaria escurece um pouco para o alto (líquenes) e junto ao chão (musgo e salpicos)."""
    z = co[:, 2]
    x, y = co[:, 0], co[:, 1]
    g = fbm(x * 0.7 + 300, y * 0.7 + z * 0.6, 4.0, 77)
    v = 0.92 + 0.16 * g
    t = np.clip((z - 386.0) / 9.0, 0, 1)
    pe = np.clip(1 - (z - 383.0) / 1.6, 0, 1) * 0.16
    c = np.stack([v * (1 - 0.05 * t), v * (1 - 0.04 * t), v * (1 - 0.05 * t)], 1)
    return np.clip(c * (1 - pe)[:, None] * np.array([0.97, 1.0, 0.96])[None, :] ** pe[:, None], 0, 1)


def patine_muro(co):
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x + 500, y + 500 + z * 0.7, 6.0, 31)
    v = 0.86 + 0.26 * g
    pe = np.clip(1 - (z - cota(x, y)) / 1.4, 0, 1) * 0.16
    c = np.stack([v * 0.99, v * (1.0 + 0.015 * (g - 0.5)), v * 0.97], 1)
    return np.clip(c * (1 - pe)[:, None], 0, 1)


def patine_igreja(co):
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x * 0.5 + 900, y * 0.5 + z * 0.5, 5.0, 53)
    v = 0.94 + 0.12 * g
    pe = np.clip(1 - (z - 377.0) / 1.5, 0, 1) * 0.12
    return np.clip(np.stack([v, v, v * 0.98], 1) * (1 - pe)[:, None], 0, 1)


# ---------------------------------------------------------------- torre de menagem

PT = np.load(AQUI / "dados" / "torre.npz")["pontos"]
_u, _v = PT[:, :2] @ EU, PT[:, :2] @ EV
_alto = (PT[:, 2] > 392.5) & ~np.isin(PT[:, 3], (3, 4, 5, 7))
_anel = _alto & ((np.abs(_u) > AU - 0.75) | (np.abs(_v) > AV - 0.75)) & (np.abs(_u) < AU + 0.1) & (np.abs(_v) < AV + 0.1)
_cent = _alto & (np.abs(_u) < 0.6) & (np.abs(_v) < 0.6)
_zs = PT[_anel, 2]
Z_ADARVE = round(float(np.median(_zs[(_zs > 393.5) & (_zs < 394.1)])), 2)    # tábuas do adarve (moda a 393,75)
Z_SILL = round(float(np.median(_zs[(_zs > 394.4) & (_zs < 395.0)])), 2)      # peitoril das ameias (394,7)
Z_PONTA = round(float(np.percentile(_zs, 98.5)), 2)                            # pontas dos merlões (396,3)
Z_MERLAO = round(float(np.median(_zs[(_zs > 395.4) & (_zs < 395.85)])), 2)    # alto do corpo dos merlões
Z_CUME = round(float(np.percentile(PT[_cent, 2], 90)), 2)                      # vértice do telhado (395,6)
CHAO_T = {}
for _nm, (_a, _b) in {"ENE": (1, 0), "NNO": (0, 1), "OSO": (-1, 0), "SSE": (0, -1)}.items():
    _s = (PT[:, 3] == 2) & ((_a * _u > AU + 0.4) & (_a * _u < AU + 2.0) & (np.abs(_v) < AV - 1) if _a else
                            (_b * _v > AV + 0.4) & (_b * _v < AV + 2.0) & (np.abs(_u) < AU - 1))
    CHAO_T[_nm] = round(float(np.percentile(PT[_s, 2], 30)), 2)
ESP_PAR = 0.70           # espessura do parapeito (fotografia de drone)
N_MERL = 8               # merlões por face, contando os dos cunhais (fotografia de drone)
LM = 0.72                # largura dos merlões
Z_SOL_PORTA = 383.1      # soleira da porta do castelo (LiDAR)
Z_PORTA = 387.73         # soleira da porta da face NNO = patamar da escada (LiDAR)
print("torre: adarve", Z_ADARVE, "peitoril", Z_SILL, "merlões", Z_MERLAO, Z_PONTA, "cume", Z_CUME, "chão", CHAO_T)


def faces_torre():
    """Planos das quatro faces: s ao longo da face, da esquerda para a direita de quem olha de fora."""
    return {
        "NNO": (Plano(W(0, AV), -EU, n=EV), AU, lambda u, v: -u),
        "SSE": (Plano(W(0, -AV), EU, n=-EV), AU, lambda u, v: u),
        "ENE": (Plano(W(AU, 0), EV, n=EU), AV, lambda u, v: v),
        "OSO": (Plano(W(-AU, 0), -EV, n=-EU), AV, lambda u, v: -v),
    }


# vãos da torre (posição ao longo da face em u ou v; cotas absolutas)
# NNO: porta alta em arco de volta perfeita no meio da face (fotografias de norte e de nascente; LiDAR: patamar
#      de 387,73 m entre u -1,0 e 1,0), seteira alta por cima
# ENE, SSE, OSO: seteiras estreitas e altas (desenho do Município e fotografias), posições estimadas
VAOS_T = {
    "NNO": [("porta", 0.0, 1.05, Z_PORTA, 1.65), ("seteira", 0.0, 0.14, 391.2, 1.05)],
    "ENE": [("seteira", 1.2, 0.14, 388.6, 1.1), ("seteira", -1.6, 0.16, 391.0, 1.2)],
    "SSE": [("seteira", 1.4, 0.14, 389.2, 1.1), ("janela", -2.0, 0.55, 391.3, 0.8)],
    "OSO": [("seteira", 0.6, 0.14, 389.6, 1.2)],
}


def torre():
    m = Malha(["cantaria_torre", "cantaria_lisa", "sombra", "madeira_verde", "tabuado", "telha_torre", "ferro", "intradorso"])
    z0 = min(CHAO_T.values()) - 1.0
    fs = faces_torre()
    for nome, (f, meia, s_de) in fs.items():
        furos, extra = [], []
        for tipo, pos, w, za, h in VAOS_T[nome]:
            s = s_de(pos, pos) if nome in ("NNO", "SSE") else s_de(pos, pos)
            s = -pos if nome == "NNO" else (pos if nome in ("SSE", "ENE") else -pos)
            if tipo == "porta":
                c = redondo(s, w, za, h, k=12)
            elif tipo == "janela":
                c = reto(s, w, za, za + h)
            else:
                c = redondo(s, w, za, h - w / 2, k=4)
            furos.append(c)
            extra.append((tipo, s, w, za, h, c))
        painel(m, f, [(-meia, z0), (meia, z0), (meia, Z_SILL), (-meia, Z_SILL)], furos, 0)
        for tipo, s, w, za, h, c in extra:
            prof = 0.9 if tipo == "porta" else 0.7
            tunel(m, f, c, 0.0, -prof, "cantaria_lisa" if tipo != "seteira" else "intradorso")
            painel(m, f, c, (), "madeira_verde" if tipo == "porta" else "sombra", d=-prof)
            if tipo == "porta":
                arco = [p for p in c if p[1] >= za + h - 1e-6]
                anel_aduelas(m, f, arco, 0.34, 0.04, "cantaria_lisa")
                # ombreiras de cantaria
                for s0_, s1_ in ((s - w / 2 - 0.30, s - w / 2), (s + w / 2, s + w / 2 + 0.30)):
                    for j in range(3):
                        za_, zb_ = za + h * j / 3 + 0.01, za + h * (j + 1) / 3 - 0.01
                        m.caixa_face(f, s0_, s1_, 0.0, 0.04, za_, zb_, 1)
                # soleira saliente
                m.caixa_face(f, s - w / 2 - 0.2, s + w / 2 + 0.2, 0.0, 0.12, za - 0.12, za, 1)
    # parapeito: face de dentro, peitoril, adarve de tábuas
    ri_u, ri_v = AU - ESP_PAR, AV - ESP_PAR
    q_in = [W(-ri_u, -ri_v), W(ri_u, -ri_v), W(ri_u, ri_v), W(-ri_u, ri_v)]
    q_out = [W(-AU, -AV), W(AU, -AV), W(AU, AV), W(-AU, AV)]
    for i in range(4):
        a, b, c_, d = q_out[i], q_out[(i + 1) % 4], q_in[(i + 1) % 4], q_in[i]
        m.face(orientar([(*a, Z_SILL), (*b, Z_SILL), (*c_, Z_SILL), (*d, Z_SILL)], np.array([0, 0, 1.0])), 1)
        m.face(orientar([(*d, Z_ADARVE), (*c_, Z_ADARVE), (*c_, Z_SILL), (*d, Z_SILL)], -np.array([*unit((a + b) / 2), 0])), 0)
    # adarve de tábuas (anel entre o parapeito e o beiral do telhado)
    rb_u, rb_v = ri_u - 0.95, ri_v - 0.95
    q_b = [W(-rb_u, -rb_v), W(rb_u, -rb_v), W(rb_u, rb_v), W(-rb_u, rb_v)]
    for i in range(4):
        a, b, c_, d = q_in[i], q_in[(i + 1) % 4], q_b[(i + 1) % 4], q_b[i]
        m.face(orientar([(*a, Z_ADARVE), (*b, Z_ADARVE), (*c_, Z_ADARVE), (*d, Z_ADARVE)], np.array([0, 0, 1.0])), 4)
    # merlões: oito por face, os dos cunhais no canto
    nm = 0
    for nome, (f, meia, _) in fs.items():
        L = 2 * meia
        passo = (L - LM) / (N_MERL - 1)
        for k in range(N_MERL):
            if k == N_MERL - 1:
                continue  # o do cunhal da direita é o da esquerda da face seguinte
            s0 = -meia + k * passo
            base = [f.xy(s0, 0.0), f.xy(s0 + LM, 0.0), f.xy(s0 + LM, -ESP_PAR), f.xy(s0, -ESP_PAR)]
            if k == 0:  # cunhal: merlão quadrado
                base = [f.xy(s0, 0.0), f.xy(s0 + ESP_PAR, 0.0), f.xy(s0 + ESP_PAR, -ESP_PAR), f.xy(s0, -ESP_PAR)]
            merlao(m, base, Z_SILL, Z_MERLAO, Z_PONTA, 0)
            nm += 1
            if k in (2, 5):  # seteira em arco no pé de alguns merlões (fotografia de drone)
                sc = s0 + LM / 2
                cont = redondo(sc, 0.2, Z_SILL + 0.08, 0.22, k=6)
                m.face(orientar([f.p(s_, 0.012, z_) for s_, z_ in cont], np.array([f.n[0], f.n[1], 0])), 2)
    # telhado de quatro águas de telha por dentro do adarve, com o rufo de cumeeira
    beiral = [W(-rb_u, -rb_v), W(rb_u, -rb_v), W(rb_u, rb_v), W(-rb_u, rb_v)]
    zb = Z_ADARVE + 0.08
    m.piramide([np.array(p) for p in beiral], zb, Z_CUME, 5)
    for p in beiral:
        m.barra((*p, zb + 0.05), (0.0, 0.0, Z_CUME + 0.05), 0.07, 5, k=5)   # telhões das arestas
    m.cilindro((W(1.4, -1.2)[0], W(1.4, -1.2)[1]), 0.18, zb + 0.6, zb + 0.95, 2, k=10)  # chaminé de ventilação
    ob = m.objeto("torre_menagem", "torre", GRUPOS["Torre"], uv=3.0, vcor=patine_torre)
    MEDIDAS["torre"] = {"lado_u": round(2 * AU, 2), "lado_v": round(2 * AV, 2), "azimute_faces": [C.ROT_TORRE, (C.ROT_TORRE + 270) % 360],
                        "chao": CHAO_T, "adarve": Z_ADARVE, "peitoril": Z_SILL, "merlao": Z_MERLAO, "pontas": Z_PONTA,
                        "cume": Z_CUME, "porta": Z_PORTA, "merloes_por_face": N_MERL, "merloes": nm,
                        "altura_nno": round(Z_PONTA - CHAO_T["NNO"], 2)}
    return ob


def bandeira():
    """Mastro com a bandeira nacional junto ao parapeito SSE, à direita do meio (fotografia de drone)."""
    m = Malha(["ferro", "bandeira_verde", "bandeira_vermelha"])
    p = W(2.6, -AV + ESP_PAR + 0.25)
    m.cilindro(p, 0.04, Z_ADARVE, 400.8, 0, k=8)
    d = unit(W(1, 0))
    h0, h1 = 399.6, 400.7
    Lb = 1.65
    for (a, b, mt) in ((0.0, 0.4, 1), (0.4, 1.0, 2)):
        pa, pb = p + d * Lb * a, p + d * Lb * b
        m.face([(*pa, h0), (*pb, h0), (*pb, h1), (*pa, h1)], mt)
        m.face([(*pa, h1), (*pb, h1), (*pb, h0), (*pa, h0)], mt)
    m.objeto("bandeira", "torre", GRUPOS["Torre"], uv=None, recalc=False)


def escada_torre():
    """Escada de pedra encostada à face NNO, a subir de nascente para poente até ao patamar da porta (LiDAR: do
    pátio a 382,9 m em u 6,0 ao patamar de 387,73 m em u 1,0; patamar até u -1,0), com corrimão de ferro."""
    m = Malha(["muro_topo", "muro", "ferro"])
    v0, v1 = AV, AV + 0.92
    vc = (v0 + v1) / 2
    z_pe = CHAO_T["NNO"] - 0.25
    n = 19
    degraus(m, W(6.05, vc), W(1.0, vc), v1 - v0, z_pe + (Z_PORTA - z_pe) / n * 0, Z_PORTA, n, 0, fundo=z_pe - 0.8)
    # patamar
    m.prisma(ccw([W(-1.0, v0), W(1.0, v0), W(1.0, v1), W(-1.0, v1)]), z_pe - 0.8, Z_PORTA, 1, mat_topo=0)
    # corrimão pelo lado de fora (fotografia do pátio)
    pts = [(*W(6.0, v1 - 0.06), z_pe + 0.3), (*W(1.0, v1 - 0.06), Z_PORTA), (*W(-1.0, v1 - 0.06), Z_PORTA)]
    corrimao(m, pts, alt=0.95, mat=2)
    m.objeto("escada_torre", "escada_torre", GRUPOS["Torre"], uv=1.5, vcor=patine_muro)
    MEDIDAS["escada_torre"] = {"degraus": n, "pe": round(z_pe, 2), "patamar": Z_PORTA, "largura": round(v1 - v0, 2)}


# ---------------------------------------------------------------- cerca

POL = {k: shapely.Polygon(WL(v)).buffer(0) for k, v in (("torreao_sul", C.TORREAO_SUL), ("torreao_norte", C.TORREAO_NORTE),
                                                         ("muro_porta", C.MURO_PORTA), ("cerca", C.CERCA), ("muro_se", C.MURO_SE))}
POL_TORRE = shapely.Polygon(WL([(-AU, -AV), (AU, -AV), (AU, AV), (-AU, AV)]))
POL_MON = shapely.union_all(list(POL.values()) + [POL_TORRE])


def chao_min(pol, folga=1.5):
    b = pol.buffer(folga)
    xs, ys = np.array(b.exterior.coords).T
    return float(np.min(cota(xs, ys)))


def prisma_pol(m, pol, z0, z1, mat, mat_topo):
    pol = shapely.geometry.polygon.orient(pol, 1)
    m.prisma(list(np.array(pol.exterior.coords)[:-1]), z0, z1, mat, mat_topo=mat_topo)


def cerca():
    """A cerca como prismas das plantas do LiDAR até ao adarve, e os parapeitos e merlões por cima:
    - torreões da porta e muro da porta: topo a 387,58 m, merlões nos três lados de fora (fotografias da porta);
    - muralha poente: adarve a 386,95 m, merlões pelo lado de fora (fotografias do adarve e de norte);
    - muralha norte e baluarte NE: adarve a 386,35 m e parapeito liso de 0,9 m pelo lado de fora (LiDAR: 387,3);
    - muralha nascente: adarve a 386,0 m e rebordo baixo (LiDAR: 386,4)."""
    m = Malha(["muro", "muro_topo", "cantaria_lisa", "sombra"])
    amei_torreoes = {"larg": 0.70, "passo": 1.30, "z1": 389.25, "zp": 389.95, "seteiras": 3}
    amei_poente = {"larg": 0.70, "passo": 1.30, "z1": 388.65, "zp": 389.30, "seteiras": 3}
    zb = {}
    for k in POL:
        zb[k] = chao_min(POL[k]) - 0.8
    # ---- torreões e muro da porta
    for k in ("torreao_sul", "torreao_norte"):
        prisma_pol(m, POL[k], zb[k], C.Z_TORREOES, 0, 1)
    # o muro da porta: as faces de fora e de dentro (com o vão) são da porta(); aqui só o alicerce e o topo
    prisma_pol(m, POL["muro_porta"], zb["muro_porta"], Z_SOL_PORTA - 1.2, 0, 0)
    mp = shapely.geometry.polygon.orient(POL["muro_porta"], 1)
    m.face([(x_, y_, C.Z_TORREOES) for x_, y_ in np.array(mp.exterior.coords)[:-1]], 1)
    n_m = 0
    # torreão sul: merlões a poente, a sul e a norte (para a passagem)
    ts = C.TORREAO_SUL
    n_m += parapeito(m, WL([(-5.22, 2.55), (-11.75, 2.55), (-11.75, 6.1), (-7.3, 6.1)]), POL["torreao_sul"], C.Z_TORREOES, 388.55, 0.62, 0, 1, amei_torreoes)
    n_m += parapeito(m, WL([(-7.3, 8.3), (-11.35, 8.45), (-11.35, 11.2), (-10.6, 12.05), (-6.85, 12.0)]), POL["torreao_norte"], C.Z_TORREOES, 388.55, 0.62, 0, 1, amei_torreoes)
    # muro da porta: um merlão a meio, por cima do arco (fotografias de frente), e o parapeito
    n_m += parapeito(m, WL([(-7.3, 6.1), (-7.3, 8.3)]), POL["muro_porta"], C.Z_TORREOES, 388.55, 0.62, 0, 1, None)
    fpm = Plano(W(-7.3, 7.2), EV, n=-EU)
    merlao(m, [fpm.xy(-0.4, 0), fpm.xy(0.4, 0), fpm.xy(0.4, -0.62), fpm.xy(-0.4, -0.62)], 388.55, 389.3, 389.95, 0)
    m.face(orientar([fpm.p(s_, 0.012, z_) for s_, z_ in redondo(0.0, 0.18, 388.65, 0.25, k=6)], np.array([-EU[0], -EU[1], 0])), 3)
    n_m += 1
    # ---- cerca (muralhas poente, norte, nascente e baluarte), em troços com o adarve de cada um
    cer = POL["cerca"]
    troco = {
        "poente": (shapely.Polygon(WL([(-8, 11.9), (-3.0, 11.9), (-3.0, 20.6), (-8, 20.6)])), C.Z_POENTE),
        "norte": (shapely.Polygon(WL([(-8, 20.6), (-3.0, 20.6), (-3.0, 18.5), (11.0, 18.5), (11.0, 24.4), (17, 18.0), (17, 31), (-8, 31)])), C.Z_NORTE),
        "nascente": (shapely.Polygon(WL([(8.0, 18.5), (11.0, 18.5), (11.0, 24.4), (17, 18.0), (17, 2.0), (8.0, 2.0)])), C.Z_NASCENTE),
    }
    for nome, (corte, z) in troco.items():
        g = cer.intersection(corte)
        for p in (g.geoms if hasattr(g, "geoms") else [g]):
            if p.geom_type == "Polygon" and p.area > 0.3:
                prisma_pol(m, p, zb["cerca"], z, 0, 1)
    # degraus entre os adarves: poente (386,95) -> norte (386,35) no canto NO
    # parapeitos
    n_m += parapeito(m, WL([(-6.85, 12.0), (-6.85, 15.3), (-6.45, 17.0), (-6.5, 18.4), (-6.2, 20.1)]), cer, C.Z_POENTE, 388.1, 0.65, 0, 1, amei_poente)
    parapeito(m, WL([(-6.2, 20.1), (-4.5, 22.1), (-3.7, 22.2), (-2.3, 23.3), (0.3, 24.0), (3.5, 25.1), (6.7, 26.3), (9.0, 26.9), (10.5, 27.8), (12.3, 28.3)]),
              cer, C.Z_NORTE, 387.25, 0.55, 0, 1, None)
    parapeito(m, WL([(12.3, 28.3), (14.3, 24.7), (15.9, 22.7), (14.7, 21.1), (13.7, 20.9), (11.7, 18.9)]), cer, C.Z_NORTE, 386.75, 0.5, 0, 1, None)
    parapeito(m, WL([(11.15, 18.9), (11.4, 15.2), (11.8, 13.0), (12.0, 10.7), (12.7, 8.9), (12.75, 6.1), (13.1, 5.3), (13.4, 4.2)]), cer, C.Z_NASCENTE, 386.4, 0.45, 0, 1, None)
    # degraus do torreão norte (387,58) para o adarve poente (386,95) (fotografia do adarve)
    for i in range(4):
        v0 = 12.0 + i * 0.3
        m.prisma(ccw(WL([(-6.2, v0), (-4.9, v0), (-4.9, v0 + 0.3), (-6.2, v0 + 0.3)])), C.Z_POENTE - 0.3, C.Z_TORREOES - (i + 1) * 0.63 / 4.2, 1, mat_topo=1)
    # muro fino a sudeste, do cunhal NE da torre ao cunhal SE da cerca, com o capeamento
    prisma_pol(m, POL["muro_se"], zb["muro_se"], C.Z_MURO_SE, 0, 2)
    ob = m.objeto("cerca", "cerca", GRUPOS["Muralhas"], uv=3.0, vcor=patine_muro)
    MEDIDAS["cerca"] = {"area_m2": round(POL_MON.area - POL_TORRE.area, 1), "merloes": n_m,
                        "adarves": {"torreoes": C.Z_TORREOES, "poente": C.Z_POENTE, "norte": C.Z_NORTE, "nascente": C.Z_NASCENTE}}
    return ob


def terraco():
    """Terraço de 384,1 m por dentro da muralha norte e banco ao longo da nascente; plataforma de lajes ao pé da
    face OSO da torre (384,0) e os alicerces a sul da torre (383,3), todos do LiDAR; e as escadas do pátio."""
    m = Malha(["muro", "muro_topo", "relva", "muro_topo", "ferro"])
    p = shapely.Polygon(WL(C.TERRACO)).buffer(0).difference(POL["cerca"].buffer(-0.02))
    for g in (p.geoms if hasattr(p, "geoms") else [p]):
        if g.area > 0.5:
            prisma_pol(m, g, chao_min(g, 0.5) - 0.6, C.Z_TERRACO, 0, 2)
    oso = shapely.Polygon(WL(C.TERRACO_OSO))
    prisma_pol(m, oso, chao_min(oso, 0.8) - 0.6, 384.0, 0, 1)
    al = shapely.Polygon(WL(C.ALICERCES))
    prisma_pol(m, al, chao_min(al, 0.8) - 0.4, 383.3, 0, 1)
    # muretes dos alicerces (fotografias de nascente: compartimentos de pedra a sul da torre)
    for a, b in (((-1.2, -7.6), (6.6, -7.6)), ((2.6, -9.3), (2.6, -5.9))):
        pa, pb = np.array(W(*a)), np.array(W(*b))
        t = unit(pb - pa)
        nn = np.array([-t[1], t[0]]) * 0.25
        m.prisma(ccw([pa - nn, pb - nn, pb + nn, pa + nn]), 383.0, 383.75, 0, mat_topo=1)
    # escadas (posições estimadas onde o LiDAR mostra os degraus ou os patamares)
    # pátio -> terraço norte (degraus com corrimão nas fotografias de norte)
    degraus(m, W(3.0, 16.9), W(3.0, 19.45), 1.1, 382.3, C.Z_TERRACO, 10, 3)
    # terraço -> adarve norte, encostada à face de dentro da muralha (LiDAR: degraus a 385,6–386,0 em u -2)
    degraus(m, W(0.6, 19.15), W(-3.0, 19.15), 0.85, C.Z_TERRACO, C.Z_NORTE, 13, 3)
    # pátio -> banco nascente -> adarve nascente (LiDAR: rampa de degraus a v 6 entre 384,1 e 386,0)
    degraus(m, W(7.1, 4.95), W(8.6, 4.95), 0.8, 382.85, C.Z_TERRACO, 7, 3)
    degraus(m, W(9.1, 5.55), W(11.25, 5.55), 0.8, C.Z_TERRACO, C.Z_NASCENTE, 10, 3)
    # pátio -> terraço, ao longo da muralha poente (LiDAR: patamar de 383,8 m em u -4)
    degraus(m, W(-3.8, 12.6), W(-3.8, 17.4), 0.9, 382.95, C.Z_TERRACO, 7, 3)
    corrimao(m, [(*W(3.6, 16.9), 382.4), (*W(3.6, 19.45), C.Z_TERRACO)], alt=0.9, mat=4)
    m.objeto("terracos", "cerca", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)


# ---------------------------------------------------------------- porta do castelo

def porta():
    """A porta, entre os torreões, no muro recuado (u -7,3 a -5,1): arco quebrado de 1,55 m de vão com aduelas, a
    porta de duas folhas de madeira verde abertas para dentro, e por dentro um arco mais largo (fotografias de fora
    e de dentro). Soleira a 383,1 m (LiDAR)."""
    m = Malha(["muro", "cantaria_lisa", "intradorso", "madeira_verde", "ferro", "sombra", "muro_topo"])
    z_sol = Z_SOL_PORTA
    fo = Plano(W(-7.3, 7.2), EV, n=-EU)           # face de fora do muro da porta (s = v - 7,2)
    fi = Plano(W(-5.1, 7.2), -EV, n=EU)           # face de dentro (s = -(v - 7,2))
    w_o, h_o = 1.55, 1.75                          # vão de fora: largura e arranque do arco (fecho a c. 386,0)
    w_i, h_i = 2.15, 1.85                          # arco de dentro
    c_o = ogiva(0.0, w_o, z_sol, h_o, k=8, raio=0.85)
    c_i = ogiva(0.0, w_i, z_sol, h_i, k=8, raio=0.8)
    meia = 1.1
    zt = C.Z_TORREOES
    painel(m, fo, [(-meia, z_sol - 1.2), (meia, z_sol - 1.2), (meia, zt), (-meia, zt)], [c_o], 0)
    painel(m, fi, [(-meia, z_sol - 1.2), (meia, z_sol - 1.2), (meia, zt), (-meia, zt)], [c_i], 0)
    # o vão de fora tem 0,7 m de fundo; a seguir abre para o arco de dentro (derrame)
    prof = 2.2
    tunel(m, fo, c_o, 0.0, -0.7, 2)
    # degrau entre os dois contornos, a 0,7 m: face de muro com o furo do vão de fora
    c_i_espelho = [(-s, z) for s, z in c_i]
    c_i_fo = [(s, z) for s, z in c_i]  # o mesmo contorno visto da face de fora (simétrico)
    painel(m, Plano(fo.xy(0, -0.7), -fo.t, n=fo.n * -1.0), [(s, z) for s, z in c_i_espelho], [[(-s, z) for s, z in c_o]], 0)
    tunel(m, fo, c_i_fo, -0.7, -prof, 2)
    # chão da passagem (lajes)
    m.face(orientar([(*fo.xy(-w_i / 2, -0.7), z_sol), (*fo.xy(w_i / 2, -0.7), z_sol), (*fo.xy(w_i / 2, -prof), z_sol), (*fo.xy(-w_i / 2, -prof), z_sol)], np.array([0, 0, 1.0])), 6)
    m.face(orientar([(*fo.xy(-w_o / 2, 0), z_sol), (*fo.xy(w_o / 2, 0), z_sol), (*fo.xy(w_o / 2, -0.7), z_sol), (*fo.xy(-w_o / 2, -0.7), z_sol)], np.array([0, 0, 1.0])), 6)
    # aduelas do arco de fora
    arco_o = [p for p in c_o if p[1] >= z_sol + h_o - 1e-6]
    anel_aduelas(m, fo, arco_o, 0.36, 0.05, 1)
    for s0_, s1_ in ((-w_o / 2 - 0.32, -w_o / 2), (w_o / 2, w_o / 2 + 0.32)):
        for j in range(4):
            za_, zb_ = z_sol + h_o * j / 4 + 0.01, z_sol + h_o * (j + 1) / 4 - 0.01
            m.caixa_face(fo, s0_, s1_, 0.0, 0.05, za_, zb_, 1)
    # impostas
    for s_ in (-w_o / 2 - 0.36, w_o / 2):
        m.caixa_face(fo, s_, s_ + 0.36, 0.0, 0.09, z_sol + h_o - 0.14, z_sol + h_o, 1)
    # portas de duas folhas de madeira verde, abertas para dentro e encostadas às paredes da passagem (fotografia de
    # frente fa66: vê-se a folha norte aberta e o céu pelo vão): perfil de cada folha = meio arco quebrado; ferragens
    arco_dir = [(s_, z_) for s_, z_ in c_o if s_ > -1e-6 and z_ >= z_sol + h_o - 1e-6]
    arco_dir.sort(key=lambda q: -q[0])
    ss_, zz_ = np.array([q[0] for q in arco_dir])[::-1], np.array([q[1] for q in arco_dir])[::-1]
    dp, folga, esp = -0.30, 0.05, 0.06
    lw = w_o / 2 - folga - 0.02
    for lado in (-1, 1):
        dobr = fo.xy(lado * (w_o / 2 - folga), dp)
        nc = -lado * fo.t                     # para o meio da passagem
        fl = Plano(dobr, -fo.n, n=nc)         # s' = da dobradiça para dentro do castelo
        sp = np.linspace(0, lw, 9)
        topo = np.interp(w_o / 2 - folga - sp, ss_, zz_) - 0.03
        cont = [(0.0, z_sol + 0.02)] + [(float(sp[-1]), z_sol + 0.02)] + [(float(a_), float(b_)) for a_, b_ in zip(sp[::-1], topo[::-1])]
        painel(m, fl, cont, (), 3)
        fl_b = Plano(fl.xy(0, -esp), -fo.n, n=-nc)
        painel(m, fl_b, cont, (), 3)
        tunel(m, fl, cont, 0.0, -esp, 3)
        # ferragens: duas barras, a fechadura junto ao bordo livre, a junta das tábuas
        for z_ in (z_sol + 0.45, z_sol + 1.45):
            m.caixa_face(fl, 0.03, lw - 0.03, 0.0, 0.022, z_, z_ + 0.07, 4)
        for sv in (0.2, 0.4, 0.6):
            if sv < lw - 0.05:
                m.caixa_face(fl, sv - 0.01, sv + 0.01, 0.0, 0.01, z_sol + 0.05, float(np.interp(sv, sp, topo)) - 0.06, 5)
        if lado > 0:
            m.caixa_face(fl, lw - 0.2, lw - 0.04, 0.0, 0.035, z_sol + 1.0, z_sol + 1.3, 4)
        # gonzos na parede
        for z_ in (z_sol + 0.4, z_sol + 1.5):
            m.caixa_face(fl, -0.02, 0.06, -esp - folga + 0.005, 0.0, z_, z_ + 0.08, 4)
    m.objeto("porta", "porta", GRUPOS["Muralhas"], uv=2.5, vcor=patine_muro)
    MEDIDAS["porta"] = {"soleira": z_sol, "vao_fora": [w_o, round(c_o[len(c_o) // 2 + 4][1] - z_sol, 2)], "vao_dentro": w_i}
    return fo, z_sol


def escadaria_porta(z_sol):
    """Escadaria de granito da encosta para a porta, entre os torreões (LiDAR: de 380,55 m em u -15,5 a 383,1 m na
    soleira; fotografias: lanço de baixo mais largo, patamar e lanço de cima), e o penedo arredondado encostado ao
    torreão sul (fotografias; LiDAR: o alto do penedo a 385,9 m)."""
    m = Malha(["muro_topo", "rocha"])
    vc = 7.35
    degraus(m, W(-15.6, vc), W(-12.4, vc), 2.4, 380.55, 381.65, 5, 0, fundo=379.6)
    m.prisma(ccw(WL([(-12.4, vc - 1.2), (-11.0, vc - 1.2), (-11.0, vc + 1.2), (-12.4, vc + 1.2)])), 379.8, 381.65, 0, mat_topo=0)
    degraus(m, W(-11.0, vc), W(-7.3, vc), 1.7, 381.65, z_sol, 8, 0, fundo=380.8)
    m.objeto("escadaria_porta", "porta", GRUPOS["Muralhas"], uv=1.5, vcor=patine_muro)
    # penedo: elipsoide deformado
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    c = np.array([*W(-11.5, 4.3), 382.7])   # penedo baixo e largo à frente do torreão sul, junto à escada (fotografia)
    for vt in bm.verts:
        p = np.array(vt.co[:])
        r = 1.0 + 0.10 * math.sin(3 * p[0] + 1.3) * math.cos(2 * p[1]) + 0.06 * math.sin(5 * p[2] + p[0]) + 0.04 * math.cos(7 * p[1] - 2 * p[2])
        q = p * r * np.array([1.6, 2.15, 1.55])
        if q[2] > 0.95:   # alto achatado (fotografia)
            q[2] = 0.95 + 0.4 * (q[2] - 0.95)
        # rodar para o eixo da muralha
        ang = math.radians(90 - C.ROT_TORRE)
        x_ = q[0] * math.cos(ang) - q[1] * math.sin(ang)
        y_ = q[0] * math.sin(ang) + q[1] * math.cos(ang)
        vt.co = (c[0] + x_, c[1] + y_, c[2] + max(q[2], -1.4))
    me = bpy.data.meshes.new("penedo")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(MATS["penedo"])
    me.shade_smooth()
    ob = bpy.data.objects.new("penedo", me)
    bpy.context.scene.collection.objects.link(ob)
    ob["elemento"] = "porta"
    ob.parent = GRUPOS["Muralhas"]
    uv_mundo(ob, 2.0)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    g = fbm(co[:, 0] * 2 + 50, co[:, 1] * 2 + co[:, 2], 1.5, 9)
    g2 = fbm(co[:, 0] * 6 + 9, co[:, 1] * 6 - co[:, 2] * 3, 1.0, 21)
    cv = np.array([0.34, 0.33, 0.305])[None, :] * (0.78 + 0.45 * g)[:, None]             # granito cinzento (× textura)
    cv = np.where((g2 > 0.62)[:, None], cv * 1.35, cv)                                    # manchas de líquen claro
    musgo = np.clip(np.clip((co[:, 2] - (c[2] - 0.6)) / 1.2, 0, 1) * 0.7 + (g2 - 0.45) * 2.0, 0, 0.9)   # manchas de musgo, mais por cima (fotografias)
    cv = cv * (1 - musgo)[:, None] + np.array([0.20, 0.25, 0.10])[None, :] * (0.8 + 0.4 * g2)[:, None] * musgo[:, None]
    at = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    at.data.foreach_set("color", np.concatenate([np.clip(cv, 0, 1), np.ones((len(cv), 1))], 1).astype(np.float32).ravel())


# ---------------------------------------------------------------- santuário

_ai = math.radians(C.IGREJA_AZ)
ES = np.array([math.sin(_ai), math.cos(_ai)])     # ao longo da nave, para norte (altar-mor)
EQ = np.array([math.cos(_ai), -math.sin(_ai)])    # para ENE
IC = np.array(C.IGREJA_C)


def S(s, q):
    return IC + ES * s + EQ * q


def SL(pts):
    return [tuple(S(*p)) for p in pts]


def telhado_2aguas(m, s0, s1, q_meia, z_beir, z_cume, mat, beir=0.35, empena=(True, True)):
    """Duas águas ao longo de s, com beiral; e os oitões (empenas) nas pontas."""
    for lado in (-1, 1):
        a = S(s0 - 0.15, 0.0)
        b = S(s1 + 0.15, 0.0)
        c_ = S(s1 + 0.15, lado * (q_meia + beir))
        d = S(s0 - 0.15, lado * (q_meia + beir))
        zb = z_beir - beir * (z_cume - z_beir) / q_meia
        m.face(orientar([(*a, z_cume), (*b, z_cume), (*c_, zb), (*d, zb)], np.array([0, 0, 1.0])), mat)
        m.face(orientar([(*a, z_cume - 0.12), (*b, z_cume - 0.12), (*c_, zb - 0.12), (*d, zb - 0.12)], np.array([0, 0, -1.0])), mat)
        m.face([(*d, zb - 0.12), (*c_, zb - 0.12), (*c_, zb), (*d, zb)], mat)
    m.barra((*S(s0 - 0.15, 0), z_cume + 0.04), (*S(s1 + 0.15, 0), z_cume + 0.04), 0.09, mat, k=5)


def ameias_cornija(m, s0, s1, q, z, passo=0.85, mat=0):
    """Cornija ameada (fotografias de nascente): pequenos merlões piramidais sobre a cornija."""
    n = max(2, int(round((s1 - s0) / passo)))
    for s in np.linspace(s0 + 0.2, s1 - 0.2, n + 1):
        base = SL([(s - 0.16, q - 0.16), (s + 0.16, q - 0.16), (s + 0.16, q + 0.16), (s - 0.16, q + 0.16)])
        m.prisma(ccw(base), z, z + 0.28, mat, topo=False)
        m.piramide([np.array(p) for p in ccw(base)], z + 0.28, z + 0.62, mat)


def santuario():
    """O Santuário de Nossa Senhora do Pilar (1680), com as medidas do LiDAR: nave de 19,2 × 10,8 m (paredes), cume
    a 390,5 m e beirais a c. 386,5 m; capela-mor a norte (7 × 9,7 m, cume a 386,6 m) com telhado de três águas;
    sacristia a poente da capela-mor (água única de 383,7 a 381,6 m); adro a sul com escadas. A fachada sul segue a
    fotografia de frente: cunhais em pilastra com pináculos, frontão triangular com a cruz, nicho com a imagem
    entre as duas janelas de cima, porta ao meio e duas janelas de baixo."""
    m = Malha(["cantaria_igreja", "cantaria_igreja", "telha", "sombra", "vidro", "madeira_igreja", "imagem", "ferro"])
    z_chao = 377.95                      # adro e soleira (LiDAR)
    z_fund = 376.0
    s_f, s_n = -9.6, 9.6                 # fachada e parede norte da nave
    qm = 5.4                             # meia largura da nave (paredes)
    z_corn = 386.2
    # ---- fachada (sul), medida na fotografia de frente (c. 37 px por metro na fachada de 10,8 m)
    ff = Plano(S(s_f, 0), EQ, n=-ES)     # fachada: s = q, d para fora (sul)
    JAN = [(-2.7, 378.75, 380.4), (2.7, 378.75, 380.4), (-2.65, 382.3, 384.0), (2.65, 382.3, 384.0)]
    furos = [reto(0.0, 1.75, z_chao, 380.85)] + [reto(q_, 1.15, a_, b_) for q_, a_, b_ in JAN]
    painel(m, ff, [(-qm, z_fund), (qm, z_fund), (qm, z_corn), (-qm, z_corn)], furos, 0)
    # porta: folhas de madeira com almofadas, ombreiras em pilastra, entablamento e frontão interrompido com a
    # cartela, de onde desce o pendente do nicho
    tunel(m, ff, furos[0], 0.0, -0.45, 1)
    painel(m, ff, furos[0], (), 5, d=-0.45)
    for lado in (-1, 1):
        for j in range(3):
            za_ = z_chao + 0.25 + j * 0.86
            m.caixa_face(ff, *sorted((lado * 0.12, lado * 0.75)), -0.45, -0.41, za_, za_ + 0.66, 5)
    m.caixa_face(ff, -0.03, 0.03, -0.45, -0.40, z_chao, 380.85, 7)
    for lado in (-1, 1):
        m.caixa_face(ff, *sorted((lado * 0.875, lado * 1.2)), 0.0, 0.13, z_chao, 380.85, 1)
        m.caixa_face(ff, *sorted((lado * 0.84, lado * 1.25)), 0.0, 0.17, z_chao, z_chao + 0.35, 1)
    m.caixa_face(ff, -1.38, 1.38, 0.0, 0.2, 380.85, 381.2, 1)
    for lado in (-1, 1):
        m.barra((*ff.xy(lado * 1.42, 0.12), 381.25), (*ff.xy(lado * 0.48, 0.12), 381.95), 0.11, 1, k=4)
    m.caixa_face(ff, -0.42, 0.42, 0.0, 0.2, 381.2, 382.15, 1)          # cartela
    m.caixa_face(ff, -0.3, 0.3, 0.0, 0.24, 382.15, 382.6, 1)
    m.caixa_face(ff, -0.18, 0.18, 0.0, 0.2, 382.6, 383.1, 1)
    # janelas: vão com grade de pedra (fotografia: quadrícula clara), moldura, peitoril e cornija
    for q_, za, zb in JAN:
        c = reto(q_, 1.15, za, zb)
        tunel(m, ff, c, 0.0, -0.3, 1)
        painel(m, ff, c, (), 4, d=-0.3)
        sa, sb = q_ - 0.575, q_ + 0.575
        for k in range(1, 4):
            s_ = sa + 1.15 * k / 4
            m.caixa_face(ff, s_ - 0.025, s_ + 0.025, -0.14, -0.09, za, zb, 1, topo=False)
        for k in range(1, 5):
            z_ = za + (zb - za) * k / 5
            m.caixa_face(ff, sa, sb, -0.14, -0.09, z_ - 0.025, z_ + 0.025, 1)
        m.caixa_face(ff, sa - 0.17, sa, 0.0, 0.08, za, zb, 1)
        m.caixa_face(ff, sb, sb + 0.17, 0.0, 0.08, za, zb, 1)
        m.caixa_face(ff, sa - 0.17, sb + 0.17, 0.0, 0.08, zb, zb + 0.17, 1)
        m.caixa_face(ff, sa - 0.3, sb + 0.3, 0.0, 0.17, zb + 0.17, zb + 0.34, 1)   # cornija
        m.caixa_face(ff, sa - 0.22, sb + 0.22, 0.0, 0.14, za - 0.16, za, 1)        # peitoril
    # retábulo do nicho, à frente da fachada e a entrar no frontão: pendente, peanha, pilastras com volutas,
    # fundo em arco, a imagem de Nossa Senhora e o remate com frontão
    for (w_, z0_, z1_, d_) in ((0.36, 383.1, 383.7, 0.18), (0.62, 383.7, 384.2, 0.2), (1.2, 384.2, 384.6, 0.3)):
        m.caixa_face(ff, -w_ / 2, w_ / 2, 0.0, d_, z0_, z1_, 1)
    painel(m, ff, redondo(0.0, 0.95, 384.6, 1.55, k=12), (), 3, d=0.03)
    for lado in (-1, 1):
        m.caixa_face(ff, *sorted((lado * 0.5, lado * 0.78)), 0.0, 0.24, 384.6, 386.95, 1)
        m.caixa_face(ff, *sorted((lado * 0.78, lado * 1.4)), 0.0, 0.14, 384.7, 385.25, 1)
        m.caixa_face(ff, *sorted((lado * 0.78, lado * 1.15)), 0.0, 0.14, 385.25, 385.95, 1)
        m.cilindro(ff.xy(lado * 1.32, 0.12), 0.16, 385.05, 385.35, 1, k=8)
    m.caixa_face(ff, -0.5, 0.5, 0.0, 0.08, 386.55, 386.95, 1)
    m.caixa_face(ff, -0.92, 0.92, 0.0, 0.3, 386.95, 387.2, 1)
    for lado in (-1, 1):
        m.barra((*ff.xy(lado * 0.95, 0.15), 387.22), (*ff.xy(0.0, 0.15), 387.75), 0.08, 1, k=4)
    m.cilindro(ff.xy(0.0, 0.15), 0.1, 387.7, 388.05, 1, k=8, r1=0.03)
    pi = ff.xy(0.0, 0.22)
    m.cilindro(pi, 0.22, 384.62, 385.75, 6, k=10, r1=0.13)    # a imagem (manto)
    m.cilindro(pi, 0.1, 385.75, 386.0, 6, k=8)
    m.cilindro(pi, 0.12, 386.0, 386.1, 6, k=8, r1=0.05)      # coroa
    # pilastras dos cunhais com capitel e embasamento
    for sgn in (-1, 1):
        q0_, q1_ = sorted((sgn * (qm - 0.92), sgn * (qm + 0.02)))
        m.caixa_face(ff, q0_, q1_, 0.0, 0.18, z_fund, z_corn - 0.35, 1)
        m.caixa_face(ff, q0_ - 0.06, q1_ + 0.06, 0.0, 0.26, z_corn - 0.35, z_corn - 0.15, 1)
        m.caixa_face(ff, q0_ - 0.06, q1_ + 0.06, 0.0, 0.26, z_fund, z_chao + 0.4, 1)
    m.caixa_face(ff, -qm, qm, 0.0, 0.1, z_fund, z_chao + 0.3, 1)
    # paredes laterais e de trás da nave
    fe = Plano(S(0, qm), ES, n=EQ)       # nascente (s = s da nave)
    fw = Plano(S(0, -qm), -ES, n=-EQ)    # poente (s = -s)
    fn = Plano(S(s_n, 0), -EQ, n=ES)     # norte (s = -q)
    furos_e = [reto(-4.2, 1.3, 378.1, 380.9), reto(-6.0, 0.95, 383.0, 384.6), reto(1.5, 0.95, 383.0, 384.6), reto(6.0, 0.95, 383.0, 384.6)]
    furos_w = [reto(-(-6.0), 0.95, 383.0, 384.6), reto(-1.5, 0.95, 383.0, 384.6), reto(-6.0, 0.95, 383.0, 384.6)]
    for f, fur in ((fe, furos_e), (fw, furos_w)):
        painel(m, f, [(s_f, z_fund), (s_n, z_fund), (s_n, z_corn), (s_f, z_corn)] if f is fe else [(-s_n, z_fund), (-s_f, z_fund), (-s_f, z_corn), (-s_n, z_corn)], fur, 0)
        for j, c in enumerate(fur):
            tunel(m, f, c, 0.0, -0.35, 1)
            painel(m, f, c, (), 5 if (f is fe and j == 0) else 4, d=-0.35)
            sa, sb = min(p[0] for p in c) - 0.15, max(p[0] for p in c) + 0.15
            za, zb = min(p[1] for p in c), max(p[1] for p in c)
            m.caixa_face(f, sa, sb, 0.0, 0.05, zb, zb + 0.15, 1)
            m.caixa_face(f, sa, sa + 0.15, 0.0, 0.05, za, zb, 1)
            m.caixa_face(f, sb - 0.15, sb, 0.0, 0.05, za, zb, 1)
    painel(m, fn, [(-qm, z_fund), (qm, z_fund), (qm, z_corn), (-qm, z_corn)], [], 0)
    # cornija corrida e pilastras dos cunhais das paredes laterais
    for sgn in (-1, 1):
        q0, q1 = (qm, qm + 0.3) if sgn > 0 else (-qm - 0.3, -qm)
        m.prisma(ccw(SL([(s_f - 0.3, q0), (s_n + 0.3, q0), (s_n + 0.3, q1), (s_f - 0.3, q1)])), z_corn - 0.25, z_corn + 0.1, 1, mat_topo=1)
        ameias_cornija(m, s_f, s_n, sgn * (qm + 0.12), z_corn + 0.1, mat=1)
        for s_ in (s_f, s_n - 0.7):
            m.prisma(ccw(SL([(s_, sgn * qm - 0.01), (s_ + 0.7, sgn * qm - 0.01), (s_ + 0.7, sgn * (qm + 0.14)), (s_, sgn * (qm + 0.14))])), z_fund, z_corn - 0.25, 1, mat_topo=1)
    for q0_, q1_ in ((-qm - 0.3, -1.42), (1.42, qm + 0.3)):   # cornija da fachada, interrompida pelo retábulo do nicho
        m.prisma(ccw(SL([(s_f - 0.3, q0_), (s_f + 0.01, q0_), (s_f + 0.01, q1_), (s_f - 0.3, q1_)])), z_corn - 0.15, z_corn + 0.12, 1, mat_topo=1)
    # ---- telhado da nave
    z_cume = 390.5
    telhado_2aguas(m, s_f + 0.05, s_n, qm + 0.3, z_corn + 0.12, z_cume, 2, beir=0.15)
    # oitão norte (parede acima da cornija, até ao telhado)
    m.face(orientar([(*S(s_n, -qm), z_corn), (*S(s_n, qm), z_corn), (*S(s_n, 0), z_cume - 0.1)], np.array([*ES, 0])), 0)
    # frontão da fachada (fotografia: vértice ao nível do cume, c. 390,6 m): tímpano, cornijas inclinadas,
    # pináculos sobre os capitéis dos cunhais e a cruz num pedestal
    z_ap = 390.7
    m.face(orientar([(*S(s_f, -qm), z_corn + 0.1), (*S(s_f, qm), z_corn + 0.1), (*S(s_f, 0), z_ap)], np.array([*-ES, 0])), 0)
    m.face(orientar([(*S(s_f + 0.6, -qm), z_corn + 0.1), (*S(s_f + 0.6, qm), z_corn + 0.1), (*S(s_f + 0.6, 0), z_ap)], np.array([*ES, 0])), 0)
    for sgn in (-1, 1):
        a, b = np.array([*S(s_f - 0.14, sgn * (qm + 0.25)), z_corn + 0.12]), np.array([*S(s_f - 0.14, 0), z_ap + 0.14])
        m.barra(a, b, 0.2, 1, k=4)
        p = S(s_f - 0.08, sgn * (qm - 0.45))
        m.prisma(ccw(SL([(s_f - 0.38, sgn * (qm - 0.78)), (s_f + 0.3, sgn * (qm - 0.78)), (s_f + 0.3, sgn * (qm - 0.12)), (s_f - 0.38, sgn * (qm - 0.12))])), z_corn + 0.1, z_corn + 0.95, 1, mat_topo=1)
        m.prisma(ccw(SL([(s_f - 0.45, sgn * (qm - 0.85)), (s_f + 0.37, sgn * (qm - 0.85)), (s_f + 0.37, sgn * (qm - 0.05)), (s_f - 0.45, sgn * (qm - 0.05))])), z_corn + 0.95, z_corn + 1.07, 1, mat_topo=1)
        m.cilindro(p, 0.2, z_corn + 1.07, z_corn + 1.3, 1, k=10, r1=0.3)
        m.cilindro(p, 0.3, z_corn + 1.3, z_corn + 1.7, 1, k=10, r1=0.14)
        m.cilindro(p, 0.14, z_corn + 1.7, z_corn + 2.6, 1, k=8, r1=0.0)
    m.prisma(ccw(SL([(s_f - 0.2, -0.3), (s_f + 0.35, -0.3), (s_f + 0.35, 0.3), (s_f - 0.2, 0.3)])), z_ap - 0.2, z_ap + 0.4, 1, mat_topo=1)
    m.prisma(ccw(SL([(s_f - 0.02, -0.07), (s_f + 0.12, -0.07), (s_f + 0.12, 0.07), (s_f - 0.02, 0.07)])), z_ap + 0.4, 392.9, 1, mat_topo=1)
    m.prisma(ccw(SL([(s_f - 0.02, -0.4), (s_f + 0.12, -0.4), (s_f + 0.12, 0.4), (s_f - 0.02, 0.4)])), 392.1, 392.25, 1, mat_topo=1)
    # ---- capela-mor
    s_c0, s_c1, qc = s_n, 19.3, 3.5
    z_cc, z_ccume = 384.45, 386.6
    m.prisma(ccw(SL([(s_c0 - 0.2, -qc), (s_c1, -qc), (s_c1, qc), (s_c0 - 0.2, qc)])), z_fund, z_cc, 0, mat_topo=0)
    fc = Plano(S(14.5, qc), ES, n=EQ)
    for s_ in (12.2, 16.6):              # s da nave; no plano fc conta-se a partir de s = 14,5 (antes ficavam 14,5 m a norte, no terreiro)
        sf_ = s_ - 14.5
        c = reto(sf_, 0.85, 381.6, 383.0)
        painel(m, fc, c, (), 4, d=0.02)
        m.caixa_face(fc, sf_ - 0.58, sf_ + 0.58, 0.0, 0.06, 383.0, 383.15, 1)
    m.prisma(ccw(SL([(s_c0, -qc - 0.25), (s_c1 + 0.25, -qc - 0.25), (s_c1 + 0.25, qc + 0.25), (s_c0, qc + 0.25)])), z_cc - 0.2, z_cc + 0.08, 1, mat_topo=1)
    for sgn in (-1, 1):
        ameias_cornija(m, s_c0 + 0.3, s_c1, sgn * (qc + 0.1), z_cc + 0.08, mat=1)
    # telhado: duas águas com a ponta norte de três águas (fotografia de nascente)
    hw = qc + 0.45
    zb = z_cc + 0.05
    k = 2.2
    pa = S(s_c0, 0)
    pr = S(s_c1 + 0.45 - k, 0)
    cs = [S(s_c0, -hw), S(s_c1 + 0.45, -hw), S(s_c1 + 0.45, hw), S(s_c0, hw)]
    for f_ in ([(*cs[0], zb), (*cs[1], zb), (*pr, z_ccume), (*pa, z_ccume)], [(*cs[1], zb), (*cs[2], zb), (*pr, z_ccume)],
               [(*cs[2], zb), (*cs[3], zb), (*pa, z_ccume), (*pr, z_ccume)]):
        m.face(orientar(f_, np.array([0, 0, 1.0])), 2)
    m.face(orientar([(*cs[3], zb), (*cs[0], zb), (*pa, z_ccume)], np.array([*-ES, 0])), 0)
    m.barra((*pa, z_ccume + 0.04), (*pr, z_ccume + 0.04), 0.08, 2, k=5)
    # ---- sacristia (a poente da capela-mor, água única)
    s_s0, s_s1, q_s0, q_s1 = 11.6, 18.6, -8.6, -qc
    m.prisma(ccw(SL([(s_s0, q_s0), (s_s1, q_s0), (s_s1, q_s1), (s_s0, q_s1)])), z_fund, 381.3, 0, topo=False)
    for s_ in (s_s0, s_s1):
        m.face(orientar([(*S(s_, q_s0), 381.3), (*S(s_, q_s1), 381.3), (*S(s_, q_s1), 383.45)], np.array([*(ES if s_ > 15 else -ES), 0])), 0)
    m.face(orientar([(*S(s_s0 - 0.3, q_s0 - 0.35), 381.15), (*S(s_s1 + 0.3, q_s0 - 0.35), 381.15), (*S(s_s1 + 0.3, q_s1), 383.6), (*S(s_s0 - 0.3, q_s1), 383.6)], np.array([0, 0, 1.0])), 2)
    m.face(orientar([(*S(s_s0 - 0.3, q_s0 - 0.35), 381.03), (*S(s_s1 + 0.3, q_s0 - 0.35), 381.03), (*S(s_s1 + 0.3, q_s1), 383.48), (*S(s_s0 - 0.3, q_s1), 383.48)], np.array([0, 0, -1.0])), 2)
    fsw = Plano(S(15.0, q_s0), -ES, n=-EQ)
    for s_ in (13.3, 16.6):              # s da nave; no plano fsw (eixo -ES a partir de s = 15) é 15 - s (antes ficavam no terreiro)
        painel(m, fsw, reto(15.0 - s_, 0.7, 379.5, 380.6), (), 4, d=0.02)
    m.objeto("santuario", "santuario", GRUPOS["Santuario"], uv=3.0, vcor=patine_igreja)
    # ---- adro e escadas (LiDAR: adro a 377,95 m de s -9,6 a -13,4; degraus para sul até 377,2 m)
    ma = Malha(["muro_topo", "calcada"])
    # adro e escadaria larga, mais larga do que a fachada (fotografia de frente)
    adro = SL([(-12.9, -6.6), (s_f + 0.05, -6.6), (s_f + 0.05, 6.6), (-12.9, 6.6)])
    ma.prisma(ccw(adro), 376.0, z_chao, 0, mat_topo=1)
    degraus(ma, S(-14.6, 0.0), S(-12.9, 0.0), 14.6, 377.2, z_chao - 0.19, 4, 0, fundo=376.0)
    # degraus da porta lateral nascente (fotografia de nascente)
    pd = S(-4.2, qm)
    degraus(ma, S(-4.2, qm + 1.25), S(-4.2, qm + 0.02), 1.9, float(cota(*S(-4.2, qm + 1.5))), 378.1, 3, 0, fundo=376.4)
    ma.objeto("adro", "santuario", GRUPOS["Santuario"], uv=1.5, vcor=patine_muro)
    MEDIDAS["santuario"] = {"nave": [round(s_n - s_f, 2), 2 * qm], "cume_nave": z_cume, "cornija": z_corn, "frontao": z_ap,
                            "cruz": 393.0, "capela_mor": [round(s_c1 - s_c0, 2), 2 * qc], "cume_capela": z_ccume,
                            "adro": z_chao, "azimute_eixo": round(C.IGREJA_AZ % 360, 2)}


# ---------------------------------------------------------------- marcos do terreiro

def marcos():
    """Marcos de granito (pilaretes) no terreiro à frente do castelo (fotografias de nascente)."""
    m = Malha(["cantaria_lisa"])
    for p in ((-2.5, -12.5), (-0.5, -14.5), (1.8, -16.3), (4.2, -17.8), (-4.6, -10.6)):
        z = float(cota(*p))
        m.cilindro(p, 0.13, z - 0.2, z + 0.85, 0, k=8)
        m.cilindro(p, 0.13, z + 0.85, z + 1.0, 0, k=8, r1=0.04)
    m.objeto("marcos", "terreno", GRUPOS["Terreno"], uv=1.0)


# ---------------------------------------------------------------- terreno, envolvente e árvores

# o pátio, por dentro da cerca (u, v): entre a torre, a muralha poente, o terraço norte e o banco nascente
PATIO = WL([(-5.1, 5.15), (-5.1, 12.0), (-4.3, 13.0), (-4.3, 17.6), (-1.0, 18.4), (2.0, 19.3), (4.2, 20.0),
            (7.6, 19.9), (8.4, 18.2), (8.5, 5.2), (9.0, 4.6), (10.3, 3.9), (5.22, 3.9), (5.22, 5.15)])
# o santuário (nave, capela-mor, sacristia, adro e escadas), para tirar casas e árvores de cima dele
POL_IGREJA = shapely.Polygon(SL([(-15.0, -9.2), (19.8, -9.2), (19.8, 7.0), (-15.0, 7.0)]))


def construir_terreno():
    passo = 0.5
    xs = np.arange(X0, X1 + 1e-6, passo)
    ys = np.arange(Y0, Y1 + 1e-6, passo)
    X, Y = np.meshgrid(xs, ys)
    Z = cota(X, Y)
    pts_all = shapely.points(X.ravel(), Y.ravel())
    mur = shapely.contains(POL_MON.buffer(0.2), pts_all).reshape(X.shape)
    # junto às faces dos muros e da torre o MDT sobe para o topo (interpolação por baixo do muro): baixa-se o chão
    # até 1,5 m das faces para o mínimo das células à volta que não são muro
    banda = ndi.binary_dilation(mur, iterations=3) & ~mur
    zl = np.where(mur, 1e4, Z)
    zmin = ndi.minimum_filter(zl, size=5)
    Z = np.where(banda, np.minimum(Z, np.maximum(zmin, Z - 1.5)), Z)
    ny, nx = X.shape
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    idx = np.arange(ny * nx).reshape(ny, nx)
    Fq = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    mats = ["chao", "relva", "asfalto"]
    ob = malha_rapida("Terreno", V, Fq, mats[0], GRUPOS["Terreno"], "terreno")
    for mt in mats[1:]:
        ob.data.materials.append(MATS[mt])
    # ---- cor do chão do monte por vértice: erva verde, lajes de granito, terreiro e trilhos, com passagens
    # suaves (as classes saem do ortofotomapa e do MDT na grelha dos vértices e são alisadas c. 0,6 m)
    Xv, Yv = V[:, 0].reshape(ny, nx), V[:, 1].reshape(ny, nx)
    ci = np.clip(((Xv - OX0) / OP).astype(int), 0, ORTO.shape[1] - 1)
    ri = np.clip(((OY1 - Yv) / OP).astype(int), 0, ORTO.shape[0] - 1)
    o = ndi.uniform_filter(ORTO, size=(7, 7, 1))[ri, ci]
    o3 = ndi.uniform_filter(ORTO, size=(3, 3, 1))[ri, ci]
    lum = o.mean(2)
    sat = o.max(2) - o.min(2)
    verde = (o[..., 1] > o[..., 0] * 1.03) & (o[..., 1] > o[..., 2] * 1.06)
    rel = ndi.gaussian_filter(MDT - ndi.gaussian_filter(MDT, 3.0), 0.7)
    rl = amostra(rel, LX0, LY0, LP, Xv, Yv)
    gy, gx = np.gradient(ndi.gaussian_filter(MDT, 1.0), LP)
    dc = amostra(np.degrees(np.arctan(np.hypot(gx, gy))), LX0, LY0, LP, Xv, Yv)
    pts_v = shapely.points(Xv.ravel(), Yv.ravel())
    zona_t = shapely.Polygon([(-16, 2), (-6, -6), (-8, -26), (-6, -60), (40, -63), (38, 6), (12, 2)])
    em_zona = shapely.contains(zona_t, pts_v).reshape(ny, nx) & ~shapely.contains(POL_IGREJA.buffer(0.5), pts_v).reshape(ny, nx)

    def limpa(msk, area_min, fecho=2):
        msk = ndi.binary_closing(msk, iterations=fecho)
        msk = ndi.binary_opening(msk, iterations=1)
        lab, k = ndi.label(msk)
        if k:
            tam = ndi.sum(msk, lab, range(1, k + 1)) * passo * passo
            msk = np.isin(lab, 1 + np.flatnonzero(tam >= area_min))
        return ndi.binary_fill_holes(msk) if fecho > 1 else msk

    claro = (sat < 0.11) & (lum > 0.42) & ~verde
    terreiro = em_zona & (sat < 0.09) & (lum > 0.38) & (dc <= 14)
    # o acesso de serviço que sobe de sul até ao terreiro é calçado como ele (fotografias), não asfalto
    servico = [shapely.LineString(f["geometria"]["coordinates"]).buffer(2.4, cap_style="flat") for f in F["caminhos"]
               if f["geometria"]["type"] == "LineString" and f["classe"] == "service"]
    if servico:
        terreiro |= shapely.contains(shapely.union_all(servico).intersection(shapely.box(-20, -70, 31.5, 10)), pts_v).reshape(ny, nx)
    terreiro = limpa(terreiro, 30.0, 3)
    # lajes nuas também onde o ortofotomapa está na sombra (encosta nascente; a fotografia de drone de nascente
    # mostra-as): solo do LiDAR sem vegetação à volta e com mais de 12° de declive
    cls_v = CLS[np.clip(np.round((Yv - LY0) / LP).astype(int), 0, CLS.shape[0] - 1), np.clip(np.round((Xv - LX0) / LP).astype(int), 0, CLS.shape[1] - 1)]
    veg_v = ndi.maximum_filter(((cls_v >= 3) & (cls_v <= 5)).astype(np.uint8), size=3) > 0
    nua = (cls_v == 2) & ~veg_v & (dc > 12) & (lum < 0.33) & ~verde
    rocha = limpa(((claro & (dc > 8)) | (lum > 0.55) | (rl > 0.6) | nua) & ~terreiro, 6.0, 1)
    w_r = np.clip((ndi.gaussian_filter(rocha.astype(float), 1.2) - 0.25) / 0.5, 0, 1)
    w_t = np.clip((ndi.gaussian_filter(terreiro.astype(float), 1.0) - 0.25) / 0.5, 0, 1)
    w_v = ndi.gaussian_filter(verde.astype(float), 2.0)
    g = fbm(Xv * 0.9, Yv * 0.9, 6.0, 71)
    g2 = fbm(Xv * 3.0 + 7, Yv * 3.0, 2.0, 73)
    erva = np.array(COR_ERVA)[None, None, :] * (1 - w_v[..., None]) + np.array(COR_RELVA)[None, None, :] * w_v[..., None]
    erva = erva * (0.82 + 0.36 * g)[..., None]
    # granito: a luminância e o tom do ortofotomapa (riscas da água, líquenes), mais cinzento, à luz linear
    # granito: cinzento claro, com as riscas do ortofotomapa (luminância relativa à média de c. 4 m, para as
    # lajes na sombra não ficarem escuras) e manchas avermelhadas de líquenes e urze (fotografias de drone)
    l3 = o3.mean(2)
    riscas = np.clip(l3 / np.maximum(ndi.gaussian_filter(l3, 8.0), 0.05), 0.72, 1.25)
    pedra = np.array([0.33, 0.318, 0.30])[None, None, :] * (riscas * (0.9 + 0.2 * g2))[..., None]
    g3 = fbm(Xv * 0.6 - 5, Yv * 1.4 + 3, 3.0, 79)
    w_lq = np.clip((g3 - 0.58) / 0.12, 0, 1) * 0.55
    pedra = pedra * (1 - w_lq[..., None]) + np.array([0.36, 0.22, 0.18])[None, None, :] * w_lq[..., None]
    saibro = np.array([0.29, 0.275, 0.25])[None, None, :] * (0.88 + 0.24 * g2)[..., None]
    cor = erva * (1 - w_r[..., None]) + pedra * w_r[..., None]
    cor = cor * (1 - w_t[..., None]) + saibro * w_t[..., None]
    # trilhos de terra batida (OSM)
    trilhos = [shapely.LineString(f["geometria"]["coordinates"]).buffer(0.7, cap_style="flat") for f in F["caminhos"]
               if f["geometria"]["type"] == "LineString" and f["classe"] in ("path", "track")]
    if trilhos:
        w_tr = shapely.contains(shapely.union_all(trilhos).difference(zona_t), pts_v).reshape(ny, nx) & (w_r < 0.5)
        w_tr = ndi.gaussian_filter(w_tr.astype(float), 0.8)[..., None]
        cor = cor * (1 - w_tr) + np.array([0.24, 0.19, 0.13])[None, None, :] * w_tr
    cor = np.clip(cor.reshape(-1, 3) / 0.79, 0, 1)   # a textura de grão tem média 0,79 (linear)
    # ---- materiais por face: pátio relvado (fotografias de drone) e estrada; o resto é chão do monte
    cx = V[Fq].mean(1)
    pts = shapely.points(cx[:, 0], cx[:, 1])
    mi_ = np.zeros(len(Fq), np.int32)
    asf = shapely.Polygon()
    for f in F["caminhos"]:
        if f["geometria"]["type"] == "LineString" and f["classe"] in ("residential", "tertiary", "unclassified", "secondary"):
            asf = asf.union(shapely.LineString(f["geometria"]["coordinates"]).buffer(2.6, cap_style="flat"))
    if not asf.is_empty:
        mi_[shapely.contains(asf, pts)] = 2
    mi_[shapely.contains(shapely.Polygon(PATIO), pts)] = 1
    ob.data.polygons.foreach_set("material_index", mi_)
    cc = cor[Fq]                                  # cor por canto (4 por face); branco nas faces de outro material
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
    MEDIDAS["terreno"] = {"m2": {"granito": round(float((w_r > 0.5).sum()) * passo * passo), "terreiro": round(float((w_t > 0.5).sum()) * passo * passo), "patio": round(float((mi_ == 1).sum()) * passo * passo)}}


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
    """Os poucos edifícios à volta (casa do guarda, restaurante, anexos) em volumes simples de um só cinzento."""
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
            continue  # a torre e o santuário também estão no OSM
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
    """Árvores: máximos locais da vegetação (classes 3 a 5) fora do monumento e do santuário; a árvore grande do
    pátio fica, com a copa limitada para não entrar nos muros nem na torre."""
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
    patio = shapely.Polygon(PATIO)
    contorno = POL_MON.boundary
    n = 0
    for (r, c) in ndi.center_of_mass(topo, lab, range(1, k + 1)):
        r, c = int(round(r)), int(round(c))
        x, y = LX0 + c * LP, LY0 + r * LP
        if not (X0 + 2 < x < X1 - 2 and Y0 + 2 < y < Y1 - 2):
            continue
        pt = shapely.Point(x, y)
        if mon_l.contains(pt):
            continue  # pontos de vegetação sobre os muros, a torre ou o santuário: não são árvores
        zt, zc = float(cota(x, y)), MDS[r, c]
        alt = zc - zt
        if alt < 2.5:
            continue
        rad = float(np.clip(np.sqrt(veg[max(r - 10, 0):r + 11, max(c - 10, 0):c + 11].sum() * 0.25 / math.pi), 1.3, 5.5))
        # nenhuma copa a atravessar o monumento: raio até 0,3 m da face mais próxima
        rad = min(rad, tirar.distance(pt) - 0.3)
        if patio.contains(pt):
            rad = min(rad, contorno.distance(pt) - 0.3)
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


def muro_terreiro():
    """Muro baixo de pedra no bordo nascente do terreiro, por cima das lajes da encosta (fotografia de drone de
    nascente; traçado na crista estreita do LiDAR de 25 cm); altura (c. 0,85 m acima do terreiro) estimada."""
    m = Malha(["muro", "muro_topo"])
    linha = [(36.5, 3.5), (31.5, -1.5), (29.0, -7.0), (30.9, -15.0), (31.9, -20.0), (32.2, -28.0), (32.0, -40.0), (31.8, -55.0)]
    for a, b in zip(linha[:-1], linha[1:]):
        g = shapely.LineString([a, b]).buffer(0.28, cap_style="flat", join_style="mitre")
        xs, ys = np.array(g.buffer(0.8).exterior.coords).T
        zt = float(np.max(cota(xs, ys)[xs < np.max(xs) - 0.6])) + 0.85   # o terreiro fica a poente
        z0 = float(np.min(cota(xs, ys))) - 0.6
        prisma_pol(m, g, z0, zt, 0, 1)
    m.objeto("muro_terreiro", "terreno", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)


# ---------------------------------------------------------------- construção

print("torre…")
torre()
bandeira()
escada_torre()
print("cerca…")
cerca()
terraco()
F_PORTA, Z_SOL = porta()
escadaria_porta(Z_SOL)
print("santuário…")
santuario()
marcos()
muro_terreiro()
print("terreno…")
construir_terreno()
TIRAR = shapely.union_all([POL_MON, POL_IGREJA, shapely.Polygon(WL([(-16.0, 2.0), (-7.0, 2.0), (-7.0, 13.0), (-16.0, 13.0)]))])
print("envolvente…")
construir_envolvente(TIRAR)
construir_arvores(TIRAR)


def _olhar(x, y, up=0.35):
    d = unit((x, y))
    return [round(float(d[0]), 2), up, round(float(-d[1]), 2)]


MEDIDAS["ancoras"] = {
    "torre": _anc(0, 0, Z_PONTA + 1.2),
    "coroamento": _anc(*W(AU + 0.6, -AV - 0.6), Z_PONTA - 0.3),
    "porta_torre": _anc(*W(0.0, AV + 0.8), Z_PORTA + 3.0),
    "escada_torre": _anc(*W(4.0, AV + 1.2), 386.0),
    "porta": _anc(*W(-9.0, 7.2), 387.0),
    "torreoes": _anc(*W(-8.5, 10.0), C.Z_TORREOES + 2.6),
    "cerca": _anc(*W(-5.6, 18.0), C.Z_POENTE + 2.2),
    "baluarte": _anc(*W(13.5, 24.5), C.Z_NORTE + 1.8),
    "patio": _anc(*W(2.0, 11.0), 385.0),
    "santuario": _anc(*IC, 393.6),
    "fachada": _anc(*S(-9.8, 0.0), 389.0),
}
MEDIDAS["olhar"] = {
    "coroamento": _olhar(*W(1, -1), 0.25),
    "porta_torre": _olhar(*W(0.3, 1), 0.15),
    "escada_torre": _olhar(*W(0.3, 1), 0.25),
    "porta": _olhar(*W(-1, 0), 0.15),
    "torreoes": _olhar(*W(-1, 0.1), 0.25),
    "cerca": _olhar(*W(-1, 0.4), 0.3),
    "baluarte": _olhar(*W(0.6, 1), 0.35),
    "patio": _olhar(*W(0.1, 1), 0.7),
    "fachada": _olhar(*(-ES), 0.15),
}
MEDIDAS["texturas_foto"] = fontes_fotos()  # créditos das fotos de pedra e telha
(AQUI / "dados" / "medidas.json").write_text(json.dumps(MEDIDAS, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in MEDIDAS.items() if k not in ("ancoras", "olhar")}, ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
(AQUI / "saida").mkdir(exist_ok=True)
glb = AQUI / "saida" / "lanhoso-bruto.glb"
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
    final = SAIDA / "lanhoso.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "lanhoso.blend"))

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

    # capa: de sueste, do alto, com o santuário à frente e o castelo atrás
    vista("capa", (70, -80), 432, (2, -12), 384, lente=34, res=(1280, 800), destino=SAIDA / "capa.jpg")
    if "--capa" not in sys.argv:
        # d74a: drone a sul, por cima do santuário, a olhar para a torre
        vista("d74a_sul", (4, -48), 410, (2, 6), 388, lente=28, res=(1200, 900))
        # 84ef / 2a72: do terreiro a ENE, a face ENE da torre e a escada
        vista("84ef_ene", tuple(W(26.0, 4.0)), 385.0, tuple(W(0.0, 2.0)), 390.0, lente=24, res=(1000, 750), perto=0.3)
        # 3cd5: de norte, por fora da muralha norte
        vista("3cd5_norte", tuple(W(2.0, 46.0)), 395.0, tuple(W(0.0, 10.0)), 386.0, lente=28, res=(1100, 750), perto=0.3)
        # 77d45 / fa66: a porta de poente, da escadaria
        vista("fa66_porta", tuple(W(-21.0, 6.0)), 382.6, tuple(W(-7.3, 7.2)), 385.5, lente=26, res=(900, 1000), perto=0.3)
        # image 10: a porta de frente, do fundo da escadaria
        vista("i10_porta_frente", tuple(W(-19.0, 6.6)), float(cota(*W(-19.0, 6.6))) + 1.6, tuple(W(-7.3, 7.0)), 386.4, lente=20, res=(900, 760), perto=0.3)
        # 00e6 de perto: a fachada do santuário de frente, à altura da foto
        vista("00e6_perto", tuple(S(-24.0, 0.3)), 379.2, tuple(S(-9.6, 0.0)), 384.6, lente=32, res=(800, 1000), perto=0.3)
        # 84ef: drone de nascente, com as lajes de granito da encosta nascente
        vista("84ef_drone", (95, -30), 430, (-5, -12), 380, lente=30, res=(1280, 853))
        # 5b79: de dentro do pátio para a porta
        vista("5b79_patio", tuple(W(4.0, 14.0)), 385.2, tuple(W(-6.0, 7.5)), 385.0, lente=22, res=(1100, 800), perto=0.3)
        # 8ae8: do baluarte NE para a torre
        vista("8ae8_ne", tuple(W(12.5, 23.0)), 388.4, tuple(W(0.0, 3.0)), 389.0, lente=24, res=(1100, 750), perto=0.3)
        # 6c4d: o adarve poente, de norte para sul
        vista("6c4d_adarve", tuple(W(-5.5, 19.0)), 388.6, tuple(W(-6.0, 6.0)), 388.0, lente=22, res=(800, 1000), perto=0.2)
        # 00e6: a fachada do santuário, do terreiro
        vista("00e6_fachada", tuple(S(-30.0, 2.0)), 380.0, tuple(S(-9.6, 0.0)), 385.5, lente=28, res=(900, 1000), perto=0.3)
        # coroamento de perto, de sudoeste
        vista("coroamento", tuple(W(-9.0, -9.0)), 398.5, tuple(W(0.0, 0.0)), 394.5, lente=28, res=(1000, 750), perto=0.3)
        # planta (ortográfica, norte para cima), para comparar com o ortofotomapa
        co.location = Vector(((X0 + X1) / 2, (Y0 + Y1) / 2, 900))
        co.rotation_euler = (0, 0, 0)
        cam.type = "ORTHO"
        cam.ortho_scale = X1 - X0
        sc.render.resolution_x, sc.render.resolution_y = 1500, int(1500 * (Y1 - Y0) / (X1 - X0))
        sc.render.filepath = str(pasta / "topo.jpg")
        if not SO or "topo" in SO:
            bpy.ops.render.render(write_still=True)
