"""Constrói a maqueta 3D do Castelo de Almourol (estado atual) em Blender e exporta-a para
public/maquetas/almourol/almourol.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow rasterio
    python pipeline/maquetas/almourol/construir.py            # GLB
    python pipeline/maquetas/almourol/construir.py --vistas   # GLB + capa + vistas de verificação

Dados (em dados/): lidar.npz e superficie.npz (lidar.py: LiDAR 2024 da DGT, folhas 178278 e 178277),
muralhas.json (tracar.py: estações medidas nos panos e círculos dos cubelos), castelo.npz (pontos do castelo),
orto.jpg (ortofoto.py: ortofotomapa 2025 da DGT).

O que se modela, e de onde vem:
- a torre de menagem, rodada 29,3° (faces a ENE, NNO, OSO e SSE), com a planta e as cotas do LiDAR: terraço a
  59,25 m, cinco merlões por face, a casota branca da escada sobre o terraço (2014), a porta e as janelas das
  fotografias;
- a cerca: oito panos medidos estação a estação (face de fora, espessura, adarve, parapeito, merlões) e oito
  cubelos redondos (centro, raio e cotas do LiDAR), com o muro interior que separa o recinto de cima (norte, com a
  torre) do recinto de baixo (sul);
- a porta principal a poente (arco com a lápide por cima) e a porta entre os dois recintos;
- o chão dos recintos e das escavações, a rocha granítica aflorante, a ilha, as margens e o Tejo à cota medida;
- as árvores do LiDAR (nenhuma sobre o monumento) e os edifícios da margem norte em volumes cinzentos.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pipeline/, para o kit comum
from kit.geometria import normais, amostra, _anc  # noqa: E402
from kit.texturas import fbm, textura_relva, imagem  # noqa: E402
from kit.cena import cor_vertices, uv_mundo  # noqa: E402
from kit.fotos import textura_foto, fontes_fotos  # noqa: E402
from kit.silhares import textura_silhares  # noqa: E402
from kit.materiais import MATS, malha_rapida, material  # noqa: E402
from kit.arquitetura import aduelas, vao, guarda  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "almourol.json").read_text())
MUR = json.loads((AQUI / "dados" / "muralhas.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "almourol"
LID = np.load(AQUI / "dados" / "lidar.npz")
LP, LX0, LY0 = float(LID["passo"]), float(LID["x0"]), float(LID["y0"])
MDT = LID["mdt"] / 100.0 + float(LID["base"])
MDS = LID["mds"] / 100.0 + float(LID["base"])
CLS = LID["classe"]
AGUA = LID["agua"]
SUPF = np.load(AQUI / "dados" / "superficie.npz")
SP, SX0, SY0 = float(SUPF["passo"]), float(SUPF["x0"]), float(SUPF["y0"])
SUP = SUPF["z"] / 100.0 + float(SUPF["base"])
ORTO = np.asarray(Image.open(AQUI / "dados" / "orto.jpg").convert("RGB"), float) / 255.0  # 25 cm, norte em cima
OX0, OY1, OP = -350.0, 215.0, 0.25

# base redonda (por omissão desde 10 nov.): a ilha inteira, o Tejo e as duas margens
BASE_C = np.array([-120.0, -12.0])
BASE_R = 226.0
NIVEL = 13.23            # Tejo: cota da água no LiDAR de junho de 2024 (mediana do MDT sobre a classe 9)
rng = np.random.default_rng(1171)


def _agua():
    """Superfície do Tejo: o MDT à cota da água (até +0,35 m) sem nada por cima, mais os pontos da classe 9; os pontos
    de água são raros no LiDAR, mas o MDT interpolado sobre o rio fica plano à cota da água (confere com o
    ortofotomapa de 2025)."""
    w = ((MDT <= NIVEL + 0.35) & (MDS - MDT < 1.0)) | LID["agua"]
    w = ndi.binary_closing(ndi.binary_opening(w, iterations=2), iterations=4)
    lab, k = ndi.label(w)
    w = np.isin(lab, 1 + np.nonzero(ndi.sum(w, lab, range(1, k + 1)) > 2000)[0])
    terra = ~w
    lab, k = ndi.label(terra)
    return w | np.isin(lab, 1 + np.nonzero(ndi.sum(terra, lab, range(1, k + 1)) < 200)[0])


AGUA = _agua()


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def cota(x, y):
    """Chão (MDT de 50 cm)."""
    return amostra(MDT, LX0, LY0, LP, x, y)


# ---------------------------------------------------------------- referencial da torre de menagem (LiDAR)
# O retângulo mínimo dos pontos do terraço dá as faces a 60,7° e 150,7° de azimute: u = azimute 60,7° (ENE),
# v = azimute 330,7° (NNO). Os pontos do terraço vão de u -3,58 a 3,68 e de v -3,54 a 3,88.
ROT = math.radians(29.3)
EU = np.array([math.cos(ROT), math.sin(ROT)])
EV = np.array([-math.sin(ROT), math.cos(ROT)])
TC = np.array([0.05, 0.17])   # centro do corpo da torre em (u, v)
AU, AV = 3.63, 3.71           # meias larguras (7,26 × 7,42 m)
Z_TERRACO = 59.25             # terraço (LiDAR)
Z_SOLEIRA_T = 59.95           # soleira das ameias
Z_MERLAO_T = 60.95            # pontas dos merlões (LiDAR: 60,9–61,0)
CHAO_T = {"ENE": 41.3, "OSO": 42.2, "NNO": 41.6, "SSE": 38.4}  # chão junto a cada face (LiDAR)


def W(u, v):
    return EU * (u + TC[0]) + EV * (v + TC[1])


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
        q = [W(u0, v0), W(u1, v0), W(u1, v1), W(u0, v1)]
        self.prisma(q, z0, z1, mat, topo=topo, fundo=fundo)

    def caixa_face(self, f, s0, s1, d0, d1, z0, z1, mat=0, topo=True, fundo=False):
        q = [f.xy(s0, d0), f.xy(s1, d0), f.xy(s1, d1), f.xy(s0, d1)]
        self.prisma(q, z0, z1, mat, topo=topo, fundo=fundo)

    def barra(self, p0, p1, r, mat=0, k=6):
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
        if vcor is not None:
            cor_vertices(me, vcor)
        return ob


class Plano:
    """Plano vertical: o = origem (x, y), t = eixo ao longo, n = normal para fora (à esquerda de t por omissão)."""

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


# ---------------------------------------------------------------- cena

for _ob in list(bpy.data.objects):  # a cena de arranque do bpy traz um cubo, uma luz e uma câmara
    bpy.data.objects.remove(_ob)
RAIZ_OB = bpy.data.objects.new("Castelo_Almourol", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = CFG["origem"]["lat"], CFG["origem"]["lon"]
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
# "Muralhas_grupo" e "Torre_grupo" são o que o visor enquadra ao abrir; Terreno/Vegetacao/Envolvente ficam fora
for g in ["Terreno", "Torre", "Muralhas", "Envolvente", "Vegetacao", "Rochedos"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

# Pedra: alvenaria de granito e xisto argamassada, de tom quente (ocre, castanho-dourado, algum rosado), como em
# todas as fotografias de 2025 do Commons e nas enviadas (adarves, torres, muros dos recintos). A foto
# old_stone_wall (Poly Haven, CC0) dá o desenho de pedras irregulares com juntas largas; com uv=2,0 e escala=0,5
# cada imagem cobre 4 m de parede.
TEX, ALT = textura_foto("old_stone_wall", n=1024, metros=4.0, cor=(0.70, 0.56, 0.39), contraste=1.2, saturacao=0.62,
                        variacao=0.10, liquen=0.04, semente=31)
material("muro", (1, 1, 1), 0.92, imagem("muro", TEX), normal=imagem("muro_n", normais(ALT, 4.0)), forca=1.0, escala=0.5, vcor=True)
TEX, ALT = textura_foto("old_stone_wall", n=1024, metros=4.0, cor=(0.71, 0.59, 0.44), contraste=1.1, saturacao=0.55,
                        variacao=0.08, semente=32)
material("muro_topo", (1, 1, 1), 0.95, imagem("muro_topo", TEX), normal=imagem("muro_topo_n", normais(ALT, 4.0)), forca=0.8, escala=0.5, vcor=True)
# torre de menagem: aparelho mais regular, em fiadas, com cunhais de cantaria (fotografias da torre, 2025)
TEX, ALT = textura_silhares(n=1024, metros=4.0, fiada=(0.28, 0.42), comp=(0.35, 0.8), cor=(0.72, 0.59, 0.43), desvio=0.10,
                            junta_m=0.016, junta_cor=(0.55, 0.47, 0.38), liquen=0.03, irregular=0.18, semente=33)
material("cantaria_torre", (1, 1, 1), 0.9, imagem("cantaria_torre", TEX), normal=imagem("cantaria_torre_n", normais(ALT, 4.0)),
         forca=0.9, escala=0.5, vcor=True)
material("cantaria_lisa", (0.70, 0.62, 0.50), 0.85)   # ombreiras, aduelas e lápides de granito claro
material("lapide", (0.90, 0.88, 0.83), 0.7)          # lápides das inscrições (pedra clara)
material("sombra", (0.05, 0.045, 0.04), 1.0)
material("madeira", (0.30, 0.20, 0.13), 0.8)         # folhas da porta principal (fotografia 50 do Commons)
material("ferro", (0.13, 0.13, 0.14), 0.5)
material("branco", (0.86, 0.85, 0.82), 0.7)          # casota da escada no terraço da torre (fotografias aéreas)
material("saibro", (0.70, 0.60, 0.45), 0.95)         # chão de saibro dos recintos e do caminho
TEX, ALT = textura_foto("rock_01", n=512, metros=3.0, cor=(0.46, 0.44, 0.40), contraste=1.0, liquen=0.03, semente=34)
material("rocha", (1, 1, 1), 0.85, imagem("rocha", TEX), normal=imagem("rocha_n", normais(ALT, 3.0)), forca=0.7)
TEX, ALT = textura_relva()
material("relva", (1, 1, 1), 0.95, imagem("relva", TEX), normal=imagem("relva_n", normais(ALT, 1.0)), forca=0.4)
material("mato", (0.47, 0.44, 0.30), 0.95)           # erva seca e mato rasteiro (junho)
material("areia", (0.80, 0.74, 0.62), 0.95)          # areais do rio
material("asfalto", (0.40, 0.40, 0.41), 0.95)        # parque e estrada da margem norte
material("balastro", (0.47, 0.45, 0.42), 0.95)       # linha da Beira Baixa
material("cinza_viz", (0.74, 0.74, 0.73), 0.9)       # edifícios da margem: um só cinzento
material("base", (0.33, 0.26, 0.20), 0.95)
material("agua", (0.16, 0.25, 0.27), 0.12)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)

MEDIDAS = {}


def patine_muro(co, nrm):
    """Manchas grandes (pedra reposta e velha), pé mais escuro, faces a sul e poente mais quentes."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x + 500, y + 500 + z * 0.7, 6.0, 31)
    v = 0.86 + 0.26 * g
    sol = np.clip(-nrm[:, 1] * 0.6 - nrm[:, 0] * 0.3, 0, 1)
    tom = np.stack([1.0 + 0.04 * sol, 1.0 + 0.0 * sol, 1.0 - 0.06 * sol], 1)
    pe = np.clip(1 - (z - cota(x, y)) / 1.6, 0, 1) * 0.12
    return np.clip(v[:, None] * tom * (1 - pe)[:, None], 0, 1)


def patine_torre(co, nrm):
    z = co[:, 2]
    du, dv = nrm[:, :2] @ EU, nrm[:, :2] @ EV
    sol = np.clip(np.maximum(-dv, -du), 0, 1)[:, None]
    tom = sol * np.array([1.0, 0.93, 0.84]) + (1 - sol) * np.array([0.92, 0.91, 0.90])
    t = np.clip((z - 50.0) / 10.0, 0, 1)[:, None]
    return np.clip(tom * (1 - 0.06 * t), 0, 1)


# ---------------------------------------------------------------- muralhas

PERIODO = 1.62      # merlões: 0,95 m de largura e ameias de c. 0,67 m (contados nas fotografias dos adarves)
L_MERLAO = 0.95
ESP_PAR = 0.55      # espessura do parapeito
ALT_MERLAO = 1.05   # da soleira das ameias à ponta (LiDAR: topo − adarve c. 2,0 m; parapeito c. 1,0 m)

# espessura de cada pano (mediana das estações, sem as torres) e lado do parapeito: nos panos da cerca o parapeito
# fica do lado de fora (à esquerda do sentido de marcha); no muro interior, do lado do recinto de baixo (à direita)
ESP = {"oeste_sul": 2.2, "oeste_norte": 2.0, "norte": 2.0, "nordeste": 2.6, "leste": 2.1, "sul": 1.6,
       "interior_oeste": 1.9, "interior_leste": 2.1}
PARAPEITO_DIR = {"interior_oeste", "interior_leste"}
TORRES = dict(MUR["torres"])
# dois cubelos meio embebidos que o traçado automático não apanhou: o topo do pano sobe c. 2 m (43,5–43,7, como o
# cubelo sueste) ao longo de 3–4 m e a face de fora sai 1–2,4 m (estações do LiDAR); confirmados nas fotografias
# aéreas de sueste (dois cubelos entre os cantos, no pano leste e no pano sul)
TORRES["leste_meio"] = {"c": [23.75, -8.25], "r": 1.9, "topo": 43.57, "piso": 42.45, "chao_min": 28.8}
TORRES["sul_meio"] = {"c": [-5.1, -18.0], "r": 1.8, "topo": 43.72, "piso": 42.6, "chao_min": 29.1}


def perto_torre(p, folga=0.4):
    for t in TORRES.values():
        if math.hypot(p[0] - t["c"][0], p[1] - t["c"][1]) < t["r"] + folga:
            return True
    return False


def dentro_torre_menagem(p, folga=0.3):
    q = np.asarray(p) - W(0, 0)
    return abs(q @ EU) < AU + folga and abs(q @ EV) < AV + folga


def suave(v, k=7, s=1.5):
    v = np.asarray(v, float)
    v = ndi.median_filter(v, size=k, mode="nearest")
    return ndi.gaussian_filter1d(v, s, mode="nearest")


def estacoes(nome):
    e = MUR["panos"][nome]["estacoes"]
    c = np.array([x["c"] for x in e])
    n = np.array([x["n"] for x in e])
    fora = np.array([x["fora"] for x in e])
    dentro = np.array([x["dentro"] for x in e])
    topo = np.array([x["topo"] for x in e])
    adarve = np.array([x["adarve"] for x in e])
    esp = ESP[nome]
    torre = np.array([perto_torre(p, 0.8) for p in c])
    if nome in PARAPEITO_DIR:
        meio = suave(np.clip((fora + dentro) / 2, -1.2, 1.2), 9, 2.0)
        ext = c + n * (meio + esp / 2)[:, None]
    else:
        f = np.where(torre, np.nan, np.clip(fora, -1.6, 1.6))
        idx = np.arange(len(f))
        ok = ~np.isnan(f)
        f = np.interp(idx, idx[ok], f[ok]) if ok.any() else np.zeros(len(f))
        ext = c + n * suave(f, 9, 2.0)[:, None]
    tp = np.where(torre, np.nan, topo)
    idx = np.arange(len(tp))
    ok = ~np.isnan(tp)
    tp = np.interp(idx, idx[ok], tp[ok])
    topo_s = suave(tp, 9, 2.0)
    soleira = topo_s - ALT_MERLAO
    ad = np.where(torre, np.nan, adarve)
    ad = np.interp(idx, idx[~np.isnan(ad)], ad[~np.isnan(ad)])
    ad = np.minimum(suave(ad, 9, 2.0), soleira - 0.85)
    ad = np.maximum(ad, soleira - 1.25)
    return c, n, ext, esp, soleira, topo_s, ad


def pe_muro(pts):
    """Cota do pé do muro: o chão mais baixo à volta de cada ponto (o terreno corta o resto)."""
    out = []
    for p in pts:
        zs = [cota(p[0] + dx, p[1] + dy) for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))]
        out.append(min(zs) - 1.5)
    return np.array(out)


MERLOES = []   # (x, y, z) de cada merlão, para medir e conferir


# portas (vãos de passagem com arco de volta perfeita): centro em planta, largura, soleira, altura das ombreiras
# (arranque do arco), lado dos degraus (+1 do lado do parapeito/exterior, −1 do outro) e lápide por cima
PORTAS = {
    # porta principal, a poente: ao fundo de um corredor entre o cubelo oeste-sul e o cubelo oeste, com penedos de
    # granito à esquerda de quem entra, três degraus, arco de volta perfeita de cantaria e uma lápide comprida e baixa
    # por cima (Commons, «Castelo de Almourol em Julho de 2025», fotos 79–82, e «em Agosto 2025», foto 17). Pela
    # porta vê-se o recinto de baixo e os merlões do pano sul ao nível dos olhos, por isso a porta abre para o
    # recinto de baixo, a sul do muro interior (chão de dentro 37,9; de fora 37,3 no LiDAR); folha de madeira aberta
    # por dentro (foto 79)
    "oeste_sul": [dict(c=(-9.0, -5.15), larg=1.4, zs=37.95, omb=1.75, degraus=+1, lapide=(1.6, 0.34), aduelas=True,
                       folha=True)],
    # porta no muro interior a poente da torre, ao nível do recinto de baixo (fotografia aérea de SSO)
    "interior_oeste": [dict(c=(-5.0, -4.3), larg=1.2, zs=38.0, omb=1.9, degraus=0, lapide=None, aduelas=True)],
    # porta do muro interior a nascente da torre, com lápide por cima e a rocha à esquerda (fotografias aéreas de
    # SSO e de sueste; Commons «em Agosto 2025», foto 25); o adarve tem aqui uma quebra no LiDAR (43,2)
    "interior_leste": [dict(c=(13.3, 2.0), larg=1.2, zs=39.75, omb=1.9, degraus=0, lapide=(0.9, 0.55), aduelas=True)],
}


def _s_linha(L):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))])


def _no_s(L, s, q):
    return np.array([np.interp(q, s, L[:, k]) for k in range(L.shape[1])])


def _inserir(arrs, s, novos):
    """Acrescenta estações exatas nas abcissas `novos` (para os vãos começarem e acabarem numa estação)."""
    s2 = np.unique(np.concatenate([s, novos]))
    out = []
    for a in arrs:
        a = np.asarray(a, float)
        if a.ndim == 1:
            out.append(np.interp(s2, s, a))
        else:
            out.append(np.stack([np.interp(s2, s, a[:, k]) for k in range(a.shape[1])], 1))
    return out, s2


def pano(m, nome):
    """Um pano de muralha: faces de fora e de dentro até ao pé, adarve, parapeito e merlões; portas como vãos."""
    c, n, ext, esp, sol, topo, ad = estacoes(nome)
    portas = []
    if nome in PORTAS:
        s = _s_linha(ext)
        novos = []
        for d in PORTAS[nome]:
            k = int(np.argmin(np.linalg.norm(ext - np.array(d["c"]), axis=1)))
            # abcissa do pé da perpendicular do centro sobre a linha
            sc = s[k]
            for j in (k - 1, k):
                if 0 <= j < len(s) - 1:
                    a, b = ext[j], ext[j + 1]
                    t = float(np.clip(np.dot(np.array(d["c"]) - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1))
                    if np.linalg.norm(a + t * (b - a) - d["c"]) <= np.linalg.norm(ext[k] - d["c"]) + 1e-6:
                        sc = s[j] + t * (s[j + 1] - s[j])
            dd = dict(d, s0=sc - d["larg"] / 2, s1=sc + d["larg"] / 2)
            dd["apex"] = dd["zs"] + dd["omb"] + d["larg"] / 2
            portas.append(dd)
            novos += [dd["s0"], dd["s1"]]
        (c, n, ext, sol, topo, ad), s = _inserir([c, n, ext, sol, topo, ad], s, novos)
        n = n / np.linalg.norm(n, axis=1)[:, None]
    lado = -1.0 if nome in PARAPEITO_DIR else 1.0   # sinal da normal do lado do parapeito
    fora_l = ext                     # linha da face do lado do parapeito (ou exterior)
    dent_l = ext - n * esp           # face do outro lado
    if lado < 0:
        fora_l, dent_l = ext - n * esp, ext
    npar = n * lado                  # normal que aponta para o lado do parapeito
    par_l = fora_l - npar * ESP_PAR
    pe_f = pe_muro(fora_l + npar * 0.4)
    pe_d = pe_muro(dent_l - npar * 0.4)
    N = len(c)
    keep = np.array([not (perto_torre(p, -0.3) or dentro_torre_menagem(p, -0.2)) for p in (fora_l + dent_l) / 2])
    s_ext = _s_linha(ext)
    for i in range(N - 1):
        if not (keep[i] and keep[i + 1]):
            continue
        a, b = i, i + 1
        sm = (s_ext[a] + s_ext[b]) / 2
        vao = next((d for d in portas if d["s0"] - 1e-6 <= sm <= d["s1"] + 1e-6), None)
        if vao is None:
            F = [(*fora_l[a], pe_f[a]), (*fora_l[b], pe_f[b]), (*fora_l[b], sol[b]), (*fora_l[a], sol[a])]
            D = [(*dent_l[b], pe_d[b]), (*dent_l[a], pe_d[a]), (*dent_l[a], ad[a]), (*dent_l[b], ad[b])]
            m.face(F, "muro")
            m.face(D, "muro")
        else:
            zs, za = vao["zs"], vao["apex"]
            m.face([(*fora_l[a], pe_f[a]), (*fora_l[b], pe_f[b]), (*fora_l[b], zs), (*fora_l[a], zs)], "muro")
            m.face([(*fora_l[a], za), (*fora_l[b], za), (*fora_l[b], sol[b]), (*fora_l[a], sol[a])], "muro")
            m.face([(*dent_l[b], pe_d[b]), (*dent_l[a], pe_d[a]), (*dent_l[a], zs), (*dent_l[b], zs)], "muro")
            m.face([(*dent_l[b], za), (*dent_l[a], za), (*dent_l[a], ad[a]), (*dent_l[b], ad[b])], "muro")
        m.face([(*dent_l[a], ad[a]), (*par_l[a], ad[a]), (*par_l[b], ad[b]), (*dent_l[b], ad[b])], "muro_topo")
        m.face([(*par_l[a], ad[a]), (*par_l[a], sol[a]), (*par_l[b], sol[b]), (*par_l[b], ad[b])], "muro")
        m.face([(*par_l[a], sol[a]), (*fora_l[a], sol[a]), (*fora_l[b], sol[b]), (*par_l[b], sol[b])], "muro_topo")
    # topos dos troços (junto às torres e nas pontas)
    for i in range(N):
        if not keep[i]:
            continue
        for viz in (i - 1, i + 1):
            if viz < 0 or viz >= N or not keep[viz]:
                m.face([(*fora_l[i], pe_f[i]), (*dent_l[i], pe_d[i]), (*dent_l[i], ad[i]), (*par_l[i], ad[i]),
                        (*par_l[i], sol[i]), (*fora_l[i], sol[i])], "muro")
    for d in portas:
        vao_porta(m, d, fora_l, dent_l, s_ext, npar)
    # merlões ao longo da linha do parapeito
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(fora_l, axis=0), axis=1))])
    for s0 in np.arange(0.5, s[-1] - 0.5, PERIODO):
        s1 = s0 + L_MERLAO
        if s1 > s[-1] - 0.2:
            break
        ia, ib = np.searchsorted(s, s0), np.searchsorted(s, s1)
        ia, ib = min(ia, N - 1), min(ib, N - 1)
        if not (keep[ia] and keep[ib]):
            continue
        pa = np.array([np.interp(s0, s, fora_l[:, 0]), np.interp(s0, s, fora_l[:, 1])])
        pb = np.array([np.interp(s1, s, fora_l[:, 0]), np.interp(s1, s, fora_l[:, 1])])
        na = npar[ia]
        z0 = float(np.interp((s0 + s1) / 2, s, sol))
        z1 = float(np.interp((s0 + s1) / 2, s, topo))
        q = [pa, pb, pb - na * ESP_PAR, pa - na * ESP_PAR]
        m.prisma(q, z0 - 0.02, z1, "muro_topo" if False else "muro", topo=True)
        MERLOES.append((*((pa + pb) / 2), z1))
    return dict(ext=fora_l, dent=dent_l, ad=ad, sol=sol, topo=topo)


VAOS_PORTA = []   # (nome do pano, centro xy, soleira, normal) de cada porta, para o raycast de conferência


def vao_porta(m, d, fora_l, dent_l, s_ext, npar):
    """Passagem da porta: soleira, ombreiras, intradorso do arco e tímpanos dos dois lados; aduelas e ombreiras de
    cantaria salientes do lado do parapeito (exterior), lápide e degraus."""
    s0, s1, zs, zm = d["s0"], d["s1"], d["zs"], d["zs"] + d["omb"]
    r = (s1 - s0) / 2
    za = d["apex"]
    Fa, Fb = _no_s(fora_l, s_ext, s0), _no_s(fora_l, s_ext, s1)
    Da, Db = _no_s(dent_l, s_ext, s0), _no_s(dent_l, s_ext, s1)
    nn = unit(_no_s(npar, s_ext, (s0 + s1) / 2))
    m.face([(*Fa, zs), (*Da, zs), (*Db, zs), (*Fb, zs)], "cantaria_lisa")
    m.face([(*Fa, zs), (*Fa, zm), (*Da, zm), (*Da, zs)], "muro")
    m.face([(*Fb, zs), (*Db, zs), (*Db, zm), (*Fb, zm)], "muro")
    k = 12
    th = np.linspace(math.pi, 0, k + 1)

    def arco(A, B, rr, ang):
        t = 0.5 + rr * math.cos(ang) / (2 * r)
        return (*(A + (B - A) * t), zm + rr * math.sin(ang))

    for j in range(k):
        m.face([arco(Fa, Fb, r, th[j]), arco(Da, Db, r, th[j]), arco(Da, Db, r, th[j + 1]), arco(Fa, Fb, r, th[j + 1])], "muro")
    for (A, B) in ((Fa, Fb), (Da, Db)):
        ca, cb = (*A, za), (*B, za)
        for j in range(k):
            canto = ca if th[j] > math.pi / 2 - 1e-6 and th[j + 1] >= math.pi / 2 - 1e-6 else cb
            m.face([canto, arco(A, B, r, th[j]), arco(A, B, r, th[j + 1])], "muro")
        m.face([ca, (*A, zm), arco(A, B, r, math.pi)], "muro")
        m.face([cb, arco(A, B, r, 0.0), (*B, zm)], "muro")
    if d.get("aduelas"):
        off = nn * 0.04
        A, B = Fa + off, Fb + off
        nad = 11   # aduelas separadas por juntas de c. 2 cm (fotografia da porta)
        for j in range(nad):
            a0 = math.pi - math.pi * j / nad - 0.012
            a1 = math.pi - math.pi * (j + 1) / nad + 0.012
            if j == 0:
                a0 = math.pi
            if j == nad - 1:
                a1 = 0.0
            for (b0, b1) in ((a0, (a0 + a1) / 2), ((a0 + a1) / 2, a1)):
                m.face([arco(A, B, r, b0), arco(A, B, r, b1), arco(A, B, r + 0.36, b1), arco(A, B, r + 0.36, b0)], "cantaria_lisa")
        t = unit(Fb - Fa)
        for (P, sg) in ((Fa, -1), (Fb, 1)):
            q = [P, P + t * sg * 0.3, P + t * sg * 0.3 + off, P + off]
            m.prisma(q if sg > 0 else q[::-1], zs, zm, "cantaria_lisa", topo=False)
    if d.get("lapide"):
        lw, lh = d["lapide"]
        mid = (Fa + Fb) / 2
        t = unit(Fb - Fa)
        q = [mid - t * lw / 2, mid + t * lw / 2, mid + t * lw / 2 + nn * 0.05, mid - t * lw / 2 + nn * 0.05]
        m.prisma(q, za + 0.45, za + 0.45 + lh, "lapide")
    if d.get("folha"):
        # folha de madeira aberta, encostada à ombreira do lado de dentro
        t = unit(Fb - Fa)
        P = Db - nn * 0.02
        q = [P - t * 0.02, P - t * 0.09, P - t * 0.09 - nn * (s1 - s0), P - t * 0.02 - nn * (s1 - s0)]
        m.prisma(q, zs + 0.02, zm + r * 0.6, "madeira", fundo=True)
    if d.get("degraus"):
        lado = d["degraus"]
        P = ((Fa + Fb) / 2) if lado > 0 else ((Da + Db) / 2)
        nd = nn * lado
        t = unit(Fb - Fa)
        q_ = P + nd * 1.6
        chao = float(np.atleast_1d(cota([q_[0]], [q_[1]]))[0])
        nd_ = int(round((zs - chao) / 0.17)) - 1 if zs - chao > 0.12 else 0
        for j in range(nd_):
            zt = zs - (j + 1) * (zs - chao) / (nd_ + 1)
            p0, p1 = P + nd * (0.05 + 0.38 * j), P + nd * (0.05 + 0.38 * (j + 1))
            q = [p0 - t * 0.95, p0 + t * 0.95, p1 + t * 0.95, p1 - t * 0.95]
            e1, e2 = q[1] - q[0], q[2] - q[1]
            if e1[0] * e2[1] - e1[1] * e2[0] < 0:
                q = q[::-1]
            m.prisma(q, chao - 0.6, zt, "cantaria_lisa")
    VAOS_PORTA.append((d["c"], zs, (nn).tolist()))


# portas dos cubelos ao nível do adarve: (azimute matemático do vão a partir do centro, soleira, verga)
# nordeste: a escadaria norte acaba numa porta da torre (fotografia do alto da torre virada a nordeste)
PORTAS_CUBELO = {"nordeste": (164.0, 46.45, 48.35)}


def cubelo(m, nome, t):
    """Cubelo redondo: corpo do chão ao parapeito, piso no alto, parapeito em anel e merlões."""
    c, r = np.array(t["c"]), t["r"]
    topo = t["topo"]
    sol = topo - ALT_MERLAO
    piso = min(sol - 0.9, t["piso"] if t["piso"] else sol - 1.0)
    piso = max(piso, sol - 1.3)
    k = 32
    a = np.linspace(0, 2 * math.pi, k, endpoint=False)
    ring = [c + r * np.array([math.cos(x), math.sin(x)]) for x in a]
    rin = [c + (r - ESP_PAR) * np.array([math.cos(x), math.sin(x)]) for x in a]
    pe = [cota(*(p + unit(p - c) * 0.6)) - 1.5 for p in ring]
    porta = PORTAS_CUBELO.get(nome)
    for i in range(k):
        j = (i + 1) % k
        am = (a[i] + a[j] + (2 * math.pi if j == 0 else 0)) / 2
        if porta and abs((am - math.radians(porta[0]) + math.pi) % (2 * math.pi) - math.pi) < 0.5 / r:
            z0p, z1p = porta[1], porta[2]
            m.face([(*ring[i], pe[i]), (*ring[j], pe[j]), (*ring[j], z0p), (*ring[i], z0p)], "muro")
            m.face([(*ring[i], z1p), (*ring[j], z1p), (*ring[j], sol), (*ring[i], sol)], "muro")
            # vão com 1 m de fundo: soleira, lados, verga e o escuro do interior da torre
            d_ = [c + (r - 1.0) * np.array([math.cos(x), math.sin(x)]) for x in (a[i], a[j] + (2 * math.pi if j == 0 else 0))]
            m.face([(*ring[i], z0p), (*ring[j], z0p), (*d_[1], z0p), (*d_[0], z0p)], "cantaria_lisa")
            m.face([(*ring[j], z1p), (*ring[i], z1p), (*d_[0], z1p), (*d_[1], z1p)], "cantaria_lisa")
            m.face([(*ring[i], z0p), (*d_[0], z0p), (*d_[0], z1p), (*ring[i], z1p)], "muro")
            m.face([(*d_[1], z0p), (*ring[j], z0p), (*ring[j], z1p), (*d_[1], z1p)], "muro")
            m.face([(*d_[0], z0p), (*d_[1], z0p), (*d_[1], z1p), (*d_[0], z1p)], "sombra")
        else:
            m.face([(*ring[i], pe[i]), (*ring[j], pe[j]), (*ring[j], sol), (*ring[i], sol)], "muro")
        m.face([(*ring[i], sol), (*ring[j], sol), (*rin[j], sol), (*rin[i], sol)], "muro_topo")
        m.face([(*rin[j], piso), (*rin[i], piso), (*rin[i], sol), (*rin[j], sol)], "muro")
    m.face([(*p, piso) for p in rin], "muro_topo")
    nm = max(4, int(round(2 * math.pi * (r - ESP_PAR / 2) / PERIODO)))
    for j in range(nm):
        a0 = 2 * math.pi * j / nm
        da = L_MERLAO / r
        arc = np.linspace(a0, a0 + da, 3)
        out = [c + r * np.array([math.cos(x), math.sin(x)]) for x in arc]
        inn = [c + (r - ESP_PAR) * np.array([math.cos(x), math.sin(x)]) for x in arc[::-1]]
        pm = (out[1] + inn[1]) / 2
        if any(dentro_torre_menagem(p, 0.0) for p in (pm,)):
            continue
        m.prisma(out + inn, sol - 0.02, topo, "muro")
        MERLOES.append((*pm, topo))
    MEDIDAS.setdefault("cubelos", {})[nome] = {"centro": [round(float(c[0]), 2), round(float(c[1]), 2)], "raio": round(r, 2),
                                               "soleira": round(sol, 2), "merloes_pontas": round(topo, 2), "merloes": nm,
                                               "chao_min": round(t["chao_min"], 2)}


def escada_norte(m):
    """Escadaria do adarve norte: sobe encostada à face de dentro do pano norte, de poente para nascente, do chão do
    recinto de cima (41,0) ao adarve (46,5). No LiDAR é uma rampa de 7,2 m a 2,0–2,5 m do eixo do pano; na
    fotografia do alto da torre virada a nordeste vê-se a escada comprida e maciça junto ao muro."""
    A, B = np.array([12.4, 12.4]), np.array([23.4, 7.6])
    t = unit(B - A)
    n = np.array([t[1], -t[0]])          # para dentro (sul)
    s0, s1, z0, z1 = 0.75, 7.95, 41.0, 46.45
    k = int(round((z1 - z0) / 0.19))
    for j in range(k):
        sa = s0 + (s1 - s0) * j / k
        sb = s1 + 0.6 if j == k - 1 else s0 + (s1 - s0) * (j + 1) / k
        zt = z0 + (z1 - z0) * (j + 1) / k
        q = [A + t * sa + n * 1.6, A + t * sb + n * 1.6, A + t * sb + n * 2.75, A + t * sa + n * 2.75]
        if (q[1] - q[0])[0] * (q[2] - q[1])[1] - (q[1] - q[0])[1] * (q[2] - q[1])[0] < 0:
            q = q[::-1]
        m.prisma(q, z0 - 1.0, zt, "muro", mat_topo="muro_topo")
    MEDIDAS["escada_norte"] = {"degraus": k, "espelho_m": round((z1 - z0) / k, 3), "cobertor_m": round((s1 - s0) / k, 3)}


def muralhas():
    m = Malha(["muro", "muro_topo", "cantaria_lisa", "lapide", "madeira"])
    INFO = {}
    escada_norte(m)
    for nome in MUR["panos"]:
        INFO[nome] = pano(m, nome)
    m.objeto("cerca", "cerca", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)
    mc = Malha(["muro", "muro_topo", "sombra", "cantaria_lisa"])
    for nome, t in TORRES.items():
        cubelo(mc, nome, t)
    mc.objeto("cubelos", "cubelos", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)
    MEDIDAS["panos"] = {k: {"adarve_mediana": round(float(np.median(v["ad"])), 2),
                            "merloes_mediana": round(float(np.median(v["topo"])), 2)} for k, v in INFO.items()}
    return INFO


# ---------------------------------------------------------------- torre de menagem

def face_torre(nuv, off):
    nuv = np.array(nuv, float)
    tuv = np.array([nuv[1], -nuv[0]])
    o = W(*(nuv * off))
    return Plano(o, EU * tuv[0] + EV * tuv[1], EU * nuv[0] + EV * nuv[1])


FACES = {"ENE": (face_torre((1, 0), AU), AV), "NNO": (face_torre((0, 1), AV), AU),
         "OSO": (face_torre((-1, 0), AU), AV), "SSE": (face_torre((0, -1), AV), AU)}
# vãos por face (s ao longo da face, da esquerda para a direita de quem olha de fora): (s0, s1, z0, z1, tipo)
# tipo: "porta" (arco), "janela" (retangular com verga), "fresta"
VAOS_T = {
    "NNO": [(-0.06, 0.06, 47.0, 48.4, "fresta"), (-0.06, 0.06, 52.5, 53.9, "fresta")],
    "ENE": [(-0.45, 0.45, 54.6, 55.9, "janela"), (-0.06, 0.06, 49.0, 50.4, "fresta")],
    "SSE": [(-0.06, 0.06, 50.0, 51.4, "fresta"), (-0.4, 0.4, 54.9, 56.1, "janela")],
    "OSO": [(-0.06, 0.06, 53.0, 54.4, "fresta")],
}


def parede_vaos(m, f, meia, z0, z1, vaos, mat):
    """Face plana de -meia a meia, de z0 a z1, com buracos retangulares nos vãos (faces em tiras)."""
    xs = sorted({-meia, meia, *[v[0] for v in vaos], *[v[1] for v in vaos]})
    for sa, sb in zip(xs[:-1], xs[1:]):
        sm = (sa + sb) / 2
        cortes = sorted([(v[2], v[3]) for v in vaos if v[0] < sm < v[1]])
        zc = z0
        for za, zb in cortes:
            m.face([f.p(sa, 0, zc), f.p(sb, 0, zc), f.p(sb, 0, za), f.p(sa, 0, za)], mat)
            zc = zb
        m.face([f.p(sa, 0, zc), f.p(sb, 0, zc), f.p(sb, 0, z1), f.p(sa, 0, z1)], mat)


def torre():
    m = Malha(["cantaria_torre", "cantaria_lisa", "sombra", "madeira", "branco", "muro_topo", "ferro"])
    zb = 36.5
    for nome, (f, meia) in FACES.items():
        parede_vaos(m, f, meia, zb, Z_SOLEIRA_T, VAOS_T[nome], "cantaria_torre")
        for s0, s1, z0, z1, tipo in VAOS_T[nome]:
            prof = 0.7 if tipo != "janela" else 0.5
            # fundo e lados do vão
            m.face([f.p(s0, -prof, z0), f.p(s1, -prof, z0), f.p(s1, -prof, z1), f.p(s0, -prof, z1)], "sombra")
            m.face([f.p(s0, 0, z0), f.p(s0, -prof, z0), f.p(s0, -prof, z1), f.p(s0, 0, z1)], "cantaria_lisa")
            m.face([f.p(s1, -prof, z0), f.p(s1, 0, z0), f.p(s1, 0, z1), f.p(s1, -prof, z1)], "cantaria_lisa")
            m.face([f.p(s0, -prof, z0), f.p(s0, 0, z0), f.p(s1, 0, z0), f.p(s1, -prof, z0)], "cantaria_lisa")
            m.face([f.p(s1, -prof, z1), f.p(s1, 0, z1), f.p(s0, 0, z1), f.p(s0, -prof, z1)], "cantaria_lisa")
            if tipo == "janela":
                # verga e peitoril de granito salientes
                m.caixa_face(f, s0 - 0.15, s1 + 0.15, -0.02, 0.06, z1, z1 + 0.22, "cantaria_lisa")
                m.caixa_face(f, s0 - 0.1, s1 + 0.1, -0.02, 0.08, z0 - 0.12, z0, "cantaria_lisa")
        # cunhais de cantaria (silhares alternados, ligeiramente salientes)
        for s in (-meia, meia):
            z = zb + 2.0
            k = 0
            while z < Z_SOLEIRA_T - 0.3:
                h = 0.45
                lg = 0.75 if k % 2 == 0 else 0.45
                sa, sb_ = (s, s - lg) if s > 0 else (s, s + lg)
                m.caixa_face(f, min(sa, sb_), max(sa, sb_), -0.01, 0.025, z, z + h - 0.02, "cantaria_lisa", topo=False)
                z += h
                k += 1
        # merlões: cinco por face contando os dos cunhais (pontos do LiDAR no terraço)
        for sc in np.linspace(-meia + 0.45, meia - 0.45, 5):
            m.caixa_face(f, sc - 0.45, sc + 0.45, -0.55, 0.0, Z_SOLEIRA_T - 0.01, Z_MERLAO_T, "cantaria_torre")
        # parapeito por dentro
        m.face([f.p(meia, -0.55, Z_TERRACO), f.p(-meia, -0.55, Z_TERRACO), f.p(-meia, -0.55, Z_SOLEIRA_T),
                f.p(meia, -0.55, Z_SOLEIRA_T)], "cantaria_torre")
        m.face([f.p(-meia, 0, Z_SOLEIRA_T), f.p(meia, 0, Z_SOLEIRA_T), f.p(meia, -0.55, Z_SOLEIRA_T),
                f.p(-meia, -0.55, Z_SOLEIRA_T)], "muro_topo")
    # cruz pátea em relevo por cima da janela virada a nascente (Wikipédia: «uma cruz pátea sobre uma janela»)
    f, meia = FACES["ENE"]
    zc = 56.75
    m.caixa_face(f, -0.07, 0.07, -0.01, 0.04, zc - 0.3, zc + 0.3, "cantaria_lisa")
    m.caixa_face(f, -0.3, 0.3, -0.01, 0.04, zc - 0.07, zc + 0.07, "cantaria_lisa")
    for (a0, a1, b0, b1) in ((-0.17, 0.17, zc + 0.22, zc + 0.3), (-0.17, 0.17, zc - 0.3, zc - 0.22)):
        m.caixa_face(f, a0, a1, -0.01, 0.04, b0, b1, "cantaria_lisa")
    for (a0, a1) in ((-0.3, -0.22), (0.22, 0.3)):
        m.caixa_face(f, a0, a1, -0.01, 0.04, zc - 0.17, zc + 0.17, "cantaria_lisa")
    # terraço
    m.face([(*W(-AU + 0.55, -AV + 0.55), Z_TERRACO), (*W(AU - 0.55, -AV + 0.55), Z_TERRACO),
            (*W(AU - 0.55, AV - 0.55), Z_TERRACO), (*W(-AU + 0.55, AV - 0.55), Z_TERRACO)], "muro_topo")
    # casota da escada (branca, com cobertura plana; LiDAR: u 0 a 3, v −0,8 a 2,5, topo 61,0)
    m.caixa_uv(0.1, 3.0, -0.8, 2.5, Z_TERRACO, 60.95, "branco")
    m.caixa_uv(0.0, 3.08, -0.88, 2.58, 60.95, 61.05, "branco")
    fporta = Plano(W(1.2, -0.8), EU, -EV)
    m.face([fporta.p(-0.4, 0.01, Z_TERRACO), fporta.p(0.4, 0.01, Z_TERRACO), fporta.p(0.4, 0.01, Z_TERRACO + 1.95),
            fporta.p(-0.4, 0.01, Z_TERRACO + 1.95)], "sombra")
    m.objeto("torre", "torre", GRUPOS["Torre"], uv=2.0, vcor=patine_torre)


# ---------------------------------------------------------------- portas

def porta_principal(m):
    """Porta a poente, no pano oeste entre o cubelo oeste e o muro interior (fotografias 30, 51–53 do Commons:
    arco de volta perfeita de cantaria, degraus e a lápide por cima; o cubelo à direita de quem entra, a rocha à
    esquerda). Posição pelo LiDAR (o adarve continua por cima)."""
    pass


# ---------------------------------------------------------------- terreno

def chao_castelo():
    """Chão dentro do castelo: a superfície do LiDAR sem as células de alvenaria, preenchida por baixo dos muros
    (o MDT dá ao recinto de cima, classificado como edifício, um chão interpolado)."""
    ch = ndi.grey_opening(SUP, size=(29, 29))
    alv = ndi.binary_opening((SUP - ch) > 1.8, iterations=1)
    alv = ndi.binary_dilation(alv, iterations=2)
    z = SUP.copy()
    z[alv] = np.nan
    idx = ndi.distance_transform_edt(np.isnan(z), return_distances=False, return_indices=True)
    return z[tuple(idx)]


CASTELO_PL = None


def construir_terreno():
    passo = 1.0
    xs = np.arange(BASE_C[0] - BASE_R, BASE_C[0] + BASE_R + 1e-6, passo)
    ys = np.arange(BASE_C[1] - BASE_R, BASE_C[1] + BASE_R + 1e-6, passo)
    X, Y = np.meshgrid(xs, ys)
    Z = cota(X, Y)
    # dentro do castelo (e até 1,5 m fora dos muros), o chão vem da superfície sem as alvenarias
    CH = chao_castelo()
    dentro = shapely.contains_xy(CASTELO_PL.buffer(1.5), X, Y)
    Z = np.where(dentro, amostra(CH, SX0, SY0, SP, X, Y), Z)
    # rio: o fundo desce com a distância à margem (o MDT sobre a água é a própria água)
    ag = amostra(AGUA.astype(float), LX0, LY0, LP, X, Y) > 0.5
    dist = ndi.distance_transform_edt(ag) * passo
    Z = np.where(ag, NIVEL - np.clip(0.4 + 0.25 * dist, 0.4, 4.0), Z)
    Z = np.where(~ag & (Z < NIVEL + 0.05), NIVEL + 0.05, Z)
    r = np.hypot(X - BASE_C[0], Y - BASE_C[1])
    ok = r <= BASE_R
    idx = -np.ones(X.shape, int)
    idx[ok] = np.arange(ok.sum())
    V = np.stack([X[ok], Y[ok], Z[ok]], 1)
    I, J = np.nonzero(ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:])
    Fq = np.stack([idx[I, J], idx[I, J + 1], idx[I + 1, J + 1], idx[I + 1, J]], 1)
    mats = ["mato", "relva", "rocha", "saibro", "asfalto", "areia", "balastro"]
    ob = malha_rapida("Terreno", V, Fq, mats[0], GRUPOS["Terreno"], "terreno")
    for mt in mats[1:]:
        ob.data.materials.append(MATS[mt])
    cx = V[Fq].mean(1)
    ci = np.clip(((cx[:, 0] - OX0) / OP).astype(int), 0, ORTO.shape[1] - 1)
    ri = np.clip(((OY1 - cx[:, 1]) / OP).astype(int), 0, ORTO.shape[0] - 1)
    o = ndi.uniform_filter(ORTO, size=(7, 7, 1))[ri, ci]
    lum = o.mean(1)
    sat = o.max(1) - o.min(1)
    ny, nx = ok.shape[0] - 1, ok.shape[1] - 1
    grelha = np.zeros((ny, nx), int)

    def manchas(sel, minimo=10, sig=1.0):
        g = np.zeros((ny, nx), float)
        g[I, J] = sel
        g = ndi.gaussian_filter(g, sig) > 0.5
        lab_, k_ = ndi.label(g)
        if k_:
            tam = ndi.sum(g, lab_, range(1, k_ + 1))
            g &= np.isin(lab_, 1 + np.nonzero(tam >= minimo)[0])
        return g[I, J]

    mi = np.zeros(len(Fq), np.int32)
    verde = (o[:, 1] > o[:, 0] * 1.02) & (o[:, 1] > o[:, 2] * 1.05)
    mi[manchas(verde, 30)] = 1
    # rocha: penedos de granito (relevo local acentuado no MDT, ou cinzento claro e pouco saturado no ortofotomapa)
    rel = ndi.gaussian_filter(MDT - ndi.gaussian_filter(MDT, 3.0), 0.7)
    rl = amostra(rel, LX0, LY0, LP, cx[:, 0], cx[:, 1])
    decl = amostra(np.hypot(*np.gradient(ndi.gaussian_filter(MDT, 1.0), LP)), LX0, LY0, LP, cx[:, 0], cx[:, 1])
    rocha = ((rl > 0.3) | (decl > 1.1) | ((lum > 0.62) & (sat < 0.08))) & (cx[:, 2] > NIVEL + 0.2)
    mi[manchas(rocha, 8, 1.1)] = 2
    areia = (lum > 0.62) & (sat > 0.10) & (sat < 0.25) & (o[:, 0] > o[:, 2] * 1.12) & (cx[:, 2] < NIVEL + 2.5)
    mi[manchas(areia, 25)] = 5
    # parque de estacionamento e estrada do cais, na margem norte a nascente (ortofotomapa 2025)
    asf = (lum > 0.50) & (sat < 0.07) & (cx[:, 0] > -5) & (cx[:, 1] > 70) & (cx[:, 1] < 145)
    mi[manchas(asf, 40, 1.2)] = 4
    # linha de caminho de ferro (ortofotomapa: faixa cinzenta a norte, y 120–170)
    lin = (cx[:, 1] > 110) & (cx[:, 1] < 180) & (lum > 0.40) & (lum < 0.60) & (sat < 0.07)
    mi[manchas(lin & ~manchas(asf, 40, 1.2), 60, 1.2)] = 6
    # dentro do castelo: saibro nos recintos, rocha onde o chão sobressai
    dc = shapely.contains_xy(CASTELO_PL, cx[:, 0], cx[:, 1])
    # à volta dos muros (até 4 m), chão de terra batida e saibro do caminho de ronda exterior (fotografias aéreas)
    aro = shapely.contains_xy(CASTELO_PL.buffer(4.0), cx[:, 0], cx[:, 1]) & ~dc & (mi != 2)
    mi[aro] = 3
    mi[dc] = 3
    rc = amostra(ndi.gaussian_filter(CH - ndi.gaussian_filter(CH, 6.0), 1.0), SX0, SY0, SP, cx[:, 0], cx[:, 1])
    dch = amostra(np.hypot(*np.gradient(ndi.gaussian_filter(CH, 1.0), SP)), SX0, SY0, SP, cx[:, 0], cx[:, 1])
    # rocha aflorante: o penedo por baixo da torre e do muro interior (fotografias de dentro e aéreas)
    mi[dc & manchas((rc > 0.35) | (dch > 0.75), 4, 0.6)] = 2
    ob.data.polygons.foreach_set("material_index", mi)
    uv_mundo(ob, 2.0)
    # simplifica longe do castelo (as zonas planas e o fundo do rio ficam com poucos triângulos)
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = 0.42
    mod.use_collapse_triangulate = True
    vg = ob.vertex_groups.new(name="perto")
    dperto = np.hypot(V[:, 0] - 10, V[:, 1] + 2)
    vg.add([int(i) for i in np.nonzero(dperto < 45)[0]], 1.0, "REPLACE")
    mod.vertex_group = "perto"
    mod.invert_vertex_group = True
    mod.vertex_group_factor = 1.0
    dg = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear()
    ob.data = me2
    MEDIDAS["terreno_triangulos"] = len(me2.polygons)
    # lados da base redonda
    m = Malha(["base"])
    zb = NIVEL - 6.0
    ang = np.linspace(0, 2 * math.pi, 361)
    borda = []
    for a in ang:
        p = BASE_C + (BASE_R - 0.5) * np.array([math.cos(a), math.sin(a)])
        z = float(cota(*p))
        z = NIVEL - 2.0 if amostra(AGUA.astype(float), LX0, LY0, LP, [p[0]], [p[1]])[0] > 0.5 else max(z, NIVEL + 0.05)
        borda.append((p[0], p[1], z))
    for a, b in zip(borda[:-1], borda[1:]):
        m.face([(a[0], a[1], zb), (b[0], b[1], zb), b, a], 0)
    m.face([(p[0], p[1], zb) for p in borda[:-1]][::-1], 0)
    m.objeto("base", None, GRUPOS["Terreno"], uv=None)


def construir_rio():
    """O Tejo: plano à cota NIVEL sobre a água do LiDAR (e 2 m para dentro das margens), recortado pela base."""
    ag = ndi.binary_dilation(AGUA, iterations=4)
    passo = 2.0
    xs = np.arange(BASE_C[0] - BASE_R, BASE_C[0] + BASE_R + 1e-6, passo)
    ys = np.arange(BASE_C[1] - BASE_R, BASE_C[1] + BASE_R + 1e-6, passo)
    X, Y = np.meshgrid(xs, ys)
    usa = (amostra(ag.astype(float), LX0, LY0, LP, X, Y) > 0.01) & (np.hypot(X - BASE_C[0], Y - BASE_C[1]) < BASE_R - 0.6)
    idx = -np.ones(X.shape, int)
    idx[usa] = np.arange(usa.sum())
    V = np.stack([X[usa], Y[usa], np.full(usa.sum(), NIVEL)], 1)
    I, J = np.nonzero(usa[:-1, :-1] & usa[1:, :-1] & usa[:-1, 1:] & usa[1:, 1:])
    Fq = np.stack([idx[I, J], idx[I, J + 1], idx[I + 1, J + 1], idx[I + 1, J]], 1)
    ob = malha_rapida("Terreno_rio", V, Fq, "agua", GRUPOS["Terreno"], "rio")
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = 0.25
    dg = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear()
    ob.data = me2
    me2.materials.clear()
    me2.materials.append(MATS["agua"])


# ---------------------------------------------------------------- envolvente e árvores

def construir_envolvente():
    """Edifícios da margem norte (bar e apoio do cais): manchas da classe 6 com mais de 2,5 m, longe do castelo,
    em volumes cinzentos de cobertura plana à cota mediana do MDS."""
    from rasterio import features
    from rasterio.transform import Affine
    h = MDS - MDT
    ed = ndi.binary_opening((CLS == 6) & (h > 2.5), iterations=1)
    lab, k = ndi.label(ed)
    tr = Affine(LP, 0, LX0 - LP / 2, 0, LP, LY0 - LP / 2)
    m = Malha(["cinza_viz"])
    n = 0
    for geo, val in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=tr):
        p = shapely.geometry.shape(geo).buffer(0).simplify(0.5)
        if p.area < 15 or p.distance(CASTELO_PL) < 5 or p.centroid.distance(shapely.Point(*BASE_C)) > BASE_R - 3:
            continue
        p = p.buffer(0.4, join_style="mitre").buffer(-0.4, join_style="mitre")
        if p.geom_type != "Polygon":
            continue
        base = np.array(p.exterior.coords)[:-1]
        if not shapely.Polygon(base).exterior.is_ccw:
            base = base[::-1]
        b = shapely.Polygon(base)
        gx, gy = np.meshgrid(np.arange(*b.bounds[0::2], 0.5), np.arange(*b.bounds[1::2], 0.5))
        di = shapely.contains_xy(b, gx, gy)
        if di.sum() < 4:
            continue
        zs = amostra(MDS, LX0, LY0, LP, gx[di], gy[di])
        zt = amostra(MDT, LX0, LY0, LP, gx[di], gy[di])
        m.prisma(base, float(np.percentile(zt, 10)) - 0.5, float(np.percentile(zs, 70)), 0)
        n += 1
    # pontão de desembarque na margem sul da ilha (ortofotomapa 2025: cais em T de madeira, c. 13 m)
    for q in ([(-70.0, -31.0), (-68.2, -31.0), (-68.2, -42.0), (-70.0, -42.0)],
              [(-74.5, -42.0), (-63.5, -42.0), (-63.5, -44.4), (-74.5, -44.4)]):
        m.prisma(q[::-1], NIVEL - 1.0, NIVEL + 0.8, 0)
    n += 1
    MEDIDAS["edificios_margem"] = n
    if n:
        m.objeto("envolvente", "envolvente", GRUPOS["Envolvente"], uv=2.0)


def construir_arvores():
    """Árvores: máximos locais da vegetação (classes 3 a 5) dentro da base, fora do castelo; copas em dois lobos."""
    h = MDS - MDT
    veg = (CLS >= 3) & (CLS <= 5) & (h > 2.0)
    veg = ndi.binary_opening(veg, iterations=1)
    hs = ndi.gaussian_filter(np.where(veg, h, 0.0), 1.5)
    topo = (hs == ndi.maximum_filter(hs, size=11)) & veg & (hs > 2.5)
    lab, k = ndi.label(topo)
    V, Fc, C = [], [], []
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
    v0 = np.array([x.co[:] for x in bm.verts])
    f0 = [[x.index for x in fa.verts] for fa in bm.faces]
    bm.free()
    mon = CASTELO_PL.buffer(1.0)
    n = 0
    for (r, c) in ndi.center_of_mass(topo, lab, range(1, k + 1)):
        r, c = int(round(r)), int(round(c))
        x, y = LX0 + c * LP, LY0 + r * LP
        if math.hypot(x - BASE_C[0], y - BASE_C[1]) > BASE_R - 3:
            continue
        if mon.contains(shapely.Point(x, y)):
            continue
        zt, zc = MDT[r, c], MDS[r, c]
        alt = zc - zt
        rad = float(np.clip(np.sqrt(veg[max(r - 8, 0):r + 9, max(c - 8, 0):c + 9].sum() * 0.25 / math.pi), 1.2, 4.8))
        # junto ao castelo, a copa não pode entrar no muro
        dmon = CASTELO_PL.distance(shapely.Point(x, y))
        rad = min(rad, max(dmon - 0.3, 0.8))
        tom = np.array([0.20, 0.30, 0.13]) * rng.uniform(0.85, 1.15)
        for (dx, dy, dz, s) in ((0, 0, -0.35, 1.0), (0.35, -0.25, -0.55, 0.75)):
            b = len(V)
            pts = (v0 + rng.normal(0, 0.06, v0.shape)) * np.array([rad * s, rad * s, alt * 0.32 * s]) + \
                np.array([x + dx * rad, y + dy * rad, zc + dz * alt * 0.45])
            V.extend(map(tuple, pts))
            Fc.extend([[b + i for i in fa] for fa in f0])
            tt = np.clip((pts[:, 2] - zt) / max(alt, 0.1), 0, 1)
            C.extend(map(tuple, tom[None, :] * (0.55 + 0.5 * tt)[:, None]))
        b = len(V)
        for j in range(5):
            a = 2 * math.pi * j / 5
            V.append((x + 0.18 * math.cos(a), y + 0.18 * math.sin(a), zt - 0.2))
            V.append((x + 0.13 * math.cos(a), y + 0.13 * math.sin(a), zc - alt * 0.45))
        for j in range(5):
            jj = (j + 1) % 5
            Fc.append([b + 2 * j, b + 2 * jj, b + 2 * jj + 1, b + 2 * j + 1])
        C.extend([(0.33, 0.27, 0.21)] * 10)
        n += 1
    print("árvores", n)
    MEDIDAS["arvores"] = n
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

print("muralhas…")
INFO = muralhas()
_cont = []
for k in ("oeste_sul", "oeste_norte", "norte", "nordeste", "leste", "sul"):
    _cont.extend(map(tuple, INFO[k]["ext"]))
CASTELO_PL = shapely.Polygon(_cont).buffer(0)
CASTELO_PL = shapely.union_all([CASTELO_PL] + [shapely.Point(*t["c"]).buffer(t["r"]) for t in TORRES.values()]).buffer(0.2)
if CASTELO_PL.geom_type == "MultiPolygon":
    CASTELO_PL = max(CASTELO_PL.geoms, key=lambda g: g.area)


def terracos_escavacao():
    """Recinto de baixo: a escavação arqueológica deixou o chão em socalcos (38,1 / 36,6 / 34,4 / 33,0 no LiDAR,
    com saltos de 1,5–3,6 m). Cada salto leva um muro de suporte de alvenaria e, em cima, uma guarda de ferro
    (fotografias de dentro: guardas de ferro à volta das zonas escavadas e muros baixos de pedra)."""
    from skimage import measure
    CH = chao_castelo()
    rb = shapely.Polygon([(-8.6, -4.4), (-2, -4.3), (4.6, -2), (6.6, -4.2), (8.8, -5), (10.4, -3.2), (11.4, -0.4),
                          (13.6, 2.4), (17, 4.2), (20.4, 5.4), (23.4, 7.6), (33.4, 4.4), (17.4, -18.4), (-13, -13.6)])
    rb = rb.intersection(CASTELO_PL.buffer(-2.4))
    rb = rb.difference(shapely.LineString([(-8.6, -4.4), (-2, -4.3), (4.6, -2), (6.6, -4.2), (8.8, -5), (10.4, -3.2),
                                           (11.4, -0.4), (13.6, 2.4), (17, 4.2), (20.4, 5.4), (23.4, 7.6)]).buffer(1.6))
    rb = rb.difference(shapely.Polygon([W(-AU, -AV), W(AU, -AV), W(AU, AV), W(-AU, AV)]).buffer(1.0))
    for t in TORRES.values():
        rb = rb.difference(shapely.Point(*t["c"]).buffer(t["r"] + 1.0))
    G = np.hypot(*np.gradient(CH, SP))

    def ch(x, y):
        return amostra(CH, SX0, SY0, SP, np.atleast_1d(x), np.atleast_1d(y))

    m = Malha(["muro", "muro_topo", "ferro"])
    linhas = []
    for lev in (37.35, 35.6, 33.8):
        for cont in measure.find_contours(CH, lev):
            xy = np.stack([SX0 + cont[:, 1] * SP, SY0 + cont[:, 0] * SP], 1)
            ok = shapely.contains_xy(rb, xy[:, 0], xy[:, 1]) & (amostra(G, SX0, SY0, SP, xy[:, 0], xy[:, 1]) > 1.2)
            lab, k = ndi.label(ok)
            for j in range(1, k + 1):
                seg = xy[lab == j]
                if len(seg) < 4:
                    continue
                ls = shapely.LineString(seg).simplify(0.25)
                if ls.length < 1.5:
                    continue
                linhas.append(np.array(ls.coords))
    n_m = 0
    for P in linhas:
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1)[:, None] + 1e-9
        N_ = np.stack([-T[:, 1], T[:, 0]], 1)
        za, zb = ch(*(P + N_ * 0.9).T), ch(*(P - N_ * 0.9).T)
        cima = np.where((za > zb)[:, None], N_, -N_)        # normal para o lado de cima
        zup = suave(np.maximum(za, zb), 3, 1.0)
        zlo = np.minimum(za, zb) - 0.3
        e = 0.3
        A, B = P + cima * e, P - cima * e                   # face de cima (tardoz) e face à vista
        for i in range(len(P) - 1):
            m.face([(*B[i], zlo[i]), (*B[i + 1], zlo[i + 1]), (*B[i + 1], zup[i + 1] + 0.1), (*B[i], zup[i] + 0.1)], "muro")
            m.face([(*A[i + 1], zup[i + 1] - 0.4), (*A[i], zup[i] - 0.4), (*A[i], zup[i] + 0.1), (*A[i + 1], zup[i + 1] + 0.1)], "muro")
            m.face([(*B[i], zup[i] + 0.1), (*B[i + 1], zup[i + 1] + 0.1), (*A[i + 1], zup[i + 1] + 0.1), (*A[i], zup[i] + 0.1)], "muro_topo")
        guarda(m, [np.array([*(P[i] + cima[i] * 0.1), zup[i] + 0.1]) for i in range(len(P))], "ferro", alt=1.0, passo=1.4,
               r=0.022, travessas=(0.5,))
        n_m += 1
    MEDIDAS["escavacao"] = {"socalcos": n_m, "comprimento_m": round(float(sum(shapely.LineString(P).length for P in linhas)), 1)}
    if n_m:
        m.objeto("escavacao", "escavacao", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)


print("torre…")
torre()
terracos_escavacao()
print("terreno…")
construir_terreno()
construir_rio()
print("envolvente…")
construir_envolvente()
construir_arvores()


def construir_rochedos():
    """Penedos de granito à volta do castelo e no corredor da porta principal (fotografias aéreas e da porta):
    máximos do relevo local do MDT (penedos com mais de 0,6 m) até 30 m dos muros, a pelo menos 2,5 m uns dos outros,
    em blocos arredondados achatados com a textura de rocha."""
    rel = MDT - ndi.gaussian_filter(MDT, 4.0)
    mx = (rel == ndi.maximum_filter(rel, size=7)) & (rel > 0.6)
    rr, cc = np.nonzero(mx)
    xs, ys = LX0 + cc * LP, LY0 + rr * LP
    d = shapely.distance(CASTELO_PL, shapely.points(xs, ys))
    sel = (d > 1.0) & (d < 30) & ~AGUA[rr, cc]
    cand = sorted(zip(rel[rr, cc][sel], xs[sel], ys[sel]), reverse=True)
    # penedos do corredor da porta principal (lado norte, ao pé do cubelo oeste; fotografias 80–82)
    fixos = [(1.4, -11.4, -2.9, 1.1), (1.1, -10.1, -3.0, 0.8), (1.0, -12.8, -3.3, 1.0)]
    escolhidos = []
    for (rl, x, y) in cand:
        if all(math.hypot(x - a, y - b) > 2.5 for (_, a, b, _) in escolhidos + fixos):
            escolhidos.append((rl, x, y, float(np.clip(rl * 1.6, 0.8, 2.4))))
        if len(escolhidos) >= 70:
            break
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    v0 = np.array([x.co[:] for x in bm.verts])
    f0 = [[x.index for x in fa.verts] for fa in bm.faces]
    bm.free()
    V, F = [], []
    for (rl, x, y, rad) in escolhidos + fixos:
        z0 = float(np.atleast_1d(cota([x], [y]))[0])
        g = rng.normal(0, 0.12, v0.shape)
        esc = np.array([rad * rng.uniform(0.9, 1.3), rad * rng.uniform(0.8, 1.1), max(rl, 0.7) * 0.9])
        ang = rng.uniform(0, math.pi)
        R2 = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
        pts = ((v0 + g) * esc) @ R2.T + np.array([x, y, z0 - max(rl, 0.7) * 0.25])
        b = len(V)
        V.extend(map(tuple, pts))
        F.extend([[b + i for i in fa] for fa in f0])
    me = bpy.data.meshes.new("rochedos")
    me.from_pydata(V, [], F)
    me.update()
    me.materials.append(MATS["rocha"])
    me.shade_smooth()
    ob = bpy.data.objects.new("rochedos", me)
    bpy.context.scene.collection.objects.link(ob)
    uv_mundo(ob, 2.0)
    ob["elemento"] = "rochedos"
    ob.parent = GRUPOS["Rochedos"]
    MEDIDAS["rochedos"] = len(escolhidos) + len(fixos)


construir_rochedos()

PT = np.load(AQUI / "dados" / "castelo.npz")["pontos"]
uvp = np.stack([(PT[:, :2] - W(0, 0)) @ EU, (PT[:, :2] - W(0, 0)) @ EV], 1)
dentro = (np.abs(uvp[:, 0]) < AU) & (np.abs(uvp[:, 1]) < AV) & ~np.isin(PT[:, 3], (3, 4, 5))
MEDIDAS["torre"] = {"ponto_mais_alto": round(float(PT[dentro, 2].max()), 2), "planta": [2 * AU, 2 * AV],
                    "terraco": Z_TERRACO, "merloes": Z_MERLAO_T, "chao": CHAO_T,
                    "altura_terraco_recinto_norte": round(Z_TERRACO - CHAO_T["NNO"], 2),
                    "altura_merloes_recinto_sul": round(Z_MERLAO_T - CHAO_T["SSE"], 2), "azimute_faces": [60.7, 150.7]}
MEDIDAS["castelo"] = {"area_m2": round(CASTELO_PL.area), "perimetro_m": round(CASTELO_PL.exterior.length)}
MEDIDAS["rio_nivel"] = NIVEL
MEDIDAS["base"] = {"centro": BASE_C.tolist(), "raio": BASE_R}


def _olhar(x, y, up=0.35):
    d = unit((x, y))
    return [round(float(d[0]), 2), up, round(float(-d[1]), 2)]


MEDIDAS["ancoras"] = {
    "torre": _anc(*W(0, 0), 62.5),
    "cerca": _anc(*TORRES["norte"]["c"], 52.5),
    "cubelos": _anc(*TORRES["nordeste"]["c"], 52.5),
}
MEDIDAS["texturas_foto"] = fontes_fotos()
(AQUI / "dados" / "medidas.json").write_text(json.dumps(MEDIDAS, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in MEDIDAS.items() if k not in ("ancoras", "olhar", "texturas_foto")}, ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
(AQUI / "saida").mkdir(exist_ok=True)
glb = AQUI / "saida" / "almourol-bruto.glb"
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
    final = SAIDA / "almourol.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "almourol.blend"))

# ---------------------------------------------------------------- capa e vistas de verificação

if "--vistas" in sys.argv:
    from mathutils import Vector
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--amostras=")), 24))
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
    d = Vector((math.sin(math.radians(AZ_SOL)), math.cos(math.radians(AZ_SOL)), 0)) * math.cos(math.radians(50))
    d.z = math.sin(math.radians(50))
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

    vista("capa", (95, -80), 105, (8, -2), 42, lente=36, res=(1280, 800), destino=SAIDA / "capa.jpg")
    if "--capa" not in sys.argv:
        # u7: drone a poente, alto, a olhar para nascente (foto 7 do Commons, "Bridging Photos Sky View")
        vista("u7_poente", (-62, -10), 95, (8, 0), 42, lente=30, res=(1000, 760))
        # 86caaf7d: drone a sul-sudoeste, a olhar para NNE (recinto de baixo à frente)
        vista("f86_sul", (-10, -70), 92, (8, 0), 42, lente=32, res=(1140, 700))
        # 1de3e527: drone a sueste
        vista("f1de_sueste", (75, -62), 80, (6, -2), 40, lente=30, res=(1280, 720))
        # 8d3e3af7: de poente, de longe, sobre a ponta rochosa da ilha
        vista("f8d_poente", (-210, -40), 95, (0, -5), 30, lente=40, res=(1030, 772))
        # eb55b599: do rio, a norte-nordeste, ao nível da água
        vista("feb_rio", (60, 120), 16, (5, 0), 38, lente=40, res=(1000, 750))
        vista("topo_castelo", (6, -2), 160, (6, -1.99), 30, lente=30, res=(1000, 1000))
        vista("porta_poente", (-16.5, -4.8), 39.3, (-9.0, -5.15), 39.2, lente=30, res=(1000, 750))
        vista("f48_escada", (0.5, 0.5), 66.0, (20, 8), 44, lente=24, res=(1200, 700))
        vista("norte_baixo", (6, 55), 24, (8, 8), 38, lente=30, res=(1200, 750))
        vista("leste_baixo", (60, -6), 22, (20, -5), 34, lente=30, res=(1200, 750))
        vista("f8e_recinto", (5.5, -7.0), 47.0, (22, -9), 36.0, lente=20, res=(1360, 766))
        vista("portas_recinto", (6, -16), 44.5, (4, -1), 40.5, lente=24, res=(1200, 750))
        vista("geral", (-120, -260), 260, (-120, -10), 15, lente=30, res=(1280, 900))
