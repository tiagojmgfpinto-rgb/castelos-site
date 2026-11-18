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
from kit.arquitetura import aduelas, vao, guarda, barra  # noqa: E402

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
material("terra", (0.50, 0.42, 0.31), 0.97)          # fundo do fosso escavado ao longo do pano leste
# rampa calcetada de seixos até à porta de nascente (fotografias da rampa): a foto old_stone_wall reduzida a metade
# (pedras de 15–25 cm), cinzenta-ocre, cada imagem cobre 2 m
TEX, ALT = textura_foto("old_stone_wall", n=512, metros=2.0, metros_foto=1.0, cor=(0.44, 0.40, 0.35), contraste=1.35,
                        saturacao=0.45, variacao=0.08, semente=35)
material("calcada", (1, 1, 1), 0.9, imagem("calcada", TEX), normal=imagem("calcada_n", normais(ALT, 2.0)), forca=0.8, escala=1.0)
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


def _patamares(v, tol=0.55, minimo=7):
    """Valores por estação -> patamares nivelados: troços seguidos que não se afastam mais de `tol` da mediana ficam
    à mesma cota; troços com menos de `minimo` estações (3,5 m) juntam-se ao vizinho de cota mais próxima. Os panos
    reais têm a linha dos merlões e o adarve horizontais, com saltos onde o pano muda de cota."""
    v = np.asarray(v, float)
    runs = [[0]]
    for i in range(1, len(v)):
        if abs(v[i] - np.median(v[runs[-1]])) <= tol:
            runs[-1].append(i)
        else:
            runs.append([i])
    while len(runs) > 1:
        curtos = [j for j, r in enumerate(runs) if len(r) < minimo]
        if not curtos:
            break
        j = min(curtos, key=lambda j: len(runs[j]))
        med = np.median(v[runs[j]])
        viz = [k for k in (j - 1, j + 1) if 0 <= k < len(runs)]
        k = min(viz, key=lambda k: abs(np.median(v[runs[k]]) - med))
        a, b = sorted((j, k))
        runs[a] = runs[a] + runs[b]
        del runs[b]
    out = np.empty(len(v))
    for r in runs:
        out[r] = np.median(v[r])
    return out


def _endireitar(ext, n, tol=0.45, P=None):
    """Linha da face com o traçado reto: Douglas-Peucker sobre as estações (tolerância 0,45 m), ou a linha P já
    feita. Devolve as novas estações (as antigas reprojetadas por abcissa mais os cantos da linha reta, que passam a
    ser estações), as normais dos troços retos (com o sentido da original) em esquadria nos cantos, para as faces de
    dentro e de fora se encontrarem num canto vivo, e o índice fracionário de cada estação nova nas antigas."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ext, axis=0), axis=1))])
    fi0 = np.arange(len(ext), dtype=float)
    if s[-1] < 1.0:
        return ext, n, fi0
    if P is None:
        P = np.array(shapely.LineString(ext).simplify(tol, preserve_topology=False).coords)
    sp = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    q = s / s[-1] * sp[-1]
    cantos = sp[1:-1]
    longe = np.array([np.all(np.abs(cantos - x) > 0.08) for x in q]) if len(cantos) else np.ones(len(q), bool)
    longe[0] = longe[-1] = True
    S = np.unique(np.concatenate([q[longe], cantos]))
    novo = np.stack([np.interp(S, sp, P[:, 0]), np.interp(S, sp, P[:, 1])], 1)
    T = np.diff(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None] + 1e-12
    NS = np.stack([-T[:, 1], T[:, 0]], 1)                # normal à esquerda de cada troço
    ref = np.stack([np.interp(S, q, n[:, 0]), np.interp(S, q, n[:, 1])], 1)
    seg = np.clip(np.searchsorted(sp, S, side="right") - 1, 0, len(T) - 1)
    sg = 1.0 if np.sum(NS[seg] * ref) >= 0 else -1.0
    nn = NS[seg] * sg
    for j, x in enumerate(cantos):
        k = int(np.argmin(np.abs(S - x)))
        b = NS[j] + NS[j + 1]
        b /= np.linalg.norm(b) + 1e-12
        nn[k] = sg * b / max(float(b @ NS[j]), 0.5)
    return novo, nn, np.interp(S, q, fi0)


def _linha_paralela(c, d, n0, tol=0.45):
    """Linha à distância d do eixo c (já endireitado), do lado da normal n0, com cantos em esquadria."""
    eixo = shapely.LineString(c).simplify(tol, preserve_topology=False)
    T = np.diff(np.array(eixo.coords)[:2], axis=0)[0]
    lado = 1.0 if (-T[1] * n0[0] + T[0] * n0[1]) >= 0 else -1.0
    L = eixo.offset_curve(lado * d, join_style="mitre", mitre_limit=4.0)
    if L.geom_type != "LineString":
        L = max(L.geoms, key=lambda g: g.length)
    P = np.array(L.coords)
    if np.linalg.norm(P[0] - c[0]) > np.linalg.norm(P[-1] - c[0]):
        P = P[::-1]
    return P


PONTAS_TORRE = {}   # pano -> (ponta inicial, ponta final) prolongadas até à torre de menagem (sem parapeito)


def _prolongar_ate_torre(ext, n, esp):
    """Pano que acaba junto à torre de menagem: prolonga a ponta pela tangente até as duas faces entrarem 5 cm na
    torre (o LiDAR perde o pano rente à torre e a muralha ficava aberta)."""
    novos = []
    for ponta, viz in ((0, 1), (len(ext) - 1, len(ext) - 2)):
        if not dentro_torre_menagem(ext[ponta] - n[ponta] * esp / 2, 1.5):
            novos.append(None)
            continue
        t = unit(ext[ponta] - ext[viz])
        nu = unit(n[ponta])
        p = None
        for d in np.arange(0.0, 3.01, 0.05):
            a = ext[ponta] + t * d
            if dentro_torre_menagem(a, -0.05) and dentro_torre_menagem(a - nu * esp, -0.05):
                p = a
                break
        if p is None:
            # pano quase paralelo à face da torre (o pano interior de nascente sai do cunhal sueste no alinhamento da
            # face nascente): a ponta encosta-se à torre na perpendicular dessa face
            pts = [ext[ponta], ext[ponta] - nu * esp]
            uv = [(q - W(0, 0)) @ EU for q in pts], [(q - W(0, 0)) @ EV for q in pts]
            ex_u = max(abs(x) for x in uv[0]) - AU
            ex_v = max(abs(x) for x in uv[1]) - AV
            if ex_u >= ex_v:
                eixo, sg, ex = EU, np.sign(np.mean(uv[0])), max(abs(x) - AU for x in uv[0])
            else:
                eixo, sg, ex = EV, np.sign(np.mean(uv[1])), max(abs(x) - AV for x in uv[1])
            if 0 < ex + 0.1 < 2.0:
                p = ext[ponta] - eixo * sg * (ex + 0.1)
                d = ex + 0.1
        novos.append(p if p is not None and d > 0.02 else None)
    return novos


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
        # muro interior: face a distância constante do eixo, paralela e com cantos vivos (as normais do LiDAR viram
        # nos cantos apertados e a face dobrava sobre si mesma)
        meio = suave(np.clip((fora + dentro) / 2, -1.2, 1.2), 9, 2.0)
        P = _linha_paralela(c, float(np.median(meio)) + esp / 2, n[0])
        ext = c + n * (meio + esp / 2)[:, None]
        ext, n2, fi = _endireitar(ext, n, P=P)
    else:
        f = np.where(torre, np.nan, np.clip(fora, -1.6, 1.6))
        idx = np.arange(len(f))
        ok = ~np.isnan(f)
        f = np.interp(idx, idx[ok], f[ok]) if ok.any() else np.zeros(len(f))
        ext = c + n * suave(f, 9, 2.0)[:, None]
        ext, n2, fi = _endireitar(ext, n)
    tp = np.where(torre, np.nan, topo)
    idx = np.arange(len(tp))
    ok = ~np.isnan(tp)
    tp = np.interp(idx, idx[ok], tp[ok])
    # o LiDAR alterna pontas dos merlões e soleiras das ameias: a linha dos merlões é o máximo em 2,5 m, nivelada
    topo_s = _patamares(ndi.maximum_filter1d(tp, 5, mode="nearest"))
    soleira = topo_s - ALT_MERLAO
    ad = np.where(torre, np.nan, adarve)
    ad = np.interp(idx, idx[~np.isnan(ad)], ad[~np.isnan(ad)])
    ad = _patamares(ndi.median_filter(ad, 9, mode="nearest"), tol=0.45)
    ad = np.minimum(ad, soleira - 0.85)
    ad = np.maximum(ad, soleira - 1.25)
    # tudo nas estações novas (as do LiDAR reprojetadas e os cantos)
    rs = lambda v: np.interp(fi, idx, v)  # noqa: E731
    c = np.stack([rs(c[:, 0]), rs(c[:, 1])], 1)
    soleira, topo_s, ad = rs(soleira), rs(topo_s), rs(ad)
    n = n2
    ini, fim = _prolongar_ate_torre(ext, n, esp)
    PONTAS_TORRE[nome] = (ini is not None, fim is not None)
    if ini is not None:
        c, ext, n = np.vstack([c[:1], c]), np.vstack([ini, ext]), np.vstack([n[:1], n])
        soleira, topo_s, ad = (np.concatenate([v[:1], v]) for v in (soleira, topo_s, ad))
    if fim is not None:
        c, ext, n = np.vstack([c, c[-1:]]), np.vstack([ext, fim]), np.vstack([n, n[-1:]])
        soleira, topo_s, ad = (np.concatenate([v, v[-1:]]) for v in (soleira, topo_s, ad))
    return c, n, ext, esp, soleira, topo_s, ad


# ---------------------------------------------------------------- recinto de baixo: plataformas da escavação
# O chão do recinto de baixo é feito de plataformas planas, com muros de suporte direitos entre elas (planta do painel
# informativo à entrada, Commons «em Julho de 2025» foto 91; fotografias do alto da torre e do caminho). As cotas são
# as do MDS de 50 cm, que no recinto é quase todo chão: o fosso escavado ao longo do pano leste desce em três
# patamares (32,9 / 34,5 / 36,8), uma faixa a 36,6 corre entre o fosso e a rampa calcetada, a rampa sobe de 38,15 a
# 39,7 encostada ao muro interior até à porta de nascente, e a sul ficam duas plataformas a 36,4 e 37,35.
# Referencial do pano leste: u ao longo do pano (de sueste para nordeste), v para dentro do recinto.
EA = math.atan2(4.4 + 18.4, 33.4 - 17.4)
EO = np.array([17.4, -18.4])
EU_ = np.array([math.cos(EA), math.sin(EA)])
EV_ = np.array([-math.sin(EA), math.cos(EA)])


def E(u, v):
    return EO + EU_ * u + EV_ * v


def uv_e(x, y):
    dx, dy = np.asarray(x, float) - EO[0], np.asarray(y, float) - EO[1]
    return dx * EU_[0] + dy * EU_[1], dx * EV_[0] + dy * EV_[1]


def _rect_e(u0, u1, v0, v1):
    return shapely.Polygon([E(u0, v0), E(u1, v0), E(u1, v1), E(u0, v1)])


def _rampa(x, y):
    u, _ = uv_e(x, y)
    return 38.15 + np.clip((u - 6.0) / 9.0, 0.0, 1.0) * 1.55


_contorno = []
for _k in ("interior_oeste", "interior_leste", "nordeste", "leste", "sul", "oeste_sul"):
    _contorno.extend(tuple(x["c"]) for x in MUR["panos"][_k]["estacoes"])
RB = shapely.set_precision(shapely.Polygon(_contorno).buffer(0), 0.01)   # recinto de baixo, pelo eixo dos panos
RAMPA_CALCADA = _rect_e(4.0, 30.0, 9.6, 30.0)        # a parte da rampa que é calçada (fotografias da porta nascente)
_PLAT_SPEC = [
    ("fosso_1", _rect_e(0.6, 11.0, -3.0, 6.6), 32.9),
    ("fosso_2", _rect_e(11.0, 20.4, -3.0, 6.6), 34.5),
    ("fosso_3", _rect_e(20.4, 30.0, -3.0, 6.6), 36.8),
    ("faixa", _rect_e(-2.0, 30.0, 6.6, 9.6), 36.6),
    ("sul", shapely.box(-3.5, -20.0, 16.0, -10.6), 36.4),
    ("poente", shapely.box(-16.0, -20.0, -3.5, -9.3), 37.35),
    ("rampa", RB, _rampa),
]
PLAT = []
_usado = shapely.Polygon()
for _nome, _pol, _z in _PLAT_SPEC:
    # coordenadas presas a uma grelha de 1 cm, para as arestas comuns coincidirem e não sobrarem tiras finas
    _p = shapely.set_precision(shapely.set_precision(_pol, 0.01).intersection(RB), 0.01).difference(_usado)
    _p = shapely.set_precision(_p, 0.01)
    if hasattr(_p, "geoms"):   # restos de menos de 2 m² (cantos debaixo dos cubelos) saem
        _p = shapely.MultiPolygon([g for g in _p.geoms if g.area >= 2.0]) if any(g.area >= 2.0 for g in _p.geoms) else shapely.Polygon()
        if len(getattr(_p, "geoms", [])) == 1:
            _p = _p.geoms[0]
    _usado = _usado.union(_p)
    if not _p.is_empty:
        PLAT.append((_nome, _p, _z))


def z_plat(z, x, y):
    return z(x, y) if callable(z) else np.full(np.shape(x), float(z))


def chao_plat(x, y):
    """Cota do chão do recinto de baixo em (x, y); NaN fora das plataformas."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.full(x.shape, np.nan)
    for _, p, z in PLAT:
        ok = shapely.contains_xy(p, x, y)
        if ok.any():
            out[ok] = z_plat(z, x[ok], y[ok])
    return out


def chao_xy(x, y):
    """Chão onde se pisa: a plataforma do recinto de baixo, se houver, senão o MDT."""
    z = chao_plat(np.atleast_1d(x), np.atleast_1d(y))
    m = np.isnan(z)
    if m.any():
        z[m] = cota(np.atleast_1d(x)[m], np.atleast_1d(y)[m])
    return z


def pe_muro(pts):
    """Cota do pé do muro: o chão mais baixo à volta de cada ponto (o terreno corta o resto)."""
    out = []
    for p in pts:
        zs = [cota(p[0] + dx, p[1] + dy) for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))]
        zp = chao_plat([p[0] + dx for dx in (0, 1.5, -1.5, 0, 0)], [p[1] + dy for dy in (0, 0, 0, 1.5, -1.5)])
        zs += [float(v) for v in zp if not np.isnan(v)]
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
    "oeste_sul": [dict(c=(-9.0, -5.15), larg=1.4, zs=37.95, omb=1.75, degraus=+1, lapide=(1.6, 0.34), lapide_dz=0.75,
                       aduelas=True, aduelas_dentro=True, folha=True)],
    # (a porta no muro interior a poente da torre, que vinha de uma só foto aérea, saiu: a fotografia da porta
    # principal tirada de dentro, Commons «em Julho de 2025» foto 73, mostra ali muro e penedos, sem vão)
    # porta do muro interior a nascente da torre, ao cimo da rampa calcetada que sobe encostada ao muro: três degraus
    # salientes, arco de volta perfeita com aduelas, lápide branca pequena logo por cima do arco, penedos à esquerda e
    # o painel informativo branco ao lado dos degraus (Commons «em Agosto 2025», fotos 21 e 25, e «Castelo de Almourol
    # - Portugal» 245694640 e 3333289147); a quebra do adarve no LiDAR (43,2) fica 1,5 m a sudoeste; a rampa chega a 39,7
    "interior_leste": [dict(c=(14.6, 2.95), larg=1.2, zs=40.3, omb=1.85, degraus=+1, lapide=(0.7, 0.32),
                            lapide_dz=0.40, aduelas=True, painel=-1)],
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
    # fora só as estações dentro dos cubelos e as que têm as duas faces bem dentro da torre de menagem (a ponta
    # prolongada fica 5 cm dentro da torre e fecha o pano)
    keep = np.array([not (perto_torre((a + b) / 2, -0.3) or (dentro_torre_menagem(a, -0.3) and dentro_torre_menagem(b, -0.3)))
                     for a, b in zip(fora_l, dent_l)])
    s_ext = _s_linha(ext)
    ponta_ini, ponta_fim = PONTAS_TORRE.get(nome, (False, False))
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
        if (i == 0 and ponta_ini) or (i == N - 2 and ponta_fim):
            # troço que encosta à torre: sem parapeito, o adarve vai de face a face e o parapeito fecha com um topo
            m.face([(*fora_l[b], sol[b]), (*fora_l[a], sol[a]), (*fora_l[a], ad[a]), (*fora_l[b], ad[b])], "muro")
            m.face([(*dent_l[a], ad[a]), (*fora_l[a], ad[a]), (*fora_l[b], ad[b]), (*dent_l[b], ad[b])], "muro_topo")
            k_ = b if i == 0 else a
            m.face([(*par_l[k_], ad[k_]), (*fora_l[k_], ad[k_]), (*fora_l[k_], sol[k_]), (*par_l[k_], sol[k_])], "muro")
            continue
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
    # merlões ao longo da linha do parapeito, troço reto a troço reto: num canto de fora (convexo) os merlões dos
    # dois troços encontram-se no canto; num canto de dentro param antes, para não se atravessarem
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(fora_l, axis=0), axis=1))])
    tg = np.diff(fora_l, axis=0)
    tg /= np.linalg.norm(tg, axis=1)[:, None] + 1e-12
    cortes = [(s[1], 0.3)] if ponta_ini else [(0.0, 0.5)]
    for i in range(2 if ponta_ini else 1, N - 2 if ponta_fim else N - 1):
        cosv = float(tg[i - 1] @ tg[i])
        if cosv < math.cos(math.radians(12)):
            convexo = float(tg[i] @ unit(npar[i - 1])) < 0
            cortes.append((s[i], 0.0 if convexo else ESP_PAR + 0.08))
    cortes.append((s[-2], 0.3) if ponta_fim else (s[-1], 0.2))
    for (ra, fa), (rb, fb) in zip(cortes[:-1], cortes[1:]):
        a0, a1 = ra + fa, rb - fb
        if a1 - a0 < L_MERLAO:
            continue
        k = int((a1 - a0 + (PERIODO - L_MERLAO)) // PERIODO)
        folga = (a1 - a0) - (k * PERIODO - (PERIODO - L_MERLAO))
        # o primeiro e o último merlão encostam às pontas nos cantos de fora; nos outros casos a folga reparte-se
        if fa == 0.0 and fb == 0.0 and k > 1:
            passo = (a1 - a0 - L_MERLAO) / (k - 1)
            inicios = [a0 + j * passo for j in range(k)]
        elif fa == 0.0:
            inicios = [a0 + j * PERIODO for j in range(k)]
        elif fb == 0.0:
            inicios = [a1 - L_MERLAO - j * PERIODO for j in range(k)][::-1]
        else:
            inicios = [a0 + folga / 2 + j * PERIODO for j in range(k)]
        im = min(int(np.searchsorted(s, (ra + rb) / 2)), N - 1)
        tr = tg[max(im - 1, 0)]
        na = np.array([-tr[1], tr[0]])
        if na @ npar[max(im - 1, 0)] < 0:
            na = -na
        for s0 in inicios:
            s1 = s0 + L_MERLAO
            ia, ib = min(int(np.searchsorted(s, s0)), N - 1), min(int(np.searchsorted(s, s1)), N - 1)
            if not (keep[ia] and keep[ib]):
                continue
            pa = np.array([np.interp(s0, s, fora_l[:, 0]), np.interp(s0, s, fora_l[:, 1])])
            pb = np.array([np.interp(s1, s, fora_l[:, 0]), np.interp(s1, s, fora_l[:, 1])])
            z0 = float(np.interp((s0 + s1) / 2, s, sol))
            z1 = float(np.interp((s0 + s1) / 2, s, topo))
            q = [pa, pb, pb - na * ESP_PAR, pa - na * ESP_PAR]
            m.prisma(q, z0 - 0.02, z1, "muro", topo=True)
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
    lados_ad = [(Fa, Fb, nn)] + ([(Da, Db, -nn)] if d.get("aduelas_dentro") else [])
    for (A0, B0, nlado) in (lados_ad if d.get("aduelas") else []):
        off = nlado * 0.04
        A, B = A0 + off, B0 + off
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
        t = unit(B0 - A0)
        for (P, sg) in ((A0, -1), (B0, 1)):
            q = [P, P + t * sg * 0.3, P + t * sg * 0.3 + off, P + off]
            e1, e2 = q[1] - q[0], q[2] - q[1]
            if e1[0] * e2[1] - e1[1] * e2[0] < 0:
                q = q[::-1]
            m.prisma(q, zs, zm, "cantaria_lisa", topo=False)
    if d.get("lapide"):
        lw, lh = d["lapide"]
        mid = (Fa + Fb) / 2
        t = unit(Fb - Fa)
        q = [mid - t * lw / 2, mid + t * lw / 2, mid + t * lw / 2 + nn * 0.05, mid - t * lw / 2 + nn * 0.05]
        dz = d.get("lapide_dz", 0.45)
        m.prisma(q, za + dz, za + dz + lh, "lapide")
    if d.get("folha"):
        # folha de madeira aberta, encostada à ombreira do lado de dentro
        t = unit(Fb - Fa)
        P = Db - nn * 0.02
        q = [P - t * 0.02, P - t * 0.09, P - t * 0.09 - nn * (s1 - s0), P - t * 0.02 - nn * (s1 - s0)]
        m.prisma(q, zs + 0.02, zm + r * 0.6, "madeira", fundo=True)
    if d.get("degraus"):
        lado = d["degraus"]
        P = ((Fa + Fb) / 2) if lado > 0 else ((Da + Db) / 2)
        t = unit(Fb - Fa)
        nd = np.array([-t[1], t[0]])                 # degraus a direito, perpendiculares ao pano
        nd = nd * np.sign(np.dot(nd, nn * lado))
        q_ = P + nd * 1.6
        chao = float(chao_xy([q_[0]], [q_[1]])[0])
        nd_ = int(round((zs - chao) / 0.17)) - 1 if zs - chao > 0.12 else 0
        for j in range(nd_):
            zt = zs - (j + 1) * (zs - chao) / (nd_ + 1)
            p0, p1 = P + nd * (0.05 + 0.38 * j), P + nd * (0.05 + 0.38 * (j + 1))
            q = [p0 - t * 0.95, p0 + t * 0.95, p1 + t * 0.95, p1 - t * 0.95]
            e1, e2 = q[1] - q[0], q[2] - q[1]
            if e1[0] * e2[1] - e1[1] * e2[0] < 0:
                q = q[::-1]
            m.prisma(q, chao - 0.6, zt, "cantaria_lisa")
        if d.get("painel"):
            # painel informativo: laje branca em pé ao lado dos degraus (fotografias da porta)
            c_ = P + t * d["painel"] * 2.1 + nd * 0.8
            q = [c_ - t * 0.24 - nd * 0.06, c_ + t * 0.24 - nd * 0.06, c_ + t * 0.24 + nd * 0.06, c_ - t * 0.24 + nd * 0.06]
            e1, e2 = q[1] - q[0], q[2] - q[1]
            if e1[0] * e2[1] - e1[1] * e2[0] < 0:
                q = q[::-1]
            zc = float(chao_xy([c_[0]], [c_[1]])[0])
            m.prisma(q, zc - 0.3, zc + 1.7, "lapide")
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
    # porta de entrada da torre, a 3 m do chão do recinto de cima, retangular, com uma escada metálica a subir
    # encostada à face (Commons «em Agosto 2025» foto 31; LiDAR: patamar a 44,5 m de s 0,4 a 1,9, lanço a
    # subir de poente desde s −3,0, chão a 42,0)
    "NNO": [(0.75, 1.65, 44.6, 46.6, "porta"), (-0.06, 0.06, 47.0, 48.4, "fresta"), (-0.06, 0.06, 52.5, 53.9, "fresta")],
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


def escada_torre(m):
    """Escada metálica da porta da torre: um lanço reto encostado à face norte-noroeste, a subir de poente
    para nascente, e um patamar diante da porta; degraus de chapa, duas longarinas, pilares finos e guarda do lado
    de fora e à volta do patamar (Commons «em Agosto 2025» foto 31; cotas do LiDAR a 0,6–1,4 m da face)."""
    f, _ = FACES["NNO"]
    z_ch, z_pat = 42.0, 44.5
    sa, sb = -3.0, 0.45            # pé do lanço e começo do patamar (s ao longo da face)
    d0, d1 = 0.12, 1.08            # largura do lanço (afastamento da face)
    k = 14
    esp_ = (z_pat - z_ch) / k
    cob = (sb - sa) / k
    for j in range(k - 1):
        s0_, z = sa + cob * j, z_ch + esp_ * (j + 1)
        m.caixa_face(f, s0_, s0_ + cob + 0.03, d0, d1, z - 0.035, z, "ferro")
    # longarinas: chapas inclinadas dos dois lados do lanço
    for d in (d0 - 0.02, d1 + 0.02):
        for dd in (0.0, 0.015):
            pts = [f.p(sa, d + dd, z_ch - 0.02), f.p(sb, d + dd, z_pat - 0.05), f.p(sb, d + dd, z_pat + 0.02),
                   f.p(sa, d + dd, z_ch + 0.3)]
            m.face(pts if dd else pts[::-1], "ferro")
    # patamar e pilares
    s_pat0, s_pat1, d_pat = sb, 1.95, 1.15
    m.caixa_face(f, s_pat0, s_pat1, 0.0, d_pat, z_pat - 0.06, z_pat, "ferro")
    for s_ in (s_pat0 + 0.05, s_pat1 - 0.05):
        barra(m, np.array([*f.xy(s_, d_pat - 0.05), z_ch - 0.2]), np.array([*f.xy(s_, d_pat - 0.05), z_pat - 0.06]),
              0.035, "ferro")
    barra(m, np.array([*f.xy((sa + sb) / 2, d1), z_ch - 0.2]),
          np.array([*f.xy((sa + sb) / 2, d1), (z_ch + z_pat) / 2 - 0.1]), 0.03, "ferro")
    # guarda: do lado de fora do lanço e à volta do patamar até à torre
    lanco = [np.array([*f.xy(sa + 0.1, d1 + 0.03), z_ch + esp_]), np.array([*f.xy(sb, d1 + 0.03), z_pat])]
    guarda(m, lanco, "ferro", alt=1.0, passo=1.2, r=0.02, travessas=(0.5,))
    pat = [np.array([*f.xy(sb, d_pat - 0.03), z_pat]), np.array([*f.xy(s_pat1 - 0.03, d_pat - 0.03), z_pat]),
           np.array([*f.xy(s_pat1 - 0.03, 0.05), z_pat])]
    guarda(m, pat, "ferro", alt=1.0, passo=0.75, r=0.02, travessas=(0.5,))
    MEDIDAS["escada_torre"] = {"degraus": k, "espelho_m": round(esp_, 3), "cobertor_m": round(cob, 3),
                               "patamar_m": z_pat, "soleira_porta_m": 44.6}


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
            if tipo == "porta":
                # ombreiras, verga e soleira de granito, ligeiramente salientes
                m.caixa_face(f, s0 - 0.14, s0, -0.02, 0.05, z0, z1, "cantaria_lisa")
                m.caixa_face(f, s1, s1 + 0.14, -0.02, 0.05, z0, z1, "cantaria_lisa")
                m.caixa_face(f, s0 - 0.14, s1 + 0.14, -0.02, 0.05, z1, z1 + 0.32, "cantaria_lisa")
                m.caixa_face(f, s0 - 0.05, s1 + 0.05, -0.02, 0.06, z0 - 0.1, z0, "cantaria_lisa")
                # folha de madeira aberta para dentro, encostada à ombreira de poente
                m.caixa_face(f, s0 + 0.02, s0 + 0.07, -0.68, -0.08, z0 + 0.02, z1 - 0.04, "madeira")
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
    escada_torre(m)
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
    # a escada metálica da porta da torre não é chão: o LiDAR sobe por ela até 44,8 m; o chão diante da face
    # norte-noroeste fica à cota medida 2,2 m para fora, ao longo da face
    f, _ = FACES["NNO"]
    pe = shapely.Polygon([f.xy(-3.8, -0.6), f.xy(2.6, -0.6), f.xy(2.6, 2.2), f.xy(-3.8, 2.2)])
    sob = shapely.contains_xy(pe, X, Y)
    if sob.any():
        ss = (np.stack([X[sob], Y[sob]], 1) - f.o) @ f.t
        zs_ = [float(amostra(CH, SX0, SY0, SP, *f.xy(x, 2.2))) for x in np.arange(-3.8, 2.61, 0.2)]
        Z[sob] = np.minimum(Z[sob], np.interp(ss, np.arange(-3.8, 2.61, 0.2), zs_))
    # recinto de baixo: o terreno fica 0,25 m por baixo das plataformas (e da mais baixa a menos de 1 m), que o tapam
    zp = chao_plat(X, Y)
    zp = ndi.minimum_filter(np.where(np.isnan(zp), np.inf, zp - 0.25), size=3)
    Z = np.where(np.isfinite(zp), zp, Z)
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


def recinto_baixo():
    """Recinto de baixo: plataformas planas (chão de saibro, fosso de terra, rampa calcetada) e, entre elas, muros de
    suporte direitos de alvenaria. Onde o salto passa de 1 m há guarda de ferro; ao longo da rampa o muro sobe 0,75 m
    acima dela e faz de parapeito (fotografias da rampa e do alto da torre). Mais um alicerce baixo a x = 3, que a
    planta do painel desenha a dividir a plataforma sul."""
    m = Malha(["saibro", "terra", "calcada"])
    mw = Malha(["muro", "muro_topo", "ferro"])
    obst = [shapely.Point(*t["c"]).buffer(t["r"] + 0.6) for t in TORRES.values()]
    obst.append(shapely.Polygon([W(-AU, -AV), W(AU, -AV), W(AU, AV), W(-AU, AV)]).buffer(0.6))
    obst = shapely.union_all(obst + [RB.exterior.buffer(1.3)])
    # superfícies: a rampa partida nos joelhos (u = 6 e 15) para as faces serem planas
    cortes = [_rect_e(-60, 6.0, -60, 60), _rect_e(6.0, 15.0, -60, 60), _rect_e(15.0, 60, -60, 60)]
    area = {}
    for nome, pol, z in PLAT:
        pecas = [pol.intersection(c) for c in cortes] if callable(z) else [pol]
        for pc in pecas:
            partes = [(pc.intersection(RAMPA_CALCADA), "calcada"), (pc.difference(RAMPA_CALCADA), "saibro")] \
                if nome == "rampa" else [(pc, "terra" if nome.startswith("fosso") else "saibro")]
            for parte, mat in partes:
                parte = parte.buffer(0)
                if parte.is_empty or parte.area < 0.01:
                    continue
                for tri in shapely.constrained_delaunay_triangles(parte).geoms:
                    P = np.array(tri.exterior.coords)[:3]
                    if tri.area < 1e-6:
                        continue
                    zz = z_plat(z, P[:, 0], P[:, 1])
                    if (P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1]) - (P[1, 1] - P[0, 1]) * (P[2, 0] - P[0, 0]) < 0:
                        P, zz = P[::-1], zz[::-1]
                    m.face([(P[q, 0], P[q, 1], zz[q]) for q in range(3)], mat)
        area[nome] = round(pol.area, 1)
    comp = {"suporte": 0.0, "parapeito": 0.0, "guarda": 0.0}

    def troco(a, t, nout, z, run):
        """Um troço seguido de muro ao longo da aresta: face à vista na aresta, tardoz 0,45 m para dentro."""
        parapeito = run[0][3]
        alto = 0.75 if parapeito else 0.15
        for (s0, s1, zo, _) in run:
            pa, pb = a + t * s0, a + t * s1
            za, zb = float(z_plat(z, *pa)), float(z_plat(z, *pb))
            ia, ib = pa - nout * 0.45, pb - nout * 0.45
            mw.face([(*pa, zo - 0.3), (*pb, zo - 0.3), (*pb, zb + alto), (*pa, za + alto)], "muro")
            mw.face([(*pa, za + alto), (*pb, zb + alto), (*ib, zb + alto), (*ia, za + alto)], "muro_topo")
            mw.face([(*ib, zb - 0.1), (*ia, za - 0.1), (*ia, za + alto), (*ib, zb + alto)], "muro")
        # topos das pontas
        for (s_, fim) in ((run[0][0], False), (run[-1][1], True)):
            p = a + t * s_
            zp_ = float(z_plat(z, *p))
            zo = run[-1][2] if fim else run[0][2]
            q = [(*p, zo - 0.3), (*(p - nout * 0.45), zo - 0.3), (*(p - nout * 0.45), zp_ + alto), (*p, zp_ + alto)]
            mw.face(q[::-1] if fim else q, "muro")
        L = run[-1][1] - run[0][0]
        comp["parapeito" if parapeito else "suporte"] += L
        queda = max(float(z_plat(z, *(a + t * r[0]))) - r[2] for r in run)
        if not parapeito and queda >= 0.9 and L > 0.6:
            pts = [np.array([*(a + t * s_ - nout * 0.22), float(z_plat(z, *(a + t * s_))) + alto])
                   for s_ in np.linspace(run[0][0] + 0.1, run[-1][1] - 0.1, max(2, int(L / 0.5) + 1))]
            guarda(mw, pts, "ferro", alt=1.0, passo=1.4, r=0.022, travessas=(0.5,))
            comp["guarda"] += L

    for nome, pol, z in PLAT:
        for g in (pol.geoms if hasattr(pol, "geoms") else [pol]):
            g = shapely.geometry.polygon.orient(g, 1.0)
            for anel in [g.exterior] + list(g.interiors):
                C = np.array(anel.coords)
                for a, b in zip(C[:-1], C[1:]):
                    L = float(np.linalg.norm(b - a))
                    if L < 0.05:
                        continue
                    t = (b - a) / L
                    nout = np.array([t[1], -t[0]])      # exterior anti-horário: o lado de fora fica à direita
                    k = max(1, int(math.ceil(L / 0.2)))
                    S = np.linspace(0, L, k + 1)
                    Pm = a + t * ((S[:-1] + S[1:]) / 2)[:, None]
                    zin = z_plat(z, Pm[:, 0], Pm[:, 1])
                    zout = chao_plat(Pm[:, 0] + nout[0] * 0.3, Pm[:, 1] + nout[1] * 0.3)
                    livre = ~shapely.contains_xy(obst, Pm[:, 0], Pm[:, 1])
                    sel = livre & ~np.isnan(zout) & (zin - zout > 0.1)
                    calc = shapely.contains_xy(RAMPA_CALCADA.buffer(0.3), Pm[:, 0], Pm[:, 1]) & (nome == "rampa")
                    run = []
                    for i in range(k):
                        if sel[i] and (not run or run[-1][3] == bool(calc[i])):
                            run.append((S[i], S[i + 1], float(zout[i]), bool(calc[i])))
                            continue
                        if run:
                            troco(a, t, nout, z, run)
                        run = [(S[i], S[i + 1], float(zout[i]), bool(calc[i]))] if sel[i] else []
                    if run:
                        troco(a, t, nout, z, run)
    # alicerce baixo a x = 3 na plataforma sul (planta do painel)
    for (pa, pb) in [((3.0, -10.75), (3.0, -15.6))]:
        pa, pb = np.array(pa), np.array(pb)
        t = unit(pb - pa)
        nn = np.array([-t[1], t[0]])
        q = [pa - nn * 0.3, pb - nn * 0.3, pb + nn * 0.3, pa + nn * 0.3]
        if (q[1] - q[0])[0] * (q[2] - q[1])[1] - (q[1] - q[0])[1] * (q[2] - q[1])[0] < 0:
            q = q[::-1]
        mw.prisma(q, 36.1, 36.85, "muro", mat_topo="muro_topo")
    MEDIDAS["recinto_baixo"] = {"plataformas_m2": area, "muros_m": {k: round(v, 1) for k, v in comp.items()},
                                "cotas": {n: (z if not callable(z) else "38,15→39,7") for n, _, z in PLAT}}
    m.objeto("recinto_baixo_chao", "recinto_baixo", GRUPOS["Terreno"], uv=2.0)
    mw.objeto("recinto_baixo_muros", "escavacao", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)


print("torre…")
torre()
recinto_baixo()
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
    # texturas JPEG -> WebP (pipeline/kit/webp.py), sem tocar na geometria Draco
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / "kit" / "webp.py"), str(final)], check=True)
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
