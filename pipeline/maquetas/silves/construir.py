"""Constrói a maqueta 3D do Castelo e da Muralha de Silves (estado atual) em Blender e exporta-a para
public/maquetas/silves/silves.glb (glTF binário com Draco e texturas WebP).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow rasterio
    python pipeline/maquetas/silves/construir.py            # GLB
    python pipeline/maquetas/silves/construir.py --vistas   # GLB + capa + vistas de verificação

Dados (em dados/): lidar.npz, superficie.npz e orto_lidar.jpg (lidar.py: LiDAR 2024 da DGT, folhas 172025, 172026,
173025 e 173026, com a cor RGB dos pontos), muralhas.json (tracar.py: estações medidas nos panos da alcáçova e da
cerca, retângulos das torres), fontes.json (recolher.py: plantas do OSM via Overture).

O que se modela, e de onde vem:
- a alcáçova: nove panos medidos estação a estação (face de fora, espessura, adarve, soleira das ameias, merlões
  retangulares de 1,0 m com ameias de 0,5 m, contados nas fotografias), as dez torres retangulares medidas no LiDAR
  (torre de menagem, do Segredo, das Mulheres, do Centro, da porta e as duas albarrãs com os braços em arco), a
  porta em arco no pano sul-poente com a escada do átrio, a casa da guarda, a cisterna da Moura com as quatro
  abóbadas, a estátua de D. Sancho I e o mastro da bandeira;
- a cerca da vila: os seis troços que sobrevivem (planta do OSM, alturas do LiDAR), as torres que o OSM regista e o
  torreão da Porta da Cidade (Porta de Loulé), com o arco de passagem;
- o chão do LiDAR, a vila em volumes cinzentos (edifícios do LiDAR) e as árvores do LiDAR.
Coordenadas da cena: x para este, y para norte (PT-TM06 menos a origem, que é o centro da alcáçova), z = altitude,
em metros (1:1). O glTF converte para y para cima (-z é o norte no visor).
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
from kit.arquitetura import guarda  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from comum import CENTRO_BASE, RAIO_BASE  # noqa: E402

RAIZ = AQUI.parents[2]
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / "silves.json").read_text())
MUR = json.loads((AQUI / "dados" / "muralhas.json").read_text())
FON = json.loads((AQUI / "dados" / "fontes.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / "silves"
LID = np.load(AQUI / "dados" / "lidar.npz")
LP, LX0, LY0 = float(LID["passo"]), float(LID["x0"]), float(LID["y0"])
MDT = LID["mdt"] / 100.0 + float(LID["base"])
MDS = LID["mds"] / 100.0 + float(LID["base"])
CLS = LID["classe"]
SUPF = np.load(AQUI / "dados" / "superficie.npz")
SP, SX0, SY0 = float(SUPF["passo"]), float(SUPF["x0"]), float(SUPF["y0"])
SUP = SUPF["z"] / 100.0 + float(SUPF["base"])
ORTO = np.asarray(Image.open(AQUI / "dados" / "orto_lidar.jpg").convert("RGB"), float) / 255.0  # 25 cm, norte em cima
OX0, OY1, OP = -360.0, 230.0, 0.25

BASE_C = np.array(CENTRO_BASE)
BASE_R = RAIO_BASE
rng = np.random.default_rng(1189)


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def cota(x, y):
    """Chão (MDT de 50 cm)."""
    return amostra(MDT, LX0, LY0, LP, x, y)


def sup(x, y):
    """Superfície do LiDAR (classes 1, 2 e 6, células de 25 cm)."""
    return amostra(SUP, SX0, SY0, SP, x, y)


def chao_xy(x, y):
    return np.atleast_1d(cota(np.atleast_1d(x), np.atleast_1d(y)))


def ccw(q):
    q = [np.asarray(p, float) for p in q]
    a = sum(q[i][0] * q[(i + 1) % len(q)][1] - q[(i + 1) % len(q)][0] * q[i][1] for i in range(len(q)))
    return q if a > 0 else q[::-1]

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



# ---------------------------------------------------------------- cena

for _ob in list(bpy.data.objects):  # a cena de arranque do bpy traz um cubo, uma luz e uma câmara
    bpy.data.objects.remove(_ob)
RAIZ_OB = bpy.data.objects.new("Castelo_Silves", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = CFG["origem"]["lat"], CFG["origem"]["lon"]
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
# "Muralhas_grupo", "Torre_grupo" e "Interior_grupo" são o que o visor enquadra ao abrir (a alcáçova); a cerca da vila
# fica em "Muralha_vila_grupo" e o resto em Terreno/Vegetacao/Envolvente, todos fora do enquadramento
for g in ["Terreno", "Torre", "Muralhas", "Interior", "Muralha_vila", "Envolvente", "Vegetacao"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

# Pedra: grés vermelho de Silves («grés de Silves»), aparelhado em fiadas irregulares de blocos de 0,2–0,4 m nos panos
# e mais regulares nas torres (fotografias do Commons: 004, 005, 019, 020, 023 e as do Tiago). Cor medida nas fotos ao
# sol: vermelho-acastanhado, mais escuro e saturado que o granito do norte; o AgX do visor tira saturação, por isso a
# cor de base fica um pouco mais quente do que a média das fotos.
TEX, ALT = textura_foto("old_stone_wall", n=1024, metros=4.0, cor=(0.58, 0.33, 0.24), contraste=1.2, saturacao=0.55,
                        variacao=0.10, liquen=0.02, semente=41)
material("muro", (1, 1, 1), 0.92, imagem("muro", TEX), normal=imagem("muro_n", normais(ALT, 4.0)), forca=1.0, escala=0.5, vcor=True)
TEX, ALT = textura_foto("old_stone_wall", n=1024, metros=4.0, cor=(0.60, 0.37, 0.27), contraste=1.1, saturacao=0.5,
                        variacao=0.08, semente=42)
material("muro_topo", (1, 1, 1), 0.95, imagem("muro_topo", TEX), normal=imagem("muro_topo_n", normais(ALT, 4.0)), forca=0.8, escala=0.5, vcor=True)
TEX, ALT = textura_silhares(n=1024, metros=4.0, fiada=(0.28, 0.48), comp=(0.4, 1.0), cor=(0.58, 0.34, 0.25), desvio=0.10,
                            junta_m=0.018, junta_cor=(0.55, 0.43, 0.35), liquen=0.0, irregular=0.4, semente=43)
material("cantaria_torre", (1, 1, 1), 0.9, imagem("cantaria_torre", TEX), normal=imagem("cantaria_torre_n", normais(ALT, 4.0)),
         forca=0.9, escala=0.5, vcor=True)
# cerca da vila: taipa forrada de grés, mais ocre e gasta que a alcáçova (fotografias das muralhas da vila)
TEX, ALT = textura_foto("old_stone_wall", n=1024, metros=4.0, cor=(0.62, 0.42, 0.31), contraste=1.15, saturacao=0.5,
                        variacao=0.12, liquen=0.03, semente=44)
material("cerca", (1, 1, 1), 0.93, imagem("cerca", TEX), normal=imagem("cerca_n", normais(ALT, 4.0)), forca=0.9, escala=0.5, vcor=True)
# torres da cerca: taipa à vista, bege-ocre, sobre embasamento de grés (fotos 081–083)
TEX, ALT = textura_foto("old_stone_wall", n=512, metros=4.0, cor=(0.72, 0.59, 0.44), contraste=0.8, saturacao=0.35,
                        variacao=0.14, semente=46)
material("taipa", (1, 1, 1), 0.95, imagem("taipa", TEX), normal=imagem("taipa_n", normais(ALT, 4.0)), forca=0.5, escala=0.5, vcor=True)
material("cantaria_lisa", (0.60, 0.40, 0.31), 0.85)   # aduelas e ombreiras de grés aparelhado
material("reboco", (0.80, 0.70, 0.56), 0.9)           # remendos rebocados das torres e o pilar da albarrã
material("terraco", (0.62, 0.36, 0.27), 0.9)          # terraços das torres (tijoleira vermelha nas fotos aéreas)
# chão do recinto: lajes irregulares de grés vermelho com juntas de terra (foto 004); a foto seaworn_stone_tiles
# (Poly Haven, CC0) recolorida, cada imagem cobre 4 m
TEX, ALT = textura_foto("seaworn_stone_tiles", n=512, metros=4.0, cor=(0.60, 0.40, 0.31), contraste=1.1, saturacao=0.4,
                        variacao=0.12, semente=47)
material("lajeado", (1, 1, 1), 0.95, imagem("lajeado", TEX), normal=imagem("lajeado_n", normais(ALT, 4.0)), forca=0.5)
material("abobada", (0.80, 0.62, 0.54), 0.9)          # extradorso rebocado das abóbadas da cisterna (orto do LiDAR)
material("sombra", (0.05, 0.045, 0.04), 1.0)
material("ferro", (0.13, 0.13, 0.14), 0.5)
material("madeira", (0.30, 0.20, 0.13), 0.8)          # folhas da porta da alcáçova (foto 101)
material("vidro", (0.12, 0.14, 0.16), 0.2)
material("bronze", (0.26, 0.29, 0.25), 0.45)          # estátua de D. Sancho I (bronze patinado, foto 023)
material("pedestal", (0.72, 0.68, 0.60), 0.8)
material("branco", (0.88, 0.86, 0.81), 0.8)           # edifícios rebocados do recinto
TEX, ALT = textura_foto("roof_09", n=512, metros=4.0, cor=(0.66, 0.36, 0.25), contraste=1.0, saturacao=0.5, semente=45)
material("telha", (1, 1, 1), 0.85, imagem("telha", TEX), normal=imagem("telha_n", normais(ALT, 4.0)), forca=0.7, escala=0.5)
material("bandeira_verde", (0.10, 0.40, 0.18), 0.8)
material("bandeira_vermelha", (0.80, 0.12, 0.12), 0.8)
TEX, ALT = textura_relva()
material("relva", (1, 1, 1), 0.95, imagem("relva", TEX), normal=imagem("relva_n", normais(ALT, 1.0)), forca=0.4)
material("mato", (0.52, 0.47, 0.33), 0.95)
material("terra", (0.62, 0.45, 0.33), 0.97)           # terra vermelha das encostas (grés desfeito)
material("saibro", (0.70, 0.55, 0.42), 0.95)
material("asfalto", (0.40, 0.40, 0.41), 0.95)
material("calcada", (0.66, 0.64, 0.60), 0.95)
material("cinza_viz", (0.74, 0.74, 0.73), 0.9)        # a vila: um só cinzento
material("base", (0.33, 0.24, 0.19), 0.95)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)

MEDIDAS = {}


def patine_muro(co, nrm):
    """Manchas grandes (pedra reposta e velha), pé mais escuro, faces a sul e poente mais quentes, norte mais frio."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x + 500, y + 500 + z * 0.7, 6.0, 41)
    v = 0.86 + 0.26 * g
    sol = np.clip(-nrm[:, 1] * 0.6 - nrm[:, 0] * 0.3, 0, 1)
    tom = np.stack([1.0 + 0.04 * sol, 1.0 - 0.01 * sol, 1.0 - 0.06 * sol], 1)
    norte = np.clip(nrm[:, 1], 0, 1)[:, None] * np.array([-0.06, -0.04, -0.01])
    pe = np.clip(1 - (z - cota(x, y)) / 1.6, 0, 1) * 0.12
    return np.clip((v[:, None] * tom + norte) * (1 - pe)[:, None], 0, 1)


def patine_torre(co, nrm):
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x * 1.3 + 90, y * 1.3 + z * 0.9, 4.0, 43)
    sol = np.clip(-nrm[:, 1] * 0.6 - nrm[:, 0] * 0.3, 0, 1)[:, None]
    tom = sol * np.array([1.03, 0.98, 0.93]) + (1 - sol) * np.array([0.93, 0.92, 0.92])
    return np.clip(tom * (0.9 + 0.18 * g)[:, None], 0, 1)


# ---------------------------------------------------------------- muralhas da alcáçova

PERIODO = 1.5       # merlões retangulares de c. 1,0 m e ameias de c. 0,5 m (contados nas fotos 004 e 023)
L_MERLAO = 1.0
ESP_PAR = 0.5       # espessura do parapeito
ALT_MERLAO = 1.0    # da soleira das ameias à ponta (LiDAR: topo − adarve 1,9–2,0 m; parapeito c. 1,0 m)

# espessura de cada pano (mediana das estações sem as torres e os corpos encostados, arredondada)
ESP = {"norte_oeste": 3.0, "norte_diagonal": 3.0, "norte": 3.0, "nordeste": 2.6, "leste": 2.5, "sueste": 2.5,
       "sul": 2.4, "sul_oeste": 2.6, "oeste": 3.0}
TORRES = {k: v for k, v in MUR["torres"].items() if v}
TORRE_PL = {k: shapely.Polygon(v["cantos"]) for k, v in TORRES.items()}
# a casa da guarda (corpo coberto sobre a entrada, x −26,5..−11,5, y −71..−51,5; LiDAR: terraço 55,5–56, parapeito
# 57,5) conta como torre para os panos: o pano sul acaba nela a nascente e o sul-poente a poente
CASA_GUARDA = [(-26.6, -71.0), (-11.4, -71.0), (-11.4, -51.6), (-26.6, -51.6)]
OBST_PL = dict(TORRE_PL, casa_guarda=shapely.Polygon(CASA_GUARDA))


def perto_torre(p, folga=0.4):
    pt = shapely.Point(float(p[0]), float(p[1]))
    return any(pl.buffer(folga).contains(pt) for pl in OBST_PL.values())


def pe_muro(pts):
    """Cota do pé do muro: o chão mais baixo à volta de cada ponto, 1,5 m abaixo (o terreno corta o resto)."""
    return np.array([min(cota(p[0] + dx, p[1] + dy) for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))) - 1.5
                     for p in pts])


def estacoes(nome):
    e = [x for x in MUR["panos"][nome]["estacoes"] if x]
    c = np.array([x["c"] for x in e])
    n = np.array([x["n"] for x in e])
    fora = np.array([x["fora"] for x in e])
    topo = np.array([x["topo"] for x in e])
    adarve = np.array([x["adarve"] for x in e])
    esp = ESP[nome]
    torre = np.array([perto_torre(p, 1.9) for p in c])   # o perfil ignora células a menos de 1,9 m das torres
    f = np.where(torre, np.nan, np.clip(fora, -1.6, 2.2))
    idx = np.arange(len(f))
    ok = ~np.isnan(f)
    f = np.interp(idx, idx[ok], f[ok]) if ok.any() else np.full(len(f), esp / 2)
    ext = c + n * suave(f, 9, 2.0)[:, None]
    ext, n2, fi = _endireitar(ext, n)
    tp = np.where(torre, np.nan, topo)
    ok = ~np.isnan(tp)
    tp = np.interp(idx, idx[ok], tp[ok]) if ok.any() else topo
    # o LiDAR alterna pontas dos merlões e soleiras das ameias: a linha dos merlões é o máximo em 2,5 m, nivelada
    topo_s = _patamares(ndi.maximum_filter1d(tp, 5, mode="nearest"))
    soleira = topo_s - ALT_MERLAO
    ad = np.where(torre, np.nan, adarve)
    ok = ~np.isnan(ad)
    ad = np.interp(idx, idx[ok], ad[ok]) if ok.any() else adarve
    ad = _patamares(ndi.median_filter(ad, 9, mode="nearest"), tol=0.45)
    ad = np.minimum(ad, soleira - 0.85)
    ad = np.maximum(ad, soleira - 1.25)
    rs = lambda v: np.interp(fi, idx, v)  # noqa: E731
    c = np.stack([rs(c[:, 0]), rs(c[:, 1])], 1)
    return c, n2, ext, esp, rs(soleira), rs(topo_s), rs(ad)


MERLOES = []   # (x, y, z) de cada merlão, para medir e conferir

# Portas abertas nos panos: nenhuma. A porta da alcáçova é a da casa da guarda (ver casa_guarda()).
PORTAS = {}

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
    lado = 1.0   # o parapeito fica do lado de fora (à esquerda do sentido de marcha)
    fora_l = ext                     # linha da face do lado do parapeito (ou exterior)
    dent_l = ext - n * esp           # face do outro lado
    if lado < 0:
        fora_l, dent_l = ext - n * esp, ext
    npar = n * lado                  # normal que aponta para o lado do parapeito
    par_l = fora_l - npar * ESP_PAR
    pe_f = pe_muro(fora_l + npar * 0.4)
    pe_d = pe_muro(dent_l - npar * 0.4)
    s_tmp = _s_linha(ext)
    for d in portas:   # debaixo do vão o pé desce abaixo da soleira (a escada do átrio fica mais baixa que o recinto)
        k_ = (s_tmp >= d["s0"] - 1e-6) & (s_tmp <= d["s1"] + 1e-6)
        pe_f[k_] = np.minimum(pe_f[k_], d["zs"] - 0.3)
        pe_d[k_] = np.minimum(pe_d[k_], d["zs"] - 0.3)
    N = len(c)
    # fora só as estações dentro dos cubelos e as que têm as duas faces bem dentro da torre de menagem (a ponta
    # prolongada fica 5 cm dentro da torre e fecha o pano)
    keep = np.array([not perto_torre((a + b) / 2, -0.3) for a, b in zip(fora_l, dent_l)])
    s_ext = _s_linha(ext)
    ponta_ini, ponta_fim = False, False
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



VAOS_PORTA = []   # (centro xy, soleira, normal) de cada porta, para o teste de raios


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



def muralhas():
    m = Malha(["muro", "muro_topo", "cantaria_lisa", "sombra", "madeira"] if "madeira" in MATS else
              ["muro", "muro_topo", "cantaria_lisa", "sombra"])
    INFO = {}
    for nome in MUR["panos"]:
        INFO[nome] = pano(m, nome)
    m.objeto("alcacova_panos", "muralhas", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)
    # guardas de ferro ao longo da borda de dentro dos adarves (fotografia 004: prumos e duas travessas)
    g = Malha(["ferro"])
    for nome, v in INFO.items():
        d, ad = v["dent"], v["ad"]
        pts = [(*(d[i] + unit(v["ext"][i] - d[i]) * 0.12), ad[i]) for i in range(len(d)) if not perto_torre(d[i], 0.6)]
        trecho = []
        for p in pts + [None]:
            if p is not None and (not trecho or math.dist(p[:2], trecho[-1][:2]) < 1.2):
                trecho.append(p)
                continue
            if len(trecho) > 3:
                guarda(g, trecho[::2] + ([trecho[-1]] if len(trecho) % 2 == 0 else []), "ferro", alt=1.0, passo=1.6,
                       r=0.02, travessas=(0.5,))
            trecho = [p] if p is not None else []
    g.objeto("guardas_adarves", "muralhas", GRUPOS["Muralhas"], uv=None, recalc=False)
    MEDIDAS["panos"] = {k: {"adarve_mediana": round(float(np.median(v["ad"])), 2),
                            "merloes_mediana": round(float(np.median(v["topo"])), 2),
                            "espessura": ESP[k]} for k, v in INFO.items()}
    MEDIDAS["merloes_alcacova"] = len(MERLOES)
    return INFO


# ---------------------------------------------------------------- torres, coroamentos e passagens em arco

def _inset(q, d):
    """Vértices do polígono convexo q (anti-horário) recuados d para dentro, em esquadria (mesma ordem)."""
    q = [np.asarray(p, float) for p in q]
    n = len(q)
    out = []
    for i in range(n):
        a, b, c = q[i - 1], q[i], q[(i + 1) % n]
        t0, t1 = unit(b - a), unit(c - b)
        n0, n1 = np.array([-t0[1], t0[0]]), np.array([-t1[1], t1[0]])   # normais para dentro (polígono anti-horário)
        bis = n0 + n1
        out.append(b + bis * d / max(1.0 + float(n0 @ n1), 0.3))
    return out


def merloes_linha(m, P0, P1, nin, sol, topo, mat, ini=0.0, fim=0.0, cantos=True):
    """Merlões retangulares ao longo de P0→P1 (face de fora), com espessura ESP_PAR para o lado nin; o primeiro e o
    último encostam às pontas (cantos das torres) quando cantos=True."""
    P0, P1 = np.asarray(P0, float), np.asarray(P1, float)
    L = float(np.linalg.norm(P1 - P0))
    t = (P1 - P0) / max(L, 1e-9)
    a0, a1 = ini, L - fim
    if a1 - a0 < L_MERLAO:
        return 0
    k = max(1, int(round((a1 - a0 + (PERIODO - L_MERLAO)) / PERIODO)))
    if cantos and k > 1:
        passo = (a1 - a0 - L_MERLAO) / (k - 1)
        inicios = [a0 + j * passo for j in range(k)]
    else:
        folga = (a1 - a0) - (k * PERIODO - (PERIODO - L_MERLAO))
        inicios = [a0 + folga / 2 + j * PERIODO for j in range(k)]
    for s0 in inicios:
        pa, pb = P0 + t * s0, P0 + t * (s0 + L_MERLAO)
        m.prisma(ccw([pa, pb, pb + nin * ESP_PAR, pa + nin * ESP_PAR]), sol - 0.02, topo, mat)
        MERLOES.append((*((pa + pb) / 2), topo))
    return k


def coroa(m, q, sol, topo, piso, mat="muro", mat_piso="terraco", merloes=True, faces_sem=()):
    """Coroamento de um corpo de planta convexa q: parapeito (soleira das ameias), merlões e terraço."""
    q = ccw(q)
    qi = _inset(q, ESP_PAR)
    n = len(q)
    nm = 0
    for i in range(n):
        j = (i + 1) % n
        m.face([(*q[i], sol), (*q[j], sol), (*qi[j], sol), (*qi[i], sol)], mat)
        m.face([(*qi[j], piso), (*qi[i], piso), (*qi[i], sol), (*qi[j], sol)], mat)
        if merloes and i not in faces_sem:
            t = unit(q[j] - q[i])
            nin = np.array([-t[1], t[0]])
            nm += merloes_linha(m, q[i], q[j], nin, sol, topo, mat, ini=ESP_PAR)
    m.face([(*p, piso) for p in qi], mat_piso)
    return nm


def corpo(m, q, z_top, mat, folga_pe=1.5, z_pe=None):
    q = ccw(q)
    pe = [z_pe if z_pe is not None else min(cota(*(p + unit(p - np.mean(q, 0)) * 0.8)), cota(*p)) - folga_pe for p in q]
    for i in range(len(q)):
        j = (i + 1) % len(q)
        m.face([(*q[i], pe[i]), (*q[j], pe[j]), (*q[j], z_top), (*q[i], z_top)], mat)
    return pe


def torre_ret(m, nome, q, topo, mat="cantaria_torre"):
    sol = topo - ALT_MERLAO
    piso = topo - 2.0
    corpo(m, q, sol, mat)
    nm = coroa(m, q, sol, topo, piso, mat=mat)
    e = np.diff(np.vstack([q, q[:1]]), axis=0)
    MEDIDAS.setdefault("torres", {})[nome] = {"planta": [round(float(np.hypot(*e[0])), 2), round(float(np.hypot(*e[1])), 2)],
                                              "merloes_pontas": round(topo, 2), "terraco": round(piso, 2), "merloes": nm}


def parede_arco(m, P0, P1, z0, z1, arcos, mat, k=12):
    """Parede vertical plana de P0 a P1 (z0..z1) com vãos em arco de volta perfeita: arcos = [(s0, s1, zs, zimp)]."""
    P0, P1 = np.asarray(P0, float), np.asarray(P1, float)
    L = float(np.linalg.norm(P1 - P0))
    t = (P1 - P0) / L

    def p(s, z):
        q = P0 + t * s
        return (q[0], q[1], z)

    cortes = sorted(arcos)
    s = 0.0
    for (s0, s1, zs, zimp) in cortes:
        if s0 > s:
            m.face([p(s, z0), p(s0, z0), p(s0, z1), p(s, z1)], mat)
        r = (s1 - s0) / 2
        if zs > z0:
            m.face([p(s0, z0), p(s1, z0), p(s1, zs), p(s0, zs)], mat)
        th = np.linspace(math.pi, 0, k + 1)
        arco = [(s0 + r + r * math.cos(a), zimp + r * math.sin(a)) for a in th]
        m.face([p(s0, zimp), p(s0, z1), p(s0 + r, z1)] + [p(*arco[j]) for j in range(k // 2, -1, -1)], mat)
        m.face([p(s0 + r, z1), p(s1, z1), p(s1, zimp)] + [p(*arco[j]) for j in range(k, k // 2 - 1, -1)], mat)
        s = s1
    if s < L:
        m.face([p(s, z0), p(L, z0), p(L, z1), p(s, z1)], mat)


def intradorso(m, A0, A1, B0, B1, zsA, zsB, zimp, mat, k=12, ombreiras=(True, True)):
    """Ombreiras, intradorso e soleira de um arco que vai da face A (A0→A1) à face B (B0→B1)."""
    A0, A1, B0, B1 = (np.asarray(v, float) for v in (A0, A1, B0, B1))
    r = float(np.linalg.norm(A1 - A0)) / 2
    th = np.linspace(math.pi, 0, k + 1)

    def arc(P, Q, a):
        return (*(P + (Q - P) * (0.5 + 0.5 * math.cos(a))), zimp + r * math.sin(a))

    for j in range(k):
        m.face([arc(A0, A1, th[j]), arc(B0, B1, th[j]), arc(B0, B1, th[j + 1]), arc(A0, A1, th[j + 1])], mat)
    if ombreiras[0]:
        m.face([(*A0, zsA), (*B0, zsB), (*B0, zimp), (*A0, zimp)], mat)
    if ombreiras[1]:
        m.face([(*A1, zsA), (*A1, zimp), (*B1, zimp), (*B1, zsB)], mat)
    m.face([(*A0, zsA), (*A1, zsA), (*B1, zsB), (*B0, zsB)], "cantaria_lisa")


def bloco_arco(m, A, B, w, z_pe, z_top, s0, s1, zs, zimp, mat, topo=True, pontas=(True, True)):
    """Corpo retangular de eixo A→B e largura w, do pé ao topo, atravessado na transversal por uma passagem em arco
    entre s0 e s1 (abcissas ao longo do eixo), soleira zs = (lado esquerdo, lado direito) e arranque zimp."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    L = float(np.linalg.norm(B - A))
    t = (B - A) / L
    nl = np.array([-t[1], t[0]])
    E0_, E1_ = A + nl * w / 2, B + nl * w / 2     # face esquerda
    D0_, D1_ = A - nl * w / 2, B - nl * w / 2     # face direita
    zsl, zsr = zs
    parede_arco(m, E0_, E1_, z_pe, z_top, [(s0, s1, zsl, zimp)], mat)
    parede_arco(m, D0_, D1_, z_pe, z_top, [(s0, s1, zsr, zimp)], mat)
    intradorso(m, E0_ + t * s0, E0_ + t * s1, D0_ + t * s0, D0_ + t * s1, zsl, zsr, zimp, mat)
    if pontas[0]:
        m.face([(*D0_, z_pe), (*E0_, z_pe), (*E0_, z_top), (*D0_, z_top)], mat)
    if pontas[1]:
        m.face([(*E1_, z_pe), (*D1_, z_pe), (*D1_, z_top), (*E1_, z_top)], mat)
    if topo:
        m.face([(*E0_, z_top), (*D0_, z_top), (*D1_, z_top), (*E1_, z_top)], "muro_topo")
    return E0_, E1_, D0_, D1_


def parapeitos_braco(m, E0_, E1_, D0_, D1_, ad, mat="muro"):
    """Parapeitos e merlões dos dois lados do adarve de um braço de albarrã."""
    sol, topo = ad + 1.0, ad + 2.0
    for (P0, P1, Q0, Q1) in ((E0_, E1_, D0_, D1_), (D1_, D0_, E1_, E0_)):
        nin = unit((Q0 - P0))
        a, b = P0 + nin * ESP_PAR, P1 + nin * ESP_PAR
        m.prisma(ccw([P0, P1, b, a]), ad, sol, mat, mat_topo="muro_topo")
        merloes_linha(m, P0, P1, nin, sol, topo, mat, ini=0.2, fim=0.2, cantos=False)


def torres():
    m = Malha(["cantaria_torre", "terraco", "muro", "muro_topo", "cantaria_lisa", "reboco"])
    for nome, t in TORRES.items():
        q = [np.array(p) for p in t["cantos"]]
        if nome == "albarra_sueste":
            # o retângulo do LiDAR (14,1 × 7,2 m) é a torre (7,2 × 7,2, a sueste) mais o braço que a liga ao pano
            # sueste, atravessado por um arco alto (foto 020, tirada de nascente)
            q = ccw(q)
            L = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
            i = int(np.argmax(L))
            a, b = q[i], q[(i + 1) % 4]
            c, d = q[(i + 2) % 4], q[(i + 3) % 4]
            # ponta de fora: a mais longe do centro da alcáçova
            fora_b = np.linalg.norm(b) > np.linalg.norm(a)
            lado = 7.18
            if fora_b:
                tt = unit(b - a)
                torre_q = [b - tt * lado, b, c, c - tt * lado]
                eixo_A, eixo_B = (a + d) / 2, (b - tt * lado + c - tt * lado) / 2
            else:
                tt = unit(a - b)
                torre_q = [a, a - tt * lado, d - tt * lado, d]
                eixo_A, eixo_B = (b + c) / 2, (a - tt * lado + d - tt * lado) / 2
            torre_ret(m, nome, torre_q, t["topo"])
            # braço: 3,2 m de largura, adarve à cota do pano sueste, arco de 4 m de vão e c. 7 m de altura
            ad = 52.98
            ch = float(cota(*((eixo_A + eixo_B) / 2)))
            Lb = float(np.linalg.norm(eixo_B - eixo_A))
            E0_, E1_, D0_, D1_ = bloco_arco(m, eixo_A - unit(eixo_B - eixo_A) * 1.0, eixo_B + unit(eixo_B - eixo_A) * 0.2,
                                            3.2, ch - 1.5, ad, 1.0 + Lb / 2 - 2.0, 1.0 + Lb / 2 + 2.0, (ch, ch), ch + 5.0,
                                            "muro", pontas=(False, False))
            parapeitos_braco(m, E0_, E1_, D0_, D1_, ad)
            BRACOS["albarra_sueste"] = dict(eixo=[eixo_A.tolist(), eixo_B.tolist()], adarve=ad, chao=round(ch, 2),
                                            arco=[4.0, round(ch + 7.0, 2)])
            continue
        torre_ret(m, nome, q, t["topo"])
    # braço da albarrã nordeste: do canto do pano nordeste/leste à face poente da torre, com arco por baixo
    # (fotos 020 e 030: arco com merlões por cima e um pilar rebocado)
    t = TORRES["albarra_nordeste"]
    q = ccw([np.array(p) for p in t["cantos"]])
    face = min(range(4), key=lambda i: np.linalg.norm((q[i] + q[(i + 1) % 4]) / 2 - np.array([49.6, 28.2])))
    fm = (q[face] + q[(face + 1) % 4]) / 2
    A = np.array([47.3, 29.9])
    ad = 58.69
    ch = float(cota(*((A + fm) / 2)))
    Lb = float(np.linalg.norm(fm - A))
    E0_, E1_, D0_, D1_ = bloco_arco(m, A, fm + unit(fm - A) * 0.2, 3.0, ch - 1.5, ad, Lb / 2 - 1.3, Lb / 2 + 1.3,
                                    (ch, ch), ch + 4.5, "muro", pontas=(False, False))
    parapeitos_braco(m, E0_, E1_, D0_, D1_, ad)
    BRACOS["albarra_nordeste"] = dict(eixo=[A.tolist(), fm.tolist()], adarve=ad, chao=round(ch, 2))
    MEDIDAS["bracos_albarras"] = BRACOS
    m.objeto("torres", "torres", GRUPOS["Torre"], uv=2.0, vcor=patine_torre)


BRACOS = {}


class Bloco:
    """Referencial (u, v) de um corpo retangular: u ao longo da face «poente» (v = 0), de sul para norte, v para
    dentro (nascente); a face «norte» é u = LU e a «nascente» v = LV."""

    def __init__(self, O, t, LU, LV):
        self.O, self.t = np.asarray(O, float), unit(t)
        self.e = np.array([self.t[1], -self.t[0]])
        self.LU, self.LV = LU, LV

    def xy(self, u, v):
        return self.O + self.t * u + self.e * v

    def q(self):
        return ccw([self.xy(0, 0), self.xy(self.LU, 0), self.xy(self.LU, self.LV), self.xy(0, self.LV)])


def passagem_cotovelo(m, B, z_pe, z_top, ent, sai, z_in, z_out, imp1, imp2, mat, folhas=False):
    """Corpo retangular com uma passagem em cotovelo: entra em arco pela face poente (v = 0) entre u = ent[0] e
    ent[1], segue para nascente até ao corredor de saída (v = sai[0]..sai[1]), que corre para norte, em rampa, até
    ao arco da face norte (u = LU). Faces de fora, corredores abobadados, soleiras e, se folhas, as duas folhas de
    madeira abertas para dentro no arco de entrada (foto 101)."""
    ua0, ua1 = ent
    vb0, vb1 = sai
    ub0 = ua0 - 0.4
    X = B.xy
    zi1, zi2 = z_in + imp1, max(z_in, z_out) + imp2
    # faces de fora: poente e norte com os arcos, sul e nascente lisas
    parede_arco(m, X(0, 0), X(B.LU, 0), z_pe, z_top, [(ua0, ua1, z_in, zi1)], mat)
    parede_arco(m, X(B.LU, 0), X(B.LU, B.LV), z_pe, z_top, [(vb0, vb1, z_out, zi2)], mat)
    m.face([(*X(B.LU, B.LV), z_pe), (*X(0, B.LV), z_pe), (*X(0, B.LV), z_top), (*X(B.LU, B.LV), z_top)], mat)
    m.face([(*X(0, B.LV), z_pe), (*X(0, 0), z_pe), (*X(0, 0), z_top), (*X(0, B.LV), z_top)], mat)
    # corredor de entrada (de v = 0 a vb0), abóbada de berço
    intradorso(m, X(ua0, 0), X(ua1, 0), X(ua0, vb0), X(ua1, vb0), z_in, z_in, zi1, mat)
    # corredor de saída: parede poente com o arco do corredor de entrada, parede nascente, fundo a sul, abóbada
    parede_arco(m, X(ub0, vb0), X(B.LU, vb0), z_in, zi2, [(ua0 - ub0, ua1 - ub0, z_in, zi1)], mat)
    m.face([(*X(ub0, vb0), z_in), (*X(ub0, vb1), z_in), (*X(ub0, vb1), zi2 + (vb1 - vb0) / 2), (*X(ub0, vb0), zi2 + (vb1 - vb0) / 2)], mat)
    intradorso(m, X(ub0, vb1), X(ub0, vb0), X(B.LU, vb1), X(B.LU, vb0), z_in, z_out, zi2, mat, ombreiras=(True, False))
    if folhas:
        w = (ua1 - ua0) / 2
        for (u_, sg) in ((ua0, 1), (ua1, -1)):
            p0 = X(u_ + sg * 0.04, 0.35)
            q_ = [p0, X(u_ + sg * 0.12, 0.35), X(u_ + sg * 0.12, 0.35 + w), X(u_ + sg * 0.04, 0.35 + w)]
            m.prisma(ccw(q_), z_in + 0.02, zi1 + w * 0.5, "madeira")
    return dict(z_in=round(z_in, 2), z_out=round(z_out, 2), arco_entrada=[round(ua1 - ua0, 2), round(zi1 + (ua1 - ua0) / 2, 2)],
                arco_saida=[round(vb1 - vb0, 2), round(zi2 + (vb1 - vb0) / 2, 2)])


# Casa da guarda e porta da alcáçova: corpo coberto (x −26,6..−11,4, y −71..−51,6; LiDAR: terraço 55,5–56 m,
# parapeito 57,5) encostado à torre da porta. A porta é um arco de volta perfeita na face poente, logo a norte da
# torre da porta e recuado em relação a ela, com a estátua de D. Sancho I diante da torre (foto 023, tirada do largo
# a sudoeste, x −38,5, y −74,2); o arco é c. 1,4 vezes a altura da estátua (c. 3,4 m ao fecho). Por dentro, o
# corredor segue para nascente e vira a norte (foto 101, tirada dentro dele, x −20,5, y −62,3, a olhar para a porta:
# folhas de madeira abertas e chão de calçada) e sobe em rampa até à porta do recinto, na face norte. Cotas: largo
# 47,4 m, recinto 50,6 m (LiDAR). O traçado exato do corredor de saída é hipótese.
CG = Bloco((-26.6, -71.0), (0.0, 1.0), 19.4, 15.2)


def casa_guarda():
    m = Malha(["muro", "muro_topo", "terraco", "cantaria_lisa", "sombra", "madeira"])
    sol, topo, piso = 56.6, 57.6, 55.6
    z_pe = 46.0
    info = passagem_cotovelo(m, CG, z_pe, sol, (5.0, 7.4), (4.8, 7.8), 47.4, 50.6, 2.2, 2.4, "muro", folhas=True)
    q = CG.q()
    # merlões nas faces de fora; a face norte (recinto) fica só com o parapeito
    norte = [i for i in range(4) if np.allclose((q[i] + q[(i + 1) % 4]) / 2, CG.xy(CG.LU, CG.LV / 2), atol=0.1)]
    nm = coroa(m, q, sol, topo, piso, mat="muro", mat_piso="terraco", faces_sem=tuple(norte))
    m.objeto("casa_guarda", "porta", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)
    MEDIDAS["casa_guarda"] = dict(info, planta=[CG.LV, CG.LU], terraco=piso, merloes_pontas=topo, merloes=nm)


# ---------------------------------------------------------------- interior da alcáçova

# Cisterna da Moura (Wikipédia: 20 × 16 × 7 m, quatro naves de abóbada de berço sobre doze colunas). No LiDAR e na
# ortofoto do LiDAR veem-se as quatro abóbadas rebocadas, rosadas, com os eixos a norte-noroeste, o extradorso a
# 52,4–53,6 m, abaixo do chão à volta (54–55 m), e um murete à volta.
CISTERNA = [(20.8, 20.0), (40.6, 17.4), (42.6, 32.6), (22.9, 35.2)]


def escada_adarve():
    """Escada de pedra encostada à face de dentro do pano norte-oeste, a subir para poente até ao adarve, com um
    patamar a meio (foto 004 do Commons, cujo GPS cai do outro lado do recinto; LiDAR: rampa de 53,4 a 57,8 m entre
    x −35,0 e −41,8, com patamar a 55,6 m entre x −37,7 e −38,5, faixa de 1,9 m junto ao muro)."""
    m = Malha(["muro", "muro_topo"])
    y_muro = 44.8 - ESP["norte_oeste"] / 2 + 0.05
    y_fora = y_muro - 1.95
    ch = 53.4
    lances = [(-35.0, -37.7, ch, 55.6), (-38.5, -41.8, 55.6, 57.85)]
    nd = 0
    for (xa, xb, za, zb) in lances:
        n = max(1, round((zb - za) / 0.2))
        dx = (xb - xa) / n
        for k in range(n):
            x0, x1 = xa + dx * k, xa + dx * (k + 1)
            z1 = za + (zb - za) * (k + 1) / n
            m.prisma(ccw([(x0, y_fora), (x1, y_fora), (x1, y_muro), (x0, y_muro)]), ch - 3.5, z1, 0, mat_topo="muro_topo")
            nd += 1
    # patamar
    m.prisma(ccw([(-37.7, y_fora), (-38.5, y_fora), (-38.5, y_muro), (-37.7, y_muro)]), ch - 3.5, 55.6, 0,
             mat_topo="muro_topo")
    m.objeto("escada_adarve", "muralha_alcacova", GRUPOS["Muralhas"], uv=2.0, vcor=patine_muro)
    MEDIDAS["escada_adarve"] = dict(degraus=nd, largura=round(y_muro - y_fora, 2), cotas=[ch, 55.6, 57.85])


def cisterna():
    m = Malha(["abobada", "muro", "muro_topo"])
    q = [np.array(p) for p in CISTERNA]
    u = q[1] - q[0]          # ao longo das naves em largura (este)
    v = q[3] - q[0]          # eixo das naves (norte-noroeste)
    nb = 4
    z_arr, flecha = 52.4, 1.2
    k = 10
    for i in range(nb):
        a0, a1 = i / nb, (i + 1) / nb
        for j in range(k):
            t0, t1 = a0 + (a1 - a0) * j / k, a0 + (a1 - a0) * (j + 1) / k
            h0 = z_arr + flecha * math.sin(math.pi * (t0 - a0) / (a1 - a0))
            h1 = z_arr + flecha * math.sin(math.pi * (t1 - a0) / (a1 - a0))
            P0, P1 = q[0] + u * t0, q[0] + u * t1
            m.face([(*P0, h0), (*P1, h1), (*(P1 + v), h1), (*(P0 + v), h0)], "abobada")
        # tímpanos das pontas
        for (base, sg) in ((q[0], 1), (q[0] + v, -1)):
            pts = [(*(base + u * (a0 + (a1 - a0) * j / k)), z_arr + flecha * math.sin(math.pi * j / k)) for j in range(k + 1)]
            m.face([(*(base + u * a0), z_arr - 0.6)] + pts + [(*(base + u * a1), z_arr - 0.6)], "abobada")
    # murete à volta (0,45 m de espessura, 0,9 m acima do chão de fora)
    qq = ccw(q)
    fora = _inset(qq, -0.45)
    zc = max(float(np.median([sup(*p) for p in fora])), 54.0)
    for i in range(4):
        j = (i + 1) % 4
        m.face([(*fora[i], zc - 2.5), (*fora[j], zc - 2.5), (*fora[j], zc + 0.5), (*fora[i], zc + 0.5)], "muro")
        m.face([(*qq[j], z_arr - 0.6), (*qq[i], z_arr - 0.6), (*qq[i], zc + 0.5), (*qq[j], zc + 0.5)], "muro")
        m.face([(*qq[i], zc + 0.5), (*qq[j], zc + 0.5), (*fora[j], zc + 0.5), (*fora[i], zc + 0.5)], "muro_topo")
    m.objeto("cisterna", "cisterna", GRUPOS["Interior"], uv=2.0, vcor=patine_muro)
    MEDIDAS["cisterna"] = {"naves": nb, "extradorso": [z_arr, z_arr + flecha], "murete": round(zc + 0.5, 2),
                           "planta": [round(float(np.linalg.norm(u)), 1), round(float(np.linalg.norm(v)), 1)]}


ESTATUA = np.array([-32.6, -71.4])   # diante da face poente da torre da porta (foto 023)


def estatua_mastro():
    """Estátua de D. Sancho I (Leopoldo de Almeida, 1965): pedestal baixo de pedra e figura de bronze de pé, de
    espada; mastro da bandeira no terraço da torre da porta (o LiDAR tem o topo a c. 66 m)."""
    m = Malha(["pedestal", "bronze", "ferro", "bandeira_verde", "bandeira_vermelha"])
    c = ESTATUA
    z0 = float(cota(*c))
    m.prisma(ccw([c + [-0.7, -0.7], c + [0.7, -0.7], c + [0.7, 0.7], c + [-0.7, 0.7]]), z0 - 0.3, z0 + 0.5, "pedestal")
    # figura: pernas, corpo com manto (tronco de cone de 12 lados), cabeça e espada
    def cone(cx, cy, za, zb, ra, rb, mat, k=12):
        for j in range(k):
            a, b = 2 * math.pi * j / k, 2 * math.pi * (j + 1) / k
            m.face([(cx + ra * math.cos(a), cy + ra * math.sin(a), za), (cx + ra * math.cos(b), cy + ra * math.sin(b), za),
                    (cx + rb * math.cos(b), cy + rb * math.sin(b), zb), (cx + rb * math.cos(a), cy + rb * math.sin(a), zb)], mat)
        m.face([(cx + rb * math.cos(2 * math.pi * j / k), cy + rb * math.sin(2 * math.pi * j / k), zb) for j in range(k)], mat)
    zf = z0 + 0.5
    cone(c[0], c[1], zf, zf + 1.25, 0.42, 0.26, "bronze")      # manto até à cintura
    cone(c[0], c[1], zf + 1.25, zf + 1.85, 0.26, 0.30, "bronze")  # peito e ombros
    cone(c[0], c[1], zf + 1.85, zf + 2.15, 0.13, 0.12, "bronze", 8)  # cabeça
    m.barra((c[0] - 0.35, c[1] - 0.25, zf + 0.05), (c[0] - 0.3, c[1] - 0.2, zf + 1.3), 0.03, "bronze")  # espada
    # mastro
    p = np.array([-29.2, -73.3])
    t = TORRES["porta"]["topo"]
    m.barra((p[0], p[1], t - 2.0), (p[0], p[1], 66.0), 0.06, "ferro", k=8)
    m.face([(p[0] + 0.08, p[1], 65.8), (p[0] + 0.08 + 1.2, p[1], 65.8), (p[0] + 0.08 + 1.2, p[1], 64.6), (p[0] + 0.08, p[1], 64.6)], "bandeira_verde")
    m.face([(p[0] + 1.28, p[1], 65.8), (p[0] + 3.08, p[1], 65.8), (p[0] + 3.08, p[1], 64.6), (p[0] + 1.28, p[1], 64.6)], "bandeira_vermelha")
    m.objeto("estatua_mastro", "estatua", GRUPOS["Interior"], uv=None, recalc=False)


def edificios_interior():
    """Edifícios modernos dentro da alcáçova (classe 6 do LiDAR, 2,5–5 m de altura): volumes rebocados de branco
    com cobertura de telha de quatro águas à cota do MDS."""
    from rasterio import features
    from rasterio.transform import Affine
    h = MDS - MDT
    ed = ndi.binary_opening((CLS == 6) & (h > 2.0) & (h < 6.5), iterations=1)
    lab, _ = ndi.label(ed)
    tr = Affine(LP, 0, LX0 - LP / 2, 0, LP, LY0 - LP / 2)
    m = Malha(["branco", "telha"])
    n = 0
    for geo, _v in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=tr):
        p = shapely.geometry.shape(geo).buffer(0)
        if p.area < 20 or not ALCACOVA_PL.buffer(-3.0).contains(p.centroid):
            continue
        p = shapely.minimum_rotated_rectangle(p)
        if any(p.intersects(t.buffer(0.5)) for t in OBST_PL.values()):
            continue
        base = ccw(np.array(p.exterior.coords)[:-1])
        gx, gy = np.meshgrid(np.arange(*p.bounds[0::2], 0.5), np.arange(*p.bounds[1::2], 0.5))
        di = shapely.contains_xy(p.buffer(-0.5), gx, gy)
        if di.sum() < 4:
            continue
        zs = amostra(MDS, LX0, LY0, LP, gx[di], gy[di])
        zt = amostra(SUP, SX0, SY0, SP, gx[di], gy[di])
        chao = float(np.percentile(chao_castelo_xy(gx[di], gy[di]), 20))
        cume = float(np.percentile(zs, 90))
        beiral = max(chao + 2.4, cume - 1.4)
        del zt
        corpo(m, base, beiral, "branco", z_pe=chao - 0.5)
        cx = np.mean(base, 0)
        m.face([(*base[0], beiral), (*base[1], beiral), (*cx, cume)], "telha")
        m.face([(*base[1], beiral), (*base[2], beiral), (*cx, cume)], "telha")
        m.face([(*base[2], beiral), (*base[3], beiral), (*cx, cume)], "telha")
        m.face([(*base[3], beiral), (*base[0], beiral), (*cx, cume)], "telha")
        n += 1
    if n:
        m.objeto("edificios_recinto", "interior", GRUPOS["Interior"], uv=2.0)
    MEDIDAS["edificios_recinto"] = n


# ---------------------------------------------------------------- cerca da vila

ESP_CERCA = 2.8    # Wikipédia: muros de taipa de 2,8 m forrados de grés


def cerca():
    m = Malha(["cerca", "muro_topo"])
    info = {}
    for nome, v in MUR["cerca"].items():
        e = v["estacoes"]
        c = np.array([x["c"] for x in e])
        n = np.array([x["n"] for x in e])
        tp = np.array([x["topo"] for x in e])
        cf = np.array([x["chao_fora"] for x in e])
        cd = np.array([x["chao_dentro"] for x in e])
        # topo: mediana em 7 m e patamares; nunca menos de 1,5 m acima do chão de dentro (o muro existe em toda a
        # extensão que o OSM regista; onde as casas o escondem, o LiDAR dá-lhe a altura dos telhados encostados)
        tp = _patamares(ndi.median_filter(tp, 7, mode="nearest"), tol=0.8, minimo=4)
        tp = np.maximum(tp, np.maximum(cd, cf) + 1.5)
        esp = ESP_CERCA if nome != "cerca_porta" else 2.4
        fora_l = c + n * esp / 2
        dent_l = c - n * esp / 2
        pe_f, pe_d = pe_muro(fora_l + n * 0.4), pe_muro(dent_l - n * 0.4)
        torre = np.array([any(shapely.Polygon(t["poligono"]).buffer(0.3).contains(shapely.Point(*p))
                              for t in MUR["cerca_torres"].values()) or
                          shapely.Polygon(TORREAO.q()).buffer(1.6).contains(shapely.Point(*p)) for p in c])
        for i in range(len(c) - 1):
            if torre[i] and torre[i + 1]:
                continue
            a, b = i, i + 1
            m.face([(*fora_l[a], pe_f[a]), (*fora_l[b], pe_f[b]), (*fora_l[b], tp[b]), (*fora_l[a], tp[a])], "cerca")
            m.face([(*dent_l[b], pe_d[b]), (*dent_l[a], pe_d[a]), (*dent_l[a], tp[a]), (*dent_l[b], tp[b])], "cerca")
            m.face([(*dent_l[a], tp[a]), (*fora_l[a], tp[a]), (*fora_l[b], tp[b]), (*dent_l[b], tp[b])], "muro_topo")
        for i in (0, len(c) - 1):
            m.face([(*fora_l[i], pe_f[i]), (*dent_l[i], pe_d[i]), (*dent_l[i], tp[i]), (*fora_l[i], tp[i])], "cerca")
        # topos onde o muro salta de patamar
        for i in range(len(c) - 1):
            if abs(tp[i + 1] - tp[i]) > 0.3:
                k_ = i if tp[i] > tp[i + 1] else i + 1
                m.face([(*fora_l[k_], min(tp[i], tp[i + 1])), (*dent_l[k_], min(tp[i], tp[i + 1])),
                        (*dent_l[k_], tp[k_]), (*fora_l[k_], tp[k_])], "cerca")
        info[nome] = {"comprimento_m": round(float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))), 1),
                      "topo_mediana": round(float(np.median(tp)), 2),
                      "altura_fora_mediana": round(float(np.median(tp - cf)), 2)}
    m.objeto("cerca_vila", "cerca", GRUPOS["Muralha_vila"], uv=2.0, vcor=patine_muro)
    # torres da cerca: prismas da planta do OSM até ao topo do LiDAR, terraço com parapeito (as mais altas que o
    # muro) — sem merlões, que as fotografias das torres da vila não mostram
    mt = Malha(["cerca", "muro_topo", "terraco", "cantaria_lisa", "muro", "taipa", "vidro", "branco", "madeira", "ferro"])
    tinfo = {}
    for nome, t in MUR["cerca_torres"].items():
        if nome == "porta_cidade":
            continue
        q = ccw([np.array(p) for p in t["cantos"]])
        topo = t["topo"]
        # embasamento de grés até c. 40 % da altura e taipa por cima (fotos 081–083)
        zm = t["chao"] + 0.4 * (topo - t["chao"])
        corpo(mt, q, zm, "cerca")
        corpo(mt, q, topo, "taipa", z_pe=zm)
        mt.face([(*p, topo) for p in q], "muro_topo")
        tinfo[nome] = {"topo": round(topo, 2), "altura": round(topo - t["chao"], 2)}
    porta_cidade(mt)
    mt.objeto("torres_vila", "cerca", GRUPOS["Muralha_vila"], uv=2.0, vcor=patine_muro)
    MEDIDAS["cerca"] = info
    MEDIDAS["torres_cerca"] = tinfo


# Torreão da Porta da Cidade (Porta de Loulé, séc. XIV; OSM «Arcos da Porta de Loulé»): torre maciça de grés
# vermelho de c. 11,6 × 20 m saliente a sul da cerca, sem merlões, com o terraço a c. 32,5 m e um parapeito baixo
# (LiDAR: 33,3 m). Entra-se por um arco alto na face poente (fotos 086–088, tiradas a poente, x −163..−195) e a
# passagem vira a norte, para dentro da vila (porta em cotovelo; o corredor por dentro é hipótese). Duas janelas de
# moldura branca no andar de cima da face poente (fotos 087 e 088).
TORREAO = Bloco((-145.9, -263.3), (-3.9, 20.0), 20.0, 11.6)
PORTA_CIDADE = {}


def porta_cidade(m):
    t = MUR["cerca_torres"]["porta_cidade"]
    B = TORREAO
    z_in = float(cota(*B.xy(14.1, -2.0)))     # largo a poente
    z_out = float(cota(*B.xy(B.LU + 2.0, 5.0)))   # rua a norte, dentro da vila
    piso = round(t["meio"], 2)
    info = passagem_cotovelo(m, B, min(z_in, z_out) - 1.5, piso, (12.8, 15.4), (3.5, 6.5), z_in, z_out, 6.9, 6.6, "cerca")
    q = B.q()
    qi = _inset(q, 0.5)
    for i in range(4):
        j = (i + 1) % 4
        m.face([(*q[i], piso), (*q[j], piso), (*q[j], piso + 0.8), (*q[i], piso + 0.8)], "cerca")
        m.face([(*q[i], piso + 0.8), (*q[j], piso + 0.8), (*qi[j], piso + 0.8), (*qi[i], piso + 0.8)], "muro_topo")
        m.face([(*qi[j], piso), (*qi[i], piso), (*qi[i], piso + 0.8), (*qi[j], piso + 0.8)], "cerca")
    m.face([(*p, piso) for p in qi], "terraco")
    # janelas de sacada da face poente (foto 086): vão de c. 1,3 × 2,0 m com moldura branca larga e cornija, a de
    # norte por cima do arco; varanda de pedra com guarda de ferro
    for u in (5.8, 14.1):
        zj = z_in + 10.0
        a, b = B.xy(u - 0.65, -0.02), B.xy(u + 0.65, -0.02)
        m.face([(*a, zj), (*b, zj), (*b, zj + 2.0), (*a, zj + 2.0)], "vidro")
        for (u0, u1, z0, z1) in ((u - 0.95, u - 0.65, zj, zj + 2.0), (u + 0.65, u + 0.95, zj, zj + 2.0),
                                 (u - 0.95, u + 0.95, zj + 2.0, zj + 2.3), (u - 1.1, u + 1.1, zj + 2.3, zj + 2.45)):
            m.prisma(ccw([B.xy(u0, 0), B.xy(u1, 0), B.xy(u1, -0.12), B.xy(u0, -0.12)]), z0, z1, "branco")
        # varanda: laje e guarda (prumos e corrimão)
        m.prisma(ccw([B.xy(u - 1.0, 0), B.xy(u + 1.0, 0), B.xy(u + 1.0, -0.55), B.xy(u - 1.0, -0.55)]), zj - 0.15, zj,
                 "branco")
        for uu in np.linspace(u - 0.95, u + 0.95, 9):
            m.barra((*B.xy(uu, -0.5), zj), (*B.xy(uu, -0.5), zj + 0.95), 0.015, "ferro", k=4)
        m.barra((*B.xy(u - 0.95, -0.5), zj + 0.95), (*B.xy(u + 0.95, -0.5), zj + 0.95), 0.025, "ferro", k=4)
        for uu in (u - 0.95, u + 0.95):
            m.barra((*B.xy(uu, -0.02), zj + 0.95), (*B.xy(uu, -0.5), zj + 0.95), 0.025, "ferro", k=4)
            m.barra((*B.xy(uu, -0.05), zj + 0.1), (*B.xy(uu, -0.05), zj + 0.95), 0.015, "ferro", k=4)
    PORTA_CIDADE.update(info, terraco=piso, planta=[B.LV, B.LU], chao_norte=round(z_out, 2), chao_sul=round(z_in, 2))
    MEDIDAS["porta_cidade"] = dict(PORTA_CIDADE)


# ---------------------------------------------------------------- terreno

def _contorno_alcacova():
    pts = []
    for k in ("norte_oeste", "norte_diagonal", "norte", "nordeste", "leste", "sueste", "sul", "sul_oeste", "oeste"):
        pts.extend(tuple(x["c"]) for x in MUR["panos"][k]["estacoes"] if x)
    p = shapely.Polygon(pts).buffer(0)
    p = shapely.union_all([p] + list(OBST_PL.values())).buffer(0.3)
    if p.geom_type == "MultiPolygon":
        p = max(p.geoms, key=lambda g: g.area)
    return shapely.Polygon(p.exterior)


ALCACOVA_PL = _contorno_alcacova()
_cerca_l = [shapely.LineString([x["c"] for x in v["estacoes"]]).buffer(ESP_CERCA / 2 + 0.3, cap_style="flat")
            for v in MUR["cerca"].values()]
_cerca_t = [shapely.Polygon(t["poligono"]).buffer(0.3) for t in MUR["cerca_torres"].values()]
CERCA_PL = shapely.union_all(_cerca_l + _cerca_t)
OBRA_PL = shapely.union_all([ALCACOVA_PL, CERCA_PL] + [shapely.Polygon(CASA_GUARDA).buffer(0.3)])
# para as árvores: dentro da alcáçova há árvores (jardim e recinto), por isso só se afastam dos muros, das torres,
# da casa da guarda, da cisterna e da cerca
_muros = [shapely.LineString([x["c"] for x in v["estacoes"] if x]).buffer(2.2) for v in MUR["panos"].values()]
CASTELO_PL = shapely.union_all(_muros + [p.buffer(0.5) for p in OBST_PL.values()] + [CERCA_PL, shapely.Polygon(CISTERNA).buffer(0.8)])


def _chao_castelo():
    """Chão dentro da alcáçova: a superfície do LiDAR sem as alvenarias nem os edifícios do recinto, preenchida por
    baixo deles (vizinho mais próximo)."""
    ch = ndi.grey_opening(SUP, size=(61, 61))
    alv = ndi.binary_opening((SUP - ch) > 1.2, iterations=1)
    alv = ndi.binary_dilation(alv, iterations=2)
    z = SUP.copy()
    z[alv] = np.nan
    idx = ndi.distance_transform_edt(np.isnan(z), return_distances=False, return_indices=True)
    return ndi.median_filter(z[tuple(idx)], size=3)


CH = _chao_castelo()


def chao_castelo_xy(x, y):
    return amostra(CH, SX0, SY0, SP, x, y)


def construir_terreno():
    passo = 1.0
    xs = np.arange(BASE_C[0] - BASE_R, BASE_C[0] + BASE_R + 1e-6, passo)
    ys = np.arange(BASE_C[1] - BASE_R, BASE_C[1] + BASE_R + 1e-6, passo)
    X, Y = np.meshgrid(xs, ys)
    Z = cota(X, Y)
    dentro = shapely.contains_xy(ALCACOVA_PL.buffer(1.0), X, Y)
    Z = np.where(dentro, chao_castelo_xy(X, Y), Z)
    # debaixo da casa da guarda o chão fica abaixo do corredor da porta (o corpo fecha-se por cima)
    cg = shapely.contains_xy(shapely.Polygon(CG.q()).buffer(-0.05), X, Y)
    Z = np.where(cg, 46.8, Z)
    # passagem sob o arco do braço da albarrã de sueste (foto 020: o arco abre-se do chão): o chão desce à soleira
    br = BRACOS["albarra_sueste"]
    A_, B_ = np.array(br["eixo"][0]), np.array(br["eixo"][1])
    ta = unit(B_ - A_)
    na = np.array([-ta[1], ta[0]])
    M_ = (A_ + B_) / 2
    s_ = (X - M_[0]) * ta[0] + (Y - M_[1]) * ta[1]
    d_ = (X - M_[0]) * na[0] + (Y - M_[1]) * na[1]
    sob = (np.abs(s_) < 2.2) & (np.abs(d_) < 5.0)
    Z = np.where(sob, np.minimum(Z, br["chao"] + 0.15 * np.maximum(np.abs(d_) - 1.6, 0)), Z)
    # debaixo do torreão da Porta da Cidade o chão fica abaixo da passagem
    tor = shapely.Polygon(TORREAO.q()).buffer(-0.05)
    so = shapely.contains_xy(tor, X, Y)
    Z = np.where(so, min(PORTA_CIDADE["chao_norte"], PORTA_CIDADE["chao_sul"]) - 0.4, Z)
    r = np.hypot(X - BASE_C[0], Y - BASE_C[1])
    ok = r <= BASE_R
    idx = -np.ones(X.shape, int)
    idx[ok] = np.arange(ok.sum())
    V = np.stack([X[ok], Y[ok], Z[ok]], 1)
    I, J = np.nonzero(ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:])
    Fq = np.stack([idx[I, J], idx[I, J + 1], idx[I + 1, J + 1], idx[I + 1, J]], 1)
    mats = ["terra", "relva", "lajeado", "asfalto", "calcada", "mato", "saibro"]
    ob = malha_rapida("Terreno", V, Fq, mats[0], GRUPOS["Terreno"], "terreno")
    for mt in mats[1:]:
        ob.data.materials.append(MATS[mt])
    cx = V[Fq].mean(1)
    ci = np.clip(((cx[:, 0] - OX0) / OP).astype(int), 0, ORTO.shape[1] - 1)
    ri = np.clip(((OY1 - cx[:, 1]) / OP).astype(int), 0, ORTO.shape[0] - 1)
    o = ndi.uniform_filter(ORTO, size=(9, 9, 1))[ri, ci]
    lum = o.mean(1)
    sat = o.max(1) - o.min(1)
    ny, nx = ok.shape[0] - 1, ok.shape[1] - 1

    def manchas(sel, minimo=10, sig=1.0):
        g = np.zeros((ny, nx), float)
        g[I, J] = sel
        g = ndi.gaussian_filter(g, sig) > 0.5
        lab_, k_ = ndi.label(g)
        if k_:
            tam = ndi.sum(g, lab_, range(1, k_ + 1))
            g &= np.isin(lab_, 1 + np.nonzero(tam >= minimo)[0])
        return g[I, J]

    mi = np.full(len(Fq), 5, np.int32)        # mato e erva seca por omissão (encostas)
    verde = (o[:, 1] > o[:, 0] * 1.0) & (o[:, 1] > o[:, 2] * 1.06)
    mi[manchas(verde, 30)] = 1
    terra = (o[:, 0] > o[:, 1] * 1.12) & (sat > 0.10)
    mi[manchas(terra & ~verde, 25)] = 0
    # ruas e largos da vila (cinzentos pouco saturados na ortofoto): asfalto, ou calçada nas ruas estreitas
    decl = amostra(np.hypot(*np.gradient(ndi.gaussian_filter(MDT, 2.0), LP)), LX0, LY0, LP, cx[:, 0], cx[:, 1])
    cinza = (sat < 0.07) & (lum > 0.35) & (decl < 0.18)
    mi[manchas(cinza, 30, 1.2)] = 3
    # dentro da alcáçova: lajes de grés e saibro, relva onde está verde (jardim e canteiros)
    dc = shapely.contains_xy(ALCACOVA_PL, cx[:, 0], cx[:, 1])
    mi[dc] = 2
    mi[dc & manchas(verde, 12)] = 1
    # largo da estátua, à porta: calçada (fotografia 023)
    lg = (cx[:, 0] > -52) & (cx[:, 0] < -26) & (cx[:, 1] > -82) & (cx[:, 1] < -60) & ~dc
    mi[lg] = 4
    ob.data.polygons.foreach_set("material_index", mi)
    uv_mundo(ob, 2.0)
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = 0.35
    mod.use_collapse_triangulate = True
    vg = ob.vertex_groups.new(name="perto")
    perto = shapely.contains_xy(OBRA_PL.buffer(6.0), V[:, 0], V[:, 1])
    vg.add([int(i) for i in np.nonzero(perto)[0]], 1.0, "REPLACE")
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
    ang = np.linspace(0, 2 * math.pi, 361)
    borda = []
    for a_ in ang:
        p = BASE_C + (BASE_R - 0.5) * np.array([math.cos(a_), math.sin(a_)])
        borda.append((p[0], p[1], float(cota(*p))))
    zb = min(b[2] for b in borda) - 4.0
    for a_, b_ in zip(borda[:-1], borda[1:]):
        m.face([(a_[0], a_[1], zb), (b_[0], b_[1], zb), b_, a_], 0)
    m.face([(p[0], p[1], zb) for p in borda[:-1]][::-1], 0)
    m.objeto("base", None, GRUPOS["Terreno"], uv=None)


# ---------------------------------------------------------------- envolvente

def construir_envolvente():
    """A vila em volumes cinzentos: manchas da classe 6 do LiDAR (edifícios) com mais de 2,5 m, partidas em corpos
    pela altura do MDS (degraus de 2,5 m), cada corpo com cobertura plana à cota mediana do MDS. Ficam fora a
    alcáçova, a cerca e as suas torres (que a cerca já desenha) e os edifícios do recinto."""
    from rasterio import features
    from rasterio.transform import Affine
    h = MDS - MDT
    ed = ndi.binary_opening((CLS == 6) & (h > 2.5), iterations=1)
    xs = LX0 + np.arange(MDS.shape[1]) * LP
    ys = LY0 + np.arange(MDS.shape[0]) * LP
    GX, GY = np.meshgrid(xs, ys)
    fora = ~shapely.contains_xy(OBRA_PL.buffer(1.0), GX, GY)
    fora &= np.hypot(GX - BASE_C[0], GY - BASE_C[1]) < BASE_R - 2
    ed &= fora
    banda = np.where(ed, np.round(ndi.median_filter(MDS, 5) / 2.5), 0).astype(np.int32)
    lab = np.zeros(banda.shape, np.int32)
    nlab = 0
    for b in np.unique(banda[ed]):
        l_, k_ = ndi.label(ed & (banda == b))
        lab[l_ > 0] = l_[l_ > 0] + nlab
        nlab += k_
    tr = Affine(LP, 0, LX0 - LP / 2, 0, LP, LY0 - LP / 2)
    m = Malha(["cinza_viz"])
    n = 0
    for geo, val in features.shapes(lab, mask=lab > 0, transform=tr):
        p = shapely.geometry.shape(geo).buffer(0)
        if p.area < 8:
            continue
        p = p.simplify(0.5).buffer(0.3, join_style="mitre").buffer(-0.3, join_style="mitre")
        if p.geom_type != "Polygon" or p.area < 8:
            continue
        base = np.array(p.exterior.coords)[:-1]
        if len(base) < 3:
            continue
        if not shapely.Polygon(base).exterior.is_ccw:
            base = base[::-1]
        gx, gy = np.meshgrid(np.arange(*p.bounds[0::2], 0.5), np.arange(*p.bounds[1::2], 0.5))
        di = shapely.contains_xy(p, gx, gy)
        if di.sum() < 3:
            continue
        zs = amostra(MDS, LX0, LY0, LP, gx[di], gy[di])
        zt = amostra(MDT, LX0, LY0, LP, base[:, 0], base[:, 1])
        m.prisma(base, float(zt.min()) - 0.8, float(np.percentile(zs, 60)), 0)
        n += 1
    MEDIDAS["edificios_vila"] = n
    m.objeto("vila", "envolvente", GRUPOS["Envolvente"], uv=None)



# ---------------------------------------------------------------- árvores

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
print("torres…")
torres()
casa_guarda()
escada_adarve()
print("interior…")
cisterna()
estatua_mastro()
edificios_interior()
print("cerca…")
cerca()
print("terreno…")
construir_terreno()
print("vila…")
construir_envolvente()
print("árvores…")
construir_arvores()

PT = np.load(AQUI / "dados" / "castelo.npz")["pontos"]
_sel = shapely.contains_xy(TORRE_PL["menagem"], PT[:, 0], PT[:, 1]) & ~np.isin(PT[:, 3], (3, 4, 5, 7))
MEDIDAS["alcacova"] = {"area_m2": round(ALCACOVA_PL.area), "perimetro_m": round(ALCACOVA_PL.exterior.length),
                       "torres": len(TORRES), "ponto_mais_alto_menagem": round(float(np.percentile(PT[_sel, 2], 99)), 2)}
MEDIDAS["base"] = {"centro": BASE_C.tolist(), "raio": BASE_R}


def _centro(nome):
    return np.mean(np.array(TORRES[nome]["cantos"]), 0)


MEDIDAS["ancoras"] = {
    "menagem": _anc(*_centro("menagem"), TORRES["menagem"]["topo"] + 1.5),
    "porta": _anc(-32.6, -60.4, 52.0),
    "albarra_sueste": _anc(*_centro("albarra_sueste"), 56.5),
    "albarra_nordeste": _anc(*_centro("albarra_nordeste"), 63.0),
    "cisterna": _anc(31.7, 26.3, 55.5),
    "segredo": _anc(*_centro("segredo"), TORRES["segredo"]["topo"] + 1.5),
    "mulheres": _anc(*_centro("mulheres"), TORRES["mulheres"]["topo"] + 1.5),
    "porta_cidade": _anc(-141.5, -252.9, 36.0),
    "cerca_norte": _anc(-176.5, 29.0, 60.0),
}
MEDIDAS["texturas_foto"] = fontes_fotos()
(AQUI / "dados" / "medidas.json").write_text(json.dumps(MEDIDAS, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in MEDIDAS.items() if k not in ("ancoras", "texturas_foto", "torres", "panos")}, ensure_ascii=False))

# ---------------------------------------------------------------- exportação

SAIDA.mkdir(parents=True, exist_ok=True)
(AQUI / "saida").mkdir(exist_ok=True)
glb = AQUI / "saida" / "silves-bruto.glb"
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
    final = SAIDA / "silves.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    # texturas JPEG -> WebP (pipeline/kit/webp.py), sem tocar na geometria Draco
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / "kit" / "webp.py"), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / "silves.blend"))

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
        if cam_z is None:   # câmara à altura dos olhos sobre o chão da maqueta
            dentro_ = ALCACOVA_PL.contains(shapely.Point(*cam_xy))
            cam_z = float(chao_castelo_xy(*cam_xy) if dentro_ else cota(*cam_xy)) + 1.7
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

    vista("capa", (-95, -150), 135, (0, 0), 50, lente=34, res=(1280, 800), destino=SAIDA / "capa.jpg")
    if "--capa" not in sys.argv:
        # câmaras nas posições do GPS do Commons (posicoes.txt) e vistas de conferência
        vista("f023_porta", (-38.5, -74.2), None, (-30, -64), 55.0, lente=20, res=(800, 1100))
        vista("f004_recinto", (-31.5, 41.6), None, (-45, 46), 57.0, lente=26, res=(1100, 825))
        vista("f020_leste", (90, -25), 45.0, (35, -30), 50.0, lente=32, res=(1200, 800))
        vista("f019_sul", (-20, -260), 40.0, (0, 0), 55.0, lente=60, res=(1100, 733))
        vista("norte_baixo", (-30, 160), 60.0, (0, 30), 58.0, lente=32, res=(1200, 750))
        vista("topo", (0, 0), 260, (0, 0.01), 50, lente=30, res=(1000, 1000))
        vista("porta_cidade", (-150, -300), 35.0, (-141, -255), 26.0, lente=28, res=(1200, 800))
        vista("geral", (-110, -420), 300, (-110, -90), 30, lente=30, res=(1280, 900))
        vista("albarra_sueste", (60, -75), 50.0, (36, -50), 50.0, lente=28, res=(1200, 800))
        vista("f101_corredor", (-20.3, -64.8), 49.1, (-30, -64.8), 48.9, lente=22, res=(800, 1100))
        vista("f086_porta_cidade", (-170.0, -252.0), None, (-147, -253.5), 25.0, lente=30, res=(1100, 1000))
        vista("f081_cerca_sw", (-221.9, -258.2), None, (-245, -262), 24.0, lente=28, res=(1200, 800))
