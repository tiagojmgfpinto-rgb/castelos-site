"""Constrói a maqueta 3D do Forte de São Sebastião (Angra do Heroísmo, ilha Terceira), estado atual, em Blender, e
exporta-a para public/maquetas/sao-sebastiao-angra/sao-sebastiao-angra.glb (glTF binário com Draco).

Corre com o módulo bpy (Blender 5.2) instalado em Python 3.13:
    pip install bpy==5.2.3 numpy scipy shapely pillow rasterio
    python pipeline/maquetas/sao-sebastiao-angra/construir.py            # GLB
    python pipeline/maquetas/sao-sebastiao-angra/construir.py --vistas   # GLB + capa + vistas de verificação

Dados (em dados/): lidar.npz e superficie.npz (lidar.py: LiDAR 2024 da Região Autónoma dos Açores, bloco
24-4-2-5), planta.json (tracar.py: aresta de cima das faces de fora, ajustada ao LiDAR), orto.jpg (cor dos pontos).

Como se modela (a nuvem de pontos tem 35 milhões de pontos; a superfície do forte está em células de 25 cm):
- as faces de fora (escarpas em talude, cordão e parapeito) seguem a magistral medida, estação a estação de 25 cm:
  o pé vem do terreno (ou do mar) à frente, o topo e a espessura do parapeito, as canhoneiras e os merlões vêm da
  superfície do LiDAR por trás da face;
- tudo o que está dentro das muralhas (terraplenos, adarves, rampas, o recinto, a Pousada, a Casa do Governador, a
  bateria baixa) é a própria superfície do LiDAR de 25 cm, com os materiais pelo sítio e pela inclinação: alvenaria
  de basalto nas faces a prumo, reboco branco na Pousada, amarelo na Casa do Governador, telha, relva, tijoleira;
- peças à parte: o Portão de Armas com a ponte de dois arcos e as armas reais, a porta de dentro, as guaritas
  redondas dos salientes, as janelas e portas, o mar, o casario em volumes cinzentos e as árvores.
Coordenadas da cena: x para este, y para norte (PTRA08 / UTM 26N menos a origem), z = altitude, em metros (1:1).
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
from kit.geometria import normais, amostra, _anc  # noqa: E402
from kit.texturas import fbm, textura_relva, imagem  # noqa: E402
from kit.cena import cor_vertices, uv_mundo  # noqa: E402
from kit.fotos import textura_foto, fontes_fotos  # noqa: E402
from kit.silhares import textura_silhares  # noqa: E402
from kit.materiais import MATS, malha_rapida, material  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
ID = "sao-sebastiao-angra"
CFG = json.loads((RAIZ / "src" / "data" / "maquetas" / f"{ID}.json").read_text())
PLANTA = json.loads((AQUI / "dados" / "planta.json").read_text())
SAIDA = RAIZ / "public" / "maquetas" / ID
LID = np.load(AQUI / "dados" / "lidar.npz")
LP, LX0, LY0 = float(LID["passo"]), float(LID["x0"]), float(LID["y0"])
MDT = LID["mdt"] / 100.0
MDS = LID["mds"] / 100.0
CLS = LID["classe"]
SUPF = np.load(AQUI / "dados" / "superficie.npz")
SP, SX0, SY0 = float(SUPF["passo"]), float(SUPF["x0"]), float(SUPF["y0"])
SUP = ndi.median_filter(SUPF["z"] / 100.0, size=3)   # tira postes, pessoas e carros soltos
ORTO = np.asarray(Image.open(AQUI / "dados" / "orto.jpg").convert("RGB"), float) / 255.0  # 25 cm, norte em cima
OX0, OY1, OP = -200.0, 200.0, 0.25
rng = np.random.default_rng(1643)

# base redonda (por omissão desde 10 nov.): o forte inteiro, a bateria baixa, a marina a poente e o casario a norte
BASE_C = np.array([0.0, -22.0])
BASE_R = 112.0


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def cota(x, y):
    """Chão (MDT de 50 cm)."""
    return amostra(MDT, LX0, LY0, LP, x, y)


def sup(x, y):
    """Superfície do LiDAR (classes 1, 2, 6, 11, 17 e 19, sem vegetação), 25 cm."""
    return amostra(SUP, SX0, SY0, SP, x, y)


def orto(x, y, k=1):
    c = np.clip(np.round((np.asarray(x) - OX0) / OP).astype(int), 0, ORTO.shape[1] - 1)
    r = np.clip(np.round((OY1 - np.asarray(y)) / OP).astype(int), 0, ORTO.shape[0] - 1)
    return ORTO[r, c]


# ---------------------------------------------------------------- mar
# Sem pontos de água na nuvem (o voo não tem retorno no mar): o mar são as células sem classe e com o MDT a zero,
# ligadas ao largo. O nível é a mediana dos poucos retornos da superfície da água (ondas e espuma).
def _agua():
    w = (CLS == 0) & (MDT < 0.3)
    w = ndi.binary_opening(w, iterations=2)
    lab, k = ndi.label(w)
    w = np.isin(lab, 1 + np.nonzero(ndi.sum(w, lab, range(1, k + 1)) > 4000)[0])
    terra = ~w
    lab, k = ndi.label(terra)
    return w | np.isin(lab, 1 + np.nonzero(ndi.sum(terra, lab, range(1, k + 1)) < 60)[0])


AGUA = _agua()
_ret = MDS[AGUA & (MDS > 0.05) & (MDS < 2.0)]
NIVEL = float(np.clip(np.median(_ret) if len(_ret) > 50 else 0.4, 0.2, 0.8))
print("nível do mar", round(NIVEL, 2), "células de mar", int(AGUA.sum()))


def no_mar(x, y):
    return amostra(AGUA.astype(float), LX0, LY0, LP, x, y) > 0.5


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
            self.face([(p[0], p[1], z1[i] if np.ndim(z1) else z1) for i, p in enumerate(base)],
                      mat if mat_topo is None else mat_topo)
        if fundo:
            self.face([(p[0], p[1], z0[i] if np.ndim(z0) else z0) for i, p in reversed(list(enumerate(base)))], mat)

    def caixa(self, c, t, comp, larg, z0, z1, mat=0, topo=True, fundo=False):
        """Caixa de planta comp × larg centrada em c, com o comprimento ao longo de t."""
        t = unit(t)
        n = np.array([-t[1], t[0]])
        c = np.asarray(c, float)
        q = [c - t * comp / 2 - n * larg / 2, c + t * comp / 2 - n * larg / 2, c + t * comp / 2 + n * larg / 2,
             c - t * comp / 2 + n * larg / 2]
        self.prisma(q, z0, z1, mat, topo=topo, fundo=fundo)

    def anel(self, c, r0, z0, r1, z1, mat=0, k=20):
        """Tronco de cone (ou cilindro) aberto entre (r0, z0) e (r1, z1)."""
        ang = np.linspace(0, 2 * math.pi, k, endpoint=False)
        for j in range(k):
            a, b = ang[j], ang[(j + 1) % k]
            self.face([(c[0] + r0 * math.cos(a), c[1] + r0 * math.sin(a), z0), (c[0] + r0 * math.cos(b), c[1] + r0 * math.sin(b), z0),
                       (c[0] + r1 * math.cos(b), c[1] + r1 * math.sin(b), z1), (c[0] + r1 * math.cos(a), c[1] + r1 * math.sin(a), z1)], mat)

    def disco(self, c, r, z, mat=0, k=20, cima=True):
        ang = np.linspace(0, 2 * math.pi, k, endpoint=False)
        pts = [(c[0] + r * math.cos(a), c[1] + r * math.sin(a), z) for a in ang]
        self.face(pts if cima else pts[::-1], mat)

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


def decimar(ob, ratio, grupo=None, proteger=None):
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    if proteger is not None and len(proteger):
        vg = ob.vertex_groups.new(name="perto")
        vg.add([int(i) for i in proteger], 1.0, "REPLACE")
        mod.vertex_group = "perto"
        mod.invert_vertex_group = True
        mod.vertex_group_factor = 1.0
    dg = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear()
    mats = [m for m in ob.data.materials]
    ob.data = me2
    if not me2.materials:
        for m in mats:
            me2.materials.append(m)
    return me2


# ---------------------------------------------------------------- cena

for _ob in list(bpy.data.objects):  # a cena de arranque do bpy traz um cubo, uma luz e uma câmara
    bpy.data.objects.remove(_ob)
RAIZ_OB = bpy.data.objects.new("Forte_Sao_Sebastiao", None)
bpy.context.scene.collection.objects.link(RAIZ_OB)
RAIZ_OB["origem_lat"], RAIZ_OB["origem_lon"] = CFG["origem"]["lat"], CFG["origem"]["lon"]
RAIZ_OB["variante"] = CFG["variante"]
RAIZ_OB["versao"] = CFG["versao"]
GRUPOS = {}
# "Muralhas_grupo" e "Edificios_grupo" são o que o visor enquadra ao abrir; Terreno/Vegetacao/Envolvente ficam fora
for g in ["Terreno", "Muralhas", "Edificios", "Envolvente", "Vegetacao"]:
    e = bpy.data.objects.new(g + "_grupo", None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = RAIZ_OB
    GRUPOS[g] = e

# Pedra: alvenaria de basalto escuro (cinzento-negro, pedras irregulares com juntas de argamassa clara e manchas de
# líquen cinzento-esbranquiçado), como no Street View da frente de terra e nas fotografias do Commons (guaritas,
# panos do mar, Portão de Armas). Com uv=2,0 e escala=0,5 cada imagem cobre 4 m de parede.
TEX, ALT = textura_foto("old_stone_wall", n=1024, metros=4.0, cor=(0.37, 0.36, 0.34), contraste=1.35, saturacao=0.25,
                        variacao=0.12, liquen=0.10, liquen_cor=(0.70, 0.71, 0.66), semente=41)
material("alvenaria", (1, 1, 1), 0.93, imagem("alvenaria", TEX), normal=imagem("alvenaria_n", normais(ALT, 4.0)), forca=1.1,
         escala=0.5, vcor=True)
TEX, ALT = textura_foto("old_stone_wall", n=1024, metros=4.0, cor=(0.45, 0.44, 0.41), contraste=1.15, saturacao=0.2,
                        variacao=0.10, liquen=0.12, liquen_cor=(0.74, 0.74, 0.69), semente=42)
material("alvenaria_topo", (1, 1, 1), 0.95, imagem("alvenaria_topo", TEX), normal=imagem("alvenaria_topo_n", normais(ALT, 4.0)),
         forca=0.8, escala=0.5, vcor=True)
# cantaria: cordão, Portão de Armas, guaritas e cunhais em silhares de basalto e traquito cinzento mais claro
TEX, ALT = textura_silhares(n=1024, metros=4.0, fiada=(0.30, 0.45), comp=(0.45, 0.9), cor=(0.52, 0.51, 0.48), desvio=0.09,
                            junta_m=0.014, junta_cor=(0.66, 0.65, 0.62), liquen=0.05, irregular=0.12, semente=43)
material("cantaria", (1, 1, 1), 0.88, imagem("cantaria", TEX), normal=imagem("cantaria_n", normais(ALT, 4.0)), forca=0.9,
         escala=0.5, vcor=True)
material("cantaria_lisa", (0.62, 0.61, 0.58), 0.85)   # aduelas, ombreiras e moldura das armas (Street View)
material("reboco_branco", (0.90, 0.89, 0.86), 0.8)    # Pousada (fotosfera e fotografias)
material("ocre_casa", (0.93, 0.80, 0.47), 0.82)       # Casa do Governador (fotografia 18 do Commons)
material("moldura_branca", (0.94, 0.93, 0.90), 0.7)   # cunhais, cornijas e molduras dos vãos da Casa do Governador
material("cobertura", (0.36, 0.36, 0.37), 0.9)        # terraços e coberturas planas da Pousada
TEX, ALT = textura_foto("roof_09", n=512, metros=4.0, cor=(0.62, 0.33, 0.22), contraste=1.1, saturacao=0.5, variacao=0.08,
                        semente=44)
material("telha", (1, 1, 1), 0.8, imagem("telha", TEX), normal=imagem("telha_n", normais(ALT, 4.0)), forca=0.7, escala=0.5)
# tijoleira vermelha dos adarves e caminhos (fotosfera dentro da Pousada e fotografia 18)
TEX, ALT = textura_silhares(n=512, metros=2.0, fiada=(0.10, 0.10), comp=(0.20, 0.20), cor=(0.52, 0.30, 0.23), desvio=0.08,
                            junta_m=0.008, junta_cor=(0.45, 0.40, 0.36), liquen=0.0, irregular=0.05, semente=45)
material("tijoleira", (1, 1, 1), 0.9, imagem("tijoleira", TEX), normal=imagem("tijoleira_n", normais(ALT, 2.0)), forca=0.5, escala=1.0)
TEX, ALT = textura_foto("old_stone_wall", n=512, metros=2.0, metros_foto=1.0, cor=(0.62, 0.61, 0.58), contraste=1.1,
                        saturacao=0.2, variacao=0.06, semente=46)
material("calcada", (1, 1, 1), 0.92, imagem("calcada", TEX), normal=imagem("calcada_n", normais(ALT, 2.0)), forca=0.6, escala=1.0)
TEX, ALT = textura_relva()
material("relva", (1, 1, 1), 0.95, imagem("relva", TEX), normal=imagem("relva_n", normais(ALT, 1.0)), forca=0.4)
material("saibro", (0.50, 0.44, 0.39), 0.95)          # chão de terra batida (castanho-avermelhado, ortoimagem)
TEX, ALT = textura_foto("rock_01", n=512, metros=3.0, cor=(0.24, 0.24, 0.23), contraste=1.15, saturacao=0.2, semente=47)
material("rocha", (1, 1, 1), 0.85, imagem("rocha", TEX), normal=imagem("rocha_n", normais(ALT, 3.0)), forca=0.8)
material("mato", (0.42, 0.45, 0.28), 0.95)
material("asfalto", (0.33, 0.33, 0.34), 0.95)
material("cinza_viz", (0.74, 0.74, 0.73), 0.9)        # casario da envolvente: um só cinzento
material("base", (0.20, 0.19, 0.18), 0.95)
material("agua", (0.07, 0.24, 0.33), 0.12)
material("sombra", (0.04, 0.04, 0.04), 1.0)
material("madeira", (0.32, 0.20, 0.13), 0.8)
material("ferro", (0.12, 0.12, 0.13), 0.5)
material("vidro", (0.12, 0.15, 0.18), 0.15)
material("arvore", (1, 1, 1), 0.9, None, vcor=True)

MEDIDAS = {}


def patine(co):
    """Manchas largas (pedra lavada e escurecida), o pé mais escuro e húmido."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    g = fbm(x + 300, y + 300 + z * 0.8, 5.0, 41)
    v = 0.84 + 0.30 * g
    pe = np.clip(1 - (z - np.minimum(cota(x, y), 30)) / 2.5, 0, 1) * 0.10
    return np.clip(np.stack([v, v * 1.0, v * 0.98], 1) * (1 - pe)[:, None], 0, 1)


# ---------------------------------------------------------------- planta e estações das faces de fora

def _limpar(pts, minimo=1.0, fechada=True):
    """Tira os vértices que deixam lados com menos de `minimo` m (o traçado tem dois ou três quase repetidos)."""
    pts = [np.asarray(p, float) for p in pts]
    mudou = True
    while mudou and len(pts) > 3:
        mudou = False
        n = len(pts)
        for i in range(n if fechada else n - 1):
            j = (i + 1) % n
            if np.linalg.norm(pts[j] - pts[i]) < minimo:
                if not fechada and j == n - 1:
                    del pts[i]
                elif not fechada and i == 0:
                    del pts[1]
                else:
                    pts[i] = (pts[i] + pts[j]) / 2
                    del pts[j]
                mudou = True
                break
    return np.array(pts)


FORTE = _limpar(PLANTA["forte"])
if not shapely.Polygon(FORTE).exterior.is_ccw:
    FORTE = FORTE[::-1]
FORTE_PL = shapely.Polygon(FORTE)
BAIXO = _limpar(PLANTA["baixo"], fechada=False)
# a bateria baixa fecha-se pelos lados do forte que lhe ficam por cima (de onde acaba a linha até onde começa)
_ia = int(np.argmin(np.linalg.norm(FORTE - BAIXO[-1], axis=1)))
_ib = int(np.argmin(np.linalg.norm(FORTE - BAIXO[0], axis=1)))
_n = len(FORTE)
_c1 = [FORTE[(_ia + k) % _n] for k in range(1, (_ib - _ia) % _n)]
_c2 = [FORTE[(_ia - k) % _n] for k in range(1, (_ia - _ib) % _n)]
_fecho = min((_c1, _c2), key=lambda c: shapely.Polygon(list(BAIXO) + c).buffer(0).intersection(FORTE_PL).area
             if len(c) else 1e9)
BAIXO_PL = shapely.Polygon(list(BAIXO) + _fecho).buffer(0)
ZONA = shapely.union_all([FORTE_PL, BAIXO_PL]).buffer(0.05).buffer(-0.05)
print("forte", round(FORTE_PL.area), "m²; bateria baixa", round(BAIXO_PL.area), "m²")


class Linha:
    """Polilinha de uma face de fora, com estações de 25 cm. n = normal para fora; M = vetor de esquadria
    (com M·n = 1 nos dois lados de cada vértice, um ponto a d metros para fora fica a d metros de cada lado)."""

    def __init__(self, nome, pts, fechada, dentro_pl, passo=0.25):
        self.nome, self.fechada = nome, fechada
        P = np.asarray(pts, float)
        ne = len(P) if fechada else len(P) - 1
        T = np.array([unit(P[(i + 1) % len(P)] - P[i]) for i in range(ne)])
        N = np.stack([T[:, 1], -T[:, 0]], 1)
        meio = (P[0] + P[1]) / 2
        if dentro_pl.contains(shapely.Point(*(meio + N[0] * 0.8))):
            N = -N
        self.P, self.T, self.N = P, T, N
        Mv = []
        for v in range(len(P)):
            a = (v - 1) % ne if (fechada or v > 0) else None
            b = v if (fechada or v < ne) else None
            if a is None:
                Mv.append(N[b])
            elif b is None:
                Mv.append(N[a])
            else:
                s = N[a] + N[b]
                m = s / max(1 + N[a] @ N[b], 1e-3)
                ln = np.linalg.norm(m)
                Mv.append(m if ln < 3.0 else m / ln * 3.0)
        Mv = np.array(Mv)
        est = []
        for i in range(ne):
            A, B = P[i], P[(i + 1) % len(P)]
            L = np.linalg.norm(B - A)
            k = max(2, int(round(L / passo)))
            ts = np.linspace(0, 1, k + 1)
            if i < ne - 1 or fechada:
                ts = ts[:-1]
            for t in ts:
                est.append((i, t, A + t * (B - A), (1 - t) * Mv[i] + t * Mv[(i + 1) % len(P)]))
        if fechada:
            est.append(est[0])
        self.ed = np.array([e[0] for e in est])
        self.t = np.array([e[1] for e in est])
        self.p = np.array([e[2] for e in est])
        self.M = np.array([e[3] for e in est])
        self.n = N[self.ed]
        self.tg = T[self.ed]
        self.s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))])

    def xy(self, d):
        """Pontos das estações a d metros para fora (d < 0 para dentro); d pode ser um vetor (um por estação)."""
        d = np.asarray(d, float)
        return self.p + self.M * (d[:, None] if d.ndim else d)


LF = Linha("forte", FORTE, True, FORTE_PL)
LB = Linha("baixo", BAIXO, False, BAIXO_PL)


def perfil_dentro(L, ds, dentro_pl):
    """Superfície do LiDAR em cada estação, a cada distância ds para dentro, pela esquadria (nos ângulos a linha de
    medida vai pela bissetriz); os pontos que caem fora do polígono ficam NaN."""
    out = np.full((len(L.p), len(ds)), np.nan)
    dirs = -L.M / np.linalg.norm(L.M, axis=1)[:, None]
    for k, d in enumerate(ds):
        q = L.p + dirs * d
        z = sup(q[:, 0], q[:, 1])
        ok = shapely.contains_xy(dentro_pl.buffer(0.3), q[:, 0], q[:, 1])
        out[ok, k] = z[ok]
    return out


def runs(mask):
    """Corridas de True: lista de (início, fim exclusivo)."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def medir_linha(L, dentro_pl):
    """Para cada estação: topo do parapeito (com canhoneiras e merlões), adarve, espessura do parapeito e pé."""
    ds = np.arange(0.125, 8.0, 0.25)
    pr = perfil_dentro(L, ds, dentro_pl)
    pr[:, 0] = np.where(np.isnan(pr[:, 0]), sup(L.p[:, 0], L.p[:, 1]), pr[:, 0])
    pr = np.where(np.isnan(pr), np.nanmax(pr[:, :3], axis=1)[:, None], pr)
    n = len(L.p)
    walk_l = np.median(pr[:, (ds > 3.2) & (ds < 5.0)], axis=1)
    top_l = np.max(pr[:, ds < 1.3], axis=1)
    ne = int(L.ed.max()) + 1
    esp = np.zeros(n)
    tem_par = np.zeros(n, bool)
    for i in range(ne):
        sel = L.ed == i
        if sel.sum() < 4:
            continue
        mp = np.median(pr[sel], axis=0)
        w = float(np.median(mp[(ds > 3.2) & (ds < 7.5)]))
        tp = float(np.max(mp[ds < 1.3]))
        if tp - w < 0.35:
            continue
        lim = w + 0.4 * (tp - w)
        k = int(np.argmax(mp[ds < 1.3]))
        while k < len(ds) and mp[k] > lim:
            k += 1
        e = float(np.clip(ds[k] if k < len(ds) else 3.0, 0.5, 3.2))
        esp[sel], tem_par[sel] = e, True
    # adarve logo por dentro do parapeito
    walk = np.empty(n)
    top_raw = np.empty(n)
    for j in range(n):
        e = esp[j]
        sel = (ds > e + 0.4) & (ds < e + 1.8) if tem_par[j] else (ds > 0.6) & (ds < 2.0)
        walk[j] = np.median(pr[j, sel])
        top_raw[j] = np.max(pr[j, ds < max(e - 0.1, 0.3)]) if tem_par[j] else walk[j]
    walk = ndi.gaussian_filter1d(ndi.percentile_filter(walk, 70, size=21, mode="nearest"), 2, mode="nearest")
    # topo: merlões e canhoneiras pela queda do topo em relação ao alto dos vizinhos (6 m)
    alto = ndi.percentile_filter(top_raw, 85, size=25, mode="nearest")
    baixo_ = top_raw < walk + 0.5 * (alto - walk)
    vao = np.zeros(n, bool)
    for a, b in runs(baixo_ & tem_par & (alto - walk > 0.6)):
        if 2 <= b - a <= 32:
            vao[a:b] = True
    topo = np.where(tem_par, ndi.median_filter(top_raw, 7, mode="nearest"), walk)
    for a, b in runs(vao):
        topo[a:b] = max(float(np.median(top_raw[a:b])), float(np.median(walk[a:b])) + 0.3)
    for a, b in runs(tem_par & ~vao):
        if b - a <= 16:   # merlão ou troço curto: topo plano
            topo[a:b] = float(np.median(top_raw[a:b]))
    topo = np.maximum(topo, walk + np.where(tem_par, 0.3, 0.0))
    # cordão: c. 1,3 m abaixo do topo do parapeito, quase à cota do adarve (Street View, frente de terra)
    ref = ndi.percentile_filter(np.where(tem_par, topo - 1.3, walk), 80, size=41, mode="nearest")
    zc = np.where(tem_par, np.clip(np.minimum(ndi.gaussian_filter1d(ref, 3), walk + 0.2), topo - 1.6, topo - 0.4), walk)
    return dict(topo=topo, walk=walk, esp=esp, tem_par=tem_par, vao=vao, zc=zc, top_raw=top_raw)


def pe_linha(L, med, talude, chao_fora):
    """Pé da escarpa: o chão (ou o mar) à frente, um pouco mais fundo para a escarpa entrar no terreno."""
    h0 = med["zc"] - 8.0
    for _ in range(2):
        off = talude * np.clip(med["zc"] - h0, 0, 40)
        zs = []
        for dd in (0.4, 0.8, 1.2, 1.6):
            q = L.xy(off + dd)
            zs.append(chao_fora(q[:, 0], q[:, 1]))
        h0 = np.median(np.array(zs), axis=0)
    pe = ndi.gaussian_filter1d(ndi.minimum_filter1d(h0, 5, mode="nearest"), 2, mode="nearest") - 0.6
    return pe, talude * np.clip(med["zc"] - pe, 0, 40)


SUP_MIN = ndi.minimum_filter(SUP, size=9)


def chao_fora(x, y):
    """Chão à frente de uma face: o MDT, o fundo da bateria baixa (superfície do LiDAR) ou o mar."""
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    z = cota(x, y)
    inb = shapely.contains_xy(BAIXO_PL.buffer(-0.2), x, y)
    z = np.where(inb, amostra(SUP_MIN, SX0, SY0, SP, x, y), z)
    return np.where(no_mar(x, y), NIVEL - 1.2, z)


def construir_linha(L, med, pe, off, nome):
    m = Malha(["alvenaria", "alvenaria_topo", "cantaria"])
    zc, topo, walk, esp, tp = med["zc"], med["topo"], med["walk"], med["esp"], med["tem_par"]
    Pp = L.xy(off)          # pé
    P0 = L.p                # aresta da face (cordão)
    Pc = L.xy(0.13)         # frente do cordão
    for j in range(len(L.p) - 1):
        k = j + 1
        a0, b0 = Pp[j], Pp[k]
        m.face([(*a0, pe[j]), (*b0, pe[k]), (*P0[k], zc[k]), (*P0[j], zc[j])], 0)
        if tp[j] and tp[k]:
            ej, ek = esp[j], esp[k]
            Ij, Ik = P0[j] - L.M[j] * ej, P0[k] - L.M[k] * ek
            m.face([(*P0[j], zc[j] - 0.10), (*P0[k], zc[k] - 0.10), (*Pc[k], zc[k] - 0.02), (*Pc[j], zc[j] - 0.02)], 2)
            m.face([(*Pc[j], zc[j] - 0.02), (*Pc[k], zc[k] - 0.02), (*Pc[k], zc[k] + 0.10), (*Pc[j], zc[j] + 0.10)], 2)
            m.face([(*Pc[j], zc[j] + 0.10), (*Pc[k], zc[k] + 0.10), (*P0[k], zc[k] + 0.16), (*P0[j], zc[j] + 0.16)], 2)
            m.face([(*P0[j], zc[j]), (*P0[k], zc[k]), (*P0[k], topo[k]), (*P0[j], topo[j])], 0)
            m.face([(*P0[j], topo[j]), (*P0[k], topo[k]), (*Ik, topo[k]), (*Ij, topo[j])], 1)
            m.face([(*Ij, topo[j]), (*Ik, topo[k]), (*Ik, walk[k] - 0.4), (*Ij, walk[j] - 0.4)], 0)
            # topos a cotas diferentes (merlões, canhoneiras): faces a prumo de lado
            if abs(topo[k] - topo[j]) > 0.15 and (j + 2 >= len(L.p) or True):
                pass
        else:
            Ij, Ik = P0[j] - L.M[j] * 0.6, P0[k] - L.M[k] * 0.6
            m.face([(*P0[j], zc[j]), (*P0[k], zc[k]), (*Ik, zc[k]), (*Ij, zc[j])], 1)
            m.face([(*Ij, zc[j]), (*Ik, zc[k]), (*Ik, zc[k] - 1.0), (*Ij, zc[j] - 1.0)], 0)
        # fecho onde o parapeito começa ou acaba
        if tp[j] != tp[k]:
            q = k if tp[k] else j
            Iq = P0[q] - L.M[q] * esp[q]
            m.face([(*P0[q], walk[q] - 0.4), (*Iq, walk[q] - 0.4), (*Iq, topo[q]), (*P0[q], topo[q])], 0)
    if not L.fechada:   # pontas da bateria baixa: a escarpa fecha-se contra o forte
        for q in (0, len(L.p) - 1):
            Iq = P0[q] - L.M[q] * max(esp[q], 0.6)
            m.face([(*Pp[q], pe[q]), (*P0[q], zc[q]), (*P0[q], topo[q]), (*Iq, topo[q]), (*Iq, pe[q])], 0)
    return m.objeto("Muralha_" + nome, "muralhas", GRUPOS["Muralhas"], uv=2.0, vcor=patine)


print("faces de fora…")
MED_F = medir_linha(LF, FORTE_PL)
PE_F, OFF_F = pe_linha(LF, MED_F, 0.18, chao_fora)
MED_B = medir_linha(LB, BAIXO_PL)
PE_B, OFF_B = pe_linha(LB, MED_B, 0.18, chao_fora)
for nome, L, med in (("forte", LF, MED_F), ("baixo", LB, MED_B)):
    print(nome, "estações", len(L.p), "com parapeito", int(med["tem_par"].sum()), "vãos", len(runs(med["vao"])),
          "topo", round(float(med["topo"].min()), 2), round(float(med["topo"].max()), 2))

DIAG = r'''
for nome, L in (("forte", LF), ("baixo", LB)):
    for i, P in enumerate(L.P):
        j = np.nonzero(L.ed == i)[0]
        print(nome, i, P.round(2), "s", round(float(L.s[j[0]]), 1) if len(j) else "-")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for nome, L, med, pe in (("forte", LF, MED_F, PE_F), ("baixo", LB, MED_B, PE_B)):
    n = int(np.ceil(L.s[-1] / 120))
    fig, axs = plt.subplots(n, 1, figsize=(22, 3.2 * n), dpi=60)
    axs = np.atleast_1d(axs)
    for k, ax in enumerate(axs):
        sel = (L.s >= k * 120) & (L.s < (k + 1) * 120)
        ax.plot(L.s[sel], med["top_raw"][sel], "k-", lw=0.6)
        ax.plot(L.s[sel], med["topo"][sel], "r-", lw=1)
        ax.plot(L.s[sel], med["walk"][sel], "g-", lw=1)
        ax.plot(L.s[sel], med["zc"][sel], "b-", lw=1)
        ax.plot(L.s[sel], pe[sel], "m-", lw=1)
        for v in np.nonzero(np.diff(L.ed[sel]))[0]:
            ax.axvline(L.s[sel][v + 1], color="gray", lw=0.5)
            ax.text(L.s[sel][v + 1], ax.get_ylim()[1] if False else float(np.nanmax(med["top_raw"][sel])), str(L.ed[sel][v + 1]), fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/tmp/claude-0/ss/pl/diag_{nome}.jpg")
for nome, L, med in (("forte", LF, MED_F),):
    for j in range(0, len(L.p), 40):
        print(nome, j, L.ed[j], L.p[j].round(1), "s", round(float(L.s[j]), 1), "esp", med["esp"][j], "par", med["tem_par"][j],
              "top", round(float(med["top_raw"][j]), 2), "topo", round(float(med["topo"][j]), 2), "walk", round(float(med["walk"][j]), 2))
sys.exit()

'''
if "--diag" in sys.argv:
    exec(compile(DIAG, "diag", "exec"))


# ---------------------------------------------------------------- zonas dos edifícios (pela ortoimagem e pelo LiDAR)
# Pousada (reboco branco e coberturas planas cinzentas): o bloco noroeste com o restaurante de fachada curva, a ala
# poente e o bloco sudoeste. Casa do Governador (amarela, telhado de telha em duas águas duplas). Casa da guarda sobre
# a passagem do Portão de Armas (telhado de telha de uma água, a descer para o recinto).
POUSADA = shapely.union_all([
    shapely.Polygon([(-36.4, 34.0), (-28.8, 34.0), (-28.8, 29.7), (-10.6, 29.7), (-10.6, 21.0), (-12.5, 18.8),
                     (-22.0, 16.8), (-32.5, 13.2), (-36.4, 13.2)]),
    shapely.Polygon([(-36.6, 13.4), (-28.0, 13.4), (-28.0, -24.5), (-36.6, -24.5)]),
    shapely.Polygon([(-41.8, -21.5), (-30.0, -21.5), (-30.0, -30.8), (-41.8, -30.8)]),
])
CASA = shapely.Polygon([(8.7, 0.3), (18.6, -7.3), (34.2, 13.0), (32.8, 16.0), (17.8, 16.6), (16.4, 10.2)])
# No saliente sudoeste o LiDAR tem uma peça alongada (c. 3 m ao longo de NNE, 1 m de largo), maior do que as guaritas:
# é a espadana de dois sinos que se vê na fotografia de Diego Delso (Commons, 2020) sobre o baluarte do lado do mar.
ESPADANA = (-55.0, -19.9)   # centro detetado; a sineira fica centrada na mancha, em (-56; -20)
ESPADANA_EIXO = (0.34, 0.94)


def _raio_g(c, base):
    return base * 1.0 if np.hypot(c[0] - ESPADANA[0], c[1] - ESPADANA[1]) < 3.0 else base


GUARDA = shapely.Polygon([(-12.5, 21.2), (3.6, 15.8), (4.2, 24.0), (-11.8, 29.3)])


def lista_estacoes():
    """Todas as estações das duas linhas, com o que é preciso para rebaixar a superfície junto às faces."""
    P, D, N, E, W, Z, PE, OF, F = [], [], [], [], [], [], [], [], []
    for L, med, pe, off, fo in ((LF, MED_F, PE_F, OFF_F, True), (LB, MED_B, PE_B, OFF_B, False)):
        P.append(L.p)
        D.append(-L.M / np.linalg.norm(L.M, axis=1)[:, None])
        N.append(L.n)
        E.append(np.where(med["tem_par"], med["esp"], 0.0))
        W.append(med["walk"])
        Z.append(med["zc"])
        PE.append(pe)
        OF.append(off)
        F.append(np.full(len(L.p), fo))
    return {k: np.concatenate(v) for k, v in zip("P D N E W Z PE OF F".split(), (P, D, N, E, W, Z, PE, OF, F))}


EST = lista_estacoes()
KD = cKDTree(EST["P"])


def rebaixar(X, Y, Z, dentro=True):
    """Superfície junto às faces de fora: por baixo do parapeito fica à cota do adarve (o parapeito modelado tapa-a);
    à frente das escarpas (bateria baixa e terreno) fica à cota do pé, para não furar o talude."""
    q = np.stack([X.ravel(), Y.ravel()], 1)
    dist, j = KD.query(q, k=3)
    Zr = Z.ravel().copy()
    for c in range(3):
        jj = j[:, c]
        v = q - EST["P"][jj]
        di = np.einsum("ij,ij->i", v, EST["D"][jj])       # para dentro
        perto = dist[:, c] < 4.5
        if dentro:
            par = perto & (di >= -0.05) & (di < EST["E"][jj] + 0.15) & (EST["E"][jj] > 0)
            Zr = np.where(par, np.minimum(Zr, EST["W"][jj] - 0.05), Zr)
            sem = perto & (di >= -0.05) & (di < 0.65) & (EST["E"][jj] == 0)
            Zr = np.where(sem, np.minimum(Zr, EST["Z"][jj] - 0.03), Zr)
        fr = perto & (di < 0.05) & (-di < EST["OF"][jj] + 1.2)
        if dentro:
            fr &= EST["F"][jj] & ~shapely.contains_xy(FORTE_PL, q[:, 0], q[:, 1])
        Zr = np.where(fr, np.minimum(Zr, EST["PE"][jj] + 0.75), Zr)
    return Zr.reshape(Z.shape)


def construir_superficie():
    """O forte por dentro: a superfície do LiDAR de 25 cm (terraplenos, adarves, rampas, recinto, Pousada, Casa do
    Governador, bateria baixa), com o material de cada face pelo sítio e pela inclinação."""
    fin = shapely.Polygon(LF.xy(-np.where(MED_F["tem_par"], MED_F["esp"] - 0.3, 0.3))).buffer(0)
    bin_ = shapely.Polygon(list(LB.xy(-np.where(MED_B["tem_par"], MED_B["esp"] - 0.3, 0.3))) + _fecho).buffer(0)
    clip = shapely.union_all([fin, bin_]).buffer(0)
    x0, y0, x1, y1 = ZONA.bounds
    xs = np.arange(np.floor(x0) - 1, x1 + 1, SP)
    ys = np.arange(np.floor(y0) - 1, y1 + 1, SP)
    X, Y = np.meshgrid(xs, ys)
    Z = sup(X, Y)
    Z = rebaixar(X, Y, Z, dentro=True)
    for c, zt, _ in GUAR:   # a cúpula da guarita não é chão
        k = _raio_g(c, 1.0)
        g = np.hypot(X - c[0], Y - c[1]) < 1.4 * k
        if g.any():
            anel = (np.hypot(X - c[0], Y - c[1]) > 1.6 * k) & (np.hypot(X - c[0], Y - c[1]) < 2.6 * k)
            Z[g] = np.minimum(Z[g], float(np.median(Z[anel])) if anel.any() else zt - 3.3)
    cx, cy = (X[:-1, :-1] + X[1:, 1:]) / 2, (Y[:-1, :-1] + Y[1:, 1:]) / 2
    ok = shapely.contains_xy(clip, cx, cy)
    I, J = np.nonzero(ok)
    usa = np.zeros(X.shape, bool)
    for di_, dj_ in ((0, 0), (0, 1), (1, 0), (1, 1)):
        usa[I + di_, J + dj_] = True
    idx = -np.ones(X.shape, int)
    idx[usa] = np.arange(usa.sum())
    V = np.stack([X[usa], Y[usa], Z[usa]], 1)
    Fq = np.stack([idx[I, J], idx[I, J + 1], idx[I + 1, J + 1], idx[I + 1, J]], 1)
    # materiais por face
    mats = ["alvenaria", "reboco_branco", "ocre_casa", "telha", "cobertura", "relva", "tijoleira", "calcada", "saibro",
            "alvenaria_topo"]
    zc4 = V[Fq][:, :, 2]
    fx, fy = cx[I, J], cy[I, J]
    a = V[Fq[:, 1]] - V[Fq[:, 0]]
    b = V[Fq[:, 3]] - V[Fq[:, 0]]
    nz = np.abs(np.cross(a, b)[:, 2]) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-9)
    ingreme = (nz < 0.55) | (zc4.max(1) - zc4.min(1) > 0.6)
    zf = zc4.mean(1)
    chao_local = amostra(ndi.grey_opening(SUP, size=(41, 41)), SX0, SY0, SP, fx, fy)
    alto = zf > chao_local + 2.2
    em_p = shapely.contains_xy(POUSADA, fx, fy)
    em_c = shapely.contains_xy(CASA, fx, fy)
    em_g = shapely.contains_xy(GUARDA, fx, fy)
    o = ndi.uniform_filter(ORTO, size=(3, 3, 1))
    oc = o[np.clip(np.round((OY1 - fy) / OP).astype(int), 0, ORTO.shape[0] - 1), np.clip(np.round((fx - OX0) / OP).astype(int), 0, ORTO.shape[1] - 1)]
    lum, sat = oc.mean(1), oc.max(1) - oc.min(1)
    verde = (oc[:, 1] > oc[:, 0] * 1.04) & (oc[:, 1] > oc[:, 2] * 1.04)
    tijolo = (oc[:, 0] > oc[:, 1] * 1.12) & (oc[:, 0] > oc[:, 2] * 1.2)
    claro = (lum > 0.55) & (sat < 0.12)
    mi = np.full(len(Fq), 8, np.int32)
    mi[claro] = 7
    mi[tijolo] = 6
    mi[verde] = 5
    mi[em_p & alto] = 4
    mi[(em_c | em_g) & alto] = 3
    mi[ingreme] = 0
    mi[ingreme & em_p] = 1
    mi[ingreme & em_c] = 2
    ob = malha_rapida("Forte_superficie", V, Fq, mats[0], GRUPOS["Muralhas"], "muralhas")
    for mt in mats[1:]:
        ob.data.materials.append(MATS[mt])
    ob.data.polygons.foreach_set("material_index", mi)
    MEDIDAS["superficie_celulas"] = int(len(Fq))
    uv_superficie(ob)
    me2 = decimar(ob, 0.35)
    uv_superficie(ob)
    cor_vertices(me2, patine)
    return ob, clip


def uv_superficie(ob):
    """UV em metros: chão em planta; faces a prumo ao longo da face (a direção da face vem da superfície alisada, para
    a textura não saltar nos degraus de 25 cm da grelha)."""
    me = ob.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    vi = np.empty(len(me.loops), int)
    me.loops.foreach_get("vertex_index", vi)
    nrm = np.empty(len(me.polygons) * 3)
    me.polygons.foreach_get("normal", nrm)
    lt = np.empty(len(me.polygons), int)
    me.polygons.foreach_get("loop_total", lt)
    n = np.repeat(nrm.reshape(-1, 3), lt, axis=0)
    p = co[vi]
    gy, gx = np.gradient(ndi.gaussian_filter(SUP, 3.0), SP)
    ux = amostra(gx, SX0, SY0, SP, p[:, 0], p[:, 1])
    uy = amostra(gy, SX0, SY0, SP, p[:, 0], p[:, 1])
    h = np.maximum(np.hypot(ux, uy), 1e-6)
    hz = np.abs(n[:, 2]) > 0.6
    u = np.where(hz, p[:, 0], (p[:, 0] * -uy + p[:, 1] * ux) / h)
    v = np.where(hz, p[:, 1], p[:, 2])
    me.uv_layers.active.data.foreach_set("uv", (np.stack([u, v], 1) / 2.0).ravel().astype(np.float32))


def detetar_guaritas():
    """Guaritas: nos salientes (ângulo < 125°) onde o LiDAR tem uma cúpula pequena, 3 a 6 m acima do adarve, a menos de
    6 m do vértice. Devolve (centro, cota do alto da cúpula, para fora). O parapeito à volta fica à cota dos vizinhos."""
    feitas = []
    for L, med in ((LF, MED_F), (LB, MED_B)):
        P = L.P
        for v in range(len(P)):
            if not L.fechada and v in (0, len(P) - 1):
                continue
            a, b = P[v - 1] - P[v], P[(v + 1) % len(P)] - P[v]
            ang = math.degrees(math.acos(np.clip(unit(a) @ unit(b), -1, 1)))
            js = np.nonzero(np.linalg.norm(L.p - P[v], axis=1) < 6)[0]
            if not len(js) or ang > 125:
                continue
            fora = L.M[js[np.argmin(np.linalg.norm(L.p[js] - P[v], axis=1))]]
            if not ZONA.buffer(0.5).contains(shapely.Point(*(P[v] - unit(fora) * 1.5))):
                continue
            ref = float(np.median(med["walk"][js]))
            gx, gy = np.meshgrid(np.arange(P[v][0] - 6, P[v][0] + 6, SP), np.arange(P[v][1] - 6, P[v][1] + 6, SP))
            zz = sup(gx, gy)
            perto = np.hypot(gx - P[v][0], gy - P[v][1]) < 6
            zt = float(zz[perto].max())
            sel = perto & (zz > zt - 1.0)
            if not (2.4 < zt - ref < 6.0) or not (6 <= sel.sum() <= 140):
                continue
            c = np.array([gx[sel].mean(), gy[sel].mean()])
            if any(np.linalg.norm(c - f) < 4 for f, _, _ in feitas):
                continue
            feitas.append((c, zt, unit(fora)))
            # o parapeito à volta da guarita: à cota dos vizinhos (o LiDAR apanha a guarita no topo)
            rg = _raio_g(c, 2.2)
            ja = np.nonzero(np.linalg.norm(L.p - c, axis=1) < rg)[0]
            jb = np.nonzero((np.linalg.norm(L.p - c, axis=1) >= rg) & (np.linalg.norm(L.p - c, axis=1) < rg + 3))[0]
            if len(ja) and len(jb):
                med["topo"][ja] = min(float(np.median(med["topo"][jb])), float(np.median(med["topo"][ja])))
    print("guaritas", [(f.round(1).tolist(), round(z, 2)) for f, z, _ in feitas])
    return feitas


GUAR = detetar_guaritas()
MURO_F = construir_linha(LF, MED_F, PE_F, OFF_F, "forte")
MURO_B = construir_linha(LB, MED_B, PE_B, OFF_B, "bateria_baixa")
print("superfície do forte…")
SUPERF, CLIP = construir_superficie()


# ---------------------------------------------------------------- terreno, mar e base

def construir_terreno():
    passo = 0.5
    xs = np.arange(BASE_C[0] - BASE_R, BASE_C[0] + BASE_R + 1e-6, passo)
    ys = np.arange(BASE_C[1] - BASE_R, BASE_C[1] + BASE_R + 1e-6, passo)
    X, Y = np.meshgrid(xs, ys)
    Z = cota(X, Y)
    ag = amostra(AGUA.astype(float), LX0, LY0, LP, X, Y) > 0.5
    dist = ndi.distance_transform_edt(ag) * passo
    Z = np.where(ag, NIVEL - np.clip(0.6 + 0.3 * dist, 0.6, 8.0), Z)
    Z = np.where(~ag & (Z < NIVEL + 0.05), NIVEL + 0.05, Z)
    Z = rebaixar(X, Y, Z, dentro=False)
    # a rocha da costa no MDT de 0,5 m sai em picos de um pixel: suaviza-se longe das muralhas
    longe = ~shapely.contains_xy(ZONA.buffer(3.0), X, Y) & ~ag
    Zs = ndi.gaussian_filter(Z, 1.3)
    Z = np.where(longe & (Z < 14), Zs, Z)
    # o MDE do voo tem a ponte do Portão de Armas como chão: por baixo dela fica o fundo do fosso dos dois lados
    j0 = int(np.argmin(np.linalg.norm(LF.p - np.array([1.0, 29.6]), axis=1)))
    n_, t_ = LF.n[j0], LF.tg[j0]
    pp = LF.p[j0]
    ponte = shapely.Polygon([pp + n_ * 0.3 - t_ * 3.4, pp + n_ * 11.5 - t_ * 3.4, pp + n_ * 11.5 + t_ * 3.4, pp + n_ * 0.3 + t_ * 3.4])
    lados = [float(cota(*(pp + n_ * d + t_ * sg * 4.5))) for d in np.arange(1, 10, 1.0) for sg in (-1, 1)]
    sob = shapely.contains_xy(ponte, X, Y)
    Z[sob] = np.minimum(Z[sob], float(np.median(lados)))
    r = np.hypot(X - BASE_C[0], Y - BASE_C[1])
    cxg, cyg = (X[:-1, :-1] + X[1:, 1:]) / 2, (Y[:-1, :-1] + Y[1:, 1:]) / 2
    dentro_q = (np.hypot(cxg - BASE_C[0], cyg - BASE_C[1]) < BASE_R - 0.3) & ~shapely.contains_xy(ZONA.buffer(-0.4), cxg, cyg)
    I, J = np.nonzero(dentro_q)
    usa = np.zeros(X.shape, bool)
    for a_, b_ in ((0, 0), (0, 1), (1, 0), (1, 1)):
        usa[I + a_, J + b_] = True
    idx = -np.ones(X.shape, int)
    idx[usa] = np.arange(usa.sum())
    V = np.stack([X[usa], Y[usa], Z[usa]], 1)
    Fq = np.stack([idx[I, J], idx[I, J + 1], idx[I + 1, J + 1], idx[I + 1, J]], 1)
    mats = ["mato", "relva", "rocha", "saibro", "asfalto", "calcada"]
    ob = malha_rapida("Terreno", V, Fq, mats[0], GRUPOS["Terreno"], "terreno")
    for mt in mats[1:]:
        ob.data.materials.append(MATS[mt])
    fx, fy = cxg[I, J], cyg[I, J]
    o = ndi.uniform_filter(ORTO, size=(5, 5, 1))
    oc = o[np.clip(np.round((OY1 - fy) / OP).astype(int), 0, ORTO.shape[0] - 1), np.clip(np.round((fx - OX0) / OP).astype(int), 0, ORTO.shape[1] - 1)]
    lum, sat = oc.mean(1), oc.max(1) - oc.min(1)
    zf = V[Fq][:, :, 2].mean(1)
    decl = amostra(np.hypot(*np.gradient(ndi.gaussian_filter(MDT, 1.0), LP)), LX0, LY0, LP, fx, fy)
    mi = np.full(len(Fq), 3, np.int32)
    mi[(lum < 0.42) & (sat < 0.08)] = 4
    mi[(lum > 0.55) & (sat < 0.10)] = 5
    mi[(oc[:, 1] > oc[:, 0] * 1.03) & (oc[:, 1] > oc[:, 2] * 1.03)] = 1
    rocha = ((decl > 0.9) & (zf < 20)) | ((zf < NIVEL + 6) & (lum < 0.35)) | (zf < NIVEL + 0.2)
    mi[rocha] = 2
    ob.data.polygons.foreach_set("material_index", mi)
    uv_mundo(ob, 2.0)
    decimar(ob, 0.35)
    MEDIDAS["terreno_triangulos"] = len(ob.data.polygons)
    # lados da base redonda
    m = Malha(["base"])
    zb = NIVEL - 9.0
    borda = []
    for a in np.linspace(0, 2 * math.pi, 361):
        p = BASE_C + (BASE_R - 0.3) * np.array([math.cos(a), math.sin(a)])
        z = NIVEL - 1.5 if no_mar([p[0]], [p[1]])[0] else max(float(cota(*p)), NIVEL + 0.05)
        borda.append((p[0], p[1], z))
    for a, b in zip(borda[:-1], borda[1:]):
        m.face([(a[0], a[1], zb), (b[0], b[1], zb), b, a], 0)
    m.face([(p[0], p[1], zb) for p in borda[:-1]][::-1], 0)
    m.objeto("Terreno_base", None, GRUPOS["Terreno"], uv=None)


def construir_mar():
    ag = ndi.binary_dilation(AGUA, iterations=4)
    passo = 2.0
    xs = np.arange(BASE_C[0] - BASE_R, BASE_C[0] + BASE_R + 1e-6, passo)
    ys = np.arange(BASE_C[1] - BASE_R, BASE_C[1] + BASE_R + 1e-6, passo)
    X, Y = np.meshgrid(xs, ys)
    usa = (amostra(ag.astype(float), LX0, LY0, LP, X, Y) > 0.01) & (np.hypot(X - BASE_C[0], Y - BASE_C[1]) < BASE_R - 0.4)
    idx = -np.ones(X.shape, int)
    idx[usa] = np.arange(usa.sum())
    V = np.stack([X[usa], Y[usa], np.full(usa.sum(), NIVEL)], 1)
    I, J = np.nonzero(usa[:-1, :-1] & usa[1:, :-1] & usa[:-1, 1:] & usa[1:, 1:])
    Fq = np.stack([idx[I, J], idx[I, J + 1], idx[I + 1, J + 1], idx[I + 1, J]], 1)
    ob = malha_rapida("Terreno_mar", V, Fq, "agua", GRUPOS["Terreno"], "mar")
    decimar(ob, 0.25)
    ob.data.materials.clear()
    ob.data.materials.append(MATS["agua"])
    MEDIDAS["mar"] = {"nivel": round(NIVEL, 2)}


print("terreno e mar…")
construir_terreno()
construir_mar()


# ---------------------------------------------------------------- guaritas

def guarita(m, c, piso, para_fora):
    """Guarita redonda dos salientes (fotografias 15, 21, 29 e 30 do Commons e Street View): mísula em cone aos
    degraus, corpo cilíndrico de cantaria com três frestas e a porta para dentro, cornija, cúpula em meia laranja e
    remate em bola. Medidas pelas fotografias; a cota vem do ponto mais alto da cúpula no LiDAR."""
    u = unit(para_fora)
    rb = 0.82
    for r0, r1, z0, z1 in ((0.95, 0.95, piso - 0.30, piso), (0.80, 0.95, piso - 0.62, piso - 0.30),
                           (0.62, 0.80, piso - 0.95, piso - 0.62), (0.40, 0.62, piso - 1.30, piso - 0.95),
                           (0.12, 0.40, piso - 1.65, piso - 1.30)):
        m.anel(c, r0, z0, r1, z1, 0)
        m.anel(c, r1, z1, r1 * 0.97, z1, 0)
    m.disco(c, 0.12, piso - 1.65, 0, cima=False)
    zt = piso + 2.05
    m.anel(c, rb, piso, rb, zt, 1, k=24)
    m.anel(c, rb, zt, 0.98, zt, 0, k=24)
    m.anel(c, 0.98, zt, 0.98, zt + 0.16, 0, k=24)
    w = np.array([-u[1], u[0]])
    for ang, larg, z0, z1, mat in ((0, 0.11, piso + 0.8, piso + 1.5, 2), (math.radians(70), 0.11, piso + 0.8, piso + 1.5, 2),
                                   (math.radians(-70), 0.11, piso + 0.8, piso + 1.5, 2),
                                   (math.pi, 0.55, piso + 0.02, piso + 1.75, 2)):
        d = u * math.cos(ang) + w * math.sin(ang)
        m.caixa(c + d * (rb + 0.005), np.array([-d[1], d[0]]), larg, 0.03, z0, z1, mat)
    prev_r, prev_z = 0.98, zt + 0.16
    for k in range(1, 7):
        fi = math.pi / 2 * k / 6
        r_, z_ = 0.92 * math.cos(fi), zt + 0.16 + 0.85 * math.sin(fi)
        if k == 1:
            m.anel(c, prev_r, prev_z, 0.92, prev_z, 0, k=24)
            prev_r = 0.92
        m.anel(c, prev_r, prev_z, max(r_, 0.02), z_, 0, k=24)
        prev_r, prev_z = r_, z_
    m.anel(c, 0.10, prev_z - 0.02, 0.10, prev_z + 0.15, 0, k=10)
    for k in range(6):   # bola
        a0, a1 = math.pi * k / 6 - math.pi / 2, math.pi * (k + 1) / 6 - math.pi / 2
        zc_ = prev_z + 0.15 + 0.17
        m.anel(c, 0.17 * math.cos(a0), zc_ + 0.17 * math.sin(a0), 0.17 * math.cos(a1), zc_ + 0.17 * math.sin(a1), 0, k=10)
    return prev_z + 0.49




def espadana(m, c, zt):
    """Sineira do baluarte sudoeste: torrinha de alvenaria de c. 3,6 × 3,4 m com dois vãos em arco para os sinos
    virados para fora e para dentro, cornija de cantaria e cruz no remate. A planta e o topo vêm da mancha do LiDAR a
    35 m (c. 4 × 4 m); os vãos e a cruz estão postos pela fotografia de Diego Delso (Commons, 2020)."""
    t = np.array(ESPADANA_EIXO, float); t /= np.linalg.norm(t)
    n = np.array([-t[1], t[0]])
    c = np.array([-56.0, -20.0])
    z0, z1 = zt - 4.6, zt
    m.caixa(c, t, 3.6, 3.4, z0, z1, 1)
    m.caixa(c, t, 3.8, 3.6, z1 - 0.25, z1, 0)
    m.caixa(c, t, 3.5, 3.3, z1, z1 + 0.06, 0)
    m.caixa(c, t, 0.7, 0.5, z1 + 0.06, z1 + 0.4, 0)
    m.caixa(c, t, 0.14, 0.14, z1 + 0.4, z1 + 1.2, 0)
    m.caixa(c, t, 0.55, 0.14, z1 + 0.82, z1 + 0.96, 0)
    for sgn in (-1, 1):
        for lado in (-1, 1):
            cc = c + t * sgn * 0.85 + n * lado * 1.72
            pts = [(cc[0] + t[0] * s_, cc[1] + t[1] * s_, z) for s_, z in contorno_arco(0.8, z1 - 2.25, z1 - 1.25)]
            m.face(pts if lado > 0 else pts[::-1], 2)


def construir_guaritas(det):
    m = Malha(["cantaria", "alvenaria", "sombra"])
    MEDIDAS["guaritas"] = []
    MEDIDAS["espadana"] = None
    for c, zt, fora in det:
        if np.hypot(c[0] - ESPADANA[0], c[1] - ESPADANA[1]) < 3.0:
            espadana(m, c, zt)
            MEDIDAS["espadana"] = {"centro": c.round(2).tolist(), "topo_lidar": round(zt, 2)}
            continue
        topo = guarita(m, c, zt - 3.25, fora)
        MEDIDAS["guaritas"].append({"centro": c.round(2).tolist(), "topo_lidar": round(zt, 2), "topo": round(topo, 2)})
    if det:
        m.objeto("Guaritas", "guaritas", GRUPOS["Muralhas"], uv=1.5, vcor=patine)


# ---------------------------------------------------------------- Portão de Armas, ponte e porta de dentro

def ponto_escarpa(j, s, z, extra=0.0):
    """Ponto na face da escarpa (em talude) da estação j, deslocado s ao longo da face, à cota z."""
    zc, pe, off = MED_F["zc"][j], PE_F[j], OFF_F[j]
    d = off * np.clip((zc - z) / max(zc - pe, 0.1), 0, 1) if z < zc else 0.0
    q = LF.p[j] + LF.tg[j] * s + LF.n[j] * (d + extra)
    return (q[0], q[1], z)


def contorno_arco(w, z0, zn, k=12):
    """Contorno de um vão de volta perfeita: (s, z) do pé esquerdo, pelo arco, ao pé direito."""
    r = w / 2
    pts = [(-r, z0)]
    for a in np.linspace(math.pi, 0, k + 1):
        pts.append((r * math.cos(a), zn + r * math.sin(a)))
    pts.append((r, z0))
    return pts


def portao_armas():
    m = Malha(["cantaria_lisa", "sombra", "cantaria", "madeira", "alvenaria", "calcada"])
    G = np.array([1.0, 29.6])
    j0 = int(np.argmin(np.linalg.norm(LF.p - G, axis=1)))
    n, t = LF.n[j0], LF.tg[j0]
    # deck da ponte (LiDAR: 27,6 m) e fim da ponte do lado de fora (onde a superfície cai)
    deck = float(np.median([sup(*(LF.p[j0] + n * d + t * s)) for d in (4, 5, 6) for s in (-0.8, 0, 0.8)]))
    fim = 4.0
    while fim < 18 and sup(*(LF.p[j0] + n * fim)) > deck - 1.0:
        fim += 0.25
    fim = min(fim, 10.5)   # Street View: a ponte acaba no contraforte do fosso; depois segue o caminho em aterro
    W, sole, nasc = 3.0, deck, deck + 1.9
    # vão escuro e moldura de cantaria (aduelas e ombreiras), na face em talude
    cont = contorno_arco(W, sole, nasc)
    m.face([ponto_escarpa(j0, s, z, 0.03) for s, z in cont], 1)
    ext = contorno_arco(W + 0.9, sole, nasc)
    for (s0, z0), (s1, z1), (S0, Z0), (S1, Z1) in zip(cont[:-1], cont[1:], ext[:-1], ext[1:]):
        m.face([ponto_escarpa(j0, s0, z0, 0.06), ponto_escarpa(j0, S0, Z0 if z0 > sole else sole, 0.06),
                ponto_escarpa(j0, S1, Z1 if z1 > sole else sole, 0.06), ponto_escarpa(j0, s1, z1, 0.06)], 0)
    # pedra de fecho e cornija por cima do arco
    zf = nasc + W / 2
    m.face([ponto_escarpa(j0, -0.25, zf - 0.1, 0.09), ponto_escarpa(j0, 0.25, zf - 0.1, 0.09),
            ponto_escarpa(j0, 0.3, zf + 0.55, 0.09), ponto_escarpa(j0, -0.3, zf + 0.55, 0.09)], 0)
    # armas reais no alto da cortina, sobre o arco (Street View): moldura com frontão e o escudo coroado
    zc = MED_F["zc"][j0]
    tp = MED_F["topo"][j0]
    za, zb = zc - 1.1, tp + 0.25
    for (s0, s1, z0, z1, ex, mat) in ((-0.85, 0.85, za, zb, 0.14, 0), (-0.55, 0.55, za + 0.25, zb - 0.35, 0.2, 2)):
        pts = [ponto_escarpa(j0, s0, z0, ex), ponto_escarpa(j0, s1, z0, ex), ponto_escarpa(j0, s1, z1, ex), ponto_escarpa(j0, s0, z1, ex)]
        m.face(pts, mat)
        base = [ponto_escarpa(j0, s0, z0, 0.0), ponto_escarpa(j0, s1, z0, 0.0), ponto_escarpa(j0, s1, z1, 0.0), ponto_escarpa(j0, s0, z1, 0.0)]
        for a_ in range(4):
            b_ = (a_ + 1) % 4
            m.face([base[a_], base[b_], pts[b_], pts[a_]], mat)
    m.face([ponto_escarpa(j0, -0.95, zb, 0.14), ponto_escarpa(j0, 0.95, zb, 0.14), ponto_escarpa(j0, 0.0, zb + 0.5, 0.14)], 0)
    m.face([ponto_escarpa(j0, -0.95, zb, 0.0), ponto_escarpa(j0, -0.95, zb, 0.14), ponto_escarpa(j0, 0.0, zb + 0.5, 0.14), ponto_escarpa(j0, 0.0, zb + 0.5, 0.0)], 0)
    m.face([ponto_escarpa(j0, 0.95, zb, 0.14), ponto_escarpa(j0, 0.95, zb, 0.0), ponto_escarpa(j0, 0.0, zb + 0.5, 0.0), ponto_escarpa(j0, 0.0, zb + 0.5, 0.14)], 0)
    # ponte de dois arcos (Street View e LiDAR): tabuleiro, guardas baixas com capeamento, pilar ao meio
    d0 = OFF_F[j0] * np.clip((zc - deck) / max(zc - PE_F[j0], 0.1), 0, 1) - 0.3
    fundo = float(min(cota(*(LF.p[j0] + n * d)) for d in np.arange(d0 + 0.5, fim, 0.5))) - 0.3
    larg = 3.4

    def P(d, s, z):
        q = LF.p[j0] + n * d + t * s
        return (q[0], q[1], z)

    m.prisma([P(d0, -larg / 2, 0)[:2], P(fim, -larg / 2, 0)[:2], P(fim, larg / 2, 0)[:2], P(d0, larg / 2, 0)[:2]],
             deck - 0.55, deck, 4, mat_topo=5)
    for sg in (-1, 1):   # guardas
        s0, s1 = sg * larg / 2, sg * (larg / 2 - 0.45)
        m.prisma([P(d0, s0, 0)[:2], P(fim, s0, 0)[:2], P(fim, s1, 0)[:2], P(d0, s1, 0)[:2]][:: 1 if sg < 0 else -1], deck, deck + 0.75, 4)
        m.prisma([P(d0, s0 + sg * 0.05, 0)[:2], P(fim, s0 + sg * 0.05, 0)[:2], P(fim, s1 - sg * 0.05, 0)[:2],
                  P(d0, s1 - sg * 0.05, 0)[:2]][:: 1 if sg < 0 else -1], deck + 0.75, deck + 0.9, 2)
    vaos = np.linspace(d0, fim, 3)
    meio = (vaos[0] + vaos[-1]) / 2
    for a, b in ((vaos[0], meio - 0.5), (meio + 0.5, vaos[-1])):
        # pilares e encontros por baixo, arcos abatidos entre eles
        pass
    for dd0, dd1 in ((d0 - 0.4, d0 + 0.2), (meio - 0.5, meio + 0.5), (fim - 0.6, fim + 0.3)):
        m.prisma([P(dd0, -larg / 2, 0)[:2], P(dd1, -larg / 2, 0)[:2], P(dd1, larg / 2, 0)[:2], P(dd0, larg / 2, 0)[:2]],
                 fundo, deck - 0.55, 4, topo=False)
    for a, b in ((d0 + 0.2, meio - 0.5), (meio + 0.5, fim - 0.6)):
        r = (b - a) / 2
        zn = max(deck - 0.55 - r * 0.75, fundo + 0.3)
        arco = [(a + r - r * math.cos(f), zn + r * 0.75 * math.sin(f)) for f in np.linspace(0, math.pi, 11)]
        for (x0, z0), (x1, z1) in zip(arco[:-1], arco[1:]):   # intradorso
            m.face([P(x0, larg / 2, z0), P(x1, larg / 2, z1), P(x1, -larg / 2, z1), P(x0, -larg / 2, z0)], 2)
        for sg in (-1, 1):   # tímpanos
            pts = [P(x, sg * larg / 2, z) for x, z in arco] + [P(b, sg * larg / 2, deck - 0.55), P(a, sg * larg / 2, deck - 0.55)]
            m.face(pts if sg > 0 else pts[::-1], 4)
            m.face([P(x, sg * larg / 2, fundo) for x, _ in (arco[0], arco[-1])] + [P(b, sg * larg / 2, zn), P(a, sg * larg / 2, zn)], 1)
    # porta de dentro: a saída da passagem abobadada na fachada da casa da guarda, para o recinto
    k = 2.0
    while k < 20 and sup(*(LF.p[j0] - n * k)) > deck + 0.3:
        k += 0.25
    Pi = LF.p[j0] - n * (k - 0.25)
    chao_i = float(sup(*(LF.p[j0] - n * (k + 1.5))))
    # pano de alvenaria da fachada (tapa os degraus da grelha) até ao alto da face medida
    alto_i = float(np.percentile([sup(*(Pi + n * 0.6 + t * sv)) for sv in np.arange(-4, 4.1, 0.5)], 30))
    pan = [Pi + t * sv - n * 0.22 for sv in (-4.5, 4.5)]
    m.face([(*pan[0], chao_i - 0.3), (*pan[1], chao_i - 0.3), (*pan[1], alto_i - 0.05), (*pan[0], alto_i - 0.05)][::-1], 4)
    Pi = Pi - n * 0.22
    cont = contorno_arco(W, chao_i, chao_i + 2.1)
    q = [(Pi + t * s - n * 0.03) for s, _ in cont]
    m.face([(*qq, z) for qq, (_, z) in zip(q, cont)][::-1], 1)
    ext = contorno_arco(W + 0.8, chao_i, chao_i + 2.1)
    for (s0, z0), (s1, z1), (S0, Z0), (S1, Z1) in zip(cont[:-1], cont[1:], ext[:-1], ext[1:]):
        a0, a1 = Pi + t * s0 - n * 0.06, Pi + t * s1 - n * 0.06
        b0, b1 = Pi + t * S0 - n * 0.06, Pi + t * S1 - n * 0.06
        m.face([(*a0, z0), (*a1, z1), (*b1, max(Z1, chao_i)), (*b0, max(Z0, chao_i))], 0)
    MEDIDAS["portao_armas"] = {"centro": LF.p[j0].round(2).tolist(), "soleira": round(deck, 2), "ponte_m": round(fim - d0, 1),
                               "passagem_m": round(k, 1), "chao_recinto": round(chao_i, 2), "fundo_fosso": round(fundo, 2)}
    print("portão", MEDIDAS["portao_armas"])
    return m.objeto("Portao_Armas", "portao", GRUPOS["Muralhas"], uv=1.5, vcor=patine)


print("guaritas e Portão de Armas…")
construir_guaritas(GUAR)
PORTAO = portao_armas()


# ---------------------------------------------------------------- casario e árvores

def construir_envolvente():
    """Casario de Angra e edifícios do porto: manchas da classe 6 com mais de 2,5 m, fora do forte, em volumes
    cinzentos com a cobertura à cota do MDS (percentil 70)."""
    from rasterio import features
    from rasterio.transform import Affine
    h = MDS - MDT
    ed = ndi.binary_opening((CLS == 6) & (h > 2.5), iterations=1)
    lab, k = ndi.label(ed)
    tr = Affine(LP, 0, LX0 - LP / 2, 0, LP, LY0 - LP / 2)
    m = Malha(["cinza_viz"])
    n = 0
    longe = ZONA.buffer(2.5)
    for geo, val in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=tr):
        p = shapely.geometry.shape(geo).buffer(0)
        p = p.difference(longe).buffer(-0.3).buffer(0.3)
        for g in getattr(p, "geoms", [p]):
            g = g.simplify(0.4)
            if g.geom_type != "Polygon" or g.area < 12 or g.centroid.distance(shapely.Point(*BASE_C)) > BASE_R - 2:
                continue
            g = g.intersection(shapely.Point(*BASE_C).buffer(BASE_R - 0.6))
            if g.geom_type != "Polygon" or g.area < 12:
                continue
            base = np.array(g.exterior.coords)[:-1]
            if not shapely.Polygon(base).exterior.is_ccw:
                base = base[::-1]
            gx, gy = np.meshgrid(np.arange(*g.bounds[0::2], 0.5), np.arange(*g.bounds[1::2], 0.5))
            di = shapely.contains_xy(g, gx, gy)
            if di.sum() < 4:
                continue
            zs = amostra(MDS, LX0, LY0, LP, gx[di], gy[di])
            zt = amostra(MDT, LX0, LY0, LP, gx[di], gy[di])
            m.prisma(base, float(np.percentile(zt, 5)) - 0.5, float(np.percentile(zs, 70)), 0)
            n += 1
    MEDIDAS["edificios_envolvente"] = n
    if n:
        m.objeto("Envolvente_casario", "envolvente", GRUPOS["Envolvente"], uv=2.0)


def construir_arvores():
    """Árvores: as copas do LiDAR (classes 3 a 5) dentro da base, fora dos muros, em dois lobos."""
    V, Fc, C = [], [], []
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    v0 = np.array([x.co[:] for x in bm.verts])
    f0 = [[x.index for x in fa.verts] for fa in bm.faces]
    bm.free()
    muros = shapely.union_all([shapely.LineString(LF.p).buffer(3.0), shapely.LineString(LB.p).buffer(3.0)])
    n = 0
    for x, y, topo, _, r in LID["arvores"]:
        x, y = float(x), float(y)
        if math.hypot(x - BASE_C[0], y - BASE_C[1]) > BASE_R - 3 or muros.contains(shapely.Point(x, y)):
            continue
        zt = float(sup(x, y)) if ZONA.contains(shapely.Point(x, y)) else float(cota(x, y))
        alt = float(topo) - zt
        if alt < 2.5:
            continue
        rad = float(np.clip(r, 1.2, 5.0))
        tom = np.array([0.20, 0.31, 0.14]) * rng.uniform(0.85, 1.15)
        for (dx, dy, dz, s_) in ((0, 0, -0.35, 1.0), (0.35, -0.25, -0.55, 0.75)):
            b = len(V)
            pts = (v0 + rng.normal(0, 0.06, v0.shape)) * np.array([rad * s_, rad * s_, alt * 0.32 * s_]) + \
                np.array([x + dx * rad, y + dy * rad, float(topo) + dz * alt * 0.45])
            V.extend(map(tuple, pts))
            Fc.extend([[b + i for i in fa] for fa in f0])
            tt = np.clip((pts[:, 2] - zt) / max(alt, 0.1), 0, 1)
            C.extend(map(tuple, tom[None, :] * (0.55 + 0.5 * tt)[:, None]))
        b = len(V)
        for j in range(5):
            a = 2 * math.pi * j / 5
            V.append((x + 0.18 * math.cos(a), y + 0.18 * math.sin(a), zt - 0.2))
            V.append((x + 0.13 * math.cos(a), y + 0.13 * math.sin(a), float(topo) - alt * 0.45))
        for j in range(5):
            jj = (j + 1) % 5
            Fc.append([b + 2 * j, b + 2 * jj, b + 2 * jj + 1, b + 2 * j + 1])
        C.extend([(0.33, 0.27, 0.21)] * 10)
        n += 1
    MEDIDAS["arvores"] = n
    if not n:
        return
    me = bpy.data.meshes.new("arvores")
    me.from_pydata(V, [], Fc)
    me.update()
    me.materials.append(MATS["arvore"])
    at = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    at.data.foreach_set("color", np.concatenate([np.array(C), np.ones((len(C), 1))], 1).astype(np.float32).ravel())
    me.shade_smooth()
    ob = bpy.data.objects.new("Vegetacao_arvores", me)
    bpy.context.scene.collection.objects.link(ob)
    ob["elemento"] = "vegetacao"
    ob.parent = GRUPOS["Vegetacao"]



# ---------------------------------------------------------------- fachadas dos edifícios

def face_medida(A, B, fora_ref):
    """Face a prumo de um edifício perto da linha A→B: a aresta medida na superfície do LiDAR (onde cai mais para o lado
    de fora), o alto (beirado ou platibanda) e o chão à frente. fora_ref: um ponto do lado de fora."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    t = unit(B - A)
    n = np.array([-t[1], t[0]])
    if (np.asarray(fora_ref) - A) @ n < 0:
        n = -n
    L = np.linalg.norm(B - A)
    ds = np.arange(-2.5, 2.5, 0.125)
    pts, tops, chaos = [], [], []
    for sv in np.arange(0.6, L - 0.6, 0.5):
        q = A + t * sv
        v = sup(q[0] + n[0] * ds, q[1] + n[1] * ds)
        g = -np.gradient(v, 0.125)
        k = int(np.argmax(g))
        if g[k] > 2.0:
            pts.append((sv, ds[k]))
            tops.append(float(np.median([sup(*(q + n * (ds[k] - e))) for e in (0.5, 0.8, 1.1)])))
            chaos.append(float(np.median([sup(*(q + n * (ds[k] + e))) for e in (1.2, 1.6, 2.0)])))
    if len(pts) < 4:
        return None
    Pq = np.array(pts)
    c = np.polyfit(Pq[:, 0], Pq[:, 1], 1)
    for _ in range(2):
        r = np.abs(np.polyval(c, Pq[:, 0]) - Pq[:, 1])
        w = r < max(0.3, np.percentile(r, 70))
        c = np.polyfit(Pq[w, 0], Pq[w, 1], 1)
    o0, o1 = np.polyval(c, 0), np.polyval(c, L)
    if abs(o0 - o1) > 0.8:   # o traçado aproximado já tem a direção certa: só se desloca
        o0 = o1 = float(np.median(Pq[:, 1]))
    A2, B2 = A + n * o0, B + n * o1
    return dict(A=A2, B=B2, n=n, t=t, L=L, topo=float(np.percentile(tops, 30)), chao=float(np.median(chaos)))


def fachada(m, f, mat, vaos, desloc=0.16, moldura="moldura_branca", cunhais=False, cornija=False):
    """Pano liso à frente da face medida (tapa os degraus de 25 cm da grelha), com os vãos: (s, z0, largura, altura,
    arco, tipo) e, se pedido, cunhais e cornija brancos (Casa do Governador)."""
    A, B, n, t, L = f["A"], f["B"], f["n"], f["t"], f["L"]
    z0, z1 = f["chao"] - 0.3, f["topo"] - 0.04

    def P(s_, z, d=0.0):
        q = A + t * s_ + n * (desloc + d)
        return (q[0], q[1], z)

    m.face([P(0, z0), P(L, z0), P(L, z1), P(0, z1)], mat)
    m.face([P(0, z0, -desloc - 0.3), P(0, z0), P(0, z1), P(0, z1, -desloc - 0.3)], mat)
    m.face([P(L, z0), P(L, z0, -desloc - 0.3), P(L, z1, -desloc - 0.3), P(L, z1)], mat)
    for (sv, za, w, h, arco, tipo) in vaos:
        mt = "sombra" if tipo == "porta" else "vidro"
        k = 8 if arco else 1
        cont = [(sv - w / 2, za), (sv + w / 2, za)]
        if arco:
            cont += [(sv + w / 2 * math.cos(a), za + h - w / 2 + w / 2 * math.sin(a)) for a in np.linspace(0, math.pi, k + 1)]
        else:
            cont += [(sv + w / 2, za + h), (sv - w / 2, za + h)]
        m.face([P(a_, b_, 0.01) for a_, b_ in cont], mt)
        # moldura
        e = 0.14
        out = [(sv - w / 2 - e, za - (e if tipo != "porta" else 0)), (sv + w / 2 + e, za - (e if tipo != "porta" else 0))]
        if arco:
            out += [(sv + (w / 2 + e) * math.cos(a), za + h - w / 2 + (w / 2 + e) * math.sin(a)) for a in np.linspace(0, math.pi, k + 1)]
        else:
            out += [(sv + w / 2 + e, za + h + e), (sv - w / 2 - e, za + h + e)]
        cc = cont + [cont[0]]
        oo = out + [out[0]]
        for i in range(len(cont)):
            if tipo == "porta" and i == 0:
                continue
            m.face([P(*cc[i], 0.03), P(*oo[i], 0.03), P(*oo[i + 1], 0.03), P(*cc[i + 1], 0.03)], moldura)
    if cunhais:
        for s0 in (0.0, L - 0.45):
            m.face([P(s0, z0, 0.02), P(s0 + 0.45, z0, 0.02), P(s0 + 0.45, z1, 0.02), P(s0, z1, 0.02)], moldura)
        m.face([P(0, f["chao"] - 0.3, 0.02), P(L, f["chao"] - 0.3, 0.02), P(L, f["chao"] + 0.45, 0.02), P(0, f["chao"] + 0.45, 0.02)], moldura)
    if cornija:
        m.face([P(-0.1, z1 - 0.35, 0.03), P(L + 0.1, z1 - 0.35, 0.03), P(L + 0.1, z1, 0.12), P(-0.1, z1, 0.12)], moldura)


def janelas(f, passo=3.0, larg=1.0, alt=1.35, porta=None, pisos=None, margem=1.2):
    """Vãos em fiada a passo regular, por piso (3,2 m de pé-direito); porta = s da porta de arco no rés do chão."""
    L = f["L"]
    h = f["topo"] - f["chao"]
    npisos = pisos or max(1, min(3, int((h - 0.6) // 3.2)))
    n_ = max(1, int((L - 2 * margem) // passo) + 1)
    ss = np.linspace(margem, L - margem, n_) if n_ > 1 else [L / 2]
    v = []
    for k in range(npisos):
        za = f["chao"] + 0.9 + 3.2 * k
        for sv in ss:
            if porta is not None and k == 0 and abs(sv - porta) < passo * 0.6:
                continue
            v.append((sv, za, larg, alt, False, "janela"))
    if porta is not None:
        v.append((porta, f["chao"], 1.6, 2.9, True, "porta"))
    return v


def construir_fachadas():
    """Fachadas medidas na superfície do LiDAR, com os vãos pelas fotografias (estimativa: número e ritmo das janelas
    pela fotografia 18 da Casa do Governador e pela fotosfera da Pousada; não há alçados publicados)."""
    m = Malha(["ocre_casa", "moldura_branca", "reboco_branco", "alvenaria", "vidro", "sombra"])
    feitas = {}
    # Casa do Governador: fachada sudoeste (lado curto, para o relvado), sueste (comprida) e noroeste
    for nome, A, B, fora, porta in (("casa_so", (9.2, 0.2), (18.6, -6.9), (10, -8), "meio"),
                                    ("casa_se", (18.6, -6.9), (33.6, 13.0), (30, 0), None),
                                    ("casa_no", (9.2, 0.2), (16.6, 9.8), (8, 8), None)):
        f = face_medida(A, B, fora)
        if f is None:
            continue
        fachada(m, f, "ocre_casa", janelas(f, porta=f["L"] / 2 if porta else None), cunhais=True, cornija=True)
        feitas[nome] = f
    # Pousada: ala poente (fachada para o recinto), bloco noroeste (fachada sul em dois troços) e bloco sudoeste
    for nome, A, B, fora in (("pousada_ala", (-28.0, 8.0), (-28.0, -21.0), (-20, -6)),
                             ("pousada_no1", (-32.0, 13.6), (-22.0, 16.7), (-25, 8)),
                             ("pousada_no2", (-22.0, 16.7), (-12.5, 18.9), (-15, 10)),
                             ("pousada_so", (-30.0, -21.8), (-30.0, -30.6), (-24, -26))):
        f = face_medida(A, B, fora)
        if f is None or f["topo"] - f["chao"] < 2.4:
            continue
        fachada(m, f, "reboco_branco", janelas(f, passo=2.6, larg=1.1, alt=1.5), moldura="reboco_branco")
        feitas[nome] = f
    MEDIDAS["fachadas"] = {k: {"de": v["A"].round(2).tolist(), "a": v["B"].round(2).tolist(), "alto": round(v["topo"], 2),
                               "chao": round(v["chao"], 2)} for k, v in feitas.items()}
    print("fachadas", {k: (round(v["topo"], 1), round(v["chao"], 1), round(v["L"], 1)) for k, v in feitas.items()})
    if feitas:
        m.objeto("Edificios_fachadas", "edificios", GRUPOS["Edificios"], uv=2.0)


print("fachadas…")
construir_fachadas()

print("casario e árvores…")
construir_envolvente()
construir_arvores()


# ---------------------------------------------------------------- exportação


def _olhar(x, y, up=0.35):
    d = unit((x, y))
    return [round(float(d[0]), 2), up, round(float(-d[1]), 2)]


MEDIDAS["forte"] = {"area_m2": round(FORTE_PL.area), "perimetro_m": round(FORTE_PL.exterior.length),
                    "bateria_baixa_m2": round(BAIXO_PL.area)}
MEDIDAS["base"] = {"centro": BASE_C.tolist(), "raio": BASE_R}
MEDIDAS["texturas_foto"] = fontes_fotos()
(AQUI / "dados" / "medidas.json").write_text(json.dumps(MEDIDAS, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in MEDIDAS.items() if k not in ("ancoras", "texturas_foto")}, ensure_ascii=False))

SAIDA.mkdir(parents=True, exist_ok=True)
(AQUI / "saida").mkdir(exist_ok=True)
glb = AQUI / "saida" / f"{ID}-bruto.glb"
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
    final = SAIDA / f"{ID}.glb"
    subprocess.run(["npx", "-y", "@gltf-transform/cli@4.2.1", "draco", str(glb), str(final)], check=True)
    # texturas JPEG -> WebP (pipeline/kit/webp.py), sem tocar na geometria Draco
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / "kit" / "webp.py"), str(final)], check=True)
    print("GLB Draco:", final, round(final.stat().st_size / 1e3), "kB")
if "--blend" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=str(AQUI / "saida" / f"{ID}.blend"))

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

    vista("capa", (110, -120), 120, (0, -20), 24, lente=36, res=(1280, 800), destino=SAIDA / "capa.jpg")
    if "--capa" not in sys.argv:
        vista("topo", (0, -22), 260, (0, -21.99), 20, lente=30, res=(1000, 1000))
        vista("norte_sv", (-7.5, 70.6), 27.5, (2, 30), 30, lente=28, res=(1200, 800))
        vista("noroeste", (-90, 110), 70, (-10, 0), 25, lente=32, res=(1200, 800))
        vista("sudeste_mar", (90, -110), 45, (-5, -20), 20, lente=32, res=(1200, 800))
        vista("poente_marina", (-150, -40), 40, (-20, -20), 22, lente=34, res=(1200, 800))
        vista("recinto", (-2, -45), 34, (5, 10), 30, lente=26, res=(1200, 800))
        vista("portao", (-12, 46), 33, (1.5, 31), 28, lente=30, res=(1200, 800))
        vista("porta_dentro", (-8, 4), 29, (-2, 17), 28.5, lente=28, res=(1200, 800))
        vista("guarita_nw", (-52, 70), 40, (-38.8, 60.2), 36, lente=40, res=(1000, 800))
        vista("frente_poente", (-112, 52), 16, (-46, 12), 28, lente=30, res=(1400, 700))
        vista("espadana", (-72, -2), 42, (-55, -19.9), 33, lente=45, res=(1000, 800))
        vista("portao_fora", (1.5, 46), 29.5, (1.0, 30), 30, lente=30, res=(900, 1100))
