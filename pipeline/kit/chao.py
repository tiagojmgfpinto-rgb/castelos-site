"""Chão por classes nítidas ("metro a metro", novembro de 2026).

O chão das maquetas deixa de ser uma malha única pintada ponto a ponto com a cor da ortofoto (que dava manchas,
sombras pintadas e estradas em rocha). Passa a ser feito de polígonos de classe (asfalto, calçada, terra, relva,
mato, rocha...) traçados sobre a ortofoto e o LiDAR, e cada classe é um objeto com o seu material nítido. As
fronteiras entre classes ficam exatas, à resolução do polígono e não da malha.

Uso, num construir.py (depois de ter a grelha Z do terreno, com os retoques que a maqueta faça):

    from kit import chao
    classes = chao.ler("dados/chao.json")                   # {classe: polígono shapely}, em metros da cena
    pecas = chao.malhas(classes, xs, Z, raio=RAIO, passo=1.0)  # {classe: (V, F)}
    for nome, (V, F) in pecas.items(): ...                    # um objeto por classe, com o material da classe

O dados/chao.json faz-se com o chao.py de cada maqueta: uma proposta automática (ortofoto + classes do LiDAR +
declive, filtrada por maioria e passada a polígonos) e por cima as correções traçadas à mão, quadrícula a
quadrícula, que mandam sempre. A ortofoto serve só para decidir as fronteiras: nunca entra como textura.
"""
import json
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage as ndi

# Ordem de prioridade: quando dois polígonos se sobrepõem, fica a classe que vem primeiro.
PRIORIDADE = ["edificio", "agua", "asfalto", "calcada", "lajeado", "escadas", "saibro", "terra", "rocha", "relva", "erva", "mato"]


# ---------------------------------------------------------------- polígonos

def resolver(classes, ordem=PRIORIDADE, dentro=None):
    """Torna os polígonos disjuntos pela ordem de prioridade e recorta-os a `dentro`."""
    out, ocupado = {}, shapely.Polygon()
    nomes = [n for n in ordem if n in classes] + [n for n in classes if n not in ordem]
    for n in nomes:
        g = shapely.make_valid(classes[n])
        if dentro is not None:
            g = g.intersection(dentro)
        g = g.difference(ocupado)
        g = _so_poligonos(g)
        if not g.is_empty:
            out[n] = g
            ocupado = ocupado.union(g)
    return out


def _so_poligonos(g):
    ps = [p for p in getattr(g, "geoms", [g]) if isinstance(p, (shapely.Polygon, shapely.MultiPolygon))]
    ps = [q for p in ps for q in getattr(p, "geoms", [p]) if q.area > 1e-4]
    return shapely.MultiPolygon(ps) if ps else shapely.Polygon()


def preencher(classes, dentro, omissao="terra"):
    """Dá classe aos buracos que ficaram (o alisamento e a área mínima deixam frestas): cada buraco vai para a
    classe com quem partilha mais fronteira. Assim todo o disco tem chão."""
    resto = dentro.difference(shapely.union_all(list(classes.values())))
    out = dict(classes)
    for b in shapely.get_parts(_so_poligonos(resto)):
        anel = b.buffer(0.05)
        melhor, comp = omissao, 0.0
        for n, g in classes.items():
            c = g.intersection(anel).area
            if c > comp:
                melhor, comp = n, c
        out[melhor] = out.get(melhor, shapely.Polygon()).union(b)
    return {n: _so_poligonos(g) for n, g in out.items()}


def raster_para_poligonos(rot, x0, y0, passo, nomes, area_min=2.0, simplificar=0.3, alisar=0.4):
    """Passa um raster de rótulos (linha 0 = sul, como as grelhas do LiDAR) a polígonos por classe.

    `rot[r, c]` é o índice em `nomes` (ou -1); a célula (r, c) cobre x0 + c·passo ± passo/2. Os degraus do raster
    alisam-se com um buffer de ida e volta e uma simplificação de `simplificar` m."""
    from rasterio import features
    from rasterio.transform import Affine
    tr = Affine(passo, 0, x0 - passo / 2, 0, passo, y0 - passo / 2)
    out = {}
    for i, n in enumerate(nomes):
        m = (rot == i).astype(np.uint8)
        if not m.any():
            continue
        ps = [shapely.geometry.shape(g) for g, v in features.shapes(m, mask=m.astype(bool), transform=tr) if v == 1]
        g = shapely.union_all(ps)
        if alisar:
            g = g.buffer(alisar, join_style="mitre").buffer(-2 * alisar, join_style="mitre").buffer(alisar, join_style="mitre")
        g = g.simplify(simplificar)
        g = _so_poligonos(g)
        g = shapely.MultiPolygon([p for p in g.geoms if p.area >= area_min]) if not g.is_empty else g
        if not g.is_empty:
            out[n] = g
    return out


def maioria(rot, n_classes, raio=2, fixos=None):
    """Filtro de maioria (janela (2·raio+1)²): tira os píxeis soltos que davam manchas."""
    k = 2 * raio + 1
    cont = np.stack([ndi.uniform_filter((rot == i).astype(np.float32), k, mode="nearest") for i in range(n_classes)])
    out = cont.argmax(0)
    out[rot < 0] = -1
    if fixos is not None:
        out[fixos] = rot[fixos]
    return out


def gravar(caminho, classes, meta=None):
    dados = {"meta": meta or {}, "classes": {n: shapely.geometry.mapping(shapely.set_precision(g, 0.01))
                                             for n, g in classes.items()}}
    Path(caminho).write_text(json.dumps(dados, ensure_ascii=False))


def ler(caminho):
    d = json.loads(Path(caminho).read_text())
    return {n: shapely.geometry.shape(g) for n, g in d["classes"].items()}


def poligono(pts):
    """Atalho para correções traçadas à mão: lista de (x, y) em metros da cena."""
    return shapely.Polygon(pts)


def faixa(pts, largura, pontas="flat"):
    """Faixa de `largura` m ao longo de uma linha (estradas, caminhos, passeios)."""
    cap = {"flat": "flat", "round": "round", "square": "square"}[pontas]
    return shapely.LineString(pts).buffer(largura / 2, cap_style=cap, join_style="round")


# ---------------------------------------------------------------- proposta automática (chão natural)

NATURAIS = ["asfalto", "calcada", "terra", "relva", "mato", "rocha", "erva", "saibro"]  # (asfalto e calçada só por correção)


def proposta(pasta, verde_min=0.025, claro=(0.58, 0.09), rocha_decl=48.0, saibro=True, maioria_raio=3,
             area_min=8.0, simplificar=0.5, alisar=1.0, casa_folga=4.0, sombra=None, copa="mato"):
    """Proposta automática para o chão natural, a partir de dados/lidar.npz (MDT, MDS e classes a `passo` m, linha
    0 = sul, ±meio m à volta da origem da cena) e de dados/orto.jpg (a mesma área, norte em cima).

    O que não é verde nem copa é erva rasa; verde = relva; copas (classes 3–5 a mais de 1,2 m) = mato; declive acima de
    `rocha_decl` sem copa = rocha; claro e neutro, plano e longe das casas = saibro (pó dos carreiros e terreiros).
    Debaixo dos edifícios fica a classe vizinha. Calçada e asfalto não se adivinham pela cor (sombras e telhados
    enganam): ficam para as correções à mão. casa_folga: metros à volta dos edifícios do LiDAR onde o claro não é
    saibro (telhados deslocados na ortofoto). sombra: luminância abaixo da qual (fora das copas) o píxel é sombra e
    fica com a classe do vizinho iluminado mais próximo. copa: classe do chão debaixo das copas ("mato" no monte; None
    num parque de relva aparada, onde o chão debaixo das árvores é o do vizinho ao sol). Devolve ({classe: polígono}, meio)."""
    from PIL import Image
    pasta = Path(pasta)
    lid = np.load(pasta / "dados" / "lidar.npz")
    passo = float(lid["passo"])
    meio = float(lid["meio"])
    mdt, mds, cls = lid["mdt"] / 100.0, lid["mds"] / 100.0, lid["classe"]
    n = mdt.shape[0]
    im = Image.open(pasta / "dados" / "orto.jpg").convert("RGB")
    o = (np.asarray(im.resize((n, n), Image.BOX), float) / 255.0)[::-1]
    lum = o.mean(2)
    sat = o.max(2) - o.min(2)
    verde = o[..., 1] - (o[..., 0] + o[..., 2]) / 2
    gy, gx = np.gradient(mdt, passo)
    decl = np.degrees(np.arctan(np.hypot(gx, gy)))
    sob_copa = (cls >= 3) & (cls <= 5) & (mds - mdt > 1.2)
    edif = cls == 6
    ix = {k: i for i, k in enumerate(NATURAIS)}
    rot = np.full(lum.shape, ix["erva"])
    rot[verde > verde_min] = ix["relva"]
    if saibro:
        perto_casa = ndi.binary_dilation(edif, iterations=max(1, int(round(casa_folga / passo))))
        rot[(lum > claro[0]) & (sat < claro[1]) & (decl < 30) & ~perto_casa] = ix["saibro"]
    rot[(decl > rocha_decl) & ~sob_copa] = ix["rocha"]
    if sombra is not None:
        esc = (lum < sombra) & ~sob_copa & ~edif
        if esc.any() and not esc.all():
            _, (ri, ci) = ndi.distance_transform_edt(esc | sob_copa, return_indices=True)
            rot = rot[ri, ci]
    if copa:
        rot[sob_copa] = ix[copa]
    elif sob_copa.any() and not sob_copa.all():
        _, (ri, ci) = ndi.distance_transform_edt(sob_copa, return_indices=True)
        rot = rot[ri, ci]
    if edif.any():
        _, (ri, ci) = ndi.distance_transform_edt(edif, return_indices=True)
        rot = rot[ri, ci]
    rot = maioria(rot, len(NATURAIS), raio=maioria_raio)
    rot = maioria(rot, len(NATURAIS), raio=1)
    return raster_para_poligonos(rot, -meio, -meio, passo, NATURAIS, area_min=area_min, simplificar=simplificar,
                                 alisar=alisar), meio


def aplicar(classes, correcoes, meio):
    """Aplica as correções à mão por ordem (cada uma manda sobre o que já lá estava), resolve as sobreposições e tapa
    as frestas, dentro do disco de raio meio − 1. O polígono de uma correção pode ser uma função das classes que já
    lá estão (por exemplo `pela_classe(zona, "erva", "saibro")`: o chão não verde de uma zona passa a calçada)."""
    cl = dict(classes)
    for classe, g, _nota in correcoes:
        if callable(g):
            g = g(cl)
        for k in list(cl):
            cl[k] = cl[k].difference(g)
        cl[classe] = cl.get(classe, shapely.Polygon()).union(g)
    d = disco(meio - 1)
    return preencher(resolver(cl, dentro=d), d)


def pela_cor(pasta, zona, lum_min=0.6, sat_max=0.16, verde_max=0.03, fechar=1, area_min=3.0, alisar=0.5):
    """Para as correções: os píxeis da ortofoto (dados/orto.jpg, à resolução do LiDAR) dentro da `zona` que são claros,
    pouco saturados e sem verde, fora das copas. Serve para os carreiros estreitos (2–3 m) que o filtro de maioria da
    proposta apaga: a zona diz onde há carreiro (um corredor à volta do traçado do OSM) e a cor diz a forma exata."""
    from PIL import Image
    from rasterio import features
    from rasterio.transform import Affine
    pasta = Path(pasta)
    lid = np.load(pasta / "dados" / "lidar.npz")
    passo, meio = float(lid["passo"]), float(lid["meio"])
    n = lid["mdt"].shape[0]
    o = np.asarray(Image.open(pasta / "dados" / "orto.jpg").convert("RGB").resize((n, n), Image.BOX), float)[::-1] / 255
    copa = (lid["classe"] >= 3) & (lid["classe"] <= 5) & (lid["mds"] - lid["mdt"] > 120)
    xs = -meio + passo * np.arange(n)  # centros das células, como em raster_para_poligonos
    X, Y = np.meshgrid(xs, xs)
    m = (o.mean(2) > lum_min) & (o.max(2) - o.min(2) < sat_max) & (o[..., 1] - (o[..., 0] + o[..., 2]) / 2 < verde_max)
    m &= ~copa & shapely.contains_xy(zona, X, Y)
    if fechar:
        m = ndi.binary_opening(ndi.binary_closing(m, iterations=fechar), iterations=1)
    tr = Affine(passo, 0, -meio - passo / 2, 0, passo, -meio - passo / 2)
    g = [shapely.geometry.shape(gj) for gj, v in features.shapes(m.astype(np.uint8), mask=m, transform=tr) if v == 1]
    g = shapely.union_all(g).buffer(alisar).buffer(-alisar).simplify(0.25) if g else shapely.Polygon()
    return shapely.union_all([p for p in shapely.get_parts(g) if p.area >= area_min]).intersection(zona)


def pela_classe(zona, *nomes):
    """Para as correções: a parte da `zona` que a proposta (ou as correções anteriores) deu como `nomes`. Serve para
    trocar o chão não verde de uma zona (erva, saibro) por calçada ou asfalto, com a fronteira da ortofoto."""
    def f(cl):
        return zona.intersection(shapely.union_all([cl[n] for n in nomes if n in cl]))
    return f


# ---------------------------------------------------------------- malhas

def _bilinear(xs, Z, x, y):
    """Interpolação bilinear na grelha regular Z (Z[r, c] em x = xs[c], y = xs[r]). É linear ao longo das arestas
    das células, por isso os vértices que caem numa aresta ficam na mesma reta que os cantos: não há fendas
    entre uma célula inteira e a vizinha cortada por uma fronteira."""
    p = xs[1] - xs[0]
    c = np.clip((x - xs[0]) / p, 0, len(xs) - 1.000001)
    r = np.clip((y - xs[0]) / p, 0, len(xs) - 1.000001)
    c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
    fc, fr = c - c0, r - r0
    return (Z[r0, c0] * (1 - fc) * (1 - fr) + Z[r0, c0 + 1] * fc * (1 - fr)
            + Z[r0 + 1, c0] * (1 - fc) * fr + Z[r0 + 1, c0 + 1] * fc * fr)


def disco(raio, segmentos=360):
    a = np.linspace(0, 2 * np.pi, segmentos, endpoint=False)
    return shapely.Polygon(np.stack([np.cos(a) * raio, np.sin(a) * raio], 1))


def malhas(classes, xs, Z, raio=None, recorte=None, segmentos=360):
    """Uma malha (V, F) por classe, sobre a grelha Z (passo = xs[1] − xs[0]).

    Células inteiramente dentro de uma classe ficam como dois triângulos; as que uma fronteira corta são
    recortadas e trianguladas com Delaunay restrito, por isso a fronteira fica exata. `raio` recorta a um disco
    (base redonda) e `recorte` é um polígono a tirar (por exemplo o chão debaixo de um volume fechado)."""
    p = float(xs[1] - xs[0])
    lim = disco(raio, segmentos) if raio else None
    if lim is not None or recorte is not None:
        cl = {}
        for n, g in classes.items():
            if lim is not None:
                g = g.intersection(lim)
            if recorte is not None:
                g = g.difference(recorte)
            g = _so_poligonos(g)
            if not g.is_empty:
                cl[n] = g
        classes = cl
    # caixas das células
    n = len(xs) - 1
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")  # I: linha (y), J: coluna (x)
    I, J = I.ravel(), J.ravel()
    caixas = shapely.box(xs[J], xs[I], xs[J] + p, xs[I] + p)
    arvore = shapely.STRtree(caixas)
    out = {}
    for nome, g in classes.items():
        idx = arvore.query(g, predicate="intersects")
        if len(idx) == 0:
            continue
        cheio = shapely.contains_properly(g, caixas[idx]) | shapely.covers(g, caixas[idx])
        tris = []
        # células inteiras: dois triângulos cada
        ci = idx[cheio]
        if len(ci):
            x0, y0 = xs[J[ci]], xs[I[ci]]
            a = np.stack([x0, y0], 1)
            b = np.stack([x0 + p, y0], 1)
            c = np.stack([x0 + p, y0 + p], 1)
            d = np.stack([x0, y0 + p], 1)
            tris.append(np.stack([a, b, c], 1))
            tris.append(np.stack([a, c, d], 1))
        # células cortadas
        cc = idx[~cheio]
        if len(cc):
            pecas = shapely.intersection(caixas[cc], g)
            pecas = shapely.get_parts(pecas)
            pecas = pecas[shapely.get_type_id(pecas) == 3]
            pecas = pecas[shapely.area(pecas) > 1e-5]
            if len(pecas):
                t = shapely.get_parts(shapely.constrained_delaunay_triangles(pecas))
                t = t[shapely.area(t) > 1e-7]
                co = shapely.get_coordinates(shapely.get_exterior_ring(t)).reshape(-1, 4, 2)[:, :3]
                tris.append(co)
        if not tris:
            continue
        T = np.concatenate(tris)
        # sentido anti-horário visto de cima
        cr = (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 1, 1] - T[:, 0, 1]) * (T[:, 2, 0] - T[:, 0, 0])
        T[cr < 0] = T[cr < 0][:, ::-1]
        pts = T.reshape(-1, 2)
        chave = np.round(pts * 1000).astype(np.int64)
        uniq, inv = np.unique(chave, axis=0, return_inverse=True)
        V2 = uniq / 1000.0
        z = _bilinear(np.asarray(xs, float), Z, V2[:, 0], V2[:, 1])
        V = np.column_stack([V2, z])
        F = inv.reshape(-1, 3)
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        out[nome] = (V, F)
    return out


def aplanar(xs, Z, geoms, sigma=1.5, margem=1.0):
    """Alisa o terreno debaixo do chão pavimentado (ruas, calçadas, lajes): o MDT de 0,5 m tem o ruído das pedras,
    carros e muretes, e uma calçada não ondula. Dentro de `geoms` o Z passa a ser o Z filtrado (gaussiano de
    `sigma` m), com transição de `margem` m para fora. Devolve a grelha nova."""
    p = float(xs[1] - xs[0])
    g = shapely.union_all([x for x in geoms if x is not None and not x.is_empty])
    if g.is_empty:
        return Z
    X, Y = np.meshgrid(xs, xs)
    d = shapely.distance(g, shapely.points(X.ravel(), Y.ravel())).reshape(X.shape)
    peso = np.clip(1 - d / max(margem, 1e-6), 0, 1)
    Zs = ndi.gaussian_filter(Z, sigma / p, mode="nearest")
    return Z * (1 - peso) + Zs * peso


def borda_disco(xs, Z, raio, segmentos=360):
    """Anel (x, y, z) da borda redonda, para a saia da base."""
    a = np.linspace(0, 2 * np.pi, segmentos, endpoint=False)
    x, y = np.cos(a) * raio, np.sin(a) * raio
    return np.stack([x, y, _bilinear(np.asarray(xs, float), Z, x, y)], 1)


def lancis(classes, a, b, largura=0.15):
    """Linhas de fronteira entre as classes a e b (por exemplo asfalto | calçada), para construir lancis."""
    if a not in classes or b not in classes:
        return []
    li = classes[a].boundary.intersection(classes[b].buffer(0.05))
    li = shapely.line_merge(shapely.union_all([g for g in shapely.get_parts(li) if g.length > 0.3]))
    return [g for g in shapely.get_parts(li) if g.length > 1.0]
