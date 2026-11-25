"""Parte Blender do chão por classes (kit/chao.py): materiais nítidos por classe e um objeto por classe.

    from kit import chao, chao_blender
    pecas = chao.malhas(classes, xs, Z, raio=RAIO)
    chao_blender.materiais(PALETA)                         # cria os materiais "chao_<classe>" em kit.materiais.MATS
    obs = chao_blender.objetos(pecas, pai=GRUPO_TERRENO)   # objetos "Chao_<classe>", com UV do mundo

Cada material é uma textura procedimental nítida (cubos, lajes, grão de asfalto, saibro, relva...) multiplicada por
uma cor por vértice com variação larga e suave (±5 %, ruído), para não ficar chapado. Nada vem da ortofoto.
A PALETA de cada maqueta dá a cor média medida nas fotografias do sítio, por classe.
"""
import bpy
import numpy as np

from .materiais import MATS, material
from scipy import ndimage as ndi

from .texturas import fbm, imagem
from .geometria import normais

# cor média (sRGB 0-1) e metros que a textura cobre; as maquetas mudam as cores pelas fotografias do sítio
PALETA_BASE = {
    "asfalto": {"cor": (0.33, 0.33, 0.34), "metros": 4.0},
    "calcada": {"cor": (0.70, 0.68, 0.64), "metros": 4.0},
    "lajeado": {"cor": (0.76, 0.73, 0.67), "metros": 6.0},
    "escadas": {"cor": (0.72, 0.70, 0.66), "metros": 4.0},
    "saibro": {"cor": (0.80, 0.75, 0.64), "metros": 6.0},
    "terra": {"cor": (0.62, 0.54, 0.42), "metros": 6.0},
    "relva": {"cor": (0.42, 0.50, 0.27), "metros": 8.0},
    "erva": {"cor": (0.58, 0.56, 0.38), "metros": 8.0},
    "mato": {"cor": (0.30, 0.36, 0.20), "metros": 8.0},
    "rocha": {"cor": (0.66, 0.64, 0.58), "metros": 6.0},
    "agua": {"cor": (0.22, 0.32, 0.36), "metros": 8.0},
}


def _neutra(tex):
    """Normaliza uma textura para média 1 (a cor vem da paleta)."""
    return np.clip(tex / tex.reshape(-1, 3).mean(0), 0, 2.5)


def pfbm(n, celulas, semente, oit=4):
    """Ruído fractal periódico (n × n píxeis, `celulas` na oitava mais larga): a textura repete sem costuras."""
    r = np.random.default_rng(semente)
    s, tot, a = np.zeros((n, n)), 0.0, 1.0
    for o in range(oit):
        c = int(celulas * 2 ** o)
        if c > n // 2:
            break
        g = r.random((c, c))
        s += a * ndi.zoom(g, n / c, order=3, mode="grid-wrap", grid_mode=True)[:n, :n]
        tot += a
        a *= 0.5
    return np.clip(s / tot, 0, 1)


def _juntas(n, linhas, semente, desfasar=True):
    """Aparelho em fiadas periódicas: devolve (id do bloco, junta 0-1) para lajes ou cubos."""
    r = np.random.default_rng(semente)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    alt = r.uniform(0.75, 1.25, linhas)
    yb = np.concatenate([[0], np.cumsum(alt / alt.sum() * n)])
    fila = np.clip(np.searchsorted(yb, yy, side="right") - 1, 0, linhas - 1)
    ident = np.zeros((n, n), int)
    dist = np.full((n, n), 1e9)
    for i in range(linhas):
        m = fila == i
        dy = np.minimum(yy[m] - yb[i], yb[i + 1] - yy[m])
        k = max(2, int(round(n / (yb[i + 1] - yb[i]) * r.uniform(0.6, 1.3)))) if desfasar else linhas
        cs = r.uniform(0.6, 1.4, k)
        xb = np.concatenate([[0], np.cumsum(cs / cs.sum() * n)]) + (r.uniform(0, n) if desfasar else 0)
        x = (xx[m] - xb[0]) % n
        xb = xb - xb[0]
        col = np.clip(np.searchsorted(xb, x, side="right") - 1, 0, k - 1)
        dx = np.minimum(x - xb[col], xb[col + 1] - x)
        ident[m] = i * 1000 + col
        dist[m] = np.minimum(dx, dy)
    return ident, dist


def _tex(nome, metros, n=1024):
    m_px = n / metros
    r = np.random.default_rng(len(nome) * 7 + 1)
    if nome in ("calcada", "escadas", "lajeado"):
        lado = 0.10 if nome != "lajeado" else 0.55
        linhas = max(2, int(round(metros / lado)))
        ident, dist = _juntas(n, linhas, 31 if nome != "lajeado" else 37, desfasar=True)
        tom = r.uniform(0.86, 1.12, ident.max() + 1)[ident]
        junta = dist < (0.012 if nome != "lajeado" else 0.008) * m_px
        g = pfbm(n, 8, 51)
        t = (tom * (0.92 + 0.16 * g))[..., None] * np.ones(3)
        t = np.where(junta[..., None], t * 0.62, t)
        a = np.clip(dist / (0.03 * m_px), 0, 1)
    elif nome == "asfalto":
        g = pfbm(n, int(metros * 2), 41, oit=6)
        brita = r.random((n, n)) > 0.94
        t = (0.88 + 0.22 * g)[..., None] * np.ones(3)
        t = np.where(brita[..., None], t * 1.2, t)
        a = g
    elif nome in ("saibro", "terra"):
        g = pfbm(n, int(metros * 1.5), 23 if nome == "terra" else 29, oit=6)
        pedra = ndi.binary_dilation(r.random((n, n)) > (0.992 if nome == "terra" else 0.985), iterations=1)
        t = (0.84 + 0.32 * g)[..., None] * np.ones(3)
        t = np.where(pedra[..., None], t * 1.18, t)
        a = g
    elif nome in ("relva", "mato", "erva"):
        g = pfbm(n, int(metros), {"relva": 7, "mato": 11, "erva": 19}[nome], oit=6)
        g2 = pfbm(n, int(metros * 8), 9, oit=3)
        amarelo = {"relva": np.array([1.30, 1.18, 0.85]), "mato": np.array([1.10, 0.85, 0.70]), "erva": np.array([1.12, 1.0, 0.82])}[nome]
        tq = np.clip((g - 0.5) * 3, 0, 1)[..., None]
        t = np.ones((n, n, 3)) * (1 - tq) + amarelo * tq
        t = t * (0.82 + 0.36 * g2)[..., None]
        a = g2
    else:
        g = pfbm(n, int(metros), 17, oit=6)
        t = (0.8 + 0.4 * g)[..., None] * np.ones(3)
        a = g
    return _neutra(np.asarray(t, float)), np.asarray(a, float)


def materiais(paleta=None, n=1024):
    """Cria (ou devolve) os materiais "chao_<classe>" em MATS. paleta: {classe: {"cor": (r,g,b), "metros": m}}."""
    pal = {k: dict(v) for k, v in PALETA_BASE.items()}
    for k, v in (paleta or {}).items():
        pal.setdefault(k, {}).update(v)
    for nome, p in pal.items():
        chave = f"chao_{nome}"
        if chave in MATS:
            continue
        # 1024 px só onde se veem juntas (cubos, lajes); o resto chega com 512 px (o GLB não incha)
        t, a = _tex(nome, p["metros"], n if nome in ("calcada", "lajeado", "escadas") else n // 2)
        t = np.clip(t * np.array(p["cor"]), 0, 1)
        nm = imagem(f"{chave}_n", normais(a, p["metros"])) if nome in ("calcada", "lajeado", "escadas") else None
        material(chave, (1, 1, 1), 0.95 if nome != "agua" else 0.2, imagem(chave, t), vcor=True,
                 normal=nm, forca=0.6)
        MATS[chave]["metros"] = float(p["metros"])
    return pal


def objetos(pecas, pai=None, elemento="terreno", semente=3, paleta=None, prefixo="Chao"):
    """Um objeto por classe ("<prefixo>_<classe>"), com UV do mundo (a textura cobre `metros` m) e cor por vértice
    com variação larga e suave. Devolve {classe: objeto}."""
    pal = materiais(paleta)
    out = {}
    for nome, (V, F) in pecas.items():
        me = bpy.data.meshes.new(f"chao_{nome}")
        me.vertices.add(len(V))
        me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        me.loops.add(F.size)
        me.loops.foreach_set("vertex_index", F.astype(np.int32).ravel())
        me.polygons.add(len(F))
        me.polygons.foreach_set("loop_start", (np.arange(len(F)) * 3).astype(np.int32))
        me.update(calc_edges=True)
        me.materials.append(MATS[f"chao_{nome}"])
        # UV do mundo (vista de cima): a textura cobre `metros` m
        m = pal.get(nome, {"metros": 6.0})["metros"]
        uv = me.uv_layers.new(name="UVMap")
        lv = np.empty(F.size, np.int32)
        me.loops.foreach_get("vertex_index", lv)
        uv.data.foreach_set("uv", (V[lv, :2] / m).astype(np.float32).ravel())
        # variação larga (8–20 m), ±5 %
        v = 0.95 + 0.10 * fbm(V[:, 0], V[:, 1], 14.0, semente + len(nome))
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        ca.data.foreach_set("color", np.column_stack([np.repeat(v[:, None] ** 2.2, 3, 1), np.ones(len(V))])
                            .astype(np.float32).ravel())
        me.color_attributes.active_color = ca
        me.shade_smooth()
        ob = bpy.data.objects.new(f"{prefixo}_{nome}", me)
        bpy.context.scene.collection.objects.link(ob)
        if elemento:
            ob["elemento"] = elemento
        ob.parent = pai
        out[nome] = ob
    return out


def separar_faces(ob, indices, classes, paleta=None, pai=None, elemento=None, prefixo="Chao_topo", omissao="calcada"):
    """Tira de `ob` as faces com material_index em `indices` (o chão desenhado em cima de outras peças: o topo das
    plataformas e dos terraplenos, os pátios) e fá-las objetos do chão por classes, com a classe de dados/chao.json
    no centro de cada face (sem classe: a mais próxima, ou `omissao`). Devolve {classe: objeto}."""
    import bmesh
    import shapely
    me = ob.data
    nv = len(me.vertices)
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    npol = len(me.polygons)
    mi = np.empty(npol, int)
    me.polygons.foreach_get("material_index", mi)
    ls = np.empty(npol, int)
    me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(npol, int)
    me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), int)
    me.loops.foreach_get("vertex_index", lv)
    sel = np.nonzero(np.isin(mi, list(indices)))[0]
    if not len(sel):
        return {}
    tris = []
    for i in sel:
        v = lv[ls[i]:ls[i] + lt[i]]
        tris += [(v[0], v[k], v[k + 1]) for k in range(1, len(v) - 1)]
    tris = np.array(tris)
    cen = co[tris].mean(1)
    nomes = [n for n in classes if not classes[n].is_empty]
    lab = np.full(len(tris), -1)
    for k, n in enumerate(nomes):
        lab[(lab < 0) & shapely.contains_xy(classes[n], cen[:, 0], cen[:, 1])] = k
    falta = lab < 0
    if falta.any() and (~falta).any():
        from scipy.spatial import cKDTree
        _, j = cKDTree(cen[~falta, :2]).query(cen[falta, :2])
        lab[falta] = lab[~falta][j]
    elif falta.any():
        nomes.append(omissao)
        lab[:] = len(nomes) - 1
    pecas = {}
    for k, n in enumerate(nomes):
        t = tris[lab == k]
        if not len(t):
            continue
        u, inv = np.unique(t.ravel(), return_inverse=True)
        pecas[n] = (co[u], inv.reshape(-1, 3))
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    alvo = set(indices)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index in alvo], context="FACES")
    bm.to_mesh(me)
    bm.free()
    me.update()
    return objetos(pecas, pai=pai if pai is not None else ob.parent, elemento=elemento or ob.get("elemento"),
                   paleta=paleta, prefixo=prefixo)
