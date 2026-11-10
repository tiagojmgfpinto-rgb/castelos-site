"""Ajudas de cena do Blender: UV em coordenadas do mundo, cena vazia e modelos de árvore.

Funções comuns às maquetas, tiradas sem alterações dos construir.py (ver pipeline/kit/LEIAME.md)."""
import bpy  # noqa: I001 (o bpy tem de vir antes do bmesh)
import bmesh
import numpy as np

__all__ = ["uv_mundo", "limpar", "arvore_modelo"]


def uv_mundo(ob, escala):
    """UV em metros: faces horizontais em planta, verticais ao longo da face (as fiadas seguem a cota)."""
    me = ob.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    n_l = len(me.loops)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    vi = np.empty(n_l, int)
    me.loops.foreach_get("vertex_index", vi)
    nrm = np.empty(len(me.polygons) * 3)
    me.polygons.foreach_get("normal", nrm)
    nrm = nrm.reshape(-1, 3)
    lt = np.empty(len(me.polygons), int)
    me.polygons.foreach_get("loop_total", lt)
    pol = np.repeat(np.arange(len(me.polygons)), lt)
    n = nrm[pol]
    p = co[vi]
    hz = np.abs(n[:, 2]) > 0.7
    h = np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-6)
    u = np.where(hz, p[:, 0], (p[:, 0] * -n[:, 1] + p[:, 1] * n[:, 0]) / h)
    v = np.where(hz, p[:, 1], p[:, 2])
    me.uv_layers.active.data.foreach_set("uv", (np.stack([u, v], 1) / escala).ravel().astype(np.float32))


def limpar():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def arvore_modelo(subdiv):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    v = np.array([x.co[:] for x in bm.verts])
    f = [[x.index for x in fa.verts] for fa in bm.faces]
    bm.free()
    return v, f


class Obra:
    """Folga das copas às construções já feitas (auditoria de novembro de 2026: copas a entrar em muros e
    telhados). Junta numa BVH todas as malhas da cena menos as que se excluem (terreno, base, vegetação)."""

    def __init__(self, excluir=("Terreno", "Base", "Vegetacao")):
        from mathutils.bvhtree import BVHTree
        v, f = [], []
        for ob in bpy.data.objects:
            if ob.type != "MESH" or any(ob.name.startswith(e) for e in excluir):
                continue
            mw = ob.matrix_world
            b = len(v)
            v.extend([tuple(mw @ p.co) for p in ob.data.vertices])
            f.extend([[b + i for i in p.vertices] for p in ob.data.polygons])
        self.bvh = BVHTree.FromPolygons(v, f) if f else None
        self.encolhidas = self.retiradas = 0

    def folga(self, pts):
        from mathutils import Vector
        if self.bvh is None:
            return 99.0
        d = [self.bvh.find_nearest(Vector(p))[3] for p in pts]
        return min([x for x in d if x is not None] or [99.0])

    def ajustar(self, x, y, z, h, raio, alt, alt_max_raio, alcance=1.1, minimo=1.2, tronco_min=0.6):
        """Copa de raio `raio` e altura `alt` com o topo medido a z + h. Devolve (raio, alt, tronco) com a copa
        encolhida até à construção mais próxima, ou None se o tronco bate na obra ou não cabe uma copa
        razoável (a árvore não se planta)."""
        tronco = max(h - alt, tronco_min)
        if self.folga([(x, y, z + k) for k in (0.5, tronco * 0.5, tronco)]) < 0.35:
            self.retiradas += 1
            return None
        centros = lambda t, a: [(x, y, z + t + a * f) for f in (0.15, 0.5, 0.85)]
        fo = self.folga(centros(tronco, alt))
        if fo >= raio * alcance + 0.15:
            return raio, alt, tronco
        if fo < minimo:
            self.retiradas += 1
            return None
        raio = (fo - 0.15) / alcance
        alt = min(alt, alt_max_raio * raio)
        tronco = max(h - alt, tronco_min)
        if self.folga(centros(tronco, alt)) < raio * alcance + 0.05:
            self.retiradas += 1
            return None
        self.encolhidas += 1
        return raio, alt, tronco
