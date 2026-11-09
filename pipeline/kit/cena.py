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
