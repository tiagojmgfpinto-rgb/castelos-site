"""Materiais do Blender (Principled BSDF com texturas, normais e cor por vértice) e o registo MATS onde ficam.

Funções comuns às maquetas, tiradas sem alterações dos construir.py (ver pipeline/kit/LEIAME.md)."""
import bpy
import numpy as np

# Registo dos materiais da maqueta (nome -> bpy.types.Material), partilhado com o construir.py que o importa.
MATS = {}

__all__ = ["MATS", "malha_rapida", "material"]


def malha_rapida(nome, V, Fq, mat, pai, elemento=None):
    me = bpy.data.meshes.new(nome)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(Fq.size)
    me.loops.foreach_set("vertex_index", Fq.astype(np.int32).ravel())
    me.polygons.add(len(Fq))
    me.polygons.foreach_set("loop_start", (np.arange(len(Fq)) * 4).astype(np.int32))
    me.update(calc_edges=True)
    me.materials.append(MATS[mat])
    ob = bpy.data.objects.new(nome, me)
    bpy.context.scene.collection.objects.link(ob)
    if elemento:
        ob["elemento"] = elemento
    ob.parent = pai
    return ob


def material(nome, cor, rug=0.9, tex=None, vcor=False, normal=None, forca=1.0, escala=None):
    mt = bpy.data.materials.new(nome)
    mt.use_nodes = True
    nt = mt.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    mapa = None
    if escala is not None:
        uvn = nt.nodes.new("ShaderNodeUVMap")
        mapa = nt.nodes.new("ShaderNodeMapping")
        mapa.inputs["Scale"].default_value = (escala, escala, 1.0)
        nt.links.new(uvn.outputs["UV"], mapa.inputs["Vector"])
    if normal is not None:
        normal.colorspace_settings.name = "Non-Color"
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = normal
        if mapa is not None:
            nt.links.new(mapa.outputs["Vector"], tn.inputs["Vector"])
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = forca
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Base Color"].default_value = (*[c ** 2.2 for c in cor], 1)
    bsdf.inputs["Roughness"].default_value = rug
    if tex is not None:
        ti = nt.nodes.new("ShaderNodeTexImage")
        ti.image = tex
        if mapa is not None:
            nt.links.new(mapa.outputs["Vector"], ti.inputs["Vector"])
        if vcor:
            va = nt.nodes.new("ShaderNodeVertexColor")
            va.layer_name = "Col"
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            nt.links.new(ti.outputs["Color"], mix.inputs[6])
            nt.links.new(va.outputs["Color"], mix.inputs[7])
            nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        else:
            nt.links.new(ti.outputs["Color"], bsdf.inputs["Base Color"])
    elif vcor:
        va = nt.nodes.new("ShaderNodeVertexColor")
        va.layer_name = "Col"
        nt.links.new(va.outputs["Color"], bsdf.inputs["Base Color"])
    MATS[nome] = mt
    return mt
