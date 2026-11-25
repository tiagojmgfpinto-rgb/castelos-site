"""Planta ortogonal de uma maqueta (vista de cima, câmara ORTHO), para comparar com a ortofoto quadrícula a quadrícula.

Abre o .blend que o construir.py grava com --blend=FICHEIRO e renderiza com o Cycles (o Workbench e o EEVEE precisam de
GPU). A imagem fica com o norte para cima e `px` píxeis por metro, centrada em (cx, cy) da cena.

    python pipeline/maquetas/penela/construir.py --blend=/tmp/penela.blend
    python pipeline/kit/planta.py /tmp/penela.blend 0 0 300 4 /tmp/planta.jpg
    python pipeline/kit/quadricula.py penela --classes --planta /tmp/planta.jpg:150
"""
import sys

import bpy


def main():
    blend, cx, cy, lado, px, saida = sys.argv[1], *map(float, sys.argv[2:6]), sys.argv[6]
    bpy.ops.wm.open_mainfile(filepath=blend)
    sc = bpy.context.scene
    cam = bpy.data.cameras.new("planta")
    cam.type = "ORTHO"
    cam.ortho_scale = lado
    cam.clip_end = 5000
    co = bpy.data.objects.new("planta", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    co.location = (cx, cy, 1000)
    co.rotation_euler = (0, 0, 0)
    # luz própria (o .blend pode ter sido gravado antes das vistas): céu claro e sol a noroeste, 50° de altura
    import math
    if sc.world is None or not any(o.type == "LIGHT" for o in sc.objects):
        w = bpy.data.worlds.new("mundo_planta")
        sc.world = w
        w.use_nodes = True
        w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.80, 0.76, 0.68, 1)
        w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.65
        sol = bpy.data.lights.new("sol_planta", "SUN")
        sol.energy = 3.6
        so = bpy.data.objects.new("sol_planta", sol)
        sc.collection.objects.link(so)
        so.rotation_euler = (math.radians(40), 0, math.radians(135))
        sc.view_settings.view_transform = "AgX"
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 8
    sc.cycles.use_denoising = True
    sc.render.resolution_x = sc.render.resolution_y = int(round(lado * px))
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 90
    sc.render.filepath = saida
    bpy.ops.render.render(write_still=True)
    print("planta", saida)


if __name__ == "__main__":
    main()
