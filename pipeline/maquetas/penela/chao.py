"""Classes do chão da maqueta de Penela -> dados/chao.json (método "metro a metro", novembro de 2026).

1. Proposta automática a 0,5 m: classes do LiDAR 2024 (copas, edifícios), declive do MDT e cor da ortofoto de 2025
   (verde = relva, claro e neutro = calçada, escuro e neutro junto a uma rua do OSM = asfalto, o resto terra),
   com filtro de maioria e passada a polígonos alisados.
2. Correções traçadas à mão nas quadrículas de 40 m (kit/quadricula.py), que mandam sempre: lista CORRECOES abaixo,
   por ordem; cada uma é (classe, polígono em metros da cena, nota do que se viu).

A ortofoto só decide as fronteiras: o chão da maqueta leva materiais nítidos por classe, nunca a fotografia.

    python pipeline/maquetas/penela/chao.py
    python pipeline/kit/quadricula.py penela --classes      # ver o resultado quadrícula a quadrícula
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import shapely
from PIL import Image
from scipy import ndimage as ndi

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1]))
from kit import chao  # noqa: E402

CFG = json.loads((AQUI.parents[2] / "src" / "data" / "maquetas" / "penela.json").read_text())
F = json.loads((AQUI / "dados" / "fontes.json").read_text())
LAT0, LON0 = CFG["origem"]["lat"], CFG["origem"]["lon"]
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110574
CENTRO = np.array([-40.0, 0.0])

LID = np.load(AQUI / "dados" / "lidar.npz")
PASSO, MEIO = float(LID["passo"]), float(LID["meio"])
MDT = LID["mdt"] / 100.0
MDS = LID["mds"] / 100.0
CLS = LID["classe"]
N = MDT.shape[0]
XS = np.linspace(-MEIO, MEIO, N)
X, Y = np.meshgrid(XS, XS)  # linha 0 = sul

NOMES = ["asfalto", "calcada", "terra", "relva", "mato", "rocha", "erva", "saibro"]  # (asfalto e calçada só por correção)
IX = {n: i for i, n in enumerate(NOMES)}


def ll(c):
    return np.array([(c[0] - LON0) * KX, (c[1] - LAT0) * KY]) - CENTRO


def orto_05():
    """Ortofoto reamostrada para a grelha de 0,5 m do LiDAR (linha 0 = sul), em 0-1."""
    im = Image.open(AQUI / "dados" / "orto.jpg").convert("RGB")
    a = np.asarray(im.resize((N, N), Image.BOX), float) / 255.0
    return a[::-1]


def proposta():
    o = orto_05()
    lum = o.mean(2)
    sat = o.max(2) - o.min(2)
    verde = o[..., 1] - (o[..., 0] + o[..., 2]) / 2
    gy, gx = np.gradient(MDT, PASSO)
    decl = np.degrees(np.arctan(np.hypot(gx, gy)))
    copa = (CLS >= 3) & (CLS <= 5) & (MDS - MDT > 1.2)
    edif = CLS == 6

    # o chão que não é verde nem copa é erva rasa (seca no verão da ortofoto de 2025; verde nas fotografias de
    # primavera); terra nua só por correção
    rot = np.full(lum.shape, IX["erva"])
    rot[verde > 0.025] = IX["relva"]
    # calçada e asfalto não se adivinham pela cor (as sombras e os telhados enganam): só pelas correções
    perto_casa = ndi.binary_dilation(edif, iterations=8)  # telhados e pátios claros enganam: 4 m à volta das casas
    rot[(lum > 0.58) & (sat < 0.09) & (decl < 30) & ~perto_casa] = IX["saibro"]  # pó de calcário dos carreiros e terreiros
    rot[(decl > 48) & ~copa] = IX["rocha"]
    rot[copa] = IX["mato"]
    # debaixo dos edifícios não se vê o chão: fica a classe vizinha mais próxima (os volumes tapam-no)
    if edif.any():
        _, (ri, ci) = ndi.distance_transform_edt(edif, return_indices=True)
        rot = rot[ri, ci]
    rot = chao.maioria(rot, len(NOMES), raio=3)
    rot = chao.maioria(rot, len(NOMES), raio=1)
    return chao.raster_para_poligonos(rot, -MEIO, -MEIO, PASSO, NOMES, area_min=8.0, simplificar=0.5, alisar=1.0)


# ---------------------------------------------------------------- correções à mão (quadrícula a quadrícula)
# (classe, polígono ou faixa, o que se viu na ortofoto / LiDAR / Street View)
def osm_linha(i):
    return [tuple(ll(c)) for c in F["caminhos"][i]["geometria"]["coordinates"]]


CORRECOES = [
    # ruas e caminhos do OSM, com a largura medida na ortofoto (quadrículas de 40 m)
    *[("asfalto", chao.faixa(osm_linha(i), 5.0, "round"), f"rua do OSM n.º {i} (residential), 5 m")
      for i, f in enumerate(F["caminhos"]) if f["classe"] in ("residential", "unknown") and i not in (60, 62, 65)],
    ("asfalto", chao.faixa(osm_linha(60), 3.6, "round"),
     "Rua D. Afonso Henriques: viela entre as duas filas de casas (x ≈ −21 a −18 na ortofoto), c. 3,5 m"),
    *[("asfalto", chao.faixa(osm_linha(i), 4.0, "round"), f"serventia do OSM n.º {i}, 4 m") for i in (40,)],
    ("asfalto", chao.faixa(osm_linha(65), 5.5, "round"),
     "Rua do Sol (estrada da encosta nascente e norte): asfalto com linhas brancas nas bermas (Google Maps), c. 5,5 m"),
    *[("calcada", chao.faixa(osm_linha(i), 2.5, "round"), f"escadas do OSM n.º {i}") for i, f in enumerate(F["caminhos"])
      if f["classe"] == "steps"],
    *[("terra", chao.faixa(osm_linha(i), 1.6, "round"), f"carreiro do OSM n.º {i} (encosta, no mato)") for i in (52, 53, 54, 68)],
    # dentro da cerca
    ("saibro", chao.faixa(osm_linha(56)[:3], 7.5, "round"),
     "caminho poente dentro da cerca (OSM 56): faixa clara e lisa de c. 7–8 m na ortofoto, x −28 a −18"),
    ("calcada", chao.poligono([(-12, -55), (-12, -70), (-11, -100), (-9, -130), (25, -130), (25, -100), (4, -98),
                               (1, -88), (-1, -70), (-3, -56)]),
     "rua larga e largo ao pé do castelo (OSM 62 e 64): pavimento claro na ortofoto entre a fachada das casas (x ≈ −12) "
     "e o penedo, alargando no largo com árvores a sul (y < −100)"),
    ("rocha", chao.poligono([(-3, -56), (-1, -70), (1, -88), (4, -98), (22, -98), (24, -80), (22, -62), (18, -55), (8, -50)]),
     "penedo de calcário à vista por baixo da torre quinária e do pano sul: branco com fendas na ortofoto, rocha na "
     "fotografia da rampa"),
    ("calcada", chao.faixa([(14, -104), (11.5, -90), (10.5, -75), (10, -62), (11.5, -57)], 4.0, "round"),
     "subida calcetada por fora, sobre o penedo, até à Brecha (OSM 64 está c. 3 m a nascente da faixa clara)"),
    ("calcada", chao.poligono([(7.6, -57.6), (11.7, -61.2), (18.3, -53.7), (14.2, -50.1)]),
     "chão da Brecha das Desaparecidas (10 m de passagem, a laje que lá estava)"),
    ("calcada", chao.faixa([(14.5, -52.5), (12.8, -45.0), (11.7, -33.2), (6.8, -9.7), (1.6, 5.2)], 6.0, "round"),
     "rampa calcetada da Brecha para a igreja (OSM 64; fotografia do Google Maps: calçada de cubos, torre à direita)"),
    ("lajeado", chao.poligono([(-10, 36), (-2, 38), (1, 31), (3, 14), (6, 8), (6.5, 1), (1, 0), (-4, 9), (-8, 21)]),
     "adro a poente da igreja (OSM 66): lajes claras na fotografia do castelejo para noroeste; faixa clara na ortofoto"),
    ("relva", chao.poligono([(15.5, 2), (16, -48), (25, -52), (30, -40), (33, -25), (36, -10), (35, 2), (28, 9)]),
     "pátio nascente entre o muro ameado (x ≈ 15) e a cerca: relva (fotografias de primavera; seca no verão de 2025)"),
    ("saibro", chao.faixa([(16.5, -3), (25, -7), (34, -12)], 1.3, "round"),
     "carreiro claro no pátio nascente (ortofoto e fotografia do castelejo)"),
    ("relva", chao.poligono([(-9, -45), (-15.5, -5), (-11, 4), (-5, 8), (0.5, 0), (4, -8), (8, -20), (9, -45)]),
     "patamares entre o caminho poente e a rampa: relva rasa com muretes (Google Maps, fotografia do castelejo)"),
    ("relva", chao.poligono([(-33, 8), (-30, 30), (-22, 33), (-12, 23), (-8, 12), (-11, 5), (-20, 2)]),
     "patamares a poente da Casa Paroquial: relva com arbustos"),
]


def main():
    cl = proposta()
    for classe, g, _nota in CORRECOES:
        for n in list(cl):
            cl[n] = cl[n].difference(g)
        cl[classe] = cl.get(classe, shapely.Polygon()).union(g)
    cl = chao.resolver(cl, dentro=chao.disco(MEIO - 1))
    cl = chao.preencher(cl, chao.disco(MEIO - 1))
    chao.gravar(AQUI / "dados" / "chao.json", cl, {"fonte": "ortofoto DGT 2025 + LiDAR DGT 2024 + correções à mão",
                                                    "correcoes": [c[2] for c in CORRECOES]})
    for n, g in cl.items():
        print(f"{n:8s} {g.area:9.0f} m²  {len(shapely.get_parts(g)):4d} polígonos")


if __name__ == "__main__":
    main()
