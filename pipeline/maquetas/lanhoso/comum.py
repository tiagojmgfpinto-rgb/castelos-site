"""Constantes partilhadas da maqueta do Castelo de Lanhoso (Póvoa de Lanhoso).

Referencial: origem no centro da torre de menagem, x para este, y para norte (PT-TM06 menos a origem), cotas =
altitudes. A torre e a cerca usam também um referencial próprio (u, v): u para o azimute 73,25° (ENE) e v para
343,25° (NNO), as direções das faces da torre medidas no LiDAR. As plantas abaixo estão em (u, v), em metros.
"""
import math

# origem = centro da torre de menagem (medido no LiDAR de 2025, folha 187514)
LAT0, LON0 = 41.586850, -8.281361
E0, N0 = -12361.95, 213063.98  # PT-TM06 (EPSG:3763)
# caixa dos dados (m); a sul acaba no limite da folha 187514 do LiDAR (N = 213000, y = -63,98)
X0, X1, Y0, Y1 = -95.0, 85.0, -63.5, 76.0
BASE = 330.0

ROT_TORRE = 73.25
_r = math.radians(ROT_TORRE)
EU = (math.sin(_r), math.cos(_r))
EV = (-math.cos(_r), math.sin(_r))


def W(u, v):
    """(u, v) do referencial da torre para (x, y) da cena."""
    return (EU[0] * u + EV[0] * v, EU[1] * u + EV[1] * v)


# torre de menagem (LiDAR: percentis 0,5/99,5 dos pontos acima de 392,5 m)
MEIA_U, MEIA_V = 5.22, 5.15

# ---- cerca (planta traçada no LiDAR: células com relevo local > 1,2 m num disco de 3,5 m; ver LEIAME) -------
# torreão sul da porta, encostado à face OSO da torre
TORREAO_SUL = [(-11.75, 2.55), (-5.22, 2.55), (-5.22, 6.1), (-11.75, 6.1)]
# torreão norte da porta, que segue para norte como muralha poente
TORREAO_NORTE = [(-11.35, 8.45), (-7.3, 8.3), (-5.1, 8.3), (-5.1, 12.0), (-10.6, 12.05), (-11.35, 11.2)]
# muro da porta (entre os torreões, com o arco ogival), recuado da frente dos torreões
MURO_PORTA = [(-7.3, 6.1), (-5.1, 6.1), (-5.1, 8.3), (-7.3, 8.3)]
# muralha poente (com merlões), canto noroeste chanfrado, muralha norte, baluarte nordeste e muralha nascente
CERCA = [
    (-6.85, 12.0), (-6.85, 15.3), (-6.45, 17.0), (-6.5, 18.4), (-6.2, 20.1), (-4.5, 22.1), (-3.7, 22.2),
    (-2.3, 23.3), (0.3, 24.0), (3.5, 25.1), (6.7, 26.3), (9.0, 26.9), (10.5, 27.8), (12.3, 28.3),
    (14.3, 24.7), (15.9, 22.7), (14.7, 21.1), (13.7, 20.9), (11.7, 18.9), (11.15, 18.9), (11.4, 15.2),
    (11.8, 13.0), (12.0, 10.7), (12.7, 8.9), (12.75, 6.1), (13.1, 5.3), (13.4, 4.2), (11.9, 3.75), (10.3, 3.9),
    (11.3, 5.1), (11.3, 6.1), (10.6, 6.6), (10.2, 8.0), (9.6, 12.0), (9.4, 16.0), (8.9, 18.4), (9.0, 20.2),
    (10.4, 20.7), (10.8, 21.5), (12.6, 22.7), (12.6, 23.7), (11.7, 25.6), (7.3, 24.3), (5.8, 23.6), (3.7, 23.1),
    (2.5, 22.1), (0.8, 21.5), (-0.1, 20.3), (-1.9, 19.7), (-3.1, 19.7), (-4.3, 20.2), (-4.3, 15.5),
    (-4.6, 13.4), (-5.1, 12.5), (-5.1, 12.0),
]
# muro fino que fecha o pátio a sul, do cunhal NE da torre ao cunhal SE da cerca (LiDAR: topo a 386,0)
MURO_SE = [(5.22, 3.25), (12.0, 3.55), (12.0, 4.1), (5.22, 3.9)]
# adarves (cotas do LiDAR, moda dos pontos do topo entre os merlões)
Z_TORREOES = 387.58      # topo dos torreões e do muro da porta
Z_POENTE = 386.95        # adarve da muralha poente
Z_NORTE = 386.35         # adarve da muralha norte e do baluarte NE
Z_NASCENTE = 386.0       # adarve da muralha nascente
Z_MURO_SE = 386.0
# terraço por dentro da muralha norte e banco ao longo da nascente (LiDAR: 384,1)
TERRACO = [(-4.3, 17.6), (-1.0, 18.4), (2.0, 19.3), (4.2, 20.0), (7.6, 19.9), (8.4, 18.2), (8.5, 5.2), (9.0, 4.6),
           (10.3, 3.9), (11.3, 5.1), (11.3, 6.1), (10.6, 6.6), (10.2, 8.0), (9.6, 12.0), (9.4, 16.0), (8.9, 18.4),
           (9.0, 20.2), (10.4, 20.7), (10.8, 21.5), (12.6, 22.7), (12.6, 23.7), (11.7, 25.6), (7.3, 24.3), (5.8, 23.6),
           (3.7, 23.1), (2.5, 22.1), (0.8, 21.5), (-0.1, 20.3), (-1.9, 19.7), (-3.1, 19.7), (-4.3, 20.2)]
Z_TERRACO = 384.1
# plataforma de lajes ao pé da face OSO da torre (LiDAR: 384,0) e alicerces a sul da torre (383,3)
TERRACO_OSO = [(-8.4, -6.1), (-5.22, -6.1), (-5.22, 2.55), (-8.4, 2.55)]
ALICERCES = [(-1.2, -9.3), (6.6, -9.3), (6.6, -5.9), (-1.2, -5.9)]

# ---- santuário de Nossa Senhora do Pilar ----------------------------------------------------------------------
# eixo da nave a 352,25° (NNO), centro da nave (5,60, -43,96); s ao longo do eixo (+ para o altar-mor, a norte),
# q para ENE
IGREJA_C = (5.60, -43.96)
IGREJA_AZ = -7.75
