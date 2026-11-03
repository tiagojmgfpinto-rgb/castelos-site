"""Constantes partilhadas da maqueta do Castelo de Lindoso."""
from pyproj import Transformer

# origem = centro da torre de menagem (medido no LiDAR de 2025)
LAT0, LON0 = 41.867167, -8.199305
E0, N0 = -5495.86, 244189.75  # PT-TM06 (EPSG:3763)
X0, X1, Y0, Y1 = -130.0, 80.0, -160.0, 70.0  # caixa da maqueta (m)
BASE = 430.0

# referencial da torre de menagem (LiDAR): u para o azimute 64,25° (ENE), v para 334,25° (NNO)
ROT_TORRE = 64.25
MEIA_TORRE = 5.25  # meia largura do corpo (10,5 m entre faces)

# zona do castelo (para o traçado dos muros) e o que não é muro: a torre de menagem, os dois edifícios do museu,
# a torre da porta (modelados à parte)
ZONA_CASTELO = [(-40, 12), (-20, 26), (2, 24), (19, 36), (28, 12), (44, 2), (36, -44), (-14, -54), (-22, -32), (-38, 0)]
TORRE_DA_PORTA = [(6.6, -28.6), (10.7, -28.4), (10.8, -32.6), (6.5, -32.7)]  # a sul, com a porta em arco (LiDAR)
ALA_OESTE = [(-14.6, -0.9), (-7.9, 0.4), (-1.4, -20.0), (-8.1, -21.8)]       # museu: ala encostada à cerca poente
CORREDOR = [(-4.5, -10.0), (0.2, -10.0), (0.2, -20.4), (-1.4, -20.0)]          # galeria envidraçada
ALA_SUL = [(-8.1, -21.8), (7.2, -20.0), (7.8, -26.4), (-6.6, -28.2)]           # museu: ala encostada à cerca sul
# passagem da porta sul pela cerca, por trás da torre da porta (sai a nascente da ala sul do museu)
PASSAGEM_SUL = [(7.4, -28.6), (10.0, -28.5), (10.0, -25.6), (7.4, -25.7)]
# Porta Norte: atravessa a ponta poente da cortina norte da muralha abaluartada, de frente para a face NNO da torre
# (fotografia da porta com a torre por trás); eixo da cortina a 60°, centro do bloco no eixo do muro
PORTA_NORTE = {"c": (-0.95, 10.4), "az": 60.0, "s0": -1.9, "s1": 1.7, "meia": 0.85}


def _bloco(c, az, s0, s1, meia, folga=0.15):
    import math as _m
    t = (_m.sin(_m.radians(az)), _m.cos(_m.radians(az)))
    n = (-t[1], t[0])
    return [(c[0] + t[0] * s + n[0] * d, c[1] + t[1] * s + n[1] * d)
            for s, d in ((s0 - folga, -meia - folga), (s1 + folga, -meia - folga), (s1 + folga, meia + folga), (s0 - folga, meia + folga))]


EXCLUIR_MUROS = [
    TORRE_DA_PORTA, PASSAGEM_SUL,
    _bloco(PORTA_NORTE["c"], PORTA_NORTE["az"], PORTA_NORTE["s0"], PORTA_NORTE["s1"], PORTA_NORTE["meia"]),
    [(-7.6, 2.75), (2.75, 7.6), (7.6, -2.75), (-2.75, -7.6)],             # torre de menagem (com 0,4 m de folga)
    ALA_OESTE, CORREDOR, ALA_SUL,
]
