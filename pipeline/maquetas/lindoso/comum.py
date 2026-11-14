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
# cubelo poente da porta sul (LiDAR: bloco maciço com o topo a c. 471 m); antes era a "torre da porta" com o arco
TORRE_DA_PORTA = [(6.6, -28.6), (10.7, -28.4), (10.8, -32.6), (6.5, -32.7)]
ALA_OESTE = [(-13.7, -3.9), (-7.9, 0.4), (-1.4, -20.0), (-8.1, -21.8)]       # museu: ala encostada à cerca poente; o topo norte fica dentro do troço NO da cerca
CORREDOR = [(-4.5, -10.0), (0.2, -10.0), (0.2, -20.4), (-1.4, -20.0)]          # galeria envidraçada
ALA_SUL = [(-8.1, -21.8), (7.2, -20.0), (7.8, -26.4), (-6.6, -28.2)]           # museu: ala encostada à cerca sul
# Porta sul (porta de armas): o arco abre-se no reentrante entre o cubelo poente e a massa a nascente (LiDAR: chão
# a 461,3 m de x 10,5 a 15,3 e de y -32 a -29; cerca de y -29 a -24,2), com a passagem a direito pela cerca até ao
# pátio, c. 4 m a nascente do cunhal da ala sul do museu (revisão do perito, 13 nov. 2026; Commons, Hugo Ferreira;
# L. Fontes 2013, p. 96; foto de drone f296a0e3). Antes passava por trás do cubelo e o museu tapava o arco de dentro.
PASSAGEM_SUL = [(10.3, -29.4), (15.5, -29.4), (15.5, -24.1), (10.3, -24.1)]
EIXO_PORTA_SUL = 13.0  # x do eixo da passagem
# Porta Norte: atravessa a ponta poente da cortina norte da muralha abaluartada, de frente para a face NNO da torre
# (fotografia da porta com a torre por trás); eixo da cortina a 60°, centro do bloco no eixo do muro
# O vão (1,5 × 2,4 m) abre-se num rebaixo para a ponte levadiça; o muro segue dos dois lados (fotografia e
# Street View: a sudoeste da porta está coberto de hera, que o LiDAR classifica como vegetação)
PORTA_NORTE = {"c": (-0.95, 10.4), "az": 60.0, "s0": -1.1, "s1": 1.1, "meia": 0.9}


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


def _faixa(c, az, s0, s1, d0, d1):
    import math as _m
    t = (_m.sin(_m.radians(az)), _m.cos(_m.radians(az)))
    n = (-t[1], t[0])
    return [(c[0] + t[0] * s + n[0] * d, c[1] + t[1] * s + n[1] * d) for s, d in ((s0, d0), (s1, d0), (s1, d1), (s0, d1))]


# Muros que o traçado automático não apanha:
# - a cortina a sudoeste da Porta Norte, coberta de hera (só no MDS, com todas as classes); topo = MDS menos a hera;
# - o troço noroeste da cerca medieval, do cunhal oeste da torre ao cunhal noroeste: por dentro encosta-lhe o
#   telhado do museu (467,3 m), por isso não se destaca no top-hat; face de fora medida no LiDAR (parapeito a 468 m).
CORTINA_PN_SO = _faixa(PORTA_NORTE["c"], PORTA_NORTE["az"], -12.0, PORTA_NORTE["s0"] - 0.15, -1.0, 1.0)
CERCA_NO_FORA = [(-7.0, 2.6), (-9.5, 1.3), (-12.2, -0.6), (-13.6, -1.9), (-16.6, -4.1)]
CERCA_NO = CERCA_NO_FORA + [(x + 0.91, y - 1.31) for x, y in reversed(CERCA_NO_FORA)]
# porta da traição (Street View e fotografia): arco estreito na face de fora do troço NO, junto ao cunhal da torre,
# com a rampa metálica que sobe da lice ao longo da face
POSTIGO = {"o": (-8.23, 1.75), "t": (-0.82, -0.57), "vao": (0.9, 2.2), "rampa": 6.5}
# porta pequena na base da cerca sul, a c. 4,5 m do cunhal sudoeste (fotografia de drone de sul e Street View)
PORTA_CERCA_SUL = {"o": (-7.0, -31.8), "t": (0.995, 0.10), "vao": (0.9, 1.7)}

