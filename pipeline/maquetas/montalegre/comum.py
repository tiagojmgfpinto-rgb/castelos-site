"""Constantes partilhadas da maqueta do Castelo de Montalegre (Terras de Barroso).

Referencial: origem no centro da torre de menagem, x para este, y para norte (PT-TM06 menos a origem), cotas =
altitudes. A torre de menagem usa também um referencial próprio (u, v): u para o azimute 97,5° (ESE) e v para
7,5° (NNE), as direções das faces medidas no LiDAR; as outras torres têm o seu (ver construir.py).
"""
import math

# origem = centro da torre de menagem (medido no LiDAR de 2024, folha 228540)
LAT0, LON0 = 41.826068, -7.790920
E0, N0 = 28427.67, 239679.35  # PT-TM06 (EPSG:3763)
# caixa dos dados (m)
X0, X1, Y0, Y1 = -85.0, 75.0, -95.0, 55.0
BASE = 980.0

ROT_TORRE = 97.5   # azimute do eixo u da torre de menagem
_r = math.radians(ROT_TORRE)
EU = (math.sin(_r), math.cos(_r))
EV = (-math.cos(_r), math.sin(_r))


def W(u, v):
    """(u, v) do referencial da torre para (x, y) da cena."""
    return (EU[0] * u + EV[0] * v, EU[1] * u + EV[1] * v)


# centro do pátio, para o perfil da cerca por azimute (tracar.py)
CENTRO_PATIO = (-4.0, -22.5)

# ---- torres (LiDAR 2024: retângulo mínimo dos pontos do coroamento; centro, azimute do eixo u, meias medidas)
# u e v de cada torre: u para o azimute dado, v 90° à esquerda de u (para u + 270°)
TORRES = {
    "menagem": {"c": (0.0, 0.0), "az": 97.5, "mu": 6.05, "mv": 6.02},
    "se": {"c": (2.05, -46.62), "az": 175.75, "mu": 5.03, "mv": 5.07},      # torre sueste (a grande, com merlões)
    "s": {"c": (-14.05, -40.57), "az": 128.0, "mu": 4.37, "mv": 3.75},      # torre sul (telhado por dentro do parapeito liso)
    "so": {"c": (-23.68, -27.84), "az": 161.5, "mu": 2.51, "mv": 2.91},     # torre pequena de sudoeste (pirâmide baixa)
}


def eixos(az):
    r = math.radians(az)
    eu = (math.sin(r), math.cos(r))
    ev = (-math.cos(r), math.sin(r))
    return eu, ev


def canto(nome, u, v):
    t = TORRES[nome]
    eu, ev = eixos(t["az"])
    return (t["c"][0] + eu[0] * u + ev[0] * v, t["c"][1] + eu[1] * u + ev[1] * v)


def planta_torre(nome, folga=0.0):
    t = TORRES[nome]
    a, b = t["mu"] + folga, t["mv"] + folga
    return [canto(nome, -a, -b), canto(nome, a, -b), canto(nome, a, b), canto(nome, -a, b)]
