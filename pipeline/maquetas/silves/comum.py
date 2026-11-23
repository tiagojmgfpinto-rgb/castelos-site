"""Constantes comuns da maqueta de Silves: origem, caixa do LiDAR e caixas de pormenor."""
from pyproj import Transformer

# Origem provisória no centro da alcáçova (PT-TM06 E -27050, N -274940); afinada em tracar.py se mudar.
E0, N0 = -27050.0, -274940.0
LON0, LAT0 = Transformer.from_crs(3763, 4326, always_xy=True).transform(E0, N0)
X0, X1, Y0, Y1 = -360.0, 260.0, -390.0, 230.0  # caixa do LiDAR (m)
CENTRO_BASE, RAIO_BASE = (-110.0, -100.0), 245.0  # cobre a alcáçova e toda a cerca da vila
CAIXAS = {  # pontos completos (classes 1-6) para medir com detalhe
    "alcacova": (-75.0, 70.0, -90.0, 80.0),
    "cerca": (-330.0, 60.0, -300.0, 80.0),
}
