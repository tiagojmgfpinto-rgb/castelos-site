# Maqueta 3D da Fortaleza de Valença (estado atual)

Versão 0.2, **provisória**. Gerada por código, à escala 1:1 e georreferenciada (origem no centro do polígono da fortificação no OSM, 42.030224, −8.645232; x para este, y para norte; no visor, −z é o norte; as cotas são altitudes). Abrange a praça Magistral, a Coroada, o revelim da Porta da Coroada, os fossos e as esplanadas (30,8 ha). O relevo, o traçado das escarpas, as cotas, as alturas e as pegadas são medidos no LiDAR da DGT de 2025.

## Como reconstruir

```sh
# 1. fortificação, portas, edifícios, ruas e usos do solo do OSM (via Overture Maps) -> dados/fontes.json
pip install pyarrow shapely
python pipeline/maquetas/valenca/recolher.py

# 2. LiDAR da DGT -> dados/lidar.npz, dados/casas.json e dados/orto.jpg
#    Do Centro de Dados da DGT (cdd.dgterritorio.gov.pt), para uma pasta:
#    MDT-50cm e MDS-50cm das folhas 157562 e 157563 (2025) e a nuvem de pontos LO-157562 e LO-157563.
pip install rasterio pyproj scipy pillow "laspy[lazrs]" numpy
python pipeline/maquetas/valenca/lidar.py PASTA

# 3. traçado das escarpas no MDT -> dados/escarpas.json
pip install scikit-image
python pipeline/maquetas/valenca/tracar.py

# 4. maqueta em Blender (módulo bpy) -> public/maquetas/valenca/valenca.glb (Draco) e capa.jpg
pip install bpy==5.2.3 numpy scipy shapely   # Python 3.13
python pipeline/maquetas/valenca/construir.py --vistas
```

Os ficheiros de `dados/` estão no repositório, por isso os passos 1 a 3 só são precisos para refazer as medições.
O `bpy` do PyPI não traz o Draco: o script comprime o GLB com `npx @gltf-transform/cli draco` (é preciso Node).
Opções do `construir.py`: `--sem-glb` (só a cena), `--vistas` (capa e vistas de verificação em `vistas/`), `--capa` (só a capa), `--so=a,b` (só estas vistas), `--sufixo=x` (acrescenta x ao nome das vistas), `--amostras=N` (amostras do Cycles). Há uma vista de perto de cada porta (`p_<porta>`) e de cada igreja (`i_<igreja>`). Cada construção grava em `dados/medidas.json` o que se mediu e os pontos das placas.

## O que se mede no LiDAR e o que é estimativa

- **Escarpas** (`tracar.py`): máscara das células do MDT com declive acima de 58° e mais de 2 m de desnível numa janela de 3 m; esqueleto, troços ligados e simplificados (Douglas-Peucker, 0,6 m), cantos afiados, peças juntas nas esquinas e partidas onde o lado baixo muda. Cada peça fica com o lado baixo (o fosso) à direita. Mede-se metro a metro o topo (máximo até 6 m para dentro), o pé (percentil 15 entre 0,5 e 6 m para fora), a aresta e o fim do talude, com mediana móvel de 5 pontos. Ficam só as peças a menos de 60 m do contorno da fortificação do OSM.
- **Perfil dos panos**: talude entre 1/10 e 1/4 da altura (pela largura da faixa da escarpa no MDT), cordão de meia-cana, parapeito de alvenaria de 1,1 m acima do cordão nos panos com mais de 5 m, terrapleno de 3 m por trás e cunhais de cantaria nos ângulos salientes. Onde um muro baixo encosta a um pano alto, a mudança de altura faz-se a 0,8 m do canto, sem rampa. Nos muros estreitos (as calçadas entre muros das portas da Coroada e do Meio), mede-se para dentro a distância à outra face: a face não passa do meio do muro e o maciço chega só ao meio, abaixo do chão lajeado da calçada, que fica plano à cota da soleira.
- **Guaritas**: nos ângulos salientes com mais de 45°, em panos altos com mais de 8 m de cada lado e a mais de 6 m de outra guarita (saem 37; a Burgenwelt conta 34). A forma segue uma fotografia.
- **Terreno**: MDT de 50 cm, rebaixado ao plano da face debaixo e à frente das escarpas (o MDT põe a face em rampa) e simplificado para 300 mil triângulos; textura da ortoimagem da nuvem de pontos, com telhados e copas apagados por convolução normalizada.
- **Portas**: posição do OSM, encostada à escarpa mais próxima com a normal certa; soleira no chão à frente da face. A Porta da Coroada (calçada de 3 a 4 m, à cota 73,4) e a face sul da Porta do Meio (calçada de 3 a 8 m, à cota 71,8) têm posição e soleira fixas, guardas de granito sobre os muros da calçada e frontispícios pelas fotografias. As outras frontarias são tipos (frontão por fora, arco de túnel por dentro).
- **Igrejas**: pegada, cumeeiras e altura das torres no MDS; fachadas de Santo Estêvão, da Misericórdia, de Santa Maria dos Anjos e do Bom Jesus pelas fotografias (a pegada vem do OSM). A capela do Senhor do Encontro é uma hipótese lida no telhado.
- **Casario**: quarteirões do OSM partidos em casas de c. 7,5 m de frente (duas filas se tiverem mais de 13 m de fundo), mais as casas do LiDAR que o OSM não tem; altura no MDS, telhado de quatro águas, cor sorteada. Ficam de fora as casas que caem nas igrejas e à frente das portas.
- **Árvores**: posições e alturas das copas na nuvem de pontos; folhosas ou pinheiros (altos e estreitos).
- **Estimativas**: perfil e espessura dos panos, parapeitos, guaritas, frontarias das portas além da Coroada, fachadas sem fotografia, cores das casas, forma das árvores e da estátua.
