# Maqueta 3D da Fortaleza de Monção (estado atual)

Versão 0.1, **provisória**. Gerada por código, à escala 1:1 e georreferenciada (origem no centro das linhas de muralha do OSM, 42.07785, −8.48020; x para este, y para norte; no visor, −z é o norte; as cotas são altitudes). Abrange a vila dentro dos baluartes, os fossos, a frente do rio e uma faixa do Minho (44 ha). O relevo, o traçado dos panos, as cotas, as alturas e as pegadas são medidos no LiDAR da DGT de 2025. Segue o método das maquetas de Valença e de Peniche.

## Como reconstruir

```sh
# 1. muralhas, portas, edifícios, ruas e usos do solo do OSM (via Overture Maps) -> dados/fontes.json
pip install pyarrow shapely
python pipeline/maquetas/moncao/recolher.py

# 2. LiDAR da DGT -> dados/lidar.npz, dados/casas.json e dados/orto.jpg
#    Do Centro de Dados da DGT (cdd.dgterritorio.gov.pt), para uma pasta:
#    MDT-50cm e MDS-50cm das folhas 170568 e 171568 (2025) e a nuvem de pontos LO-170568 e LO-171568.
pip install rasterio pyproj scipy pillow "laspy[lazrs]" numpy
python pipeline/maquetas/moncao/lidar.py PASTA

# 3. traçado dos panos -> dados/escarpas.json
pip install scikit-image
python pipeline/maquetas/moncao/tracar.py

# 4. maqueta em Blender (módulo bpy) -> public/maquetas/moncao/moncao.glb (Draco) e capa.jpg
pip install bpy==5.2.3 numpy scipy shapely   # Python 3.13
python pipeline/maquetas/moncao/construir.py --vistas

# 5. imagens da ficha (ortoimagem e relevo) -> public/imagens/moncao/
python pipeline/maquetas/moncao/imagens.py
```

Os ficheiros de `dados/` estão no repositório, por isso os passos 1 a 3 só são precisos para refazer as medições.
O `bpy` do PyPI não traz o Draco: o script comprime o GLB com `npx @gltf-transform/cli draco` (é preciso Node).
Opções do `construir.py`: `--sem-glb`, `--vistas`, `--so=a,b`, `--sufixo=x`, `--amostras=N`, como em Valença.

## O que se mede no LiDAR e o que é estimativa

- **Panos** (`tracar.py`): em Monção muitos panos vêm na classe 6 (edifício) da nuvem de pontos, por isso o MDT os apaga. O traçado corre sobre uma superfície mista: o MDS a menos de 4,5 m das linhas de muralha do OSM, onde a classe é 2 ou 6, e o MDT no resto. Depois é o método de Valença (declive acima de 58° e mais de 2 m de desnível em 3 m, esqueleto, troços ligados). Ficam só as peças a menos de 30 m das linhas do OSM, fora dos socalcos do rio e dos cortes das estradas (polígonos `EXCLUIR`). Saem 92 panos, com c. 5,1 km e 6,8 m de altura mediana.
- **Perfil e alvenaria**: talude, cordão, parapeito e terrapleno como em Valença. A alvenaria é de pedras grandes e irregulares, cinzento-rosadas com manchas de ferrugem, juntas fundas, líquenes e musgo, pelas fotografias; o topo dos terraplenos é relva.
- **Guaritas**: nos ângulos salientes com mais de 38°, em panos altos com mais de 6 m de cada lado (saem 15), mais a da Porta do Rosal, sobre mísula, por cima do arco.
- **Rio Minho**: o LiDAR não tem pontos sobre a água (classe 0) e o MDT fica a c. 3 m. A água é um plano a 3,0 m nas manchas de classe 0 abaixo de 3,45 m com mais de 400 m², com o fundo rebaixado para dar a cor da água mais funda.
- **Portas**: a Porta do Rosal (por fora e por dentro) segue as fotografias, com a posição do OSM e o vão aberto nos panos: arco de granito de 2,7 m, túnel, lápide, guarita; por dentro, contrafortes, lojas, escadas para o adarve e gradeamento. A Porta de Salvaterra é um tipo (não houve fotografia).
- **Igrejas**: pegada, cumeeira e torre no MDS. A Matriz (portal de arquivoltas, óculos quadrilobados, rosácea, torre com relógio e sineira) e a Misericórdia (pilastras, frontão de enrolamentos, sineira) seguem as fotografias, com medidas aproximadas.
- **Ponte da EN202**: a ponte pedonal metálica sobre o corte da estrada a nascente, com o tabuleiro (33,9 m) e o vão (16,5 m) medidos no MDS; as silhuetas de castelo do guarda-corpo seguem a fotografia.
- **Casario e árvores**: como em Valença (volumes simples do OSM e do LiDAR, altura no MDS; árvores nas posições medidas).

## Fotografias

As fotografias de referência (blogues e Google Maps) são de terceiros: serviram para as cores e as formas e não estão no repositório nem na ficha.
