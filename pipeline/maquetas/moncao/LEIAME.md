# Maqueta 3D da Fortaleza de Monção (estado atual)

Versão 0.2, **provisória** (revista com o Street View e as fotografias do Tiago: muralhas, portas, igrejas e pedra). Gerada por código, à escala 1:1 e georreferenciada (origem no centro das linhas de muralha do OSM, 42.07785, −8.48020; x para este, y para norte; no visor, −z é o norte; as cotas são altitudes). Abrange a vila dentro dos baluartes, os fossos, a frente do rio e uma faixa do Minho (44 ha). O relevo, o traçado dos panos, as cotas, as alturas e as pegadas são medidos no LiDAR da DGT de 2025. Segue o método das maquetas de Valença e de Peniche.

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

# 2b. ortofotomapa da DGT (OrtoSat 2023, WMS ortos.dgterritorio.gov.pt) -> dados/ortofoto.jpg (25 cm)
python pipeline/maquetas/moncao/ortofoto.py

# 3. superfície com os pontos das classes 1, 2 e 6 -> dados/superficie.npz; traçado dos panos -> dados/escarpas.json
python pipeline/maquetas/moncao/pontos.py PASTA
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

- **Panos** (`pontos.py`, `tracar.py`): em Monção muitos panos vêm na classe 6 (edifício) e os parapeitos na classe 1 (não classificado) da nuvem de pontos, e o MDT apaga-os. `pontos.py` guarda em `dados/superficie.npz` o máximo das classes 1, 2 e 6, o máximo da classe 1 e o mínimo da classe 2, por célula de 0,5 m. O traçado corre sobre essa superfície (ou o MDT, se for mais alto) numa faixa de 40 m à volta das linhas de muralha do OSM, fora das casas. Depois é o método de Valença (declive acima de 58° e mais de 2 m de desnível em 3 m, esqueleto, troços ligados). Fica cada peça que tenha plataforma do lado de dentro (terrapleno ou adarve a menos de 2,5 m do topo em pelo menos 40% do comprimento) ou que corra junto às linhas do OSM, fora dos socalcos do rio, dos cortes das estradas, dos degraus da Misericórdia e da encosta norte entre a vila e o rio, onde os degraus são socalcos e o passadiço de madeira em ziguezague (polígonos `EXCLUIR`). Saem 143 panos, com c. 6,2 km e 6,4 m de altura mediana. As casas do LiDAR com mais de 1000 m² são recusadas (eram panos).
- **Perfil e alvenaria**: talude, cordão, parapeito e terrapleno como em Valença; os terraplenos são as manchas onde os pontos ficam mais de 0,5 m acima do MDT. A alvenaria é uma textura de 8 m (escala própria no material, que sai no glTF como `KHR_texture_transform`), medida no Street View da Porta do Rosal: fiadas de 0,28 a 0,5 m, pedras de 0,45 a 1,1 m (e algumas pequenas), granito bege, cinzento e cor de salmão (cores tiradas da fotografia ao sol), juntas fundas e escuras de 2 a 4 cm, cantos arredondados, manchas de ferrugem, líquenes e musgo; o topo dos terraplenos é relva.
- **Guaritas**: nos ângulos salientes com mais de 38°, em panos altos com mais de 6 m de cada lado (saem 24), mais a da Porta do Rosal, sobre mísula, por cima do arco.
- **Rio Minho**: o LiDAR não tem pontos sobre a água (classe 0) e o MDT fica a c. 3 m. A água é um plano a 3,0 m nas manchas de classe 0 abaixo de 3,45 m com mais de 400 m², com o fundo rebaixado para dar a cor da água mais funda.
- **Portas**: os panos já não são cortados nas portas: `vaos()` abre na face do pano o vão do arco (o pano continua por cima e dos lados), e o túnel atravessa o terrapleno. A Porta do Rosal segue as fotografias: por fora, arco de granito de 2,1 m, túnel de 21 m e guarita sobre o cordão; por dentro, contrafortes, lojas, arcos, escadas para o adarve e gradeamento. A Porta de Salvaterra segue a fotografia: frontispício com pilastras almofadadas, arco de 2,6 m com impostas e fecho, entablamento à altura do cordão e frontão curvo interrompido com tarja.
- **Igrejas**: pegada, cumeeira e torre no MDS. A Matriz (portal de arquivoltas, óculos quadrilobados, rosácea; torre com relógio só na face da frente, sineira em arco, pilastras nos cunhais e remate em pirâmide com bola e cata-vento) e a Misericórdia seguem as fotografias, com medidas aproximadas. A Matriz: portal aberto no pano com arco de resguardo e porta em arco, empena lisa (a cachorrada fica só nas cornijas laterais), rosácea de c. 2,2 m com a moldura, sineira vazada com o sino à vista. A vista `i_matriz` usa `perto=4.5` (corte da câmara) porque a fachada abre para uma rua de c. 8 m e a fotografia foi tirada dela. A fachada da Misericórdia está virada a **norte**, para o largo (confirmado no ortofotomapa e no Street View): cunhais de granito, frontão branco com relógio, pináculos e cruz, portal com porta verde, nicho e óculo; a poente, a casa da Misericórdia e a sineira. No visor, `olhar` no JSON põe a câmara do lado das fachadas ao clicar na placa.
- **Ponte da EN202**: a ponte pedonal metálica sobre o corte da estrada a nascente, com o tabuleiro (33,9 m) e o vão (16,5 m) medidos no MDS; as silhuetas de castelo do guarda-corpo seguem a fotografia.
- **Chão e pavimentos** (`textura_terreno()`, `pavimentos()`): o ortofotomapa da DGT (OrtoSat 2023, cor verdadeira, a camada "Ortofotomapa 2023" do WebSIG da Câmara de Monção), descarregado pelo `ortofoto.py` do WMS `ortos.dgterritorio.gov.pt/wms/ortosat2023` e reamostrado a 25 cm para a grelha da maqueta (`dados/ortofoto.jpg`). Está c. 2,25 m a este e 0,5 m a norte da nuvem de pontos do LiDAR 2025 (correlação dos pormenores do chão, classe 2); o `ortofoto.py` corrige esse desvio (`DESVIO_E`, `DESVIO_N`). Os telhados preenchem-se com o chão à volta (as copas ficam, como sombra de mata debaixo das árvores) e as sombras fortes levantam-se um pouco. O pavimento verdadeiro (asfalto, cubos, a calçada da praça, os jardins e a relva) fica como no ortofotomapa; nas ruas do OSM só se tiram os carros estacionados e as sombras, que passam à cor mediana de cada troço. A encosta norte (`ENCOSTA_N`, com os socalcos e o passadiço de madeira) e os caminhos de terra ficam como na imagem.
- **Passadiço**: o passadiço de madeira em ziguezague da encosta norte segue os caminhos pedonais do OSM dentro de `ENCOSTA_N`: tabuleiro de 1,8 m a 0,35 m do terreno, com prumos e corrimãos de madeira.
- **Casario e árvores**: como em Valença (volumes simples do OSM e do LiDAR, altura no MDS; árvores nas posições medidas).

## Fotografias

As fotografias de referência (blogues e Google Maps) são de terceiros: serviram para as cores e as formas e não estão no repositório nem na ficha.
