# Maqueta 3D da Fortaleza de Peniche (estado atual)

Versão 0.3, **provisória**. Gerada por código, à escala 1:1 e georreferenciada (origem num ponto do recinto, 39.353123, −9.381143, PT-TM06 −107575, −34245; x para este, y para norte; no visor, −z é o norte; as cotas são altitudes). Abrange a fortaleza, as arribas, o mar à volta e o princípio da vila (11,6 ha). O relevo, as plataformas, os parapeitos, as cotas, as alturas e as pegadas são medidos no LiDAR da DGT de 2024.

## Como reconstruir

```sh
# 1. edifícios, ruas e usos do solo do OSM (via Overture Maps) -> dados/fontes.json
pip install pyarrow shapely
python pipeline/maquetas/peniche/recolher.py

# 2. LiDAR da DGT -> dados/lidar.npz, dados/casas.json e dados/orto.jpg
#    Do Centro de Dados da DGT (cdd.dgterritorio.gov.pt), para uma pasta:
#    MDT-50cm e MDS-50cm da folha 092266 (2024) e a nuvem de pontos LO-092266-06-2024.
pip install rasterio pyproj scipy pillow "laspy[lazrs]" numpy
python pipeline/maquetas/peniche/lidar.py PASTA

# 3. plataformas (terraplenos e edifícios) e parapeitos -> dados/plataformas.json
pip install scikit-image
python pipeline/maquetas/peniche/plataformas.py

# 4. maqueta em Blender (módulo bpy) -> public/maquetas/peniche/peniche.glb (Draco) e capa.jpg
pip install bpy==5.2.3 numpy scipy shapely   # Python 3.13
python pipeline/maquetas/peniche/construir.py --vistas

# 5. imagens da ficha -> public/imagens/peniche/
python pipeline/maquetas/peniche/imagens.py
```

Os ficheiros de `dados/` estão no repositório, por isso os passos 1 a 3 só são precisos para refazer as medições.
O `bpy` do PyPI não traz o Draco: o script comprime o GLB com `npx @gltf-transform/cli draco` (é preciso Node).
Opções do `construir.py`: `--sem-glb` (só a cena), `--vistas` (capa e vistas de verificação em `vistas/`), `--capa` (só a capa), `--so=a,b` (só estas vistas), `--sufixo=x` (acrescenta x ao nome das vistas), `--amostras=N` (amostras do Cycles). Cada construção grava em `dados/medidas.json` o que se mediu e os pontos das placas.

## O que se mede no LiDAR e o que é estimativa

- **Classe 6**: aqui os terraplenos, as muralhas e os edifícios vêm todos na classe 6 da nuvem de pontos, e o MDT é interpolado por baixo deles. Por isso o MDT não serve para traçar os panos; serve para as rochas, a praça e os fossos.
- **Plataformas** (`plataformas.py`): a massa da classe 6 dentro do contorno da fortaleza, partida onde o MDS tem degraus de mais de 0,7 m; cada peça vira um polígono. O topo é uma triangulação do contorno (amostrado 1,1 m para dentro no MDS mediano 5×5) e de uma grelha de 0,7 m; os lados descem até ao terreno com talude de 1/10. Os triângulos de topo muito inclinados ganham material de muro.
- **Escarpas da costa**: por baixo de cada parapeito, do lado mais baixo, um pano de alvenaria em talude (1/8) desce da cota do chão de dentro até ao pé, medido no MDT 2 a 4,5 m para fora (mediana e mínimo ao longo do troço, no máximo 12 m de pano; junto à água entra no mar). Os intervalos curtos entre troços altos também levam pano. O terreno à frente, até 4,6 m, desce ao pé (o MDT faz uma rampa que tapava o pano). Até 8 m das escarpas tiram-se do terreno as covas (fecho morfológico de 4,5 m, sem subir acima do MDT) e as agulhas (abertura de 2,5 m). Junto ao mar, quando o relevo dá dois traços para o mesmo parapeito (metade ou mais a menos de 3,5 m de um mais comprido), fica só o mais comprido. Por baixo fica a rocha.
- **Panos das plataformas**: contorno simplificado a 0,75 m (cada pano é um plano), coroamento com mediana ao longo do pano (c. 5 m) e topo dos terraplenos no MDS filtrado (mediana 9×9 e gaussiana); cordão de pedra a 1,2 m do coroamento nos panos com mais de 3,5 m. O tabuleiro da ponte e as suas guardas, que também vêm na classe 6 e no relevo, são tirados das plataformas e dos parapeitos.
- **Parapeitos**: os da costa vêm classificados como solo. Traçam-se pelo relevo local do MDS (MDS menos abertura 9×9 acima de 0,45 m), esqueleto e simplificação (0,35 m). De 0,5 em 0,5 m mede-se o topo (percentil 20 numa janela de 9) e os merlões ficam onde o percentil 85 passa 0,35 m acima, de 3,2 em 3,2 m. Um parapeito anda de nível: onde o topo do traço cai 2,5 m abaixo do nível do troço, ou numa ponta que desce mais de 0,6 m por metro, o relevo do MDS é a aresta da arriba e não um muro, e esse bocado sai (eram os troços amarelos que desciam pela rocha); os degraus a meio de um parapeito ficam. São amarelos a menos de 25 m do mar aberto e de pedra para o lado da vila; a parte de baixo, abaixo do chão de dentro, é alvenaria.
- **Guaritas**: nos ângulos salientes com mais de 45° e mais de 5 m de queda, a mais de 26 m umas das outras (saem 10), mais a da porta da ponte. A forma segue uma fotografia.
- **Fortim Redondo**: centro (−10,5; −63,3), raio (11,8 m), terrapleno (17,05 m, só a sul de y = −57,3), pátio à cota da praça (14,3 m) e frente reta em y = −50,8, com dois panos retos até ao círculo, tudo lido no MDS; muro em talude amarelo, cordão de pedra, merlões, portal com frontão, cartela e armas, lápide e sineira com varanda e sino pelas fotografias e pela descrição do MNRL.
- **Prisão e torre**: pegadas da classe 6 e alturas do MDS; platibanda, soco, janelas por piso (2,6 m), abertas na espessura da parede (o pano das fachadas tem 0,3 m e os vãos são reentrâncias com grades a meio) e topo sudeste amarelo pelas fotografias. Torre branca a (−9,5; 60), até 36 m.
- **Ponte, porta e túnel**: tabuleiro a 10,75 m e extremos no LiDAR; quatro arcos (vãos de 4,6 a 5,4 m), nichos nos pilares e guardas pela fotografia; debaixo do tabuleiro o chão desce até c. 3,2 m para os vãos ficarem abertos, e o fosso (polígono GULLY) tem textura de rocha; porta com guarita por cima.
- **Túnel da água do mar**: leva a água da baía (a sul) ao fosso da ponte, o charco junto à entrada. As bocas estão nos entalhes que o MDT mostra nas duas margens: na baía em (72; 20,3), ao pé da escarpa do parapeito da costa, e no fosso em (82,5; 44,2), na face do pano em talude, onde as fotografias mostram o arco com água dentro. Cada boca tem paramento de alvenaria, arco de aduelas e pés-direitos de cantaria; o canal à frente desce abaixo da água e os rochedos ficam afastados. Por dentro, abóbada de berço de 3,8 m de vão com o fecho a 3,2 m e água à cota do mar, numa reta entre as bocas (c. 25 m). O traçado por dentro e as medidas da abóbada são estimativa.
- **Mar**: plano a 0,4 m; água onde o MDT está abaixo desse nível, em manchas de mais de 150 m² (o mar aberto, o porto e o fosso da ponte). O fundo desce com a distância à costa; cor turquesa nos baixios e azul-petróleo no fundo, espuma a menos de 1,6 m da costa, ondulação em mapa de normais. Rochedos à tona tirados da nuvem de pontos, com banda molhada.
- **Terreno**: MDT de 50 cm simplificado para 320 mil triângulos; as faces muito inclinadas fora das plataformas levam a textura fotográfica de calcário das arribas. Cor da ortoimagem da nuvem de pontos, escurecida e aquecida.
- **Relvados**: polígonos desenhados a partir das fotografias (em frente do Fortim e ao longo da muralha da costa), relva 6 cm acima do terreno e lancil branco de 0,2 × 0,12 m.
- **Vila**: edifícios do OSM fora da fortaleza, volumes simples com altura do MDS. Árvores nas posições da nuvem de pontos; quatro palmeiras pelas fotografias.
- **Estimativas**: perfil e espessura dos panos, merlões, guaritas, portal e sineira do Fortim, janelas da prisão, arcos da ponte, cor das casas, forma das árvores e dos rochedos.

As fotografias que o autor do projeto mandou serviram só de referência para as formas e as cores; não estão no repositório.

## Realismo (12 nov. 2026)

- Alvenaria e cantaria fotográficas (`kit/fotos.py`, fotos CC0 da Poly Haven: `old_stone_wall` e `stone_brick_wall_001`), recoloridas para o calcário cinzento-amarelado das fotografias; telha `roof_09`. As arribas usam `cliff_side` (calcário em camadas partidas) no cinzento das rochas de Peniche; a antiga textura de estratos desenhada à mão fazia um ondulado amarelo e foi retirada (12 nov. 2026). Créditos em `dados/medidas.json` (`texturas_foto`).
- As casas da vila, fora da fortaleza, ficam num só cinzento claro (paredes e telhados), como em Melgaço, para o destaque ficar na fortaleza.
