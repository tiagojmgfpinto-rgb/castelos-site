# Maqueta 3D da Torre de Menagem de Braga (estado atual)

Versão 0.2, **provisória**. Só a torre, com o largo à volta e as casas vizinhas em volumes simples (quadrado de 44 m). Gerada por código, à escala 1:1 e georreferenciada: origem no centro da torre (41,551322, −8,423892), x para este e y para norte em PT-TM06, cotas = altitudes.

## Como reconstruir

```sh
# 1. plantas das casas e ruas do OSM (via Overture Maps) -> dados/fontes.json
pip install pyarrow shapely pyproj
python pipeline/maquetas/braga/recolher.py

# 2. LiDAR e ortofotomapa da DGT -> dados/lidar.npz, dados/torre.npz, dados/orto.jpg
#    Do Centro de Dados da DGT (cdd.dgterritorio.gov.pt), numa pasta: MDT-50cm e MDS-50cm da folha 175510 (2025),
#    a nuvem de pontos LO-175510-07-2025 e o Ortofotomapa 2021 ortos2021_cog_25cm_rgbi_jpg_070-1.
pip install rasterio "laspy[lazrs]" numpy pillow
python pipeline/maquetas/braga/lidar.py PASTA

# 3. maqueta em Blender (módulo bpy) -> public/maquetas/braga/braga.glb (Draco) e capa.jpg; vistas/ com --vistas
pip install bpy==5.2.3 numpy scipy shapely   # Python 3.13
python pipeline/maquetas/braga/construir.py --vistas

# 4. imagens da ficha -> public/imagens/braga/
python pipeline/maquetas/braga/imagens.py
```

Os ficheiros de `dados/` estão no repositório. Opções do `construir.py`: `--sem-glb`, `--vistas`, `--so=a,b`, `--amostras=N`, `--sol=AZ`.

## O que se mede e o que é estimativa

- **Planta e rotação** (LiDAR): o retângulo mínimo dos pontos do topo dá faces a 71,7° e 161,7°. A torre tem c. 10,4 × 10,0 m entre faces. O LiDAR quase não tem pontos nas paredes; o erro em planta é de 0,2 a 0,3 m. O código trabalha num referencial próprio da torre (u para ENE, v para NNO), e as faces chamam-se ENE, NNO, OSO e SSE.
- **Cotas** (LiDAR): praça a 191,2–191,6 m; soleira das ameias a 215,65 m; pontas dos merlões a 216,6–217,0 m; telhado de quatro águas por dentro do parapeito, entre 214,6 m (beiral) e 216,15 m (cume), que também se vê no ortofotomapa.
- **Escada** (LiDAR): 1.º lanço encostado à face SSE (de u 0,4 a 4,9), patamar no cunhal, 2.º lanço encostado à face ENE até ao patamar da porta (196,6 m). Guarda-corpo de pedra c. 0,9 m acima dos degraus, com o topo medido (195,4 e 197,55 m). Os degraus são contados para dar c. 0,19 m de espelho.
- **Coroamento** (fotografias): balcões de 2,45 m ao longo da face, salientes 0,55 m, parapeito de 215,1 a 216,3 m, sobre três mísulas de quatro fiadas por face (a do cunhal com cinco); quatro merlões piramidais em cada face do balcão e três de 0,68 m entre os balcões. A fotografia de frente da face sul (torre3) foi endireitada: leu-se com 87,5 px/m na horizontal (largura da face) e 103 px/m na vertical (altura do LiDAR).
- **Vãos** (fotografias): SSE, janela geminada (luzes de 0,52 × 1,35 m, 211,5–212,85 m, vidraça em losangos, arco de descarga e peitoril) e duas seteiras (207,7–208,9 e 203,3–204,6 m); NNO, janela em arco de 1,1 m (fotografia de 2006, fechada com portada azul); ENE, a porta alta em arco no topo da escada e o escudo de D. Dinis por cima (Wikipédia). As janelas de ENE e OSO e o escudo de OSO são estimativas (a Wikipédia menciona-os, não há fotografia). Uma gárgula por face, como na face sul.
- **Cantaria**: textura de 6 m feita pelas fotografias (fiadas de 0,30 a 0,48 m, média 0,38 m pela periodicidade das juntas; silhares de 0,55 a 1,6 m; granito cinzento de grão médio, com pintas de biotite e feldspato, pouca diferença de tom entre silhares, manchas de óxido e escorrências; cor calibrada pela mediana das fotografias ao sol e dessaturada para o visor, que mostra a textura sem a transformação de cor do Cycles). A pedra escurece para o alto (cor por vértice). A escada tem uma textura própria, mais clara, com silhares maiores e mais líquenes.
- **Envolvente**: plantas do OSM recortadas, cornija (25.º percentil) e cume (92.º) do MDS; telhados por recuo da planta, num só cinzento, num quadrado de 44 m. A árvore a noroeste está na posição do LiDAR; um grupo de pontos do alto da torre classificado como vegetação é ignorado.
- **Padrão em coluna** (LiDAR e fotografia de 2006): ao pé do cunhal noroeste, topo do bloco a 193,4 m; proporções da fotografia.
- **Vistas de verificação** (`vistas/`): `f1` a `f5` repetem os pontos de vista das cinco fotografias (`f3_longe` sem as casas, de longe, como a teleobjetiva). As comparações lado a lado estão na pasta do projeto, em `braga-maqueta/`.

## Fotografias

As cinco fotografias enviadas pelo Tiago (duas com marca de autor de terceiros) serviram só de referência e não estão no repositório nem na ficha.

## Realismo (12 nov. 2026)

- Cantaria da torre e da escada com granito fotografado (`kit/fotos.py`, `stone_brick_wall_001`, foto CC0 da Poly Haven) em vez da cantaria procedimental: a imagem cobre 4 m (UV 2,0 × escala 0,5), com fiadas de c. 0,3–0,4 m na torre e silhares maiores na escada, recolorida para o cinzento quente medido nas fotografias. A patine por vértice mantém-se. Telha `roof_09`. Créditos em `dados/medidas.json` (`texturas_foto`). As casas da envolvente já eram cinzentas.
