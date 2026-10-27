# Maqueta 3D do Castelo de Guimarães (estado atual)

Versão 0.1, **provisória**. Gerada por código, à escala 1:1 e georreferenciada (origem no centro da torre de menagem, 41.447926, −8.290429; x para este, y para norte; no visor, −z é o norte). Inclui o castelo, a Igreja de São Miguel do Castelo, o Paço dos Duques de Bragança e a estátua de D. Afonso Henriques. O relevo, as plantas das torres, as cotas e as alturas são medidos no LiDAR da DGT de 2025.

## Como reconstruir

```sh
# 1. contorno do castelo, Paço, igreja, casario e ruas do OSM (via Overture Maps) -> dados/fontes.json
pip install pyarrow shapely
python pipeline/maquetas/guimaraes/recolher.py

# 2. LiDAR da DGT -> dados/lidar.npz, dados/casas.json e dados/orto.jpg
#    Descarregar do Centro de Dados da DGT (cdd.dgterritorio.gov.pt) para uma pasta:
#    MDT-50cm e MDS-50cm das folhas 186498 e 187498 (2025) e a nuvem de pontos
#    LO-186498-07-2025 e LO-187498-07-2025.
pip install rasterio pyproj scipy pillow "laspy[lazrs]" numpy
python pipeline/maquetas/guimaraes/lidar.py PASTA

# 3. maqueta em Blender (módulo bpy) -> public/maquetas/guimaraes/guimaraes.glb (Draco) e capa.jpg
pip install bpy==5.2.3 numpy rasterio shapely scikit-image   # Python 3.13
python pipeline/maquetas/guimaraes/construir.py --vistas
```

A nuvem de pontos de 2025 traz cor (formato 8, RGB). O `lidar.py` faz com ela uma ortoimagem de 50 cm (cor do ponto mais alto de cada célula, lacunas preenchidas pela célula vizinha), por isso não é precisa ortofoto.
O `lidar.npz`, o `casas.json` e o `orto.jpg` estão no repositório, por isso o passo 2 só é preciso para refazer as medições.
O `bpy` do PyPI não traz o Draco, por isso o script comprime o GLB com `npx @gltf-transform/cli draco` (é preciso Node).
`--vistas` grava a capa e as vistas de verificação em `vistas/`. Cada construção grava em `dados/medidas.json` o que se mediu (cotas absolutas em metros).

## O que se mede no LiDAR e o que é estimativa

- **Terreno**: MDT de 50 cm amostrado a 1 m, sem retoques.
- **Torres de canto e torre de menagem**: planta tirada do MDS (retângulo mínimo das células acima do topo da muralha, à volta de um ponto escolhido), topo dentro da planta, pé no anel exterior. O OSM só tem o contorno exterior, desviado c. 1 m para oeste (corrigido com `DESVIO_OSM`).
- **Torres das portas e eixo dos panos**: desenhados à mão sobre o MDS e a ortoimagem (`PANOS`, `torres` no `construir.py`).
- **Muralhas**: topo dos merlões medido ao longo do eixo de 2 em 2 m; pé exterior e chão interior no MDT.
- **Paço**: planta do OSM (anel exterior e pátio); altura das fachadas medida em três profundidades; telhados separados por corpos com uma bacia hidrográfica (watershed) do MDS, cada corpo com telhado de quatro águas quando é retangular ou com telhado pela distância ao bordo quando não é; chaminés nos picos do MDS mais de 1,2 m acima da vizinhança.
- **Igreja**: nave e capela-mor tiradas das células de classe edifício, com cumeeiras e beirados medidos.
- **Estátua**: posição do OSM, altura total medida na nuvem de pontos.
- **Estimativas**: espessura das muralhas, desenho das ameias chanfradas e parapeito, frestas, janelas da alcáçova, arcos das portas, percurso do passadiço, janelas e arcarias do Paço, figura da estátua e a forma das árvores.

## Onde está a leitura das fontes

Tudo o que é interpretação está em `src/data/maquetas/guimaraes.json`: estado, grau de confiança, medidas, fontes, limitações e as placas do visor. Cada elemento é um objeto próprio no GLB (`userData.elemento`).

## Elementos e grau de confiança

| Elemento | Estado | Confiança | Medidas e base | Fontes |
|---|---|---|---|---|
| Torre de menagem | estado-atual | alta | Planta c. 13 × 15 m (LiDAR). Topo dos merlões a 268,3 m, 27,1 m acima do pé e c. 26 m acima da praça de armas (SIPA: c. 27 m). | sipa, mm, lidar, fotos |
| Passadiço da torre de menagem | estado-atual | média | — | sipa, fotos |
| Porta principal | estado-atual | alta | Soleira a 241,5 m, c. 2 m acima do largo exterior; torres de 12 a 13 m acima do pé (LiDAR). | sipa, lidar, orto, fotos |
| Porta da Traição | estado-atual | média | Soleira a c. 238,5 m; torres de 14 a 15 m acima do pé (LiDAR). | sipa, lidar, orto |
| Alcáçova | estado-atual | média | Topo a 255,3–257,1 m, 17 a 18 m acima do pé exterior e 13 a 15 m acima da praça (LiDAR). | sipa, lidar, fotos |
| Pano leste (sul da porta) | estado-atual | alta | Topo a 249,3–256,8 m, 10,5 a 15,5 m acima do pé exterior (LiDAR). | sipa, lidar, orto, fotos |
| Pano leste (norte da porta) | estado-atual | alta | Topo a 250,3–255,8 m, 10,4 a 13,7 m acima do pé exterior (LiDAR). | sipa, lidar, orto, fotos |
| Pano oeste (norte) | estado-atual | alta | Topo a 251,1–255,7 m, 15,4 a 20,9 m acima do pé exterior (LiDAR). | sipa, lidar, orto, fotos |
| Pano oeste (sul) | estado-atual | alta | Topo a 250,8–257,8 m, 11,2 a 17,4 m acima do pé exterior (LiDAR). | sipa, lidar, orto, fotos |
| Pano sul | estado-atual | alta | Topo a 250,3–256,9 m, 11,4 a 18,1 m acima do pé exterior (LiDAR). | sipa, lidar, orto, fotos |
| Torre nordeste | estado-atual | alta | 19,1 m acima do pé; topo a 256,2 m (LiDAR). | sipa, lidar, orto, fotos |
| Torre sul | estado-atual | alta | 19,2 m acima do pé; topo a 256,9 m (LiDAR). | sipa, lidar, orto, fotos |
| Torre sudoeste | estado-atual | alta | 19,6 m acima do pé; topo a 257,6 m (LiDAR). | sipa, lidar, orto, fotos |
| Torre noroeste | estado-atual | alta | 21,1 m acima do pé; topo a 255,7 m (LiDAR). | sipa, lidar, orto, fotos |
| Igreja de São Miguel do Castelo | estado-atual | alta | Nave c. 16,6 × 9,4 m, cumeeira a 239,4 m; capela-mor c. 6,1 × 6,9 m, cumeeira a 237,7 m (LiDAR). | sipa, wiki, lidar, osm, fotos |
| Paço dos Duques de Bragança | estado-atual | média | Fachadas de 15 a 21 m acima do pé, cumeeiras a c. 247 m; 17 corpos de telhado e 36 chaminés medidos (LiDAR). | paco, osm, lidar, orto, fotos |
| Estátua de D. Afonso Henriques | estado-atual | média | 4,9 m do chão ao topo da figura, com o pedestal (LiDAR). | wiki, osm, lidar, fotos |
| Terreno | estado-atual | alta | — | lidar, orto |
| Casario | estado-atual | média | 12 edifícios (8 do OSM, 4 da nuvem de pontos), cume medido em todos. | osm, lidar |
| Caminhos e ruas | estado-atual | média | — | osm, orto |
| Vegetação | estado-atual | média | — | lidar |

Fontes: `sipa` SIPA IPA.00001060 · `mm` Museus e Monumentos de Portugal · `paco` Paço dos Duques · `cmg` CM Guimarães · `wiki` Wikipédia · `osm` OpenStreetMap via Overture 2026-10-21.0 · `lidar` DGT LiDAR 2025 (CC BY 4.0) · `orto` ortoimagem da cor dos pontos LiDAR · `fotos` fotografias fornecidas.

## O que falta para passar de provisória a validada

- Planta cotada (levantamento DGEMN ou da Museus e Monumentos) para confirmar a espessura das muralhas, a planta das torres das portas e o interior da praça de armas (escadas, vestígios escavados).
- Posição das frestas e seteiras, da antiga porta entaipada da torre sul e do medalhão.
- Torreões cónicos, janelas e arcarias do Paço a partir de alçados ou fotografias de cada fachada.
