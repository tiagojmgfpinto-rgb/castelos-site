# Maqueta 3D do Castelo de Montalegre (estado atual)

Versão 0.4, **provisória** (0.4: a escada do adarve poente passa a ser o próprio muro em degraus de granito, de lado a lado, com guardas de aço galvanizado claro dos dois lados, como nas fotografias de norte e de nascente; 0.3: as entradas de fora refeitas, a poterna e a porta do sul com face plana, aduelas, vão recuado e porta de madeira, a porta do sul mudada para o azimute 230° e a porta do corredor de aço no vão do muro de pedra; 0.2: portas das torres sueste, sul e de sudoeste e arco de dentro da poterna pela fotografia de 2022 do alto da torre de menagem; grade da cisterna; pátio com relva e manchas de terra). Só o castelo: a torre de menagem com os merlões, os balcões com matacães, as guaritas dos cunhais, o telhado, a casota da escada, a bandeira, a porta alta com o patamar e a escada de aço corten; as três torres do sul; a cerca com os adarves, as guardas de ferro, a poterna e a porta em arco do sul; o corredor de entrada de aço corten no troço noroeste, a escada de aço do adarve e o passadiço de madeira; a cisterna; as duas ruínas em ferradura a norte; a barbacã com o caminho lajeado e a vedação. A envolvente é mínima: relvado, árvores do LiDAR afastadas dos muros e os edifícios da vila em volumes simples de um só cinzento. Gerada por código, à escala 1:1 e georreferenciada: origem no centro da torre de menagem (41,826068, −7,790920; PT-TM06 28427,67, 239679,35), x para este e y para norte, cotas = altitudes. Caixa do modelo: x −58..52, y −78..32.

## Como reconstruir

```sh
# 1. edifícios e ruas do OSM (via Overture Maps) -> dados/fontes.json
python pipeline/maquetas/montalegre/recolher.py

# 2. LiDAR da DGT -> dados/lidar.npz, dados/superficie.npz (com a cor RGB), dados/torre.npz
#    Numa pasta: MDT-50cm e MDS-50cm da folha 228540 (2024) e a nuvem de pontos LO-228540-09-2024.
python pipeline/maquetas/montalegre/lidar.py PASTA

# 3. ortofotomapa 2021 da DGT (WMS) -> dados/orto.jpg
python pipeline/maquetas/montalegre/ortofoto.py

# 4. perfil da cerca e barbacã no LiDAR -> dados/plantas.json
python pipeline/maquetas/montalegre/tracar.py

# 5. maqueta em Blender (módulo bpy) -> public/maquetas/montalegre/montalegre.glb (Draco), capa.jpg, dados/medidas.json
pip install bpy==5.2.3 numpy scipy shapely rasterio   # Python 3.13
python pipeline/maquetas/montalegre/construir.py --vistas

# 6. dados do visor -> src/data/maquetas/montalegre.json
python pipeline/maquetas/montalegre/visor.py

# 7. imagens da ficha -> public/imagens/montalegre/ (fotografias do Commons, ortofotomapa e relevo)
python pipeline/maquetas/montalegre/imagens.py PASTA_DAS_FOTOS_DO_COMMONS
```

Opções do `construir.py`: `--sem-glb`, `--vistas`, `--capa`, `--so=a,b`, `--amostras=N`, `--sol=AZ`, `--blend`.

## Referenciais

- **Torres** (`comum.TORRES`): cada torre tem o centro, o azimute do eixo u e as meias medidas, do retângulo mínimo dos pontos do coroamento no LiDAR. v fica 90° à esquerda de u. Torre de menagem: u para 97,5° (ESE), v para 7,5° (NNE).
- **Cerca** (`perfil.py`): perfil por azimute a partir do centro do pátio (−4,0, −22,5), de 2,5° em 2,5°: raio da face de dentro e de fora do topo e cota do adarve. As linhas que caem nas torres, nos degraus de aço e em pontos soltos são substituídas por interpolação, com mediana de 5.

## O que se mede e o que é estimativa

| Elemento | Medido (LiDAR 2024) | Das fotografias ou estimado |
|---|---|---|
| Torre de menagem | 12,1 × 12,0 m; chão 1012,5; adarve 1037,53; peitoril 1038,35; merlões 1039,05/1039,72; telhado (beiral 1037,6 a 2,05 m das faces, vértice 1039,6); casota (u −1,85..−0,15, v −4,6..−1,0, 1040,15); mastro; posições dos balcões (s 0 a sul, −0,25 a norte, 0,25 a nascente, 2,25 a poente) e pisos a c. 1034 | 9 merlões por face; guaritas e balcões (mísulas, matacães, 2 merlões por lado); porta, patamar (1017,6) e escada de aço da face sul, que está tapada no LiDAR; seteiras; par de mísulas da face nascente |
| Torre de sueste | 10,1 × 10,1 m; adarve 1027,55; merlões 1029,2/1029,8; telhado 1027,6–1029,2 | 11 merlões por face; na face do pátio, porta alta (1020,1) e porta em arco ao nível do chão; seteiras |
| Torre do sul | 8,74 × 7,5 m; parapeito liso 1025,95; telhado 1025,08–1026,3 | porta em arco para o adarve na face NE (1019,83); seteiras |
| Torre de sudoeste | 5,0 × 5,8 m; rebordo 1023,15; pirâmide 1022,9–1023,7 | porta em arco para o adarve (0,8 × 1,9) |
| Cerca | planta e cota do adarve por azimute: 1016,6 a nascente, rampa até 1019,7–1020,1 junto à torre sueste, 1019,5–1019,8 a sul e a poente | guardas de ferro; poterna (az. 137,5°) e porta do sul (az. 230°, c. 2,3 m da torre sul), 1,05 × 2,05 m, recuadas 0,75 m, com portas de madeira |
| Corredor de aço | az. 279,5–352°; cobertura 1016,25; 1,6–2 m de largura (pontos avermelhados) | muro de pedra por baixo (0,55 m acima do pátio); juntas das chapas; porta, placa e degraus (az. 323°); saída para o pátio (az. 340°) |
| Escada do adarve | de 1019,42 (az. 263°) a 1016,25 (az. 279°) | 17 degraus de granito na largura do muro, guardas de aço galvanizado dos dois lados (fotografias de norte e de nascente) |
| Passadiço | faixa mais clara e lisa entre as ruínas, de (−18,3, 15,6) à porta | tábuas, balizadores |
| Ruínas | ferraduras de c. 1 m, abertas a nascente, centros (−21,3, 11,4) e (−12,4, 15,2) | — |
| Cisterna | quadrado x −12,2..−7,2, y −30,4..−25,4 (ortofotomapa) | bocal de 0,55 m, fundo escuro |
| Barbacã | bordo onde o MDT cai do patamar (cortes de 10 m) | muro de 0,6 m, caminho lajeado de 2 m (ortofotomapa), vedação |

## Fotografias de referência e pontos de vista (`vistas/`)

As fotografias enviadas (autorizadas para a galeria em 2026-11-04, menos o cartaz e a captura do Google Maps) e as vistas que as reproduzem:
`62d1_sul` (drone de sul-sudoeste), `3b33_poente` (do relvado a poente, antes do corredor de aço), `05d9_norte` (de norte, ao longo do passadiço), `0e257_escada` (do pé da escada de aço), `4f30_nascente` (drone de nascente), `f05_torre_se` (drone por cima da torre sueste), `gm_torre_sul` (do alto da torre de menagem para sul, como a fotografia de 2022 enviada no dia 4), `escada_adarve` e `escada_adarve_norte` (a escada do adarve poente), `poterna_fora`, `porta_sul_fora` e `porta_corredor_fora` (as entradas de fora, de perto), `coroamento` e `topo` (planta para comparar com o ortofotomapa).

## Geoportal da Câmara

O MunWebGis (`sig.cm-montalegre.pt/MunWebGis/Viewer.aspx?serviceName=pdm`) serve o ortofotomapa de 2011 em TMS: `https://sig.cm-montalegre.pt/cache/1.0.0/malegre_ortos2011/{z}/{x}/{y}.png`, Datum 73 (EPSG:27493), origem (−100000, 150000); a z = 9 cada mosaico tem 90,3 m e o castelo está em x = 1422, y = 993. O servidor recusa pedidos diretos (403); funciona com `fetch` dentro de uma página aberta no Playwright. Mostra o castelo antes do corredor de aço e do passadiço.

## Limitações

As mesmas do visor (`visor.py`): sem planta cotada; a face sul da torre está tapada no LiDAR; as linhas de muralha exteriores de Duarte de Armas já não existem; o modelo do Sketchfab de 2019 (Rui Oliveira) é uma reconstituição de fases históricas e só serviu de referência de forma.
