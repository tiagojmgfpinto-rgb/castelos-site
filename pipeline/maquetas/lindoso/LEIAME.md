# Maqueta 3D do Castelo de Lindoso e dos espigueiros (estado atual)

Versão 0.3, **provisória**. A torre de menagem, a cerca medieval com a torre da porta, a Porta Norte, a porta da traição e as guaritas, a muralha abaluartada do século XVII com os baluartes e o revelim, os edifícios do museu no pátio e os espigueiros da eira. As casas à volta são volumes simples num só cinzento. Gerada por código, à escala 1:1 e georreferenciada: origem no centro da torre (41,867167, −8,199305; PT-TM06 −5495,86, 244189,75), x para este e y para norte, cotas = altitudes. Caixa do modelo: x −124..74, y −156..64.

## Como reconstruir

```sh
# 1. plantas das casas, espigueiros e ruas do OSM (via Overture Maps) -> dados/fontes.json
pip install pyarrow shapely pyproj
python pipeline/maquetas/lindoso/recolher.py

# 2. LiDAR da DGT -> dados/lidar.npz, dados/superficie.npz (com a cor RGB), dados/torre.npz
#    Do Centro de Dados da DGT (cdd.dgterritorio.gov.pt), numa pasta: MDT-50cm e MDS-50cm da folha 194545 (2025)
#    e a nuvem de pontos LO-194545-07-2025.
pip install rasterio "laspy[lazrs]" numpy scipy pillow
python pipeline/maquetas/lindoso/lidar.py PASTA

# 3. ortofotomapa 2021 da DGT (WMS) -> dados/orto.jpg
python pipeline/maquetas/lindoso/ortofoto.py

# 4. muros traçados no LiDAR -> dados/muralhas.json; manchas de espigueiros -> dados/espigueiros_lidar.json
pip install scikit-image
python pipeline/maquetas/lindoso/tracar.py
python pipeline/maquetas/lindoso/espigueiros.py

# 5. maqueta em Blender (módulo bpy) -> public/maquetas/lindoso/lindoso.glb (Draco), capa.jpg e dados/medidas.json;
#    vistas/ com --vistas
pip install bpy==5.2.3 numpy scipy shapely rasterio   # Python 3.13
python pipeline/maquetas/lindoso/construir.py --vistas

# 6. dados do visor (placas, elementos, limitações) -> src/data/maquetas/lindoso.json
python pipeline/maquetas/lindoso/visor.py

# 7. imagens da ficha -> public/imagens/lindoso/ (fotografias, ortofotomapa e relevo)
python pipeline/maquetas/lindoso/imagens.py PASTA_DAS_FOTOS
```

Os ficheiros de `dados/` estão no repositório. As constantes partilhadas (origem, rotação da torre, plantas do museu, da torre da porta e da Porta Norte) estão em `comum.py`. Opções do `construir.py`: `--sem-glb`, `--vistas`, `--capa`, `--so=a,b`, `--amostras=N`, `--sol=AZ`, `--blend`.

## O que se mede e o que é estimativa

- **Torre** (LiDAR, `dados/torre.npz`): quadrado de 10,5 m rodado 64,25° (faces a ENE, NNO, OSO e SSE; a rotação vem do histograma da orla da torre). Chão junto às faces: NNO 461,87, SSE 462,52, ENE 462,74, OSO 462,73. Soleira das ameias a 473,45 m, beiral do telhado a 473,05 m, vértice do telhado de quatro águas a 474,96 m, pontas dos merlões a 475,2 m (percentil 98,5 da orla, sem o mastro, que chega a 479 m). Sete merlões por face com remate piramidal (fotografias).
- **Janela da face SSE** (fotografia do pátio): centro a 60 % da largura, vista de fora, e a 52 % da altura até às ameias; c. 1,0 × 1,55 m, com portadas verdes. A porta na face OSO e as seteiras são estimativas.
- **Muros** (LiDAR, `tracar.py`): células de 25 cm com mais de 0,7 m de relevo local (top-hat com disco de 3 m) e mais de 0,9 m acima do MDT, para as lices e o pátio, que o MDT tem como chão, ficarem de fora; manchas alisadas e simplificadas a 0,45 m, cantos afiados. Topo medido em cada vértice (percentil 85 por dentro, mediana com os vizinhos); chão de fora no MDT. A cerca medieval (topo acima de 466 m, cantaria) e a muralha abaluartada (abaixo, alvenaria) traçam-se em separado: juntas, o topo triangulado fazia rampas de 471 para 464 m. Nos muros baixos, o miolo a mais de 1,7 m da orla (os terraplenos dos baluartes) sai do muro e passa a chão, levantado ao LiDAR. Topo com a mediana de 7 vértices.
- **Torre da porta** (LiDAR e fotografia de drone de sul): planta de 4,1 × 4,2 m e topo a 471,1 m; passagem em arco de volta perfeita de 2,0 × 3,2 m com arquivolta de 2,5 × 3,55 m, que segue pela cerca até ao pátio (chão de 461,2 m fora a 461,6 m dentro). Profundidade e murete do topo estimados.
- **Porta Norte** (LiDAR, fotografia e Street View): o bloco mais alto (c. 466,6 m) na ponta poente da cortina norte, a c. 3 m da face NNO da torre; a fotografia, tirada no eixo da porta, mostra a torre de frente por trás dela. Ponte levadiça: vão de 1,5 × 2,4 m com aduelas e portão de grade, aberto num rebaixo de 1,9 × 2,65 m onde o tabuleiro encaixa quando sobe; dois braços de madeira saem por rasgos acima do arco, com correntes até às pontas do tabuleiro de pranchas (1,8 × 2,5 m), baixado sobre o fosso (fundo estimado a 1,1 m). O muro segue dos dois lados: a sudoeste está coberto de hera, que o LiDAR classifica como vegetação (classes 3–5), por isso a planta vem do MDS e o topo é o MDS menos 0,4 m (`CORTINA_PN_SO` em `comum.py`); a hera vai em tufos.
- **Troço noroeste da cerca**: do cunhal oeste da torre ao cunhal noroeste, face de fora medida no LiDAR (parapeito a 468 m) e desenhada à mão (`CERCA_NO`); por dentro encosta-lhe o telhado do museu (467,3 m), e o top-hat não o separa. O topo norte da ala oeste do museu fica dentro deste troço.
- **Porta da traição** (Street View e fotografia): arco de 0,9 × 2,2 m na face de fora do troço noroeste, a 1,5 m do cunhal oeste da torre, com a rampa metálica e guarda que sobe da lice ao longo da face (c. 6,5 m; o LiDAR mostra-a 0,3–0,6 m acima da lice). Como o muro traçado é maciço, o vão é desenhado para fora da face, com 0,3 m de ombreiras.
- **Porta da cerca sul** (fotografia de drone de sul e Street View): vão retangular de 0,9 × 1,7 m com ombreiras e padieira, na base da cerca sul a c. 4,5 m do cunhal sudoeste; posição a c. 1 m.
- **Guaritas**: no cunhal sueste da cerca (fotografia, com mísula e gárgula) e na ponta do baluarte poente (ao fundo na mesma fotografia); 1,25 m de diâmetro e 2,1 m de corpo, estimados.
- **Museu**: plantas das duas alas e da galeria (LiDAR e satélite, em `comum.py`); cornija (percentil 15 numa faixa junto às paredes) e cume (percentil 97) do LiDAR; portas e janelas esquemáticas.
- **Espigueiros**: plantas estreitas do OSM (65) e manchas do LiDAR sem par no OSM (`espigueiros.py`); comprimento, largura e orientação da planta, cume do MDS (só quando fica a 2,9–4,3 m do chão; acima disso há uma árvore por cima e o cume vai para 3,55 m; plantas com menos de 3,4 m de comprido são arrumos). Pés e mesas aos pares (c. 1,7 m entre pares), soleira, esteios de granito a cada c. 0,95 m com grades de ripas de madeira, padieira, empenas de granito, porta em arco num topo, telhado de lajes a 27° com beiral, cumeeira, cruz e pináculo (fotografias). A largura do corpo é a da planta menos 0,7 m (a planta é a do telhado).
- **Chão**: MDT de 50 cm; os terraplenos que o MDT não tem como chão vêm do MDS; junto aos muros o chão baixa até 1,2 m na faixa de 1,25 m onde o MDT sobe para o topo por interpolação. Junto aos muros e por baixo dos muros finos o chão fica no mais baixo das células à volta (sem isto o LiDAR do topo fazia dentes de erva pelas faces). Pátio relvado; no monte, erva verde com manchas amareladas (Street View), relvado onde o ortofotomapa é verde e afloramentos só onde o relevo local do MDT passa de 0,22 m ou a pedra é cinzenta no ortofotomapa; ruas do OSM.
- **Envolvente**: plantas do OSM e, onde faltam, manchas de edifícios do LiDAR (classe 6), sem as plantas dos espigueiros; cornija e cume do MDS (no máximo 4 m de telhado; arrumos com menos de 20 m² até 4 m do chão, porque debaixo de uma árvore o MDS dá a copa). Árvores nas posições do LiDAR, sem as que caíam sobre o castelo ou os espigueiros.

## Fontes de conferência

- Fotografias enviadas pelo Tiago (as de marca de água de outro autor só serviram de referência).
- Imagem de satélite e ortofotomapas 2021 e 2023 da plataforma do Município de Ponte da Barca: `https://pontedabarca.city-platform.com/v2/app?a=pdm&z=20&c=41.8672,-8.1993` (no Playwright, fechar o painel para ver o mapa inteiro). As capturas foram georreferenciadas em Web Mercator e alinhadas com o LiDAR por correlação de arestas; têm deslocamento de relevo (os topos dos muros caem para fora), por isso a planta segue o LiDAR.

## Visor

O visor procura o grupo `Muralhas_grupo` para enquadrar e o objeto `Terreno` para a caixa do chão; os elementos têm `userData.elemento` (`torre`, `cerca`, `baluartes`, `porta_sul`, `porta_norte`, `postigo`, `guaritas`, `museu`, `espigueiros`, `terreno`, `envolvente`, `vegetacao`). As âncoras das placas e os pontos de vista são calculados no `construir.py` e gravados em `dados/medidas.json`; o `visor.py` lê-os.

## Vistas de verificação

`vistas/` repete os pontos de vista das fotografias (`f296_sul`, `73c3_norte`, `5a85_sudoeste`, `26f5_torre`, `d049_porta_norte`, `guarita_se`, `33d8_espigueiro`, `d846_espigueiros`) e do Street View (`sv_porta_norte`, `sv_postigo`, `sv_cerca_sul`), e acrescenta `porta_sul`, `perto_postigo`, `perto_porta_sul`, `ar_oeste` e `topo` (para comparar com o ortofotomapa).
