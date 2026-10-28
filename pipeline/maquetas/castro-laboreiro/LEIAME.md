# Maqueta 3D do Castelo de Castro Laboreiro (estado atual)

Versão 0.1, **provisória**. Gerada por código, à escala 1:1 e georreferenciada (origem no centro do recinto, 42.022746, −8.158370; x para este, y para norte; no visor, −z é o norte). É só o monte do castelo, num círculo de 92 m de raio: os dois recintos, as portas, o reduto da ponta sul, os vestígios da capela, o marco geodésico e os afloramentos de granito à volta. O relevo, o traçado dos panos, as cotas e as alturas são medidos no LiDAR da DGT de 2025.

## Como reconstruir

```sh
# 1. contorno da cerca, muro entre os recintos, portas e trilhos do OSM (via Overture Maps) -> dados/fontes.json
pip install pyarrow shapely
python pipeline/maquetas/castro-laboreiro/recolher.py

# 2. LiDAR da DGT -> dados/lidar.npz, dados/casas.json e dados/orto.jpg
#    Descarregar do Centro de Dados da DGT (cdd.dgterritorio.gov.pt) para uma pasta:
#    MDT-50cm e MDS-50cm das folhas 197562 e 198562 (2025) e a nuvem de pontos
#    LO-197562-07-2025 e LO-198562-07-2025.
pip install rasterio pyproj scipy pillow "laspy[lazrs]" numpy
python pipeline/maquetas/castro-laboreiro/lidar.py PASTA

# 3. maqueta em Blender (módulo bpy) -> public/maquetas/castro-laboreiro/castro-laboreiro.glb (Draco) e capa.jpg
pip install bpy==5.2.3 numpy scipy shapely   # Python 3.13
python pipeline/maquetas/castro-laboreiro/construir.py --vistas
```

O `lidar.npz`, o `casas.json` (vazio: não há casas no monte) e o `orto.jpg` estão no repositório, por isso o passo 2 só é preciso para refazer as medições. A ortoimagem é feita com a cor dos pontos LiDAR de 2025 (cor do ponto mais alto de cada célula de 50 cm).
O `bpy` do PyPI não traz o Draco, por isso o script comprime o GLB com `npx @gltf-transform/cli draco` (é preciso Node).
`--vistas` grava a capa e as vistas de verificação em `vistas/`. Cada construção grava em `dados/medidas.json` o que se mediu (cotas absolutas em metros) e os pontos das placas.

## O que se mede no LiDAR e o que é estimativa

- **Terreno e afloramentos**: MDT de 50 cm, sem reamostrar. A rocha separa-se da erva pela cor da nuvem de pontos (clara e pouco saturada) e pelo declive, e vai num objeto próprio (`Afloramentos`), com as fissuras escurecidas pela curvatura do terreno. Debaixo dos panos e do reduto, onde a nuvem classifica o topo das muralhas como solo, o terreno desce ao chão vizinho.
- **Panos**: linha do meio desenhada sobre o relevo local do MDS (o que sobressai numa janela de 6 m) e encostada à crista medida, metro a metro. O topo é o percentil 92 do MDS num raio de 1 m, em todas as classes (as muralhas aparecem na nuvem ora como solo ora como vegetação), filtrado ao longo do pano; pé exterior e chão interior no MDT. A frente poente da cerca e o pano poente do castelo quase não se destacam no LiDAR: seguem o OSM, como troços de 0,9 a 2,4 m acima do chão de dentro.
- **Reduto**: plataforma de 12 × 9 m lida no MDS, com o topo medido.
- **Capela**: planta quadrada e altura das paredes no MDS; o vão a norte é onde o MDS mostra o corte.
- **Marco geodésico**: pico estreito de 2 m no MDS, no ponto mais alto.
- **Portas**: Porta do Sol, porta entre os recintos e portal norte na posição do OSM; Porta da Fonte a 7 m da Porta do Sol pelo pano (fotografia). Soleira no MDT; tipo de arco pelo SIPA.
- **Estimativas**: espessura das muralhas (1,7 m no castelo pela legenda de Duarte de Armas citada no SIPA; 2 m na cerca e 1,5 a 1,6 m nos troços baixos, pelas fotografias), coroamento plano, largura e altura dos vãos, aduelas, impostas e mísulas, terrapleno do reduto, forma do marco, das árvores e das moitas.
- **Fora da maqueta**: a torre de menagem (destruída em 1659), as torres e o cubelo de Duarte de Armas, a cisterna, os alicerces do quartel e as escadas embutidas nas muralhas não se vêem no LiDAR e não há planta que os situe.

## Onde está a leitura das fontes

Tudo o que é interpretação está em `src/data/maquetas/castro-laboreiro.json`: estado, grau de confiança, medidas, fontes, limitações e as placas do visor. Cada elemento é um objeto próprio no GLB (`userData.elemento`).

Fontes: `sipa` SIPA IPA.00002273 · `cdp` Castelos de Portugal · `wiki` Wikipédia · `osm` OpenStreetMap via Overture 2026-10-21.0 · `lidar` DGT LiDAR 2025 (CC BY 4.0) · `orto` ortoimagem da cor dos pontos LiDAR · `fotos` fotografias fornecidas.

## O que falta para passar de provisória a validada

- Planta cotada (levantamento da DGEMN ou das escavações de 1975–1980) para confirmar a espessura das muralhas, a posição da cisterna, do quartel e das escadas embutidas, e o traçado da frente poente.
- Fotografias de perto do portal norte, da Porta da Fonte e do reduto, para as medidas dos vãos e o remate do reduto.
