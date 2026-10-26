# Maqueta 3D do Castelo de Penela (estado atual)

Versão 0.2, **provisória**. Gerada por código, à escala 1:1 e georreferenciada (origem nas coordenadas oficiais 40.031162, −8.389862; x para este, y para norte; no visor, −z é o norte). Desde a 0.2 o relevo, as cotas e as alturas são medidos no LiDAR da DGT de 2024 e o traçado é conferido na ortofoto de 2025.

## Como reconstruir

```sh
# 1. traçado e edifícios do OSM (via Overture Maps) -> dados/fontes.json
pip install pyarrow shapely
python pipeline/maquetas/penela/recolher.py

# 2. LiDAR e ortofoto da DGT -> dados/lidar.npz e dados/orto.jpg
#    Descarregar do Centro de Dados da DGT (cdd.dgterritorio.gov.pt) para uma pasta:
#    MDT-50cm e MDS-50cm das folhas 177341 e 178341 (2024), a nuvem de pontos LO-177341 e
#    LO-178341 (2024) e a ortofoto ortos2025_cog_25cm_rgbi_jpg_251-4.
pip install rasterio pyproj scipy pillow "laspy[lazrs]" numpy
python pipeline/maquetas/penela/lidar.py PASTA

# 3. maqueta em Blender (módulo bpy) -> public/maquetas/penela/penela.glb (Draco) e capa.jpg
pip install bpy==5.2.3 numpy          # Python 3.13
python pipeline/maquetas/penela/construir.py --vistas
```

O `lidar.npz` e o `orto.jpg` estão no repositório (c. 1,7 MB), por isso o passo 2 só é preciso para refazer as medições com outras folhas.
O `bpy` do PyPI não traz o Draco, por isso o script comprime o GLB com `npx @gltf-transform/cli draco` (é preciso Node).
`--vistas` grava a capa e as vistas de verificação em `vistas/` (topo, norte, sul, nascente, poente, duas oblíquas como as fotografias aéreas 1 e 2 e o pano poente).
Cada construção grava em `dados/medidas.json` o que se mediu (cotas absolutas em metros), que é a base das medidas citadas no `penela.json`.

## O que se mede no LiDAR e o que é estimativa

- **Terreno**: MDT de 50 cm amostrado a 1 m, sem retoques. As escarpas são as reais.
- **Traçado**: OSM densificado de 2 em 2 m e encostado à crista da muralha no MDS, até 4,5 m para cada lado, só onde a crista está claramente acima da linha.
- **Muralhas e castelejo**: topo dos merlões medido em cada vértice (percentil 95 do MDS num raio de 1,6 m, sem células de vegetação, mediana móvel ao longo do pano). Pé exterior e chão interior no MDT.
- **Torres e cubelo**: topo dentro da planta e pé no anel exterior.
- **Igreja, casa paroquial e vila**: cumes e beirados medidos dentro da planta do OSM.
- **Árvores**: máximos locais da altura da vegetação (MDS − MDT, pontos de classe 3 a 5), com o raio da copa pela área de vegetação de cada máximo.
- **Estimativas**: espessura das muralhas, desenho das ameias e parapeito, altura do coruchéu, vãos de portas e janelas, e a forma das árvores.

## Onde está a leitura das fontes

Tudo o que é interpretação está em `src/data/maquetas/penela.json`: estado (estado atual, reconstituição, hipótese), grau de confiança, medidas, fontes e limitações. O visor mostra esses dados quando se escolhe um elemento. Cada elemento é um objeto próprio no GLB (`userData.elemento`).

## Elementos e grau de confiança

| Elemento | Estado | Confiança | Medidas e base | Fontes |
|---|---|---|---|---|
| Castelejo | estado-atual | alta | Recinto c. 13 × 25 m, traçado do OSM com a face nascente encostada à crista no MDS. Topo dos muros a 304–305 m, c. 10–14 m acima do chão interior (c. 291 m) e 10–16 m acima do pé exterior (LiDAR 2024). | sipa, pc, cmp, osm, lidar, orto, fotos |
| Porta da Vila (ou do Cruzeiro) | estado-atual | média | Soleira à cota 279,0 m (MDT). Vão de c. 2,6 × 3,8 m (estimativa, fotografia aérea 1). | sipa, cmp, osm, lidar, fotos |
| Porta da Traição (ou dos Campos) | estado-atual | média | Torre c. 4 × 4 m (OSM). Topo a 292,2 m, 8,6 m acima do pé exterior (LiDAR 2024). A face exata de cada abertura do cotovelo está por confirmar. | sipa, cmp, osm, lidar |
| Brecha das Desaparecidas (antiga Porta do Relógio) | estado-atual | alta | Abertura de c. 5,5 m entre o pano sul e a torre quinária (OSM e MDS). Chão à cota 287,2 m. | pc, cmp, visite, osm, lidar, fotos |
| Torre quinária | estado-atual | média | Planta c. 6,5 × 6,5 m (OSM). Topo a 300,7 m, 14,1 m acima do pé exterior (LiDAR 2024). | cmp, wiki, osm, lidar, fotos |
| Torreão do pano poente (norte) | estado-atual | alta | Planta c. 4 × 4 m (OSM). Topo a 287,8 m, 9,5 m acima do pé exterior (LiDAR 2024). | sipa, osm, lidar, fotos |
| Torreão do pano poente (centro) | estado-atual | alta | Planta c. 4 × 6,5 m (OSM). Topo a 291,2 m, 13,6 m acima do pé exterior (LiDAR 2024). | sipa, osm, lidar, fotos |
| Torreão do pano poente (sul) | estado-atual | alta | Planta c. 3,3 × 2,3 m (OSM). Topo a 291,0 m, 12,5 m acima do pé exterior (LiDAR 2024). | sipa, osm, lidar, fotos |
| Cubelo do ângulo noroeste | estado-atual | média | Topo a 287,6 m, 15,2 m acima do pé da escarpa (LiDAR 2024). Raio c. 2,8 m (estimativa a partir da fotografia). | pc, osm, lidar, fotos |
| Pano norte | estado-atual | alta | Topo dos merlões a 291–297 m, subindo a 304 m junto ao castelejo. Altura exterior de 7 a 13 m e só 1 a 3,5 m acima do chão interior (LiDAR 2024). | sipa, osm, lidar, orto, fotos |
| Pano noroeste | estado-atual | alta | Topo a 285–292 m. Altura exterior de 7 a 15 m, sobre a escarpa (LiDAR 2024). | osm, lidar, orto, fotos |
| Pano poente | estado-atual | alta | Topo a 287–293 m. Altura exterior de 7 a 14,5 m (LiDAR 2024), dentro do intervalo publicado de 7 a 19 m. | sipa, cmp, osm, lidar, orto, fotos |
| Pano sul | estado-atual | alta | Topo a 291,5–294,7 m. Altura exterior de 8 a 13 m (LiDAR 2024). | osm, lidar, orto, fotos |
| Pano nascente | estado-atual | alta | Topo a 295–303 m. Altura exterior de 7 a 11 m sobre o rochedo (LiDAR 2024). | osm, lidar, orto, fotos |
| Igreja de São Miguel | estado-atual | média | Planta do OSM (c. 16 × 25 m). Cumeeira a 301,6 m, 10,3 m acima do adro; beirados a 296,9 m (sul) e 298,3 m (norte); vértice do coruchéu a 311,3 m, 20 m acima do adro (LiDAR 2024). Coruchéu, vãos e relógio pelas fotografias. | sipa, wiki, osm, lidar, fotos |
| Casa paroquial e espaço museológico | estado-atual | média | Planta do OSM. Cume a 294,4 m e cornija a c. 291,8 m, c. 6,4 m acima do chão (LiDAR 2024). | sipa, wiki, osm, lidar, fotos |
| Muro de suporte do terreiro da igreja | estado-atual | média | Degrau medido no MDT em c. 38 m de comprimento, até 5,5 m de desnível. Onde o MDT não mostra degrau o muro não é desenhado. | lidar, orto, fotos |
| Terreno | estado-atual | alta | — | lidar, orto |
| Vila de Penela | estado-atual | média | — | osm, lidar |
| Caminhos e ruas | estado-atual | média | — | osm, orto |
| Vegetação | estado-atual | média | — | lidar, fotos |

Fontes: `sipa` SIPA IPA.00001597 · `pc` Património Cultural 69810 · `cmp` CM Penela · `visite` Visite Penela · `wiki` Wikipédia · `osm` OpenStreetMap via Overture 2026-10-21.0 · `lidar` DGT LiDAR 2024 (CC BY 4.0) · `orto` DGT ORTOS 2025 (CC BY 4.0) · `fotos` fotografias aéreas e de pormenor fornecidas.

## O que falta para passar de provisória a validada

- Planta cotada (levantamento DGEMN/IPPAR ou da Câmara) para confirmar a espessura das muralhas e o traçado onde o OSM foi corrigido.
- Posição da cisterna, das seteiras, troneiras e bombardeiras, e das aberturas da Porta da Traição.
- Confirmação no local das identificações de cada torre e da Porta da Vila.
