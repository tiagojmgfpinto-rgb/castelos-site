# Maqueta 3D do Castelo de Penela (estado atual)

Versão 0.1, **provisória**. Gerada por código, à escala 1:1 e georreferenciada (origem nas coordenadas oficiais 40.031162, −8.389862; x para este, y para norte; no visor, −z é o norte).

## Como reconstruir

```sh
# 1. dados abertos (OSM via Overture Maps e Copernicus DEM) -> dados/fontes.json
pip install pyarrow shapely rasterio numpy
python pipeline/maquetas/penela/recolher.py

# 2. maqueta em Blender (módulo bpy) -> public/maquetas/penela/penela.glb (Draco) e capa.jpg
pip install bpy==5.2.3 numpy          # Python 3.13
python pipeline/maquetas/penela/construir.py --vistas
```

O `bpy` do PyPI não traz o Draco, por isso o script comprime o GLB com `npx @gltf-transform/cli draco` (é preciso Node).
`--vistas` grava a capa e as vistas de verificação em `vistas/` (topo, norte, sul, nascente, poente, duas oblíquas como as fotografias aéreas 1 e 2 e o pano poente).

## Onde está a leitura das fontes

Tudo o que é interpretação está em `src/data/maquetas/penela.json`: cotas dos recintos, altura e escarpa de cada pano, estado (estado atual, reconstituição, hipótese), grau de confiança, medidas, fontes e limitações. O visor mostra esses dados quando se escolhe um elemento. Para corrigir uma altura ou uma identificação, muda-se o JSON e volta-se a correr o `construir.py`. Cada elemento é um objeto próprio no GLB (`userData.elemento`).

## Elementos e grau de confiança

| Elemento | Estado | Confiança | Medidas e base | Fontes |
|---|---|---|---|---|
| Castelejo | estado-atual | média | Recinto c. 13 × 25 m (OSM). Rochedo c. 4 m acima do terreiro e muros c. 6 m acima do rochedo: estimativas a partir das fotografias aéreas 1 e 2 e da fotografia do interior. | sipa, pc, cmp, osm, fotos |
| Porta da Vila (ou do Cruzeiro) | estado-atual | média | Vão de c. 2,6 × 3,8 m (estimativa). | sipa, cmp, osm, fotos |
| Porta da Traição (ou dos Campos) | estado-atual | média | Torre c. 4 × 4 m (OSM). A face exata de cada abertura do cotovelo está por confirmar. | sipa, cmp, osm |
| Brecha das Desaparecidas (antiga Porta do Relógio) | estado-atual | alta | Abertura c. 4,5 m (OSM e fotografia de pormenor). | pc, cmp, visite, osm, fotos |
| Torre quinária | estado-atual | média | Planta c. 6,5 × 6,5 m (OSM). Altura exterior c. 14 m (estimativa, fotografia de pormenor). | cmp, wiki, osm, fotos |
| Torreão do pano poente (norte) | estado-atual | alta | Planta c. 4 × 4 m (OSM). Altura c. 2,5 m acima do pano (estimativa). | sipa, osm, fotos |
| Torreão do pano poente (centro) | estado-atual | alta | Planta c. 4 × 6,5 m (OSM). Altura c. 2,5 m acima do pano (estimativa). | sipa, osm, fotos |
| Torreão do pano poente (sul) | estado-atual | alta | Planta c. 3,3 × 2,3 m (OSM). Altura c. 2,5 m acima do pano (estimativa). | sipa, osm, fotos |
| Cubelo do ângulo noroeste | estado-atual | média | Raio c. 2,8 m e altura c. 2 m acima do pano (estimativas a partir da fotografia). | pc, osm, fotos |
| Pano norte | estado-atual | alta | Altura exterior c. 8 m, sobre escarpa rochosa de c. 6 m (estimativas, fotografias aéreas 2 e 3). | sipa, osm, fotos |
| Pano noroeste | estado-atual | alta | Altura exterior c. 8 m, sobre escarpa de arenito de c. 10 m (estimativas, fotografia aérea 2). | osm, fotos |
| Pano poente | estado-atual | alta | Altura exterior c. 12 m (estimativa, fotografia aérea 1). Publicado só o intervalo de 7 a 19 m. | sipa, cmp, osm, fotos |
| Pano sul | estado-atual | alta | Altura exterior c. 9 m (estimativa, fotografia de pormenor). | osm, fotos |
| Pano nascente | estado-atual | alta | Altura exterior c. 8 m, sobre rochedo de c. 6 m (estimativas). | osm, fotos |
| Igreja de São Miguel | estado-atual | média | Planta do OSM (c. 16 × 30 m). Alturas estimadas pelas fotografias: nave c. 11,5 m à cumeeira e torre c. 22 m ao topo do coruchéu. | sipa, wiki, osm, fotos |
| Casa paroquial e espaço museológico | estado-atual | média | Planta do OSM. Altura c. 7 m à cornija (estimativa). | sipa, wiki, osm, fotos |
| Muro de suporte do terreiro da igreja | estado-atual | baixa | Desnível c. 2,5 m (estimativa). | fotos |
| Terreno | estado-atual | baixa | — | mdt, fotos |
| Vila de Penela | estado-atual | baixa | — | osm |
| Caminhos e ruas | estado-atual | média | — | osm |
| Vegetação | estado-atual | baixa | — | fotos |
Fontes: `sipa` SIPA IPA.00001597 · `pc` Património Cultural 69810 · `cmp` CM Penela · `visite` Visite Penela · `wiki` Wikipédia · `osm` OpenStreetMap via Overture 2026-10-21.0 · `mdt` Copernicus GLO-30 · `fotos` fotografias aéreas e de pormenor fornecidas.

## O que falta para passar de provisória a validada

- Planta cotada (levantamento DGEMN/IPPAR ou da Câmara) ou ortofoto da DGT de 25 cm para conferir o traçado.
- MDT LiDAR da DGT (50 cm) para substituir o relevo de 30 m e a escarpa moldada à mão.
- Alturas medidas de cada pano, torreão e da torre sineira.
- Posição da cisterna, das seteiras, troneiras e bombardeiras, e das aberturas da Porta da Traição.
