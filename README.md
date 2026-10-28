# castelos.site

Atlas 3D dos Castelos e Fortalezas de Portugal: o território em relevo real, com a localização de cada fortificação, voos de câmara até cada monumento e fichas históricas.

## Estrutura

| Pasta | Conteúdo |
| --- | --- |
| `pipeline/terrain/` | Scripts Python que descarregam o Copernicus DEM GLO-30 e geram os mosaicos de relevo (quadtree) e as linhas cartográficas de cada território. |
| `src/content/monumentos/` | Um ficheiro Markdown por monumento. O esquema está em `src/content.config.ts`. |
| `src/components/atlas/` | Atlas 3D em React Three Fiber: terreno com níveis de detalhe, linhas, marcadores, câmara. |
| `src/components/ui/` | Interface sobre o mapa, à maneira do castillos.site: barra de ferramentas, índice com pesquisa e filtros, distritos, fontes, rosa dos ventos e ficha lateral. |
| `src/pages/` | Páginas estáticas Astro: atlas, lista de castelos, ficha de cada monumento, créditos. |

## Desenvolvimento

```bash
npm install
pip install rasterio pyproj shapely pillow numpy

# relevo (só é preciso uma vez; os mosaicos não ficam no git)
python pipeline/terrain/fetch_dem.py --region continente
python pipeline/terrain/fetch_dem.py --region acores
python pipeline/terrain/fetch_dem.py --region madeira
for r in continente acores madeira; do python pipeline/terrain/fetch_bathy.py --region $r --zoom 9; done
python pipeline/terrain/fetch_bathy.py --region continente-moldura --zoom 8   # resto da Península, mais grosseiro
bash pipeline/terrain/naturalearth.sh
python pipeline/terrain/build_tiles.py

npm run dev
npm test
```

A publicação no GitHub Pages é feita pelo workflow `.github/workflows/publicar.yml`, que corre o pipeline de relevo e compila o site.

## Dados dos monumentos

Cada monumento entra com `coordenadas.verificadas: false` até as coordenadas serem confirmadas numa fonte oficial (SIPA / Património Cultural, I.P.). Uma ficha só passa a `publicada` com fontes. Uma maqueta só aparece no atlas com `validada: true` e, se for uma reconstituição, com época e fontes.

## Fontes de dados

- Relevo: Copernicus DEM GLO-30 (programa Copernicus da UE), via AWS Open Data.
- Costa, fronteira, distritos e rios: Natural Earth (provisório até à integração da CAOP da DGT).
