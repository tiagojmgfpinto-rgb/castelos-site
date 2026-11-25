# Kit comum das maquetas

Funções que os `construir.py` das maquetas tinham copiadas, iguais, de uma para a outra. Desde a Etapa 3 da auditoria de novembro de 2026 estão aqui uma só vez e cada `construir.py` importa-as:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pipeline/, para o kit comum
from kit.geometria import unit, area, densificar  # noqa: E402
```

- `geometria.py`: Geometria plana e amostragem: vetores, polígonos, linhas, arcos e interpolação nas grelhas do LiDAR. (`unit`, `area`, `ccw`, `orientar`, `densificar`, `dentro`, `distancia`, `simplificar`, `desvio`, `mediana_movel`, `normais`, `angulos`, `partir`, `amostra`, `quad`, `arco_pts`, `arco`, `arco_quebrado`, `ogiva`, `redondo`, `reto`, `_retangulo`, `_sd`, `_ang`, `_anc`, `_arred`)
- `texturas.py`: Texturas procedimentais (ruído, fbm, cantaria, telha, relva, rocha, terra) e passagem de arrays para imagens do Blender. (`ruido`, `fbm`, `textura_cantaria`, `textura_cubos`, `textura_grao`, `textura_relva`, `textura_rocha`, `textura_telha`, `textura_terra`, `imagem`)
- `cena.py`: Ajudas de cena do Blender: UV em coordenadas do mundo, cena vazia e modelos de árvore. (`uv_mundo`, `limpar`, `arvore_modelo`)
- `materiais.py`: Materiais do Blender (Principled BSDF com texturas, normais e cor por vértice) e o registo MATS onde ficam. (`malha_rapida`, `material`)
- `arquitetura.py`: Peças de arquitetura: referencial de fachada (Quadro), vãos, arcos, aduelas, telhados, guaritas, merlões, escadas e portas. (`Quadro`, `_vao_desloc`, `aduelas`, `altura_telhado`, `barra`, `contorno_vao`, `corrimao`, `cupula`, `faixa_inclinada`, `guardas_corredor`, `guarita`, `merlao`, `oitavado`, `painel`, `parede`, `passagem`, `piramide`, `porta_revelim`, `porta_tunel`, `telhado_4aguas`, `telhado_generico`, `tunel`, `vao`)

- `fotos.py` (novembro de 2026): texturas fotográficas. `textura_foto(id, metros=..., cor=...)` devolve o mesmo par (cor, relevo) que `textura_cantaria`, a partir de fotos CC0 da Poly Haven guardadas em `fotos/` (`obter_fotos.py` descarrega-as; autores, licença e tamanho real em `fotos/fontes.json`). A foto dá o desenho real das pedras e juntas; a cor média é a do monumento, medida nas fotografias dele. `metros` tem de ser o que a imagem cobre de facto na maqueta (UV do `uv_mundo` × `KHR_texture_transform`): com `uv=2.0` e `escala=0.5` são 4 m. `fontes_fotos()` dá os créditos das fotos usadas, para gravar em `medidas.json` (`texturas_foto`).

- `chao.py`, `chao_blender.py`, `quadricula.py` e `planta.py` (novembro de 2026, método "metro a metro"): o chão por classes. Ver a secção abaixo.

### Chão por classes, "metro a metro" (25 nov. 2026)

O chão deixa de ser uma malha pintada ponto a ponto com a cor da ortofoto (manchas, sombras pintadas, estradas que viravam rocha). O Tiago pediu, a 24 nov. 2026, que se veja "metro a metro" o que é muro, estrada, relva e telhado, e que as maquetas se façam com scripts Python no Blender, à escala 1:1, a partir de plantas, ortofotos e fotografias. A ortofoto e as fotografias dizem **o que** é cada sítio; nunca entram como textura.

1. **Classes** (`chao.py` de cada maqueta → `dados/chao.json`): polígonos em metros da cena por classe (`asfalto`, `calcada`, `lajeado`, `escadas`, `saibro`, `terra`, `rocha`, `relva`, `erva`, `mato`, `agua`). Uma proposta automática (cor da ortofoto + classes do LiDAR + declive, com filtro de maioria e passada a polígonos alisados por `kit.chao.raster_para_poligonos`) só decide o que é natural (erva, relva, mato, rocha, saibro); asfalto, calçada e lajes **só por correções traçadas à mão** (lista `CORRECOES`, cada uma com a nota do que se viu e onde). `kit.chao.resolver` torna os polígonos disjuntos pela ordem de `PRIORIDADE`, `kit.chao.preencher` tapa as frestas.
2. **Ver quadrícula a quadrícula**: `python pipeline/kit/quadricula.py MAQUETA --classes --planta PLANTA.jpg:MEIO` grava folhas de 40 m com a grelha da cena (ortofoto | contornos das classes | planta da maqueta). A planta faz-se com `python pipeline/kit/planta.py FICHEIRO.blend cx cy lado px saida.jpg` (Cycles, câmara ORTHO) a partir do `.blend` que o `construir.py --blend` grava. Atenção: a ortofoto não é verdadeira-orto, os telhados aparecem deslocados uns metros na direção oposta ao avião.
3. **Malhas** (`kit.chao.malhas(classes, xs, Z, raio, recorte)`): uma malha por classe sobre a grelha Z do terreno. Células inteiras ficam dois triângulos; as que uma fronteira corta são recortadas e trianguladas (Delaunay restrito), por isso a fronteira é exata. O Z interpola-se bilinearmente na grelha, que é linear nas arestas das células: não há fendas entre classes. `kit.chao.aplanar` alisa o Z debaixo das classes pavimentadas (uma calçada não ondula com o ruído do MDT). `recorte` tira o que tem objeto próprio (falésias, chão de uma porta).
4. **Objetos** (`kit.chao_blender.objetos(pecas, pai, paleta)`): `Chao_<classe>` com material nítido e periódico (sem costuras): cubos de calçada, lajes, grão de asfalto, saibro, terra, relva, erva e mato, multiplicados por uma variação larga e suave (±5 %) por vértice. A `paleta` de cada maqueta dá a cor média medida nas fotografias do sítio. 1024 px só para cubos e lajes; o resto 512 px.

Primeira maqueta feita assim: Penela 0.8 (ver `pipeline/maquetas/penela/chao.py`). A seguir, Guimarães 0.10 (`pipeline/maquetas/guimaraes/chao.py`), que juntou ao kit:

- `chao.proposta(pasta, ...)` e `chao.aplicar(classes, CORRECOES, meio)`: a proposta e a aplicação das correções saíram dos `chao.py` (Penela fica igual byte a byte). Opções da proposta: `claro` (limiar do saibro), `casa_folga` (metros à volta dos edifícios onde o claro não conta), `sombra` (as sombras ficam com a classe do vizinho ao sol), `copa` (`"mato"` no monte; `None` num parque de relva aparada, onde o chão debaixo das árvores é o do vizinho) e `maioria_raio` (2 para não apagar carreiros de 2–3 m).
- O polígono de uma correção pode ser uma função das classes: `chao.pela_classe(zona, "erva", "saibro")` dá a parte não verde de uma zona (a vila fora do parque passa a asfalto, o adro a calçada, com a fronteira da cor); `chao.pela_cor(pasta, zona, ...)` dá os píxeis claros e sem verde de um corredor à volta do traçado do OSM (os carreiros com a forma exata, mesmo com o OSM 3 m fora).
- A semântica (cubos, lajes, saibro, asfalto) confirma-se no Street View, ponto a ponto, e fica na nota da correção.

Valença 0.7 (`pipeline/maquetas/valenca/chao.py`) trouxe as maquetas grandes e retangulares, que tinham a ortofoto drapeada como textura:

- `chao.grelha(lid)` dá os centros (xs, ys) das células de um `lidar.npz` quadrado (`meio`) ou retangular (`meiox`, `meioy`); `proposta`, `pela_cor` e `aplicar` servem os dois. Nas grelhas retangulares a `proposta` devolve como domínio o retângulo da grelha, e `aplicar` aceita um polígono no lugar do raio do disco.
- `chao.malhas_adaptativas(classes, xs, ys, Z, dominio)`: para dezenas de hectares, uma só triangulação de Delaunay restrita (`mathutils.geometry.delaunay_2d_cdt`). Os pontos do relevo vêm de uma árvore quaternária (`chao.pontos_quadtree`: blocos de até 8 m enquanto a interpolação dos cantos não se afastar mais de 15 cm do MDT, medido no Z alisado para o ruído não partir os blocos planos), e as fronteiras das classes, do contorno e dos buracos (passagens das portas) são arestas obrigatórias. Cada triângulo fica com a classe do seu centro. Sem fendas e com o número de triângulos contido.
- `chao.alinhar_osm(linha, chao.mascara_asfalto(pasta))`: em Valença o traçado das estradas do OSM chega a estar 10–20 m ao lado da faixa da ortofoto, e não sempre para o mesmo lado. Cada troço de 40 m procura o deslocamento com mais píxeis de asfalto ao sol (cinzentos, sem verde, não escuros: as sombras das muralhas também são cinzentas) e os deslocamentos alisam-se ao longo da linha. `pela_cor` ganhou `lum_max` e `abrir`.

### Padrão das maquetas (12 nov. 2026)

- Pedra, telha e rocha dos monumentos com `textura_foto`: `stone_brick_wall_001` para silhares (torres, portas), `old_stone_wall` para alvenaria (cercas, muralhas), `castle_wall_slates` para xisto, `roof_09` para telha, `rock_01` para granito à vista. Fiadas com o tamanho das fotos do monumento (c. 0,3–0,5 m).
- Casas da envolvente num só cinzento claro, sem textura (`cinza_viz`, 0,74/0,74/0,73), paredes e telhados; os monumentos com nome mantêm cor e detalhe.
- GLB com Draco (`gltf-transform draco`) e, a seguir, texturas em WebP com `pipeline/kit/webp.py` (qualidade 80, extensão `EXT_texture_webp`). O `webp.py` só reescreve as imagens e copia a geometria Draco tal como está; uma imagem só passa a WebP se ficar pelo menos 5 % mais pequena. Em novembro de 2026 tirou 12 % ao total das 13 maquetas (40,3 → 35,5 MB), até 26 % em Braga; a comparação com a mesma câmara deu diferença média abaixo de 0,5 % por píxel. (O teste antigo em Melgaço com o `gltf-transform webp` não ganhava nada.)

## Regras

- **O kit não muda resultados.** Cada função saiu sem alterações do `construir.py` de onde veio. Só saiu de cada maqueta a variante idêntica à do kit, e só quando as funções do kit que ela chama também eram idênticas nessa maqueta. As variantes próprias (por exemplo, o `uv_mundo` com outra escala, a `Malha` que usa o referencial `W` da torre) continuam no `construir.py` da maqueta.
- **O registo de materiais é o do kit.** `kit.materiais.MATS` é o dicionário que `material()` preenche; as maquetas que usam `material()` importam o `MATS` em vez de criarem o seu.
- **Mudar uma função do kit muda todas as maquetas que a usam.** Para mudar só uma, copie a função para o `construir.py` dessa maqueta e mude-a lá (a definição local sobrepõe-se à importada se vier depois do `import`).
- **Antes de entregar uma mudança no kit, verifique:**

```sh
pip install bpy==5.2.3 numpy scipy shapely rasterio pyproj matplotlib pillow   # Python 3.13
python pipeline/kit/verificar.py
```

O `verificar.py` reconstrói as maquetas e compara o GLB antes do Draco (`saida/<id>-bruto.glb`) e o `dados/medidas.json` com `impressoes.json`. Uma diferença é uma mudança no resultado: se for de propósito, grava-se a referência nova com `--gravar`. Os GLB finais em `public/` ficam de fora da comparação porque o Draco do glTF Transform nem sempre sai igual. As impressões dependem das versões em `impressoes.json` (`ambiente`); noutro ambiente, grave primeiro uma referência com o código atual e só depois mexa.

O `verificar.py` reescreve `public/maquetas/<id>/<id>.glb` e `dados/medidas.json`; se não quer publicar novas versões, reponha-os com `git checkout public/maquetas pipeline/maquetas/*/dados/medidas.json`.

## Próximos passos

- Juntar as variantes que só diferem num parâmetro (as 3 `uv_mundo`, os 4 `material`, as `Malha`), com o parâmetro explícito, verificando com o `verificar.py` que nenhuma maqueta muda.
- Passar a ler e a exportar (`construir_terreno`, árvores, exportação GLB e Draco, vistas) por funções do kit, que hoje dependem de globais de cada script.
