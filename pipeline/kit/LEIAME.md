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

### Padrão das maquetas (12 nov. 2026)

- Pedra, telha e rocha dos monumentos com `textura_foto`: `stone_brick_wall_001` para silhares (torres, portas), `old_stone_wall` para alvenaria (cercas, muralhas), `castle_wall_slates` para xisto, `roof_09` para telha, `rock_01` para granito à vista. Fiadas com o tamanho das fotos do monumento (c. 0,3–0,5 m).
- Casas da envolvente num só cinzento claro, sem textura (`cinza_viz`, 0,74/0,74/0,73), paredes e telhados; os monumentos com nome mantêm cor e detalhe.
- GLB com Draco (`gltf-transform draco`). WebP foi testado em Melgaço e não ganha nada sobre o JPEG (2,49 → 2,49 MB), por isso não se usa.

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
