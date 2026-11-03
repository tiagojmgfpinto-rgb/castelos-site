# Maqueta 3D do Castelo de Lanhoso e do Santuário de Nossa Senhora do Pilar (estado atual)

Versão 0.2, **provisória** (0.2: fachada do santuário refeita pela fotografia de frente, portas da porta do castelo fechadas com ferragens, penedo baixo e largo com musgo, chão do monte com cor por vértice, lajes de granito com riscas e passagens suaves, muro do terreiro no bordo nascente). A torre de menagem com os merlões, a porta alta e a escada de pedra; a porta do castelo entre os dois torreões ameados, com a escadaria e o penedo; a cerca com o adarve, as escadas e o terraço; o pátio; o santuário de 1680 com o adro; o terreiro, as lajes de granito do monte e as árvores. Os poucos edifícios à volta são volumes simples num só cinzento. Gerada por código, à escala 1:1 e georreferenciada: origem no centro da torre (41,586850, −8,281361; PT-TM06 −12361,95, 213063,98), x para este e y para norte, cotas = altitudes. Caixa do modelo: x −70..80, y −63..57 (a sul acaba no limite da folha 187514 do LiDAR).

## Como reconstruir

```sh
# 1. edifícios e caminhos do OSM (via Overture Maps) -> dados/fontes.json
pip install pyarrow shapely pyproj
python pipeline/maquetas/lanhoso/recolher.py

# 2. LiDAR da DGT -> dados/lidar.npz, dados/superficie.npz (com a cor RGB), dados/torre.npz
#    Do Centro de Dados da DGT (cdd.dgterritorio.gov.pt), numa pasta: MDT-50cm e MDS-50cm da folha 187514 (2025)
#    e a nuvem de pontos LO-187514-07-2025.
pip install rasterio "laspy[lazrs]" numpy scipy pillow
python pipeline/maquetas/lanhoso/lidar.py PASTA

# 3. ortofotomapa 2021 da DGT (WMS) -> dados/orto.jpg
python pipeline/maquetas/lanhoso/ortofoto.py

# 4. maqueta em Blender (módulo bpy) -> public/maquetas/lanhoso/lanhoso.glb (Draco), capa.jpg e dados/medidas.json;
#    vistas/ com --vistas
pip install bpy==5.2.3 numpy scipy shapely rasterio   # Python 3.13
python pipeline/maquetas/lanhoso/construir.py --vistas

# 5. dados do visor (placas, elementos, limitações) -> src/data/maquetas/lanhoso.json
python pipeline/maquetas/lanhoso/visor.py

# 6. imagens da ficha -> public/imagens/lanhoso/ (fotografias, ortofotomapa e relevo)
python pipeline/maquetas/lanhoso/imagens.py PASTA_DAS_FOTOS [OUTRA_PASTA]
```

Os ficheiros de `dados/` estão no repositório. As plantas (torre, torreões, muro da porta, cerca, terraços, alicerces) e o referencial do santuário estão em `comum.py`. Opções do `construir.py`: `--sem-glb`, `--vistas`, `--capa`, `--so=a,b`, `--amostras=N`, `--sol=AZ`, `--blend`.

## Referenciais

- **Torre e cerca (u, v)**: u para o azimute 73,25° (ENE) e v para 343,25° (NNO), as direções das faces da torre (histograma da orla dos pontos acima de 392,5 m). As plantas de `comum.py` estão em (u, v).
- **Santuário (s, q)**: s ao longo do eixo da nave, para o altar-mor (azimute 352,25°), q para ENE; centro da nave em (5,60, −43,96).

## O que se mede e o que é estimativa

- **Torre** (LiDAR, `dados/torre.npz`): 10,44 × 10,30 m. Chão junto às faces: ENE 382,97, NNO 383,18, OSO 383,95 (plataforma de lajes), SSE 382,90. Adarve (tábuas) a 393,80 m, peitoril das ameias a 394,73 m, corpo dos merlões a 395,79 m e pontas a 396,29 m (percentil 98,5); vértice do telhado a 395,56 m. Oito merlões por face contando os dos cunhais, alguns com seteira em arco (fotografias de drone). Telhado de quatro águas castanho-avermelhado, mastro com a bandeira junto à face SSE.
- **Porta e escada da torre**: porta na face NNO, soleira a 387,73 m (o patamar da escada no LiDAR); a escada sobe encostada à face, de ENE (382,9 m) para o patamar, 19 degraus, com corrimão de ferro (fotografias de nascente, do pátio e da porta vista de dentro). O arco ficou de volta perfeita; a Wikipédia diz quebrado.
- **Porta do castelo**: muro recuado (u −7,3..−5,1) entre os torreões, que têm o topo a 387,58 m e merlões nos lados de fora. Arco quebrado de fora com 1,55 m de vão e arranque a 1,75 m da soleira (383,1 m); por dentro, arco mais largo (2,15 m); portas de madeira verde abertas (fotografias de fora e de dentro); um merlão com seteira sobre o arco. Escadaria de 380,55 m (u −15,5) à soleira: lanço de 5 degraus, patamar e lanço de 8. O penedo, encostado ao torreão sul, tem o alto a c. 385,9 m (LiDAR); a forma é aproximada.
- **Cerca**: plantas traçadas nas células de 25 cm com relevo local > 1,2 m num disco de 3,5 m e conferidas no ortofotomapa. Adarves (moda dos pontos entre os merlões): poente 386,95 m com merlões (peitoril 388,1 m), norte 386,35 m e canto NE com parapeito liso, nascente 386,0 m. Terraço por dentro da muralha norte e banco ao longo da nascente a 384,1 m; muro fino a sueste (topo 386,0 m). Escadas para os adarves das fotografias (posições estimadas).
- **Santuário** (LiDAR): nave de 19,2 × 10,8 m, beirais a c. 386,2 m, cume 390,5 m, frontão 391,7 m, cruz 393,0 m; capela-mor de 9,7 × 7,0 m com cume a 386,6 m; sacristia a poente da capela-mor; adro a 377,95 m. A fachada segue a fotografia de frente. A torre sineira e as capelas de pirâmide ficam fora da folha do LiDAR.
- **Muro do terreiro**: muro baixo no bordo nascente do terreiro, sobre as lajes da encosta (fotografia de drone de nascente, 84ef); traçado na crista estreita do LiDAR de 25 cm; altura (0,85 m) estimada.
- **Fachada do santuário** (fotografia de frente, c. 37 px por metro): porta de 1,75 × 2,9 m com ombreiras, entablamento, frontão interrompido e cartela; janelas de 1,15 m com grade de pedra em quadrícula, moldura, peitoril e cornija (baixo 378,75–380,4 m, cima 382,3–384,0 m); retábulo do nicho à frente da fachada, com pendente, pilastras, volutas e frontãozinho, a entrar no frontão; pilastras dos cunhais com capitel; pináculos de 2,5 m; vértice do frontão a 390,7 m e cruz a 392,9 m; escadaria de 4 degraus com 14,6 m de largura.
- **Penedo da porta** (fotografia de frente da porta): baixo e largo (c. 3,2 × 4,3 m, alto a c. 384,6 m), à frente do torreão sul e encostado à escada; o LiDAR não o separa do torreão, a forma é estimada.
- **Chão**: MDT de 50 cm; junto aos muros e à torre o chão baixa até 1,5 m na faixa onde o MDT sobe por interpolação. O chão do monte é um só material com cor por canto de face (grão da erva na textura): erva verde (relva onde o ortofotomapa é verde), lajes de granito onde o ortofotomapa é claro e pouco saturado em declive acima de 8°, ou onde o LiDAR tem solo nu sem vegetação com mais de 12° (a encosta nascente, que no ortofotomapa está na sombra), terreiro de saibro e lajes onde é claro e plano entre o castelo e o santuário (com o acesso de serviço do OSM), trilhos de terra. As máscaras são limpas (fecho, abertura, manchas pequenas fora) e alisadas c. 0,6 m, para as passagens serem suaves. O granito tem as riscas do ortofotomapa (luminância relativa à média de 4 m) e manchas avermelhadas de líquenes e urze. Pátio relvado e estrada asfaltada à parte.
- **Envolvente**: plantas do OSM e manchas de edifícios do LiDAR (classe 6), sem o monumento e o santuário. Árvores nos máximos locais da vegetação do LiDAR, sem as que caíam sobre os muros, a torre ou o santuário, e com a copa afastada 0,3 m dos muros.

## Fontes de conferência

- Fotografias enviadas pelo Tiago (as pequenas, de páginas da internet, só serviram de referência).
- O geoportal do Município da Póvoa de Lanhoso (`sig.mun-planhoso.pt`) recusou o acesso (403), também no Playwright; usou-se o ortofotomapa de 2021 da DGT.

## Visor

O visor procura o grupo `Muralhas_grupo` para enquadrar e o objeto `Terreno` para a caixa do chão; os elementos têm `userData.elemento` (`torre`, `escada_torre`, `cerca`, `porta`, `santuario`, `terreno`, `envolvente`, `vegetacao`). As âncoras das placas e os pontos de vista são calculados no `construir.py` e gravados em `dados/medidas.json`; o `visor.py` lê-os.

## Vistas de verificação

`vistas/` repete os pontos de vista das fotografias (`d74a_sul`, `84ef_ene`, `84ef_drone`, `3cd5_norte`, `fa66_porta`, `i10_porta_frente`, `5b79_patio`, `8ae8_ne`, `6c4d_adarve`, `00e6_fachada`, `00e6_perto`) e acrescenta `coroamento` e `topo` (para comparar com o ortofotomapa).
