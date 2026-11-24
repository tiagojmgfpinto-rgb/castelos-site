# Maqueta 3D do Forte de São Sebastião, Angra do Heroísmo (estado atual)

Forte abaluartado (1572, reconstruído em 1698) no extremo nascente da baía de Angra, ilha Terceira. Maqueta à escala 1:1
com o forte, a bateria baixa, a costa e o mar numa base redonda (centro x 0, y −22, raio 112 m).

## Como reconstruir

```bash
# 1. LiDAR dos Açores 2024 -> dados/lidar.npz, dados/superficie.npz, dados/orto.jpg
python3 pipeline/maquetas/sao-sebastiao-angra/lidar.py 2024_TER_ANG_24-4-2-5_CCL_RGBI.laz …_01_MDE.tif
# 2. magistral do forte e da bateria baixa -> dados/planta.json
python3 pipeline/maquetas/sao-sebastiao-angra/tracar.py
# 2b. classes do chão (traços à mão em dados/chao_tracos.json + LiDAR + ortofoto) -> dados/chao_classes.png, chao_tinta.png
#     (construir.py faz a textura com textura_chao.py: materiais nítidos, nenhum píxel de fotografia)
python3 pipeline/maquetas/sao-sebastiao-angra/chao.py
# 3. maqueta (bpy) -> public/maquetas/sao-sebastiao-angra/sao-sebastiao-angra.glb (Draco + WebP), capa.jpg,
#    dados/medidas.json; vistas/ com --vistas; --diag imprime os perfis medidos
python3 pipeline/maquetas/sao-sebastiao-angra/construir.py --vistas
```

O JSON do visor (`src/data/maquetas/sao-sebastiao-angra.json`) é escrito à mão.

## Origem e referencial

- Origem no recinto: lat 38,651886, lon −27,211995 (PTRA08 / UTM 26N, E 481 553,4, N 4 278 168,5).
- x para este, y para norte, z altitude. Convergência de meridianos 0,13°, ignorada.
- Mar a 0,63 m: mediana do MDT nas células sem pontos (classe 0) junto à costa.
- A cor dos pontos (RGB) serve de ortoimagem; está desviada uns 0,5 m da geometria em algumas fiadas.

## Método

- **Magistral** (`tracar.py`): em cada lado aproximado procura-se, perfil a perfil, a queda mais exterior do MDS
  (mais de 2 m em 2 m, perto do topo) e ajusta-se uma reta robusta. Erro mediano 0,10 m no forte e 0,13 m na bateria.
- **Muralhas** (`construir.py`, classe `Linha`): estação a cada 0,25 m ao longo da magistral, com vetor de esquadria
  em cada vértice. Em cada estação mede-se no MDS: pé da escarpa (fora), cordão, topo do parapeito, espessura do
  parapeito, cota do adarve e vãos das canhoneiras. Escarpa em talude de 0,18, cordão de cantaria, parapeito com
  merlões e canhoneiras às cotas medidas.
- **Dentro**: o chão, os terraplenos, as rampas e os volumes dos edifícios são o MDS de 25 cm (mediana 3×3), com o
  material de cada célula escolhido pela zona (Casa do Governador, Pousada, corpo da guarda), pelo declive e pela cor.
- **Guaritas**: detetadas nos vértices salientes (ângulo < 125°) onde o MDS sobe 2,4–6 m acima do adarve numa mancha
  de 6–140 células; modeladas com corpo redondo, mísula, frestas, porta, cúpula e bola. A do saliente sudoeste é
  maior (mancha de c. 4 × 4 m a 35 m): é a sineira da fotografia de Diego Delso, feita à parte (`espadana()`).
- **Portão de Armas**: arco de 3 m com moldura e fecho, painel das armas reais com frontão, ponte de dois arcos sobre o
  fosso (cotas do MDS, comprimento limitado a 10,5 m) e arco interior.
- **Edifícios (0.2)**: a Pousada e a casa da guarda saem do MDS de 25 cm (mediana 5×5): células mais de 1,8 m
  acima do MDT, partidas em patamares de 0,6 m, cada patamar um polígono (rasterio `shapes`, simplificado a 0,3 m)
  extrudido do solo à cota mediana, com janelas a passo de 2,6 m nas faces livres. A superfície fica com o MDT por baixo.
- **Casa do Governador (0.2)**: referencial u (azimute 37°) / v a partir de (18,6; −6,9). Corpo A u 0–23,6, v 0–6,1;
  corpo B u 0–13, v 6,1–11,85; telhados de quatro águas, beirado 32,1 m, cume 33,45 m; pátio u 13–17 (chão a 26,8 m)
  com muro e portão em arco; ala baixa C u 17–23,4, v 6,1–12,4. Revisto na 0.5 pela grelha de 1 m do MDS em u/v: C é
  terraço plano a 30,0 m (±0,1 m), entre B e C há uma faixa coberta plana a 29,4 m (u 13–17, v 6,1–8,7) e o cume de A
  corre até ao topo nordeste, onde encosta ao prédio de 34–35 m (empena, sem tacaniça). Vãos da fachada
  noroeste de B pela fotografia «7181537601» de Vítor Oliveira: três janelas no andar com aventais recortados, duas
  portas por baixo das das pontas; topo sudoeste com três janelas por piso.
- **Ala norte da Pousada (0.2)**: bloco de cobertura plana a 34,7 m entre a casa da guarda e a Casa (planta PN em
  `construir.py`). É o prédio amarelo do óculo da fotografia «F s seb angra 2» (2009), que a 0.1 tomava pela Casa do
  Governador: fachada sul de 13,4 m com dois pares de janelas por piso, óculo ao centro e porta por baixo dele.
- **Casa da guarda (0.2)**: planta GU pela ortofoto (telhado de telha de c. 14 × 7 m), uma água de 31,65 a 33,75 m;
  o arco interior do Portão de Armas fica na fachada amarela, onde o eixo da passagem a corta (`frente_guarda()`).
- **Chão (0.2, revisto)**: classes da ortofoto sem manchas de menos de 3 m²; a terra batida dos caminhos e dos
  terraplenos tem a cor medida (c. 0,46/0,42/0,37 sRGB ao sol) e cada vértice leva o tom da ortofoto sem as sombras.
- **Basalto (0.2)**: patine por vértice com manchas largas, escorrências que descem do parapeito com comprimentos
  variados, pé húmido com musgo e líquen claro (fotografias de Diego Delso e de C. Cruz).
- **Picos do recinto (0.2)**: a araucária e os chapéus de sol que ficaram no MDS saem com uma abertura de 2,25 m onde
  o resultado é o chão do recinto (abaixo de 28,5 m).
- **Sineira (0.3)**: bloco maciço do MDS de 5,5 × 5 m no ângulo SO, cobertura plana a 35,1 m (c. 2,3 m acima do
  parapeito), faces de fora no prumo do baluarte, vãos em arco no alto (dois a poente e a sul, um a norte e a nascente),
  cata-vento e chaminé (fotografia de Diego Delso, 2020). A 0.2 tinha-a como terraço ameado com uma sineira pequena.
- **Guaritas (0.3)**: assentes no vértice do saliente, 0,35 m para fora pela bissetriz, no alto do parapeito vizinho;
  o parapeito baixa ao piso dentro do corpo e os merlões a menos de 3,2 m não passam do parapeito de 3–7 m.
- **Estrada e telhados (0.3)**: a relva exige saturação (o asfalto cinzento-esverdeado deixou de passar por relva);
  asfalto e coberturas planas com a cor da ortofoto em linear; capeamento branco nas platibandas; telha mais castanha.
- **Chão (0.5, refeito)**: a primeira 0.5 drapeava a ortofoto como textura do chão e ficava uma fotografia esborratada
  (rejeitada pelo Tiago). Agora a ortofoto só diz o que é cada sítio. `chao.py` compensa as sombras do voo (sol a 271°,
  44°) e tira os carros. Depois faz uma segmentação semeada (watershed sobre o gradiente Lab da ortofoto e do SUP),
  para cada fronteira encostar à aresta real. As sementes são:
  - traços à mão por zona (`dados/chao_tracos.json`), mais faixas de largura constante para passeios e caminhos;
  - classes do LiDAR: 11 = estrada (só o meio; dentro do forte é o caminho de terra batida), 6 com mais de 3 m =
    edifício, 4 e 5 = mato;
  - relva muito verde e mar.
  As fronteiras são alisadas a 0,75 m. As marcas brancas (passadeiras, linhas) são retângulos ajustados às manchas
  finas dentro do asfalto, e as manchas redondas saem. `textura_chao.py` pinta cada classe com o seu material, repetido
  à escala real: asfalto, asfalto gasto do parque, passeio, lajeado, relva com o tom seco/verde da ortofoto a 3 m, mato,
  terra batida castanho-arroxeada, betão do estaleiro, rocha, tinta. A textura tem 4096 px (5,7 cm), com manchas largas
  que não se repetem. Os lancis estão em volume (`construir_lancis`, c. 464 m, 15 × 13 cm, alisados a 1,2 m) onde o
  asfalto encosta a passeio, relva ou terra. A rocha continua a ser a escarpa a pique, as faces a mais de c. 63° e o que
  fica à cota do mar.
- **Muros de fora (0.5)**: muro de suporte da estrada de poente, muro do parque de poente, muro alto do estaleiro e
  muros em degraus do parque baixo a nordeste, traçados à mão na ortofoto (com um painel da altura sobre o chão local) e
  assentes no LiDAR: o traço encosta à faixa branca do capeamento (±6 m) e depois à crista (maior salto entre o ponto e
  o chão a 0,75 m de cada lado); topo e pé medidos no MDS de 25 cm; o terreno do lado baixo fica à cota do pé até 0,9 m
  da face (as células do terreno no corredor saem e o muro leva o seu chão). Nenhum tem parapeito acima do lado alto. Faces em basalto: hipótese (não se veem de cima).
- **Chão (0.2)**: classes pela cor dos pontos (relva por verde relativo, também na sombra; tijoleira; calçada; saibro),
  mediana de 5×5 células; no recinto (abaixo de 28 m) o saibro passa a bagacina vermelha.

## Medido e estimado

Medido no LiDAR 2024: planta, alturas das muralhas, do adarve e do parapeito, guaritas (posição e topo), soleira do
portão (27,5 m), fundo do fosso (25,2 m), chão do recinto (26,6 m), beirados da Casa do Governador (32,2 m) e da
Pousada (30,2–32,8 m), nível do mar, árvores.

Estimado pelas fotografias do Commons e pelo Street View: número e ritmo das janelas, forma das guaritas, porta em
arco da Casa do Governador, desenho do Portão de Armas e das armas reais, arcos da ponte. Não modelado: a passagem
abobadada por baixo do terrapleno, o interior da Pousada, mobiliário e iluminação.

## Fotografias de referência

Commons: «Fuerte de San Sebastián … 2020-07-25, DD 20» (Diego Delso, CC BY-SA 4.0), «F s seb p pipas 5» e «6» e
«F s seb angra 2» (Carlos Luis M C da Cruz, domínio público), «Monte Brasil desde a Fortaleza de São Sebastião»
(Concierge.2C, CC BY-SA 3.0), «Forte de São Sebastião from the dam Angra» (Ymblanter, CC BY-SA 4.0) e «Pousada do Forte … (7181537601)» (Vítor Oliveira, CC BY-SA 2.0), «Vista geral do
forte e da marina» (Senhormario, CC BY-SA 4.0).
