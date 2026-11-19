# Maqueta 3D do Forte de São Sebastião, Angra do Heroísmo (estado atual)

Forte abaluartado (1572, reconstruído em 1698) no extremo nascente da baía de Angra, ilha Terceira. Maqueta à escala 1:1
com o forte, a bateria baixa, a costa e o mar numa base redonda (centro x 0, y −22, raio 112 m).

## Como reconstruir

```bash
# 1. LiDAR dos Açores 2024 -> dados/lidar.npz, dados/superficie.npz, dados/orto.jpg
python3 pipeline/maquetas/sao-sebastiao-angra/lidar.py 2024_TER_ANG_24-4-2-5_CCL_RGBI.laz …_01_MDE.tif
# 2. magistral do forte e da bateria baixa -> dados/planta.json
python3 pipeline/maquetas/sao-sebastiao-angra/tracar.py
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
- **Fachadas**: painéis sobre as faces medidas no gradiente do MDS, com janelas, cunhais e cornija.

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
(Concierge.2C, CC BY-SA 3.0), «Forte de São Sebastião from the dam Angra» (Ymblanter, CC BY-SA 4.0) e «Vista geral do
forte e da marina» (Senhormario, CC BY-SA 4.0).
