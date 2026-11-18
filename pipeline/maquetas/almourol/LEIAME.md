# Maqueta 3D do Castelo de Almourol (estado atual)

Castelo templário (1171) na ponta leste da ilha de Almourol, no Tejo. Maqueta à escala 1:1 com a ilha inteira, o rio e as
duas margens numa base redonda (centro x −120, y −12, raio 226 m).

## Como reconstruir

```bash
# 1. LiDAR da DGT -> dados/lidar.npz, dados/superficie.npz, dados/castelo.npz
#    Numa pasta: LO-178278-06-2024_v01.laz e LO-178277-06-2024_v01.laz, com os MDT e MDS de 50 cm das duas folhas.
python3 pipeline/maquetas/almourol/lidar.py <pasta>
# 2. ortofotomapa 2025 da DGT (WMS Ortos2025-RGB, 25 cm) -> dados/orto.jpg
python3 pipeline/maquetas/almourol/ortofoto.py
# 3. estações dos panos e círculos dos cubelos -> dados/muralhas.json
python3 pipeline/maquetas/almourol/tracar.py
# 4. maqueta (bpy) -> public/maquetas/almourol/almourol.glb (Draco), capa.jpg, dados/medidas.json; vistas/ com --vistas
python3 pipeline/maquetas/almourol/construir.py --vistas
```

O JSON do visor (`src/data/maquetas/almourol.json`) é escrito à mão; o `construir.py` lê dele a origem e a versão.

## Origem e referencial

- Origem no centro da torre de menagem: lat 39,461973, lon −8,383981 (PT-TM06 E −21 590,16, N −22 873,01).
- x para este, y para norte, z altitude. Torre de menagem com eixos próprios: u = azimute 60,7° (ENE), v = azimute 330,7°.
- Tejo a 13,23 m (MDT sobre a água, junho de 2024). A água é o MDT à cota do rio sem nada por cima, mais a classe 9.

## O que se mede e o que é estimativa

Medido no LiDAR 2024 (pontos das classes 1–6 a 25 cm):
- **Panos:** estação a cada 0,5 m com face de fora, face de dentro, adarve (p35), merlões (p92) e chão dos dois lados.
  Adarve a 46,4–47,2 m no recinto de cima e 39,4 m nos panos sul e leste; merlões c. 2 m acima do adarve.
- **Dez cubelos** ajustados a círculos (raio 1,8–3,3 m). Oito vêm do traçado automático; os do meio dos panos leste
  (23,75; −8,25) e sul (−5,1; −18,0) foram acrescentados à mão porque o topo do pano sobe 2 m ao longo de 3–4 m e a face
  sai 1–2,4 m. Ficam seis no recinto de baixo e quatro no de cima, como diz o SIPA.
- **Torre:** planta 7,26 × 7,42 m, terraço 59,25 m, merlões 60,95 m (cinco por face), casota branca da escada u 0–3,
  v −0,8–2,5, topo 61,0 m. Porta de entrada (0.3) na face norte-noroeste, s 0,75–1,65, soleira 44,6 m, verga 46,6 m,
  com escada metálica encostada à face: o LiDAR a 0,6–1,4 m da face sobe de 42,0 m (s −3,0) a 44,5 m (patamar de
  s 0,4 a 1,9) e volta ao chão (41,7 m) a seguir; a 0,3 m vê-se a guarda (45,6–46,4 m). Confere com a fotografia do
  Commons «em Agosto 2025» n.º 31 (porta alta retangular, escada escura com patamar e guarda). O terreno por baixo da
  escada fica à cota do chão medida 2,2 m para fora da face.
- **Chão dos recintos:** recinto de cima a 41–42 m. O recinto de baixo (0.2) é feito de plataformas planas às cotas do
  MDS, desenhadas no referencial do pano leste (u ao longo do pano, v para dentro): fosso escavado em três patamares
  (32,9 / 34,5 / 36,8), faixa a 36,6 com guarda de ferro, rampa calcetada de 38,15 a 39,7 encostada ao muro interior
  com parapeito de 0,75 m, e a sul plataformas a 36,4 e 37,35 com um alicerce baixo a x = 3. Os muros de suporte são
  direitos e saem das arestas das plataformas (guarda onde o salto passa de 0,9 m). A disposição vem da planta do
  painel informativo à entrada (Commons, julho de 2025, foto 91), ajustada às cotas do MDS.
- **Escadaria norte:** rampa de 41,0 a 46,45 m em 7,2 m a 2,0–2,5 m do eixo do pano norte (29 degraus).
- Penedos: máximos do relevo local do MDT (mais de 0,6 m) até 30 m dos muros.

Pelas fotografias:
- **Porta principal** a poente, ao fundo do corredor entre os cubelos oeste-sul e oeste, com penedos do lado norte,
  três degraus, arco de volta perfeita e lápide comprida por cima (Commons, julho de 2025, fotos 79–82, GPS da câmara).
  Pelo vão vê-se o recinto de baixo e os merlões do pano sul ao nível dos olhos, por isso a porta abre a sul do muro
  interior. Soleira 37,95 m; arco 1,4 × 2,45 m (estimado).
- Porta do muro interior a nascente da torre, ao cimo da rampa calcetada: três degraus, lápide branca pequena logo
  por cima do arco, penedos à esquerda e o painel informativo ao lado (Commons, agosto de 2025, fotos 21 e 25, e
  Flickr 245694640 / 3333289147). Soleira 40,3 m. A porta a poente da torre que estava na 0.1 saiu: a fotografia da
  porta principal tirada de dentro (julho de 2025, foto 73) mostra ali muro e penedos.
- Porta do cubelo nordeste ao cimo da escadaria (foto do alto da torre virada a nordeste).
- Janelas e frestas da torre e a cruz pátea por cima da janela virada a nascente (Wikipédia e fotos): posições estimadas.

## Contradições

- A Wikipédia portuguesa diz que os merlões uniformes são acrescento romântico do século XIX; a inglesa põe-nos nas
  obras de 1940–50. Fica escrito que foram refeitos no restauro.
- A ficha falava de alvenaria calcária; as fotos e a Wikipédia mostram granito. Corrigido na ficha.
- O SIPA dá o ilhéu com 325 × 85 m; com a água de junho de 2024 o LiDAR dá c. 355 × 113 m (a ponta oeste é baixa e
  muda com o caudal).

## Por confirmar

- Porta da traição a sul (SIPA): não aparece nas fotos reunidas nem no LiDAR.
- Proporção do arco da porta principal (parece mais largo e baixo na foto).
- Escada metálica de 2013–14 dentro da torre (não modelada).
- Passadiços de madeira e tenda do recinto de baixo (temporários, não modelados).

## Visor

Grupos: Torre e Muralhas (enquadramento), Terreno, Vegetacao, Envolvente e Rochedos (fora do enquadramento). Seis
placas: torre, porta principal, muro interior, recinto de cima, recinto de baixo, cubelos e rio Tejo.

## Revisão

Autorrevisão de perito em `/mnt/project-files/revisao-perito/almourol/relatorio.md` (nota 6,4 → 7,1) e comparações
foto/maqueta em `/mnt/project-files/almourol-maqueta/`.

## Fotografias

As oito fotografias de referência enviadas não são do autor do projeto e não entram na galeria. A galeria da ficha fica
com as quatro fotografias do Commons que já lá estavam.

## Panos (0.2)

A face de cada pano é endireitada (Douglas-Peucker com 0,45 m sobre as estações) e a linha dos merlões e o adarve
ficam em patamares nivelados (o LiDAR alterna pontas dos merlões e soleiras das ameias; toma-se o máximo em 2,5 m e
juntam-se troços com menos de 0,55 m de diferença). Na 0.1 seguiam as estações suavizadas e ondulavam.

Na 0.3 os cantos da linha reta passam a ser estações, com a normal em esquadria (bissetriz a dividir pelo cosseno do
meio ângulo), para as duas faces e o parapeito se encontrarem num canto vivo; no muro interior a face é uma paralela
ao eixo (offset em esquadria), porque as normais do LiDAR viravam no canto apertado a sul da torre e a face dobrava
sobre si mesma (o «corpo redondo» que se via junto à porta nascente). Os merlões são postos troço reto a troço reto:
nos cantos de fora encostam ao canto, nos de dentro param antes para não se atravessarem. O muro interior de nascente
sai do cunhal sueste da torre quase no alinhamento da face nascente e o LiDAR perde-o rente à torre: a ponta é
prolongada até entrar 10 cm na torre, sem parapeito nesse troço.
