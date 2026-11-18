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
  v −0,8–2,5, topo 61,0 m.
- **Chão dos recintos:** recinto de cima a 41–42 m; recinto de baixo em socalcos de escavação (38,1 / 36,6 / 34,4 /
  33,0 m), com muros de suporte e guardas de ferro onde o salto passa de 1,2 m.
- **Escadaria norte:** rampa de 41,0 a 46,45 m em 7,2 m a 2,0–2,5 m do eixo do pano norte (29 degraus).
- Penedos: máximos do relevo local do MDT (mais de 0,6 m) até 30 m dos muros.

Pelas fotografias:
- **Porta principal** a poente, ao fundo do corredor entre os cubelos oeste-sul e oeste, com penedos do lado norte,
  três degraus, arco de volta perfeita e lápide comprida por cima (Commons, julho de 2025, fotos 79–82, GPS da câmara).
  Pelo vão vê-se o recinto de baixo e os merlões do pano sul ao nível dos olhos, por isso a porta abre a sul do muro
  interior. Soleira 37,95 m; arco 1,4 × 2,45 m (estimado).
- Porta do muro interior a nascente da torre, com lápide (fotos aéreas e Commons, agosto de 2025, foto 25); porta do
  muro interior a poente, ao nível do recinto de baixo (uma só foto aérea: hipótese).
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
- Interior da porta do muro interior a poente da torre.
- Escada metálica de 2013–14 dentro da torre (não modelada).
- Passadiços de madeira e tenda do recinto de baixo (temporários, não modelados).

## Visor

Grupos: Torre e Muralhas (enquadramento), Terreno, Vegetacao, Envolvente e Rochedos (fora do enquadramento). Seis
placas: torre, porta principal, muro interior, recinto de cima, recinto de baixo e cubelos.

## Revisão

Autorrevisão de perito em `/mnt/project-files/revisao-perito/almourol/relatorio.md` (nota 6,4 → 7,1) e comparações
foto/maqueta em `/mnt/project-files/almourol-maqueta/`.

## Fotografias

As oito fotografias de referência enviadas não são do autor do projeto e não entram na galeria. A galeria da ficha fica
com as quatro fotografias do Commons que já lá estavam.
