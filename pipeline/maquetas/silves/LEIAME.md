# Maqueta do Castelo e da Muralha de Silves (estado atual, 0.1 provisória)

Pedido do Tiago a 18 nov. 2026, pelo prompt-modelo v4 ("vamos fazer a maqueta do Castelo e Muralha de Silves… ao mais
ínfimo detalhe"). Âmbito: a alcáçova ao máximo detalhe, os troços da cerca da vila que sobrevivem com a Porta da
Cidade, a vila em cinzento e base redonda.

## Como se reconstrói

```
python pipeline/maquetas/silves/lidar.py PASTA      # PASTA com as 4 folhas MDT, MDS e LAZ da DGT
python pipeline/maquetas/silves/tracar.py           # dados/muralhas.json
python pipeline/maquetas/silves/recolher.py         # dados/fontes.json (OSM via Overture)
python pipeline/maquetas/silves/construir.py --vistas
python pipeline/maquetas/silves/visor_elementos.py  # elementos do JSON do visor
```

`dados/castelo.npz` (pontos do LiDAR à volta da alcáçova, 68 MB) não vai para o repositório: refaz-se com `lidar.py`.

## Referencial

- Origem: centro da alcáçova, PT-TM06 E −27050, N −274940 (lat 37,191034, lon −8,437765). Não é a torre de menagem,
  que fica no canto norte (x 20, y 67).
- x para este, y para norte, z = altitude (m). No visor, −z é o norte.
- Base redonda de 245 m de raio centrada em (−110, −100): cobre a alcáçova e toda a cerca que sobrevive.

## Fontes

- DGT, LiDAR 2024, folhas 172025, 172026, 173025 e 173026 (o castelo fica no canto comum das quatro). A nuvem tem cor
  RGB: a ortofoto de conferência (`dados/orto_lidar.jpg`, 25 cm) vem do ponto mais alto de cada célula. O WMS da DGT
  não responde a partir do contentor (`ortofoto.py` fica como alternativa).
- OSM via Overture 2026-10-21.0: troços «city_wall» da cerca, torres da cerca, «Arcos da Porta de Loulé», nomes das
  torres da alcáçova (Menagem, Segredo, Mulheres, Centro, duas Albarrãs) e plantas dos edifícios.
- Wikimedia Commons com GPS (`/mnt/project-files/silves-maqueta/fotos-commons/`, `posicoes.txt`): 004 (adarve e
  merlões), 020 e 030 (albarrãs), 023 e 102 (porta e estátua, do largo a sudoeste), 101 (dentro do corredor da porta),
  081–083 (torres da cerca), 086–088 (Porta da Cidade, a poente). As fotos de Vitor Oliveira têm marca de água: só
  referência.
- Wikipédia: onze torres (duas albarrãs), ronda de c. 388 m, cisterna da Moura 20 × 16 × 7 m com quatro naves sobre
  doze colunas, cerca de taipa de 2,8 m forrada de grés, Porta da Cidade do século XIV.
- SIPA (n.º 1288): não respondeu nem pelo browser nem pelo contentor. Por consultar.

## Medidas (LiDAR)

- Panos (estação a cada 0,5 m): adarve 53,0–58,9 m, merlões c. 2 m acima do adarve, espessura 2,4–3,0 m.
- Torres (retângulo robusto das células altas, percentis 1,5–98,5): menagem 14,3 × 9,0 m, merlões 66,5 m; Segredo
  10,4 × 6,5; Mulheres 8,4 × 9,0; Centro 10,9 × 7,7; albarrã sueste 7,2 × 14,1 (torre + braço); albarrã nordeste
  6,7 × 6,5; torre da porta 5,6 × 6,1, merlões 59,5 m. O resto em `dados/medidas.json`.
- Casa da guarda: x −26,6..−11,4, y −71..−51,6, terraço 55,5–56, parapeito 57,5.
- Cisterna: extradorso das abóbadas a 52,4–53,6 m, abaixo do chão à volta (54–55 m).
- Porta da Cidade: torreão c. 20 × 11,6 m, terraço 32,5 m (parapeito 33,3), largo a poente c. 16 m.

## Decisões e hipóteses

- **Porta da alcáçova.** A foto 023 (do largo a sudoeste) mostra o arco à esquerda da torre da porta, recuado, com
  a estátua diante da torre; a foto 101 foi tirada dentro do corredor (x −20,5, y −62,3) a olhar para o arco. Daí: o
  arco está na face poente da casa da guarda e o corredor segue para nascente. A viragem a norte e a rampa até ao
  recinto (47,4 → 50,6 m) são hipótese (porta em cotovelo, como é típico das portas almóadas).
- **Albarrãs.** Os braços têm arco por baixo (fotos 020, 030 e a de Marty B no Commons). Largura do braço (3,0–3,2 m)
  e altura dos arcos estimadas.
- **Merlões.** Retangulares, c. 1,0 m com ameias de 0,5 m (fotos 004, 023, 076). Passo constante; não contados um a
  um. Cada merlão tem uma seteira estreita nas duas faces (076), como vão escuro de 0,08 × 0,5 m.
- **Cerca.** Só os troços que o OSM regista. O topo é o LiDAR (mediana em 7 m, em patamares, nunca menos de 1,5 m
  acima do chão de dentro): onde há casas encostadas pode ser a altura dos telhados. Sem merlões (as fotos dos troços
  e das torres da vila não os mostram). Torres da cerca com embasamento de grés e taipa à vista (081–083).
- **Porta da Cidade.** Arco alto e estreito na face poente, por baixo da janela norte (2,6 m de vão, c. 8 m ao fecho;
  foto 086), duas janelas de sacada com moldura branca e guarda de ferro, sem merlões. A passagem em cotovelo para
  norte é hipótese.
- **Escada do adarve.** A da foto 004 está no pano norte-oeste (o GPS da foto cai no pano nordeste, onde o LiDAR é
  plano): 22 degraus de 53,4 a 57,8 m com patamar a 55,6 m, faixa de 1,9 m junto ao muro (LiDAR).
- **Contradição.** A Wikipédia fala em onze torres; o LiDAR dá dez torres retangulares mais a casa da guarda, que
  conta como a décima primeira.

## Versão 0.2 (23 nov. 2026)

- **Escavações do Palácio das Varandas.** Os muros saem do LiDAR: relevo local (superfície menos a abertura de
  17 × 17 células) entre 0,35 e 2,2 m, só a crista, só onde a densidade é alta (zona escavada a sudeste do recinto),
  sem relva nem copas. Os polígonos são esquadriados pelo retângulo mínimo. Os passadiços (cinzento-azulado no
  ortofoto) têm estrado, prumos e guarda. 35 muros em c. 184 m². As plantas dos compartimentos são aproximadas.
- **Cisterna da Moura.** Abóbadas e bordo em vermelho, como nas fotos 053–055. O bordo segue o terreno aresta a aresta
  (murete a c. 54 m).
- **Chão do recinto.** Material lido no ortofoto em janelas de 5 × 5: saibro claro, lajeado de grés, relva e terra.
  Palmeiras onde o LiDAR dá copas pequenas e altas dentro da alcáçova.
- **Torre de menagem.** Porta em arco quebrado na face virada ao recinto (soleira 55,5 m, medida no LiDAR). As pontas
  dos panos prolongam-se pelo eixo até entrarem na torre, para não ficar fresta (vista `f046_menagem`).
- **Porta da alcáçova.** Aduelas de grés no arco exterior e um segundo arco, mais estreito, recuado (foto 101).
- **Porta da Cidade.** Telhado de quatro águas (beirado 31,9 m, cume 33,9 m) sobre platibanda, terraço a norte a
  26,9 m com guarda e escada íngreme encostada com parapeito amarelo (37 degraus desde 19,4 m; foto 087 e LiDAR).
  O telhado e a escada são lidos no LiDAR, mas o desenho das águas é hipótese.

## Por fazer / por confirmar

- Cisterna dos Cães, Porta da Traição (norte), outras escadas
  dos adarves, arco branco reconstruído, casa de chá e o jardim de traçado islâmico.
- Confirmar no SIPA as datas e o traçado da porta em cotovelo; contar os merlões face a face.
