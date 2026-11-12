# Maqueta 3D da Praça-forte de Almeida (estado atual)

Versão 0.2, **provisória** (10 de novembro de 2026). A 0.2 corrige o que o autor do projeto pediu e o que se viu no Google Maps e no Street View (ver «Correções da 0.2»). A maqueta é gerada por código, à escala 1:1 e georreferenciada.

- **Origem:** centróide do anel da contraescarpa do fosso principal no OSM (retaining_wall fc68b085, 33,1 ha).
  - WGS84: 40,725642, −6,906288.
  - PT-TM06 (EPSG:3763): 103 646,77, 118 134,25.
- **Eixos:** x para este, y para norte; no visor, −z é o norte. As cotas são altitudes.
- **Área:** base redonda (pedido do autor), um círculo de 466 m de raio centrado na origem que leva a praça, os revelins, o fosso e a esplanada; o raio está limitado pela área do LiDAR.
- **Medições:** relevo, traçado das escarpas, pontes, cotas, alturas e pegadas vêm do LiDAR da DGT de 2024 (voo de maio).

## Como reconstruir

```sh
# 1. muralhas, casas da guarda, edifícios e ruas do OSM (via Overture Maps) -> dados/fontes.json
python pipeline/maquetas/almeida/recolher.py

# 2. LiDAR da DGT, folhas 303418, 303419, 304418 e 304419 (2024).
#    Primeiro, os MDT/MDS que a DGT não deu (MDT de 303418, MDS de 303418 e 303419) a partir das nuvens LO-*.laz:
python pipeline/maquetas/almeida/rasterizar.py PASTA            # --conferir 304419 compara com uma folha da DGT
#    Depois -> dados/lidar.npz, dados/casas.json, dados/orto.jpg
python pipeline/maquetas/almeida/lidar.py PASTA

# 3. traçado das escarpas no MDT -> dados/escarpas.json; pontes (classe 26) -> dados/pontes.json
python pipeline/maquetas/almeida/tracar.py
python pipeline/maquetas/almeida/pontes.py PASTA

# 4. maqueta em Blender (bpy 5.2.3, Python 3.13) -> public/maquetas/almeida/almeida.glb (Draco) e capa.jpg
python pipeline/maquetas/almeida/construir.py --vistas
```

Opções do `construir.py`: as mesmas de Valença.

| Opção | O que faz |
|---|---|
| `--sem-glb` | só monta a cena |
| `--vistas` | grava a capa e as vistas em `vistas/` |
| `--so=a,b` | grava só as vistas indicadas |
| `--amostras=N` | número de amostras do Cycles |

As vistas `p_<porta>`, `m_<monumento>` e `c_<foto>` são as de conferência; as `c_` foram feitas da posição das fotografias de comparação. Cada construção grava em `dados/medidas.json` as medidas e as âncoras das placas.

## Dados do LiDAR

- **Classes:**
  - 2: chão, incluindo as muralhas, que estão no MDT. Por isso o traçado automático das escarpas funciona.
  - 6: edifícios.
  - 26: pontes.
- **Folhas que faltavam:** o MDT de 303418 e os MDS de 303418 e 303419 não vieram da DGT e foram feitos com `rasterizar.py`.
  - Método: média do chão e máximo das classes 1 a 6 por célula, com as falhas interpoladas.
  - Conferência com a folha 304419 da DGT: mediana 0, e 99,9 % das células a menos de 25 cm.
- **Cotas:** as altitudes vão de 710 a 763 m. O `lidar.npz` guarda-as em cm acima de 700 m (BASE = 700).

## O que se mede e o que é estimativa

**Escarpas** (`tracar.py`)
- Método de Valença, aplicado numa faixa de 45 m à volta das muralhas do OSM e de 60 m à volta da contraescarpa.
- Resultado: 64 panos, c. 8,6 km.
- Altura acima do pé: 5,7 m em mediana e até 14,2 m.
- Nos panos com mais de 7 m o cordão fica a meia altura (0,46 da altura), como nas fotografias das portas.

**Portas**
- Quatro portas das duas portas duplas (São Francisco ou da Cruz, a sul-sudeste; Santo António, a poente) e a Porta Nova.
- A posição vem do OSM e do relevo; a soleira, do tabuleiro das pontes (classe 26) e do MDT.
- O vão é aberto no pano (`vaos()`, como em Monção), sem cortar a escarpa. Na Porta Nova abre-se também a escarpa de dentro, onde o túnel sai.
- Os frontispícios seguem as fotografias:
  - colunas de tambores almofadados, frontão curvo, armas reais e guarita na porta de dentro de São Francisco;
  - calçada com obeliscos na de fora;
  - frontispício simples com guarita na de fora de Santo António.

**Casas da guarda**
- Pegada do OSM e altura do MDS (percentil 90).
- Cobertura em duas águas baixas, com a cumeeira no eixo da passagem (fotografia 19 e foto de São Francisco), mais a chaminé de Santo António.
- Cordão de meia-cana à altura do alto do arco.
- Panos de remendo ligam a ponta da escarpa ao canto da casa.

**Túneis**
- Retos e com o comprimento lido no LiDAR. O terreno por cima fica acima do extradorso.
- O degrau do terreno logo atrás da frontaria sai da malha. Sem isso, um campo de alturas tapa o vão: o teste de raios batia a 3,5 m.
- **Hipótese:** o túnel de Santo António de dentro acaba num fundo escuro a 20 m. O traçado real, que deve dobrar para o pátio a c. (−150, −65), não se vê.

**Pontes**
- Comprimento, largura, tabuleiro e guardas medidos na classe 26: São Francisco 37,4 m; Santo António 44,7 m e 18,6 m.
- Os arcos (vãos de c. 6,5 m) são estimativa.

**Guaritas:** 25, postas automaticamente nos ângulos salientes. O número é estimado.

**Monumentos**

A pegada vem do OSM e as cotas do MDS. A orientação das fachadas foi conferida pelo GPS das fotografias do Commons.

| Monumento | Fachada e fotografia | Medidas (MDS) |
|---|---|---|
| Igreja Matriz | fachada a poente; câmara de D. Perez a poente | nave 24 × 9,5 m, cumeeira 749,9; torre de 5 m a SW, remate 758,7 |
| Misericórdia | fachada a nordeste, a meio do lado da Praça da República; câmara de D. Perez 7 m à frente | empena perpendicular à rua, cumeeira c. 752,4 |
| Torre do Relógio | remate de balaustrada, pináculos e lanterna pequena com cata-vento (Street View) | 5,5 m de lado; 762,7 → c. 786 m |
| Quartel das Esquadras | arcaria nos DOIS lados compridos (Street View do Terreiro do Poço do Rancho e da Rua da Muralha); telha velha escura (ortofotomapa); chaminés na água noroeste; escadaria em V no topo nordeste (fotografia de D. Perez) | 105,5 × 14,6 m; cumeeira 755,8 |
| Castelo | — | bases de três torres redondas (N, E, W) lidas no relevo e na ortoimagem; alturas estimadas |

**Ficam no casario cinzento:** o Picadeiro, o Paiol e a Casa dos Governadores. As Casamatas são subterrâneas e só se veem no relevo do pátio.

## Correções da 0.2 (10 nov., conferidas no Google Maps e no Street View)

- **Base:** redonda em vez de recortada pelo traçado da praça.
- **Quartel das Esquadras:** a 0.1 tinha a arcaria só no lado sudeste e a telha laranja. O Street View mostra arcaria dos dois lados, com uma porta verde ao fundo de cada arco e uma faixa de duas fiadas de granito por cima, e o ortofotomapa mostra a telha escura. Faltavam as chaminés e a escadaria em V do topo nordeste.
- **Torre do Relógio:** o remate é uma lanterna pequena com cata-vento sobre a balaustrada, não uma cúpula grande.
- **Portas de Santo António:** a de dentro não tem frontão nem guarita: pilastras almofadadas, painel do brasão entre duas seteiras, friso de tríglifos e cornija, e uma janela gradeada à direita. A de fora não tem empena: frontispício retangular com cornija, lápide, brasão e guarita por cima. Saiu um cordão da casa da guarda que ficava solto sobre o fosso.
- **Muralhas:** a alvenaria escura de pedra miúda passou a cantaria de granito claro, com silhares maiores e manchas de líquen, como nas fotografias.
- **Igreja da Misericórdia:** a porta é de arco pleno com colunelos e frontão curvo inteiro; as janelas são de sacada, com varandim; o brasão real está na empena e não há óculo.
- **Janelas:** as janelas fora do eixo de cada fachada (Quartel, Misericórdia) eram desenhadas todas no centro; agora cada uma fica no seu lugar. Peniche, Valença e Monção usam a mesma função e têm o mesmo erro.
- **Muralhas sobrepostas:** o MDT de 50 cm parte muitas escarpas altas em dois degraus (a face de cima, um patamar de 4 a 6 m, que é o MDT interpolado ao pé do pano, e a de baixo), e o `tracar.py` dava dois panos paralelos a c. 5 m. No ortofotomapa o pano é um só. O `construir.py` (`fundir_degraus`) tira o troço de baixo quando corre paralelo, mais baixo e do lado do fosso de outro em mais de 60 % do comprimento (7 troços, c. 1 km, sobretudo a nordeste e a leste); o pano de cima desce até ao fosso e o patamar passa a ser chão do fosso. Os dentes do traçado (duas viragens contrárias num troço de menos de 6 m) também saem.
- **Conferido sem mudanças:** Portas de São Francisco (as duas), Porta Nova, pontes e Igreja Matriz.
- **Árvores:** o pé passa a vir da malha do terreno já simplificada (com a base maior, o MDT ficava acima dela longe da praça). Os 500 troncos assentam no chão.

## Contradições

- **Portas de Santo António:** a Wikipédia diz «rasgadas a Norte» e o SIPA (ficha) diz noroeste. O LiDAR e o OSM põem-nas a poente do centro. A maqueta segue o LiDAR.
- **Porta da Cruz:** o SIPA chama assim a porta magistral do revelim da Cruz; o Município e o OSM chamam-lhe Portas de São Francisco.
- **Área:** a Wikipédia dá 650 000 m², o que não bate com os 2,5 km de perímetro. A contraescarpa do OSM fecha 33 ha.

## Verificações (fase 5)

**Auditoria** (`auditoria-maquetas/ferramentas/auditar.py`)
- GLB: 2,6 MB com Draco; 473 mil triângulos; 55 primitivas (0.2, base redonda).
- Sem z-fighting. A base tem 7,5 % das faces com a normal trocada segundo o auditor, por ter o rebordo e o fundo a partilhar o anel; não se vê no visor.
- Árvores: 500 troncos medidos um a um, todos assentes no terreno; o auditor agrupa copas vizinhas e conta 31 grupos «a flutuar», que não são troncos soltos.
- 209 peças «a flutuar»: são beirais e telhados separados das paredes (Valença tinha 130).
- 461 peças «enterradas»: são as peças dos túneis debaixo do terrapleno, de propósito.

**Teste de raios** (três alturas e três posições por porta)
- As cinco passagens estão abertas.
- Santo António de dentro termina no fundo escuro a 20 m (hipótese).

**Visor real:** cores conferidas no Playwright.

## Fontes

- DGT, LiDAR 2024 (CC BY 4.0).
- OpenStreetMap (ODbL), via Overture Maps.
- Wikipédia, «Praça-forte de Almeida».
- Município de Almeida (circuito).
- Fotografias da Wikimedia Commons: David Perez (CC BY 3.0), Hovallef (CC BY-SA 4.0), Dinah Raphael (CC BY-SA 3.0), Xemenendura (CC BY-SA 4.0).
- Capturas de referência enviadas pelo autor do projeto: só para conferência, não vão para a galeria.
- O SIPA estava em baixo (503) durante o trabalho.

## Realismo (12 nov. 2026)

- Pedra e telha fotográficas (`kit/fotos.py`, fotos CC0 da Poly Haven): silharia das escarpas e cantaria das portas em `stone_brick_wall_001` (fiadas de c. 0,5 m nas escarpas), recolorida para o granito claro bege-acinzentado do Street View; telhas `roof_09` com os três tons (viva, velha e do Quartel). Créditos em `dados/medidas.json` (`texturas_foto`). As casas já eram cinzentas.
