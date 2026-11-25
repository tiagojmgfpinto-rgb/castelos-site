# Lições acumuladas

O que aprendemos com as maquetas anteriores e com as críticas do Tiago. Todos os papéis leem isto antes de começar; o revisor transforma em condições graves as que se aplicam ao castelo. O coordenador acrescenta aqui, no fim de cada castelo, o que o `registo.md` mostrou (com data e castelo).

## Regras do Tiago

- **Só com ordem.** Nunca começar uma maqueta sem ordem explícita do Tiago (9 nov. 2026).
- **Não estragar o que está feito.** Realismo sim, mas não mudar a forma do que ele já aprovou (merlões, muros, casas, telhados); só corrigir bugs das estruturas (dentes de serra, picos, peças a flutuar, buracos). Mudar formas só com licença dele e troço a troço (Penela, 25 nov. 2026: merlões "rústicos", panos deformados e beirados foram lidos como estragar e revertidos).
- **Fasquia.** A mesma qualidade do castillos.site, "maqueta a maqueta, metro a metro"; alvo: Guadalupe (25 nov. 2026).
- **Envolvente a cinzento.** Casas da envolvente em cinzento neutro e volumes simples, mesmo copiando o modelo de texturas do castillos, que as texturiza (2 e 25 nov. 2026). Torres de granito com cor de granito.
- **Base redonda**, não recortada pelo traçado (Almeida, 10 nov. 2026).
- **Detalhe dos monumentos principais** ao nível do claustro do Paço dos Duques (Guimarães, PR #12): arcarias, colunas, capitéis, alpendres.
- **Blender e SketchUp** sempre que possível; maquetas com scripts Python no Blender, escala 1:1, a partir de plantas, ortofotos e fotografias (23–24 nov. 2026). Ver a recomendação sobre three.js no LEIAME.
- **Fotos do Tiago** vão para a galeria da ficha, sem as que têm marca de água alheia.

## Erros já criticados (procurar sempre)

- Muralha que vira escarpa rochosa ou rampa (Peniche, Monção).
- Porta tapada, vão preto opaco, ou a muralha "caída" por cima da porta.
- Torre diferente das fotos (forma, número de pisos, remate).
- Fachada orientada ao contrário (Monção 0.1: a Misericórdia).
- Árvores do LiDAR a atravessar o monumento (Braga).
- Chão com a ortofoto drapeada como textura (rejeitado a 24 nov. 2026): o chão faz-se por classes nítidas (asfalto, calçada, lajeado, escadas, saibro, terra, rocha, relva, erva, mato, água), com as calçadas e lajes só por correções traçadas à mão e confirmadas no Street View.
- Muros que não assentam no LiDAR; telhados fora da grelha do edifício.
- Texturas de pedra que repetem em faixas (3,2 m por repetição a 324 px/m em Penela 0.8); o castillos usa 6–8 m por repetição.

## Método que funcionou

- Confirmar muralhas, portas, torres e fachadas no **Street View** e nas ortofotos antes de entregar (2 nov. 2026).
- Coordenadas do edifício, não da vila; as coordenadas SIPA às vezes estão deslocadas.
- LiDAR 2024/2025 da DGT: classe 6 (edifícios) dá as muralhas em muitas folhas, mas não em todas (Castro Laboreiro, Monção: classes 1/2); a ortofoto não é verdadeira-orto (os telhados aparecem deslocados uns metros).
- Cor da pedra medida em fotos a pleno sol; texturas fotográficas Poly Haven (CC0) com a cor média do monumento (`kit/fotos.py`), créditos em `medidas.json`.
- Um material por elemento arquitetónico (Guadalupe tem 73): cunhais em silharia, vãos recuados escuros, capa dos merlões, frisos claros, cumeeiras, arcos em sombra. É isso que dá o "detalhe" mais do que triângulos soltos.
- Draco + WebP nos GLB (`kit/webp.py`).

## Lições por castelo

(O coordenador acrescenta aqui depois de cada castelo feito com o sistema.)
