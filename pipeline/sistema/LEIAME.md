# Sistema das maquetas

Como se faz (ou se refaz) uma maqueta do castelos.site. Copia o processo que a equipa do castillos.site nos explicou em novembro de 2026, adaptado ao nosso repositório: uma equipa de agentes Claude com papéis separados e o Tiago a decidir que castelo se faz e a pôr a fasquia.

A fasquia é uma maqueta fiel, não uma foto: que se reconheça e acerte em forma, proporção, cor, remates e envolvente. O castelo pesa mais do que as rochas e as árvores. Alvo de qualidade escolhido pelo Tiago: o Real Monasterio de Guadalupe no castillos.site (131 mil triângulos na obra, um material por elemento arquitetónico: cunhais, vãos recuados, capas dos merlões, frisos, cumeeiras).

## Os papéis

Ninguém faz o trabalho de outro: quem constrói a maqueta não se dá a nota, e quem a revê não lhe toca.

| Papel | Modelo | Faz | Não faz | Encargo |
| --- | --- | --- | --- | --- |
| Coordenador | Opus (a sessão do thread) | Recebe a ordem do Tiago, lança os outros com o encargo, aprova cada passo olhando para as láminas ampliadas, congela as condições, publica e leva o registo | Não modela, salvo arranjos pequenos no fim | [encargos/coordenador.md](encargos/coordenador.md) |
| Investigador | Sonnet, só em castelos novos | Fotos de todas as faces em sete fontes, vídeo de drone, coordenadas do edifício, medidas, plantas, história, terreno | Não modela nem revê | [encargos/investigador.md](encargos/investigador.md) |
| Revisor | Sonnet, independente | Antes de modelar escreve as condições; depois revê contra elas e dá a nota | Não toca na maqueta | [encargos/revisor.md](encargos/revisor.md) |
| Modelador | Opus | Escreve o `construir.py`, constrói, corre a volta, faz o autoexame | Não se dá nota nem publica | [encargos/modelador.md](encargos/modelador.md) |

Os agentes não falam entre si: tudo passa pelo coordenador. Cada um trabalha à parte e, no fim de cada passo, devolve um aviso curto com o que fez. O que importa não vai nas mensagens mas nos ficheiros da pasta do castelo, que todos leem. O coordenador lança cada agente com a ferramenta Agent (`model: "sonnet"` ou `"opus"`) e o prompt «Lê `pipeline/sistema/encargos/<papel>.md` e faz o passo X para o castelo Y»; para lhe devolver trabalho sem explicar tudo de novo, continua o mesmo agente (SendMessage) em vez de lançar outro.

## A pasta do castelo

`REFS/<castelo>/`, com `REFS=/mnt/project-files/refs` (pasta partilhada do projeto, que sobrevive aos threads; as fotos de terceiros não vão para o git):

```
refs/<castelo>/
  investigacao.md          investigador (modelos/investigacao.md): fontes, tabela de fotos por face, medidas, história
  fotos/                   fotos de referência (só referência; no site só fotos com licença livre)
  drone/                   fotogramas do vídeo de drone
  terreno/                 recortes do MDT/MDS, ortofoto com grelha métrica, OSM (ou os dados/ da maqueta)
  condicoes.json           revisor (modelos/condicoes.json), aprovado e congelado pelo coordenador
  condicoes.congeladas     impressão das condições congeladas (nota.py congelar)
  volta1/forma/            volta.py --fase forma: só a forma, para o ponto de controlo
  volta1/completa/         volta.py: vitrina, láminas, cores, técnico, diferenças, resumo.md
  volta1/autoexame.md      modelador (modelos/autoexame.md)
  volta1/revisao.json      revisor (modelos/revisao.json), com a nota calculada pelo nota.py
  volta1/revisao.md        revisor: a revisão em texto, com as provas
  volta2/...               idem
  registo.md               coordenador (modelos/registo.md): tempos, custos, o que baixou a nota, lições
```

## O processo de um castelo

Sempre por esta ordem. As paragens a meio existem para não perder horas a polir uma coisa mal começada.

1. **Ordem do Tiago.** O Tiago diz que castelo se faz, novo ou para refazer. Nada começa por iniciativa própria (regra de 9 nov. 2026).
2. **Investigação** (investigador, só castelos novos; num castelo a refazer, o coordenador reúne o que já existe em `pipeline/maquetas/<m>/` e o investigador só preenche faltas). Fotos de todas as faces em sete fontes, com a tabela do que se procurou e que face cobre cada foto; um vídeo de drone de onde se tiram fotogramas; história, medidas e plantas. Se falta uma foto vital, pede-se ao Tiago antes de modelar, mas só depois de mostrar que se procurou nas sete fontes.
3. **Terreno e medidas** (investigador). Com as coordenadas do edifício (não da vila): MDT e MDS da DGT, ortofoto com grelha métrica, edifícios do OSM, tudo em metros à volta do castelo; alturas conferidas no LiDAR; a cor da pedra medida em fotos a pleno sol.
4. **Condições** (revisor; aprova e congela o coordenador). De 8 a 12 vistas de controlo, cada uma com a foto com que se compara e a câmara; cada material com a cor medida em duas fotos, o tamanho da peça e a argamassa; planta e alturas; condições graves (G), que impedem a aprovação, e leves (L). O que não estiver nas condições não se pode exigir depois como grave. `python pipeline/sistema/nota.py congelar refs/<c>/condicoes.json`.
5. **Volta 1 com ponto de controlo** (modelador). Primeiro só a forma das peças principais: `python pipeline/sistema/volta.py <c> --volta 1 --fase forma`. O modelador para e devolve as láminas ampliadas. O coordenador passa cada grave de forma contra a sua lámina e dá luz verde ou devolve uma lista. Só depois vêm a cor, as texturas, o chão e as comprovações de perto, e a volta completa: `volta.py <c> --volta 1`. A volta acaba com o autoexame: cada vista, cada material e cada condição com a captura e a medida.
6. **Revisão 1** (revisor). Vê a maqueta na vitrina, de longe e de perto, e preenche `revisao.json` e `revisao.md`; a nota sai do `nota.py calcular`. Tudo o que é grande vai nesta lista: não se admitem graves novas na revisão seguinte.
7. **Volta 2** (modelador). Só a lista da revisão 1, também com paragem a meio. Qualquer mudança fora da lista explica-se no autoexame; o mapa de diferenças (antes/depois) marca a vermelho o que mudou, para nada passar sem querer.
8. **Revisão 2 e fecho** (revisor e coordenador). Confere a lista e procura regressões. Se falta uma coisa pequena, arranja-se sem abrir uma terceira volta e o coordenador confirma nas láminas.
9. **Alta e publicação** (coordenador). Com nota 8 ou mais publica-se sem perguntar (merge automático, autorizado pelo Tiago a 28 out. 2026): maqueta e ficha dadas de alta, `npm test` e `npm run build`, PR, merge com o CI verde, e confirmação no site publicado de que o GLB servido é o mesmo que o local (tamanho e SHA-1). Uma maqueta refeita entra em «Adicionados recentemente». Com nota abaixo de 8 depois da revisão 2, o coordenador mostra ao Tiago as láminas e a lista do que falta e pergunta.
10. **Melhorar o processo** (coordenador), antes do castelo seguinte: tempo de cada fase, custo de cada agente, o que baixou cada nota e o que se podia ter visto antes, em `registo.md`. O que se aprendeu passa a [licoes.md](licoes.md), aos encargos, ao kit e ao CLAUDE.md.

## A nota

Calcula-a o `nota.py`, nunca a olho:

- 8,0 = todas as condições graves cumpridas.
- Cada grave que falha tira 0,5. Uma vista de controlo que falha inteira tira 1.
- Cada regressão (uma vista pior do que na maqueta publicada antes) e as comprovações técnicas reprovadas contam como uma grave falhada.
- As leves nunca baixam do 8; com as graves todas cumpridas sobem a nota meio ponto por cada quarto das leves cumpridas, até 10. Não há décimas.

## As ferramentas

| Ficheiro | Para quê |
| --- | --- |
| `volta.py` | Uma ordem por volta: constrói, vitrina, comprovações técnicas, antes/depois, láminas, cores, resumo e esqueleto do autoexame |
| `vitrina/vitrina.html`, `vitrina/vitrina.mjs` | Vitrina de testes: a mesma luz, sombras, oclusão, tone mapping e fundo do visor do site (lê `src/data/luz.json`). As cores medem-se aqui, nunca com outra luz. Sem servidor nem portas, por isso podem correr várias ao mesmo tempo |
| `tecnico.py` | Peças a flutuar ou enterradas, árvores metidas em muros, faces coplanares que piscam, normais invertidas, nada fora da peana; triângulos e materiais por categoria |
| `laminas.py` | Lámina foto / maqueta / mistura 50 % / contorno da maqueta sobre a foto; recortes de perto; cor mediana e ΔE2000 por material |
| `diferencas.py` | Antes/depois com o que mudou a vermelho e a percentagem por vista |
| `nota.py` | Validação e congelamento das condições; cálculo da nota (`--teste` corre os testes da regra) |

Dependências: `pip install -r pipeline/sistema/requirements.txt` e `npm ci` (Playwright e three.js do repositório). O `construir.py` precisa do `bpy` (ver `pipeline/kit/LEIAME.md`); sem ele no contentor, constrói-se no Blender do computador do Tiago e corre-se `volta.py --sem-construir`. Blender por turnos: no máximo quatro ao mesmo tempo no mesmo computador.

Uma volta da Penela publicada (260 mil triângulos, 8 vistas de controlo + 11 gerais, sem construir) demora cerca de 2,5 minutos no contentor.

## Blender/Python ou three.js?

O castillos passou do Blender/Python para three.js. Recomendação para nós, por agora: **continuar com Blender/Python para construir e fazer tudo o que é medir e comparar na vitrina three.js**, que é o que este sistema já faz.

- A razão principal da troca deles (as cores e a luz do Blender não serem as do site) resolve-se com a vitrina: as láminas, as cores e a nota saem sempre da mesma luz que o visitante vê.
- Temos 15 maquetas, o kit (`pipeline/kit/`), o chão por classes e o `verificar.py` em Python/bpy, e o Tiago pediu Blender e SketchUp (23 e 24 nov. 2026). Reescrever tudo em three.js custava semanas e arriscava estragar o que está feito.
- A diferença de qualidade para o castillos não vem da ferramenta, vem do processo (condições escritas antes, láminas foto contra maqueta, revisor independente, um material por elemento).
- A pensar outra vez depois dos dois primeiros castelos feitos com este sistema: se o `construir.py` for o gargalo (minutos por iteração contra segundos no three.js), passar só as peças de arquitetura para uma biblioteca JavaScript e manter o terreno e o chão em Python.

Não se muda de ferramenta sem o Tiago dizer que sim.

## Como pedir um castelo

Ao coordenador, numa frase: «Faz a maqueta de Óbidos com o sistema» ou «Refaz Penela com o sistema». O coordenador lança o investigador, mostra as condições ao Tiago se ele as quiser ver, para no ponto de controlo da forma e só publica com nota 8 ou mais.
