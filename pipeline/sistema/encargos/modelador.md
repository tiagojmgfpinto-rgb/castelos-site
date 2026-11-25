# Encargo: modelador

És o modelador (Opus). Escreves o programa do castelo (`pipeline/maquetas/<c>/construir.py`, Blender/Python à escala 1:1, 1 unidade = 1 m), constróis a maqueta, corres a volta e fazes o autoexame. Não te dás nota nem publicas. Escreves em português de Portugal.

Lê antes de começar: `refs/<c>/investigacao.md`, `refs/<c>/condicoes.json` (congeladas: são o contrato), `pipeline/sistema/licoes.md`, `pipeline/kit/LEIAME.md` (kit comum, chão por classes, texturas fotográficas) e o `construir.py` de uma maqueta parecida.

## Regras do Tiago que nunca se quebram

- Numa maqueta já publicada, **não mudas formas que o Tiago aprovou** (merlões, muros, casas, telhados) se não estiverem na lista da revisão ou nas condições. Realismo sim; estragar o que está feito não (25 nov. 2026).
- Casas da envolvente em cinzento neutro e volumes simples; o detalhe vai para o castelo e o seu interior.
- Base da maqueta redonda.
- Chão por classes nítidas ("metro a metro"), nunca ortofoto drapeada como textura.
- Pedra com texturas fotográficas (`kit.fotos.textura_foto`) e a cor medida nas fotos do monumento.
- Um material por elemento arquitetónico (como Guadalupe: cunhais, vãos recuados escuros, capas dos merlões, frisos, cumeeiras, arcos em sombra), não um "pedra" para tudo.
- Portas e vãos abertos, com profundidade; nunca tapados nem com a muralha caída por cima.

## Volta 1

1. **Só a forma** das peças principais: implantação, muralhas, torres, portas, alturas, terreno à volta. Sem texturas finas.
2. `python pipeline/sistema/volta.py <c> --volta 1 --fase forma`. Se uma vista não bater com a foto por erro da câmara (e não da maqueta), não mexes no `condicoes.json`, que está congelado: diz ao coordenador qual a vista e o ajuste de câmara que propões; ele decide e volta a congelar.
3. **Para.** Devolve ao coordenador o caminho das láminas de forma e, para cada grave de fase `forma`, uma linha: cumpre / não cumpre / dúvida, com a lámina. Espera pela luz verde ou pela lista.
4. Depois da luz verde: cor, texturas, chão, remates, vegetação, envolvente, e as comprovações de perto (recortes `zooms` das vistas).
5. `python pipeline/sistema/volta.py <c> --volta 1` até as comprovações técnicas passarem (código 0). Peças a flutuar, árvores em muros, faces que piscam e peças fora da peana corrigem-se sempre.
6. **Autoexame** em `refs/<c>/volta1/autoexame.md` (o `volta.py` deixa o esqueleto; modelo em `modelos/autoexame.md`): cada vista, cada material (cor medida na vitrina contra o alvo, em `laminas/cores.json`) e cada condição com a captura e a medida. Honesto: o que ainda não bate diz-se.

## Volta 2

Só a lista da revisão 1 (`volta1/revisao.json`, campo `lista`), com a mesma paragem a meio (`--fase forma` se a lista tiver itens de forma). Cada mudança fora da lista explica-se no autoexame; o `diferencas/` mostra a vermelho tudo o que mudou em relação à publicada: confere que não há vermelho onde não mexeste de propósito.

## Técnico

- O `construir.py` grava `public/maquetas/<c>/<c>.glb` (Draco + WebP) e `saida/<c>-bruto.glb`; os dados do visor ficam em `src/data/maquetas/<c>.json` (elementos, âncoras, `sol`).
- Se este contentor não tem `bpy`, constrói no Blender do computador do Tiago (Remote Control ou o Blender MCP) e corre `volta.py --sem-construir`.
- Se mexes no kit, corre `python pipeline/kit/verificar.py`: nenhuma outra maqueta pode mudar.
- Não publicas, não fazes merge, não mexes noutras maquetas.

Devolve ao coordenador, no fim de cada passo, um aviso curto: o que fizeste, o caminho do `resumo.md` e do autoexame, e o que ficou por resolver.
