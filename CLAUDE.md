# castelos.site

Atlas 3D dos castelos e fortalezas de Portugal, com maquetas 3D de cada monumento. Astro + React Three Fiber (`src/`), pipeline em Python (`pipeline/`). Ver `README.md`.

## Língua

Tudo em português de Portugal: código comentado, textos do site, commits, PRs e mensagens ao Tiago.

## Maquetas: regras do Tiago

- **Nunca** começar ou refazer uma maqueta sem ordem explícita do Tiago.
- Numa maqueta publicada, não mudar formas que ele aprovou; realismo (texturas, cor) e correção de bugs das estruturas sim.
- Casas da envolvente em cinzento neutro e volumes simples; base redonda; chão por classes nítidas, nunca ortofoto drapeada.
- Confirmar muralhas, portas, torres e fachadas no Street View e nas ortofotos antes de entregar.
- Lista completa e erros já criticados: `pipeline/sistema/licoes.md`.

## Maquetas: o sistema

Toda a maqueta nova ou refeita segue `pipeline/sistema/LEIAME.md`: coordenador (a sessão do thread), investigador (Sonnet), revisor independente (Sonnet) e modelador (Opus), cada um com o seu encargo em `pipeline/sistema/encargos/`. O Tiago pediu este sistema (25 nov. 2026), por isso nas maquetas o coordenador lança estes agentes sem perguntar.

- Condições escritas e congeladas antes de modelar (`nota.py congelar`); o que não está nelas não se exige depois como grave.
- Volta 1 com ponto de controlo só da forma; revisão 1; volta 2 só com a lista; revisão 2.
- Uma ordem por volta: `python pipeline/sistema/volta.py <castelo> --volta N [--fase forma]`.
- A nota sai do `nota.py calcular`: 8,0 = todas as graves cumpridas, −0,5 por grave falhada, −1 por vista falhada inteira.
- As cores medem-se na vitrina (`pipeline/sistema/vitrina/`), com a luz do visor (`src/data/luz.json`). Se `src/components/maqueta/Luz.tsx` ou a luz do `Maqueta.tsx` mudarem, mudar também `vitrina.html`.
- Pasta de cada castelo: `/mnt/project-files/refs/<castelo>/` (fotos de terceiros não entram no git; no site só fotos com licença livre).

## Publicar

- Com nota ≥ 8 o coordenador publica sem perguntar: PR, merge com o CI verde (autorização do Tiago, 28 out. 2026), e confirma no site publicado que o GLB servido é o local. Com nota < 8 depois da revisão 2, mostra ao Tiago e pergunta.
- Antes do PR: `npm test` e `npm run build`. Se mexer no kit das maquetas: `python pipeline/kit/verificar.py` (nenhuma outra maqueta pode mudar).
