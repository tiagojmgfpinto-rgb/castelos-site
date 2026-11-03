# Maqueta 3D do Castelo de Melgaço e da muralha da vila (estado atual)

Versão 0.1, **provisória**. A torre de menagem, a cerca oval com as escadas, as portas da traição, a torre pentagonal, a cisterna e a barbacã, e a muralha da vila desde o cunhal noroeste até à Porta de Baixo, com a Porta da Rua do Carvalho. As casas à volta são volumes simples num só cinzento. Gerada por código, à escala 1:1 e georreferenciada: origem no centro da torre (42,114413, −8,259790), x para este e y para norte em PT-TM06, cotas = altitudes. Caixa do modelo: x −112..46, y −40..46.

## Como reconstruir

```sh
# 1. plantas das casas, ruas, cisterna e portas do OSM (via Overture Maps) -> dados/fontes.json
pip install pyarrow shapely pyproj
python pipeline/maquetas/melgaco/recolher.py

# 2. LiDAR da DGT -> dados/lidar.npz, dados/superficie.npz, dados/torre.npz
#    Do Centro de Dados da DGT (cdd.dgterritorio.gov.pt), numa pasta: MDT-50cm e MDS-50cm da folha 189572 (2025)
#    e a nuvem de pontos LO-189572-07-2025.
pip install rasterio "laspy[lazrs]" numpy scipy pillow
python pipeline/maquetas/melgaco/lidar.py PASTA

# 3. ortofotomapa 2021 da DGT (WMS) -> dados/orto.jpg
python pipeline/maquetas/melgaco/ortofoto.py

# 4. estações da cerca e da muralha da vila -> dados/muralhas.json
python pipeline/maquetas/melgaco/tracar.py

# 5. maqueta em Blender (módulo bpy) -> public/maquetas/melgaco/melgaco.glb (Draco), capa.jpg e dados/medidas.json;
#    vistas/ com --vistas
pip install bpy==5.2.3 numpy scipy shapely rasterio   # Python 3.13
python pipeline/maquetas/melgaco/construir.py --vistas

# 6. dados do visor (placas, elementos, limitações) -> src/data/maquetas/melgaco.json
python pipeline/maquetas/melgaco/visor.py

# 7. imagens da ficha -> public/imagens/melgaco/ (fotografias, ortofotomapa e relevo)
python pipeline/maquetas/melgaco/imagens.py PASTA_DAS_FOTOS
```

Os ficheiros de `dados/` estão no repositório. Opções do `construir.py`: `--sem-glb`, `--vistas`, `--capa`, `--so=a,b`, `--amostras=N`, `--sol=AZ`, `--blend`.

## O que se mede e o que é estimativa

- **Torre** (LiDAR): rodada 26,7° (eixo u para azimute 63,3°), corpo de 8,8 m e parapeito de 9,6 m; soco a 191,4 m, mísulas a 205,85 m, parapeito a 207,0 m, soleira das ameias a 208,55 m, merlões a 209,35 m com pontas a 210,25 m; telhado de quatro águas por dentro (207,7–208,9 m, também no ortofotomapa). Cachorrada de oito mísulas de três fiadas por face e uma em cada cunhal, com os matacães abertos entre elas; quatro merlões piramidais por face mais os dos cunhais (fotografia de baixo).
- **Porta e escada da torre** (LiDAR): porta na face NNO, soleira a 195,75 m, 1,0 × 2,2 m em arco; escada metálica ao longo da face NNO, de u 5,35 a 0,25, com patamar até u −1,4. A fotografia tirada do pátio a norte confirma que a escada sobe da esquerda para a direita até à porta. As seteiras (0,10 m de largura) são estimativas de posição e altura.
- **Cerca** (LiDAR, `tracar.py`): estação a cada grau à volta do pátio, com face de dentro, face de fora, adarve, parapeito e chão dos dois lados; topo plano em cada grau. Espessura limitada a 2,1–3,1 m onde a torre pentagonal, a plataforma noroeste e a barbacã se colam ao muro. Parapeito só a norte. Adarve a 196,0–196,6 m de noroeste a nascente e a c. 192 m a sul e poente.
- **Escadas em V** (LiDAR): faixa de 1 m encostada à face de dentro; a nordeste do azimute 39° ao 71° com o pé a 53,5°, a norte do 337° ao 358° com o pé a 346,5°. O número de degraus é contado para c. 0,25 a 0,3 m de espelho (`dados/medidas.json`). As portas da traição ao pé de cada V (arco de 1,2 × 2,3 m) vêm das fotografias: a de nordeste é certa, a de norte é provável.
- **Torre pentagonal, plataforma noroeste e cisterna**: plantas do LiDAR e do OSM; torre pentagonal com topo a 196,3 m e parapeitos a 197,3 m; plataforma a 195,5 m; cisterna a 191,8 m com cume a 192,2 m.
- **Entrada e barbacã**: passagem aberta na cerca a sul (LiDAR e fotografia do alto da torre); barbacã em U com topo a 190,85 m e pátio a 187,7 m. As duas arcadas do muro sul (1,3 × 2,05 m) e a sua profundidade são estimativas pela fotografia.
- **Muralha da vila** (LiDAR): estação a cada metro ao longo de um eixo desenhado sobre o relevo; uma fita contínua com o topo de 194,5 m junto ao castelo a 184,4 m junto à Porta de Baixo, sem ameias.
- **Porta de Baixo**: posição e topo (184,4 m, chão de fora a 177,4 m) do LiDAR; o resto pela fotografia de frente, à escala da altura do LiDAR: vão de 2,3 × 3,7 m em arco de volta perfeita, arquivolta exterior de 2,9 × 4,0 m recuada 0,3 m, impostas salientes, cinco mísulas de matacão e passagem coberta de 4,5 m. O chão em frente foi aplanado (o MDT tinha pontos do muro classificados como solo). As fotografias sugerem que a muralha a norte da porta é mais baixa do que o bloco da porta; o LiDAR dá c. 184,5 m aos dois e ficou o LiDAR.
- **Porta da Rua do Carvalho**: só vem do OSM; o arco (2,6 × 3,4 m) é estimativa.
- **Pedra**: texturas de cantaria geradas por código e calibradas pelas fotografias: granito cinzento-dourado uniforme na torre e na Porta de Baixo, alvenaria em fiadas, mais cinzenta e com líquenes, na cerca e na muralha.
- **Chão**: relvado e afloramentos de granito no pátio pelo relevo do MDT; ruas do OSM em calçada; verdes do ortofotomapa.
- **Envolvente**: plantas do OSM e, onde faltam, manchas de edifícios do LiDAR (classe 6); cornija e cume do MDS; telhados por recuo da planta, ou planos quando a planta é irregular. Árvores nas posições do LiDAR, sem as que caíam sobre o monumento.

## Visor

O visor procura o grupo `Muralhas_grupo` para enquadrar e o objeto `Terreno` para a caixa do chão; os elementos têm `userData.elemento`. As âncoras das placas e os pontos de vista são calculados no `construir.py` e gravados em `dados/medidas.json`; o `visor.py` lê-os.

## Vistas de verificação

`vistas/` repete os pontos de vista das fotografias (`m12_sudoeste`, `m11_sul`, `m09_torre`, `m03_norte`, `m02_sul`, `m05_entrada`, `m06_porta_baixo`) e acrescenta `capa`, `topo` (para comparar com o ortofotomapa), `vila_noroeste`, `porta_carvalho`, `barbaca` e `torre_se`.

## Fotografias

Das doze fotografias enviadas pelo Tiago, dez estão na galeria da ficha (`imagens.py`). As duas com marca de água de outro autor ou de uma rede social (m04 e m08) serviram só de referência e não estão no repositório.
