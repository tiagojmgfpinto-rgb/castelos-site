"""Castelo de Lanhoso: escreve os dados do visor (src/data/maquetas/lanhoso.json): placas, elementos, fontes e
limitações. As âncoras, os pontos de vista e as medidas vêm de dados/medidas.json, que o construir.py grava.

    python pipeline/maquetas/lanhoso/visor.py
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[3]
M = json.loads((R / "pipeline/maquetas/lanhoso/dados/medidas.json").read_text())
anc, olh = M["ancoras"], M["olhar"]
T, CE, P, S, E = M["torre"], M["cerca"], M["porta"], M["santuario"], M["escada_torre"]


def n(v, k=1):
    return f"{v:.{k}f}".replace(".", ",")


def el(i, nome, grupo, conf, desc, med, fontes, resumo):
    e = {"id": i, "nome": nome, "grupo": grupo, "estado": "estado-atual", "confianca": conf, "descricao": desc, "medidas": med,
         "fontes": fontes, "resumo": resumo, "ancora": anc[i]}
    if i in olh:
        e["olhar"] = olh[i]
    return e


d = {
    "monumento": "lanhoso", "titulo": "Castelo de Lanhoso e Santuário de Nossa Senhora do Pilar", "variante": "estado-atual",
    "versao": "0.3 provisória", "data": "2026-11-03",
    "origem": {"lat": 41.586850, "lon": -8.281361,
               "nota": "Centro da torre de menagem (LiDAR). Eixos da maqueta: x para este, y para norte (PT-TM06 menos a origem), no visor -z é o norte, em metros à escala 1:1; as cotas são altitudes."},
    "fontes": {
        "lidar": {"titulo": "DGT, LiDAR 2025: MDT e MDS de 50 cm e nuvem de pontos classificada, folha 187514 (CC BY 4.0). Planta e cotas da torre, da cerca, dos torreões da porta, dos terraços e do santuário.", "url": "https://cdd.dgterritorio.gov.pt"},
        "orto": {"titulo": "DGT, Ortofotomapa 2021, 25 cm (CC BY 4.0). Conferência da planta, dos telhados, do terreiro e das lajes de granito do monte.", "url": "https://cdd.dgterritorio.gov.pt"},
        "osm": {"titulo": "OpenStreetMap (ODbL), via Overture Maps 2026-10-21.0: santuário, caminhos e estrada.", "url": "https://www.openstreetmap.org/#map=19/41.58685/-8.28136"},
        "wiki": {"titulo": "Wikipédia — Castelo de Lanhoso (história, portão de arco quebrado entre torreões ameados, merlões piramidais, porta alta da torre com escada de pedra)", "url": "https://pt.wikipedia.org/wiki/Castelo_de_Lanhoso"},
        "fotos": {"titulo": "Fotografias do castelo e do santuário enviadas pelo autor do projeto, novembro de 2026 (drone de sul e de poente, porta com o penedo, porta vista de dentro, adarve poente, pátio, fachada do santuário)"},
    },
    "limitacoes": [
        "Não foi usada planta cotada. A planta da torre, da cerca, dos torreões e dos terraços foi traçada no LiDAR de 2025 (pontos de solo e de edifício em células de 25 cm) e conferida no ortofotomapa e nas fotografias; o erro esperado em planta é de 0,3 a 0,5 m. Os adarves têm a cota medida (moda dos pontos entre os merlões) e as faces descem a prumo.",
        f"Torre: {n(T['lado_u'], 2)} × {n(T['lado_v'], 2)} m, faces a {n(T['azimute_faces'][0], 2)}° e {n(T['azimute_faces'][1], 2)}°; adarve a {n(T['adarve'], 2)} m, peitoril das ameias a {n(T['peitoril'], 2)} m, pontas dos merlões a {n(T['pontas'], 2)} m e vértice do telhado a {n(T['cume'], 2)} m (LiDAR). Oito merlões por face e as seteiras em alguns merlões seguem as fotografias de drone. A porta alta na face NNO tem a soleira à cota do patamar da escada no LiDAR ({n(T['porta'], 2)} m); a Wikipédia diz que o arco é quebrado e as fotografias, pequenas, não chegam para o confirmar: ficou de volta perfeita. As seteiras das faces são estimativas de posição.",
        "A porta do castelo está a poente da torre, entre os dois torreões (a Wikipédia diz 'a sul, junto à torre de menagem'); o arco quebrado, as portas verdes, o merlão sobre o arco, a escadaria e o penedo seguem as fotografias. O vão (1,55 m) e o arranque do arco são medidos à escala da escadaria; o penedo é um volume aproximado.",
        "As escadas de pedra dos adarves e do terraço norte e o muro baixo no bordo nascente do terreiro (traçado na crista estreita do LiDAR) são das fotografias, com a altura estimada. A barbacã a norte e o portão norte que a Wikipédia descreve não foram modelados (ficam abaixo, na encosta coberta de árvores, e não há fotografias).",
        f"Santuário: nave de {n(S['nave'][0])} × {n(S['nave'][1])} m e capela-mor de {n(S['capela_mor'][0])} × {n(S['capela_mor'][1])} m, cumes a {n(S['cume_nave'])} e {n(S['cume_capela'])} m e adro a {n(S['adro'], 2)} m (LiDAR). A fachada (frontão, cruz, pináculos, nicho com a imagem, janelas e porta) segue a fotografia de frente, simplificada. A torre sineira e as capelas de remate em pirâmide ficam a sul do limite da folha do LiDAR e não foram feitas.",
        "Pedra: granito cinzento liquenado em fiadas na torre, na cerca e no santuário (fotografias). O pátio é relvado, o terreiro é de saibro e lajes cinzentas e as grandes lajes de granito do monte estão onde o ortofotomapa as mostra claras e inclinadas; o resto da encosta ficou em erva verde.",
        "As árvores estão nas posições e com as alturas do LiDAR (julho de 2025), com copas esquemáticas e afastadas dos muros e da torre; a nascente o LiDAR tem menos árvores do que o ortofotomapa de 2021. Os poucos edifícios à volta são volumes simples de um só cinzento.",
        "O geoportal do Município da Póvoa de Lanhoso não deixou consultar os ortofotomapas a partir daqui (acesso recusado); a conferência foi feita com o ortofotomapa de 2021 da DGT e com as fotografias enviadas. O Google Street View não chega ao interior do castelo.",
    ],
    "distancia": 120, "distanciaMinima": 10, "direcao": [0.55, 0.55, 0.62], "sol": [-0.38, 0.74, 0.55],
    "placas": ["torre", "coroamento", "porta_torre", "escada_torre", "porta", "torreoes", "cerca", "baluarte", "patio", "santuario", "fachada"],
    "elementos": [
        el("torre", "Torre de Menagem", "Torre", "alta",
           "Torre quadrangular de cantaria de granito, no ponto mais alto do Monte de Lanhoso, junto à porta do castelo. Foi erguida no fim do século XII ou no início do XIII; em 1758 era o que restava do castelo. Por dentro do parapeito tem um telhado de quatro águas de telha e um adarve de tábuas.",
           f"{n(T['lado_u'], 2)} × {n(T['lado_v'], 2)} m de planta; pontas dos merlões a {n(T['pontas'], 2)} m, c. {n(T['altura_nno'])} m acima do chão a norte (LiDAR 2025).",
           ["lidar", "orto", "fotos", "wiki"], "Torre quadrangular no alto do monte"),
        el("coroamento", "Merlões", "Torre", "alta",
           "Oito merlões por face, contando os dos cunhais, com remate piramidal; alguns têm uma seteira em arco. A bandeira nacional fica no mastro do lado sul.",
           f"Peitoril a {n(T['peitoril'], 2)} m, pontas a {n(T['pontas'], 2)} m (LiDAR e fotografias de drone).", ["lidar", "fotos"], "Oito merlões piramidais por face"),
        el("porta_torre", "Porta da torre", "Torre", "média",
           "A porta da torre fica alta, na face voltada ao pátio, a c. 4,5 m do chão, como era costume para a defesa; hoje tem portas de madeira.",
           f"Soleira a {n(T['porta'], 2)} m (LiDAR).", ["lidar", "fotos", "wiki"], "Porta alta da torre"),
        el("escada_torre", "Escada da torre", "Torre", "alta",
           "Escada de pedra encostada à face da torre, a subir do pátio para a porta, com um corrimão de ferro.",
           f"{E['degraus']} degraus de {n(E['pe'], 2)} a {n(E['patamar'], 2)} m (LiDAR e fotografias).", ["lidar", "fotos", "wiki"], "Escada de pedra para a porta"),
        el("porta", "Porta do castelo", "Muralhas", "alta",
           "A entrada faz-se por um arco quebrado com portas de madeira verde, recuado entre dois torreões ameados. Chega-se lá por uma escadaria de granito que sobe da encosta, com um grande penedo encostado ao torreão sul.",
           f"Soleira a {n(P['soleira'], 2)} m; vão de c. {n(P['vao_fora'][0], 2)} × {n(P['vao_fora'][1])} m (LiDAR e fotografias).", ["lidar", "fotos", "wiki"], "Arco quebrado entre dois torreões"),
        el("torreoes", "Torreões da porta", "Muralhas", "alta",
           "Dois torreões quadrangulares guardam a porta, com merlões piramidais nos lados de fora; o do sul encosta à torre de menagem. Foram reconstruídos pela DGEMN a partir de 1938.",
           f"Topo a {n(CE['adarves']['torreoes'], 2)} m (LiDAR).", ["lidar", "fotos", "wiki"], "Torreões ameados da porta"),
        el("cerca", "Cerca e adarve", "Muralhas", "alta",
           "A muralha da praça de armas tem adarve com parapeito: com merlões piramidais a poente e liso a norte e a nascente. Por dentro, escadas de pedra sobem do pátio e do terraço para o adarve.",
           f"Adarve a {n(CE['adarves']['poente'], 2)} m a poente, {n(CE['adarves']['norte'], 2)} m a norte e {n(CE['adarves']['nascente'], 2)} m a nascente (LiDAR).", ["lidar", "orto", "fotos", "wiki"], "Muralha com adarve"),
        el("baluarte", "Canto nordeste", "Muralhas", "alta",
           "No canto nordeste a muralha avança num ângulo saliente sobre a encosta, com vista para o vale do Ave e para a vila.",
           f"Adarve a {n(CE['adarves']['norte'], 2)} m (LiDAR).", ["lidar", "orto", "fotos"], "Ângulo saliente a nordeste"),
        el("patio", "Praça de armas", "Muralhas", "alta",
           "O pátio relvado por dentro da cerca, com uma árvore grande e um terraço mais alto ao longo da muralha norte. Aqui ficavam a residência do alcaide e a cisterna.",
           "Pátio a 382–383 m e terraço a 384,1 m (LiDAR).", ["lidar", "fotos", "wiki"], "Pátio por dentro da cerca"),
        el("santuario", "Santuário de Nossa Senhora do Pilar", "Santuario", "alta",
           "Igreja de 1680, mandada fazer por André da Silva Machado com pedra do castelo, a sul da torre: nave, capela-mor mais baixa a norte e sacristia, de granito, com telhados de telha.",
           f"Nave de {n(S['nave'][0])} × {n(S['nave'][1])} m, cume a {n(S['cume_nave'])} m (LiDAR).", ["lidar", "orto", "fotos", "wiki"], "Igreja de 1680"),
        el("fachada", "Fachada do santuário", "Santuario", "média",
           "Fachada virada a sul, com pilastras e pináculos nos cunhais, frontão com a cruz, o nicho com a imagem de Nossa Senhora entre as duas janelas de cima, a porta e duas janelas em baixo, e o adro com degraus.",
           f"Frontão a c. {n(S['frontao'])} m e cruz a c. {n(S['cruz'])} m (LiDAR e fotografia).", ["lidar", "fotos"], "Fachada barroca a sul"),
    ],
}
d["elementos"] += [
    {"id": "terreno", "nome": "Monte, terreiro e pátio", "grupo": "Terreno", "estado": "estado-atual", "confianca": "alta",
     "descricao": "O relevo do LiDAR, com as grandes lajes de granito do Monte de Lanhoso, o terreiro entre o castelo e o santuário, o pátio relvado, os marcos de granito e a estrada.",
     "medidas": "MDT de 50 cm (LiDAR 2025).", "fontes": ["lidar", "orto", "osm"], "resumo": "Relevo, lajes de granito e terreiro"},
    {"id": "envolvente", "nome": "Edifícios à volta", "grupo": "Envolvente", "estado": "estado-atual", "confianca": "baixa",
     "descricao": "Os poucos edifícios à volta, em volumes simples e num só cinzento, só para dar a escala.",
     "medidas": "Plantas do OSM e do LiDAR; alturas do MDS.", "fontes": ["osm", "lidar"], "resumo": "Volumes simples"},
    {"id": "vegetacao", "nome": "Árvores", "grupo": "Envolvente", "estado": "estado-atual", "confianca": "baixa",
     "descricao": "As árvores do monte, nas posições e com as alturas do LiDAR; copas esquemáticas.",
     "medidas": "LiDAR 2025 (julho).", "fontes": ["lidar", "orto"], "resumo": "Árvores do monte"},
]
(R / "src/data/maquetas/lanhoso.json").write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
print("ok")
