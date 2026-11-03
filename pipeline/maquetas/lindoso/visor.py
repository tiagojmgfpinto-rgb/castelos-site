"""Castelo de Lindoso: escreve os dados do visor (src/data/maquetas/lindoso.json): placas, elementos, fontes e
limitações. As âncoras, os pontos de vista e as medidas vêm de dados/medidas.json, que o construir.py grava.

    python pipeline/maquetas/lindoso/visor.py
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[3]
M = json.loads((R / "pipeline/maquetas/lindoso/dados/medidas.json").read_text())
anc, olh = M["ancoras"], M["olhar"]
T, MU, E = M["torre"], M["museu"], M["espigueiros"]


def n(v, k=1):
    return f"{v:.{k}f}".replace(".", ",")


def el(i, nome, grupo, conf, desc, med, fontes, resumo):
    e = {"id": i, "nome": nome, "grupo": grupo, "estado": "estado-atual", "confianca": conf, "descricao": desc, "medidas": med,
         "fontes": fontes, "resumo": resumo, "ancora": anc[i]}
    if i in olh:
        e["olhar"] = olh[i]
    return e


d = {
    "monumento": "lindoso", "titulo": "Castelo de Lindoso e espigueiros", "variante": "estado-atual", "versao": "0.1 provisória",
    "data": "2026-11-03",
    "origem": {"lat": 41.867167, "lon": -8.199305,
               "nota": "Centro da torre de menagem (LiDAR). Eixos da maqueta: x para este, y para norte (PT-TM06 menos a origem), no visor -z é o norte, em metros à escala 1:1; as cotas são altitudes."},
    "fontes": {
        "lidar": {"titulo": "DGT, LiDAR 2025: MDT e MDS de 50 cm e nuvem de pontos classificada, folha 194545 (CC BY 4.0). Planta e cotas da torre, dos muros (traçados nos pontos de solo, edifícios e muros, em células de 25 cm), dos edifícios do museu e dos espigueiros.", "url": "https://cdd.dgterritorio.gov.pt"},
        "orto": {"titulo": "DGT, Ortofotomapa 2021, 25 cm (CC BY 4.0). Conferência da planta, dos telhados, do pátio e do monte dos espigueiros.", "url": "https://cdd.dgterritorio.gov.pt"},
        "geoportal": {"titulo": "Município de Ponte da Barca, plataforma de informação geográfica: imagem de satélite e ortofotomapas 2021 e 2023 (conferência da planta dos baluartes, do museu e da eira).", "url": "https://pontedabarca.city-platform.com/v2/app?a=pdm"},
        "osm": {"titulo": "OpenStreetMap (ODbL), via Overture Maps 2026-10-21.0: plantas dos espigueiros e das casas, ruas.", "url": "https://www.openstreetmap.org/#map=19/41.86717/-8.19930"},
        "wiki": {"titulo": "Wikipédia — Castelo de Lindoso (história, portas, revelim, guaritas)", "url": "https://pt.wikipedia.org/wiki/Castelo_de_Lindoso"},
        "fotos": {"titulo": "Fotografias do castelo e dos espigueiros enviadas pelo autor do projeto, novembro de 2026 (drone de sul, aéreas, torre do pátio, Porta Norte, porta da traição, pátio e museu, guarita do cunhal sueste, espigueiros)"},
    },
    "limitacoes": [
        "Não foi usada planta cotada. A planta dos muros foi traçada no LiDAR de 2025: são as células com mais de 0,7 m de relevo local (num disco de 3 m) e mais de 0,9 m acima do MDT, para as lices e o pátio ficarem de fora; o contorno foi alisado e simplificado a 0,45 m. O topo de cada muro tem a cota medida em cada vértice e as faces descem a prumo até ao chão. O LiDAR quase não tem pontos nas faces; o erro esperado em planta é de 0,3 a 0,5 m, e os muros não têm o talude nem o cordão que alguns troços possam ter.",
        "Os terraplenos largos dos baluartes ficaram como parte do muro, com o topo em erva; os pequenos muros de pedra solta do monte não foram modelados. Escadas para os adarves não se veem no LiDAR e não foram feitas.",
        f"Torre: planta de {n(T['lado'])} m, rodada {n(64.25, 2)}° (faces a ENE, NNO, OSO e SSE); soleira das ameias a {n(T['soleira_ameias'], 2)} m, pontas dos merlões a {n(T['pontas_merloes'], 2)} m e telhado de quatro águas com o vértice a {n(T['cume_telhado'], 2)} m (LiDAR). Os sete merlões por face com remate piramidal e a janela da face SSE, com portadas verdes, seguem as fotografias. A porta na face OSO e as seteiras são estimativas de posição.",
        "A Porta Norte está onde o LiDAR mostra o bloco mais alto na ponta poente da cortina norte, de frente para a face NNO da torre, como na fotografia; o vão (1,7 × 2,75 m), o portão de grade e o estrado de madeira são da fotografia, à escala da altura do muro. A torre da porta a sul, com a passagem em arco de volta perfeita (2,0 × 3,2 m), tem planta e topo do LiDAR; a profundidade da passagem é estimada. A porta da traição ao pé da torre é da fotografia, com a posição estimada.",
        "As guaritas redondas (cunhal sueste da cerca e ponta do baluarte poente) seguem as fotografias; o diâmetro (1,25 m) e a altura são estimativas.",
        "Museu: plantas do LiDAR e do satélite, cornijas e cumes do LiDAR. Portas, janelas e a galeria envidraçada são esquemáticas.",
        f"Espigueiros: {E['total']} ({E['osm']} com a planta do OSM e os restantes do LiDAR), com o comprimento, a largura, a orientação e a altura do cume medidos. Os pés, as mesas, as grades de madeira, as empenas, a porta em arco, o telhado de lajes, a cruz e o pináculo são iguais em todos e seguem as fotografias; o número de pés e de esteios depende do comprimento. As portas foram postas, alternadamente, num topo ou no outro. Alguns espigueiros fora da área da maqueta ou escondidos por árvores podem faltar.",
        "A pedra é feita por texturas tiradas das fotografias: cantaria de granito cinzento em fiadas na torre e na cerca medieval, alvenaria mais miúda na muralha abaluartada, granito liquenado nos espigueiros. O pátio é relvado; o monte tem erva seca e afloramentos de granito onde o MDT e o ortofotomapa os mostram.",
        "As casas à volta são volumes simples, num só cinzento, para dar a escala sem tirar destaque ao castelo e aos espigueiros (plantas do OSM e, onde faltam, do LiDAR; cornija e cume medidos no MDS). As árvores estão nas posições do LiDAR (julho), com copas esquemáticas.",
        "O Google Street View não chega ao interior do castelo nem à eira; a conferência foi feita com as fotografias enviadas, o ortofotomapa de 2021 e a imagem de satélite da plataforma do Município de Ponte da Barca.",
    ],
    "distancia": 170, "distanciaMinima": 12, "direcao": [-0.42, 0.62, 0.66], "sol": [-0.38, 0.74, 0.55],
    "placas": ["torre", "coroamento", "cerca", "porta_sul", "porta_norte", "postigo", "guaritas", "baluartes", "museu", "espigueiros"],
    "elementos": [
        el("torre", "Torre de Menagem", "Torre", "alta",
           "Torre quadrangular de cantaria de granito no canto norte da cerca medieval, com a face NNO para fora. Tem uma janela com portadas verdes na face voltada ao pátio e seteiras estreitas; por dentro do parapeito, um telhado de quatro águas de telha. É do tempo de D. Afonso III e D. Dinis, restaurada pela DGEMN a partir dos anos 1940.",
           f"Planta de {n(T['lado'])} × {n(T['lado'])} m; soleira das ameias a {n(T['soleira_ameias'], 2)} m, c. {n(T['soleira_ameias'] - T['chao']['SSE'])} m acima do pátio (LiDAR 2025).",
           ["lidar", "orto", "fotos", "wiki"], "Torre quadrangular no canto norte da cerca"),
        el("coroamento", "Merlões", "Torre", "alta",
           "Sete merlões por face, contando os dos cunhais, com remate em pirâmide, sobre o parapeito; a bandeira nacional no mastro do lado norte.",
           f"Pontas dos merlões a c. {n(T['pontas_merloes'], 2)} m (LiDAR e fotografias).", ["lidar", "fotos"], "Sete merlões piramidais por face"),
        el("cerca", "Cerca medieval", "Muralhas", "alta",
           "Muros altos de cantaria de granito à volta do pátio, com adarve no alto. A torre de menagem fica no canto norte e a torre da porta a sul.",
           f"Topo a 467–{n(M['muros']['topo_max'])} m, c. 6 m acima do pátio (LiDAR).", ["lidar", "orto", "geoportal", "fotos"], "Cerca de D. Afonso III e D. Dinis"),
        el("porta_sul", "Torre da porta", "Muralhas", "alta",
           "A sul, uma torre saliente da cerca guarda a entrada, uma passagem em arco de volta perfeita que segue pela cerca até ao pátio. A Wikipédia refere uma ponte levadiça de madeira nesta porta.",
           f"Topo a {n(M['torre_porta']['topo'], 2)} m; vão de c. 2,0 × 3,2 m (LiDAR e fotografia).", ["lidar", "fotos", "wiki"], "Entrada sul em arco"),
        el("porta_norte", "Porta Norte", "Muralhas", "média",
           "Porta em arco de volta perfeita, com aduelas de granito e portão de grade, na ponta poente da cortina norte, de frente para a torre de menagem. À frente tem um estrado de madeira.",
           f"Vão de c. 1,7 × 2,75 m; muro com o topo a {n(M['porta_norte']['topo'], 2)} m; chão de fora a c. 461 m (LiDAR e fotografia).", ["lidar", "fotos", "wiki"], "Porta norte junto à torre"),
        el("postigo", "Porta da traição", "Muralhas", "média",
           "Pequena porta em arco na face de fora da cerca, ao pé da torre de menagem, para sair para a lice sem passar pelas portas.",
           "Vão de c. 0,8 × 1,9 m (fotografia; posição estimada).", ["fotos"], "Postigo ao pé da torre"),
        el("guaritas", "Guaritas", "Muralhas", "média",
           "Guaritas redondas de pedra com cúpula, assentes em mísulas nos cunhais: uma no cunhal sueste da cerca, com uma gárgula por baixo, e outra na ponta do baluarte poente.",
           "C. 1,25 m de diâmetro (estimativa a partir das fotografias).", ["fotos", "wiki"], "Guaritas com cúpula nos cunhais"),
        el("baluartes", "Muralha abaluartada", "Muralhas", "alta",
           "Na Guerra da Restauração a cerca medieval foi envolvida por uma muralha abaluartada de planta estrelada, com baluartes nos cunhais, lices entre as duas cercas e um revelim a norte. As obras acabaram por volta de 1666 e o revelim principal por volta de 1720.",
           "Muros de 1,2 a 2 m de espessura, com o topo a 462–465 m, 2 a 4 m acima das lices e do campo (LiDAR).", ["lidar", "orto", "geoportal", "wiki", "fotos"], "Fortificação do século XVII"),
        el("museu", "Edifícios do museu", "Museu", "média",
           "Duas alas de paredes rebocadas a branco e telhados de quatro águas, encostadas à cerca poente e à cerca sul, ligadas por uma galeria envidraçada. Ocupam o lugar das antigas casas do alcaide e da guarnição.",
           f"Cumes a {n(MU['ala_oeste']['cume'], 2)} m (ala poente) e {n(MU['ala_sul']['cume'], 2)} m (ala sul) (LiDAR).", ["lidar", "orto", "fotos"], "Museu no pátio"),
        el("espigueiros", "Espigueiros", "Espigueiros", "alta",
           "Na eira comunitária a sul e sudoeste do castelo juntam-se mais de cinquenta espigueiros de granito, levantados do chão sobre pés com mesas para guardar o milho dos ratos. Têm grades de ripas de madeira nos lados, empenas de granito com uma porta pequena, telhado de lajes e uma cruz num dos topos.",
           f"{E['total']} espigueiros modelados, de c. 3 a 12 m de comprimento; cumes 3,3 a 3,8 m acima do chão (LiDAR e OSM).", ["lidar", "osm", "orto", "fotos"], "Eira com mais de 50 espigueiros"),
    ],
}
d["elementos"] += [
    {"id": "terreno", "nome": "Terreno, pátio e eira", "grupo": "Terreno", "estado": "estado-atual", "confianca": "alta",
     "descricao": "O relevo do LiDAR, com o pátio relvado, as lices, o monte de erva seca com os afloramentos de granito e as ruas.",
     "medidas": "MDT de 50 cm (LiDAR 2025), com os terraplenos dos baluartes do MDS.", "fontes": ["lidar", "orto", "osm"], "resumo": "Relevo, pátio, eira e ruas"},
    {"id": "envolvente", "nome": "Casas da aldeia", "grupo": "Envolvente", "estado": "estado-atual", "confianca": "baixa",
     "descricao": "As casas à volta do castelo e da eira, em volumes simples e num só cinzento, só para dar a escala.",
     "medidas": "Plantas do OSM e do LiDAR; cornijas e cumes medidos no MDS de 50 cm.", "fontes": ["osm", "lidar"], "resumo": "Volumes simples das casas à volta"},
    {"id": "vegetacao", "nome": "Árvores", "grupo": "Envolvente", "estado": "estado-atual", "confianca": "baixa",
     "descricao": "As árvores à volta do castelo e da eira, nas posições e com as alturas do LiDAR; copas esquemáticas.",
     "medidas": "LiDAR 2025 (julho).", "fontes": ["lidar", "orto"], "resumo": "Árvores à volta"},
]
(R / "src/data/maquetas/lindoso.json").write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
print("ok")
