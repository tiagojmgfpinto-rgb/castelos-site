"""Castelo de Montalegre: escreve os dados do visor (src/data/maquetas/montalegre.json): placas, elementos, fontes
e limitações. As âncoras, os pontos de vista e as medidas vêm de dados/medidas.json, que o construir.py grava.

    python pipeline/maquetas/montalegre/visor.py
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[3]
M = json.loads((R / "pipeline/maquetas/montalegre/dados/medidas.json").read_text())
anc, olh = M["ancoras"], M["olhar"]
T, CE, E, CO = M["torre"], M["cerca"], M["escada_torre"], M["corredor"]
SE, S, SO = M["torre_se"], M["torre_s"], M["torre_so"]


def n(v, k=1):
    return f"{v:.{k}f}".replace(".", ",")


def el(i, nome, grupo, conf, desc, med, fontes, resumo):
    e = {"id": i, "nome": nome, "grupo": grupo, "estado": "estado-atual", "confianca": conf, "descricao": desc, "medidas": med,
         "fontes": fontes, "resumo": resumo, "ancora": anc[i]}
    if i in olh:
        e["olhar"] = olh[i]
    return e


d = {
    "monumento": "montalegre", "titulo": "Castelo de Montalegre", "variante": "estado-atual",
    "versao": "0.2 provisória", "data": "2026-11-04",
    "origem": {"lat": 41.826068, "lon": -7.790920,
               "nota": "Centro da torre de menagem (LiDAR). Eixos da maqueta: x para este, y para norte (PT-TM06 menos a origem), no visor -z é o norte, em metros à escala 1:1; as cotas são altitudes."},
    "fontes": {
        "lidar": {"titulo": "DGT, LiDAR 2024: MDT e MDS de 50 cm e nuvem de pontos classificada com cor, folha 228540 (CC BY 4.0). Planta e cotas das quatro torres, da cerca, do corredor de aço, da barbacã e das ruínas.", "url": "https://cdd.dgterritorio.gov.pt"},
        "orto": {"titulo": "DGT, Ortofotomapa 2021, 25 cm (CC BY 4.0). Conferência da planta, dos telhados, do pátio, da cisterna e do caminho da barbacã.", "url": "https://cdd.dgterritorio.gov.pt"},
        "geoportal": {"titulo": "Município de Montalegre, MunWebGis: ortofotomapa de 2011 (conferência da planta antes do corredor de aço).", "url": "https://sig.cm-montalegre.pt/MunWebGis/Viewer.aspx?serviceName=pdm"},
        "osm": {"titulo": "OpenStreetMap (ODbL), via Overture Maps 2026-10-21.0: edifícios e ruas à volta.", "url": "https://www.openstreetmap.org/#map=19/41.82607/-7.79092"},
        "wiki": {"titulo": "Wikipédia — Castelo de Montalegre (história, torre de menagem de c. 27 m com balcões e merlões, cisterna no pátio, portas descritas em 1758)", "url": "https://pt.wikipedia.org/wiki/Castelo_de_Montalegre"},
        "fotos": {"titulo": "Fotografias do castelo enviadas pelo autor do projeto, novembro de 2026 (drone de sul, de nascente e do alto da torre sueste; de poente, de norte com o passadiço; a escada de aço da torre; o pátio e as portas das torres vistos do alto da torre de menagem, 2022)"},
        "sketchfab": {"titulo": "Rui Oliveira, Castelo de Montalegre 2019 (Sketchfab): reconstituição de fases históricas, consultada só como referência de forma", "url": "https://sketchfab.com/3d-models/castelo-de-montalegre-2019-49d4d02350d847dc98850b3a8f36ff4e"},
    },
    "limitacoes": [
        "Não foi usada planta cotada. As quatro torres são retângulos ajustados aos pontos do coroamento no LiDAR de 2024 e a cerca segue o perfil medido de 2,5° em 2,5° a partir do centro do pátio (face de dentro, face de fora e cota do adarve); o erro esperado em planta é de 0,3 a 0,5 m e as faces descem a prumo.",
        f"Torre de menagem: {n(T['lado_u'], 2)} × {n(T['lado_v'], 2)} m, adarve a {n(T['adarve'], 2)} m e pontas dos merlões a {n(T['pontas'], 2)} m, c. {n(T['altura'])} m acima do chão (LiDAR). Os nove merlões por face, os quatro balcões com matacães e as quatro guaritas dos cunhais seguem as fotografias, com as posições dos balcões e a cota dos pisos (c. 1034 m) medidas no LiDAR; as mísulas e os matacães são esquemáticos. A face sul está tapada no LiDAR: a porta, o patamar de pedra e a escada de aço corten seguem as fotografias e a cota do patamar ({n(T['porta'], 2)} m) é estimada. As seteiras são estimativas de posição.",
        f"Torres do sul: a sueste tem onze merlões por face e telhado por dentro (pontas a {n(SE['pontas'], 2)} m); a sul tem parapeito liso a {n(S['peitoril'], 2)} m e a pequena de sudoeste um rebordo fino a {n(SO['peitoril'], 2)} m com a pirâmide baixa (LiDAR). As portas seguem a fotografia de 2022 tirada do alto da torre de menagem: na torre de sueste, uma porta alta para o adarve e outra em arco ao nível do pátio; nas torres do sul e de sudoeste, portas em arco para o adarve. O tamanho das portas (c. 0,9 × 2,0 m) é estimado.",
        "A cerca não tem merlões; os adarves altos de sul e de poente têm guarda de ferro pelo lado de fora (fotografias). A poterna junto à torre sueste (com o arco de dentro, encostado à torre) e a porta em arco entre as torres sul e de sudoeste seguem as fotografias e ficaram fechadas com portas de madeira; o tamanho (1,0 × 2,0 m) é estimado. A cisterna tem a grade de ferro da fotografia de 2022.",
        f"O troço noroeste da cerca é um muro arruinado com um corredor de entrada de aço corten por cima, de cobertura plana a {n(CO['cobertura'], 2)} m (LiDAR: pontos de edifício avermelhados; faixa vermelha no ortofotomapa de 2021). A porta de chapa, a placa e os degraus seguem a fotografia de norte; a saída para o pátio junto à torre é estimada. O passadiço de madeira segue a faixa clara e lisa do LiDAR entre as duas ruínas em ferradura; os balizadores são esquemáticos.",
        "A barbacã é o muro de suporte do patamar a nascente e a sul, traçado onde o MDT cai 2,5 a 4 m; o caminho lajeado claro por cima segue o ortofotomapa e a vedação de madeira a fotografia de drone de nascente. As duas linhas de muralha exteriores desenhadas por Duarte de Armas (c. 1509) já não existem e não foram feitas.",
        "Pedra: granito cinzento em fiadas regulares nas torres e alvenaria mais miúda na cerca (fotografias). Telhados de telha vermelha. O pátio é de terra batida clara (ortofotomapa) e o resto do monte é relva. A cisterna ficou como um poço quadrado com bocal baixo.",
        "As árvores estão nas posições e com as alturas do LiDAR, com copas esquemáticas e afastadas dos muros e das torres. Os edifícios da vila à volta são volumes simples de um só cinzento.",
        "O Google Street View não chega ao interior do castelo; a conferência fez-se com os ortofotomapas da DGT (2021) e do geoportal da Câmara (2011) e com as fotografias. O modelo do Sketchfab de 2019 é uma reconstituição histórica e não foi usado para o estado atual.",
    ],
    "distancia": 110, "distanciaMinima": 10, "direcao": [-0.45, 0.55, 0.70], "sol": [0.34, 0.77, 0.54],
    "placas": ["torre", "coroamento", "balcoes", "porta_torre", "escada_torre", "torre_se", "torre_s", "torre_so", "cerca", "corredor", "passadico", "cisterna", "ruinas", "barbaca", "patio"],
    "elementos": [
        el("torre", "Torre de Menagem", "Torre", "alta",
           "A torre de menagem, a norte do pátio, é a mais alta do castelo: uma torre quadrangular gótica de cantaria de granito, com c. 27 m, salas abobadadas por dentro e telhado de quatro águas de telha por dentro do adarve. A obra do castelo acabou em 1331, com D. Afonso IV.",
           f"{n(T['lado_u'], 2)} × {n(T['lado_v'], 2)} m de planta; pontas dos merlões a {n(T['pontas'], 2)} m, c. {n(T['altura'])} m acima do chão (LiDAR 2024).",
           ["lidar", "orto", "fotos", "wiki"], "Torre gótica de c. 27 m"),
        el("coroamento", "Merlões", "Torre", "alta",
           "Nove merlões piramidais por face, contando os dos cunhais. No alto da escada interior há uma casota de pedra, com o mastro da bandeira por cima.",
           f"Peitoril a {n(T['peitoril'], 2)} m, pontas a {n(T['pontas'], 2)} m (LiDAR e fotografias).", ["lidar", "fotos"], "Nove merlões piramidais por face"),
        el("balcoes", "Balcões e guaritas", "Torre", "média",
           "Cada face tem um balcão com matacães sobre mísulas e cada cunhal uma guarita, também sobre mísulas, com dois merlões por lado. Por baixo dos matacães abriam-se os buracos para defender o pé da torre.",
           "Pisos a c. 1034 m e parapeitos a c. 1035,4 m (LiDAR); mísulas esquemáticas.", ["lidar", "fotos", "wiki"], "Balcões com matacães e guaritas"),
        el("porta_torre", "Porta da torre", "Torre", "média",
           "A porta fica alta, na face voltada ao pátio, em arco de volta perfeita, sobre um patamar de pedra com cornija.",
           f"Patamar a c. {n(T['porta'], 2)} m, c. 5 m acima do pátio (estimado nas fotografias).", ["fotos"], "Porta alta sobre o patamar"),
        el("escada_torre", "Escada de aço", "Torre", "média",
           "Uma escada moderna de aço corten, com chapas cheias de guarda, sobe do pátio ao patamar da porta ao longo da face sul.",
           f"{E['degraus']} degraus de c. {n(E['espelho'] * 100, 0)} cm, de {n(E['pe'], 2)} a {n(E['patamar'], 2)} m (estimado nas fotografias).", ["fotos"], "Escada de aço corten"),
        el("torre_se", "Torre de sueste", "Torres", "alta",
           "A maior das três torres do sul, quadrada, com onze merlões piramidais por face e telhado de telha por dentro do adarve.",
           f"{n(SE['lados'][0], 2)} × {n(SE['lados'][1], 2)} m; pontas dos merlões a {n(SE['pontas'], 2)} m, c. {n(SE['altura'])} m acima do chão (LiDAR).", ["lidar", "orto", "fotos"], "Torre quadrada com merlões"),
        el("torre_s", "Torre do sul", "Torres", "alta",
           "Torre retangular de parapeito liso, sem merlões, com o telhado de telha por dentro. Segundo a Wikipédia, a inscrição da base de uma das torres do sul regista o fim da obra em 1331 e o reparo de 1580.",
           f"{n(S['lados'][0], 2)} × {n(S['lados'][1], 2)} m; parapeito a {n(S['peitoril'], 2)} m (LiDAR).", ["lidar", "orto", "fotos", "wiki"], "Torre de parapeito liso"),
        el("torre_so", "Torre pequena", "Torres", "alta",
           "A mais pequena das torres, a sudoeste, com um rebordo fino e um telhado baixo em pirâmide; uma porta em arco abre para o adarve.",
           f"{n(SO['lados'][0], 2)} × {n(SO['lados'][1], 2)} m; rebordo a {n(SO['peitoril'], 2)} m (LiDAR).", ["lidar", "fotos"], "Torre pequena de sudoeste"),
        el("cerca", "Cerca e adarve", "Muralhas", "alta",
           "A muralha liga as quatro torres à volta da praça de armas. O adarve, sem merlões, sobe de nascente para sul; a sul e a poente tem guarda de ferro. Na base, a poterna junto à torre sueste e uma porta em arco entre as torres do sul.",
           f"Adarve a {n(CE['adarves']['nascente'], 2)} m a nascente e {n(CE['adarves']['sul'], 2)} m a sul (LiDAR).", ["lidar", "orto", "fotos", "wiki"], "Muralha com adarve"),
        el("corredor", "Corredor de aço corten", "Muralhas", "alta",
           "No troço noroeste, onde a muralha está arruinada, a entrada faz-se hoje por um corredor coberto de aço corten, assente no muro antigo, com uma porta de chapa ao fundo do passadiço. Do adarve de poente desce para ele uma escada de aço.",
           f"Cobertura a {n(CO['cobertura'], 2)} m; soleira da porta a c. {n(CO['soleira'], 2)} m (LiDAR e fotografia).", ["lidar", "orto", "fotos"], "Entrada moderna de aço"),
        el("passadico", "Passadiço", "Muralhas", "média",
           "Um passadiço de tábuas de madeira, ladeado por balizadores baixos, atravessa o relvado de norte até à porta do corredor, entre as duas ruínas em ferradura.",
           f"C. {n(M['passadico']['comprimento'], 0)} m de comprimento e {n(M['passadico']['largura'])} m de largura (LiDAR e fotografia).", ["lidar", "fotos"], "Passadiço de madeira"),
        el("cisterna", "Cisterna", "Muralhas", "alta",
           "A cisterna quadrada no meio do pátio, com o bocal de pedra.",
           "C. 5 × 5 m (ortofotomapa e LiDAR).", ["orto", "lidar", "wiki"], "Cisterna no meio do pátio"),
        el("ruinas", "Ruínas em ferradura", "Muralhas", "média",
           "Dois muros baixos de pedra em ferradura, abertos a nascente, no relvado a norte da torre de menagem.",
           "C. 1 m de altura; 5,3 e 6 m de diâmetro (LiDAR).", ["lidar", "orto", "fotos"], "Muros baixos a norte"),
        el("barbaca", "Barbacã", "Muralhas", "alta",
           "O muro de suporte que contorna o castelo a nascente e a sul, com um caminho lajeado por cima e uma vedação de madeira.",
           f"C. {n(M['barbaca']['comprimento'], 0)} m de comprimento; até c. {n(M['barbaca']['altura_max'])} m de altura (LiDAR).", ["lidar", "orto", "fotos"], "Muro de suporte com caminho"),
        el("patio", "Praça de armas", "Muralhas", "alta",
           "O pátio de terra batida por dentro da cerca, com a cisterna no meio. Aqui ficava a casa do governador descrita em 1758.",
           f"Pátio a {n(M['terreno']['patio_cota'][0], 1)}–{n(M['terreno']['patio_cota'][1], 1)} m (LiDAR).", ["lidar", "orto", "wiki"], "Pátio de terra batida"),
    ],
}
d["elementos"] += [
    {"id": "terreno", "nome": "Monte e relvado", "grupo": "Terreno", "estado": "estado-atual", "confianca": "alta",
     "descricao": "O relevo do LiDAR, com o relvado à volta do castelo, o pátio de terra batida e os caminhos lajeados.",
     "medidas": "MDT de 50 cm (LiDAR 2024).", "fontes": ["lidar", "orto", "osm"], "resumo": "Relevo e relvado"},
    {"id": "envolvente", "nome": "Edifícios à volta", "grupo": "Envolvente", "estado": "estado-atual", "confianca": "baixa",
     "descricao": "Os edifícios da vila à volta, em volumes simples e num só cinzento, só para dar a escala.",
     "medidas": "Plantas do OSM e do LiDAR; alturas do MDS.", "fontes": ["osm", "lidar"], "resumo": "Volumes simples"},
    {"id": "vegetacao", "nome": "Árvores", "grupo": "Envolvente", "estado": "estado-atual", "confianca": "baixa",
     "descricao": "As árvores à volta do castelo, nas posições e com as alturas do LiDAR; copas esquemáticas.",
     "medidas": "LiDAR 2024.", "fontes": ["lidar", "orto"], "resumo": "Árvores"},
]
(R / "src/data/maquetas/montalegre.json").write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
print("ok")
