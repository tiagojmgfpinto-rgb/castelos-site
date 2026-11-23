"""Acrescenta ao JSON do visor (src/data/maquetas/silves.json) a lista de elementos, com as medidas e as âncoras
que o construir.py gravou em dados/medidas.json. Os textos são escritos à mão aqui; corre depois do construir.py:
    python pipeline/maquetas/silves/visor_elementos.py
"""
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
JS = AQUI.parents[2] / "src" / "data" / "maquetas" / "silves.json"
M = json.loads((AQUI / "dados" / "medidas.json").read_text())
A = M["ancoras"]
T = M["torres"]


def n(v, c=1):
    return f"{v:.{c}f}".replace(".", ",")


def pl(t):
    a, b = sorted(T[t]["planta"], reverse=True)
    return f"{n(a)} × {n(b)} m"


PC = M["porta_cidade"]
CG = M["casa_guarda"]
CI = M["cisterna"]
BR = M["bracos_albarras"]
ESC = M["escavacoes"]
EL = [
    dict(id="menagem", nome="Torre de menagem", grupo="Torre", estado="estado-atual", confianca="alta",
         descricao="A maior torre da alcáçova, no canto norte, saliente da muralha. Retangular, de grés vermelho em fiadas, com terraço e merlões retangulares com seteira em todas as faces; entra-se do recinto por uma porta de arco quebrado.",
         medidas=f"Planta de {pl('menagem')}; merlões a {n(T['menagem']['merloes_pontas'], 2)} m e terraço a {n(T['menagem']['terraco'])} m de altitude, c. 12 m acima do chão de fora (LiDAR 2024).",
         fontes=["lidar", "osm", "commons"], resumo="A maior das onze torres", ancora=A["menagem"]),
    dict(id="porta", nome="Porta e casa da guarda", grupo="Muralhas", estado="estado-atual", confianca="média",
         descricao="Entra-se por um arco de volta perfeita na face poente da casa da guarda, logo a norte da torre da porta e recuado em relação a ela, com a estátua de bronze de D. Sancho I diante da torre. Dentro, o corredor abobadado, de calçada e com as folhas de madeira abertas, vira a norte e sobe até ao recinto (porta em cotovelo; o traçado do corredor de saída é hipótese).",
         medidas=f"Arco de {n(CG['arco_entrada'][0])} m de largura e c. {n(CG['arco_entrada'][1] - CG['z_in'])} m ao fecho; soleira a {n(CG['z_in'])} m e recinto a {n(CG['z_out'])} m; casa da guarda de {n(CG['planta'][0])} × {n(CG['planta'][1])} m com terraço a {n(CG['terraco'])} m (LiDAR e fotografias).",
         fontes=["lidar", "commons", "fotos"], resumo="Arco ao lado da estátua de D. Sancho I", ancora=A["porta"],
         desde=1965, desdeTexto="estátua de D. Sancho I de Leopoldo de Almeida, 1965"),
    dict(id="albarra_sueste", nome="Torre albarrã de sueste", grupo="Torre", estado="estado-atual", confianca="média",
         descricao="Torre destacada da muralha e ligada a ela por um braço com adarve e merlões, atravessado por um arco alto por onde se passa ao pé da muralha.",
         medidas=f"Torre de c. 7,2 × 7,2 m com merlões a {n(T['albarra_sueste']['merloes_pontas'], 2)} m; braço com adarve a {n(BR['albarra_sueste']['adarve'], 2)} m e chão a {n(BR['albarra_sueste']['chao'])} m; arco de c. 4 m de vão (LiDAR; largura do braço e altura do arco estimadas pelas fotos).",
         fontes=["lidar", "commons"], resumo="Torre destacada com braço em arco", ancora=A["albarra_sueste"]),
    dict(id="albarra_nordeste", nome="Torre albarrã de nordeste", grupo="Torre", estado="estado-atual", confianca="média",
         descricao="A outra albarrã, no canto nordeste, presa ao ângulo da muralha por um braço curto em arco, com merlões por cima.",
         medidas=f"Planta de {pl('albarra_nordeste')}; merlões a {n(T['albarra_nordeste']['merloes_pontas'], 2)} m; braço com adarve a {n(BR['albarra_nordeste']['adarve'], 2)} m (LiDAR).",
         fontes=["lidar", "commons"], resumo="Albarrã do canto nordeste", ancora=A["albarra_nordeste"]),
    dict(id="cisterna", nome="Cisterna da Moura", grupo="Interior", estado="estado-atual", confianca="alta",
         descricao="Grande cisterna almóada de quatro naves com abóbadas de berço sobre doze colunas. À superfície vê-se o extradorso das quatro abóbadas, rebocado e pintado de vermelho-rosa, rodeado por um murete da mesma cor que a norte e a poente segura a encosta.",
         medidas=f"Planta de c. {n(CI['planta'][0])} × {n(CI['planta'][1])} m pelo LiDAR (20 × 16 × 7 m na Wikipédia); extradorso a {n(CI['extradorso'][0])}–{n(CI['extradorso'][1])} m.",
         fontes=["lidar", "wiki"], resumo="Quatro naves abobadadas, 1,3 milhões de litros", ancora=A["cisterna"]),
    dict(id="escavacoes", nome="Escavações do Palácio das Varandas", grupo="Interior", estado="estado-atual", confianca="média",
         descricao="A sueste do recinto, as escavações puseram à vista os alicerces de taipa e grés do palácio almóada e das casas à volta, com pátios, compartimentos e ruas; percorrem-se por passadiços de madeira com guarda. Os muros são os que o LiDAR mede acima do fundo das escavações.",
         medidas=f"{ESC['muros']} troços de muro em c. {ESC['area_muros_m2']} m² de planta, de 0,4 a 2 m de altura; {ESC['passadicos']} passadiços com c. {ESC['area_passadicos_m2']} m² (LiDAR 2024 de 25 cm e ortofotomapa).",
         fontes=["lidar", "commons"], resumo="Alicerces do palácio e do bairro almóada", ancora=A["escavacoes"]),
    dict(id="segredo", nome="Torre do Segredo", grupo="Torre", estado="estado-atual", confianca="alta",
         descricao="Torre do canto noroeste, onde a muralha da vila se liga à alcáçova.",
         medidas=f"Planta de {pl('segredo')}; merlões a {n(T['segredo']['merloes_pontas'], 2)} m (LiDAR).",
         fontes=["lidar", "osm"], resumo="Onde a cerca da vila encontra a alcáçova", ancora=A["segredo"]),
    dict(id="mulheres", nome="Torre das Mulheres", grupo="Torre", estado="estado-atual", confianca="alta",
         descricao="Torre do canto sudoeste, à esquerda de quem chega à porta.",
         medidas=f"Planta de {pl('mulheres')}; merlões a {n(T['mulheres']['merloes_pontas'], 2)} m (LiDAR).",
         fontes=["lidar", "osm", "commons"], resumo="Canto sudoeste, junto à porta", ancora=A["mulheres"]),
    dict(id="muralha_alcacova", nome="Muralha da alcáçova", grupo="Muralhas", estado="estado-atual", confianca="alta",
         descricao="Nove panos de taipa forrada de grés vermelho, com adarve, guardas de ferro do lado de dentro e merlões retangulares, entre onze torres (dez medidas no LiDAR e a casa da guarda). Sobe-se ao adarve norte por uma escada de pedra com patamar, encostada ao muro.",
         medidas=f"Perímetro de c. {M['alcacova']['perimetro_m']} m e área de c. {n(M['alcacova']['area_m2'] / 10000, 2)} ha; panos de 2,4 a 3,0 m de espessura; {M['merloes_alcacova']} merlões (LiDAR e fotografias).",
         fontes=["lidar", "commons", "wiki"], resumo="Caminho de ronda de c. 388 m", ancora=[-50.0, 62.0, 20.0]),
    dict(id="cerca", nome="Muralha da vila", grupo="Muralha_vila", estado="estado-atual", confianca="média",
         descricao="Os troços da cerca da vila que sobrevivem, de taipa forrada de grés, entre as casas, com as torres que restam (embasamento de grés e taipa à vista por cima).",
         medidas="Seis troços com c. " + n(sum(v["comprimento_m"] for v in M["cerca"].values()) / 1000, 2) + " km no total e 13 torres; 2,8 m de espessura (Wikipédia); altura do LiDAR.",
         fontes=["lidar", "osm", "wiki", "commons"], resumo="Cerca urbana de taipa e grés", ancora=A["cerca_norte"]),
    dict(id="porta_cidade", nome="Porta da Cidade", grupo="Muralha_vila", estado="estado-atual", confianca="média",
         descricao="Torreão maciço de grés vermelho a sul da cerca, sem merlões, com um telhado de quatro águas escondido atrás do parapeito e duas janelas de sacada de moldura branca e guarda de ferro. Entra-se por um arco alto e estreito na face poente, por baixo da janela norte, e a passagem vira para dentro da vila (o corredor por dentro é hipótese). Do largo sobe-se por uma escada íngreme, encostada à face norte, até um terraço com guarda.",
         medidas=f"Planta de c. {n(PC['planta'][1])} × {n(PC['planta'][0])} m e terraço a {n(PC['terraco'])} m, c. {n(PC['terraco'] - min(PC['chao_sul'], PC['chao_norte']), 0)} m acima do largo (LiDAR).",
         fontes=["lidar", "osm", "commons"], resumo="Torreão da Porta de Loulé, século XIV", ancora=A["porta_cidade"], desde=1300,
         desdeTexto="século XIV"),
]
d = json.loads(JS.read_text())
d["elementos"] = EL
d["epocas"] = [
    {"id": "almoada", "nome": "Alcáçova almóada", "ate": 1940, "periodo": "séculos XII–XIII",
     "texto": "A alcáçova de taipa forrada de grés, com as torres albarrãs e a cisterna da Moura, é provavelmente almóada. D. Sancho I toma a cidade em 1189; a reconquista definitiva é de 1249.",
     "fontes": ["wiki"]},
    {"id": "restauro", "nome": "Restauro da DGEMN", "ate": 2100, "periodo": "década de 1940",
     "texto": "A DGEMN consolida a torre de menagem e refaz os parapeitos e as ameias; em 1965 põe-se a estátua de D. Sancho I à porta.",
     "fontes": ["wiki"]},
]
JS.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
print(len(EL), "elementos")
