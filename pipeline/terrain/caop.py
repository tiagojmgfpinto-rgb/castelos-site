"""Extrai da CAOP 2025 (DGT) os limites administrativos e as ilhas usados no mapa.

A Carta Administrativa Oficial de Portugal é a fonte oficial dos limites de distritos, municípios e freguesias.
Os GeoPackages (c. 140 MB em zip) não entram no repositório nem no CI: corre-se este script quando sai uma CAOP
nova e grava-se o resultado, pequeno, em pipeline/terrain/dados/caop.json, que o build_tiles.py lê.

    pip install pyogrio
    python pipeline/terrain/caop.py            # descarrega para .cache/caop e grava dados/caop.json

Usam-se os troços (cont_trocos, ram_trocos, raa_*_trocos), e não os contornos dos polígonos: cada troço é uma
linha partilhada por duas áreas, desenhada uma só vez, por isso não há falhas nem sobreposições entre distritos.
O nível do troço é o mais alto dos limites que ele separa:
    1ª Ordem  fronteira com Espanha
    3ª Ordem  distrito (Continente) ou ilha (regiões autónomas)
    4ª Ordem  município
    5ª Ordem  freguesia (não se usa)
Só entram os troços "Limite em Terra" (e "Limite na Água" nos rios de fronteira); os de costa vêm do Natural Earth.

Saída em graus (ETRS89/PTRA08 ≈ WGS84 à escala do mapa), com 5 casas decimais (c. 1 m), simplificada a 20 m (40 m nos municípios).
"""
import json
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pyogrio
from pyproj import Transformer
from shapely import from_wkb
from shapely.geometry import box, mapping
from shapely.ops import linemerge, polylabel, unary_union
from shapely import transform

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("CASTELOS_CACHE", ROOT / ".cache")) / "caop"
SAIDA = Path(__file__).resolve().parent / "dados" / "caop.json"
ANO = "2025"
URL = "https://geo2.dgterritorio.gov.pt/caop/CAOP_{}_{}-gpkg.zip"

# ficheiro, prefixo das camadas
FONTES = {
    "Continente": [("Continente_CAOP2025.gpkg", "cont")],
    "RAA": [("ArqAcores_GCentral_GOriental_CAOP2025.gpkg", "raa_cen_ori"), ("ArqAcores_GOcidental_CAOP2025.gpkg", "raa_oci")],
    "RAM": [("ArqMadeira_CAOP2025.gpkg", "ram")],
}

# nome no mapa de cada valor de distrito_ilha (as Desertas e as Selvagens são áreas secundárias de Santa Cruz e do
# Funchal: separam-se da Ilha da Madeira pela posição)
ILHAS = {
    "Ilha de Santa Maria": "Santa Maria", "Ilha de São Miguel": "São Miguel", "Ilha Terceira": "Terceira",
    "Ilha da Graciosa": "Graciosa", "Ilha de São Jorge": "São Jorge", "Ilha do Pico": "Pico", "Ilha do Faial": "Faial",
    "Ilha das Flores": "Flores", "Ilha do Corvo": "Corvo", "Ilha da Madeira": "Madeira", "Ilha de Porto Santo": "Porto Santo",
}
GRUPOS_MADEIRA = {  # caixas em graus: lon0, lat0, lon1, lat1
    "Ilhas Desertas": (-16.62, 32.33, -16.40, 32.62),
    "Ilhas Selvagens": (-16.10, 29.95, -15.80, 30.20),
}


def descarregar():
    CACHE.mkdir(parents=True, exist_ok=True)
    for regiao, ficheiros in FONTES.items():
        if all((CACHE / f).exists() for f, _ in ficheiros):
            continue
        z = CACHE / f"{regiao}.zip"
        print("a descarregar", URL.format(regiao, ANO))
        urlretrieve(URL.format(regiao, ANO), z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(CACHE)
        z.unlink()


def ler(ficheiro, camada):
    """Lista de (propriedades, geometria shapely) e o CRS da camada."""
    meta, _, geoms, campos = pyogrio.raw.read(CACHE / ficheiro, layer=camada)
    nomes = meta["fields"].tolist()
    linhas = [({n: campos[i][k] for i, n in enumerate(nomes)}, from_wkb(g)) for k, g in enumerate(geoms)]
    return linhas, meta["crs"]


def graus(geom, crs):
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return transform(geom, lambda c: np.column_stack(t.transform(c[:, 0], c[:, 1])))


def linhas_json(geom):
    if geom.is_empty:
        return []
    partes = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
    return [[[round(x, 5), round(y, 5)] for x, y in p.coords] for p in partes if len(p.coords) >= 2]


def ponto_json(p):
    return [round(p.x, 5), round(p.y, 5)]


def caixa_json(g):
    return [round(v, 5) for v in g.bounds]


def main():
    descarregar()
    out = {
        "fonte": f"CAOP {ANO}, Direção-Geral do Território (https://www.dgterritorio.gov.pt/cartografia/cartografia-tematica/caop)",
        "fronteira": [], "distritos": [], "concelhos": [], "nomesDistritos": [], "ilhas": [], "contorno": [],
    }
    contorno = []
    ilhas = {}
    for regiao, ficheiros in FONTES.items():
        for ficheiro, pre in ficheiros:
            trocos, crs = ler(ficheiro, f"{pre}_trocos")
            por_nivel = {"1ª Ordem": [], "3ª Ordem": [], "4ª Ordem": []}
            for p, g in trocos:
                if p["significado_linha"] == "Limite e Linha de Costa":
                    continue
                if p["nivel_limite_admin"] in por_nivel:
                    por_nivel[p["nivel_limite_admin"]].append(g)
            for nivel, chave in (("1ª Ordem", "fronteira"), ("3ª Ordem", "distritos"), ("4ª Ordem", "concelhos")):
                if not por_nivel[nivel]:
                    continue
                g = linemerge(unary_union(por_nivel[nivel])).simplify(40 if chave == "concelhos" else 20)
                out[chave] += linhas_json(graus(g, crs))

            # contorno de Portugal (terra), para o relevo e a costa baterem certo com os limites
            for p, g in ler(ficheiro, f"{pre}_municipios")[0]:
                contorno.append(graus(g.buffer(1).simplify(30), crs))

            if regiao == "Continente":
                # nome de cada distrito no ponto mais afastado dos seus limites (não cai num canto nem fora dele)
                for p, g in ler(ficheiro, f"{pre}_distritos")[0]:
                    maior = max(g.geoms, key=lambda x: x.area) if g.geom_type == "MultiPolygon" else g
                    c = polylabel(maior, tolerance=200)
                    out["nomesDistritos"].append({"nome": p["distrito"], "lonlat": ponto_json(graus(c, crs))})
                continue

            # ilhas: os municípios agrupados pelo campo distrito_ilha
            for p, g in ler(ficheiro, f"{pre}_municipios")[0]:
                ilhas.setdefault(p["distrito_ilha"], []).append(graus(g, crs))

    for nome_caop, gs in ilhas.items():
        u = unary_union(gs)
        partes = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
        if nome_caop == "Ilha da Madeira":
            for grupo, cx in GRUPOS_MADEIRA.items():
                sel = [q for q in partes if q.intersects(box(*cx))]
                g = unary_union(sel)
                out["ilhas"].append({"nome": grupo, "arquipelago": "Madeira", "lonlat": ponto_json(g.centroid), "caixa": caixa_json(g)})
            partes = [q for q in partes if not any(q.intersects(box(*cx)) for cx in GRUPOS_MADEIRA.values())]
        # a ilha principal (sem os ilhéus) dá o centro; a caixa inclui tudo
        g = unary_union(partes)
        maior = max(partes, key=lambda x: x.area)
        arq = "Madeira" if nome_caop in ("Ilha da Madeira", "Ilha de Porto Santo") else "Açores"
        out["ilhas"].append({"nome": ILHAS[nome_caop], "arquipelago": arq, "lonlat": ponto_json(maior.representative_point()), "caixa": caixa_json(g)})

    u = unary_union(contorno)
    polis = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
    for q in polis:
        if q.area < 1e-7:  # ilhéus com menos de c. 1 ha
            continue
        anel = lambda r: [[round(x, 5), round(y, 5)] for x, y in r.coords]
        # sem buracos: os que aparecem são frinchas de milímetros entre municípios vizinhos (Portugal não tem enclaves)
        out["contorno"].append([anel(q.exterior)])

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    n = {k: (len(v), sum(len(l) for l in v)) for k, v in out.items() if k in ("fronteira", "distritos", "concelhos")}
    print(f"{SAIDA.relative_to(ROOT)}: {SAIDA.stat().st_size / 1024:.0f} KB; linhas e vértices {n}; "
          f"{len(out['nomesDistritos'])} distritos; {len(out['ilhas'])} ilhas; contorno com {len(out['contorno'])} polígonos")


if __name__ == "__main__":
    main()
