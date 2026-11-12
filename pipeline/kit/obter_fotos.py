"""Descarrega as texturas fotográficas (Poly Haven, CC0) que o kit usa e grava-as em pipeline/kit/fotos/.

    python pipeline/kit/obter_fotos.py            # só as que faltam
    python pipeline/kit/obter_fotos.py --todas    # volta a descarregar todas

Para cada textura grava <id>_cor.jpg (cor difusa, 1024 px) e <id>_relevo.png (deslocamento, 1024 px, 8 bits),
e acrescenta a fonte a fotos/fontes.json (autores, licença, página, tamanho real em metros). Os ficheiros
ficam no repositório para que as maquetas se reconstruam sempre iguais, mesmo que a Poly Haven mude.
"""
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent / "fotos"
API = "https://api.polyhaven.com"

# id -> para que serve no kit
TEXTURAS = {
    "old_stone_wall": "alvenaria de pedra irregular (muralhas, cercas, muros)",
    "stone_brick_wall_001": "cantaria de silhares grandes e rústicos (torres, cubelos)",
    "seaworn_stone_tiles": "cantaria regular de fiadas baixas (portas, paramentos aparelhados)",
    "castle_wall_slates": "alvenaria de xisto e lajes (muros de xisto, Penela)",
    "roof_09": "telha de canudo velha (telhados dos monumentos)",
    "rock_01": "granito em afloramento (penedos e rocha à vista)",
}


def obter(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "castelos-site-pipeline"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> None:
    AQUI.mkdir(exist_ok=True)
    todas = "--todas" in sys.argv
    fontes_p = AQUI / "fontes.json"
    fontes = json.loads(fontes_p.read_text()) if fontes_p.exists() else {}
    for tid, uso in TEXTURAS.items():
        cor, rel = AQUI / f"{tid}_cor.jpg", AQUI / f"{tid}_relevo.png"
        if cor.exists() and rel.exists() and tid in fontes and not todas:
            continue
        info = json.loads(obter(f"{API}/info/{tid}"))
        fich = json.loads(obter(f"{API}/files/{tid}"))
        Image.open(io.BytesIO(obter(fich["Diffuse"]["1k"]["jpg"]["url"]))).convert("RGB").save(cor, quality=92)
        d = np.asarray(Image.open(io.BytesIO(obter(fich["Displacement"]["1k"]["png"]["url"]))), float)
        if d.ndim == 3:
            d = d[..., :3].mean(-1)
        d = (d - d.min()) / max(d.max() - d.min(), 1e-9)  # PNG de 8 ou 16 bits -> 0..1
        Image.fromarray((d * 255).round().astype(np.uint8)).save(rel, optimize=True)
        fontes[tid] = {
            "titulo": info["name"],
            "autores": sorted(info.get("authors", {})),
            "licenca": "CC0 1.0 (domínio público)",
            "pagina": f"https://polyhaven.com/a/{tid}",
            "metros": [d / 1000 for d in info.get("dimensions", [2000, 2000])],
            "uso": uso,
        }
        print(tid, fontes[tid]["metros"], fontes[tid]["autores"])
    fontes_p.write_text(json.dumps(fontes, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
