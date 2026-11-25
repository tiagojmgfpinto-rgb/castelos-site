"""Comparação automática antes/depois: marca a vermelho o que mudou entre duas séries de vistas da vitrina.

Serve para que nada mude sem querer na volta 2 (e para mostrar ao Tiago o que mudou numa maqueta já publicada).
Para cada imagem com o mesmo nome nas duas pastas grava antes | depois | mapa (depois em cinzento, vermelho onde
mudou) e a percentagem de píxeis mudados. O ruído da oclusão ambiental e das sombras é filtrado: conta só uma
diferença maior que LIMIAR num canal, em manchas de pelo menos 3 × 3 píxeis.

Uso:
    python pipeline/sistema/diferencas.py PASTA_ANTES PASTA_DEPOIS PASTA_SAIDA
Grava diferencas.json: {vista: percentagem}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from laminas import fonte

LIMIAR = 28


def mapa(antes: Image.Image, depois: Image.Image) -> tuple[Image.Image, float]:
    a = np.asarray(antes.convert("RGB"), np.int16)
    b = np.asarray(depois.convert("RGB"), np.int16)
    mud = (np.abs(a - b).max(2) > LIMIAR).astype(np.uint8) * 255
    m = Image.fromarray(mud).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(5))
    cinza = depois.convert("L").point(lambda v: 90 + v * 0.55).convert("RGB")
    cinza.paste(Image.new("RGB", cinza.size, (230, 20, 20)), mask=m)
    return cinza, float((np.asarray(m) > 0).mean() * 100)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    pa, pd, ps = map(Path, argv[:3])
    ps.mkdir(parents=True, exist_ok=True)
    res = {}
    for f in sorted(pd.glob("*.png")):
        if f.stem.endswith("_mascara") or not (pa / f.name).exists():
            continue
        antes, depois = Image.open(pa / f.name), Image.open(f)
        if antes.size != depois.size:
            antes = antes.resize(depois.size)
        mp, pct = mapa(antes, depois)
        w, h = depois.size
        out = Image.new("RGB", (3 * w + 24, h + 44), (245, 242, 235))
        for i, (img, t) in enumerate([(antes, "Antes (publicada)"), (depois, "Depois"), (mp, f"Mudou: {pct:.1f} % (vermelho)")]):
            out.paste(img.convert("RGB"), (i * (w + 12), 44))
            ImageDraw.Draw(out).text((i * (w + 12) + 8, 10), t, font=fonte(22), fill=(20, 20, 20))
        out.save(ps / f"{f.stem}.jpg", quality=85)
        res[f.stem] = round(pct, 2)
    (ps / "diferencas.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k}: {v:.1f} % mudou")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    sys.exit(main(sys.argv[1:]))
