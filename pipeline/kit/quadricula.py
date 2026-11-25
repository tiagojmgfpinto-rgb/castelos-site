"""Folhas de quadrícula para ver o chão "metro a metro" (novembro de 2026).

Corta a ortofoto (e, se houver, a planta ortogonal da maqueta e o painel de alturas do LiDAR) em quadrículas
com a grelha da cena desenhada (linhas de 5 m, rótulos de 10 m em metros da cena: x para este, y para norte),
com o contorno das classes do chão por cima. Serve para traçar as correções à mão e para comparar a maqueta
com a ortofoto quadrícula a quadrícula, sem precisar do Blender.

    python pipeline/kit/quadricula.py MAQUETA [--lado 40] [--px 12] [--classes] [--planta vistas/planta.png]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CORES = {"asfalto": (40, 40, 48), "calcada": (230, 230, 230), "lajeado": (200, 190, 170), "escadas": (255, 120, 0),
         "terra": (190, 130, 60), "saibro": (235, 215, 160), "relva": (60, 200, 60), "erva": (170, 190, 90), "mato": (20, 110, 30), "rocha": (150, 150, 170),
         "edificio": (255, 0, 0), "agua": (0, 120, 255)}


def fonte(t=12):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, t)
    return ImageFont.load_default()


class Imagem:
    """Imagem georreferenciada na cena: `meio` m para cada lado da origem, linha 0 = norte."""

    def __init__(self, img, meio, centro=(0.0, 0.0)):
        self.img = img.convert("RGB")
        self.meio = meio
        self.cx, self.cy = centro
        self.k = self.img.size[0] / (2 * meio)  # px por metro

    def recorte(self, x0, y0, lado, px):
        k = self.k
        l = (x0 - (self.cx - self.meio)) * k
        t = ((self.cy + self.meio) - (y0 + lado)) * k
        r = l + lado * k
        b = t + lado * k
        return self.img.crop((int(round(l)), int(round(t)), int(round(r)), int(round(b)))).resize(
            (int(lado * px), int(lado * px)), Image.LANCZOS)


def grelha(im, x0, y0, lado, px, passo=5, rotulo=10, cor=(255, 255, 0)):
    d = ImageDraw.Draw(im)
    f = fonte(11)
    for v in np.arange(np.ceil(x0 / passo) * passo, x0 + lado + 1e-6, passo):
        X = (v - x0) * px
        d.line([(X, 0), (X, im.size[1])], fill=cor if v % rotulo == 0 else (cor[0] // 2, cor[1] // 2, 0), width=1)
        if v % rotulo == 0:
            d.text((X + 2, 2), f"x{v:.0f}", fill=cor, font=f)
    for v in np.arange(np.ceil(y0 / passo) * passo, y0 + lado + 1e-6, passo):
        Y = (y0 + lado - v) * px
        d.line([(0, Y), (im.size[0], Y)], fill=cor if v % rotulo == 0 else (cor[0] // 2, cor[1] // 2, 0), width=1)
        if v % rotulo == 0:
            d.text((2, Y + 2), f"y{v:.0f}", fill=cor, font=f)
    return im


def contornos(im, classes, x0, y0, lado, px, largura=2):
    import shapely
    d = ImageDraw.Draw(im)
    caixa = shapely.box(x0, y0, x0 + lado, y0 + lado)
    for n, g in classes.items():
        g = g.intersection(caixa)
        for p in shapely.get_parts(g):
            for anel in [p.exterior, *p.interiors] if hasattr(p, "exterior") else []:
                c = np.asarray(anel.coords)
                pts = [((x - x0) * px, (y0 + lado - y) * px) for x, y in c]
                if len(pts) > 1:
                    d.line(pts, fill=CORES.get(n, (255, 0, 255)), width=largura)
            if not p.is_empty and p.area > 6:
                q = p.representative_point()
                d.text(((q.x - x0) * px - 10, (y0 + lado - q.y) * px - 6), n[:4], fill=CORES.get(n, (255, 0, 255)),
                       font=fonte(10))
    return im


def folha(paineis, titulo, px):
    w, h = paineis[0].size
    out = Image.new("RGB", (w * len(paineis) + 8 * (len(paineis) - 1), h + 22), (20, 20, 20))
    for i, p in enumerate(paineis):
        out.paste(p, (i * (w + 8), 22))
    ImageDraw.Draw(out).text((4, 4), titulo, fill=(255, 255, 255), font=fonte(13))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maqueta")
    ap.add_argument("--lado", type=float, default=40)
    ap.add_argument("--px", type=float, default=12)
    ap.add_argument("--raio", type=float, default=None)
    ap.add_argument("--classes", action="store_true")
    ap.add_argument("--planta", default=None, help="planta ortogonal da maqueta (png) e o seu meio-lado: ficheiro:meio")
    ap.add_argument("--so", default=None, help="só a quadrícula x0,y0")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    pasta = Path(__file__).resolve().parents[1] / "maquetas" / a.maqueta
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    meta = json.loads((pasta / "dados" / "orto.json").read_text()) if (pasta / "dados" / "orto.json").exists() else {}
    meio = meta.get("meio", 160.0)
    orto = Imagem(Image.open(pasta / "dados" / "orto.jpg"), meio, tuple(meta.get("centro", (0.0, 0.0))))
    planta = None
    if a.planta:
        f, m = a.planta.split(":")
        planta = Imagem(Image.open(f), float(m))
    classes = {}
    if a.classes and (pasta / "dados" / "chao.json").exists():
        from kit.chao import ler
        classes = ler(pasta / "dados" / "chao.json")
    saida = Path(a.saida) if a.saida else pasta / "vistas" / "quadriculas"
    saida.mkdir(parents=True, exist_ok=True)
    raio = a.raio or meio
    L = a.lado
    xs = np.arange(-raio, raio, L)
    for y0 in xs[::-1]:
        for x0 in xs:
            if a.so and (x0, y0) != tuple(map(float, a.so.split(","))):
                continue
            if np.hypot(x0 + L / 2, y0 + L / 2) > raio + L * 0.7:
                continue
            ps = []
            o = orto.recorte(x0, y0, L, a.px)
            if classes:
                o2 = contornos(o.copy(), classes, x0, y0, L, a.px)
                ps.append(grelha(o, x0, y0, L, a.px))
                ps.append(grelha(o2, x0, y0, L, a.px))
            else:
                ps.append(grelha(o, x0, y0, L, a.px))
            if planta:
                ps.append(grelha(planta.recorte(x0, y0, L, a.px), x0, y0, L, a.px))
            nome = f"q_{x0:+04.0f}_{y0:+04.0f}"
            folha(ps, f"{a.maqueta}  x {x0:.0f}..{x0 + L:.0f}  y {y0:.0f}..{y0 + L:.0f}  (ortofoto | classes | maqueta)",
                  a.px).save(saida / f"{nome}.jpg", quality=85)
    print("quadrículas em", saida)


if __name__ == "__main__":
    main()
