"""Láminas foto contra maqueta e medição de cores na vitrina.

Uma lámina por vista de controlo, em grelha 2 × 2 e ampliada:
    foto           | maqueta (vitrina)
    mistura a 50 % | contorno da maqueta (vermelho) sobre a foto
mais, para cada recorte pedido em `zooms`, uma faixa foto | maqueta ampliada (as comprovações de perto).

Cores: para cada material das condições com amostras, a cor média (sRGB) na maqueta e nas fotos e a diferença
ΔE2000 (até ~3 não se nota, 3–6 nota-se lado a lado, acima de 10 é outra cor).

Uso (normalmente chamado pelo volta.py):
    python pipeline/sistema/laminas.py CONDICOES.json PASTA_VITRINA PASTA_SAIDA [--refs PASTA_REFS]
As fotos das vistas são caminhos relativos à pasta refs/<castelo>/ (a pasta do condicoes.json, por omissão).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FONTE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def fonte(t: int):
    try:
        return ImageFont.truetype(FONTE, t)
    except OSError:
        return ImageFont.load_default(size=t)


def ajustar_foto(foto: Image.Image, w: int, h: int, recorte=None) -> Image.Image:
    """Recorta a foto (recorte em píxeis da foto, opcional) ao formato da vista e redimensiona-a."""
    if recorte:
        foto = foto.crop(tuple(recorte))
    fw, fh = foto.size
    alvo = w / h
    if fw / fh > alvo:
        nw = round(fh * alvo)
        foto = foto.crop(((fw - nw) // 2, 0, (fw - nw) // 2 + nw, fh))
    elif fw / fh < alvo:
        nh = round(fw / alvo)
        foto = foto.crop((0, (fh - nh) // 2, fw, (fh - nh) // 2 + nh))
    return foto.resize((w, h), Image.LANCZOS)


def contorno(foto: Image.Image, mascara: Image.Image, cor=(230, 30, 30), largura=2) -> Image.Image:
    """Contorno da silhueta da obra (máscara branca) desenhado sobre a foto."""
    m = mascara.convert("L").point(lambda v: 255 if v > 127 else 0)
    borda = ImageChops.difference(m, m.filter(ImageFilter.MinFilter(3)))
    if largura > 1:
        borda = borda.filter(ImageFilter.MaxFilter(2 * (largura // 2) + 1))
    out = foto.convert("RGB").copy()
    out.paste(Image.new("RGB", out.size, cor), mask=borda)
    return out


def rotulo(img: Image.Image, texto: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    f = fonte(max(14, img.width // 45))
    x0, y0, x1, y1 = d.textbbox((10, 8), texto, font=f)
    d.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(20, 20, 20))
    d.text((10, 8), texto, font=f, fill=(255, 255, 255))
    return img


def lamina(vista: dict, foto: Image.Image | None, render: Image.Image, mascara: Image.Image | None, condicoes: list[dict]) -> Image.Image:
    w, h = render.size
    render = render.convert("RGB")
    if foto is None:
        foto = Image.new("RGB", (w, h), (90, 90, 90))
        rotulo(foto, "SEM FOTO")
    mistura = Image.blend(foto, render, 0.5)
    cont = contorno(foto, mascara) if mascara is not None else foto.copy()
    paineis = [rotulo(foto.copy(), "Foto"), rotulo(render.copy(), "Maqueta (vitrina)"), rotulo(mistura, "Mistura 50 %"), rotulo(cont, "Contorno da maqueta sobre a foto")]
    cab = 56
    rodape = 26 * max(1, len(condicoes)) + 16
    out = Image.new("RGB", (2 * w + 12, 2 * h + 12 + cab + rodape), (245, 242, 235))
    for i, p in enumerate(paineis):
        out.paste(p, ((i % 2) * (w + 12), cab + (i // 2) * (h + 12)))
    d = ImageDraw.Draw(out)
    titulo = f"{vista['id']} · {vista.get('nome', '')}"
    if vista.get("fonte"):
        titulo += f"  —  foto: {vista['fonte']}"
    d.text((10, 12), titulo, font=fonte(26), fill=(20, 20, 20))
    y = cab + 2 * h + 12 + 8
    for c in condicoes:
        d.text((10, y), f"{c['id']} [{c['tipo']}] {c['texto']}"[:220], font=fonte(18), fill=(120, 20, 20) if c["tipo"] == "G" else (60, 60, 60))
        y += 26
    return out


def zooms(vista: dict, foto: Image.Image, render: Image.Image) -> list[Image.Image]:
    """Recortes [x0, y0, x1, y1] (píxeis da vista) ampliados lado a lado: foto | maqueta."""
    out = []
    for i, z in enumerate(vista.get("zooms", [])):
        a, b = foto.crop(tuple(z)), render.convert("RGB").crop(tuple(z))
        esc = max(1, 700 // max(1, a.height))
        a = a.resize((a.width * esc, a.height * esc), Image.LANCZOS)
        b = b.resize(a.size, Image.LANCZOS)
        img = Image.new("RGB", (a.width * 2 + 12, a.height), (245, 242, 235))
        img.paste(rotulo(a, f"{vista['id']} perto {i + 1}: foto"), (0, 0))
        img.paste(rotulo(b, "maqueta"), (a.width + 12, 0))
        out.append(img)
    return out


# ---------- cor ----------

def srgb_para_lab(rgb) -> np.ndarray:
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e2000(lab1, lab2) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm**7 / (Cm**7 + 25**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    if C1p * C2p == 0:
        dh = 0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hpm = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hpm = (h1p + h2p) / 2
    else:
        hpm = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    T = 1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm)) + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63))
    dth = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm**7 / (Cpm**7 + 25**7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return float(np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh)))


def media(img: Image.Image, caixa) -> list[int]:
    a = np.asarray(img.convert("RGB").crop(tuple(caixa)), float).reshape(-1, 3)
    return [round(v) for v in np.median(a, 0)]  # mediana: ignora juntas escuras e reflexos


def hexa(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def cores(cond: dict, vitrina: Path, refs: Path) -> list[dict]:
    out = []
    for m in cond.get("materiais", []):
        am = m.get("amostra_maqueta")
        if not am:
            continue
        rp = vitrina / f"{am['vista']}.png"
        if not rp.exists():
            continue
        maq = media(Image.open(rp), am["caixa"])
        fotos = [media(Image.open(refs / a["foto"]), a["caixa"]) for a in m.get("medido_em", []) if (refs / a["foto"]).exists()]
        alvo = m.get("cor_alvo")
        alvo_rgb = [int(alvo[i:i + 2], 16) for i in (1, 3, 5)] if alvo else (np.mean(fotos, 0).tolist() if fotos else None)
        r = {"id": m["id"], "nome": m.get("nome", ""), "maqueta": hexa(maq), "alvo": hexa(alvo_rgb) if alvo_rgb else None,
             "fotos": [hexa(f) for f in fotos]}
        if alvo_rgb:
            r["dE2000"] = round(delta_e2000(srgb_para_lab(maq), srgb_para_lab(alvo_rgb)), 1)
            tol = m.get("tolerancia_dE", 8)
            r["tolerancia_dE"] = tol
            r["cumpre"] = r["dE2000"] <= tol
        out.append(r)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    cpath, vitrina, saida = Path(argv[0]), Path(argv[1]), Path(argv[2])
    refs = Path(argv[argv.index("--refs") + 1]) if "--refs" in argv else cpath.parent
    cond = json.loads(cpath.read_text())
    saida.mkdir(parents=True, exist_ok=True)
    feitas = []
    for v in cond.get("vistas", []):
        rp = vitrina / f"{v['id']}.png"
        if not rp.exists():
            print(f"{v['id']}: sem render na vitrina")
            continue
        render = Image.open(rp)
        mp = vitrina / f"{v['id']}_mascara.png"
        mascara = Image.open(mp) if mp.exists() else None
        foto = None
        if v.get("foto") and (refs / v["foto"]).exists():
            foto = ajustar_foto(Image.open(refs / v["foto"]).convert("RGB"), *render.size, v.get("recorte"))
        cs = [c for c in cond.get("condicoes", []) if v["id"] in c.get("vistas", [])]
        lamina(v, foto, render, mascara, cs).save(saida / f"{v['id']}.jpg", quality=88)
        feitas.append(v["id"])
        if foto is not None:
            for i, z in enumerate(zooms(v, foto, render)):
                z.save(saida / f"{v['id']}_perto{i + 1}.jpg", quality=90)
    cs = cores(cond, vitrina, refs)
    (saida / "cores.json").write_text(json.dumps(cs, indent=1, ensure_ascii=False))
    print(f"{len(feitas)} láminas em {saida}; {len(cs)} cores medidas")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
