"""Texturas fotográficas: pedra, telha e rocha fotografadas (Poly Haven, CC0), com a cor de cada monumento.

As fotos estão em pipeline/kit/fotos/ (obter_fotos.py descarrega-as e regista as fontes em fotos/fontes.json).
`textura_foto` devolve o mesmo par (cor, relevo) que `kit.texturas.textura_cantaria`, numa imagem de n x n px
que cobre `metros` de lado, por isso entra no lugar dela sem mudar UV nem materiais:

    TEX, ALT = textura_foto("old_stone_wall", cor=(0.60, 0.58, 0.54))
    material("muro", (1, 1, 1), 0.92, imagem("muro", TEX), normal=imagem("muro_n", normais(ALT, 3.5)), ...)

A foto dá o desenho real das pedras, das juntas e do grão; a cor média vem do monumento (medida nas
fotografias dele), para a pedra não ficar com o tom do sítio onde a foto foi tirada. Por cima juntam-se
variações largas de tom e, se pedido, manchas de líquen, ambas periódicas para a imagem repetir sem costura.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

__all__ = ["textura_foto", "fontes_fotos", "FONTES"]

PASTA = Path(__file__).resolve().parent / "fotos"
FONTES = json.loads((PASTA / "fontes.json").read_text()) if (PASTA / "fontes.json").exists() else {}
USADAS = set()  # ids usados nesta maqueta (para os créditos)


def fontes_fotos():
    """Créditos das fotos usadas na maqueta atual, para gravar em medidas.json ou no LEIAME."""
    return {t: FONTES[t] for t in sorted(USADAS)}


def _ler(tid):
    cor = np.asarray(Image.open(PASTA / f"{tid}_cor.jpg").convert("RGB"), float) / 255.0
    rel = np.asarray(Image.open(PASTA / f"{tid}_relevo.png").convert("L"), float) / 255.0
    return cor ** 2.2, rel  # cor linear para as médias; volta a sRGB no fim


def _ruido_periodico(n, celulas, semente):
    """Ruído suave com período n (grelha aleatória de celulas x celulas ampliada com repetição)."""
    g = np.random.default_rng(semente).random((celulas, celulas))
    z = ndi.zoom(np.tile(g, (3, 3)), n / celulas, order=3, mode="grid-wrap")
    return z[n:2 * n, n:2 * n]


def _fbm_periodico(n, semente, base=4, oit=4):
    s, amp, tot = np.zeros((n, n)), 1.0, 0.0
    for o in range(oit):
        s += amp * _ruido_periodico(n, base * 2 ** o, semente + o)
        tot += amp
        amp *= 0.5
    return s / tot


def textura_foto(tid, n=1024, metros=6.0, cor=(0.60, 0.58, 0.54), contraste=1.0, saturacao=0.5, metros_foto=None,
                 variacao=0.10, liquen=0.0, liquen_cor=(0.84, 0.84, 0.80), semente=7, rodar=False):
    """Foto `tid` repetida em n x n px = `metros` x `metros` (o que a imagem cobre de facto na maqueta: UV e
    KHR_texture_transform incluídos), recolorida para a cor média `cor` (sRGB 0..1).

    metros_foto: tamanho real da foto (por omissão o de fontes.json); acerta-se para caber um número inteiro
    de vezes em `metros`, para a imagem final repetir sem costura. contraste > 1 acentua pedras e juntas.
    saturacao: quanto da cor própria de cada pedra (desvio à média) se mantém (0 = só a cor dada).
    variacao: amplitude das manchas largas de tom (pedra mais clara ou mais escura por zonas).
    liquen: fração aproximada da superfície com manchas claras de líquen (0 = nenhuma).
    rodar: roda a foto 90° (fiadas verticais passam a horizontais)."""
    USADAS.add(tid)
    foto, rel = _ler(tid)
    if rodar:
        foto, rel = np.rot90(foto), np.rot90(rel)
    mf = metros_foto or FONTES.get(tid, {}).get("metros", [2.0])[0]
    k = max(1, int(round(metros / mf)))  # repetições da foto em `metros`
    lado = n // k
    f = np.asarray(Image.fromarray((foto ** (1 / 2.2) * 255).astype(np.uint8)).resize((lado, lado), Image.LANCZOS), float) / 255.0
    f = f ** 2.2
    r = np.asarray(Image.fromarray((rel * 255).astype(np.uint8)).resize((lado, lado), Image.LANCZOS), float) / 255.0
    f, r = np.tile(f, (k, k, 1)), np.tile(r, (k, k))
    if k * lado != n:  # n não divisível por k: estica 1–2 px para a imagem repetir sem costura
        f = ndi.zoom(f, (n / f.shape[0], n / f.shape[1], 1), order=1, mode="grid-wrap")
        r = ndi.zoom(r, n / r.shape[0], order=1, mode="grid-wrap")
    # luminância e cor própria de cada pixel, relativas à média da foto
    lum = f @ np.array([0.2126, 0.7152, 0.0722])
    # tira os defeitos da foto que se notariam a repetir (buracos negros, uma pedra branca, etiquetas):
    # a luminância fica entre os percentis 1,5 e 98,5
    # buracos escuros e lascas claras mais pequenos do que uma pedra: luminância abaixo de 35 % ou acima de 160 % da mediana local (as juntas ficam)
    # (janela de c. 6 % da foto), toma-se a cor da mediana local
    med = np.stack([ndi.median_filter(f[..., c], size=max(5, lado // 16), mode="wrap") for c in range(3)], -1)
    lmed = med @ np.array([0.2126, 0.7152, 0.0722])
    buraco = ndi.binary_dilation((lum < 0.35 * lmed) | (lum > 1.6 * lmed), iterations=1)[..., None]
    f = np.where(buraco, med, f)
    lum = f @ np.array([0.2126, 0.7152, 0.0722])
    lo, hi = np.quantile(lum, [0.015, 0.985])
    f = f * (np.clip(lum, lo, hi) / np.maximum(lum, 1e-4))[..., None]
    lum = np.clip(lum, lo, hi)
    lm = lum.mean()
    croma = f / np.maximum(lum[..., None], 1e-4)
    croma = 1 + (croma / croma.reshape(-1, 3).mean(0) - 1) * saturacao
    alvo = np.array(cor, float) ** 2.2
    tex = alvo * ((lum / lm) ** contraste)[..., None] * croma
    if variacao:
        v = _fbm_periodico(n, semente, base=3)
        tex *= (1 + variacao * 2 * (v - 0.5))[..., None]
    if liquen:
        # manchas pequenas e salpicadas (as fotos dos castelos mostram líquen em pintas, não em placas)
        m = _fbm_periodico(n, semente + 11, base=int(metros * 6), oit=3)
        lim = np.quantile(m, 1 - liquen)
        a = np.clip((m - lim) / 0.03, 0, 1)[..., None] * 0.40
        tex = tex * (1 - a) + (np.array(liquen_cor) ** 2.2) * (lum / lm)[..., None] * a
    tex = np.clip(tex, 0, 1) ** (1 / 2.2)
    return tex, ndi.gaussian_filter(r, 0.7)
