"""Texturas procedimentais (ruído, fbm, cantaria, telha, relva, rocha, terra) e passagem de arrays para imagens do Blender.

Funções comuns às maquetas, tiradas sem alterações dos construir.py (ver pipeline/kit/LEIAME.md)."""
import math
import bpy
import numpy as np
from scipy import ndimage as ndi

__all__ = ["ruido", "fbm", "textura_cantaria", "textura_cubos", "textura_grao", "textura_relva", "textura_rocha", "textura_telha", "textura_terra", "imagem"]


def ruido(x, y, escala, semente):
    g = np.random.default_rng(semente).random((64, 64))
    u, v = np.asarray(x) / escala, np.asarray(y) / escala
    i, j = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i, v - j
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a, b = g[i % 64, j % 64], g[(i + 1) % 64, j % 64]
    c, d = g[i % 64, (j + 1) % 64], g[(i + 1) % 64, (j + 1) % 64]
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def fbm(x, y, escala, semente, oit=4):
    s, amp, tot = 0.0, 1.0, 0.0
    for o in range(oit):
        s = s + amp * ruido(x, y, escala / 2**o, semente + o)
        tot += amp
        amp *= 0.5
    return s / tot


def textura_cantaria(n=1024, metros=6.0, semente=1170, fiada=(0.30, 0.48), comp=(0.55, 1.6), liquen=0.80,
                     tinta=(1.0, 1.0, 1.0), paleta=None, prob=None, junta_m=0.005, irregular=0.0, var=(0.86, 1.12),
                     liq_forca=0.55):
    """Aparelho de granito em fiadas (6 m por repetição): silhares de fiada e comprimento dados, juntas finas,
    arestas gastas, grão de biotite e feldspato, manchas de óxido, escorrências e líquenes. Com irregular > 0
    as juntas verticais inclinam e as fiadas ondulam (alvenaria da cerca e da muralha). Devolve a cor e o relevo."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    if paleta is None:
        # granito de Melgaço nas fotografias: cinzento quente, dourado ao sol, com silhares mais claros e outros
        # tingidos de ocre (óxido de ferro)
        paleta = np.array([
            (0.600, 0.565, 0.500),  # cinzento-dourado (a maioria)
            (0.660, 0.630, 0.570),  # claro
            (0.500, 0.475, 0.430),  # escuro, gasto
            (0.640, 0.565, 0.460),  # ocre
            (0.575, 0.505, 0.420),  # ocre-acastanhado
            (0.560, 0.555, 0.535),  # cinzento frio
        ])
        prob = [0.40, 0.16, 0.16, 0.15, 0.05, 0.08]
    prob = np.array(prob) / np.sum(prob)
    cel = np.zeros((n, n), int)
    cores = []
    y = 0.0
    k = 0
    while y < n - 1:
        hf = r.uniform(*fiada) * m_px
        if y + hf > n - fiada[0] * 0.6 * m_px:
            hf = n - y
        y0, y1 = int(round(y)), int(round(y + hf))
        x = r.uniform(0, 1.0) * m_px
        x_ini = x
        while x < x_ini + n:
            wf = r.uniform(*comp) * m_px
            xs = (np.arange(int(round(x)), int(round(x + wf)))) % n
            cel[y0:y1, xs] = k
            cores.append(paleta[r.choice(len(paleta), p=prob)] * r.uniform(*var))
            k += 1
            x += wf
        y += hf
    cores = np.array(cores)
    yy, xx = np.mgrid[0:n, 0:n]
    amp = 0.04 + irregular
    wx = ((fbm(xx, yy, 0.2 * m_px, semente + 3) - 0.5) * amp * m_px).astype(int)
    wy = ((fbm(xx, yy, 0.2 * m_px, semente + 4) - 0.5) * (0.03 + irregular * 0.6) * m_px).astype(int)
    celw = cel[(yy + wy) % n, (xx + wx) % n]
    tex = cores[celw]
    borda = np.zeros((n, n), bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        borda |= celw != np.roll(np.roll(celw, dy, 0), dx, 1)
    dist = ndi.distance_transform_edt(~borda) / m_px
    junta = dist < junta_m
    gasto = np.clip(dist / (0.04 + irregular * 0.5), 0, 1)
    alt = np.sqrt(gasto) * (~junta)
    tex = tex * (0.86 + 0.14 * alt[..., None])
    tex = np.where(junta[..., None], tex * 0.55 + np.array([0.36, 0.35, 0.33]) * 0.45, tex)
    tex = tex * (0.92 + 0.16 * fbm(xx, yy, 0.30 * m_px, semente + 5))[..., None]
    g1 = ndi.gaussian_filter(r.random((n, n)), 0.6, mode="wrap")
    g1 = (g1 - g1.mean()) / g1.std()
    g2 = r.random((n, n))
    tex = tex * (1 + 0.05 * g1[..., None])
    tex = np.where((g2 < 0.045)[..., None], tex * 0.62, tex)
    tex = np.where((g2 > 0.93)[..., None], tex * 0.80 + 0.17, tex)
    ox = np.clip((fbm(xx, yy, 0.6 * m_px, semente + 8) - 0.58) * 3.0, 0, 1)[..., None]
    tex = tex * (1 - 0.30 * ox) + np.array([0.62, 0.50, 0.36]) * 0.30 * ox
    esc = fbm(xx * 6.0, yy * 0.25, 0.9 * m_px, semente + 9)
    esc = np.clip((esc - 0.55) * 2.2, 0, 1)[..., None]
    tex = tex * (1 - 0.22 * esc)
    # líquenes: manchas brancas acinzentadas (muito visíveis nas fotografias, sobretudo na torre e no alto
    # dos muros) e algumas amarelo-esverdeadas
    liq = fbm(xx, yy, 0.25 * m_px, semente + 6)
    tex = np.where((liq > liquen)[..., None], tex * (1 - liq_forca) + np.array([0.86, 0.86, 0.82]) * liq_forca, tex)
    liq2 = fbm(xx, yy, 0.18 * m_px, semente + 11)
    tex = np.where((liq2 > liquen + 0.04)[..., None], tex * 0.6 + np.array([0.66, 0.66, 0.42]) * 0.4, tex)
    return np.clip(tex * np.array(tinta), 0, 1), ndi.gaussian_filter(alt, 0.8) + 0.05 * g1


def textura_cubos(n=1024, metros=8.0, semente=91, base=(0.58, 0.57, 0.54)):
    """Calçada de cubos de granito (cubos de c. 10 cm, juntas de areia)."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    c = 0.105 * m_px
    gi = (xx // c).astype(int) + 1000 * ((yy + (xx // (3 * c)).astype(int) % 2 * c * 0.5) // c).astype(int)
    tom = r.uniform(0.85, 1.12, gi.max() + 1)[gi]
    fx, fy = (xx % c) / c, ((yy + (xx // (3 * c)).astype(int) % 2 * c * 0.5) % c) / c
    junta = (fx < 0.12) | (fy < 0.12)
    tex = np.array(base) * tom[..., None] * (0.9 + 0.2 * fbm(xx, yy, 1.5 * m_px, semente + 1))[..., None]
    tex = np.where(junta[..., None], np.array([0.40, 0.38, 0.35]), tex)
    return np.clip(tex, 0, 1), (~junta).astype(float)


def textura_grao(n=256):
    yy, xx = np.mgrid[0:n, 0:n] / n
    g = fbm(xx * 30, yy * 30, 2.0, 31)
    return np.repeat((0.84 + 0.22 * g)[..., None], 3, 2).clip(0, 1)


def textura_relva(n=512, metros=8.0, semente=7):
    """Relvado do recinto (fotografias: verde médio, com manchas amareladas e terra à vista nos carreiros)."""
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    g = fbm(xx, yy, 1.2 * m_px, semente)
    g2 = fbm(xx, yy, 0.15 * m_px, semente + 5)
    base = np.array([0.36, 0.46, 0.22])
    amarelo = np.array([0.55, 0.55, 0.30])
    t = np.clip((g - 0.45) * 2.5, 0, 1)[..., None]
    tex = base * (1 - t) + amarelo * t
    tex = tex * (0.85 + 0.3 * g2[..., None])
    pinta = np.random.default_rng(semente).random((n, n))
    tex = np.where((pinta > 0.9)[..., None], tex * 1.15, tex)
    return np.clip(tex, 0, 1), g2


def textura_rocha(n=512, metros=6.0, semente=13):
    """Afloramentos de granito do recinto: cinzento claro, gasto, com líquenes e fendas."""
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    g = fbm(xx, yy, 0.8 * m_px, semente)
    tex = np.array([0.60, 0.58, 0.54]) * (0.8 + 0.4 * g)[..., None]
    fenda = np.abs(fbm(xx, yy, 1.5 * m_px, semente + 3) - 0.5) < 0.012
    tex = np.where(fenda[..., None], tex * 0.55, tex)
    liq = fbm(xx, yy, 0.2 * m_px, semente + 6)
    tex = np.where((liq > 0.66)[..., None], tex * 0.5 + 0.42, tex)
    return np.clip(tex, 0, 1), g


def textura_telha(n=256, metros=2.0):
    yy, xx = np.mgrid[0:n, 0:n] / n * metros
    onda = 0.5 + 0.5 * np.cos(2 * math.pi * xx / 0.22)
    fiada = (yy % 0.42) / 0.42
    v = 0.78 + 0.22 * onda - 0.18 * (fiada > 0.9)
    g = fbm(xx * 8, yy * 8, 1.0, 77)
    v = v * (0.9 + 0.2 * g)
    return np.repeat(v[..., None], 3, 2).clip(0, 1), onda - 0.4 * (fiada > 0.9)


def textura_terra(n=512, metros=6.0, semente=23):
    m_px = n / metros
    yy, xx = np.mgrid[0:n, 0:n]
    g = fbm(xx, yy, 0.6 * m_px, semente)
    tex = np.array([0.60, 0.52, 0.42]) * (0.85 + 0.3 * g)[..., None]
    pedra = np.random.default_rng(semente).random((n, n)) > 0.985
    tex = np.where(ndi.binary_dilation(pedra, iterations=2)[..., None], tex * 1.15, tex)
    return np.clip(tex, 0, 1), g


def imagem(nome, arr):
    h, w, _ = arr.shape
    img = bpy.data.images.new(nome, w, h, alpha=False)
    rgba = np.concatenate([arr, np.ones((h, w, 1))], 2).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.file_format = "JPEG"
    img.pack()
    return img
