"""Textura do chão de São Sebastião sem nenhum píxel de fotografia aérea: cada célula leva o material da sua classe
(dados/chao_classes.png, feito por chao.py a partir de traços à mão na ortofoto e das classes do LiDAR), com texturas
nítidas e repetidas à escala real e fronteiras alisadas a 5,7 cm. Cobre x -116..116, y -138..94 (232 m) em n x n px.
Devolve sRGB 0..1 com as linhas de baixo para cima (y para cima), como o kit.texturas.imagem espera."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from kit.fotos import textura_foto, _fbm_periodico

X0, Y0, L = -116.0, -138.0, 232.0
CL = ['asfalto', 'passeio', 'relva', 'mato', 'terra', 'branco', 'tijoleira', 'betao', 'rocha', 'asfalto2', 'calcada', 'muro']
# cores médias (sRGB) medidas na ortofoto compensada e tiradas do amarelado do voo
COR = {
    'asfalto': (0.30, 0.305, 0.30),   # estrada (ortofoto 0,35/0,36/0,31)
    'asfalto2': (0.40, 0.395, 0.37),  # parque de poente, asfalto gasto mais claro (0,39/0,38/0,31)
    'passeio': (0.58, 0.575, 0.55),   # passeios de betão e lancis claros (0,46/0,47/0,39)
    'calcada': (0.66, 0.64, 0.59),    # lajeado claro do terreiro da Pousada
    'relva': (0.36, 0.42, 0.22),
    'mato': (0.25, 0.31, 0.17),
    'terra': (0.47, 0.40, 0.36),      # terra batida castanho-arroxeada dos caminhos do recinto (ortofoto R-G c. +10)
    'tijoleira': (0.52, 0.37, 0.30),   # adarves (fotosfera dentro da Pousada), menos vermelha do que a 0.4
    'betao': (0.57, 0.52, 0.47),      # chão do estaleiro, betão manchado de ferrugem (0,53/0,51/0,43)
    'rocha': (0.23, 0.23, 0.22),      # basalto da costa
    'muro': (0.38, 0.37, 0.35),
    'branco': (0.86, 0.86, 0.83),     # tinta das passadeiras e linhas; cascos dos barcos
}
LANCIL = (0.60, 0.59, 0.56)


def _tile_ruido(n, metros, cor, fino=0.10, largo=0.06, semente=1):
    """Superfície lisa (asfalto, betão, tinta): grão fino e manchas largas, sem padrão de foto."""
    rng = np.random.default_rng(semente)
    g = ndi.gaussian_filter(rng.random((n, n)), 0.7, mode='wrap')
    g = (g - g.mean()) / (g.std() + 1e-6)
    v = _fbm_periodico(n, semente + 3, base=max(2, int(metros / 3)))
    lin = np.array(cor) ** 2.2 * (1 + fino * g)[..., None] * (1 + largo * 2 * (v - 0.5))[..., None]
    return np.clip(lin, 0, 1) ** (1 / 2.2)


def _tile_relva(n, metros, cor, semente=7):
    rng = np.random.default_rng(semente)
    v = _fbm_periodico(n, semente, base=max(2, int(metros / 1.5)))
    t = np.clip((v - 0.5) * 1.2, -1, 1)[..., None]
    base = np.array(cor) ** 2.2
    seca = np.array((0.50, 0.48, 0.30)) ** 2.2
    lin = base * (1 - np.clip(t, 0, 1) * 0.5) + seca * np.clip(t, 0, 1) * 0.5
    lin = lin * (1 - np.clip(-t, 0, 1) * 0.15)
    g = ndi.gaussian_filter(rng.random((n, n)), 0.6, mode='wrap')
    lin = lin * (0.82 + 0.36 * (g - g.min()) / (np.ptp(g) + 1e-6))[..., None]
    return np.clip(lin, 0, 1) ** (1 / 2.2)


def _tiles(px_m):
    """Cada material como (array n x n sRGB, metros que cobre)."""
    T = {}
    T['asfalto'] = (_tile_ruido(512, 8.0, COR['asfalto'], 0.12, 0.08, 11), 8.0)
    T['asfalto2'] = (_tile_ruido(512, 8.0, COR['asfalto2'], 0.12, 0.12, 12), 8.0)
    T['passeio'] = (textura_foto('seaworn_stone_tiles', n=512, metros=4.0, cor=COR['passeio'], contraste=0.55, saturacao=0.1,
                                 variacao=0.06, semente=13)[0], 4.0)
    T['calcada'] = (textura_foto('seaworn_stone_tiles', n=512, metros=4.0, cor=COR['calcada'], contraste=0.7, saturacao=0.15,
                                 variacao=0.05, semente=14)[0], 4.0)
    T['relva'] = (_tile_relva(512, 8.0, COR['relva'], 15), 8.0)
    T['mato'] = (textura_foto('rock_01', n=512, metros=4.0, cor=COR['mato'], contraste=1.2, saturacao=0.1, variacao=0.15,
                              semente=16)[0], 4.0)
    T['terra'] = (textura_foto('rock_01', n=512, metros=3.0, cor=COR['terra'], contraste=0.6, saturacao=0.25, variacao=0.10,
                               semente=17)[0], 3.0)
    T['tijoleira'] = (textura_foto('seaworn_stone_tiles', n=512, metros=3.0, cor=COR['tijoleira'], contraste=0.6,
                                   saturacao=0.2, variacao=0.06, semente=18)[0], 3.0)
    T['betao'] = (_tile_ruido(512, 8.0, COR['betao'], 0.08, 0.16, 19), 8.0)
    T['rocha'] = (textura_foto('rock_01', n=512, metros=3.0, cor=COR['rocha'], contraste=1.15, saturacao=0.2, semente=20)[0], 3.0)
    T['muro'] = (textura_foto('old_stone_wall', n=512, metros=4.0, cor=COR['muro'], contraste=1.0, saturacao=0.2, semente=21)[0],
                 4.0)
    T['branco'] = (_tile_ruido(256, 4.0, COR['branco'], 0.04, 0.04, 22), 4.0)
    return T


def textura_chao(dados, n=4096):
    dados = Path(dados)
    k = np.asarray(Image.open(dados / 'chao_classes.png'))[::-1]          # linha 0 = y -168, coluna 0 = x -200, 25 cm
    p = L / n
    c = (np.arange(n) + 0.5) * p
    xs, ys = X0 + c, Y0 + c
    ci, ri = (xs + 200) / 0.25 - 0.5, (ys + 168) / 0.25 - 0.5
    # janela das classes que interessa (com margem) e máscaras suaves de cada classe, argmax à resolução final
    r0, r1 = int(ri[0]) - 4, int(ri[-1]) + 5
    c0, c1 = int(ci[0]) - 4, int(ci[-1]) + 5
    kw = k[r0:r1, c0:c1]
    RR, CC = np.meshgrid(ri - r0, ci - c0, indexing='ij')
    best = np.full((n, n), -1.0, np.float32)
    lab = np.zeros((n, n), np.uint8)
    for i in range(len(CL)):
        m = kw == i
        if not m.any():
            continue
        s = ndi.gaussian_filter(m.astype(np.float32), 0.9 if CL[i] != 'branco' else 0.5)
        v = ndi.map_coordinates(s, [RR, CC], order=1, mode='nearest').astype(np.float32)
        sel = v > best
        best[sel] = v[sel]
        lab[sel] = i
    del RR, CC, best
    out = np.zeros((n, n, 3), np.float32)
    T = _tiles(n / L)
    YY, XX = np.meshgrid(ys, xs, indexing='ij')
    for i, nome in enumerate(CL):
        sel = lab == i
        if not sel.any():
            continue
        t, m = T[nome]
        tn = t.shape[0]
        iy = ((YY[sel] % m) / m * tn).astype(int) % tn
        ix = ((XX[sel] % m) / m * tn).astype(int) % tn
        out[sel] = t[iy, ix]
    # manchas largas que não se repetem (a repetição de 8 m da relva via-se em planta como um xadrez)
    rng = np.random.default_rng(5)
    ruido = ndi.gaussian_filter(rng.random((240, 240)), 4.0, mode='wrap')
    ruido = (ruido - ruido.mean()) / (ruido.std() + 1e-6)
    mz = ndi.map_coordinates(ruido, [(YY - Y0) * 240 / L, (XX - X0) * 240 / L], order=1, mode='wrap').astype(np.float32)
    amp = np.where(np.isin(lab, (CL.index('relva'), CL.index('mato'), CL.index('terra'))), 0.10,
                   np.where(lab == CL.index('branco'), 0.0, 0.05)).astype(np.float32)
    out *= np.clip(1 + amp * mz, 0.7, 1.3)[..., None]
    del mz, amp
    # relva e mato: tom seco ou verde da ortofoto, só a baixa frequência (3 m), a meio caminho da cor do material
    tinta = np.asarray(Image.open(dados / 'chao_tinta.png').convert('RGB'))[::-1].astype(np.float32) / 255.0  # 1 m, y -168
    tv = np.stack([ndi.map_coordinates(tinta[..., j], [(YY + 168) - 0.5, (XX + 200) - 0.5], order=1, mode='nearest')
                   for j in range(3)], -1)
    ref = np.array((0.31, 0.34, 0.23), np.float32)
    ganho = np.clip(tv / ref, 0.7, 1.45)
    ganho = ganho / ganho.mean(-1, keepdims=True) * np.clip(ganho.mean(-1, keepdims=True), 0.85, 1.2)
    rv = np.isin(lab, (CL.index('relva'), CL.index('mato')))
    out[rv] = np.clip(out[rv] * (0.5 + 0.5 * ganho[rv]), 0, 1)
    del tv, ganho, YY, XX
    # lancil claro de 15 cm onde o asfalto encosta a passeio, relva, mato, terra ou calçada
    asf = np.isin(lab, (CL.index('asfalto'), CL.index('asfalto2'), CL.index('branco')))
    perto = ndi.distance_transform_edt(~asf) * p <= 0.15
    viz = np.isin(lab, [CL.index(c_) for c_ in ('passeio', 'relva', 'mato', 'terra', 'calcada')])
    out[perto & viz] = LANCIL
    return out, lab


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    out, lab = textura_chao(Path(__file__).resolve().parent / 'dados', int(sys.argv[2]) if len(sys.argv) > 2 else 2048)
    Image.fromarray((out[::-1] * 255).astype(np.uint8)).save(sys.argv[1])
