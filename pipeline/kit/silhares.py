"""Silharia de granito em fiadas, com a cor e o grão de cada pedra (revisão do perito, etapa 3, 15 nov. 2026).

As texturas fotográficas (kit/fotos.py) têm dois defeitos que a revisão apontou em quase todas as maquetas:
- old_stone_wall é uma alvenaria de pedras redondas ("seixos") que não existe em Melgaço, Lanhoso nem Montalegre,
  onde os panos são silhares em fiadas de 0,40–0,55 m;
- stone_brick_wall_001 tem 2,5 m de lado: repetida, faz faixas horizontais a cada 2–2,5 m (Guimarães, Braga), e
  a recoloração por uma só cor média deixa todas as pedras iguais (desvio RGB 9–10 contra 35–59 nas fotos).

Aqui o desenho é das pedras e o grão é da foto: cada silhar tem a sua fiada, o seu comprimento, o seu tom (tirado de
uma paleta medida nas fotos do monumento) e um pedaço próprio da foto de granito (rock_01, com deslocamento aleatório
por pedra, para o grão não repetir). As juntas são finas e escuras; as arestas gastas ou almofadadas.

    TEX, ALT = textura_silhares(metros=6.0, fiada=(0.40, 0.55), comp=(0.7, 1.2), cor=(0.66, 0.63, 0.57))

devolve o mesmo par (cor sRGB 0..1, relevo) que textura_foto, e entra no lugar dela sem mudar UV nem materiais.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from kit.fotos import PASTA, USADAS

__all__ = ["textura_silhares"]

LUM = np.array([0.2126, 0.7152, 0.0722])


def _fbm_periodico(n, semente, base=4, oit=4):
    """Ruído fractal periódico (repete com período n), 0..1, por filtragem de ruído branco no domínio da
    frequência: muito mais rápido do que ampliar grelhas a 2048 px. base = número de manchas por lado."""
    r = np.random.default_rng(semente)
    f = np.fft.fftfreq(n)
    k2 = f[:, None] ** 2 + f[None, :] ** 2
    s, amp, tot = np.zeros((n, n)), 1.0, 0.0
    for o in range(oit):
        sig = base * 2 ** o / n  # frequência de corte
        w = np.fft.ifft2(np.fft.fft2(r.standard_normal((n, n))) * np.exp(-k2 / (2 * sig ** 2))).real
        s += amp * w / w.std()
        tot += amp
        amp *= 0.5
    s /= tot
    return np.clip(0.5 + s * 0.18, 0, 1)


def _grao(lado_px, metros_foto=1.5):
    """Grão do granito (rock_01): luminância relativa (média 1) e relevo, com lado_px por metros_foto."""
    USADAS.add("rock_01")
    cor = np.asarray(Image.open(PASTA / "rock_01_cor.jpg").convert("RGB").resize((lado_px, lado_px), Image.LANCZOS), float) / 255
    rel = np.asarray(Image.open(PASTA / "rock_01_relevo.png").convert("L").resize((lado_px, lado_px), Image.LANCZOS), float) / 255
    lum = (cor ** 2.2) @ LUM
    # tira as variações largas da foto (sombras, manchas): fica só o grão
    larga = ndi.gaussian_filter(lum, lado_px / 12, mode="wrap")
    g = lum / np.maximum(larga, 1e-3)
    croma = (cor ** 2.2) / np.maximum(lum[..., None], 1e-3)
    croma = croma / croma.reshape(-1, 3).mean(0)
    return g / g.mean(), croma, rel - ndi.gaussian_filter(rel, lado_px / 12, mode="wrap")


def textura_silhares(n=2048, metros=6.0, fiada=(0.40, 0.55), comp=(0.6, 1.2), cor=(0.62, 0.60, 0.55),
                     desvio=0.16, paleta=None, junta_m=0.012, junta_cor=(0.30, 0.28, 0.26), almofada=0.0,
                     gasto_m=0.035, grao=0.55, liquen=0.04, liquen_cor=(0.80, 0.80, 0.76), oxido=0.0,
                     escorrencias=0.0, irregular=0.0, semente=31, fiadas_fixas=None):
    """Silhares em fiadas, n x n px = metros x metros (a imagem repete sem costura nos dois sentidos).

    fiada: altura mínima e máxima das fiadas (m); fiadas_fixas: lista de alturas que se repetem (m), em vez de sortear.
    comp: comprimento mínimo e máximo dos silhares (m).
    cor: cor média sRGB da pedra (medida nas fotos do monumento).
    desvio: desvio-padrão do tom de cada pedra (luminância relativa; 0.16 dá c. 30–40 de desvio RGB no render).
    paleta: [(cor sRGB relativa, probabilidade)] de tingimentos por pedra (ocre, rosado, cinzento frio), em relação a cor.
    junta_m, junta_cor: largura e cor das juntas. almofada: 0 = face plana, 1 = silhar almofadado (aresta boleada funda).
    gasto_m: largura da aresta gasta. grao: força do grão da foto. liquen: fração com pintas de líquen claro.
    oxido: fração com manchas ocre. escorrencias: força das escorrências escuras verticais.
    irregular: 0 = juntas a prumo e fiadas direitas; > 0 juntas inclinadas e fiadas onduladas (alvenaria aparelhada)."""
    r = np.random.default_rng(semente)
    m_px = n / metros
    # --- desenho das pedras (periódico: as fiadas enchem exatamente n; cada fiada dá a volta em x)
    alturas = []
    if fiadas_fixas:
        tot = sum(fiadas_fixas)
        k = max(1, round(metros / tot))
        alturas = list(fiadas_fixas) * k
    else:
        y = 0.0
        while y < metros - 1e-6:
            h = r.uniform(*fiada)
            if metros - (y + h) < fiada[0] * 0.75:
                h = metros - y
            alturas.append(h)
            y += h
    esc = metros / sum(alturas)
    cel = np.full((n, n), -1, int)
    tom, tinta = [], []
    pal = paleta or [((1.0, 1.0, 1.0), 1.0)]
    pc = np.array([p[0] for p in pal], float)
    pp = np.array([p[1] for p in pal], float)
    pp = pp / pp.sum()
    y, k = 0.0, 0
    for h in alturas:
        y0, y1 = int(round(y * esc * m_px)), int(round((y + h) * esc * m_px))
        x = r.uniform(0, comp[1]) * m_px
        x_fim = x + n
        while x < x_fim - 1:
            w = r.uniform(*comp) * m_px
            if x_fim - (x + w) < comp[0] * 0.6 * m_px:
                w = x_fim - x
            xs = np.arange(int(round(x)), int(round(x + w))) % n
            cel[np.ix_(np.arange(y0, y1), xs)] = k
            tom.append(np.exp(r.normal(0, desvio)))
            tinta.append(pc[r.choice(len(pc), p=pp)])
            k += 1
            x += w
        y += h
    cel[cel < 0] = 0
    tom, tinta = np.array(tom), np.array(tinta)
    yy, xx = np.mgrid[0:n, 0:n]
    # arestas nunca a régua: ruído fino (2–3 cm) nas juntas; irregular > 0 inclina-as e ondula as fiadas
    fx = _fbm_periodico(n, semente + 3, base=int(metros * 3)) - 0.5
    fy = _fbm_periodico(n, semente + 4, base=int(metros * 3)) - 0.5
    ex = _fbm_periodico(n, semente + 13, base=int(metros * 8), oit=2) - 0.5
    ey = _fbm_periodico(n, semente + 14, base=int(metros * 8), oit=2) - 0.5
    wx = (fx * irregular * m_px + ex * 0.04 * m_px).astype(int)
    wy = (fy * irregular * 0.5 * m_px + ey * 0.04 * m_px).astype(int)
    cel = cel[(yy + wy) % n, (xx + wx) % n]
    # --- juntas e arestas
    borda = np.zeros((n, n), bool)
    for dy, dx in ((0, 1), (1, 0)):
        d = cel != np.roll(np.roll(cel, dy, 0), dx, 1)
        borda |= d | np.roll(np.roll(d, -dy, 0), -dx, 1)
    dist = ndi.distance_transform_edt(~borda) / m_px  # m até à junta (não vê a costura: aceitável a 1–2 px)
    junta = dist < junta_m / 2
    g = np.clip((dist - junta_m / 2) / max(gasto_m, 1e-3), 0, 1)
    perfil = np.sqrt(g) if almofada <= 0 else g ** (0.35 + 0.4 * almofada)
    # --- grão da foto, com um deslocamento por pedra
    lado = int(round(1.5 * m_px))
    gl, gc, gr = _grao(lado)
    ox = r.integers(0, lado, k + 1)[cel]
    oy = r.integers(0, lado, k + 1)[cel]
    iy, ix = (yy + oy) % lado, (xx + ox) % lado
    gra = 1 + (gl[iy, ix] - 1) * grao
    croma = 1 + (gc[iy, ix] - 1) * 0.35
    rel_g = gr[iy, ix]
    # --- cor
    base = np.array(cor, float) ** 2.2
    tex = base * (tom[cel] * gra)[..., None] * (np.array(tinta) ** 2.2)[cel] * croma
    k_a = 0.20 + 0.20 * almofada
    tex = tex * (1 - k_a + k_a * perfil)[..., None]  # arestas gastas (ou almofadadas) mais escuras: sombra e sujidade
    patina = _fbm_periodico(n, semente + 6, base=int(metros * 4), oit=3)
    tex = tex * (1 + 0.22 * (patina - 0.5))[..., None]
    larga = _fbm_periodico(n, semente + 5, base=max(2, int(metros / 1.5)))
    tex = tex * (1 + 0.12 * (larga - 0.5))[..., None]
    if oxido:
        m = _fbm_periodico(n, semente + 8, base=max(2, int(metros * 1.2)))
        a = np.clip((m - np.quantile(m, 1 - oxido)) / 0.04, 0, 1)[..., None] * 0.45
        tex = tex * (1 - a) + (np.array((0.66, 0.50, 0.30)) ** 2.2) * (tex @ LUM / (base @ LUM))[..., None] * a
    if escorrencias:
        e = _fbm_periodico(n, semente + 9, base=max(2, int(metros * 4)))
        e = ndi.uniform_filter1d(e, size=int(1.8 * m_px), axis=0, mode="wrap")  # riscas verticais
        a = np.clip((e - np.quantile(e, 0.75)) / 0.03, 0, 1) * escorrencias
        tex = tex * (1 - 0.45 * a)[..., None]
    if liquen:
        m = _fbm_periodico(n, semente + 11, base=int(metros * 10), oit=3)
        a = np.clip((m - np.quantile(m, 1 - liquen)) / 0.03, 0, 1)[..., None] * 0.30
        tex = tex * (1 - a) + (np.array(liquen_cor) ** 2.2) * gra[..., None] * a
    jc = np.array(junta_cor, float) ** 2.2
    tex = np.where(junta[..., None], jc * (0.85 + 0.3 * gra[..., None] / 2), tex)
    tex = np.clip(tex, 0, 1) ** (1 / 2.2)
    alt = perfil * (~junta) * (0.6 + 0.4 * almofada) + 0.15 * rel_g * (~junta)
    return tex, ndi.gaussian_filter(alt, 0.7, mode="wrap")
