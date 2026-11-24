"""Muros das ruínas escavadas como segmentos retos, a partir da superfície LiDAR de 25 cm.

As cristas dos muros são linhas salientes de 0,4–1 m de largura: filtro de cristas (Sato) sobre a superfície,
esqueleto, retas de Hough e fusão das retas quase colineares (ângulo < 8°, afastamento < 0,6 m, intervalo < 1,2 m).
Usado por construir.py (escavações do Palácio das Varandas, 24 nov. 2026)."""
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import sato
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line


def segmentos(sub, rel, x0, y0, passo, mascara, limiar=0.06):
    """Retas (em coordenadas locais) das cristas de `sub` dentro de `mascara` (mesma grelha, linha = y)."""
    r = sato(sub, sigmas=[1, 1.5, 2], black_ridges=False)
    m = (r > limiar) & (rel > 0.25) & mascara
    lab, k = ndi.label(m, structure=np.ones((3, 3)))
    if k:
        tam = ndi.sum(m, lab, range(1, k + 1))
        m &= np.isin(lab, 1 + np.nonzero(tam >= 16)[0])
    sk = skeletonize(m)
    L = probabilistic_hough_line(sk, threshold=4, line_length=6, line_gap=3, rng=2)
    return [((x0 + a[0] * passo, y0 + a[1] * passo), (x0 + b[0] * passo, y0 + b[1] * passo)) for a, b in L]


def fundir(segs, tol_ang=8.0, d_max=0.6, gap=1.2, min_len=1.2):
    """Junta retas quase colineares numa só (média ponderada pelo comprimento)."""
    S = [[np.array(a, float), np.array(b, float)] for a, b in segs if np.hypot(*(np.subtract(b, a))) > 0.3]
    mudou = True
    while mudou:
        mudou = False
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                a, b = S[i]
                c, d = S[j]
                t1, t2 = (b - a) / np.linalg.norm(b - a), (d - c) / np.linalg.norm(d - c)
                if abs(t1 @ t2) < np.cos(np.radians(tol_ang)):
                    continue
                if t1 @ t2 < 0:
                    c, d, t2 = d, c, -t2
                L1, L2 = np.linalg.norm(b - a), np.linalg.norm(d - c)
                t = (t1 * L1 + t2 * L2) / np.linalg.norm(t1 * L1 + t2 * L2)
                n = np.array([-t[1], t[0]])
                o1, o2 = ((a + b) / 2) @ n, ((c + d) / 2) @ n
                if abs(o1 - o2) > d_max:
                    continue
                s = sorted([a @ t, b @ t])
                u = sorted([c @ t, d @ t])
                if u[0] > s[1] + gap or s[0] > u[1] + gap:
                    continue
                o = (o1 * L1 + o2 * L2) / (L1 + L2)
                s0, s1 = min(s[0], u[0]), max(s[1], u[1])
                S[i] = [t * s0 + n * o, t * s1 + n * o]
                del S[j]
                mudou = True
                break
            if mudou:
                break
    return [(tuple(a), tuple(b)) for a, b in S if np.linalg.norm(b - a) >= min_len]
