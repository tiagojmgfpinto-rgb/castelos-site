"""Geometria plana e amostragem: vetores, polígonos, linhas, arcos e interpolação nas grelhas do LiDAR.

Funções comuns às maquetas, tiradas sem alterações dos construir.py (ver pipeline/kit/LEIAME.md)."""
import math
import numpy as np
import shapely
from scipy import ndimage as ndi

__all__ = ["unit", "area", "ccw", "orientar", "densificar", "dentro", "distancia", "simplificar", "desvio", "mediana_movel", "normais", "angulos", "partir", "amostra", "quad", "arco_pts", "arco", "arco_quebrado", "ogiva", "redondo", "reto", "_retangulo", "_sd", "_ang", "_anc", "_arred"]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def ccw(pts):
    pts = [tuple(map(float, p)) for p in pts]
    return pts if shapely.Polygon(pts).exterior.is_ccw else pts[::-1]


def orientar(pts, ref):
    """Põe a face com a normal para o lado de ref (vetor 3D)."""
    p = np.asarray(pts, float)
    nrm = np.zeros(3)
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        nrm += np.cross(a, b)
    return list(pts) if np.dot(nrm, ref) >= 0 else list(pts)[::-1]


def densificar(linha, passo=2.0, fechada=False):
    pts = np.vstack([linha, linha[:1]]) if fechada else linha
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / passo)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    out = np.array(out)
    return out[:-1] if fechada else out


def dentro(px, py, poly):
    px, py = np.asarray(px, float), np.asarray(py, float)
    res = np.zeros(px.shape, bool)
    x, y = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = ((y[i] > py) != (y[j] > py)) & (px < (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i] + 1e-12) + x[i])
        res ^= cond
        j = i
    return res


def distancia(px, py, linha, fechada=False):
    """Distância mínima de pontos a uma polilinha e o ponto mais próximo."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    pts = np.vstack([linha, linha[:1]]) if fechada else linha
    best = np.full(px.shape, np.inf)
    qx, qy = np.zeros(px.shape), np.zeros(px.shape)
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        t = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / max(d @ d, 1e-9), 0, 1)
        cx, cy = a[0] + t * d[0], a[1] + t * d[1]
        dd = np.hypot(px - cx, py - cy)
        m = dd < best
        best[m], qx[m], qy[m] = dd[m], cx[m], cy[m]
    return best, qx, qy


def simplificar(poly, tol=0.6):
    pts = list(poly)
    mudou = True
    while mudou and len(pts) > 4:
        mudou = False
        for i in range(len(pts)):
            a, b, c = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[(i + 1) % len(pts)])
            d = abs((c - a)[0] * (b - a)[1] - (c - a)[1] * (b - a)[0]) / max(np.linalg.norm(c - a), 1e-9)
            if d < tol:
                pts.pop(i)
                mudou = True
                break
    return np.array(pts)


def desvio(linha, d, fechada=False):
    """Polilinha paralela à distância d (positivo = à esquerda), com esquinas em mitra."""
    n = len(linha)
    out = []
    for i in range(n):
        if fechada or 0 < i < n - 1:
            t1 = unit(linha[i] - linha[i - 1])
            t2 = unit(linha[(i + 1) % n] - linha[i])
        elif i == 0:
            t1 = t2 = unit(linha[1] - linha[0])
        else:
            t1 = t2 = unit(linha[-1] - linha[-2])
        n1 = np.array([-t1[1], t1[0]])
        nm = unit(np.array([-(t1 + t2)[1], (t1 + t2)[0]]))
        if np.linalg.norm(nm) < 1e-6:
            nm = n1
        esc = 1.0 / max(0.7, float(nm @ n1))
        out.append(linha[i] + nm * d * esc)
    return np.array(out)


def mediana_movel(v, k=5, fechada=False):
    h = k // 2
    vv = np.concatenate([v[-h:], v, v[:h]]) if fechada else np.concatenate([np.repeat(v[:1], h), v, np.repeat(v[-1:], h)])
    return np.array([np.median(vv[i:i + k]) for i in range(len(v))])


def normais(alt, forca):
    gx = (np.roll(alt, -1, 1) - np.roll(alt, 1, 1)) * forca
    gy = (np.roll(alt, -1, 0) - np.roll(alt, 1, 0)) * forca
    nv = np.stack([-gx, -gy, np.ones_like(alt)], 2)
    nv /= np.linalg.norm(nv, axis=2, keepdims=True)
    return nv * 0.5 + 0.5


def angulos(linha):
    """Viragem em cada vértice interior (graus, positivo = ângulo saliente com o exterior à direita)."""
    out = np.zeros(len(linha))
    for i in range(1, len(linha) - 1):
        t1, t2 = unit(linha[i] - linha[i - 1]), unit(linha[i + 1] - linha[i])
        out[i] = math.degrees(math.atan2(t1[0] * t2[1] - t1[1] * t2[0], t1 @ t2))
    return out


def partir(poly, frente=7.5, fundo=13.0):
    """Os quarteirões do OSM vêm muitas vezes como um só polígono: partem-se em casas, em faixas de c.
    7,5 m ao longo do eixo maior (e em duas fiadas costas com costas quando o bloco é fundo), como o
    parcelamento estreito que se vê na ortoimagem. Cada parte leva a sua altura medida no MDS."""
    g = shapely.Polygon(poly).buffer(0)
    if g.area < 110:
        return [poly]
    r = np.array(g.minimum_rotated_rectangle.exterior.coords)[:4]
    e0, e1 = r[1] - r[0], r[2] - r[1]
    if np.linalg.norm(e0) < np.linalg.norm(e1):
        e0, e1 = e1, e0
    L, W = np.linalg.norm(e0), np.linalg.norm(e1)
    u, v = e0 / L, e1 / W
    c = r.mean(0)
    nl = max(1, int(round(L / frente)))
    nw = 2 if W > fundo else 1
    partes = []
    for i in range(nl):
        for j in range(nw):
            u0, u1 = -L / 2 + L * i / nl, -L / 2 + L * (i + 1) / nl
            v0, v1 = -W / 2 + W * j / nw, -W / 2 + W * (j + 1) / nw
            q = shapely.Polygon([c + u * u0 + v * v0, c + u * u1 + v * v0, c + u * u1 + v * v1, c + u * u0 + v * v1])
            p = g.intersection(q.buffer(0.01, join_style="mitre"))
            for pp in (p.geoms if hasattr(p, "geoms") else [p]):
                if pp.geom_type == "Polygon" and pp.area > 12:
                    b_ = np.array(pp.simplify(0.3).exterior.coords)[:-1]
                    partes.append(b_ if area(b_) > 0 else b_[::-1])
    return partes or [poly]


def amostra(G, x0, y0, p, x, y, ordem=1):
    xa, ya = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    r = ndi.map_coordinates(G, [(ya - y0) / p, (xa - x0) / p], order=ordem, mode="nearest")
    return r if np.ndim(x) else float(r[0])


def quad(m, f, s0, s1, z0, z1, d=0.0, mat=0):
    m.face([f.p(s0, d, z0), f.p(s1, d, z0), f.p(s1, d, z1), f.p(s0, d, z1)], mat)


def arco_pts(sc, zc, r, k=10, a0=0.0, a1=math.pi):
    return [(sc + r * math.cos(a), zc + r * math.sin(a)) for a in np.linspace(a0, a1, k + 1)]


def arco(m, centro, u, normal, largura, soleira, nascenca, prof, mat):
    """Vão em arco pleno, como reentrância escura na face (normal aponta para fora)."""
    r = largura / 2
    pts = [(-r, soleira), (r, soleira), (r, soleira + nascenca)]
    for k in range(1, 12):
        a = math.pi * k / 12
        pts.append((r * math.cos(a), soleira + nascenca + r * math.sin(a)))
    pts.append((-r, soleira + nascenca))
    frente = [centro + u * px + normal * 0.03 for px, _ in pts]
    m.face([(*p, z) for p, (_, z) in zip(frente, pts)], mat)


def arco_quebrado(m, centro, u, normal, largura, soleira, nascenca, mat):
    """Vão em arco quebrado (ogival), como reentrância escura na face."""
    r = largura
    pts = [(-largura / 2, soleira), (largura / 2, soleira), (largura / 2, soleira + nascenca)]
    for k in range(1, 7):  # arco do lado direito, centro no arranque esquerdo
        a = math.pi / 3 * k / 6
        pts.append((-largura / 2 + r * math.cos(a), soleira + nascenca + r * math.sin(a)))
    for k in range(5, 0, -1):  # lado esquerdo, simétrico
        a = math.pi / 3 * k / 6
        pts.append((largura / 2 - r * math.cos(a), soleira + nascenca + r * math.sin(a)))
    pts.append((-largura / 2, soleira + nascenca))
    m.face([(*(centro + u * px + normal * 0.03), z) for px, z in pts], mat)


def ogiva(sc, w, za, h_arranque, k=8, raio=0.8):
    """Contorno de um vão em arco quebrado (ogival): de za ao arranque h_arranque acima, e o arco com raio = raio × w
    (de baixo à esquerda, no sentido anti-horário)."""
    r = raio * w
    sa, sb = sc - w / 2, sc + w / 2
    zs = za + h_arranque
    th = math.acos((r - w / 2) / r)  # ângulo do fecho (no eixo sc); antes passava o eixo e os arcos cruzavam-se
    out = [(sa, za), (sb, za), (sb, zs)]
    # arco direito: centro em (sb - r, zs), do ângulo 0 a th
    for a in np.linspace(0, th, k + 1)[1:]:
        out.append((sb - r + r * math.cos(a), zs + r * math.sin(a)))
    # arco esquerdo: centro em (sa + r, zs), de pi - th a pi
    for a in np.linspace(math.pi - th, math.pi, k + 1)[1:]:
        out.append((sa + r + r * math.cos(a), zs + r * math.sin(a)))
    return out


def redondo(sc, w, za, h_arranque, k=10):
    r = w / 2
    zs = za + h_arranque
    out = [(sc - r, za), (sc + r, za)]
    out += [(sc + r * math.cos(a), zs + r * math.sin(a)) for a in np.linspace(0, math.pi, k + 1)]
    return out


def reto(sc, w, za, zb):
    return [(sc - w / 2, za), (sc + w / 2, za), (sc + w / 2, zb), (sc - w / 2, zb)]


def _retangulo(base):
    b = np.array(base)
    if area(b) < 0:
        b = b[::-1]
    e = [np.linalg.norm(b[(i + 1) % 4] - b[i]) for i in range(4)]
    k = 0 if e[0] + e[2] >= e[1] + e[3] else 1  # arestas k e k+2 são as compridas
    b = np.roll(b, -k, axis=0)
    L = (np.linalg.norm(b[1] - b[0]) + np.linalg.norm(b[2] - b[3])) / 2
    W = (np.linalg.norm(b[3] - b[0]) + np.linalg.norm(b[2] - b[1])) / 2
    return b, L, W


def _sd(sz, d):
    return (sz[0], d, sz[1])


def _ang(v):
    return math.radians(v)


def _anc(x, y, z):
    return [round(float(x), 1), round(float(z), 1), round(float(-y), 1)]


def _arred(v):
    if isinstance(v, float):
        return round(v, 1)
    if isinstance(v, list):
        return [_arred(x) for x in v]
    if isinstance(v, dict):
        return {k: _arred(x) for k, x in v.items()}
    return v
