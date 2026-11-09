"""Peças de arquitetura: referencial de fachada (Quadro), vãos, arcos, aduelas, telhados, guaritas, merlões, escadas e portas.

Funções comuns às maquetas, tiradas sem alterações dos construir.py (ver pipeline/kit/LEIAME.md)."""
import math
import numpy as np
import shapely
from .geometria import _retangulo, _sd, arco_pts, area, ccw, orientar, quad, simplificar, unit

__all__ = ["Quadro", "_vao_desloc", "aduelas", "altura_telhado", "barra", "contorno_vao", "corrimao", "cupula", "faixa_inclinada", "guardas_corredor", "guarita", "merlao", "oitavado", "painel", "parede", "passagem", "piramide", "porta_revelim", "porta_tunel", "telhado_4aguas", "telhado_generico", "tunel", "vao"]


class Quadro:
    """Referencial local de uma fachada: s ao longo dela (para a direita de quem olha de fora), z para
    cima e d para fora da parede, a partir de um ponto da face."""

    def __init__(self, m, origem, normal):
        self.m = m
        self.o = np.asarray(origem, float)
        self.n = unit(np.asarray(normal, float))
        self.t = np.array([self.n[1], -self.n[0]])  # olhando de fora (contra a normal), a direita

    def p(self, s, z, d):
        q = self.o + self.t * s + self.n * d
        return (q[0], q[1], z)

    def bloco(self, s0, s1, z0, z1, d0, d1, mat):
        c = self.o + self.t * (s0 + s1) / 2 + self.n * (d0 + d1) / 2
        self.m.caixa(c, self.t, abs(s1 - s0), abs(d1 - d0), z0, z1, mat, fundo=True)

    def placa(self, sz, d0, d1, mat):
        """Extrude um polígono do plano (s, z) entre as profundidades d0 e d1."""
        sz = [tuple(q) for q in sz]
        if area(np.array(sz)) < 0:
            sz = sz[::-1]
        frente = [self.p(s, z, d1) for s, z in sz]
        tras = [self.p(s, z, d0) for s, z in sz]
        self.m.face(frente[::-1], mat)
        self.m.face(tras, mat)
        k = len(sz)
        for i in range(k):
            j = (i + 1) % k
            self.m.face([frente[i], frente[j], tras[j], tras[i]], mat)

    def arco_sz(self, s0, r0, r1, zn, d0, d1, mat, a0=0.0, a1=math.pi, k=12, aduelas=False):
        """Coroa de arco (entre os raios r0 e r1) centrada em (s0, zn), extrudida entre d0 e d1;
        com aduelas=True, partida em aduelas com juntas finas."""
        for i in range(k):
            t0 = a0 + (a1 - a0) * i / k + (0.012 if aduelas else 0)
            t1 = a0 + (a1 - a0) * (i + 1) / k - (0.012 if aduelas else 0)
            q = [(s0 + r0 * math.cos(t0), zn + r0 * math.sin(t0)), (s0 + r1 * math.cos(t0), zn + r1 * math.sin(t0)),
                 (s0 + r1 * math.cos(t1), zn + r1 * math.sin(t1)), (s0 + r0 * math.cos(t1), zn + r0 * math.sin(t1))]
            self.placa(q, d0, d1, mat)

    def vao(self, largura, z0, nascenca, d, mat, reto=False):
        """Vão escuro (arco pleno, ou verga reta) no plano d."""
        w = largura / 2
        pts = [(-w, z0), (w, z0), (w, z0 + nascenca)]
        if not reto:
            pts += [(w * math.cos(math.pi * k / 14), z0 + nascenca + w * math.sin(math.pi * k / 14)) for k in range(1, 14)]
        pts += [(-w, z0 + nascenca)]
        self.m.face([self.p(s, z, d) for s, z in pts], mat)

    def esfera(self, s, z, d, r, mat):
        self.m.esfera(self.p(s, z, d), r, mat, k=10)

    def escudo(self, s0, z0, larg, alt, d0, d1, mat, coroa=True):
        """Escudo português (base redonda) com coroa real por cima, em relevo."""
        w = larg / 2
        pts = [(s0 - w, z0 + alt), (s0 - w, z0 + alt * 0.4)]
        pts += [(s0 - w * math.cos(math.pi / 2 * k / 6), z0 + alt * 0.4 - alt * 0.4 * math.sin(math.pi / 2 * k / 6)) for k in range(1, 7)]
        pts += [(s0 + w * math.sin(math.pi / 2 * k / 6), z0 + alt * 0.4 - alt * 0.4 * math.cos(math.pi / 2 * k / 6)) for k in range(1, 6)]
        pts += [(s0 + w, z0 + alt * 0.4), (s0 + w, z0 + alt)]
        self.placa(pts, d0, d1, mat)
        self.placa([(s0 - w * 0.55, z0 + alt * 0.25), (s0 + w * 0.55, z0 + alt * 0.25), (s0 + w * 0.55, z0 + alt * 0.85), (s0 - w * 0.55, z0 + alt * 0.85)],
                   d1, d1 + 0.05, mat)  # escudete das quinas
        if coroa:
            zc = z0 + alt
            self.placa([(s0 - w * 1.05, zc), (s0 + w * 1.05, zc), (s0 + w * 1.15, zc + alt * 0.28), (s0 + w * 0.5, zc + alt * 0.2),
                        (s0, zc + alt * 0.36), (s0 - w * 0.5, zc + alt * 0.2), (s0 - w * 1.15, zc + alt * 0.28)], d0, d1, mat)
            self.esfera(s0, zc + alt * 0.45, (d0 + d1) / 2, alt * 0.09, mat)

    def almofadas(self, s0, s1, z0, z1, d0, d1, mat, fiada=0.55):
        """Pilastra de almofadado rústico: blocos salientes com juntas fundas (fotografia da Porta da Coroada)."""
        self.bloco(s0 + 0.04, s1 - 0.04, z0, z1, d0, d1 - 0.06, mat)
        for k, z in enumerate(np.arange(z0, z1 - 0.1, fiada)):
            ss = (0.0 if k % 2 == 0 else 0.06)
            self.bloco(s0 + ss, s1 - ss, z + 0.04, min(z + fiada - 0.04, z1), d1 - 0.08, d1, mat)


def _vao_desloc(Q, s0, w, z0, hn, d, mat, reto=False):
    """Como Quadro.vao mas centrado em s0."""
    o = Q.o
    Q.o = Q.o + Q.t * s0
    Q.vao(w, z0, hn, d, mat, reto=reto)
    Q.o = o


def aduelas(m, f, sc, w, za, zb, larg, sal, n=9, mat="cantaria_lisa", ombreiras=True, d0=0.0):
    """Aduelas do arco (e ombreiras de cantaria), ligeiramente salientes da face."""
    r = w / 2
    zc = zb - r
    R = r + larg
    for i in range(n):
        a0, a1 = math.pi * i / n, math.pi * (i + 1) / n
        g = 0.010
        pts = [(sc + r * math.cos(a0 + g), zc + r * math.sin(a0 + g)), (sc + R * math.cos(a0 + g), zc + R * math.sin(a0 + g)),
               (sc + R * math.cos(a1 - g), zc + R * math.sin(a1 - g)), (sc + r * math.cos(a1 - g), zc + r * math.sin(a1 - g))]
        m.face([f.p(s, d0 + sal, z) for s, z in pts], mat)
        for (s1, z1), (s2, z2) in zip(pts, pts[1:] + pts[:1]):
            m.face([f.p(s1, d0, z1), f.p(s2, d0, z2), f.p(s2, d0 + sal, z2), f.p(s1, d0 + sal, z1)], mat)
    if ombreiras:
        for s0, s1 in ((sc - r - larg, sc - r), (sc + r, sc + r + larg)):
            hz = zc - za
            k = max(1, int(round(hz / 0.45)))
            for j in range(k):
                z0_, z1_ = za + hz * j / k + 0.008, za + hz * (j + 1) / k - 0.008
                m.caixa_face(f, s0, s1, d0, d0 + sal, z0_, z1_, mat)


def altura_telhado(poly, inclinacao=26):
    s = simplificar(poly)
    if len(s) == 4 and abs(area(s)) > 0.85 * abs(area(poly)):
        return _retangulo(s)[2] / 2 * math.tan(math.radians(inclinacao))
    return min(2.4, 0.25 * math.sqrt(abs(area(poly))))


def barra(m, p0, p1, r, mat):
    """Barra fina de secção quadrada entre dois pontos (ferragens)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    d /= max(np.linalg.norm(d), 1e-9)
    a = np.cross(d, [0, 0, 1.0]) if abs(d[2]) < 0.95 else np.cross(d, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b_ = np.cross(d, a)
    q = [a * r + b_ * r, -a * r + b_ * r, -a * r - b_ * r, a * r - b_ * r]
    for i in range(4):
        j = (i + 1) % 4
        m.face([p0 + q[i], p0 + q[j], p1 + q[j], p1 + q[i]], mat)


def contorno_vao(sc, w, za, zb, arco=True, k=12):
    r = w / 2
    sa, sb = sc - r, sc + r
    c = [(sa, za), (sb, za)]
    if arco:
        c += arco_pts(sc, zb - r, r, k)
    else:
        c += [(sb, zb), (sa, zb)]
    return c


def corrimao(m, pts3, alt=0.95, mat="ferro", passo=1.1):
    """Corrimão de ferro: prumos e duas barras ao longo de uma linha 3D (o pé dos prumos)."""
    pts3 = [np.asarray(p, float) for p in pts3]
    tot = []
    for a, b in zip(pts3[:-1], pts3[1:]):
        L = np.linalg.norm(b - a)
        k = max(1, int(math.ceil(L / passo)))
        for i in range(k):
            tot.append(a + (b - a) * i / k)
    tot.append(pts3[-1])
    for p in tot:
        m.barra(p, p + np.array([0, 0, alt]), 0.022, mat, k=5)
    for h in (alt, alt * 0.5):
        for a, b in zip(pts3[:-1], pts3[1:]):
            m.barra(a + np.array([0, 0, h]), b + np.array([0, 0, h]), 0.02, mat, k=5)


def cupula(m, c, r, z0, mat, k=16, aneis=6, alt=None):
    """Meia cúpula por anéis (cobertura da torre de Santo Estêvão)."""
    alt = r if alt is None else alt
    ang = np.linspace(0, 2 * math.pi, k, endpoint=False)
    prev = [(c[0] + r * math.cos(t), c[1] + r * math.sin(t), z0) for t in ang]
    for i in range(1, aneis + 1):
        fi = math.pi / 2 * i / aneis
        rr, zz = r * math.cos(fi), z0 + alt * math.sin(fi)
        cur = [(c[0] + rr * math.cos(t), c[1] + rr * math.sin(t), zz) for t in ang]
        for j in range(k):
            jj = (j + 1) % k
            if i == aneis:
                m.face([prev[j], prev[jj], (c[0], c[1], zz)], mat)
            else:
                m.face([prev[j], prev[jj], cur[jj], cur[j]], mat)
        prev = cur


def faixa_inclinada(Q, pts, esp, d0, d1, mat):
    """Faixa (cornija) que segue uma linha poligonal do plano (s, z), com espessura esp para cima."""
    for (s0, z0), (s1, z1) in zip(pts[:-1], pts[1:]):
        Q.placa([(s0, z0), (s1, z1), (s1, z1 + esp), (s0, z0 + esp)], d0, d1, mat)


def guardas_corredor(m, F0, n, pts, NR, A_, TP):
    """Guardas baixas de granito sobre os muros da calçada que sai da porta (fotografia da Porta da Coroada)."""
    t = np.array([-n[1], n[0]])
    rel = pts - F0
    dd, ss = rel @ n, rel @ t
    sel = np.nonzero((dd > 0.5) & (dd < 40) & (np.abs(ss) < 9) & (np.abs(NR @ t) > 0.8))[0]
    for lado in (-1, 1):
        k = sel[np.sign(ss[sel]) == lado]
        k = k[np.argsort(dd[k])]
        for i, j in zip(k[:-1], k[1:]):
            if dd[j] - dd[i] > 1.5 or np.linalg.norm(pts[j] - pts[i]) > 1.5:
                continue
            a_, b_ = pts[i] + NR[i] * (A_[i] - 0.25), pts[j] + NR[j] * (A_[j] - 0.25)
            comp = float(np.linalg.norm(b_ - a_))
            if comp < 1e-3:
                continue
            z0 = float(min(TP[i], TP[j])) - 0.1
            m.caixa((a_ + b_) / 2, unit(b_ - a_), comp + 0.02, 0.45, z0, z0 + 1.0, 0)
            m.caixa((a_ + b_) / 2, unit(b_ - a_), comp + 0.02, 0.55, z0 + 1.0, z0 + 1.12, 0)


def guarita(m, c, bis, zc, topo):
    """Guarita de cantaria (fotografias): mísula em cone invertido aos degraus a sair do ângulo, corpo
    quadrado de cunhais chanfrados com frestas, cornija, cúpula e pináculo em bola."""
    u = unit(bis)
    w = np.array([-u[1], u[0]])
    L = 0.78  # meia largura do corpo
    ch = 0.22  # chanfro
    planta = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        planta += [c + u * sx * L + w * sy * (L - ch), c + u * sx * (L - ch) + w * sy * L]
    planta = np.array(planta)
    if area(planta) < 0:
        planta = planta[::-1]
    piso = zc + 0.25
    # mísula: anéis cada vez mais estreitos para baixo, até um remate em pinha
    for k, (r_, z0, z1) in enumerate(((1.0, piso - 0.35, piso), (0.86, piso - 0.65, piso - 0.35), (0.7, piso - 0.95, piso - 0.65),
                                      (0.52, piso - 1.25, piso - 0.95), (0.32, piso - 1.55, piso - 1.25))):
        m.prisma([c + (p - c) * r_ for p in planta], z0, z1, 1, fundo=True)
    m.cilindro(c, 0.18, piso - 1.85, piso - 1.55, 1, k=8, r1=0.3)
    zt = piso + 1.95
    m.prisma(planta, piso, zt, 1)
    # frestas nas três faces de fora e porta na de dentro
    for d_, larg, z0, z1 in ((u, 0.12, piso + 0.7, piso + 1.45), (w, 0.12, piso + 0.7, piso + 1.45),
                             (-w, 0.12, piso + 0.7, piso + 1.45), (-u, 0.5, piso + 0.05, piso + 1.6)):
        m.caixa(c + d_ * (L + 0.005), np.array([-d_[1], d_[0]]), larg, 0.02, z0, z1, 2)
    m.prisma([c + (p - c) * 1.1 for p in planta], zt, zt + 0.16, 1)  # cornija
    # cúpula em meia laranja (anéis) e pináculo
    prev = [c + (p - c) * 1.0 for p in planta]
    zprev = zt + 0.16
    for k in range(1, 5):
        fi = math.pi / 2 * k / 4
        r_ = math.cos(fi)
        zz = zt + 0.16 + 0.75 * math.sin(fi)
        cur = [c + (p - c) * r_ for p in planta]
        for i in range(len(planta)):
            j = (i + 1) % len(planta)
            m.face([(*prev[i], zprev), (*prev[j], zprev), (*cur[j], zz), (*cur[i], zz)], 1)
        prev, zprev = cur, zz
    m.cilindro(c, 0.12, zprev - 0.02, zprev + 0.2, 1, k=8)
    m.esfera((c[0], c[1], zprev + 0.42), 0.24, 1, k=10)


def merlao(m, base, z0, z1, zp, mat, seteira=None):
    """Merlão: prisma de z0 a z1 e remate piramidal até zp; seteira = (plano, s, z) de um vão em arco na face."""
    base = ccw(base)
    m.prisma(base, z0, z1, mat, topo=False)
    m.piramide(base, z1, zp, mat)


def oitavado(m, c, r0, r1, z0, z1, mat, rot=math.pi / 8):
    """Tronco octogonal (agulhas, tambores)."""
    a = np.linspace(0, 2 * math.pi, 8, endpoint=False) + rot
    b0 = [(c[0] + r0 * math.cos(t), c[1] + r0 * math.sin(t), z0) for t in a]
    b1 = [(c[0] + r1 * math.cos(t), c[1] + r1 * math.sin(t), z1) for t in a]
    for i in range(8):
        j = (i + 1) % 8
        if r1 > 0.01:
            m.face([b0[i], b0[j], b1[j], b1[i]], mat)
        else:
            m.face([b0[i], b0[j], (c[0], c[1], z1)], mat)
    if r1 > 0.01:
        m.face(b1, mat)


def painel(m, f, contorno, furos=(), mat=0, d=0.0):
    """Face plana do plano f (em s, z, a d para fora), com furos; triangulada (Delaunay restringida)."""
    pol = shapely.Polygon(contorno, [h for h in furos if len(h) >= 3])
    if not pol.is_valid:
        pol = pol.buffer(0)
    ref = np.array([f.n[0], f.n[1], 0.0])
    geoms = pol.geoms if hasattr(pol, "geoms") else [pol]
    for g in geoms:
        for t in shapely.constrained_delaunay_triangles(g).geoms:
            cs = np.array(t.exterior.coords)[:3]
            if shapely.Polygon(cs).area < 1e-6:
                continue
            m.face(orientar([f.p(s, d, z) for s, z in cs], ref), mat)


def parede(m, f, s0, s1, z0, z1, vaos, mat=0, passo_z=1.5, d=0.0):
    """Plano da face entre s0..s1 e z0..z1, com vãos retangulares ou em arco abertos.
    vaos: lista de (sa, sb, za, zb, arco); arco=True: zb é o fecho do arco de volta perfeita."""
    ss = sorted({s0, s1, *[v_[0] for v_ in vaos], *[v_[1] for v_ in vaos]})
    ss = [s for s in ss if s0 <= s <= s1]
    for a, b in zip(ss[:-1], ss[1:]):
        sm = (a + b) / 2
        cort = sorted([(v_[2], v_[3]) for v_ in vaos if v_[0] <= sm <= v_[1]])
        zs = [z0]
        for za, zb in cort:
            zs += [za, zb]
        zs.append(z1)
        for za, zb in zip(zs[0::2], zs[1::2]):
            if zb - za < 1e-4:
                continue
            nz = max(1, int(math.ceil((zb - za) / passo_z)))
            for i in range(nz):
                quad(m, f, a, b, za + (zb - za) * i / nz, za + (zb - za) * (i + 1) / nz, d=d, mat=mat)
    for sa, sb, za, zb, arco in vaos:
        if not arco:
            continue
        r = (sb - sa) / 2
        zc = zb - r
        pts = arco_pts((sa + sb) / 2, zc, r, 12)
        meio = len(pts) // 2
        m.face([f.p(s, d, z) for s, z in pts[:meio + 1]] + [f.p(sb, d, zb)], mat)
        m.face([f.p(s, d, z) for s, z in pts[meio:]] + [f.p(sa, d, zb)], mat)


def passagem(m, f, s0, s1, prof, z_base, z_topo, w, alt, chao_f, chao_d, mat="muro", mat_topo="muro_topo",
             mat_intra="intradorso", degrau=None, arco_dentro=None, topo=True, lados=True):
    """Bloco de muro com uma passagem em arco de volta perfeita, de lado a lado. f: plano da face de fora (d = 0,
    normal para fora); o bloco vai de s0 a s1 ao longo e de d = 0 a d = -prof; o eixo da passagem é s = 0.
    w, alt: largura e altura (do chão ao fecho) do vão; chao_f, chao_d: chão da passagem fora e dentro.
    degrau = (w2, alt2, p2): arquivolta de fora mais larga (w2 × alt2) até p2 de profundidade.
    arco_dentro = (w3, alt3): vão da face de dentro, se for diferente."""
    k = 14

    def face_com_vao(d, ww, aa, chao):
        r = ww / 2
        zap = chao + aa
        zs = zap - r
        # faixas dos lados, do pé ao topo
        quad(m, f, s0, -r, z_base, z_topo, d=d, mat=mat)
        quad(m, f, r, s1, z_base, z_topo, d=d, mat=mat)
        if chao > z_base + 0.01:
            quad(m, f, -r, r, z_base, chao, d=d, mat=mat)
        # por cima do arco: duas metades, do arranque ao fecho e até ao topo
        pts = arco_pts(0.0, zs, r, k)
        meio = len(pts) // 2
        m.face([f.p(s, d, z) for s, z in pts[:meio + 1]] + [f.p(0.0, d, z_topo), f.p(r, d, z_topo)], mat)
        m.face([f.p(s, d, z) for s, z in pts[meio:]] + [f.p(-r, d, z_topo), f.p(0.0, d, z_topo)], mat)
        return contorno_vao(0.0, ww, chao, zap, True, k)

    def forro(c_a, d_a, c_b, d_b):
        pa, pb = c_a + [c_a[0]], c_b + [c_b[0]]
        for i in range(len(c_a)):
            (sa1, za1), (sa2, za2) = pa[i], pa[i + 1]
            (sb1, zb1), (sb2, zb2) = pb[i], pb[i + 1]
            mt = mat_intra if i > 0 else "calcada"
            m.face([f.p(sa1, d_a, za1), f.p(sa2, d_a, za2), f.p(sb2, d_b, zb2), f.p(sb1, d_b, zb1)], mt)

    if degrau is None:
        c_fora = face_com_vao(0.0, w, alt, chao_f)
        d_ini = 0.0
    else:
        w2, alt2, p2 = degrau
        c2 = face_com_vao(0.0, w2, alt2, chao_f)
        chao_p = chao_f + (chao_d - chao_f) * p2 / prof
        c2b = contorno_vao(0.0, w2, chao_p, chao_p + alt2, True, k)
        forro(c2, 0.0, c2b, -p2)
        # face do degrau (entre a arquivolta e o vão), a p2 de profundidade
        c_fora = contorno_vao(0.0, w, chao_p, chao_p + alt, True, k)
        for i in range(len(c2b)):
            j = (i + 1) % len(c2b)
            if i == 0:
                continue
            m.face([f.p(*_sd(c2b[i], -p2)), f.p(*_sd(c2b[j], -p2)), f.p(*_sd(c_fora[j], -p2)), f.p(*_sd(c_fora[i], -p2))], mat)
        d_ini = -p2
    if arco_dentro is None:
        c_dentro = face_com_vao(-prof, w, alt, chao_d)
        forro(c_fora, d_ini, c_dentro, -prof)
    else:
        w3, alt3 = arco_dentro
        c_d = face_com_vao(-prof, w3, alt3, chao_d)
        c_m = contorno_vao(0.0, w, chao_d, chao_d + alt, True, k)
        forro(c_fora, d_ini, c_m, -prof + 0.4)
        forro(c_m, -prof + 0.4, c_d, -prof)  # abre em derrame para o arco de dentro
    if topo:
        m.face([f.p(s0, 0, z_topo), f.p(s1, 0, z_topo), f.p(s1, -prof, z_topo), f.p(s0, -prof, z_topo)], mat_topo)
    if lados:
        for s_ in (s0, s1):
            m.face([f.p(s_, 0, z_base), f.p(s_, -prof, z_base), f.p(s_, -prof, z_topo), f.p(s_, 0, z_topo)], mat)


def piramide(m, c, r, z0, z1, mat):
    """Pináculo: pirâmide de base quadrada."""
    b = [c + np.array([math.cos(k * math.pi / 2 + math.pi / 4), math.sin(k * math.pi / 2 + math.pi / 4)]) * r for k in range(4)]
    for k in range(4):
        m.face([(*b[k], z0), (*b[(k + 1) % 4], z0), (*c, z1)], mat)


def porta_revelim(Q, sole):
    """Porta do revelim (fotografia): moldura de cantaria com pilastras e entablamento, arco de volta
    perfeita com porta de madeira, e cartela com as armas reais por cima da cornija."""
    c, s_ = 0, 1
    z = sole
    Q.bloco(-3.0, 3.0, z - 0.5, z + 4.1, -1.4, 0.25, c)
    w, zn = 1.15, z + 2.35
    Q.vao(2 * w, z, zn - z, 0.26, s_)
    Q.placa([(-w + 0.05, z), (w - 0.05, z), (w - 0.05, zn), (-w + 0.05, zn)], 0.2, 0.24, 2)  # porta de ferro/madeira
    Q.arco_sz(0.0, w, w + 0.4, zn, 0.25, 0.36, c, k=11, aduelas=True)
    for sg in (-1, 1):
        s0, s1 = sorted((sg * 1.7, sg * 2.25))
        Q.bloco(s0, s1, z - 0.3, z + 3.45, 0.25, 0.48, c)
        Q.bloco(s0 - 0.06, s1 + 0.06, z + 3.45, z + 3.65, 0.25, 0.55, c)
    Q.bloco(-2.45, 2.45, z + 3.65, z + 3.95, 0.25, 0.6, c)
    Q.bloco(-2.6, 2.6, z + 3.95, z + 4.2, 0.25, 0.72, c)
    # cartela: escudo com coroa entre enrolamentos
    Q.bloco(-0.9, 0.9, z + 4.2, z + 5.3, -0.2, 0.4, c)
    Q.escudo(0.0, z + 4.35, 0.75, 0.8, 0.4, 0.52, c)
    for sg in (-1, 1):
        circ = [(sg * 1.05 + 0.24 * math.cos(t), z + 4.5 + 0.24 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 10, endpoint=False)]
        Q.placa(circ, -0.1, 0.45, c)
    return 5.6


def porta_tunel(Q, sole):
    """Boca de túnel do lado de dentro (fotografia): arco abatido largo em pano de silhares, com o fecho e
    a porta de madeira verde de duas folhas encostada."""
    c, s_ = 0, 1
    z = sole
    Q.bloco(-3.6, 3.6, z - 0.5, z + 5.2, -1.4, 0.22, c)
    w, zn = 1.8, z + 2.7
    Q.vao(2 * w, z, zn - z, 0.23, s_)
    Q.arco_sz(0.0, w, w + 0.55, zn, 0.18, 0.28, c, k=13, aduelas=True)
    Q.bloco(-0.22, 0.22, zn + w - 0.05, zn + w + 0.6, 0.22, 0.34, c)
    Q.placa([(-w, z), (-w + 0.08, z), (-w + 0.08, zn + 0.6), (-w, zn + 0.6)], 0.24, 1.0, 2)  # folha aberta
    return 5.2


def telhado_4aguas(m, base, z, inclinacao=26, mat=1, beiral=0.3):
    b, L, W = _retangulo(base)
    c = b.mean(0)
    b = np.array([p + unit(p - c) * beiral * 1.41 for p in b])  # beirado saliente
    L += 2 * beiral
    W += 2 * beiral
    h = W / 2 * math.tan(math.radians(inclinacao))
    mid_a, mid_b = (b[0] + b[3]) / 2, (b[1] + b[2]) / 2
    u = unit(mid_b - mid_a)
    r1, r2 = mid_a + u * min(W / 2, L / 2), mid_b - u * min(W / 2, L / 2)
    R1, R2 = (*r1, z + h), (*r2, z + h)
    B = [(*p, z) for p in b]
    m.face([B[0], B[1], R2, R1], mat)
    m.face([B[2], B[3], R1, R2], mat)
    m.face([B[1], B[2], R2], mat)
    m.face([B[3], B[0], R1], mat)
    m.face(B[::-1], mat)  # face de baixo do beirado


def telhado_generico(m, base, z, mat=1):
    b = np.array(base)
    if area(b) < 0:
        b = b[::-1]
    c = b.mean(0)
    inset = c + (b - c) * 0.45
    h = min(2.4, 0.25 * math.sqrt(abs(area(b))))
    n = len(b)
    for i in range(n):
        j = (i + 1) % n
        m.face([(*b[i], z), (*b[j], z), (*inset[j], z + h), (*inset[i], z + h)], mat)
    m.face([(*p, z + h) for p in inset], mat)


def tunel(m, f, contorno, d0, d1, mat, ref_dentro=True):
    """Faces laterais de um vão (o contorno em s, z), de d0 a d1."""
    pts = list(contorno) + [contorno[0]]
    for (s1, z1), (s2, z2) in zip(pts[:-1], pts[1:]):
        m.face([f.p(s1, d0, z1), f.p(s2, d0, z2), f.p(s2, d1, z2), f.p(s1, d1, z1)], mat)


def vao(m, f, sc, w, za, zb, prof, arco=True, mat_lado="cantaria_torre", mat_fundo="sombra", k=12, d0=0.0):
    """Ombreiras, soleira e intradorso do vão (para dentro da parede até prof) e o fundo."""
    contorno = contorno_vao(sc, w, za, zb, arco, k)
    pts = contorno + [contorno[0]]
    for (s1, z1), (s2, z2) in zip(pts[:-1], pts[1:]):
        m.face([f.p(s1, d0, z1), f.p(s2, d0, z2), f.p(s2, d0 - prof, z2), f.p(s1, d0 - prof, z1)], mat_lado)
    m.face([f.p(s, d0 - prof, z) for s, z in contorno], mat_fundo)
    return contorno
