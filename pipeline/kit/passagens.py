"""Passagens das portas pelo maciço das muralhas (revisão do perito, 13 nov. 2026).

Os vãos das portas eram planos pretos ou túneis curtos que acabavam numa parede: a porta parecia tapada.
Aqui ficam as peças comuns para abrir as passagens de lado a lado:

- anel(): o perfil interior da passagem (chão, paredes e abóbada) no plano de uma boca;
- tubo(): uma passagem varrida entre dois anéis (reta, em rampa ou em diagonal), só com o intradorso;
- buraco(): as células do terreno por cima de uma passagem que ficariam abaixo da abóbada (tiram-se) e a
  cota mais alta da borda em troços de 1 m;
- enchimento(): a alvenaria que tapa esse buraco por cima da abóbada e o alicerce por baixo do chão.

Referencial das bocas: o de Quadro (s ao longo da face, para a direita de quem olha de fora; d para fora).
"""
import math

import numpy as np


def perfil(w, h, reto=False, k=12):
    """Perfil interior (s, altura acima do chão), fechado pelo chão: arco pleno de raio w com nascença a h."""
    if reto:
        prf = [(-w, 0.0), (-w, h), (w, h), (w, 0.0)]
    else:
        prf = [(-w, 0.0)] + [(-w * math.cos(math.pi * i / k), h + w * math.sin(math.pi * i / k)) for i in range(k + 1)] + [(w, 0.0)]
    return prf + [prf[0]]


def anel(centro, t, w, h, z_chao, reto=False, sinal=1.0):
    """Pontos 3D do perfil no plano de uma boca: centro (x, y) no eixo, t a direção de s. Com sinal=-1 o
    perfil é percorrido ao contrário (a direita de uma face é a esquerda da face oposta)."""
    c, t = np.asarray(centro, float), np.asarray(t, float)
    return np.array([[*(c + t * s * sinal), z_chao + hh] for s, hh in perfil(w, h, reto)])


def tubo(m, RA, RB, mat, passo=1.0):
    """Passagem varrida entre os anéis RA e RB (com o mesmo número de pontos), em troços de c. 1 m."""
    L = float(np.hypot(*(RB[0, :2] - RA[0, :2])))
    n = max(1, int(math.ceil(L / passo)))
    aneis = [RA + (RB - RA) * (i / n) for i in range(n + 1)]
    for R0, R1 in zip(aneis[:-1], aneis[1:]):
        for j in range(len(R0) - 1):
            m.face([tuple(R0[j]), tuple(R0[j + 1]), tuple(R1[j + 1]), tuple(R1[j])], mat)
    return L


def buraco(X, Y, Z, A, B, za, zb, fecho, u0=1.0, meia=2.6):
    """Células da grelha (X, Y, Z: vértices) por cima do troço de A a B (chão de za a zb) que ficariam abaixo
    da abóbada (fecho acima do chão). Devolve a máscara das células (forma de Z menos 1 em cada eixo) e a
    borda: [(de, até, cota mais alta das células tiradas nesse metro ou None)]. u0: distância a partir do
    início onde se começa a tirar (junto às faces a frontaria tapa)."""
    passo = X[0, 1] - X[0, 0]
    cx, cy = X[:-1, :-1] + passo / 2, Y[:-1, :-1] + passo / 2
    zmin = np.minimum.reduce([Z[:-1, :-1], Z[1:, :-1], Z[:-1, 1:], Z[1:, 1:]])
    zmax = np.maximum.reduce([Z[:-1, :-1], Z[1:, :-1], Z[:-1, 1:], Z[1:, 1:]])
    A, B = np.asarray(A, float), np.asarray(B, float)
    L = float(np.hypot(*(B - A)))
    u = (B - A) / L
    cu = (cx - A[0]) * u[0] + (cy - A[1]) * u[1]
    cs = (cx - A[0]) * -u[1] + (cy - A[1]) * u[0]
    piso = za + (zb - za) * np.clip(cu / L, 0, 1)
    bur = (np.abs(cs) < meia) & (cu > u0) & (cu < L + 0.3) & (zmin < piso + fecho + 0.3)
    borda = []
    for a in np.arange(0.0, L, 1.0):
        sel = bur & (cu >= a) & (cu < a + 1.0)
        borda.append((float(a), float(min(a + 1.0, L)), float(zmax[sel].max()) if sel.any() else None))
    return bur, borda


def enchimento(Q, borda, z_chao, dz, topo_min, mat, meia=2.9):
    """Alvenaria por cima de um troço de passagem (no quadro Q, ao longo de -d), da abóbada até à borda do
    buraco no terreno ou até topo_min(de, até); e o alicerce, 4 m por baixo do chão. z_chao(u): chão a u
    metros do início; dz: altura do extradorso da abóbada acima do chão."""
    for a, b, bd in borda:
        z0 = max(z_chao(a), z_chao(b)) + dz
        z1 = max(bd if bd is not None else -1e9, topo_min(a, b))
        if z1 > z0 + 0.1:
            Q.bloco(-meia, meia, z0, z1 + 0.05, -b, -a, mat)
        zp = min(z_chao(a), z_chao(b))
        Q.bloco(-meia, meia, zp - 4.0, zp - 0.02, -b, -a, mat)
