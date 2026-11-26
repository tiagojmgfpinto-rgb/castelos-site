"""Comprovações técnicas de uma maqueta (GLB sem Draco), com limites que reprovam a volta.

Vem da auditoria técnica de novembro de 2026 (auditoria-maquetas/ferramentas/auditar.py), com limites e
a verificação da peana. Eixos do GLB: x este, y cima, z sul. Mede:
- triângulos e materiais por categoria (obra, chão, terreno, vegetação), para comparar com o alvo (Guadalupe:
  131 mil triângulos na obra, um material por elemento arquitetónico);
- árvores a flutuar, enterradas ou a atravessar muros e edifícios;
- peças de obra a flutuar (sem nada por baixo) ou totalmente enterradas;
- faces coplanares sobrepostas entre objetos diferentes (z-fighting: piscam no visor);
- normais invertidas; texturas esticadas;
- peças fora da peana (para lá do disco/retângulo do terreno).

Uso:
    python pipeline/sistema/tecnico.py MAQUETA.glb [--saida tecnico.json]
O GLB tem de vir sem Draco: o saida/<id>-bruto.glb do construir.py, ou o que a vitrina grava com --bruto.
Sai com código 1 se algum limite falhar (LIMITES abaixo).
"""
import collections
import json
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# Avisos: contam-se e mostram-se, mas não reprovam (o agrupamento das copas em bosques densos dá falsos positivos;
# confirmar na vitrina com uma vista de perto antes de mexer)
AVISOS = {"arvores/a_flutuar_>0.5m": 0, "arvores/enterradas_>1.5m": 0, "texturas_esticadas": 0}
LIMITES = {
    "arvores/a_atravessar_obra": 0,
    "obra_a_flutuar": 0,
    "obra_enterrada": 0,
    "zfighting": 0,
    "fora_da_peana": 0,
    "normais_invertidas": 0,
}


def categoria(nome):
    n = nome.lower()
    if n.startswith(("vegetacao", "arvores")):
        return "veg"
    if n.startswith(("terreno", "base", "afloramentos", "rochedos")):
        return "terreno"
    if n.startswith(("chao_", "caminhos", "praca", "relvados", "adro", "pavimento")):
        return "chao"
    return "obra"


def intersector(m):
    try:
        return trimesh.ray.ray_pyembree.RayMeshIntersector(m)
    except Exception:  # sem embree: muito mais lento, mas dá o mesmo resultado
        return trimesh.ray.ray_triangle.RayMeshIntersector(m)


def carregar(caminho):
    s = trimesh.load(caminho, force="scene")
    partes = collections.defaultdict(list)
    for nd in s.graph.nodes_geometry:
        T, gname = s.graph[nd]
        g = s.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        m = g.copy()
        m.apply_transform(T)
        partes[nd].append((gname, m))
    return s, partes


def analisar(caminho):
    s, partes = carregar(caminho)
    res = {"glb": str(caminho)}
    cat = {nd: categoria(nd) for nd in partes}
    res["triangulos"] = int(sum(m.faces.shape[0] for v in partes.values() for _, m in v))
    res["triangulos_por_categoria"] = {c: int(sum(m.faces.shape[0] for nd, v in partes.items() if cat[nd] == c for _, m in v)) for c in ("obra", "chao", "terreno", "veg")}
    mats = collections.defaultdict(set)
    for nd, v in partes.items():
        for _, m in v:
            mat = getattr(getattr(m, "visual", None), "material", None)
            mats[cat[nd]].add(getattr(mat, "name", None) or "?")
    res["materiais_por_categoria"] = {c: len(v) for c, v in mats.items()}

    def junta(c):
        ms = [m for nd, v in partes.items() if cat[nd] == c for _, m in v]
        return trimesh.util.concatenate(ms) if ms else None
    terr = junta("terreno")
    chao = junta("chao")
    veg = junta("veg")
    obra = junta("obra")
    if terr is None:
        res["erro"] = "sem terreno"
        return res
    solo_tudo = trimesh.util.concatenate([terr] + ([chao] if chao is not None else []))
    # Só as faces viradas para cima: o fundo e a saia da peana não contam como chão (por baixo dos muros o chão
    # por classes tem buracos, e o raio ia bater no fundo da peana, dezenas de metros abaixo)
    solo = solo_tudo.submesh([np.where(solo_tudo.face_normals[:, 1] > 0.1)[0]], append=True)
    rs = intersector(solo)
    ro = intersector(obra) if obra is not None else None
    tudo_nveg = trimesh.util.concatenate([solo] + ([obra] if obra is not None else []))
    rt = intersector(tudo_nveg)
    TOP = float(tudo_nveg.bounds[1][1]) + 50

    def altura(r, xz):
        o = np.column_stack([xz[:, 0], np.full(len(xz), TOP), xz[:, 1]])
        d = np.tile([0, -1, 0], (len(xz), 1))
        loc, idx, _ = r.intersects_location(o, d, multiple_hits=False)
        h = np.full(len(xz), np.nan)
        h[idx] = loc[:, 1]
        return h

    # ---------- peana ----------
    tv = solo_tudo.vertices[:, [0, 2]]
    c0 = (tv.min(0) + tv.max(0)) / 2
    rmax = np.linalg.norm(tv - c0, axis=1).max()
    redonda = np.percentile(np.linalg.norm(tv - c0, axis=1), 99.5) > 0.97 * rmax and abs(np.ptp(tv[:, 0]) - np.ptp(tv[:, 1])) < 0.02 * rmax
    fora = []
    for nd, v in partes.items():
        if cat[nd] == "terreno":
            continue
        for _, m in v:
            p = m.vertices[:, [0, 2]]
            if redonda:
                d = np.linalg.norm(p - c0, axis=1) - rmax
            else:
                d = np.maximum((tv.min(0) - p).max(1), (p - tv.max(0)).max(1))
            if (d > 0.2).any():
                fora.append((nd, round(float(d.max()), 1)))
    res["peana"] = {"forma": "redonda" if redonda else "retangular", "raio": round(float(rmax), 1)}
    res["fora_da_peana"] = len(fora)
    res["ex_fora_da_peana"] = sorted(fora, key=lambda t: -t[1])[:10]

    # ---------- árvores ----------
    arv = {}
    if veg is not None:
        comps = veg.split(only_watertight=False)
        cen = np.array([c.bounds.mean(0) for c in comps])
        # agrupar componentes (tronco + lóbulos) por proximidade em planta
        tree = cKDTree(cen[:, [0, 2]])
        grupos = -np.ones(len(comps), int)
        g = 0
        for i in range(len(comps)):
            if grupos[i] >= 0: continue
            pilha = [i]; grupos[i] = g
            while pilha:
                j = pilha.pop()
                r = max(1.0, 0.5 * (comps[j].bounds[1][[0, 2]] - comps[j].bounds[0][[0, 2]]).max())
                for k in tree.query_ball_point(cen[j, [0, 2]], r):
                    if grupos[k] < 0:
                        grupos[k] = g; pilha.append(k)
            g += 1
        arvores = [trimesh.util.concatenate([comps[i] for i in np.where(grupos == k)[0]]) for k in range(g)]
        arv['n'] = len(arvores)
        base = np.array([[a.vertices[a.vertices[:, 1].argmin()][0], a.vertices[:, 1].min(), a.vertices[a.vertices[:, 1].argmin()][2]] for a in arvores])
        hs = altura(rs, base[:, [0, 2]])
        ho = altura(rt, base[:, [0, 2]])  # inclui obra (árvore em cima de um muro)
        gap = base[:, 1] - np.fmax(hs, np.nan_to_num(ho, nan=-1e9))
        arv['a_flutuar_>0.5m'] = int(np.nansum(gap > 0.5))
        arv['enterradas_>1.5m'] = int(np.nansum(base[:, 1] - hs < -1.5))
        exemplos_f = [tuple(np.round(base[i, [0, 2]], 1)) + (round(float(gap[i]), 1),) for i in np.where(gap > 0.5)[0][:6]]
        arv['ex_flutuar'] = exemplos_f
        # atravessar obra: vértices da árvore acima do solo e abaixo do topo da obra, na mesma vertical
        cruz = []
        if ro is not None:
            for i, a in enumerate(arvores):
                v = a.vertices[::max(1, len(a.vertices) // 120)]
                top = altura(ro, v[:, [0, 2]])
                sol = altura(rs, v[:, [0, 2]])
                dentro = (~np.isnan(top)) & (v[:, 1] < top - 0.3) & (v[:, 1] > np.nan_to_num(sol, nan=-1e9) + 0.3)
                fr = dentro.mean()
                if fr > 0.08:
                    c = a.bounds.mean(0)
                    cruz.append((round(float(c[0]), 1), round(float(c[2]), 1), round(float(fr), 2)))
        arv['a_atravessar_obra'] = len(cruz)
        arv['ex_atravessar'] = sorted(cruz, key=lambda t: -t[2])[:8]
    res['arvores'] = arv

    # ---------- peças de obra a flutuar / enterradas ----------
    flut, ent = [], []
    comps, donos = [solo], ['(solo)']
    for nd, v in partes.items():
        if cat[nd] != 'obra': continue
        m = trimesh.util.concatenate([x for _, x in v])
        m = trimesh.Trimesh(m.vertices.copy(), m.faces.copy(), process=True)
        m.merge_vertices(digits_vertex=3)
        for c in m.split(only_watertight=False):
            comps.append(c); donos.append(nd)
    off = np.cumsum([0] + [len(c.faces) for c in comps])
    glob = trimesh.util.concatenate(comps)
    rg = intersector(glob)
    O, D, quem = [], [], []
    for k in range(1, len(comps)):
        c = comps[k]
        if len(c.vertices) < 4 or max(c.extents) < 0.6: continue
        ymin = c.vertices[:, 1].min()
        low = c.vertices[c.vertices[:, 1] < ymin + 0.05]
        low = low[:: max(1, len(low) // 12)]
        for p in low:
            O.append([p[0], ymin + 0.5, p[2]]); quem.append(k)
    O = np.array(O); quem = np.array(quem)
    loc, ir, it = rg.intersects_location(O, np.tile([0, -1, 0], (len(O), 1)), multiple_hits=True)
    ctri = np.searchsorted(off, it, side='right') - 1
    outro = ctri != quem[ir]
    sup = np.full(len(O), -np.inf)
    np.maximum.at(sup, ir[outro], loc[outro, 1])
    gaps = (O[:, 1] - 0.5) - sup
    gmin = collections.defaultdict(lambda: np.inf)
    for k, g in zip(quem, gaps):
        gmin[k] = min(gmin[k], g)
    NS = int(min(4e6, glob.area * 12))
    sp, sf = trimesh.sample.sample_surface(glob, NS)
    sown = np.searchsorted(off, sf, side='right') - 1
    kds = cKDTree(sp)
    for k, g in list(gmin.items()):
        if g > 0.3:
            c = comps[k]
            low = c.vertices[c.vertices[:, 1] < c.vertices[:, 1].min() + 0.05]
            viz = kds.query_ball_point(low[::max(1, len(low)//12)], 0.4)
            if any((sown[l] != k).any() for l in viz if len(l)):
                gmin[k] = 0.0
    for k, g in list(gmin.items()):
        if g > 0.3:
            c = comps[k]
            # chão à volta dos vértices de baixo (a peça pode estar num buraco do chão por classes): toca se o
            # ponto mais baixo da peça não ficar acima do chão mais alto dessa coroa
            y0 = c.vertices[:, 1].min()
            low = c.vertices[c.vertices[:, 1] < y0 + 0.05][:, [0, 2]]
            low = low[:: max(1, len(low) // 24)]
            ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            off = np.concatenate([np.column_stack([np.cos(ang), np.sin(ang)]) * r for r in (0.75, 1.5, 3.0)])
            ht = altura(rs, (low[:, None, :] + off[None]).reshape(-1, 2))
            if np.isfinite(ht).any() and y0 <= np.nanmax(ht) + 0.3:
                gmin[k] = 0.0
    for k, g in gmin.items():
        if g > 0.3:
            c = comps[k]; cc = c.bounds.mean(0)
            flut.append((donos[k], round(float(cc[0]), 1), round(float(cc[2]), 1), round(float(g), 2) if np.isfinite(g) else 'sem chao', round(float(max(c.extents)), 1)))
    for k in range(1, len(comps)):
        c = comps[k]
        if max(c.extents) < 0.6: continue
        # amostra a passo fixo mais o vértice mais alto: num pano com 4 vértices novos por segmento (2 no pé, 2 no
        # adarve) o passo par apanhava só os do pé, 1,5 m abaixo do chão, e dava o pano inteiro como enterrado
        vv = np.vstack([c.vertices[::max(1, len(c.vertices) // 40)], c.vertices[[int(np.argmax(c.vertices[:, 1]))]]])
        hv = altura(rs, vv[:, [0, 2]])
        ok = ~np.isnan(hv)
        if ok.sum() > 3 and np.all(vv[ok, 1] < hv[ok] - 0.05):
            cc = c.bounds.mean(0)
            ent.append((donos[k], round(float(cc[0]), 1), round(float(cc[2]), 1), round(float(max(c.extents)), 1)))
    res['obra_a_flutuar'] = len(flut)
    res['ex_obra_a_flutuar'] = sorted(flut, key=lambda t: -(t[3] if isinstance(t[3], float) else 99))[:12]
    res['obra_enterrada'] = len(ent)
    res['ex_obra_enterrada'] = ent[:10]

    # ---------- z-fighting entre objetos diferentes ----------
    pts, nrm, dono = [], [], []
    nomes = [nd for nd in partes if cat[nd] in ('obra', 'chao', 'terreno')]
    for k, nd in enumerate(nomes):
        m = trimesh.util.concatenate([x for _, x in partes[nd]])
        N = int(min(20000, max(300, m.area * 2)))
        p, fi = trimesh.sample.sample_surface_even(m, N) if m.area > 1 else (m.vertices, np.zeros(len(m.vertices), int))
        pts.append(p); nrm.append(m.face_normals[fi]); dono.append(np.full(len(p), k))
    pts = np.vstack(pts); nrm = np.vstack(nrm); dono = np.concatenate(dono)
    kd = cKDTree(pts)
    pares = kd.query_pairs(0.015, output_type='ndarray')
    zf = collections.Counter()
    if len(pares):
        a, b = pares[:, 0], pares[:, 1]
        sel = (dono[a] != dono[b]) & (np.einsum('ij,ij->i', nrm[a], nrm[b]) > 0.98)
        for i, j in pares[sel]:
            key = tuple(sorted((nomes[dono[i]], nomes[dono[j]])))
            zf[key] += 1
    res['zfighting'] = [(a, b, c) for (a, b), c in zf.most_common(10) if c >= 5]

    # ---------- normais invertidas ----------
    inv = []
    for nd, v in partes.items():
        for _, m in v:
            if 'vertex_normals' not in m._cache.cache and m.vertex_normals is None: continue
            vn = m.vertex_normals[m.faces].mean(1)
            fn = m.face_normals
            d = np.einsum('ij,ij->i', vn, fn)
            frac = float((d < -0.3).mean())
            if frac > 0.02:
                inv.append((nd, round(frac, 3), int((d < -0.3).sum())))
    res['normais_invertidas'] = inv

    # ---------- texturas esticadas ----------
    est = []
    for nd, v in partes.items():
        if cat[nd] == 'veg': continue
        for _, m in v:
            uv = getattr(m.visual, 'uv', None)
            if uv is None or len(uv) != len(m.vertices): continue
            t = uv[m.faces]
            e1=t[:, 1] - t[:, 0]; e2=t[:, 2] - t[:, 0]; a_uv = 0.5 * np.abs(e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0])
            a3 = m.area_faces
            ok = a3 > 0.05
            if ok.sum() < 10: continue
            r = a_uv[ok] / a3[ok]
            med = np.median(r)
            if med <= 0: continue
            mau = (r < med / 9) | (r > med * 9)
            fr = float((a3[ok][mau]).sum() / a3[ok].sum())
            # alongamento anisotrópico
            if fr > 0.05:
                est.append((nd, round(fr, 2)))
    res['texturas_esticadas'] = est
    return res


def avaliar(res, limites=None):
    falhas = []
    for chave, lim in (limites or LIMITES).items():
        v = res
        for k in chave.split("/"):
            v = v.get(k, 0) if isinstance(v, dict) else 0
        n = len(v) if isinstance(v, list) else v
        if n > lim:
            falhas.append(f"{chave}: {n} (limite {lim})")
    return falhas


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    res = analisar(argv[0])
    res["falhas"] = avaliar(res)
    res["avisos"] = avaliar(res, AVISOS)
    res["aprovado"] = not res["falhas"]
    txt = json.dumps(res, ensure_ascii=False, indent=1, default=str)
    if "--saida" in argv:
        open(argv[argv.index("--saida") + 1], "w").write(txt)
    print(txt if "--saida" not in argv else ("APROVADO" if res["aprovado"] else "REPROVADO:\n  " + "\n  ".join(res["falhas"])) + "".join(f"\n  aviso: {a}" for a in res["avisos"]))
    return 0 if res["aprovado"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
