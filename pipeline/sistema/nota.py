"""Regra da nota das maquetas, e congelamento das condições.

A regra (igual à do castillos.site, com as leves explicitadas):
- 8,0 = todas as condições graves (G) cumpridas.
- Cada condição grave que falha tira 0,5.
- Uma vista de controlo que falha inteira tira 1 (e as graves que só dizem respeito a essa vista não tiram outra vez).
- Cada regressão (uma vista que fica pior do que na maqueta publicada antes) conta como uma grave falhada: −0,5.
- Uma comprovação técnica reprovada (tecnico.py) conta como uma grave falhada: −0,5.
- As leves (L) nunca baixam do 8. Com todas as graves cumpridas, as leves sobem a nota: +0,5 por cada quarto das
  leves cumpridas (máximo 10). Não há décimas a olho: a nota sai sempre em meios pontos, só desta conta.
- Nota mínima 0.
Com 8 ou mais publica-se sem perguntar (ver encargos/coordenador.md).

As condições congelam-se quando o coordenador as aprova: a partir daí não entram condições graves novas. Uma
revisão que cite uma condição que não está nas congeladas, ou um condicoes.json mudado depois de congelado, é
recusada (o que não estava nas condições não se pode exigir depois como grave).

Uso:
    python pipeline/sistema/nota.py congelar refs/<castelo>/condicoes.json
    python pipeline/sistema/nota.py calcular refs/<castelo>/condicoes.json refs/<castelo>/volta1/revisao.json
    python pipeline/sistema/nota.py --teste
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


def impressao(cond: dict) -> str:
    """SHA-1 das condições (só as listas que contam para a nota), para ver se mudaram depois de congeladas."""
    chave = {k: cond.get(k) for k in ("vistas", "materiais", "condicoes", "planta", "alturas")}
    return hashlib.sha1(json.dumps(chave, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def congelar(cpath: Path) -> str:
    cond = json.loads(cpath.read_text())
    erros = validar(cond)
    if erros:
        raise SystemExit("Condições inválidas:\n  " + "\n  ".join(erros))
    sha = impressao(cond)
    (cpath.parent / "condicoes.congeladas").write_text(sha + "\n")
    return sha


def validar(cond: dict) -> list[str]:
    erros = []
    ids_v = [v["id"] for v in cond.get("vistas", [])]
    if not 8 <= len(ids_v) <= 12:
        erros.append(f"são precisas 8 a 12 vistas de controlo (há {len(ids_v)})")
    for v in cond.get("vistas", []):
        if not v.get("foto"):
            erros.append(f"vista {v['id']} sem foto de comparação")
        if not v.get("camera"):
            erros.append(f"vista {v['id']} sem câmara (pos, alvo, fov)")
    for m in cond.get("materiais", []):
        if len(m.get("medido_em", [])) < 2:
            erros.append(f"material {m['id']}: a cor tem de ser medida em duas fotos")
        if not m.get("peca_m"):
            erros.append(f"material {m['id']}: falta o tamanho da peça")
        if "argamassa" not in m:
            erros.append(f"material {m['id']}: falta a argamassa (ou 'sem')")
    ids_c = set()
    for c in cond.get("condicoes", []):
        if c["id"] in ids_c:
            erros.append(f"condição repetida {c['id']}")
        ids_c.add(c["id"])
        if c.get("tipo") not in ("G", "L"):
            erros.append(f"condição {c['id']}: tipo tem de ser G ou L")
        for v in c.get("vistas", []):
            if v not in ids_v:
                erros.append(f"condição {c['id']}: vista {v} não existe")
    if not any(c.get("tipo") == "G" for c in cond.get("condicoes", [])):
        erros.append("não há condições graves")
    return erros


def calcular(cond: dict, rev: dict) -> dict:
    """Nota de uma revisão. rev = {condicoes: [{id, cumpre}], vistas_falhadas: [id], regressoes: [...], tecnico_aprovado}."""
    porid = {c["id"]: c for c in cond["condicoes"]}
    estranhas = [r["id"] for r in rev.get("condicoes", []) if r["id"] not in porid]
    if estranhas:
        raise ValueError(f"a revisão cita condições que não estão nas condições congeladas: {estranhas}")
    cumpre = {r["id"]: bool(r["cumpre"]) for r in rev.get("condicoes", [])}
    faltam = [i for i in porid if i not in cumpre]
    if faltam:
        raise ValueError(f"a revisão não avalia todas as condições: faltam {faltam}")
    vf = set(rev.get("vistas_falhadas", []))
    graves = [c for c in cond["condicoes"] if c["tipo"] == "G"]
    leves = [c for c in cond["condicoes"] if c["tipo"] == "L"]
    g_falhadas = [c["id"] for c in graves if not cumpre[c["id"]]]
    # graves cobertas por uma vista falhada inteira (todas as vistas da condição falharam) não tiram outra vez
    g_contadas = [i for i in g_falhadas if not (porid[i].get("vistas") and set(porid[i]["vistas"]) <= vf)]
    regressoes = rev.get("regressoes", [])
    tecnico = 0 if rev.get("tecnico_aprovado", True) else 1
    desconto = 0.5 * len(g_contadas) + 1.0 * len(vf) + 0.5 * len(regressoes) + 0.5 * tecnico
    if desconto == 0:
        quartos = math.floor(4 * sum(cumpre[c["id"]] for c in leves) / len(leves)) if leves else 0
        nota = 8.0 + 0.5 * quartos
    else:
        nota = max(0.0, 8.0 - desconto)
    return {
        "nota": nota,
        "publica": nota >= 8.0,
        "graves_falhadas": g_falhadas,
        "vistas_falhadas": sorted(vf),
        "regressoes": regressoes,
        "tecnico_aprovado": not tecnico,
        "leves_cumpridas": f"{sum(cumpre[c['id']] for c in leves)}/{len(leves)}",
    }


def calcular_ficheiros(cpath: Path, rpath: Path) -> dict:
    cond = json.loads(cpath.read_text())
    cong = cpath.parent / "condicoes.congeladas"
    if not cong.exists():
        raise SystemExit("As condições ainda não foram congeladas (python nota.py congelar ...).")
    if cong.read_text().strip() != impressao(cond):
        raise SystemExit("O condicoes.json mudou depois de congelado. As condições não mudam a meio: reponha a versão congelada.")
    rev = json.loads(rpath.read_text())
    r = calcular(cond, rev)
    rev["resultado"] = r
    rpath.write_text(json.dumps(rev, indent=1, ensure_ascii=False))
    return r


def _teste() -> None:
    cond = {"condicoes": [
        {"id": "G1", "tipo": "G", "vistas": ["V1"]}, {"id": "G2", "tipo": "G", "vistas": ["V2"]},
        {"id": "G3", "tipo": "G", "vistas": ["V1", "V3"]},
        {"id": "L1", "tipo": "L"}, {"id": "L2", "tipo": "L"}, {"id": "L3", "tipo": "L"}, {"id": "L4", "tipo": "L"}]}
    tudo = lambda **f: [{"id": c["id"], "cumpre": f.get(c["id"], True)} for c in cond["condicoes"]]  # noqa: E731
    assert calcular(cond, {"condicoes": tudo()})["nota"] == 10.0
    assert calcular(cond, {"condicoes": tudo(L1=False)})["nota"] == 9.5
    assert calcular(cond, {"condicoes": tudo(L1=False, L2=False, L3=False, L4=False)})["nota"] == 8.0
    assert calcular(cond, {"condicoes": tudo(G1=False)})["nota"] == 7.5
    assert calcular(cond, {"condicoes": tudo(G1=False, L1=False)})["nota"] == 7.5  # as leves não contam abaixo de 8
    # V1 falhada inteira: −1; G1 (só V1) não conta outra vez; G3 (V1 e V3) conta
    assert calcular(cond, {"condicoes": tudo(G1=False, G3=False), "vistas_falhadas": ["V1"]})["nota"] == 6.5
    assert calcular(cond, {"condicoes": tudo(), "regressoes": ["V2 pior que a publicada"]})["nota"] == 7.5
    assert calcular(cond, {"condicoes": tudo(), "tecnico_aprovado": False})["nota"] == 7.5
    for mau in ({"condicoes": tudo() + [{"id": "G9", "cumpre": False}]}, {"condicoes": tudo()[1:]}):
        try:
            calcular(cond, mau)
            raise AssertionError("devia recusar")
        except ValueError:
            pass
    print("nota.py: testes passam")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--teste"]:
        _teste()
    elif a[:1] == ["congelar"] and len(a) == 2:
        print("congeladas:", congelar(Path(a[1])))
    elif a[:1] == ["calcular"] and len(a) == 3:
        print(json.dumps(calcular_ficheiros(Path(a[1]), Path(a[2])), indent=1, ensure_ascii=False))
    else:
        print(__doc__)
        sys.exit(2)
