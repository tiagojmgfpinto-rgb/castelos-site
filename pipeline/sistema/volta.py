"""Uma ordem por volta: constrói a maqueta, passa as comprovações técnicas e gera láminas, cores e diferenças.

Uso:
    python pipeline/sistema/volta.py CASTELO --volta 1 [--fase forma|completa] [--sem-construir] [--antes origin/main]

Passos (cada um grava em REFS/<castelo>/volta<N>/<fase>/):
 1. construir   python pipeline/maquetas/<castelo>/construir.py (Blender/bpy). --sem-construir usa o GLB que já está
                em public/maquetas/<castelo>/.
 2. vitrina     vistas de controlo do condicoes.json (com máscaras) e vistas gerais, na luz do site, com o GLB novo.
 3. tecnico     tecnico.py no GLB sem Draco (o saida/<id>-bruto.glb do construir.py ou o que a vitrina exporta).
 4. antes       as mesmas vistas com o GLB publicado (git show <antes>:public/maquetas/...), e o mapa das diferenças.
 5. laminas     foto | maqueta | mistura 50 % | contorno, recortes de perto e, na fase completa, cores (ΔE2000).
 6. resumo      resumo.md (o que correu, números, falhas) e o esqueleto do autoexame.md, se ainda não existir.
Sai com código 1 se as comprovações técnicas reprovarem.

REFS: variável de ambiente REFS ou /mnt/project-files/refs (a pasta partilhada do projeto: fotos de terceiros
não entram no git).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
REFS = Path(os.environ.get("REFS", "/mnt/project-files/refs"))


def correr(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def vitrina(castelo: str, glb: Path, vistas: str | Path, saida: Path, *extra: str) -> None:
    r = correr(["node", AQUI / "vitrina/vitrina.mjs", "--maqueta", castelo, "--glb", glb, "--vistas", vistas, "--saida", saida, *extra])
    if r.returncode:
        raise SystemExit(f"a vitrina falhou ({vistas})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("castelo")
    ap.add_argument("--volta", type=int, required=True)
    ap.add_argument("--fase", choices=["forma", "completa"], default="completa")
    ap.add_argument("--sem-construir", action="store_true")
    ap.add_argument("--antes", default="origin/main", help="referência git da maqueta publicada (para as diferenças)")
    a = ap.parse_args()

    c = a.castelo
    pasta = REFS / c
    cpath = pasta / "condicoes.json"
    if not cpath.exists():
        raise SystemExit(f"Falta {cpath}. O revisor escreve as condições antes de se modelar (modelos/condicoes.json).")
    if not (pasta / "condicoes.congeladas").exists():
        print("AVISO: as condições ainda não foram aprovadas e congeladas pelo coordenador (nota.py congelar).")
    out = pasta / f"volta{a.volta}" / a.fase
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time()
    tempos: dict[str, float] = {}
    resumo: list[str] = [f"# {c}: volta {a.volta}, fase {a.fase}", ""]

    # 1. construir
    maq = REPO / "pipeline/maquetas" / c
    glb = REPO / "public/maquetas" / c / f"{c}.glb"
    if not a.sem_construir:
        if not (maq / "construir.py").exists():
            raise SystemExit(f"Falta {maq / 'construir.py'}.")
        try:
            import bpy  # noqa: F401
        except ImportError:
            raise SystemExit("Sem bpy neste Python (pip install bpy==5.2.3 numpy scipy shapely rasterio pyproj matplotlib pillow, "
                             "Python 3.13), ou construir no Blender do computador do Tiago e correr com --sem-construir.")
        r = correr([sys.executable, maq / "construir.py"], cwd=REPO)
        if r.returncode:
            raise SystemExit("o construir.py falhou")
    tempos["construir"] = time.time() - t0
    if not glb.exists():
        raise SystemExit(f"Não há GLB em {glb}.")

    # 2. vitrina
    t = time.time()
    vit = out / "vitrina"
    bruto = next((maq / "saida").glob("*-bruto.glb"), None) if (maq / "saida").exists() and not a.sem_construir else None
    extra = ["--mascaras", "--contagens", out / "contagens.json"]
    if bruto is None:
        bruto = out / f"{c}-bruto.glb"
        extra += ["--bruto", bruto]
    vitrina(c, glb, cpath, vit, *extra)
    vitrina(c, glb, "geral", vit / "gerais")
    tempos["vitrina"] = time.time() - t

    # 3. tecnico
    t = time.time()
    r = correr([sys.executable, AQUI / "tecnico.py", bruto, "--saida", out / "tecnico.json"])
    tec = json.loads((out / "tecnico.json").read_text())
    tempos["tecnico"] = time.time() - t

    # 4. antes (maqueta publicada) e diferenças
    t = time.time()
    antes_glb = out / "antes.glb"
    g = subprocess.run(["git", "show", f"{a.antes}:public/maquetas/{c}/{c}.glb"], cwd=REPO, capture_output=True)
    dif = {}
    if g.returncode == 0 and g.stdout:
        antes_glb.write_bytes(g.stdout)
        vitrina(c, antes_glb, cpath, out / "antes")
        vitrina(c, antes_glb, vit / "gerais/gerais.json", out / "antes/gerais")  # mesmas câmaras gerais
        correr([sys.executable, AQUI / "diferencas.py", out / "antes", vit, out / "diferencas"])
        correr([sys.executable, AQUI / "diferencas.py", out / "antes/gerais", vit / "gerais", out / "diferencas/gerais"])
        for f in [out / "diferencas/diferencas.json", out / "diferencas/gerais/diferencas.json"]:
            if f.exists():
                dif.update(json.loads(f.read_text()))
        antes_glb.unlink()
    else:
        resumo.append(f"Sem maqueta publicada em {a.antes}: castelo novo, sem comparação antes/depois.\n")
    tempos["antes"] = time.time() - t

    # 5. láminas e cores
    t = time.time()
    correr([sys.executable, AQUI / "laminas.py", cpath, vit, out / "laminas", "--refs", pasta])
    cores = json.loads((out / "laminas/cores.json").read_text()) if a.fase == "completa" else []
    tempos["laminas"] = time.time() - t

    # 6. resumo
    cont = json.loads((out / "contagens.json").read_text())
    resumo += [
        f"- Triângulos: {tec.get('triangulos')} (obra {tec.get('triangulos_por_categoria', {}).get('obra')}; alvo Guadalupe: 131 mil na obra)",
        f"- Materiais: {len(cont.get('materiais', []))} (alvo: um por elemento arquitetónico; Guadalupe tem 73)",
        f"- Comprovações técnicas: {'APROVADAS' if tec.get('aprovado') else 'REPROVADAS'}",
    ]
    resumo += [f"  - {f}" for f in tec.get("falhas", [])] + [f"  - aviso: {f}" for f in tec.get("avisos", [])]
    if dif:
        resumo.append("- Diferenças em relação à publicada (% de píxeis mudados por vista; ver diferencas/):")
        resumo += [f"  - {k}: {v:.1f} %" for k, v in dif.items()]
    if cores:
        resumo.append("- Cores medidas na vitrina (ΔE2000 em relação ao alvo):")
        resumo += [f"  - {x['id']} {x['nome']}: maqueta {x['maqueta']}, alvo {x['alvo']}, ΔE {x.get('dE2000')} {'ok' if x.get('cumpre') else 'FORA'}" for x in cores]
    resumo.append("- Tempos (s): " + ", ".join(f"{k} {v:.0f}" for k, v in tempos.items()))
    (out / "resumo.md").write_text("\n".join(resumo) + "\n")

    auto = pasta / f"volta{a.volta}" / "autoexame.md"
    if not auto.exists():
        cond = json.loads(cpath.read_text())
        linhas = [f"# Autoexame: {c}, volta {a.volta}", "", "Preenche o modelador. Cada linha com a prova (lámina, recorte de perto, medida). Ver modelos/autoexame.md.", "",
                  "## Vistas de controlo", ""]
        linhas += [f"- [ ] {v['id']} {v.get('nome', '')}: lámina `{a.fase}/laminas/{v['id']}.jpg`. Diferenças que ainda vejo: …" for v in cond.get("vistas", [])]
        linhas += ["", "## Materiais", ""]
        linhas += [f"- [ ] {m['id']} {m.get('nome', '')}: cor medida na vitrina …, alvo {m.get('cor_alvo')}, peça {m.get('peca_m')} m, argamassa …" for m in cond.get("materiais", [])]
        linhas += ["", "## Condições", ""]
        linhas += [f"- [ ] {x['id']} [{x['tipo']}] {x['texto']}: cumpre? prova: …" for x in cond.get("condicoes", [])]
        linhas += ["", "## Comprovações técnicas", "", f"Ver `{a.fase}/tecnico.json`. Explicar cada aviso que fica.", "",
                   "## Mudanças fora da lista (só na volta 2)", "", "Cada uma explicada, com o mapa de `diferencas/`.", ""]
        auto.write_text("\n".join(linhas))
    print("\n".join(resumo))
    print(f"\nTudo em {out}")
    return 0 if tec.get("aprovado") else 1


if __name__ == "__main__":
    sys.exit(main())
