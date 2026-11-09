"""Reconstrói as maquetas e confirma que os GLB saem iguais aos de referência.

Para mexer no kit (ou numa maqueta) sem mudar o resultado sem querer:
    python pipeline/kit/verificar.py               # todas
    python pipeline/kit/verificar.py penela braga  # só estas
    python pipeline/kit/verificar.py --gravar      # aceita o resultado atual como nova referência

Compara a impressão digital (SHA-1) do GLB antes do Draco (saida/<id>-bruto.glb) e do dados/medidas.json
de cada maqueta com impressoes.json. Os GLB finais em public/ não entram na comparação: o Draco do
glTF Transform não sai sempre igual. As impressões dependem das versões do Blender e das bibliotecas
(ver "ambiente" em impressoes.json): noutras versões, grave uma referência nova antes de mexer no código.
Corre os construir.py com o mesmo Python (com o bpy instalado); o construir.py reescreve
public/maquetas/<id>/<id>.glb e dados/medidas.json, que o git mostra como alterados se mudarem.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
MAQ = AQUI.parent / "maquetas"
REF = AQUI / "impressoes.json"


def sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def ambiente() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ["bpy", "numpy", "scipy", "shapely", "rasterio", "pyproj", "matplotlib", "PIL"]:
        try:
            m = __import__(mod)
            out[mod] = m.app.version_string if mod == "bpy" else m.__version__
        except ImportError:
            out[mod] = None
    return out


def main() -> int:
    gravar = "--gravar" in sys.argv
    ids = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(p.name for p in MAQ.iterdir() if (p / "construir.py").exists())
    ref = json.loads(REF.read_text()) if REF.exists() else {"maquetas": {}}
    falhas = 0
    for m in ids:
        r = subprocess.run([sys.executable, str(MAQ / m / "construir.py")], capture_output=True, text=True)
        if r.returncode:
            print(f"{m}: o construir.py falhou\n{r.stderr[-1500:]}")
            falhas += 1
            continue
        atual = {"glb": sha1(next((MAQ / m / "saida").glob("*-bruto.glb"))), "medidas": sha1(MAQ / m / "dados" / "medidas.json")}
        esperado = ref["maquetas"].get(m)
        if gravar:
            ref["maquetas"][m] = atual
            print(f"{m}: referência gravada")
        elif esperado == atual:
            print(f"{m}: igual")
        else:
            print(f"{m}: DIFERENTE (esperado {esperado}, obtido {atual})")
            falhas += 1
    if gravar:
        ref["ambiente"] = ambiente()
        REF.write_text(json.dumps(ref, indent=1, ensure_ascii=False) + "\n")
    elif falhas:
        print(f"\n{falhas} maqueta(s) diferente(s) ou com erro. Ambiente da referência: {ref.get('ambiente')}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
