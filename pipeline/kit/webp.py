"""Passa as texturas JPEG/PNG de um GLB a WebP (extensão EXT_texture_webp), sem tocar na geometria.

Uso:  python pipeline/kit/webp.py public/maquetas/<id>/<id>.glb [--qualidade 80] [--saida outro.glb]

Corre depois do passo Draco (gltf-transform). Só lê e reescreve o contentor GLB: os bufferViews da
geometria Draco são copiados byte a byte; só as imagens mudam. Cada imagem só é trocada se o WebP
ficar pelo menos 5 % mais pequeno; caso contrário fica como estava. Imagens com transparência são
gravadas em WebP com canal alfa. Correr duas vezes sobre o mesmo ficheiro não faz nada à segunda.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

from PIL import Image

EXT = "EXT_texture_webp"
GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942


def ler_glb(dados: bytes) -> tuple[dict, bytes]:
    magic, versao, _ = struct.unpack_from("<III", dados, 0)
    if magic != GLB_MAGIC or versao != 2:
        raise ValueError("não é um GLB 2.0")
    gltf, binario, pos = None, b"", 12
    while pos < len(dados):
        tamanho, tipo = struct.unpack_from("<II", dados, pos)
        corpo = dados[pos + 8 : pos + 8 + tamanho]
        if tipo == JSON_CHUNK:
            gltf = json.loads(corpo)
        elif tipo == BIN_CHUNK:
            binario = corpo
        pos += 8 + tamanho
    if gltf is None:
        raise ValueError("GLB sem JSON")
    return gltf, binario


def escrever_glb(gltf: dict, binario: bytes) -> bytes:
    js = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode()
    js += b" " * (-len(js) % 4)
    binario += b"\0" * (-len(binario) % 4)
    total = 12 + 8 + len(js) + (8 + len(binario) if binario else 0)
    out = struct.pack("<III", GLB_MAGIC, 2, total) + struct.pack("<II", len(js), JSON_CHUNK) + js
    if binario:
        out += struct.pack("<II", len(binario), BIN_CHUNK) + binario
    return out


def para_webp(raw: bytes, qualidade: int) -> bytes:
    img = Image.open(io.BytesIO(raw))
    alfa = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA" if alfa else "RGB")
    b = io.BytesIO()
    img.save(b, "WEBP", quality=qualidade, method=6, alpha_quality=100)
    return b.getvalue()


def converter(dados: bytes, qualidade: int = 80) -> tuple[bytes, list[tuple[int, int, int]]]:
    gltf, binario = ler_glb(dados)
    vistas = gltf.get("bufferViews", [])
    novos: dict[int, bytes] = {}  # índice do bufferView -> bytes novos
    trocadas: set[int] = set()
    relatorio = []
    for i, im in enumerate(gltf.get("images", [])):
        if im.get("mimeType") not in ("image/jpeg", "image/png") or "bufferView" not in im:
            continue
        bv = vistas[im["bufferView"]]
        o = bv.get("byteOffset", 0)
        raw = binario[o : o + bv["byteLength"]]
        w = para_webp(raw, qualidade)
        if len(w) > 0.95 * len(raw):
            continue
        novos[im["bufferView"]] = w
        im["mimeType"] = "image/webp"
        trocadas.add(i)
        relatorio.append((i, len(raw), len(w)))
    if not trocadas:
        return dados, relatorio

    # Texturas que apontam para imagens trocadas passam a usar a extensão.
    for t in gltf.get("textures", []):
        if t.get("source") in trocadas:
            t.setdefault("extensions", {})[EXT] = {"source": t.pop("source")}
    for chave in ("extensionsUsed", "extensionsRequired"):
        lista = gltf.setdefault(chave, [])
        if EXT not in lista:
            lista.append(EXT)

    # Volta a empacotar o buffer pela ordem dos bufferViews, alinhado a 4 bytes.
    partes, pos = [], 0
    for k, bv in enumerate(vistas):
        if bv.get("buffer", 0) != 0:
            raise ValueError("só se suportam GLB com um único buffer")
        o = bv.get("byteOffset", 0)
        corpo = novos.get(k, binario[o : o + bv["byteLength"]])
        pad = -pos % 4
        partes.append(b"\0" * pad)
        pos += pad
        bv["byteOffset"] = pos
        bv["byteLength"] = len(corpo)
        partes.append(corpo)
        pos += len(corpo)
    novo_bin = b"".join(partes)
    gltf["buffers"][0]["byteLength"] = len(novo_bin)
    return escrever_glb(gltf, novo_bin), relatorio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("glb", type=Path, nargs="+")
    ap.add_argument("--qualidade", type=int, default=80)
    ap.add_argument("--saida", type=Path, help="só com um GLB; por omissão reescreve o próprio ficheiro")
    a = ap.parse_args()
    if a.saida and len(a.glb) > 1:
        ap.error("--saida só com um GLB")
    for f in a.glb:
        antes = f.read_bytes()
        depois, rel = converter(antes, a.qualidade)
        (a.saida or f).write_bytes(depois)
        print(f"{f.name}: {len(antes) / 1e3:.0f} kB -> {len(depois) / 1e3:.0f} kB ({len(rel)} imagens em WebP)")


if __name__ == "__main__":
    main()
