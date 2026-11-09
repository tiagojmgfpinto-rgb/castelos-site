import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const cache = new Map<string, string>();

/**
 * Acrescenta `?v=<hash>` ao endereço de um ficheiro de public/ (GLB, capa), calculado do conteúdo na compilação.
 * Quando uma maqueta muda, o endereço muda e o browser não mostra a versão antiga guardada em cache.
 * Só corre no servidor (páginas Astro e lib/monumentos.ts).
 */
export function comVersao(ficheiro: string): string;
export function comVersao(ficheiro: string | undefined): string | undefined;
export function comVersao(ficheiro: string | undefined) {
  if (!ficheiro || ficheiro.includes("?")) return ficheiro;
  if (!cache.has(ficheiro)) {
    let v = "";
    try {
      v = createHash("sha1").update(readFileSync(join(process.cwd(), "public", ficheiro))).digest("hex").slice(0, 10);
    } catch {
      /* ficheiro em falta: fica sem versão */
    }
    cache.set(ficheiro, v ? `${ficheiro}?v=${v}` : ficheiro);
  }
  return cache.get(ficheiro);
}
