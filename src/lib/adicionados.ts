import { execSync } from "node:child_process";
import { basename } from "node:path";

// Data de publicação dos monumentos que ainda não a têm escrita na ficha (campo `adicionado`).
// Vem do histórico de main: o merge que acrescentou o ficheiro é o momento em que ele entrou no site.
// Só corre na compilação. O CI faz checkout do histórico completo (fetch-depth: 0); num clone raso as datas
// seriam falsas, por isso aí não se usa o git.

let cache: Map<string, string> | null = null;
const inicioDaCompilacao = new Date().toISOString().replace(/\.\d+Z$/, "Z");

function doGit(): Map<string, string> {
  const r = new Map<string, string>();
  try {
    if (execSync("git rev-parse --is-shallow-repository", { encoding: "utf8" }).trim() === "true") return r;
    const out = execSync("git log --first-parent --diff-filter=A --format=@%cI --name-only -- src/content/monumentos", { encoding: "utf8", maxBuffer: 1 << 24 });
    let data = "";
    for (const linha of out.split("\n")) {
      if (linha.startsWith("@")) data = new Date(linha.slice(1)).toISOString().replace(/\.\d+Z$/, "Z");
      else if (linha.endsWith(".md")) r.set(basename(linha, ".md"), data); // o mais antigo fica por último e ganha
    }
  } catch {
    /* sem git: fica a hora da compilação */
  }
  return r;
}

/** Data de publicação (ISO, UTC): a escrita na ficha; senão, a do merge em main; senão, a desta compilação. */
export function adicionadoEm(id: string, escrito?: string): string {
  if (escrito) return escrito;
  cache ??= doGit();
  return cache.get(id) ?? inicioDaCompilacao;
}
