import { resumo, todosMonumentos } from "../../lib/monumentos";
import { paraAdicionado } from "../../components/ui/Recentes";

// Lista de todos os monumentos com a data de publicação. O painel "Adicionados recentemente" do mapa lê-a de
// 5 em 5 minutos, para os novos entrarem sem recarregar a página; quem tem mais de 24 h é filtrado no browser.
export async function GET() {
  const lista = (await todosMonumentos()).map((m) => paraAdicionado(resumo(m)));
  lista.sort((a, b) => b.adicionado.localeCompare(a.adicionado) || a.id.localeCompare(b.id));
  return new Response(JSON.stringify(lista), { headers: { "Content-Type": "application/json; charset=utf-8" } });
}
