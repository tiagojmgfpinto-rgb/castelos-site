import Maqueta from "./Maqueta";
import { dadosMaqueta } from "./dados";
import { BASE } from "../../lib/base";

/** Maqueta na página permanente do monumento (ilha Astro). */
export default function MaquetaPagina({ id, ficheiro, capa, provisoria }: { id: string; ficheiro: string; capa?: string; provisoria: boolean }) {
  const dados = dadosMaqueta(id);
  if (!dados) return null;
  return <Maqueta url={`${BASE}${ficheiro}`} capa={capa} dados={dados} provisoria={provisoria} onVoltarMapa={() => (location.href = `${BASE}#${id}`)} />;
}
