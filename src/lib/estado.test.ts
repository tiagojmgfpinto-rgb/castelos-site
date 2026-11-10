import { describe, expect, it } from "vitest";
import { entraNoMapa, faseDe, haQuanto, JANELA_RECENTES_MS, proximaMudanca, recentes, variantePublicada } from "./estado";

const H = 3600 * 1000;
const agora = Date.parse("2026-11-10T12:00:00Z");
const ha = (ms: number) => new Date(agora - ms).toISOString();

describe("Teste A: adicionados nas últimas 24 h", () => {
  const lista = [
    { id: "acabado-de-entrar", adicionado: ha(5 * 60 * 1000) },
    { id: "ha-duas-horas", adicionado: ha(2 * H) },
    { id: "quase-a-sair", adicionado: ha(24 * H - 1000) },
    { id: "expirado", adicionado: ha(24 * H) },
    { id: "antigo", adicionado: "2026-10-26T14:52:00Z" },
    { id: "futuro", adicionado: new Date(agora + H).toISOString() },
  ];

  it("mostra só os de menos de 24 h, o mais recente primeiro", () => {
    expect(recentes(lista, agora).map((x) => x.id)).toEqual(["acabado-de-entrar", "ha-duas-horas", "quase-a-sair"]);
  });

  it("cada um expira à sua hora, sem tirar nada da lista original", () => {
    const depois = agora + 2000; // "quase-a-sair" passa as 24 h
    expect(recentes(lista, depois).map((x) => x.id)).toEqual(["acabado-de-entrar", "ha-duas-horas"]);
    expect(lista).toHaveLength(6);
    // 24 h depois de entrar, o último também sai
    expect(recentes(lista, Date.parse(lista[0].adicionado) + JANELA_RECENTES_MS).map((x) => x.id)).toEqual(["futuro"]);
    // e o que tinha data no futuro só aparece a partir dessa hora
    expect(recentes(lista, agora).some((x) => x.id === "futuro")).toBe(false);
  });

  it("um monumento novo entra logo, com ou sem maqueta", () => {
    const novo = { id: "novo-sem-maqueta", adicionado: new Date(agora).toISOString(), noMapa: false };
    expect(recentes([...lista, novo], agora)[0].id).toBe("novo-sem-maqueta");
  });

  it("acorda exatamente quando o próximo expira", () => {
    expect(proximaMudanca(lista, agora)).toBe(Date.parse(lista[2].adicionado) + JANELA_RECENTES_MS);
    expect(proximaMudanca([{ id: "antigo", adicionado: "2026-10-26T14:52:00Z" }], agora)).toBeNull();
  });

  it("escreve o tempo decorrido em português", () => {
    expect(haQuanto(20 * 1000)).toBe("agora mesmo");
    expect(haQuanto(60 * 1000)).toBe("há 1 minuto");
    expect(haQuanto(42 * 60 * 1000)).toBe("há 42 minutos");
    expect(haQuanto(1 * H)).toBe("há 1 hora");
    expect(haQuanto(2 * H + 59 * 60 * 1000)).toBe("há 2 horas");
  });
});

describe("Testes B e C: quem entra no mapa e quem fica em Próximas adições", () => {
  const existe = (f: string) => f === "maquetas/x/x.glb";
  const publicada = { ficheiro: "maquetas/x/x.glb", provisoria: true, publicada: "2026-11-02" };

  it("a variante só conta como publicada com data, aprovação e ficheiro", () => {
    expect(variantePublicada(publicada, existe)).toBe(true);
    expect(variantePublicada({ ...publicada, publicada: undefined }, existe)).toBe(false);
    expect(variantePublicada({ ...publicada, provisoria: false }, existe)).toBe(false);
    expect(variantePublicada({ ...publicada, validada: true, provisoria: false }, existe)).toBe(true);
    expect(variantePublicada({ ...publicada, ficheiro: "maquetas/falta/falta.glb" }, existe)).toBe(false);
  });

  it("segue a tabela do pedido", () => {
    // sem maqueta
    expect(faseDe("pendente", 0)).toBe("por-fazer");
    expect(entraNoMapa("pendente", 0)).toBe(false);
    // em desenvolvimento, mesmo que já haja um ficheiro publicado de outra versão
    expect(faseDe("em-curso", 0)).toBe("em-curso");
    expect(entraNoMapa("em-curso", 1)).toBe(false);
    // concluída mas por publicar
    expect(faseDe("concluida", 0)).toBe("por-publicar");
    expect(entraNoMapa("concluida", 0)).toBe(false);
    // concluída e publicada
    expect(faseDe("concluida", 1)).toBe("publicada");
    expect(entraNoMapa("concluida", 1)).toBe(true);
  });

  it("um ficheiro publicado sem o estado concluida não chega", () => {
    expect(entraNoMapa("pendente", 1)).toBe(false);
  });
});
