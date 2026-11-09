import proj4 from "proj4";

// Um só mapa para o Continente, os Açores e a Madeira, como o castillos.site com as Canárias: cónica conforme
// de Lambert com o centro em Portugal Continental (meridiano e origem do PT-TM06), paralelos 34° e 41°.
// Igual a PT_LCC em pipeline/terrain/build_tiles.py.
export const MAPA = "portugal";
proj4.defs("PT-LCC", "+proj=lcc +lat_0=39.6682583333333 +lon_0=-8.13310833333333 +lat_1=34 +lat_2=41 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs");
const QUADRADO = { crs: "PT-LCC", rotuloProjecao: "Cónica conforme de Lambert · ETRS89", x0: -2300000, y0: 700000, size: 3200000 };

export type TerritorioId = "continente" | "acores" | "madeira";

export interface Territorio {
  id: TerritorioId;
  nome: string;
  crs: string;
  rotuloProjecao: string;
  // canto superior esquerdo e lado do quadrado dos mosaicos (m), igual ao pipeline
  x0: number;
  y0: number;
  size: number;
  // vista inicial: alvo (m, na projeção), distância (km), inclinação e rumo (graus)
  vista: { x: number; y: number; distancia: number; inclinacao: number; rumo: number; aspectoMin?: number };
}

export const TERRITORIOS: Record<TerritorioId, Territorio> = {
  // as três regiões partilham o mesmo quadrado de mosaicos (3072 km, igual ao pipeline); muda só a vista
  continente: {
    id: "continente", nome: "Portugal Continental", ...QUADRADO,
    vista: { x: 25000, y: -30000, distancia: 1000, inclinacao: 36, rumo: 0, aspectoMin: 0.62 },
  },
  acores: {
    id: "acores", nome: "Açores", ...QUADRADO,
    vista: { x: -1700000, y: 0, distancia: 880, inclinacao: 36, rumo: 0, aspectoMin: 1.1 },
  },
  madeira: {
    id: "madeira", nome: "Madeira", ...QUADRADO,
    vista: { x: -806600, y: -718800, distancia: 170, inclinacao: 36, rumo: 0, aspectoMin: 0.8 },
  },
};

/** Centro do quadrado de mosaicos, que é a origem da cena 3D. */
export function centro(t: Territorio) {
  return { cx: t.x0 + t.size / 2, cy: t.y0 - t.size / 2 };
}

/** Coordenadas geográficas -> coordenadas planas da projeção do território (m). */
export function projectar(t: Territorio, lon: number, lat: number): [number, number] {
  return proj4("EPSG:4326", t.crs, [lon, lat]) as [number, number];
}

/** Coordenadas planas (m) -> cena (km): x para este, z para sul. */
export function paraCena(t: Territorio, x: number, y: number): [number, number] {
  const { cx, cy } = centro(t);
  return [(x - cx) / 1000, -(y - cy) / 1000];
}

/** Cena (km) -> coordenadas geográficas [lon, lat]. */
export function deCena(t: Territorio, sx: number, sz: number): [number, number] {
  const { cx, cy } = centro(t);
  return proj4(t.crs, "EPSG:4326", [sx * 1000 + cx, -sz * 1000 + cy]) as [number, number];
}
