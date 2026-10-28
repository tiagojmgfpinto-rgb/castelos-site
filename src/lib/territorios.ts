import proj4 from "proj4";

// Projeções oficiais (DGT): PT-TM06/ETRS89 no Continente, PTRA08-UTM nas regiões autónomas.
proj4.defs("EPSG:3763", "+proj=tmerc +lat_0=39.6682583333333 +lon_0=-8.13310833333333 +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs");
proj4.defs("EPSG:5015", "+proj=utm +zone=26 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs");
proj4.defs("EPSG:5016", "+proj=utm +zone=28 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs");

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
  vista: { x: number; y: number; distancia: number; inclinacao: number; rumo: number };
}

export const TERRITORIOS: Record<TerritorioId, Territorio> = {
  continente: {
    id: "continente", nome: "Portugal Continental", crs: "EPSG:3763", rotuloProjecao: "Transversa de Mercator · PT-TM06/ETRS89",
    // quadrado de 1228,8 km: quase toda a Península, como uma maqueta sobre a mesa (igual ao pipeline)
    x0: -567200, y0: 607200, size: 1228800,
    vista: { x: 25000, y: -20000, distancia: 1080, inclinacao: 24, rumo: 0 },
  },
  acores: {
    id: "acores", nome: "Açores", crs: "EPSG:5015", rotuloProjecao: "PTRA08 · UTM 26N",
    x0: 100000, y0: 4560000, size: 614400,
    vista: { x: 390000, y: 4250000, distancia: 560, inclinacao: 14, rumo: 0 },
  },
  madeira: {
    id: "madeira", nome: "Madeira", crs: "EPSG:5016", rotuloProjecao: "PTRA08 · UTM 28N",
    x0: 260000, y0: 3700000, size: 153600,
    vista: { x: 335000, y: 3645000, distancia: 150, inclinacao: 16, rumo: 0 },
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
