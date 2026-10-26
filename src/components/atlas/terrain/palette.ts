import * as THREE from "three";

// Paleta de maqueta cartográfica: tintas hipsométricas claras (verde nas planícies, ocre nas
// zonas médias, castanho nas serras, pedra nos cumes), mar azul com profundidade, e Espanha
// em cinzento-areia apagado para destacar o território português.
const rampa = (pares: [number, string][]) => pares.map(([h, c]) => [h, new THREE.Color(c)] as const);

const TERRA = rampa([
  [0, "#a9b98a"],
  [100, "#b8c290"],
  [250, "#d2cd98"],
  [450, "#ddc98f"],
  [700, "#cfae78"],
  [1000, "#b98d5f"],
  [1400, "#9d7553"],
  [1800, "#a08d7c"],
  [2300, "#d9d2c6"],
]);
const VIZINHO = rampa([
  [0, "#d6d0c2"],
  [600, "#cbc3b3"],
  [1500, "#b9b0a0"],
  [2500, "#e2ddd3"],
]);
const MAR = rampa([
  [-6000, "#2e5d77"],
  [-3000, "#3b6e88"],
  [-1000, "#4f86a0"],
  [-200, "#6fa3b8"],
  [-40, "#8dbac7"],
  [0, "#a6cad1"],
]);
const tmp = new THREE.Color();

function amostrar(out: THREE.Color, r: readonly (readonly [number, THREE.Color])[], h: number) {
  if (h <= r[0][0]) return out.copy(r[0][1]);
  for (let i = 0; i < r.length - 1; i++) {
    const [h0, c0] = r[i];
    const [h1, c1] = r[i + 1];
    if (h <= h1) return out.copy(c0).lerp(c1, (h - h0) / (h1 - h0));
  }
  return out.copy(r[r.length - 1][1]);
}

export function corDoTerreno(out: THREE.Color, h: number, cls: number, ny: number) {
  if (cls === 0) return amostrar(out, MAR, Math.min(h, 0));
  if (cls === 2) return amostrar(out, VIZINHO, h);
  amostrar(out, TERRA, h);
  // encostas íngremes ficam mais terrosas
  const declive = 1 - ny;
  if (declive > 0.08) out.lerp(tmp.set("#8a6a4a"), Math.min(0.35, (declive - 0.08) * 1.4));
  return out;
}
