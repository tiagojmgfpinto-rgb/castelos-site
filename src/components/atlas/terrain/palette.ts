import * as THREE from "three";

// Paleta de maqueta cartográfica: tintas hipsométricas claras (verde nas planícies, ocre nas
// zonas médias, castanho nas serras, pedra nos cumes), mar azul com profundidade, e Espanha
// em cinzento-areia apagado para destacar o território português.
const rampa = (pares: [number, string][]) => pares.map(([h, c]) => [h, new THREE.Color(c)] as const);

const TERRA = rampa([
  [0, "#a3b57e"],
  [150, "#b9c389"],
  [350, "#d6cc93"],
  [600, "#dcc286"],
  [900, "#c9a46c"],
  [1300, "#a87e55"],
  [1800, "#8d6b4f"],
  [2300, "#d4cabb"],
]);
const VIZINHO = rampa([
  [0, "#dad6cc"],
  [600, "#cfcabd"],
  [1500, "#bdb6a8"],
  [2500, "#e4e0d8"],
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
