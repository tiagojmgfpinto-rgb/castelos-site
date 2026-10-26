import * as THREE from "three";

// Paleta de maqueta: areia nas planícies, verde-azeitona nas encostas, castanho nas serras,
// pedra clara nos cumes. Espanha surge apagada para destacar o território português.
const RAMPA: [number, string][] = [
  [0, "#cdbf98"],
  [120, "#c4b98e"],
  [300, "#aaa57a"],
  [550, "#8f9066"],
  [850, "#86795a"],
  [1200, "#7b624a"],
  [1600, "#8a7766"],
  [2000, "#b9ad9c"],
];
const CORES = RAMPA.map(([h, c]) => [h, new THREE.Color(c)] as const);
// albedo escuro: depois de iluminado, o mar coincide com o fundo da página
export const COR_MAR = new THREE.Color("#0d0c0a");
const COR_ESPANHA = new THREE.Color("#47443c");
const tmp = new THREE.Color();

export function corDoTerreno(out: THREE.Color, h: number, cls: number, ny: number) {
  if (cls === 0) {
    out.copy(COR_MAR);
    return out;
  }
  let i = 0;
  while (i < CORES.length - 1 && h > CORES[i + 1][0]) i++;
  if (i === CORES.length - 1) out.copy(CORES[i][1]);
  else {
    const [h0, c0] = CORES[i];
    const [h1, c1] = CORES[i + 1];
    out.copy(c0).lerp(c1, Math.min(1, Math.max(0, (h - h0) / (h1 - h0))));
  }
  // encostas íngremes ficam mais terrosas
  const declive = 1 - ny;
  if (declive > 0.05) out.lerp(tmp.set("#6f5a44"), Math.min(0.45, declive * 1.6));
  if (cls === 2) out.lerp(COR_ESPANHA, 0.82);
  return out;
}
