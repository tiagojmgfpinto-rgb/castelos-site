import * as THREE from "three";

/**
 * Grão do chão (25 nov. 2026, modelo do castillos.site): por cima da cor de cada classe do chão (erva, mato,
 * terra, saibro...), uma textura cinzenta de pormenor (média 0,5) com tufos de erva, torrões e pedrinhas,
 * projetada dos três eixos em coordenadas do mundo, a duas escalas e rodada entre elas, para não esticar nas
 * encostas nem se notar a repetição. Vai-se apagando com a distância: de longe fica só a cor da classe.
 * Só nas maquetas com "detalheChao" no JSON e só nos materiais "chao_*" de chão natural.
 */

const SEM_GRAO = /^chao_(calcada|lajeado|escadas|agua)/;
export const COM_GRAO = (nome: string) => /^chao_/.test(nome) && !SEM_GRAO.test(nome);

export interface OpcoesGrao {
  /** Metros cobertos pela textura nas duas escalas. */
  escalas?: [number, number];
  /** Força do grão (0 = nada). */
  forca?: number;
  /** Distância (m) a que o grão começa a apagar-se e a que desaparece. */
  fade?: [number, number];
}

let texGrao: THREE.DataTexture | null = null;

/** Hash inteiro → [0, 1) (determinístico, sem Math.random, para a textura ser sempre a mesma). */
function hash(i: number) {
  let h = (i | 0) ^ 0x9e3779b9;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** Ruído de valor periódico (período n px), suavizado com a curva de Hermite. */
function ruido(n: number, celulas: number, semente: number, saida: Float32Array, amp: number) {
  const c = celulas;
  for (let y = 0; y < n; y++) {
    const fy = (y / n) * c, y0 = Math.floor(fy), ty = fy - y0, sy = ty * ty * (3 - 2 * ty);
    for (let x = 0; x < n; x++) {
      const fx = (x / n) * c, x0 = Math.floor(fx), tx = fx - x0, sx = tx * tx * (3 - 2 * tx);
      const v = (i: number, j: number) => hash(semente * 7919 + ((j % c) * c + (i % c))) * 2 - 1;
      const a = v(x0, y0) + (v(x0 + 1, y0) - v(x0, y0)) * sx;
      const b = v(x0, y0 + 1) + (v(x0 + 1, y0 + 1) - v(x0, y0 + 1)) * sx;
      saida[y * n + x] += amp * (a + (b - a) * sy);
    }
  }
}

/**
 * Textura de 512 px, dois canais: R = tufos e folhas finas (traços curtos, mais ou menos de pé, claros e escuros),
 * G = grão de terra (manchas largas, torrões e pedrinhas). Ambos com média 0,5 e período 512 px.
 */
export function texturaGrao(): THREE.DataTexture {
  if (texGrao) return texGrao;
  const N = 512;
  const R = new Float32Array(N * N), G = new Float32Array(N * N);
  ruido(N, 32, 1, R, 0.07);
  ruido(N, 96, 2, R, 0.04);
  ruido(N, 8, 3, G, 0.10);
  ruido(N, 24, 4, G, 0.08);
  ruido(N, 80, 5, G, 0.05);
  const wrap = (x: number, y: number) => (((y % N) + N) % N) * N + (((x % N) + N) % N);
  let k = 0;
  // folhas de erva: 24 mil traços de 3 a 14 px, quase verticais na textura (que nas encostas fica na vertical)
  for (let t = 0; t < 24000; t++) {
    const x0 = hash(k++) * N, y0 = hash(k++) * N;
    const ang = -Math.PI / 2 + (hash(k++) - 0.5) * 2.0, comp = 3 + hash(k++) * 11;
    const claro = hash(k++) < 0.5, dv = claro ? 0.12 + 0.1 * hash(k++) : -(0.12 + 0.14 * hash(k++));
    for (let s = 0; s <= comp; s++) {
      const q = wrap(Math.floor(x0 + Math.cos(ang) * s), Math.floor(y0 + Math.sin(ang) * s));
      R[q] += dv * 0.6 * (claro ? 0.5 + 0.5 * (s / comp) : 1 - 0.5 * (s / comp));
    }
  }
  // torrões e pedrinhas: discos de 1 a 3 px, claros ou escuros
  for (let t = 0; t < 4000; t++) {
    const cx = Math.floor(hash(k++) * N), cy = Math.floor(hash(k++) * N), r = 0.8 + hash(k++) * 2.2;
    const dv = hash(k++) < 0.5 ? 0.18 : -0.2;
    for (let j = -3; j <= 3; j++) for (let i = -3; i <= 3; i++) {
      const d = Math.hypot(i, j);
      if (d <= r) G[wrap(cx + i, cy + j)] += dv * (1 - d / (r + 0.5));
    }
  }
  const media = (A: Float32Array) => A.reduce((s, v) => s + v, 0) / A.length;
  const mR = media(R), mG = media(G);
  const dados = new Uint8Array(N * N * 4);
  for (let q = 0; q < N * N; q++) {
    dados[4 * q] = Math.max(0, Math.min(255, Math.round((R[q] - mR + 0.5) * 255)));
    dados[4 * q + 1] = Math.max(0, Math.min(255, Math.round((G[q] - mG + 0.5) * 255)));
    dados[4 * q + 2] = 128;
    dados[4 * q + 3] = 255;
  }
  const tex = new THREE.DataTexture(dados, N, N, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.NoColorSpace;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return (texGrao = tex);
}

/** Junta o grão a um material do chão (MeshStandardMaterial do glTF). Idempotente. */
export function aplicarGrao(m: THREE.Material, anisotropia: number, op: OpcoesGrao = {}) {
  const mat = m as THREE.MeshStandardMaterial;
  if (!mat.isMeshStandardMaterial || mat.userData.grao) return;
  mat.userData.grao = true;
  const tex = texturaGrao();
  tex.anisotropy = anisotropia;
  const [e1, e2] = op.escalas ?? [1.6, 5.3];
  const [f0, f1] = op.fade ?? [90, 320];
  const u = {
    uGrao: { value: tex },
    uGraoEsc: { value: new THREE.Vector2(e1, e2) },
    uGraoForca: { value: op.forca ?? 0.32 },
    uGraoFade: { value: new THREE.Vector2(f0, f1) },
  };
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vGraoP;\nvarying vec3 vGraoN;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvGraoP = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvGraoN = normalize(mat3(modelMatrix) * objectNormal);");
    sh.fragmentShader = sh.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vGraoP;
varying vec3 vGraoN;
uniform sampler2D uGrao;
uniform vec2 uGraoEsc;
uniform float uGraoForca;
uniform vec2 uGraoFade;
vec2 graoTri(vec3 p, vec3 w, float e, mat2 rot) {
  return texture2D(uGrao, rot * p.xz / e).rg * w.y + texture2D(uGrao, rot * p.xy / e).rg * w.z + texture2D(uGrao, rot * p.zy / e).rg * w.x;
}`,
      )
      .replace(
        "#include <map_fragment>",
        `#include <map_fragment>
{
  vec3 w = pow(abs(normalize(vGraoN)), vec3(4.0));
  w /= (w.x + w.y + w.z + 1e-5);
  vec2 g1 = graoTri(vGraoP, w, uGraoEsc.x, mat2(1.0));
  vec2 g2 = graoTri(vGraoP, w, uGraoEsc.y, mat2(0.8090, 0.5878, -0.5878, 0.8090));
  float g = 0.55 * g1.r + 0.2 * g1.g + 0.25 * g2.g - 0.5;
  float perto = 1.0 - smoothstep(uGraoFade.x, uGraoFade.y, distance(vGraoP, cameraPosition));
  diffuseColor.rgb *= clamp(1.0 + 2.0 * uGraoForca * g * perto, 0.4, 1.6);
}`,
      );
  };
  mat.customProgramCacheKey = () => "grao-chao";
  mat.needsUpdate = true;
}
