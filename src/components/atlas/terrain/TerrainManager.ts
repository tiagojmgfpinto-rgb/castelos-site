import * as THREE from "three";
import { corDoTerreno } from "./palette";

/** Formato de manifest.json gerado por pipeline/terrain/build_tiles.py */
export interface Manifest {
  id: string;
  crs: string;
  x0: number;
  y0: number;
  size: number;
  tileSize: number;
  maxLevel: number;
  heightRange: [number, number];
  levels: Record<string, [number, number][]>;
}

interface TileData {
  heights: Float32Array; // (tileSize+1)^2, metros
  cls: Uint8Array;
}

interface Leaf {
  key: string; // nó z/x/y
  src: string; // mosaico de onde vêm os dados (o próprio ou um antecessor)
}

const N = 257;
const SEGMENTS = 128;
const SPLIT = 2.1; // dividir um nó quando a câmara está a menos de SPLIT x o seu lado
const MAX_CACHE_GEOM = 220;
const FUNDO_MAR = 0.12; // relevo submarino suavizado no sombreado
const REFORCO_SOMBRA = 1.6; // o sombreado acentua o relevo mais do que a geometria, como num mapa de relevo

const k = (z: number, x: number, y: number) => `${z}/${x}/${y}`;
const parse = (key: string) => key.split("/").map(Number) as [number, number, number];

export async function decodeTile(url: string): Promise<TileData> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`mosaico ${url}: ${res.status}`);
  const blob = await res.blob();
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas: OffscreenCanvas | HTMLCanvasElement =
    typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(bmp.width, bmp.height) : Object.assign(document.createElement("canvas"), { width: bmp.width, height: bmp.height });
  const ctx = canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const n = bmp.width * bmp.height;
  const heights = new Float32Array(n);
  const cls = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    heights[i] = (px[i * 4] * 256 + px[i * 4 + 1]) / 5 - 8000;
    cls[i] = px[i * 4 + 2];
  }
  bmp.close();
  return { heights, cls };
}

/**
 * Terreno em quadtree com níveis de detalhe.
 * Unidades da cena: 1 = 1 km; x para este, z para sul, y para cima (com exagero vertical).
 */
export class TerrainManager {
  readonly group = new THREE.Group();
  private exists = new Set<string>();
  private hasChildren = new Set<string>();
  private tiles = new Map<string, TileData>();
  private loading = new Map<string, Promise<void>>();
  private geoms = new Map<string, THREE.Mesh>();
  private shown = new Set<string>();
  private material: THREE.MeshStandardMaterial;
  private frustum = new THREE.Frustum();
  private projScreen = new THREE.Matrix4();
  private sizeKm: number;
  private maxH: number;
  onProgress?: (loaded: number, pending: number) => void;
  onChange?: () => void;

  constructor(
    private manifest: Manifest,
    private baseUrl: string,
    readonly exagero: number,
  ) {
    this.sizeKm = manifest.size / 1000;
    this.maxH = (manifest.heightRange[1] / 1000) * exagero;
    for (const [z, list] of Object.entries(manifest.levels)) {
      for (const [x, y] of list) {
        this.exists.add(k(+z, x, y));
        if (+z > 0) this.hasChildren.add(k(+z - 1, x >> 1, y >> 1));
      }
    }
    this.material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.93,
      metalness: 0,
      side: THREE.FrontSide,
    });
  }

  async init() {
    await this.load("0/0/0");
  }

  /** Limites do nó na cena (km). */
  private bounds(z: number, x: number, y: number) {
    const s = this.sizeKm / 2 ** z;
    const half = this.sizeKm / 2;
    return { minX: -half + x * s, minZ: -half + y * s, s };
  }

  private load(key: string): Promise<void> {
    if (this.tiles.has(key)) return Promise.resolve();
    let p = this.loading.get(key);
    if (!p) {
      const [z, x, y] = parse(key);
      p = decodeTile(`${this.baseUrl}${z}/${x}_${y}.png`)
        .then((t) => {
          this.tiles.set(key, t);
        })
        .catch((e) => console.warn(e))
        .finally(() => {
          this.loading.delete(key);
          this.onProgress?.(this.tiles.size, this.loading.size);
          this.onChange?.();
        });
      this.loading.set(key, p);
    }
    return p;
  }

  /** Antecessor mais próximo (ou o próprio) que exista no manifest. */
  private sourceFor(z: number, x: number, y: number): string {
    while (z > 0 && !this.exists.has(k(z, x, y))) {
      z--;
      x >>= 1;
      y >>= 1;
    }
    return k(z, x, y);
  }

  /** Antecessor mais próximo já carregado; pede o carregamento do desejado. */
  private loadedSource(key: string): string {
    if (this.tiles.has(key)) return key;
    this.load(key);
    let [z, x, y] = parse(key);
    while (z > 0) {
      z--;
      x >>= 1;
      y >>= 1;
      const kk = k(z, x, y);
      if (this.tiles.has(kk)) return kk;
    }
    return "0/0/0";
  }

  update(camera: THREE.Camera) {
    this.projScreen.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projScreen);
    const cam = camera.position;
    const leaves: Leaf[] = [];
    const box = new THREE.Box3();

    const visit = (z: number, x: number, y: number) => {
      const { minX, minZ, s } = this.bounds(z, x, y);
      box.min.set(minX, -0.5, minZ);
      box.max.set(minX + s, this.maxH, minZ + s);
      const key = k(z, x, y);
      const inView = this.frustum.intersectsBox(box);
      const d = box.distanceToPoint(cam);
      if (inView && z < this.manifest.maxLevel && this.hasChildren.has(key) && d < s * SPLIT) {
        for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) visit(z + 1, x * 2 + i, y * 2 + j);
        return;
      }
      if (!inView) return; // fora do campo de visão: não desenhar
      leaves.push({ key, src: this.loadedSource(this.sourceFor(z, x, y)) });
    };
    visit(0, 0, 0);

    const want = new Set<string>();
    for (const leaf of leaves) {
      const gk = `${leaf.key}|${leaf.src}`;
      want.add(gk);
      let mesh = this.geoms.get(gk);
      if (!mesh) {
        mesh = this.buildMesh(leaf);
        this.geoms.set(gk, mesh);
      }
      if (!this.shown.has(gk)) {
        this.group.add(mesh);
        this.shown.add(gk);
      }
    }
    for (const gk of [...this.shown]) {
      if (!want.has(gk)) {
        const mesh = this.geoms.get(gk);
        if (mesh) this.group.remove(mesh);
        this.shown.delete(gk);
      }
    }
    // limpar geometrias antigas que não estão à vista
    if (this.geoms.size > MAX_CACHE_GEOM) {
      for (const [gk, mesh] of this.geoms) {
        if (this.shown.has(gk)) continue;
        mesh.geometry.dispose();
        this.geoms.delete(gk);
        if (this.geoms.size <= MAX_CACHE_GEOM * 0.8) break;
      }
    }
  }

  get pendentes() {
    return this.loading.size;
  }

  /** Altitude real (m) num ponto da cena, lida do mosaico carregado mais fino. */
  altitude(xKm: number, zKm: number): number | null {
    const half = this.sizeKm / 2;
    for (let z = this.manifest.maxLevel; z >= 0; z--) {
      const s = this.sizeKm / 2 ** z;
      const tx = Math.floor((xKm + half) / s);
      const ty = Math.floor((zKm + half) / s);
      const t = this.tiles.get(k(z, tx, ty));
      if (!t) continue;
      const u = ((xKm + half - tx * s) / s) * (N - 1);
      const v = ((zKm + half - ty * s) / s) * (N - 1);
      return bilinear(t.heights, u, v);
    }
    return null;
  }

  private buildMesh(leaf: Leaf): THREE.Mesh {
    const [z, x, y] = parse(leaf.key);
    const [sz, sx, sy] = parse(leaf.src);
    const t = this.tiles.get(leaf.src)!;
    // sub-retângulo do mosaico de origem que corresponde a este nó, em amostras
    const span = (N - 1) / 2 ** (z - sz);
    const ox = (x - sx * 2 ** (z - sz)) * span;
    const oy = (y - sy * 2 ** (z - sz)) * span;
    const seg = Math.max(1, Math.min(SEGMENTS, Math.round(span)));
    const { minX, minZ, s } = this.bounds(z, x, y);
    const ex = this.exagero / 1000;
    const sampleKm = (s / span); // km por amostra de origem

    const nv = (seg + 1) * (seg + 1);
    const skirt = 4 * (seg + 1);
    const pos = new Float32Array((nv + skirt) * 3);
    const nor = new Float32Array((nv + skirt) * 3);
    const col = new Float32Array((nv + skirt) * 3);
    const c = new THREE.Color();
    const nrm = new THREE.Vector3();

    for (let j = 0; j <= seg; j++) {
      for (let i = 0; i <= seg; i++) {
        const u = ox + (i / seg) * span;
        const v = oy + (j / seg) * span;
        const hm = bilinear(t.heights, u, v);
        const cl = nearest(t.cls, u, v);
        const idx = j * (seg + 1) + i;
        // o mar é uma superfície plana logo abaixo da costa; a batimetria só entra no sombreado e na cor
        const h = cl === 0 ? -40 / this.exagero : hm;
        pos[idx * 3] = minX + (i / seg) * s;
        pos[idx * 3 + 1] = h * ex;
        pos[idx * 3 + 2] = minZ + (j / seg) * s;
        // normal por diferenças finitas na grelha de origem
        const step = Math.max(1, span / seg);
        // nas bordas do mosaico a diferença é unilateral, com o passo real (evita costuras de luz)
        const ul = Math.max(u - step, 0), ur = Math.min(u + step, N - 1);
        const vu = Math.max(v - step, 0), vd = Math.min(v + step, N - 1);
        const fx = cl === 0 ? ex * FUNDO_MAR : ex * REFORCO_SOMBRA;
        const gx = ((bilinear(t.heights, ur, v) - bilinear(t.heights, ul, v)) * fx) / ((ur - ul) * sampleKm);
        const gz = ((bilinear(t.heights, u, vd) - bilinear(t.heights, u, vu)) * fx) / ((vd - vu) * sampleKm);
        nrm.set(-gx, 1, -gz).normalize();
        nor.set([nrm.x, nrm.y, nrm.z], idx * 3);
        corDoTerreno(c, hm, cl, nrm.y);
        col.set([c.r, c.g, c.b], idx * 3);
      }
    }

    // saias nas bordas, para esconder fendas entre níveis de detalhe diferentes
    const drop = 0.1 + s * 0.06;
    const edge: number[] = [];
    for (let i = 0; i <= seg; i++) edge.push(i); // norte
    for (let i = 0; i <= seg; i++) edge.push(seg * (seg + 1) + i); // sul
    for (let j = 0; j <= seg; j++) edge.push(j * (seg + 1)); // oeste
    for (let j = 0; j <= seg; j++) edge.push(j * (seg + 1) + seg); // este
    edge.forEach((src, e) => {
      const d = nv + e;
      pos[d * 3] = pos[src * 3];
      pos[d * 3 + 1] = pos[src * 3 + 1] - drop;
      pos[d * 3 + 2] = pos[src * 3 + 2];
      nor.copyWithin(d * 3, src * 3, src * 3 + 3);
      col.copyWithin(d * 3, src * 3, src * 3 + 3);
    });

    const index: number[] = [];
    const vi = (i: number, j: number) => j * (seg + 1) + i;
    for (let j = 0; j < seg; j++) {
      for (let i = 0; i < seg; i++) {
        const a = vi(i, j), b = vi(i + 1, j), cc = vi(i, j + 1), d = vi(i + 1, j + 1);
        index.push(a, cc, b, b, cc, d);
      }
    }
    const strip = (base: number, along: (i: number) => number) => {
      for (let i = 0; i < seg; i++) {
        const a = along(i), b = along(i + 1), sa = nv + base + i, sb = nv + base + i + 1;
        // as duas faces, com a mesma normal: com DoubleSide, a face de trás teria a normal invertida
        // e apareceria como uma linha escura nas fendas entre níveis de detalhe
        index.push(a, b, sa, b, sb, sa, a, sa, b, b, sa, sb);
      }
    };
    strip(0, (i) => vi(i, 0));
    strip(seg + 1, (i) => vi(i, seg));
    strip(2 * (seg + 1), (j) => vi(0, j));
    strip(3 * (seg + 1), (j) => vi(seg, j));

    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setIndex(nv + skirt > 65535 ? new THREE.Uint32BufferAttribute(index, 1) : new THREE.Uint16BufferAttribute(index, 1));
    g.computeBoundingSphere();
    const mesh = new THREE.Mesh(g, this.material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.name = leaf.key;
    return mesh;
  }

  dispose() {
    for (const m of this.geoms.values()) m.geometry.dispose();
    this.geoms.clear();
    this.material.dispose();
  }
}

function bilinear(h: Float32Array, u: number, v: number) {
  u = Math.min(Math.max(u, 0), N - 1);
  v = Math.min(Math.max(v, 0), N - 1);
  const i = Math.min(Math.floor(u), N - 2), j = Math.min(Math.floor(v), N - 2);
  const fu = u - i, fv = v - j;
  const a = h[j * N + i], b = h[j * N + i + 1], c = h[(j + 1) * N + i], d = h[(j + 1) * N + i + 1];
  return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv;
}

function nearest(c: Uint8Array, u: number, v: number) {
  const i = Math.min(Math.max(Math.round(u), 0), N - 1);
  const j = Math.min(Math.max(Math.round(v), 0), N - 1);
  return c[j * N + i];
}
