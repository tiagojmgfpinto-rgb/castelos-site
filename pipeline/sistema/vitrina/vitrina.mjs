// Vitrina de testes: desenha vistas de uma maqueta com a mesma luz do visor do site (vitrina.html).
//
// Uso:
//   node pipeline/sistema/vitrina/vitrina.mjs --maqueta penela --saida /tmp/v [opções]
// Opções:
//   --glb CAMINHO        GLB a desenhar (por omissão public/maquetas/<m>/<m>.glb). Lê Draco.
//   --vistas FICHEIRO    condicoes.json (usa .vistas) ou lista JSON de vistas; "geral" = vistas gerais automáticas.
//                        Vista: { id, pos:[x,y,z], alvo:[x,y,z], fov, w, h, ocultar:[...] }  (x este, y altitude, z sul)
//   --mascaras           grava também <id>_mascara.png (só a obra, branco sobre preto) para o contorno.
//   --bruto CAMINHO      grava o GLB sem Draco (para tecnico.py).
//   --contagens CAMINHO  grava triângulos e materiais por grupo.
//   --w 1200 --h 800     tamanho das vistas sem tamanho próprio.
// Sem servidor nem portas: o Playwright serve os ficheiros por intercepção, por isso podem correr várias vitrinas ao mesmo tempo.
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const AQUI = path.dirname(new URL(import.meta.url).pathname);
const REPO = path.resolve(AQUI, '../../..');
const require = createRequire(`${REPO}/package.json`);
const { chromium } = require('playwright');

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
const flag = (n) => args.includes(`--${n}`);
const m = opt('maqueta');
if (!m) { console.error('falta --maqueta'); process.exit(2); }
const glb = path.resolve(opt('glb', `${REPO}/public/maquetas/${m}/${m}.glb`));
const saida = path.resolve(opt('saida', '.'));
const W0 = +opt('w', 1200), H0 = +opt('h', 800);
const dadosPath = `${REPO}/src/data/maquetas/${m}.json`;
const dados = fs.existsSync(dadosPath) ? JSON.parse(fs.readFileSync(dadosPath)) : {};
fs.mkdirSync(saida, { recursive: true });

let vistas = [];
const fv = opt('vistas', 'geral');
if (fv !== 'geral') {
  const j = JSON.parse(fs.readFileSync(fv));
  vistas = (Array.isArray(j) ? j : j.vistas || []).filter((v) => v.camera || v.pos).map((v) => ({
    id: v.id, pos: v.pos || v.camera.pos, alvo: v.alvo || v.camera.alvo, fov: v.fov || v.camera?.fov || 32,
    w: v.w || v.camera?.w || W0, h: v.h || v.camera?.h || H0, ocultar: v.ocultar || [],
  }));
}

const navegador = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });

async function abrir(w, h) {
  const p = await navegador.newPage({ viewport: { width: w, height: h } });
  await p.route('http://vitrina/**', async (rt) => {
    const u = decodeURIComponent(new URL(rt.request().url()).pathname);
    let f;
    if (u.startsWith('/three/')) f = path.join(REPO, 'node_modules/three', u.slice(7));
    else if (u.startsWith('/draco/')) f = path.join(REPO, 'public/draco', u.slice(7));
    else if (u.startsWith('/glb/')) f = glb;
    else if (u === '/luz.json') f = path.join(REPO, 'src/data/luz.json');
    else f = path.join(AQUI, u);
    const ct = f.endsWith('.js') ? 'text/javascript' : f.endsWith('.html') ? 'text/html' : f.endsWith('.json') ? 'application/json' : f.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream';
    try { await rt.fulfill({ body: fs.readFileSync(f), contentType: ct }); } catch { await rt.fulfill({ status: 404 }); }
  });
  p.on('pageerror', (e) => console.error('ERRO na página:', e.message));
  const sol = dados.sol ? dados.sol.join(',') : '';
  await p.goto(`http://vitrina/vitrina.html?m=${m}&w=${w}&h=${h}${sol ? `&sol=${sol}` : ''}`);
  await p.waitForFunction(() => window.pronto || window.erro, null, { timeout: 600000 });
  const erro = await p.evaluate(() => window.erro);
  if (erro) throw new Error(erro);
  return p;
}

/** Vistas gerais: aéreas de SO e NE, quatro lados, topo, e quatro ao nível do chão. Iguais em todas as voltas. */
async function gerais(p) {
  const [mn, mx] = await p.evaluate(() => window.caixaObra);
  const c = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];
  const R = Math.max(mx[0] - mn[0], mx[2] - mn[2]) / 2, D = Math.max(60, R * 2.6);
  const out = [
    ['G_aerea_SO', [c[0] - D * 0.62, c[1] + D * 0.6, c[2] + D * 0.5]], ['G_aerea_NE', [c[0] + D * 0.62, c[1] + D * 0.6, c[2] - D * 0.5]],
    ['G_norte', [c[0], c[1] + D * 0.45, c[2] - D * 0.9]], ['G_este', [c[0] + D * 0.9, c[1] + D * 0.45, c[2]]],
    ['G_sul', [c[0], c[1] + D * 0.45, c[2] + D * 0.9]], ['G_oeste', [c[0] - D * 0.9, c[1] + D * 0.45, c[2]]],
    ['G_topo', [c[0], c[1] + D * 1.3, c[2] + 0.01]],
  ].map(([id, pos]) => ({ id, pos, alvo: c, fov: 32, w: W0, h: H0 }));
  for (const [id, az] of [['G_chao_N', 0], ['G_chao_E', 90], ['G_chao_S', 180], ['G_chao_O', 270]]) {
    const a = (az * Math.PI) / 180, d = R * 1.25 + 8;
    const x = c[0] + Math.sin(a) * d, z = c[2] - Math.cos(a) * d;
    const h = await p.evaluate(([x, z]) => window.chao(x, z), [x, z]);
    if (h != null) out.push({ id, pos: [x, h + 1.7, z], alvo: c, fov: 55, w: W0, h: H0 });
  }
  fs.writeFileSync(path.join(saida, 'gerais.json'), JSON.stringify({ centro: c, R, caixaObra: [mn, mx], vistas: out }, null, 1));
  return out;
}

const porTamanho = new Map();
let p0 = await abrir(W0, H0);
if (fv === 'geral') vistas = await gerais(p0);
for (const v of vistas) {
  const k = `${v.w}x${v.h}`;
  if (!porTamanho.has(k)) porTamanho.set(k, []);
  porTamanho.get(k).push(v);
}
if (flag('bruto')) {
  const b64 = await p0.evaluate(() => window.exportarBruto());
  fs.writeFileSync(path.resolve(opt('bruto')), Buffer.from(b64, 'base64'));
}
if (flag('contagens')) fs.writeFileSync(path.resolve(opt('contagens')), JSON.stringify(await p0.evaluate(() => window.contagens()), null, 1));

for (const [k, lista] of porTamanho) {
  const [w, h] = k.split('x').map(Number);
  const p = w === W0 && h === H0 ? p0 : await abrir(w, h);
  for (const v of lista) {
    await p.evaluate((v) => window.render(v), v);
    await p.screenshot({ path: path.join(saida, `${v.id}.png`) });
    if (flag('mascaras')) {
      await p.evaluate((v) => window.render({ ...v, modo: 'mascara' }), v);
      await p.screenshot({ path: path.join(saida, `${v.id}_mascara.png`) });
    }
    console.log('vista', v.id);
  }
  if (p !== p0) await p.close();
}
await navegador.close();
