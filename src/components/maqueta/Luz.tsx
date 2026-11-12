import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import luz from "../../data/luz.json";

/** Luz do visor (src/data/luz.json): a mesma para todas as maquetas e para os renders de revisão. */
export const LUZ = luz;

const TONS: Record<string, THREE.ToneMapping> = {
  AgX: THREE.AgXToneMapping,
  Neutral: THREE.NeutralToneMapping,
  ACES: THREE.ACESFilmicToneMapping,
  Nenhum: THREE.NoToneMapping,
};

/** Ecrãs táteis: sem oclusão ambiental e com mapa de sombras mais pequeno (poupa bateria e memória). */
export const ecraTatil = () => typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;

/** Tone mapping e exposição da luz partilhada. Chama-se no onCreated do Canvas. */
export function aplicarTom(gl: THREE.WebGLRenderer) {
  gl.toneMapping = TONS[luz.tom] ?? THREE.AgXToneMapping;
  gl.toneMappingExposure = luz.exposicao;
}

/**
 * Luz de ambiente: o RoomEnvironment do three.js filtrado (PMREM) como mapa de ambiente da cena. Dá às faces
 * na sombra a luz rebatida que o hemisfério sozinho não dá (cada face recebe luz diferente consoante a orientação).
 */
export function Ambiente() {
  const { gl, scene } = useThree();
  useEffect(() => {
    const pmrem = new THREE.PMREMGenerator(gl);
    const sala = new RoomEnvironment();
    const tex = pmrem.fromScene(sala, luz.ambiente.sigma).texture;
    scene.environment = tex;
    scene.environmentIntensity = luz.ambiente.intensidade;
    sala.dispose();
    pmrem.dispose();
    return () => {
      if (scene.environment === tex) scene.environment = null;
      tex.dispose();
    };
  }, [gl, scene]);
  return null;
}

/**
 * Resolução da oclusão ambiental em relação ao ecrã. A metade (PR #95) poupava, mas as arestas da oclusão
 * cintilavam com o giro; fica a resolução inteira (a densidade de píxeis já está limitada a 1,5).
 */
const ESCALA_AO = 1;
/** Tempo médio por fotograma acima do qual o visor baixa a qualidade (33 ms = menos de 30 fotogramas por segundo). */
const FOTOGRAMA_LENTO = 1 / 30;

/**
 * Desenho da cena, só quando é preciso: a câmara mexeu-se ou `sujo` pede mais fotogramas (o componente que muda
 * a cena põe lá um número). Com a maqueta parada não se desenha nada e o computador descansa.
 *
 * No computador corre num EffectComposer com oclusão ambiental (GTAO), que escurece os cantos, a base das torres,
 * as ameias e os vãos, como a luz indireta do Cycles; `raio` é o raio da maqueta em metros (a oclusão trabalha à
 * escala das ameias, 1–2,5 m). Nos ecrãs táteis desenha-se direto, sem oclusão.
 *
 * Se os fotogramas forem lentos (computadores fracos, ecrãs 4K), a qualidade baixa sozinha, uma vez por cada
 * passo: primeiro a densidade de píxeis para 1, depois a oclusão desliga-se. Não volta a subir (não oscila).
 */
export function Desenho({ raio, tatil, sujo }: { raio: number; tatil: boolean; sujo: React.RefObject<number> }) {
  const { gl, scene, camera, size, setDpr, viewport } = useThree();
  const [oclusao, setOclusao] = useState(!tatil);
  const composer = useMemo(() => {
    if (tatil) return null;
    const alvo = new THREE.WebGLRenderTarget(16, 16, { samples: 4, type: THREE.HalfFloatType });
    const c = new EffectComposer(gl, alvo);
    c.addPass(new RenderPass(scene, camera));
    const ao = new GTAOPass(scene, camera, 16, 16);
    const mudarTamanho = ao.setSize.bind(ao);
    ao.setSize = (w: number, h: number) => mudarTamanho(Math.max(1, Math.round(w * ESCALA_AO)), Math.max(1, Math.round(h * ESCALA_AO)));
    corrigirGtao(ao);
    ao.blendIntensity = luz.oclusao.intensidade;
    ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 5, radiusExponent: 1.4, rings: 2, samples: 12 });
    c.addPass(ao);
    c.addPass(new OutputPass());
    return { c, ao };
  }, [gl, scene, camera, tatil]);

  useEffect(() => {
    if (!composer) return;
    const o = luz.oclusao;
    composer.ao.updateGtaoMaterial({
      radius: THREE.MathUtils.clamp(raio * o.fator, o.raioMin, o.raioMax) || o.raio,
      distanceExponent: 1.3,
      thickness: o.espessura,
      scale: o.escala,
      samples: o.amostras,
      distanceFallOff: 1,
      screenSpaceRadius: false,
    });
  }, [composer, raio]);

  useEffect(() => {
    if (composer) composer.ao.enabled = oclusao;
    sujo.current = Math.max(sujo.current, 2);
  }, [composer, oclusao, sujo]);

  useEffect(() => {
    composer?.c.setPixelRatio(viewport.dpr);
    composer?.c.setSize(size.width, size.height);
    sujo.current = Math.max(sujo.current, 2);
  }, [composer, viewport.dpr, size.width, size.height, sujo]);

  useEffect(() => () => {
    composer?.ao.dispose();
    composer?.c.dispose();
  }, [composer]);

  const ultima = useMemo(() => new THREE.Matrix4(), []);
  const medida = useRef({ antes: 0, soma: 0, n: 0, desde: performance.now() + 2500 });
  useFrame(() => {
    camera.updateMatrixWorld();
    const mexeu = !ultima.equals(camera.matrixWorld);
    const m = medida.current;
    const agora = performance.now();
    if (!mexeu && sujo.current <= 0) {
      m.antes = 0;
      return;
    }
    ultima.copy(camera.matrixWorld);
    sujo.current = Math.max(0, sujo.current - 1);
    if (composer) composer.c.render();
    else gl.render(scene, camera);
    // Medir só fotogramas seguidos (os primeiros segundos não contam: compilação dos shaders e texturas)
    if (m.antes && agora > m.desde) {
      m.soma += (agora - m.antes) / 1000;
      if (++m.n >= 45) {
        if (m.soma / m.n > FOTOGRAMA_LENTO) {
          if (viewport.dpr > 1.01) setDpr(1);
          else if (oclusao) setOclusao(false);
          m.desde = agora + 1500;
        }
        m.soma = 0;
        m.n = 0;
      }
    }
    m.antes = agora;
  }, 1);
  return null;
}

/**
 * Na GTAO do three.js, cosHorizons pode passar de 1 por erro de arredondamento; sqrt e acos dão então NaN e o
 * filtro de ruído espalha-o num quadrado preto. Limitam-se os cossenos e um valor que não é número conta como
 * sem oclusão. Se o shader mudar noutra versão do three.js, fica como está e avisa na consola.
 */
function corrigirGtao(ao: GTAOPass) {
  const m = ao.gtaoMaterial;
  const antes = m.fragmentShader;
  m.fragmentShader = antes
    .replace(
      "vec2 sinHorizons = sqrt(1. - cosHorizons * cosHorizons);",
      "cosHorizons = clamp(cosHorizons, -1., 1.);\n\t\t\t\tvec2 sinHorizons = sqrt(max(vec2(0.), 1. - cosHorizons * cosHorizons));",
    )
    .replace("ao = clamp(ao / float(DIRECTIONS), 0., 1.);", "ao = ao / float(DIRECTIONS);\n\t\t\tao = (ao == ao) ? clamp(ao, 0., 1.) : 1.;");
  if (m.fragmentShader === antes) console.warn("[maqueta] GTAO sem correção de NaN (outra versão do three.js?)");
  m.needsUpdate = true;
}
