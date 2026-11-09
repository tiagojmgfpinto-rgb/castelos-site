import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
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
 * Oclusão ambiental (GTAO): escurece os cantos, a base das torres, as ameias e os vãos, como a luz indireta do
 * Cycles. Corre num EffectComposer que passa a desenhar a cena (useFrame com prioridade 1 desliga o render
 * automático do R3F). `raio` é o raio da maqueta em metros: a oclusão trabalha à escala das ameias (1–2,5 m).
 */
export function Oclusao({ raio }: { raio: number }) {
  const { gl, scene, camera, size } = useThree();
  const composer = useMemo(() => {
    const alvo = new THREE.WebGLRenderTarget(16, 16, { samples: 4, type: THREE.HalfFloatType });
    const c = new EffectComposer(gl, alvo);
    c.addPass(new RenderPass(scene, camera));
    const ao = new GTAOPass(scene, camera, 16, 16);
    corrigirGtao(ao);
    ao.blendIntensity = luz.oclusao.intensidade;
    ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 5, radiusExponent: 1.4, rings: 2, samples: 12 });
    c.addPass(ao);
    c.addPass(new OutputPass());
    return { c, ao };
  }, [gl, scene, camera]);

  useEffect(() => {
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
    composer.c.setPixelRatio(gl.getPixelRatio());
    composer.c.setSize(size.width, size.height);
  }, [composer, gl, size.width, size.height]);

  useEffect(() => () => {
    composer.ao.dispose();
    composer.c.dispose();
  }, [composer]);

  useFrame(() => composer.c.render(), 1);
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
