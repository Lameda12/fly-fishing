// The pond: one Three.js scene per recording.
//
// Conventions follow the simulator's own viewer (embodied-fly-lab/shared/
// scene.js): Z is up, the ground plane sits at z = 0, and meshes use
// MeshStandardMaterial under a hemisphere plus one directional key light. The
// fly here is a placeholder capsule; the real NeuroMechFly meshes arrive with
// the GLB conversion.
//
// Nothing in this file is simulator output. The dock, the water, the fly's
// position and the splash are all drawn. What comes from the simulation is the
// timing the Replay drives it with, and the rates the HUD prints.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { FlyModel } from "./fly";
import { buildBoat, buildCampfire, updateCampfire, updateOars } from "./props";

const WATER_SIZE = 90;
const BOBBER_DISTANCE = 13;
const BOBBER_DIP_DEPTH = 1.5;
/** Where the line leaves the rod, in the fly anchor's frame. */
const ROD_TIP = new THREE.Vector3(3.0, 0.45, 2.5);
/** The default framing: jetty on the left, fishing ground on the right. */
const ORBIT_POSITION = new THREE.Vector3(-7, -25, 9);
const ORBIT_TARGET = new THREE.Vector3(2, 0, 1.1);
/** Where the boat sits at each end of its run. Scripted staging. */
const MOORED = new THREE.Vector3(-6.2, -5.2, 0);
const FISHING = new THREE.Vector3(5.5, 0.4, 0);
/** Where the fire sits on the jetty, and where the fly stands to use it. */
const FIRE_AT = new THREE.Vector3(-5.6, 1.4, 1.84);
const FIRE_SEAT = new THREE.Vector3(-3.6, 1.2, 1.84);
/** The fly's seat in the boat, in the hull's own frame. */
const SEAT = new THREE.Vector3(0.7, 0, -0.3);

/** A pooled splash droplet. */
interface Droplet {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  life: number;
}

function waterMaterial(): THREE.MeshStandardMaterial {
  const material = new THREE.MeshStandardMaterial({
    color: 0x27616e,
    roughness: 0.3,
    metalness: 0.14,
    transparent: true,
    opacity: 0.94,
  });
  const uniforms = { uTime: { value: 0 } };
  material.userData.uniforms = uniforms;

  // Ripples in the vertex stage: three travelling sines, with the normal taken
  // analytically from the same height field so the lighting follows the waves.
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = uniforms.uTime;
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
        uniform float uTime;
        const vec2 K0 = vec2(0.42, 0.13);
        const vec2 K1 = vec2(-0.21, 0.37);
        const vec2 K2 = vec2(0.67, -0.55);
        float waveHeight(vec2 p) {
          return 0.22 * sin(dot(p, K0) + uTime * 1.35)
               + 0.15 * sin(dot(p, K1) + uTime * 1.9)
               + 0.08 * sin(dot(p, K2) + uTime * 2.7);
        }
        vec3 waveNormal(vec2 p) {
          float dx = 0.22 * K0.x * cos(dot(p, K0) + uTime * 1.35)
                   + 0.15 * K1.x * cos(dot(p, K1) + uTime * 1.9)
                   + 0.08 * K2.x * cos(dot(p, K2) + uTime * 2.7);
          float dy = 0.22 * K0.y * cos(dot(p, K0) + uTime * 1.35)
                   + 0.15 * K1.y * cos(dot(p, K1) + uTime * 1.9)
                   + 0.08 * K2.y * cos(dot(p, K2) + uTime * 2.7);
          return normalize(vec3(-dx, -dy, 1.0));
        }`,
      )
      .replace("#include <beginnormal_vertex>", "vec3 objectNormal = waveNormal(position.xy);")
      .replace(
        "#include <begin_vertex>",
        "vec3 transformed = vec3(position.x, position.y, position.z + waveHeight(position.xy));",
      );
  };
  return material;
}

function buildDock(): THREE.Group {
  const group = new THREE.Group();
  const plankMaterial = new THREE.MeshStandardMaterial({ color: 0x6b4d31, roughness: 0.92 });
  const postMaterial = new THREE.MeshStandardMaterial({ color: 0x4a3524, roughness: 0.95 });

  for (let i = 0; i < 7; i++) {
    const plank = new THREE.Mesh(new THREE.BoxGeometry(1.5, 7, 0.35), plankMaterial);
    plank.position.set(-1.1 - i * 1.65, 0, 1.6);
    plank.castShadow = true;
    plank.receiveShadow = true;
    group.add(plank);
  }
  for (const y of [-2.7, 2.7]) {
    for (const x of [-1.6, -10.4]) {
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 3.6, 10), postMaterial);
      post.rotation.x = Math.PI / 2;
      post.position.set(x, y, 0.1);
      post.castShadow = true;
      group.add(post);
    }
  }
  return group;
}

/**
 * The placeholder fly: a capsule thorax, an abdomen, a head and two eyes.
 *
 * Used when web/public/fly.glb has not been generated. The real NeuroMechFly
 * body is converted from the fetched simulator checkout by
 * tools/build_fly_glb.py and is gitignored, because upstream ships no LICENSE
 * and this repository does not redistribute anything derived from it.
 */
function buildFlyPlaceholder(): THREE.Group {
  const group = new THREE.Group();
  const body = new THREE.MeshStandardMaterial({ color: 0x9a6a20, roughness: 0.62 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x3a2a12, roughness: 0.7 });
  const eye = new THREE.MeshStandardMaterial({ color: 0xab351e, roughness: 0.35 });

  const thorax = new THREE.Mesh(new THREE.CapsuleGeometry(0.34, 0.5, 6, 14), body);
  thorax.rotation.z = Math.PI / 2;
  thorax.castShadow = true;
  group.add(thorax);

  const abdomen = new THREE.Mesh(new THREE.CapsuleGeometry(0.3, 0.55, 6, 14), dark);
  abdomen.rotation.z = Math.PI / 2;
  abdomen.position.set(-0.78, 0, -0.04);
  abdomen.castShadow = true;
  group.add(abdomen);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.28, 16, 12), body);
  head.position.set(0.58, 0, 0.06);
  head.castShadow = true;
  group.add(head);

  for (const side of [1, -1]) {
    const lens = new THREE.Mesh(new THREE.SphereGeometry(0.15, 12, 10), eye);
    lens.position.set(0.66, side * 0.19, 0.11);
    group.add(lens);
  }

  const wingMaterial = new THREE.MeshStandardMaterial({
    color: 0xcfd8e4,
    transparent: true,
    opacity: 0.32,
    roughness: 0.15,
    side: THREE.DoubleSide,
  });
  for (const side of [1, -1]) {
    const wing = new THREE.Mesh(new THREE.PlaneGeometry(1.15, 0.42), wingMaterial);
    wing.position.set(-0.5, side * 0.3, 0.3);
    wing.rotation.set(0, 0, side * 0.22);
    group.add(wing);
  }

  group.scale.setScalar(1.4);
  return group;
}

/** The rod is the viewer's own prop, not part of the body model. */
function buildRod(): THREE.Mesh {
  const rod = new THREE.Mesh(
    new THREE.CylinderGeometry(0.05, 0.085, 3.6, 8),
    new THREE.MeshStandardMaterial({ color: 0xd8c08a, roughness: 0.6 }),
  );
  rod.position.set(1.85, 0.45, 1.35);
  rod.rotation.set(0.12, 0.72, 0);
  rod.castShadow = true;
  return rod;
}

function buildBobber(): THREE.Group {
  const group = new THREE.Group();
  const top = new THREE.Mesh(
    new THREE.SphereGeometry(0.42, 18, 14, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0xe1695e, roughness: 0.45 }),
  );
  const bottom = new THREE.Mesh(
    new THREE.SphereGeometry(0.42, 18, 14, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0xf2f4f0, roughness: 0.5 }),
  );
  const stem = new THREE.Mesh(
    new THREE.CylinderGeometry(0.045, 0.045, 0.55, 8),
    new THREE.MeshStandardMaterial({ color: 0x2b2f31, roughness: 0.8 }),
  );
  stem.rotation.x = Math.PI / 2;
  stem.position.z = 0.48;
  group.add(top, bottom, stem);
  group.position.set(BOBBER_DISTANCE, 0, 0);
  return group;
}

export class PondScene {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly water: THREE.Mesh;
  private readonly waterUniforms: { uTime: { value: number } };
  /** The fly's perch. Rides in the boat, so the body moves with it. */
  private readonly flyAnchor: THREE.Group;
  private readonly boat: THREE.Group;
  private readonly rod: THREE.Mesh;
  private readonly campfire: THREE.Group;
  /** Where the boat is, 0 at the jetty and 1 out on the fishing ground. */
  private boatOut = 0;
  private fireHeat = 0;
  private oarStroke = 0;
  private flyAtFire = false;
  private readonly flyTarget = new THREE.Vector3();
  private placeholder: THREE.Group | null;
  private body: FlyModel | null = null;
  private readonly bobber: THREE.Group;
  private readonly line: THREE.Line;
  private readonly linePoints: THREE.Vector3[];
  private readonly ring: THREE.Mesh;
  private readonly droplets: Droplet[] = [];
  private readonly dropletGeometry = new THREE.SphereGeometry(0.11, 7, 6);
  private readonly dropletMaterial = new THREE.MeshStandardMaterial({
    color: 0xbfe6ea,
    roughness: 0.2,
    emissive: 0x18343a,
  });
  private readonly resizeObserver: ResizeObserver;
  private ringLife = 0;
  private cameraMode: "orbit" | "closeup" = "orbit";

  constructor(private readonly container: HTMLElement) {
    this.scene.background = new THREE.Color(0x12232a);
    this.scene.fog = new THREE.Fog(0x12232a, 42, 118);

    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.camera = new THREE.PerspectiveCamera(44, 1, 0.1, 400);
    this.camera.up.set(0, 0, 1);
    this.camera.position.copy(ORBIT_POSITION);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.copy(ORBIT_TARGET);
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 70;

    this.scene.add(new THREE.HemisphereLight(0xa8ccd8, 0x1a2420, 1.25));
    const key = new THREE.DirectionalLight(0xffeccd, 2.1);
    key.position.set(-14, -18, 22);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 1;
    key.shadow.camera.far = 70;
    for (const edge of ["left", "right", "top", "bottom"] as const) {
      const camera = key.shadow.camera;
      camera[edge] = edge === "left" || edge === "bottom" ? -26 : 26;
    }
    this.scene.add(key);
    // Kept dim and high: a bright, low rim light reflects off the water as a
    // blown-out hotspot at grazing camera angles.
    const rim = new THREE.DirectionalLight(0x8fd4e8, 0.26);
    rim.position.set(24, 16, 20);
    this.scene.add(rim);

    const material = waterMaterial();
    this.waterUniforms = material.userData.uniforms as { uTime: { value: number } };
    this.water = new THREE.Mesh(new THREE.PlaneGeometry(WATER_SIZE, WATER_SIZE, 180, 180), material);
    this.water.receiveShadow = true;
    this.scene.add(this.water);

    const bed = new THREE.Mesh(
      new THREE.PlaneGeometry(WATER_SIZE, WATER_SIZE),
      new THREE.MeshStandardMaterial({ color: 0x14272c, roughness: 1 }),
    );
    bed.position.z = -2.4;
    this.scene.add(bed);

    this.scene.add(buildDock());
    // The fly rides in the boat, so its perch is a child of the hull rather
    // than of the scene: moving the boat moves the fly, the rod and the line.
    this.boat = buildBoat();
    this.scene.add(this.boat);
    this.rod = buildRod();
    // The fly is placed each frame rather than parented, because it rides the
    // boat for three stages and then stands at the fire for two.
    this.flyAnchor = new THREE.Group();
    this.placeholder = buildFlyPlaceholder();
    this.flyAnchor.add(this.placeholder, this.rod);
    this.scene.add(this.flyAnchor);
    this.flyAnchor.position.copy(MOORED).add(SEAT);

    this.campfire = buildCampfire();
    this.campfire.position.copy(FIRE_AT);
    this.scene.add(this.campfire);
    this.bobber = buildBobber();
    this.scene.add(this.bobber);

    // The line is redrawn every frame, so it gets its own buffer up front.
    this.linePoints = Array.from({ length: 24 }, () => new THREE.Vector3());
    const geometry = new THREE.BufferGeometry().setFromPoints(this.linePoints);
    this.line = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: 0xdfe8e4, transparent: true, opacity: 0.68 }),
    );
    this.scene.add(this.line);

    this.ring = new THREE.Mesh(
      new THREE.RingGeometry(0.5, 0.62, 48),
      new THREE.MeshBasicMaterial({
        color: 0x62c48d,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide,
        // The waves displace the surface by up to ~0.45, so the ring sits above
        // the crests and does not write depth, or it reads as a dark arc.
        depthWrite: false,
      }),
    );
    this.ring.position.set(BOBBER_DISTANCE, 0, 0.55);
    this.scene.add(this.ring);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  /**
   * Swap the placeholder for the real NeuroMechFly body, if it has been
   * converted. Returns true when the real body is now in the scene.
   */
  async attachBody(url: string): Promise<boolean> {
    const model = await FlyModel.load(url);
    if (!model) return false;
    this.body = model;
    if (this.placeholder) {
      this.flyAnchor.remove(this.placeholder);
      this.placeholder = null;
    }
    this.flyAnchor.add(model.object);
    return true;
  }

  get usingPlaceholder(): boolean {
    return this.body === null;
  }

  /**
   * Orbit frames the whole scene; close-up sits on the fly.
   *
   * Both keep OrbitControls live, so "follows the fly" here means the orbit
   * target moves onto the fly and the distance limits tighten around it, not
   * that the viewer loses control of the camera.
   */
  setCameraMode(mode: "orbit" | "closeup"): void {
    this.cameraMode = mode;
    if (mode === "closeup") {
      const fly = this.flyAnchor.getWorldPosition(new THREE.Vector3());
      this.controls.target.set(fly.x + 0.4, fly.y, fly.z + 0.4);
      this.controls.minDistance = 2;
      this.controls.maxDistance = 16;
      this.camera.position.set(fly.x + 2.6, fly.y - 5.2, fly.z + 2.4);
    } else {
      this.controls.target.copy(ORBIT_TARGET);
      this.controls.minDistance = 4;
      this.controls.maxDistance = 70;
      this.camera.position.copy(ORBIT_POSITION);
    }
    this.controls.update();
  }

  get cameraModeName(): "orbit" | "closeup" {
    return this.cameraMode;
  }

  /** A live stream of this panel's canvas, for the webm export. */
  captureStream(fps = 60): MediaStream {
    return this.renderer.domElement.captureStream(fps);
  }

  /**
   * Place the voyage: how far out the boat is, and whether the fire is lit.
   *
   * Both are scripted staging. `out` is 0 at the jetty and 1 on the fishing
   * ground; `heat` fades the fire in for the cooking and eating stages.
   */
  setVoyage({
    out,
    heat,
    bobberVisible,
    rowing = 0,
    atFire = false,
  }: {
    out: number;
    heat: number;
    bobberVisible: boolean;
    rowing?: number;
    atFire?: boolean;
  }): void {
    this.boatOut = out;
    this.fireHeat = heat;
    this.oarStroke = rowing;
    this.flyAtFire = atFire;
    this.bobber.visible = bobberVisible;
    this.line.visible = bobberVisible;
  }

  resize(): void {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /** A caught fish: a burst of droplets and an expanding ring at the bobber. */
  splash(): void {
    for (let i = 0; i < 26; i++) {
      const angle = (i / 26) * Math.PI * 2 + Math.random() * 0.3;
      const speed = 2.2 + Math.random() * 3.4;
      const droplet: Droplet = {
        mesh: new THREE.Mesh(this.dropletGeometry, this.dropletMaterial),
        velocity: new THREE.Vector3(
          Math.cos(angle) * speed * 0.55,
          Math.sin(angle) * speed * 0.55,
          3.4 + Math.random() * 3.6,
        ),
        life: 1,
      };
      droplet.mesh.position.copy(this.bobber.position).setZ(0.1);
      this.scene.add(droplet.mesh);
      this.droplets.push(droplet);
    }
    this.ringLife = 1;
  }

  /**
   * Advance the visuals.
   *
   * @param elapsedSeconds wall-clock seconds, for the water and the droplets
   * @param dip            bobber dip, 0 to 1, interpolated from the recording
   * @param idlePhase      the fly's idle bob; a drawn animation, not a gait
   */
  update(elapsedSeconds: number, dt: number, dip: number, idlePhase: number): void {
    this.waterUniforms.uTime.value = elapsedSeconds;

    this.bobber.position.z = -dip * BOBBER_DIP_DEPTH + Math.sin(elapsedSeconds * 2.1) * 0.06;
    this.bobber.rotation.x = Math.sin(elapsedSeconds * 1.7) * 0.12 + dip * 0.5;

    // A drawn idle, not a gait: the body's legs hold the model's neutral pose
    // because the task does not involve walking.
    this.flyAnchor.position.z += Math.sin(idlePhase) * 0.02;
    this.flyAnchor.rotation.y = Math.sin(idlePhase * 0.5) * 0.02;
    this.flyAnchor.rotation.z = this.flyAtFire ? Math.PI * 0.62 : this.boat.rotation.z;

    // The boat rides the swell and rocks a little; both are drawn.
    this.boat.position.lerpVectors(MOORED, FISHING, this.boatOut);
    // Freeboard: the hull is a bowl, so it has to ride high enough that its
    // side shows above the water plane or it reads as a raft.
    this.boat.position.z = 0.55 + Math.sin(elapsedSeconds * 1.9) * 0.08;
    this.boat.rotation.z = Math.atan2(FISHING.y - MOORED.y, FISHING.x - MOORED.x) * this.boatOut;
    this.boat.rotation.y = Math.sin(elapsedSeconds * 1.4) * 0.03;
    this.boat.rotation.x = Math.sin(elapsedSeconds * 2.3 + 1) * 0.035;

    // The fly rides the boat, then steps onto the jetty to cook and eat. Eased
    // rather than snapped, so the change of stage reads as a move.
    if (this.flyAtFire) this.flyTarget.copy(FIRE_SEAT);
    else this.flyTarget.copy(SEAT).applyQuaternion(this.boat.quaternion).add(this.boat.position);
    this.flyAnchor.position.lerp(this.flyTarget, 1 - Math.exp(-dt / 0.35));
    this.rod.visible = !this.flyAtFire;

    updateOars(this.boat, elapsedSeconds, this.oarStroke);
    this.campfire.visible = this.fireHeat > 0.01;
    if (this.campfire.visible) updateCampfire(this.campfire, elapsedSeconds, this.fireHeat);

    // The line hangs from the rod tip to the bobber with a little sag, and the
    // sag tightens as the bobber is pulled under.
    this.flyAnchor.updateMatrixWorld();
    const from = this.flyAnchor.localToWorld(ROD_TIP.clone());
    const to = this.bobber.position.clone().setZ(this.bobber.position.z + 0.35);
    const sag = 0.85 * (1 - dip);
    for (let i = 0; i < this.linePoints.length; i++) {
      const t = i / (this.linePoints.length - 1);
      this.linePoints[i]!.lerpVectors(from, to, t).z -= Math.sin(t * Math.PI) * sag;
    }
    this.line.geometry.setFromPoints(this.linePoints);

    if (this.ringLife > 0) {
      this.ringLife = Math.max(0, this.ringLife - dt * 1.3);
      const grown = 1 + (1 - this.ringLife) * 4.5;
      this.ring.scale.setScalar(grown);
      this.ring.position.x = this.bobber.position.x;
      (this.ring.material as THREE.MeshBasicMaterial).opacity = this.ringLife * 0.75;
    }

    for (let i = this.droplets.length - 1; i >= 0; i--) {
      const droplet = this.droplets[i]!;
      droplet.life -= dt * 1.15;
      droplet.velocity.z -= 14 * dt;
      droplet.mesh.position.addScaledVector(droplet.velocity, dt);
      droplet.mesh.scale.setScalar(Math.max(0.05, droplet.life));
      if (droplet.life <= 0 || droplet.mesh.position.z < -0.6) {
        this.scene.remove(droplet.mesh);
        this.droplets.splice(i, 1);
      }
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    this.dropletGeometry.dispose();
    this.container.removeChild(this.renderer.domElement);
  }
}
