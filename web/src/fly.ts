// The NeuroMechFly body, loaded from the GLB that tools/build_fly_glb.py makes.
//
// Two things worth knowing before reading the code.
//
// **The GLB is not in this repository.** It is converted from the fetched
// simulator checkout, which ships no LICENSE, so it is generated locally and
// gitignored for the same reason the response cache is. `load` returns null when
// it is absent and the scene falls back to its placeholder capsule, which is why
// this returns a nullable rather than throwing.
//
// **The legs are at the model's neutral pose.** A fly sitting on a dock holding
// a rod is not walking, so no gait is driven here. Upstream's CPG could drive
// these joints and the rig carries everything needed to do it (`applyCtrl`
// below is that path, and it is exercised by the neutral pose on every load),
// but inventing a walk cycle for a fly that is standing still would be drawing
// behaviour rather than showing it. The idle sway in scene.ts is scripted and
// labelled as such.

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

export interface RigJoint {
  name: string;
  /** Index into the glTF node array: the node this hinge rotates. */
  node: number;
  /** Hinges on one node compose in this order, after the node's own rotation. */
  order: number;
  axis: [number, number, number];
  /** Control index that drives it, or null when the joint is passive. */
  ctrlIndex: number | null;
  qposadr: number;
  neutral: number;
}

interface Rig {
  schemaVersion: number;
  source: string;
  joints: RigJoint[];
}

/** One node, its rotation as the GLB declares it, and the hinges that turn it. */
interface Hinged {
  object: THREE.Object3D;
  base: THREE.Quaternion;
  joints: RigJoint[];
}

export class FlyModel {
  private readonly hinged: Hinged[] = [];
  private readonly scratch = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3();

  private constructor(
    readonly object: THREE.Object3D,
    readonly rig: Rig,
    nodes: THREE.Object3D[],
  ) {
    const byNode = new Map<number, RigJoint[]>();
    for (const joint of rig.joints) {
      const list = byNode.get(joint.node);
      if (list) list.push(joint);
      else byNode.set(joint.node, [joint]);
    }
    for (const [index, joints] of byNode) {
      const target = nodes[index];
      if (!target) continue;
      joints.sort((a, b) => a.order - b.order);
      this.hinged.push({ object: target, base: target.quaternion.clone(), joints });
    }
    this.applyCtrl(null);
  }

  /**
   * Load the body. Returns null when the GLB has not been generated, so the
   * caller can fall back rather than showing a broken scene.
   *
   * `targetLength` rescales the model into the pond's units. The MJCF is in its
   * own units (mesh scale 1000), and the dock and bobber around it are drawn at
   * a size chosen for the shot, not measured from anything, so a fixed scale
   * here would be no more principled than a fitted one.
   */
  static async load(url: string, targetLength = 3.4): Promise<FlyModel | null> {
    let gltf;
    try {
      const response = await fetch(url, { method: "HEAD" });
      if (!response.ok) return null;
      gltf = await new GLTFLoader().loadAsync(url);
    } catch {
      return null;
    }

    const rig = (gltf.scene.userData?.rig ?? gltf.parser.json.extras?.rig) as Rig | undefined;
    if (!rig?.joints) return null;

    const nodes = (await gltf.parser.getDependencies("node")) as THREE.Object3D[];

    const root = new THREE.Group();
    root.add(gltf.scene);
    const model = new FlyModel(root, rig, nodes);

    // Scale and centre after the neutral pose is applied, or the bounding box
    // would be of a model in its unposed rest position.
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const size = new THREE.Vector3();
    box.getSize(size);
    const longest = Math.max(size.x, size.y, size.z) || 1;
    gltf.scene.scale.setScalar(targetLength / longest);

    const posed = new THREE.Box3().setFromObject(gltf.scene);
    const centre = new THREE.Vector3();
    posed.getCenter(centre);
    // Sit the fly on z = 0 of its own group, facing +x, as the placeholder did.
    gltf.scene.position.sub(new THREE.Vector3(centre.x, centre.y, posed.min.z));

    for (const child of gltf.scene.children) child.traverse(markShadows);
    return model;
  }

  /**
   * Pose every hinge. `ctrl` is upstream's control vector; a null or missing
   * entry leaves that joint at the neutral angle the MJCF declares.
   */
  applyCtrl(ctrl: ArrayLike<number> | null): void {
    for (const { object, base, joints } of this.hinged) {
      object.quaternion.copy(base);
      for (const joint of joints) {
        const commanded =
          ctrl && joint.ctrlIndex !== null ? ctrl[joint.ctrlIndex] : undefined;
        const angle = commanded ?? joint.neutral;
        if (angle === 0) continue;
        this.axis.set(joint.axis[0], joint.axis[1], joint.axis[2]).normalize();
        object.quaternion.multiply(this.scratch.setFromAxisAngle(this.axis, angle));
      }
    }
  }
}

function markShadows(object: THREE.Object3D): void {
  if ((object as THREE.Mesh).isMesh) {
    object.castShadow = true;
    object.receiveShadow = true;
  }
}
