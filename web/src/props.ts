// The voyage's props: the boat the fly works from, and the fire it cooks on.
//
// All drawn, none of it simulated. The boat's position across the water is set
// by which stage the voyage is in, which is scripted; the fly riding in it is
// the real NeuroMechFly body when web/public/fly.glb has been generated.

import * as THREE from "three";

/** A small rowing boat, pointing along +x, with the fly's seat near the origin. */
export function buildBoat(): THREE.Group {
  const group = new THREE.Group();
  const planks = new THREE.MeshStandardMaterial({
    color: 0x7a5533,
    roughness: 0.88,
    side: THREE.DoubleSide,
  });
  const trim = new THREE.MeshStandardMaterial({ color: 0x4e3720, roughness: 0.9 });
  const inside = new THREE.MeshStandardMaterial({ color: 0x8d6741, roughness: 0.95 });

  // The hull is the bottom half of a sphere stretched into a dinghy: an open
  // bowl with a rounded bow and stern. Two things to get right. A lathe profile
  // was tried first and produced a spiky fan, because a revolved outline has to
  // be monotonic. And three's hemisphere opens along -Y while this scene is
  // Z-up, so without the rotation below it renders as a dome, not a bowl.
  const hull = new THREE.Mesh(
    new THREE.SphereGeometry(1, 28, 16, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2),
    planks,
  );
  hull.rotation.x = Math.PI / 2;
  // Scale is applied before that rotation, so local Y becomes world Z (depth)
  // and local Z becomes world Y (beam).
  hull.scale.set(3.4, 1.45, 1.35);
  hull.castShadow = true;
  hull.receiveShadow = true;
  group.add(hull);

  // The deck the fly stands on, sitting just above the hull's waterline.
  const deck = new THREE.Mesh(new THREE.BoxGeometry(5.2, 2.0, 0.14), inside);
  deck.position.z = -0.42;
  deck.receiveShadow = true;
  group.add(deck);

  // A rim around the top of the hull, and a thwart to sit on.
  const rim = new THREE.Mesh(new THREE.TorusGeometry(1, 0.1, 10, 40), trim);
  rim.scale.set(3.4, 1.35, 1);
  rim.castShadow = true;
  group.add(rim);

  const bench = new THREE.Mesh(new THREE.BoxGeometry(0.85, 2.3, 0.16), trim);
  bench.position.set(-1.3, 0, -0.28);
  bench.castShadow = true;
  group.add(bench);

  // A pair of oars resting in the rowlocks, blades out over the water.
  const oarMaterial = new THREE.MeshStandardMaterial({ color: 0xc9ae7c, roughness: 0.75 });
  const oars: THREE.Group[] = [];
  for (const side of [1, -1]) {
    const oar = new THREE.Group();
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.085, 4.6, 8), oarMaterial);
    shaft.rotation.z = Math.PI / 2;
    oar.add(shaft);
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.85, 0.06, 0.4), oarMaterial);
    blade.position.x = -2.6;
    oar.add(blade);
    oar.position.set(-0.9, side * 1.3, 0.02);
    oar.rotation.set(side * 0.3, 0, 0);
    oar.castShadow = true;
    group.add(oar);
    oars.push(oar);
  }
  group.userData.oars = oars;
  return group;
}

/**
 * Pull on the oars. `stroke` is 0 when they are shipped and 1 while rowing, so
 * they only sweep during the row-back stage. Drawn, like everything else here.
 */
export function updateOars(boat: THREE.Group, elapsedSeconds: number, stroke: number): void {
  const oars = boat.userData.oars as THREE.Group[];
  const sweep = Math.sin(elapsedSeconds * 3.4) * 0.5 * stroke;
  oars.forEach((oar, i) => {
    const side = i === 0 ? 1 : -1;
    oar.rotation.z = sweep;
    oar.rotation.x = side * (0.3 + Math.cos(elapsedSeconds * 3.4) * 0.22 * stroke);
  });
}

/** A driftwood fire with a pan over it, for the cooking and eating stages. */
export function buildCampfire(): THREE.Group {
  const group = new THREE.Group();
  const stone = new THREE.MeshStandardMaterial({ color: 0x6d6f6b, roughness: 1 });
  for (let i = 0; i < 7; i++) {
    const angle = (i / 7) * Math.PI * 2;
    const rock = new THREE.Mesh(new THREE.DodecahedronGeometry(0.28, 0), stone);
    rock.position.set(Math.cos(angle) * 1.05, Math.sin(angle) * 1.05, 0.12);
    rock.rotation.set(i, i * 1.7, i * 0.5);
    rock.castShadow = true;
    group.add(rock);
  }

  const logMaterial = new THREE.MeshStandardMaterial({ color: 0x4a3423, roughness: 0.95 });
  for (const angle of [0.4, 2.5]) {
    const log = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.16, 1.7, 7), logMaterial);
    log.rotation.set(Math.PI / 2, 0, angle);
    log.position.z = 0.2;
    group.add(log);
  }

  // Flames: emissive cones that get scaled and spun per frame. Cheap, and it
  // reads as fire without a particle system or a texture.
  const flames = new THREE.Group();
  for (let i = 0; i < 5; i++) {
    const flame = new THREE.Mesh(
      new THREE.ConeGeometry(0.22 - i * 0.02, 0.9 + i * 0.12, 7),
      new THREE.MeshStandardMaterial({
        color: i < 2 ? 0xffd27a : 0xe8823a,
        emissive: i < 2 ? 0xffb03a : 0xc4451a,
        emissiveIntensity: 1.5,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
      }),
    );
    flame.position.set((i - 2) * 0.13, (i % 2 ? 1 : -1) * 0.1, 0.55 + i * 0.06);
    flames.add(flame);
  }
  group.add(flames);
  group.userData.flames = flames;

  const fireLight = new THREE.PointLight(0xffa542, 3.2, 14, 2);
  fireLight.position.z = 1.1;
  group.add(fireLight);
  group.userData.fireLight = fireLight;

  // The pan, and a fish in it.
  const pan = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.95, 0.82, 0.22, 20),
    new THREE.MeshStandardMaterial({ color: 0x2b2d30, roughness: 0.42, metalness: 0.75 }),
  );
  body.rotation.x = Math.PI / 2;
  pan.add(body);
  const handle = new THREE.Mesh(
    new THREE.CylinderGeometry(0.09, 0.09, 1.5, 8),
    new THREE.MeshStandardMaterial({ color: 0x2b2d30, roughness: 0.5, metalness: 0.7 }),
  );
  handle.rotation.z = Math.PI / 2;
  handle.position.set(-1.6, 0, 0);
  pan.add(handle);
  const fish = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.2, 0.62, 5, 10),
    new THREE.MeshStandardMaterial({ color: 0xd8b48a, roughness: 0.6, emissive: 0x2a1408 }),
  );
  fish.rotation.z = Math.PI / 2;
  fish.position.z = 0.2;
  pan.add(fish);
  pan.position.z = 1.35;
  pan.castShadow = true;
  group.add(pan);
  group.userData.pan = pan;

  group.visible = false;
  return group;
}

/** Animate the fire. `heat` fades it in and out as the stage changes. */
export function updateCampfire(fire: THREE.Group, elapsedSeconds: number, heat: number): void {
  const flames = fire.userData.flames as THREE.Group;
  const light = fire.userData.fireLight as THREE.PointLight;
  flames.children.forEach((flame, i) => {
    const flicker = 0.75 + Math.sin(elapsedSeconds * (7 + i * 1.7) + i) * 0.25;
    flame.scale.set(1, flicker * heat, 1);
    flame.rotation.z = Math.sin(elapsedSeconds * 3 + i) * 0.16;
  });
  light.intensity = heat * (2.6 + Math.sin(elapsedSeconds * 9) * 0.7);
  const pan = fire.userData.pan as THREE.Group;
  pan.position.z = 1.35 + Math.sin(elapsedSeconds * 2.2) * 0.02;
}
