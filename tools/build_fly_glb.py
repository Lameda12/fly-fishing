#!/usr/bin/env python3
"""Convert the NeuroMechFly body in the simulator checkout into one GLB.

The viewer cannot compile MJCF: MuJoCo is a WebAssembly build that upstream's
own browser arena loads, and this layer's viewer is a plain static site. So the
body is converted once, ahead of time, into a glTF binary whose node hierarchy
mirrors the MJCF body tree. The viewer then does forward kinematics on that tree
from joint angles, with no physics engine involved.

What this reads, all from the fetched checkout and none of it modified:

  assets/model/fly.xml        the MJCF: body tree, geoms, joints, mesh scales
  assets/model/*.stl          39 binary STL meshes (66,574 triangles total)
  assets/model_meta.json      neutral pose, and the actuator -> joint mapping

What it writes:

  web/public/fly.glb          one binary glTF, meshes plus a rig description

The rig lives in the glTF's `extras`, because glTF skins are for skinned meshes
and this is a rigid-body tree. `extras.rig.joints` lists every MJCF hinge with
the node it rotates, its axis, its neutral angle, and the control index that
drives it, so the viewer can apply upstream's own CPG output to the right node.

**The output is geometry, not behaviour.** Nothing here simulates anything. The
pose a viewer builds from it is only as honest as the angles it is fed; see the
README on what those angles are and are not.

No third-party Python packages are used, so this runs wherever run.py does.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "embodied-fly-lab"
DEFAULT_OUT = ROOT / "web" / "public" / "fly.glb"

# glTF component and target constants, spelled out so the writer reads plainly.
FLOAT = 5126
UNSIGNED_INT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


# --- small vector and quaternion helpers ------------------------------------
# MJCF quaternions are (w, x, y, z); glTF wants (x, y, z, w). Everything below
# keeps MJCF order internally and converts once, at write time.


def parse_floats(text: str | None, default: list[float]) -> list[float]:
    if text is None:
        return list(default)
    return [float(part) for part in text.split()]


def quat_from_axis_angle(axis: list[float], angle: float) -> list[float]:
    length = math.sqrt(sum(component * component for component in axis)) or 1.0
    half = angle / 2.0
    sin = math.sin(half) / length
    return [math.cos(half), axis[0] * sin, axis[1] * sin, axis[2] * sin]


def quat_multiply(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def body_quat(element: ET.Element) -> list[float]:
    """The body's fixed orientation, as an MJCF (w, x, y, z) quaternion."""
    if element.get("quat") is not None:
        return parse_floats(element.get("quat"), [1, 0, 0, 0])
    if element.get("euler") is not None:
        # The model declares radians in its compiler tag; MJCF applies XYZ.
        rx, ry, rz = parse_floats(element.get("euler"), [0, 0, 0])
        quaternion = [1.0, 0.0, 0.0, 0.0]
        for axis, angle in (([1, 0, 0], rx), ([0, 1, 0], ry), ([0, 0, 1], rz)):
            quaternion = quat_multiply(quaternion, quat_from_axis_angle(axis, angle))
        return quaternion
    if element.get("axisangle") is not None:
        values = parse_floats(element.get("axisangle"), [1, 0, 0, 0])
        return quat_from_axis_angle(values[:3], values[3])
    return [1.0, 0.0, 0.0, 0.0]


# --- STL loading ------------------------------------------------------------


def load_stl(path: Path) -> list[tuple[float, float, float]]:
    """Return one triangle's three corners per group of three, from a binary STL."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path.name} is too short to be a binary STL")
    count = struct.unpack_from("<I", data, 80)[0]
    if 84 + 50 * count != len(data):
        # The upstream assets are all binary; an ASCII one would need a different
        # reader, and failing loudly beats emitting an empty mesh.
        raise ValueError(f"{path.name} is not a binary STL ({count} triangles declared)")
    corners: list[tuple[float, float, float]] = []
    offset = 84
    for _ in range(count):
        # 12 bytes of facet normal are skipped: smooth normals are recomputed
        # below, which is what upstream's own mesh builder does.
        values = struct.unpack_from("<12f", data, offset + 12 - 12)
        corners.append((values[3], values[4], values[5]))
        corners.append((values[6], values[7], values[8]))
        corners.append((values[9], values[10], values[11]))
        offset += 50
    return corners


def build_mesh(corners, scale):
    """Weld duplicate corners, apply the mesh scale, and compute smooth normals.

    A negative component in the scale mirrors the mesh, which reverses triangle
    winding; the winding is flipped back so the recomputed normals still point
    outward. Thirty of the model's meshes are mirrored this way (every right-side
    part reuses its left-side STL).
    """
    flip = (sum(1 for component in scale if component < 0) % 2) == 1

    index_of: dict[tuple[float, float, float], int] = {}
    positions: list[float] = []
    indices: list[int] = []

    for triangle_start in range(0, len(corners), 3):
        triangle = [corners[triangle_start + i] for i in range(3)]
        if flip:
            triangle.reverse()
        for corner in triangle:
            scaled = (corner[0] * scale[0], corner[1] * scale[1], corner[2] * scale[2])
            index = index_of.get(scaled)
            if index is None:
                index = len(positions) // 3
                index_of[scaled] = index
                positions.extend(scaled)
            indices.append(index)

    normals = [0.0] * len(positions)
    for i in range(0, len(indices), 3):
        a, b, c = indices[i] * 3, indices[i + 1] * 3, indices[i + 2] * 3
        ux = positions[b] - positions[a]
        uy = positions[b + 1] - positions[a + 1]
        uz = positions[b + 2] - positions[a + 2]
        vx = positions[c] - positions[a]
        vy = positions[c + 1] - positions[a + 1]
        vz = positions[c + 2] - positions[a + 2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        for base in (a, b, c):
            normals[base] += nx
            normals[base + 1] += ny
            normals[base + 2] += nz
    for i in range(0, len(normals), 3):
        length = math.sqrt(normals[i] ** 2 + normals[i + 1] ** 2 + normals[i + 2] ** 2)
        if length > 0:
            normals[i] /= length
            normals[i + 1] /= length
            normals[i + 2] /= length
        else:
            normals[i + 2] = 1.0
    return positions, normals, indices


# --- materials --------------------------------------------------------------


def material_colors(asset: ET.Element) -> dict[str, list[float]]:
    """Base colour per MJCF material.

    Materials with an explicit rgba use it. The rest are procedurally textured
    (flat or gradient noise), and their texture's rgb1 is the base tone, which is
    what those surfaces read as. Procedural textures are not reproduced here: the
    GLB carries flat colours.
    """
    textures = {
        texture.get("name"): parse_floats(texture.get("rgb1"), [0.7, 0.7, 0.7])
        for texture in asset.findall("texture")
        if texture.get("name")
    }
    colors: dict[str, list[float]] = {}
    for material in asset.findall("material"):
        name = material.get("name")
        if not name:
            continue
        if material.get("rgba"):
            values = parse_floats(material.get("rgba"), [0.7, 0.7, 0.7, 1.0])
            colors[name] = (values + [1.0])[:4]
        else:
            rgb = textures.get(material.get("texture", ""), [0.7, 0.7, 0.7])
            colors[name] = [*rgb, 1.0]
    return colors


# --- glTF assembly ----------------------------------------------------------


class GlbBuilder:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []

    def _align(self) -> None:
        while len(self.buffer) % 4:
            self.buffer.append(0)

    def add_floats(self, values: list[float], components: int, target: int) -> int:
        self._align()
        offset = len(self.buffer)
        self.buffer.extend(struct.pack(f"<{len(values)}f", *values))
        self.buffer_views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(values) * 4, "target": target}
        )
        count = len(values) // components
        # glTF requires min/max on POSITION; giving them for every float accessor
        # is harmless and saves a special case.
        mins = [min(values[i::components]) for i in range(components)]
        maxs = [max(values[i::components]) for i in range(components)]
        self.accessors.append(
            {
                "bufferView": len(self.buffer_views) - 1,
                "componentType": FLOAT,
                "count": count,
                "type": {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[components],
                "min": mins,
                "max": maxs,
            }
        )
        return len(self.accessors) - 1

    def add_indices(self, indices: list[int]) -> int:
        self._align()
        offset = len(self.buffer)
        self.buffer.extend(struct.pack(f"<{len(indices)}I", *indices))
        self.buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(indices) * 4,
                "target": ELEMENT_ARRAY_BUFFER,
            }
        )
        self.accessors.append(
            {
                "bufferView": len(self.buffer_views) - 1,
                "componentType": UNSIGNED_INT,
                "count": len(indices),
                "type": "SCALAR",
                "min": [min(indices)] if indices else [0],
                "max": [max(indices)] if indices else [0],
            }
        )
        return len(self.accessors) - 1

    def write(self, gltf: dict, path: Path) -> int:
        self._align()
        gltf["buffers"] = [{"byteLength": len(self.buffer)}]
        gltf["bufferViews"] = self.buffer_views
        gltf["accessors"] = self.accessors

        json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_chunk += b" " * (-len(json_chunk) % 4)
        binary_chunk = bytes(self.buffer)

        total = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
        with path.open("wb") as handle:
            handle.write(struct.pack("<III", 0x46546C67, 2, total))
            handle.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
            handle.write(json_chunk)
            handle.write(struct.pack("<II", len(binary_chunk), 0x004E4942))
            handle.write(binary_chunk)
        return total


def convert(model_dir: Path, meta_path: Path, out: Path, quiet: bool = False) -> int:
    def say(text: str) -> None:
        if not quiet:
            print(text)

    root = ET.parse(model_dir / "fly.xml").getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("fly.xml has no <asset> or <worldbody>")
    meta = json.loads(meta_path.read_text())

    colors = material_colors(asset)

    # joint name (without the model prefix) -> its control index and neutral
    actuator_by_joint = {
        actuator["joint"]: actuator for actuator in meta.get("actuators", [])
    }
    neutral_qpos = meta.get("neutral_qpos", [])

    builder = GlbBuilder()
    gltf: dict = {
        "asset": {
            "version": "2.0",
            "generator": "fly-fishing tools/build_fly_glb.py",
            "copyright": (
                "Geometry from NeuroMechFly v2 / FlyGym (NeLy, EPFL), Apache-2.0, "
                "via statsleelab/embodied-fly-lab. Converted, not modified."
            ),
        },
        "scene": 0,
        "materials": [],
        "meshes": [],
        "nodes": [],
        "scenes": [{"nodes": []}],
    }

    # --- materials
    material_index: dict[str, int] = {}
    for name, rgba in sorted(colors.items()):
        if name == "grid":
            continue  # the ground plane is the viewer's, not the fly's
        material_index[name] = len(gltf["materials"])
        gltf["materials"].append(
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": rgba,
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.72,
                },
                "doubleSided": True,
                **({"alphaMode": "BLEND"} if rgba[3] < 1.0 else {}),
            }
        )

    # --- meshes, one glTF mesh per MJCF <mesh> asset
    stl_cache: dict[str, list] = {}
    mesh_index: dict[str, int] = {}
    triangles = 0
    for mesh in asset.findall("mesh"):
        name = mesh.get("name")
        file_name = mesh.get("file")
        if not name or not file_name:
            continue
        if file_name not in stl_cache:
            stl_cache[file_name] = load_stl(model_dir / file_name)
        scale = parse_floats(mesh.get("scale"), [1.0, 1.0, 1.0])
        positions, normals, indices = build_mesh(stl_cache[file_name], scale)
        triangles += len(indices) // 3
        position_accessor = builder.add_floats(positions, 3, ARRAY_BUFFER)
        normal_accessor = builder.add_floats(normals, 3, ARRAY_BUFFER)
        index_accessor = builder.add_indices(indices)
        mesh_index[name] = len(gltf["meshes"])
        gltf["meshes"].append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
                        "indices": index_accessor,
                    }
                ],
            }
        )
    say(f"  {len(gltf['meshes'])} meshes, {triangles:,} triangles, {len(gltf['materials'])} materials")

    # --- nodes, mirroring the MJCF body tree
    rig_joints: list[dict] = []
    missing_actuators: list[str] = []

    def add_body(element: ET.Element) -> int:
        name = element.get("name") or "body"
        node: dict = {"name": name}
        position = parse_floats(element.get("pos"), [0.0, 0.0, 0.0])
        if any(position):
            node["translation"] = position
        quaternion = body_quat(element)
        if quaternion != [1.0, 0.0, 0.0, 0.0]:
            # MJCF (w, x, y, z) -> glTF (x, y, z, w)
            node["rotation"] = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        gltf["nodes"].append(node)
        index = len(gltf["nodes"]) - 1
        children: list[int] = []

        for geom in element.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None or mesh_name not in mesh_index:
                continue  # non-mesh geoms are collision primitives; nothing to draw
            geom_node: dict = {
                "name": geom.get("name") or mesh_name,
                "mesh": mesh_index[mesh_name],
            }
            geom_position = parse_floats(geom.get("pos"), [0.0, 0.0, 0.0])
            if any(geom_position):
                geom_node["translation"] = geom_position
            geom_quaternion = parse_floats(geom.get("quat"), [1.0, 0.0, 0.0, 0.0])
            if geom_quaternion != [1.0, 0.0, 0.0, 0.0]:
                geom_node["rotation"] = [
                    geom_quaternion[1],
                    geom_quaternion[2],
                    geom_quaternion[3],
                    geom_quaternion[0],
                ]
            material = geom.get("material")
            if material in material_index:
                gltf["meshes"][mesh_index[mesh_name]]["primitives"][0]["material"] = (
                    material_index[material]
                )
            gltf["nodes"].append(geom_node)
            children.append(len(gltf["nodes"]) - 1)

        # Hinges on this body rotate this node, in declaration order. Every hinge
        # in this model is anchored at the body origin, which is what lets the
        # viewer compose them straight into the node's rotation.
        hinges = [joint for joint in element.findall("joint") if joint.get("type", "hinge") == "hinge"]
        for order, joint in enumerate(hinges):
            joint_name = joint.get("name") or f"{name}-joint{order}"
            anchor = parse_floats(joint.get("pos"), [0.0, 0.0, 0.0])
            if any(anchor):
                raise ValueError(
                    f"joint {joint_name} is anchored at {anchor}, not the body origin; "
                    "the viewer's forward kinematics assumes origin anchors"
                )
            bare = joint_name.split("/", 1)[-1]
            actuator = actuator_by_joint.get(bare)
            # MuJoCo numbers qpos in declaration order: the root free joint takes
            # slots 0-6, then one slot per hinge. That gives a neutral angle for
            # the unactuated hinges too, which the actuator table does not cover.
            # The assertion below checks the numbering against every actuated
            # joint rather than trusting it.
            qposadr = 7 + len(rig_joints)
            if actuator is not None and actuator["qposadr"] != qposadr:
                raise ValueError(
                    f"qpos numbering disagrees at {bare}: walked to {qposadr}, "
                    f"model_meta says {actuator['qposadr']}"
                )
            if actuator is None:
                missing_actuators.append(bare)
            rig_joints.append(
                {
                    "name": bare,
                    "node": index,
                    "order": order,
                    "axis": parse_floats(joint.get("axis"), [0.0, 0.0, 1.0]),
                    "range": parse_floats(joint.get("range"), []) or None,
                    "ctrlIndex": actuator["id"] if actuator else None,
                    "qposadr": qposadr,
                    "neutral": neutral_qpos[qposadr] if qposadr < len(neutral_qpos) else 0.0,
                }
            )

        for child in element.findall("body"):
            children.append(add_body(child))
        if children:
            node["children"] = children
        return index

    for body in worldbody.findall("body"):
        gltf["scenes"][0]["nodes"].append(add_body(body))

    actuated = sum(1 for joint in rig_joints if joint["ctrlIndex"] is not None)
    expected_nq = 7 + len(rig_joints)
    if meta.get("nq") not in (None, expected_nq):
        raise ValueError(f"walked {len(rig_joints)} hinges (nq {expected_nq}), model_meta says nq {meta['nq']}")
    say(
        f"  {len(gltf['nodes'])} nodes, {len(rig_joints)} hinges "
        f"({actuated} driven by an actuator, {len(rig_joints) - actuated} held at neutral)"
    )
    if missing_actuators:
        say(f"  unactuated: {', '.join(missing_actuators[:6])}{' ...' if len(missing_actuators) > 6 else ''}")

    gltf["extras"] = {
        "rig": {
            "schemaVersion": 1,
            "source": "statsleelab/embodied-fly-lab assets/model/fly.xml",
            "units": "millimetres, as the MJCF declares them (mesh scale 1000)",
            "joints": rig_joints,
            "adhesionCtrlIndices": meta.get("adhesion", []),
            "note": (
                "Hinges on the same node compose in `order`, applied after the node's "
                "own rotation. Every hinge is anchored at its body origin. Joints with "
                "a null ctrlIndex are not actuated and hold `neutral`."
            ),
        },
        "provenance": {
            "geometry": (
                "NeuroMechFly v2 / FlyGym (NeLy, EPFL), Apache-2.0, fetched via "
                "statsleelab/embodied-fly-lab. Converted here, never modified."
            ),
            "disclaimer": (
                "This file is geometry and a joint list. It simulates nothing, and a pose "
                "built from it is only as meaningful as the angles supplied."
            ),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    size = builder.write(gltf, out)
    say(f"  wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} ({size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sim", default=str(VENDOR), help="simulator checkout (default: vendor/embodied-fly-lab)"
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"output GLB (default: {DEFAULT_OUT})")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sim = Path(args.sim)
    model_dir = sim / "assets" / "model"
    meta_path = sim / "assets" / "model_meta.json"
    if not (model_dir / "fly.xml").exists() or not meta_path.exists():
        print(
            f"No fly model at {model_dir}\nRun: python3 tools/fetch_sim.py",
            file=sys.stderr,
        )
        return 1
    return convert(model_dir, meta_path, Path(args.out), quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
