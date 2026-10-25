import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { applyHeadOnly } from "./headOnly";

function robot() {
  const named = <T extends THREE.Object3D>(o: T, name: string) => ((o.name = name), o);
  const mesh = (name: string) => named(new THREE.Mesh(), name);
  const antennaSkin = named(new THREE.SkinnedMesh(), "Cylinder.036"); // the antenna rods are skinned to the antenna bones
  const head = named(new THREE.Object3D(), "Core001"); // GLTFLoader strips the dot from "Core.001"
  const box = (name: string, size: number, y: number) => {
    const m = named(new THREE.Mesh(new THREE.BoxGeometry(size, size, size)), name);
    m.position.y = y;
    return m;
  };
  const headMesh = box("HeadShell", 1, 0); // spans y -0.5..0.5
  const neckMesh = box("Boule.011", 0.1, -0.7); // under the shell
  const antenna = named(new THREE.Object3D(), "Antenna.L.002");
  const antennaMesh = mesh("Pivot.L");
  antenna.add(antennaMesh);
  head.add(headMesh, neckMesh, antenna);
  const body = mesh("BodyShell");
  const core = named(new THREE.Object3D(), "Core");
  core.add(body, head);
  const root = named(new THREE.Object3D(), "Armature");
  root.add(antennaSkin, named(new THREE.Object3D(), "GLOBAL").add(named(new THREE.Object3D(), "Base").add(core)));
  return { root, antennaSkin, headMesh, neckMesh, antennaMesh, body };
}

describe("applyHeadOnly", () => {
  it("hides everything but the head subtree and the skinned antennas", () => {
    const r = robot();
    applyHeadOnly(r.root, true);
    expect(r.headMesh.visible).toBe(true);
    expect(r.antennaMesh.visible).toBe(true);
    expect(r.body.visible).toBe(false);
    expect(r.neckMesh.visible).toBe(false);
    expect(r.antennaSkin.visible).toBe(true);
  });

  it("shows everything again when off", () => {
    const r = robot();
    applyHeadOnly(r.root, true);
    applyHeadOnly(r.root, false);
    for (const m of [r.headMesh, r.neckMesh, r.antennaMesh, r.body, r.antennaSkin]) expect(m.visible).toBe(true);
  });
});
