import * as THREE from "three";

// GLTFLoader strips "." from node names ("Core.001" -> "Core001"); compare normalized names.
export const norm = (s: string): string => s.replace(/[^a-z0-9]/gi, "").toLowerCase();

const _box = new THREE.Box3();

/**
 * The parts hanging under the head shell (the platform's top plate, its ball joints and bolts):
 * they read as a neck once the body is hidden. Found by position, not by name: every mesh of the
 * head whose middle sits below the bottom of the shell (its largest mesh).
 */
export function neckParts(head: THREE.Object3D): Set<THREE.Object3D> {
  head.updateWorldMatrix(true, true);
  const parts: { mesh: THREE.Object3D; mid: number; size: number }[] = [];
  let bottom = -Infinity;
  let largest = 0;
  head.traverse((o) => {
    if (!(o as THREE.Mesh).isMesh || (o as THREE.SkinnedMesh).isSkinnedMesh) return;
    _box.setFromObject(o, true);
    const size = _box.getSize(new THREE.Vector3()).lengthSq();
    parts.push({ mesh: o, mid: (_box.min.y + _box.max.y) / 2, size });
    if (size > largest) [largest, bottom] = [size, _box.min.y];
  });
  return new Set(parts.filter((p) => p.mid < bottom).map((p) => p.mesh));
}

/**
 * Show only the head (bone "Core.001" and below, without its neck) or, when `on` is false, the whole robot.
 * The only skinned meshes in Pollen's model are the two antenna rods, so they stay visible.
 */
export function applyHeadOnly(root: THREE.Object3D, on: boolean, headName = "core001"): void {
  let head: THREE.Object3D | undefined;
  root.traverse((o) => {
    if (!head && norm(o.name) === headName) head = o;
  });
  const inHead = new Set<THREE.Object3D>();
  head?.traverse((o) => inHead.add(o));
  if (head) for (const part of neckParts(head)) inHead.delete(part);
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) o.visible = !on || inHead.has(o) || (o as THREE.SkinnedMesh).isSkinnedMesh === true;
  });
}
