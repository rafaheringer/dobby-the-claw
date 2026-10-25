import type * as THREE from "three";

// GLTFLoader strips "." from node names ("Core.001" -> "Core001"); compare normalized names.
export const norm = (s: string): string => s.replace(/[^a-z0-9]/gi, "").toLowerCase();

/**
 * Show only the head (bone "Core.001" and below) or, when `on` is false, the whole robot.
 * The only skinned meshes in Pollen's model are the two antenna rods, so they stay visible.
 */
export function applyHeadOnly(root: THREE.Object3D, on: boolean, headName = "core001"): void {
  let head: THREE.Object3D | undefined;
  root.traverse((o) => {
    if (!head && norm(o.name) === headName) head = o;
  });
  const inHead = new Set<THREE.Object3D>();
  head?.traverse((o) => inHead.add(o));
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) o.visible = !on || inHead.has(o) || (o as THREE.SkinnedMesh).isSkinnedMesh === true;
  });
}
