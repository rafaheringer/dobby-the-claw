// Adapted from pollen-robotics/reachy-mini-desktop-app, src/components/viewer3d/GLTFRobot.tsx
// (commit 5796d5b94f07cd889e955e3b9e785eafee1cac89), Apache-2.0, © Pollen Robotics.
// Changes: the pose is read each frame from a getter (no React re-render per sample), a head-only
// mode hides the body, and the rest-pose bounding boxes are reported for camera framing.

import { useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import glbUrl from "../assets/robot/reachy_mini_viz.glb?url";
import type { RobotPose } from "../robot/types";
import { applyHeadOnly, neckParts, norm } from "./headOnly";

// Draco-compressed glb; the decoder is served by us (the CSP allows no CDN).
const DRACO_PATH = "/draco/";
useGLTF.preload(glbUrl, DRACO_PATH);

// The glb is exported Z-up (the robot's native frame, same as the URDF), so headPose applies directly.
const MODEL_SCALE = 0.5; // the glb was modeled ~3x real scale
const DISPLAY_YAW = -Math.PI;
const UNITS_PER_M = 1.7; // head_pose translation (metres) -> model units
const HEAD_Z_MAX = 0.044; // platform reach above / below rest, model units
const HEAD_Z_MIN = -0.085;
// Robot frame (X-fwd, Y-left, Z-up) -> glb model frame: Rz(+90), applied as R_model = M·R·M⁻¹.
const HEAD_FIX = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
const HEAD_FIX_INV = HEAD_FIX.clone().invert();
const YAW_SIGN = 1;
const ANT_AXIS = new THREE.Vector3(0, 0, 1);
const ANT_SIGN = -1;
const SMOOTH_K = 15; // exponential smoothing rate (1/s)

const BONE = { body: "Core", head: "Core.001", antL: "Antenna.L.002", antR: "Antenna.R.002" } as const;

export interface RobotBoxes {
  full: THREE.Box3;
  head: THREE.Box3;
}

interface Props {
  getPose(): RobotPose;
  headOnly: boolean;
  onBoxes?(boxes: RobotBoxes): void;
}

function toMatrix(p: number[]): THREE.Matrix4 | null {
  if (p.length !== 16) return null;
  return new THREE.Matrix4().fromArray(p).transpose(); // row-major (robot) -> column-major (three)
}

// --- Neck leg IK (Stewart platform legs re-solved each frame with CCD) ---
const LEG_IDS = ["A", "B", "C", "D", "E", "F"] as const;
const LEG_IK_ITERATIONS = 12;
type Dof = "free" | "hingeX" | "locked";
const LEG_DOF: Dof[] = ["locked", "hingeX", "locked", "free"];
const HINGE_AXIS = new THREE.Vector3(1, 0, 0);

interface LegChain {
  chain: THREE.Object3D[];
  restQ: THREE.Quaternion[];
  effector: THREE.Object3D;
  target: THREE.Object3D;
}

function setupLeg(id: string, byName: Record<string, THREE.Object3D>, head: THREE.Object3D): LegChain | null {
  const chain = ["001", "002", "003", "004"].map((s) => byName[norm(`Neck.${id}.${s}`)]);
  const ikTarget = byName[norm(`Neck.loc.IK.${id}`)];
  if (chain.some((b) => !b) || !ikTarget) return null;
  const attach = ikTarget.getWorldPosition(new THREE.Vector3());
  const effector = new THREE.Object3D();
  chain[3].add(effector);
  effector.position.copy(chain[3].worldToLocal(attach.clone()));
  const target = new THREE.Object3D();
  head.add(target);
  target.position.copy(head.worldToLocal(attach.clone()));
  return { chain, restQ: chain.map((b) => b.quaternion.clone()), effector, target };
}

const _jp = new THREE.Vector3();
const _ep = new THREE.Vector3();
const _tp = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _hw = new THREE.Vector3();
const _cx = new THREE.Vector3();
const _qr = new THREE.Quaternion();
const _qw = new THREE.Quaternion();
const _qp = new THREE.Quaternion();

function solveLeg(leg: LegChain, iterations: number): void {
  for (let i = 0; i < leg.chain.length; i++) leg.chain[i].quaternion.copy(leg.restQ[i]);
  leg.chain[0].updateWorldMatrix(false, true);
  leg.target.getWorldPosition(_tp);
  for (let it = 0; it < iterations; it++) {
    for (let i = 0; i < leg.chain.length; i++) {
      const dof = LEG_DOF[i];
      if (dof === "locked") continue;
      const joint = leg.chain[i];
      leg.effector.getWorldPosition(_ep);
      if (_ep.distanceToSquared(_tp) < 1e-8) return;
      joint.getWorldPosition(_jp);
      joint.getWorldQuaternion(_qw);
      if (dof === "free") {
        _v1.subVectors(_ep, _jp).normalize();
        _v2.subVectors(_tp, _jp).normalize();
        _qr.setFromUnitVectors(_v1, _v2);
      } else {
        _hw.copy(HINGE_AXIS).applyQuaternion(_qw).normalize();
        _v1.subVectors(_ep, _jp).projectOnPlane(_hw);
        _v2.subVectors(_tp, _jp).projectOnPlane(_hw);
        if (_v1.lengthSq() < 1e-12 || _v2.lengthSq() < 1e-12) continue;
        _v1.normalize();
        _v2.normalize();
        _qr.setFromAxisAngle(_hw, Math.atan2(_cx.crossVectors(_v1, _v2).dot(_hw), _v1.dot(_v2)));
      }
      _qr.multiply(_qw);
      joint.parent?.getWorldQuaternion(_qp);
      joint.quaternion.copy(_qp.invert().multiply(_qr));
      joint.updateWorldMatrix(false, true);
    }
  }
}

export function GLTFRobot({ getPose, headOnly, onBoxes }: Props) {
  const { scene } = useGLTF(glbUrl, DRACO_PATH);
  const model = useMemo(() => skeletonClone(scene) as THREE.Object3D, [scene]);
  const wrapper = useRef<THREE.Group>(null);
  const rig = useRef<{
    bones: Partial<Record<keyof typeof BONE, THREE.Object3D>>;
    rest: Partial<Record<keyof typeof BONE, THREE.Quaternion>>;
    headRest: { parentInv: THREE.Matrix4; q: THREE.Quaternion; p: THREE.Vector3; s: THREE.Vector3 } | null;
    legs: LegChain[];
    bodyYawAxis: THREE.Vector3;
  }>({ bones: {}, rest: {}, headRest: null, legs: [], bodyYawAxis: new THREE.Vector3(0, 0, 1) });

  // Capture bones and the rest pose once per model.
  useEffect(() => {
    const r = rig.current;
    model.updateWorldMatrix(true, true);
    const modelInv = new THREE.Matrix4().copy(model.matrixWorld).invert();
    const byName: Record<string, THREE.Object3D> = {};
    model.traverse((o) => {
      if (o.name) byName[norm(o.name)] = o;
      if ((o as THREE.Mesh).isMesh) o.frustumCulled = false; // skinned/IK parts move outside their static bounds
    });
    for (const [key, name] of Object.entries(BONE) as [keyof typeof BONE, string][]) {
      const node = byName[norm(name)];
      r.bones[key] = node;
      if (node) r.rest[key] = node.quaternion.clone();
    }
    const body = r.bones.body;
    if (body?.parent) {
      const parentModel = new THREE.Matrix4().multiplyMatrices(modelInv, body.parent.matrixWorld);
      const pq = new THREE.Quaternion();
      parentModel.decompose(new THREE.Vector3(), pq, new THREE.Vector3());
      r.bodyYawAxis.set(0, 0, 1).applyQuaternion(pq.invert()).normalize();
    }
    const head = r.bones.head;
    if (head?.parent) {
      const headModel = new THREE.Matrix4().multiplyMatrices(modelInv, head.matrixWorld);
      const parentModel = new THREE.Matrix4().multiplyMatrices(modelInv, head.parent.matrixWorld);
      const q = new THREE.Quaternion();
      const p = new THREE.Vector3();
      const s = new THREE.Vector3();
      headModel.decompose(p, q, s);
      r.headRest = { parentInv: parentModel.invert(), q, p, s };
      r.legs = LEG_IDS.map((id) => setupLeg(id, byName, head)).filter((l): l is LegChain => l !== null);
    }
  }, [model]);

  // Report the rest-pose boxes (in world space, after the display wrapper) for framing.
  useEffect(() => {
    const head = rig.current.bones.head;
    if (!wrapper.current || !head || !onBoxes) return;
    wrapper.current.updateWorldMatrix(true, true);
    // The head as the home page shows it: without the neck parts under the shell.
    const neck = neckParts(head);
    const headBox = new THREE.Box3();
    head.traverse((o) => {
      if ((o as THREE.Mesh).isMesh && !(o as THREE.SkinnedMesh).isSkinnedMesh && !neck.has(o)) headBox.union(new THREE.Box3().setFromObject(o, true));
    });
    // The antenna rods are skinned (their bounds span the whole rig), so frame their tip bones instead.
    model.traverse((o) => {
      if (["antennal003", "antennar003"].includes(norm(o.name))) headBox.expandByPoint(o.getWorldPosition(new THREE.Vector3()));
    });
    const headHeight = headBox.max.y - headBox.min.y;
    const rodsTop = headBox.max.y + headHeight * 0.9; // the rods reach well past their last bone
    headBox.max.y = rodsTop; // whole rods: a crop at the card's hard edge looks like a mistake
    const fullBox = new THREE.Box3().setFromObject(model, true);
    fullBox.max.y = Math.max(fullBox.max.y, rodsTop);
    onBoxes({ full: fullBox, head: headBox });
  }, [model, onBoxes]);

  useEffect(() => applyHeadOnly(model, headOnly), [model, headOnly]);

  const tmp = useRef({ p: new THREE.Vector3(), q: new THREE.Quaternion(), s: new THREE.Vector3(), worldQ: new THREE.Quaternion(), worldP: new THREE.Vector3(), mat: new THREE.Matrix4(), parentInv: new THREE.Matrix4() }).current;
  const sm = useRef({ init: false, smP: new THREE.Vector3(), smQ: new THREE.Quaternion(), tgtP: new THREE.Vector3(), tgtQ: new THREE.Quaternion(), smYaw: 0, smAnt: [0, 0] as [number, number] }).current;

  useFrame((_, delta) => {
    const { bones, rest, headRest: hr, legs, bodyYawAxis } = rig.current;
    const pose = getPose();
    const a = sm.init ? 1 - Math.exp(-SMOOTH_K * Math.min(delta, 0.05)) : 1;
    sm.init = true;

    if (bones.body && rest.body) {
      sm.smYaw += (pose.yawBody - sm.smYaw) * a;
      tmp.q.setFromAxisAngle(bodyYawAxis, YAW_SIGN * sm.smYaw);
      bones.body.quaternion.copy(tmp.q).multiply(rest.body);
    }

    const m = toMatrix(pose.headPose);
    if (bones.head && hr && m) {
      m.decompose(sm.tgtP, sm.tgtQ, tmp.s);
      sm.smP.lerp(sm.tgtP, a);
      sm.smQ.slerp(sm.tgtQ, a);
      tmp.p.copy(sm.smP);
      tmp.q.copy(sm.smQ).premultiply(HEAD_FIX).multiply(HEAD_FIX_INV);
      tmp.p.applyQuaternion(HEAD_FIX);
      tmp.worldQ.copy(tmp.q).multiply(hr.q);
      tmp.worldP.copy(hr.p).addScaledVector(tmp.p, UNITS_PER_M);
      tmp.worldP.z = Math.min(hr.p.z + HEAD_Z_MAX, Math.max(hr.p.z + HEAD_Z_MIN, tmp.worldP.z));
      // The SDK's head pose is relative to the fixed base, not to the turning body: the head's parent
      // (under the body bone) has just been rotated, so place the head against where that parent is now.
      const parent = bones.head.parent;
      if (parent) {
        parent.updateWorldMatrix(true, false);
        tmp.parentInv.copy(model.matrixWorld).invert().multiply(parent.matrixWorld).invert();
      } else tmp.parentInv.copy(hr.parentInv);
      tmp.mat.compose(tmp.worldP, tmp.worldQ, hr.s).premultiply(tmp.parentInv);
      tmp.mat.decompose(bones.head.position, bones.head.quaternion, bones.head.scale);
    }

    if (!headOnly) for (const leg of legs) solveLeg(leg, LEG_IK_ITERATIONS); // legs are hidden in head mode

    sm.smAnt[0] += (pose.antennas[0] - sm.smAnt[0]) * a;
    sm.smAnt[1] += (pose.antennas[1] - sm.smAnt[1]) * a;
    if (bones.antL && rest.antL) bones.antL.quaternion.copy(rest.antL).multiply(tmp.q.setFromAxisAngle(ANT_AXIS, ANT_SIGN * sm.smAnt[1]));
    if (bones.antR && rest.antR) bones.antR.quaternion.copy(rest.antR).multiply(tmp.q.setFromAxisAngle(ANT_AXIS, ANT_SIGN * sm.smAnt[0]));
  });

  return (
    <group ref={wrapper} rotation={[0, DISPLAY_YAW, 0]} scale={MODEL_SCALE}>
      <primitive object={model} rotation={[-Math.PI / 2, 0, 0]} />
    </group>
  );
}
