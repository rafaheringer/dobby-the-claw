import { ContactShadows, Environment, Lightformer } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useCallback, useRef, useState } from "react";
import * as THREE from "three";
import { robotLink } from "../robot";
import { CAMERA_FOV, fitDistance } from "./framing";
import { GLTFRobot, type RobotBoxes } from "./GLTFRobot";
import { FRONT, orbitOffset, useOrbit, type Orbit } from "./orbit";
import { useStageStatus, useTwinPanel } from "./stageStatus";

// Room left for the controls in full mode: a bottom sheet on phones, a side panel on desktop.
export const PHONE_PANEL_FRACTION = 0.44;
export const DESKTOP_PANEL_PX = 400;
const PHONE_TOP_BAR_PX = 60;
const DESKTOP_MIN_PX = 1024;

const _center = new THREE.Vector3();
const _size = new THREE.Vector3();
const BASE_ELEVATION = Math.atan2(0.05, 1); // the default view is slightly above the robot

let snapUntil = 0;
/** Jump the camera straight to its framing for the next `ms` (while the stage is faded out). */
export function snapCamera(ms: number): void {
  snapUntil = performance.now() + ms;
}

/**
 * Eases the camera toward the framing of the current mode (head close-up or whole robot), turned
 * by the person's orbit in full mode (the head close-up always faces front).
 */
function CameraRig({ boxes, full, snap }: { boxes: RobotBoxes | null; full: boolean; snap: boolean }) {
  const lookAt = useRef(new THREE.Vector3(0, 0.15, 0));
  const distance = useRef<number | null>(null);
  const orbit = useRef<Orbit>({ ...FRONT });
  const { size } = useThree();
  useFrame(({ camera }, delta) => {
    if (!boxes) return;
    const cam = camera as THREE.PerspectiveCamera;
    const box = full ? boxes.full : boxes.head;
    box.getCenter(_center);
    box.getSize(_size);
    let width = size.width;
    let height = size.height;
    const desktop = size.width >= DESKTOP_MIN_PX;
    // Keep the robot clear of the controls by shifting the projection center.
    if (full && desktop) {
      cam.setViewOffset(width, height, DESKTOP_PANEL_PX / 2, 0, width, height);
      width -= DESKTOP_PANEL_PX;
    } else if (full) {
      // Visible band: between the top bar (Voltar, mode) and the controls panel at the bottom.
      const panel = useTwinPanel.getState().phonePx || height * PHONE_PANEL_FRACTION;
      cam.setViewOffset(width, height, 0, (panel - PHONE_TOP_BAR_PX) / 2, width, height);
      height -= panel + PHONE_TOP_BAR_PX;
    } else cam.clearViewOffset();
    const visibleFov = (2 * Math.atan(Math.tan((CAMERA_FOV * Math.PI) / 360) * (height / size.height)) * 180) / Math.PI;
    const fit = fitDistance({ width: Math.max(_size.x, _size.z), height: _size.y }, visibleFov, width / height, full ? 1.2 : 1.4);
    const dt = Math.min(delta, 0.05);
    const jump = snap || performance.now() < snapUntil;
    const k = jump ? 1 : 1 - Math.exp(-6 * dt);
    const kOrbit = jump ? 1 : 1 - Math.exp(-14 * dt); // follows the finger closely, with a little glide
    distance.current = distance.current === null ? fit : distance.current + (fit - distance.current) * k;
    lookAt.current.lerp(_center, k);
    // Angles are eased, not the position: a fast drag goes around the robot instead of cutting through it.
    const goal = full ? useOrbit.getState() : FRONT;
    const o = orbit.current;
    o.az += (goal.az - o.az) * kOrbit;
    o.el += (goal.el - o.el) * kOrbit;
    o.zoom += (goal.zoom - o.zoom) * kOrbit;
    const [ox, oy, oz] = orbitOffset(o, distance.current, BASE_ELEVATION);
    cam.position.set(lookAt.current.x + ox, lookAt.current.y + oy, lookAt.current.z + oz);
    cam.lookAt(lookAt.current);
  });
  return null;
}

export function Scene({ full, snap }: { full: boolean; snap: boolean }) {
  const [boxes, setBoxes] = useState<RobotBoxes | null>(null);
  const onBoxes = useCallback((b: RobotBoxes) => {
    setBoxes(b);
    useStageStatus.setState({ ready: true });
  }, []);
  const getPose = useCallback(() => robotLink.tick(), []);

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[0.6, 1.2, 1]} intensity={2.2} />
      <directionalLight position={[-1, 0.4, -0.8]} intensity={0.9} color="#ffffff" />
      <Environment resolution={128}>
        <Lightformer form="rect" intensity={2} position={[0, 2, 2]} scale={[4, 2, 1]} />
        <Lightformer form="rect" intensity={0.8} color="#ffffff" position={[-2, 1, -1]} scale={[2, 2, 1]} />
        <Lightformer form="ring" intensity={0.8} color="#ffffff" position={[2, 1, 1]} scale={1.5} />
      </Environment>
      <GLTFRobot getPose={getPose} headOnly={!full} onBoxes={onBoxes} />
      {full && boxes && <ContactShadows position={[0, boxes.full.min.y, 0]} opacity={0.3} scale={0.8} blur={2.4} far={0.4} color="#111111" />}
      <CameraRig boxes={boxes} full={full} snap={snap} />
    </>
  );
}
