import { ContactShadows, Environment, Lightformer } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useCallback, useRef, useState } from "react";
import * as THREE from "three";
import { robotLink } from "../robot";
import { CAMERA_FOV, fitDistance } from "./framing";
import { GLTFRobot, type RobotBoxes } from "./GLTFRobot";
import { useStageStatus } from "./stageStatus";

// Room left for the controls in full mode: a bottom sheet on phones, a side panel on desktop.
export const PHONE_PANEL_FRACTION = 0.44;
export const DESKTOP_PANEL_PX = 400;
const DESKTOP_MIN_PX = 1024;

const _center = new THREE.Vector3();
const _size = new THREE.Vector3();
const _wantPos = new THREE.Vector3();

/** Eases the camera toward the framing of the current mode (head close-up or whole robot). */
function CameraRig({ boxes, full, snap }: { boxes: RobotBoxes | null; full: boolean; snap: boolean }) {
  const lookAt = useRef(new THREE.Vector3(0, 0.15, 0));
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
      const panel = height * PHONE_PANEL_FRACTION;
      cam.setViewOffset(width, height, 0, panel / 2, width, height);
      height -= panel;
    } else cam.clearViewOffset();
    const visibleFov = (2 * Math.atan(Math.tan((CAMERA_FOV * Math.PI) / 360) * (height / size.height)) * 180) / Math.PI;
    const distance = fitDistance({ width: Math.max(_size.x, _size.z), height: _size.y }, visibleFov, width / height, full ? 1.2 : 1.4);
    _wantPos.set(_center.x, _center.y + distance * 0.05, _center.z + distance);
    const k = snap ? 1 : 1 - Math.exp(-6 * Math.min(delta, 0.05));
    cam.position.lerp(_wantPos, k);
    lookAt.current.lerp(_center, k);
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
