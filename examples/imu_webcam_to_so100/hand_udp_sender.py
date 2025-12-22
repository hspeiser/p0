#!/usr/bin/env python
import argparse
import os
import socket
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


class HandLandmark:
    WRIST = 0
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    PINKY_MCP = 17


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


def compute_hand_frame(landmarks):
    wrist = landmarks[HandLandmark.WRIST]
    index_mcp = landmarks[HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = landmarks[HandLandmark.PINKY_MCP]
    middle_mcp = landmarks[HandLandmark.MIDDLE_FINGER_MCP]

    x = normalize(index_mcp - pinky_mcp)
    y = normalize(middle_mcp - wrist)
    if x is None or y is None:
        return None

    z = np.cross(x, y)
    z = normalize(z)
    if z is None:
        return None

    if z[2] < 0:
        z = -z
    y = np.cross(z, x)
    y = normalize(y)
    if y is None:
        return None

    return x, y, z


def compute_pinch_2d(landmarks):
    thumb_tip = landmarks[HandLandmark.THUMB_TIP]
    index_tip = landmarks[HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks[HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = landmarks[HandLandmark.PINKY_MCP]

    dist = np.linalg.norm(thumb_tip - index_tip)
    scale = np.linalg.norm(index_mcp - pinky_mcp)
    if scale < 1e-6:
        return dist
    return dist / scale


def select_hand(handedness, prefer):
    if not handedness:
        return None
    if prefer in {"right", "left"}:
        target = "Right" if prefer == "right" else "Left"
        for i, h in enumerate(handedness):
            if h[0].category_name == target:
                return i, h[0].score
        return None
    best = None
    for i, h in enumerate(handedness):
        score = h[0].score
        if best is None or score > best[1]:
            best = (i, score)
    return best


def draw_hand(image, landmarks, connections, color=(0, 255, 255)):
    h, w, _ = image.shape
    for c0, c1 in connections:
        p0 = landmarks[c0]
        p1 = landmarks[c1]
        x0, y0 = int(p0[0] * w), int(p0[1] * h)
        x1, y1 = int(p1[0] * w), int(p1[1] * h)
        cv2.line(image, (x0, y0), (x1, y1), color, 2)
    for p in landmarks:
        x, y = int(p[0] * w), int(p[1] * h)
        cv2.circle(image, (x, y), 3, color, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--hand", type=str, default="any", choices=["any", "right", "left"])
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.normpath(os.path.join(base_dir, "..", "webcam_to_so100_EE", "models"))
    hand_model_path = os.path.join(model_dir, "hand_landmarker.task")

    base_options_hand = mp_tasks.BaseOptions(model_asset_path=hand_model_path)
    options_hand = vision.HandLandmarkerOptions(
        base_options=base_options_hand,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
    hand_connections = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

    last_ts = 0
    dt = 1.0 / max(args.fps, 1.0)

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            ts_ms = int(time.time() * 1000)
            if ts_ms <= last_ts:
                ts_ms = last_ts + 1
            last_ts = ts_ms

            result = hand_landmarker.detect_for_video(mp_image, ts_ms)

            msg = {
                "kind": "hand",
                "t": time.time(),
                "conf": 0.0,
                "pinch": 0.0,
                "valid": False,
                "frame_valid": False,
                "pinch_valid": False,
            }

            right = select_hand(result.handedness, args.hand)
            axes = None
            pinch = None
            frame_valid = False
            pinch_valid = False
            landmarks_2d = None

            if right is not None:
                idx, score = right

                if result.hand_landmarks:
                    lm2d = result.hand_landmarks[idx]
                    landmarks_2d = np.array([[p.x, p.y] for p in lm2d], dtype=np.float32)
                    pinch = compute_pinch_2d(landmarks_2d)
                    pinch_valid = True

                if result.hand_world_landmarks:
                    lm3d = result.hand_world_landmarks[idx]
                    landmarks_3d = np.array([[p.x, p.y, p.z] for p in lm3d], dtype=np.float32)
                elif result.hand_landmarks:
                    lm3d = result.hand_landmarks[idx]
                    landmarks_3d = np.array([[p.x, p.y, p.z] for p in lm3d], dtype=np.float32)
                else:
                    landmarks_3d = None

                if landmarks_3d is not None:
                    axes = compute_hand_frame(landmarks_3d)
                    if axes is not None:
                        frame_valid = True

                msg = {
                    "kind": "hand",
                    "t": time.time(),
                    "conf": float(score),
                    "pinch": float(pinch) if pinch is not None else 0.0,
                    "valid": bool(frame_valid),
                    "frame_valid": bool(frame_valid),
                    "pinch_valid": bool(pinch_valid),
                }

                if frame_valid:
                    x, y, z = axes
                    msg.update(
                        {
                            "x": [float(v) for v in x],
                            "y": [float(v) for v in y],
                            "z": [float(v) for v in z],
                        }
                    )

            sock.sendto(str.encode(mp_json(msg)), target)

            if not args.no_display:
                if landmarks_2d is not None:
                    draw_hand(frame, landmarks_2d, hand_connections)
                    t = landmarks_2d[HandLandmark.THUMB_TIP]
                    i = landmarks_2d[HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = frame.shape
                    pt1 = (int(t[0] * w), int(t[1] * h))
                    pt2 = (int(i[0] * w), int(i[1] * h))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

                status = "OK" if frame_valid else "NO-FRAME"
                pinch_txt = f"{msg['pinch']:.3f}" if pinch_valid else "NA"
                cv2.putText(
                    frame,
                    f"hand {args.hand} conf {msg['conf']:.2f} frame {status} pinch {pinch_txt}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("hand_udp_sender", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            elapsed = time.time() - t0
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def mp_json(data):
    import json

    return json.dumps(data, separators=(",", ":"))


if __name__ == "__main__":
    main()
