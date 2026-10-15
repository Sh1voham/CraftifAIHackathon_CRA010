"""Single-image sanity test for the WorkerFallDetection YOLO11s-Pose ONNX
model (see /home/kushal/Craftifai/UseCases/WorkerFallDetection/DeepStream/pgie.txt).

Model: Models/yolo11s-pose.onnx
  - Ultralytics YOLO11s-Pose, single class ("person", Models/labels.txt),
    17 COCO keypoints. Exported via Models/export_yolo11_pose.py, which is
    the marcoslucianops DeepStream-Yolo-Pose convention: the raw Ultralytics
    head output [1, 56, 8400] is wrapped in a `DeepStreamOutput` module that
    just transposes to [1, 8400, 56] -- box decode (xyxy, 640-space pixels),
    objectness sigmoid, and keypoint pixel-space decode + per-keypoint
    visibility sigmoid are ALL already baked into the graph by Ultralytics'
    own Pose head at export time (confirmed empirically: box/keypoint
    columns range ~0-640, confidence/visibility columns range 0-1).
  - Column layout per anchor: [x1, y1, x2, y2, conf, (kx, ky, kv) x 17].
  - NMS is NOT baked in (cluster-mode=4/None in pgie.txt -> the custom
    NvDsInferParseYoloPose parser does it) -> standard single-class NMS here.

Preprocessing constants from pgie.txt: net-scale-factor=1/255,
model-color-format=0 (RGB), maintain-aspect-ratio=1 + symmetric-padding=1
-> standard Ultralytics letterbox (grey (114,114,114) pad, split evenly).

Fall heuristic (this script only, NOT part of the exported model): a
single still image has no motion signal, so "fall" is estimated from body
orientation -- a wide/short box and a near-horizontal shoulder-hip line both
indicate a person lying down rather than standing. This is a simple,
clearly-approximate heuristic for a one-frame sanity check, not a trained
classifier.
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["person"]  # Models/labels.txt

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]  # standard COCO-17 order used by Ultralytics pose models

SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]  # Ultralytics' default COCO skeleton (0-indexed)

# ---------------------------------------------------------------------------
# Step 2 — model path resolver (ships verbatim)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIRS = ("Models", "model", "models", "weights", "onnx", ".")
DEFAULT_HF_REPO = None

HF_REF = re.compile(
    r"^(?:https?://huggingface\.co/)?"
    r"(?P<repo>[\w.-]+/[\w.-]+)"
    r"(?:/(?:blob|resolve)/(?P<rev>[^/]+)/(?P<file>.+?))?/?$"
)


def looks_like_hf_ref(value):
    if Path(value).expanduser().exists():
        return False
    if value.startswith(("http://", "https://")):
        return "huggingface.co/" in value
    if value.startswith((".", "/", "~")):
        return False
    if value.endswith(".onnx") and "/blob/" not in value and "/resolve/" not in value:
        return False
    return bool(HF_REF.match(value))


def fetch_from_hf(ref, local_dir="Models"):
    from huggingface_hub import hf_hub_download, list_repo_files

    m = HF_REF.match(ref)
    repo_id, revision, filename = m.group("repo"), m.group("rev") or "main", m.group("file")
    token = os.environ.get("HF_TOKEN")

    files = list_repo_files(repo_id, revision=revision, token=token)
    if filename is None:
        onnx_files = [f for f in files if f.endswith(".onnx")]
        if not onnx_files:
            ckpts = [f for f in files if f.endswith((".pt", ".pth", ".safetensors"))]
            raise FileNotFoundError(
                f"{repo_id} has no .onnx. Checkpoints present: {ckpts or 'none'}"
            )
        if len(onnx_files) > 1:
            raise RuntimeError(
                f"{repo_id} has {len(onnx_files)} .onnx files — pick one with\n  "
                + "\n  ".join(f"-m {repo_id}/blob/{revision}/{f}" for f in onnx_files)
            )
        filename = onnx_files[0]

    path = hf_hub_download(repo_id, filename, revision=revision, local_dir=local_dir, token=token)
    for sidecar in (f"{filename}.data", f"{filename}_data", f"{filename}.data_0"):
        if sidecar in files:
            hf_hub_download(repo_id, sidecar, revision=revision, local_dir=local_dir, token=token)
    print(f"[model] fetched {repo_id}:{filename} -> {path}")
    return path


def resolve_model_path(explicit=None, search_dirs=DEFAULT_MODEL_DIRS):
    if explicit:
        if looks_like_hf_ref(explicit):
            return fetch_from_hf(explicit)
        p = Path(explicit).expanduser()
        if p.is_file():
            return str(p)
        if p.is_dir():
            search_dirs = (p,)
        else:
            raise FileNotFoundError(f"--model path does not exist: {p}")

    seen, candidates = set(), []
    for d in search_dirs:
        d = Path(d).expanduser()
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.onnx")):
            if any(part.startswith(".") for part in f.parts):
                continue
            rp = f.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            candidates.append(f)
        if candidates:
            break

    if not candidates:
        if DEFAULT_HF_REPO:
            return fetch_from_hf(DEFAULT_HF_REPO)
        raise FileNotFoundError(
            "No .onnx found in: " + ", ".join(str(d) for d in search_dirs)
            + "\nPass an explicit path with -m /path/to/model.onnx"
        )
    if len(candidates) > 1:
        listing = "\n  ".join(f"{c}  ({c.stat().st_size / 1e6:.1f} MB)" for c in candidates)
        raise RuntimeError(f"Found {len(candidates)} .onnx files — pick one with -m:\n  {listing}")

    print(f"[model] auto-resolved: {candidates[0]}")
    return str(candidates[0])


# --- decode constants (from pgie.txt) ---
INPUT_SIZE = 640
NET_SCALE_FACTOR = 0.0039215697906911373  # 1/255
CONF_THRESHOLD = 0.25    # pre-cluster-threshold
NMS_IOU_THRESHOLD = 0.45  # nms-iou-threshold
KPT_VIS_THRESHOLD = 0.5   # only draw/consider keypoints the model is confident about

# fall heuristic thresholds (this script only — see module docstring)
FALL_ASPECT_RATIO = 1.3   # box_w / box_h above this looks "lying down"
FALL_TORSO_ANGLE_DEG = 45  # shoulder-hip line within this many degrees of horizontal


class ModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.ort_session = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        inp = self.ort_session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.output_names = [o.name for o in self.ort_session.get_outputs()]

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        self.orig_h, self.orig_w = image.shape[:2]

        gain = min(INPUT_SIZE / self.orig_w, INPUT_SIZE / self.orig_h)
        new_w, new_h = round(self.orig_w * gain), round(self.orig_h * gain)
        pad_w, pad_h = (INPUT_SIZE - new_w) / 2, (INPUT_SIZE - new_h) / 2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
        left, right = round(pad_w - 0.1), round(pad_w + 0.1)
        letterboxed = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=(114, 114, 114))
        self.gain, self.pad_w, self.pad_h = gain, pad_w, pad_h

        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) * NET_SCALE_FACTOR
        blob = blob.transpose(2, 0, 1)[None, ...]
        return np.ascontiguousarray(blob)

    def run_inference(self, image_path):
        blob = self.preprocess_image(image_path)
        return self.ort_session.run(self.output_names, {self.input_name: blob})[0]

    def _to_original(self, x, y):
        return (x - self.pad_w) / self.gain, (y - self.pad_h) / self.gain

    def postprocess(self, output, confidence_threshold=CONF_THRESHOLD, nms_threshold=NMS_IOU_THRESHOLD):
        preds = output[0]  # [8400, 56]: x1,y1,x2,y2,conf,(kx,ky,kv)*17
        mask = preds[:, 4] >= confidence_threshold
        preds = preds[mask]
        if len(preds) == 0:
            return []

        boxes_xyxy = preds[:, :4]
        scores = preds[:, 4]
        keypoints = preds[:, 5:].reshape(-1, 17, 3)

        boxes_xywh = np.stack([
            boxes_xyxy[:, 0], boxes_xyxy[:, 1],
            boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
        ], axis=1)
        keep = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), confidence_threshold, nms_threshold)
        keep = np.array(keep).flatten() if len(keep) else []

        detections = []
        for i in keep:
            x1, y1, x2, y2 = boxes_xyxy[i]
            x1, y1 = self._to_original(x1, y1)
            x2, y2 = self._to_original(x2, y2)

            kpts = []
            for kx, ky, kv in keypoints[i]:
                ox, oy = self._to_original(kx, ky)
                kpts.append((ox, oy, float(kv)))

            detections.append({
                "box": [max(0, x1), max(0, y1), min(self.orig_w, x2), min(self.orig_h, y2)],
                "score": float(scores[i]),
                "keypoints": kpts,
                "fallen": self._is_fallen([max(0, x1), max(0, y1), min(self.orig_w, x2), min(self.orig_h, y2)], kpts),
            })
        return detections

    @staticmethod
    def _is_fallen(box, kpts):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        aspect_fall = h > 0 and (w / h) >= FALL_ASPECT_RATIO

        torso_fall = False
        l_sh, r_sh, l_hip, r_hip = kpts[5], kpts[6], kpts[11], kpts[12]
        if min(l_sh[2], r_sh[2], l_hip[2], r_hip[2]) >= KPT_VIS_THRESHOLD:
            shoulder_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
            hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
            dx = hip_mid[0] - shoulder_mid[0]
            dy = hip_mid[1] - shoulder_mid[1]
            angle_from_vertical = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))
            torso_fall = angle_from_vertical >= (90 - FALL_TORSO_ANGLE_DEG)

        return bool(aspect_fall or torso_fall)

    def draw_detections(self, image_path, detections, output_path):
        image = cv2.imread(image_path)
        fallen_count = 0
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            fallen = det["fallen"]
            fallen_count += int(fallen)
            box_color = (0, 0, 255) if fallen else (0, 255, 0)
            label = f'{"FALL" if fallen else "person"} {det["score"]:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(image, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            for a, b in SKELETON:
                xa, ya, va = det["keypoints"][a]
                xb, yb, vb = det["keypoints"][b]
                if va >= KPT_VIS_THRESHOLD and vb >= KPT_VIS_THRESHOLD:
                    cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (255, 200, 0), 2)
            for x, y, v in det["keypoints"]:
                if v >= KPT_VIS_THRESHOLD:
                    cv2.circle(image, (int(x), int(y)), 3, (0, 165, 255), -1)

        banner = f"person(s): {len(detections)}   fallen: {fallen_count}"
        cv2.rectangle(image, (0, 0), (420, 40), (0, 0, 0), -1)
        cv2.putText(image, banner, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if fallen_count else (0, 255, 0), 2)
        cv2.imwrite(output_path, image)
        return fallen_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-image sanity test for the YOLO11s-Pose fall-detection ONNX model.")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a Hugging Face ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-o", "--output", default="output.jpg")
    parser.add_argument("-c", "--conf-threshold", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--nms-threshold", type=float, default=NMS_IOU_THRESHOLD)
    args = parser.parse_args()

    model = ModelTester(resolve_model_path(args.model))
    print(f"[input ] {model.input_name} {model.input_shape}")
    print(f"[output] {model.output_names}")

    output = model.run_inference(args.image)
    print(f"[raw   ] output shape {output.shape}, conf min/max: "
          f"{output[0][:, 4].min():.4f}/{output[0][:, 4].max():.4f}")

    result = model.postprocess(output, confidence_threshold=args.conf_threshold, nms_threshold=args.nms_threshold)
    fallen_count = model.draw_detections(args.image, result, args.output)

    print(f"[result] {len(result)} person(s) detected, {fallen_count} flagged as fallen")
    for det in result:
        print(f"  score={det['score']:.2f}  fallen={det['fallen']}  box={[round(float(v), 1) for v in det['box']]}")
    print(f"[done  ] wrote {args.output}")
