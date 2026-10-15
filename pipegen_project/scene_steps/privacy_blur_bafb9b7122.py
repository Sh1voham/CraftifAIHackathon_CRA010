import cv2
import numpy as np

# Reference for the privacy steps (catalogue §6 Privacy), written to be
# transcribed into a per-frame C++ worker. It runs after the model has used the
# frame and before anything shows, stores or sends it: blur first and the model
# has nothing left to find.
#
# The frame is NV12 and is changed IN PLACE (y: H x W, uv: H/2 x W/2 x 2, see
# frame_conditioning.py). boxes are this frame's objects in full-frame pixels,
# as DeepStream's NvDsObjectMeta rect_params give them:
#   [{"left": x, "top": y, "width": w, "height": h, "label": "person", "source": "<task>"?}, ...]
# params = {"ops": [{"op": "pixelate_boxes", "region": "face" | "head" | "person" | "plate",
#                    "classes": ["person"], "source": "<task key>", "mode": "pixelate" | "blur", "cell": 16}]}
# "classes" empty: every box of the source task. "head" is the top 20% of a
# person box, grown by 20% on each axis, for pipelines that find people but not faces.
# Stateless: nothing is kept between frames.

HEAD_SHARE = 0.20
HEAD_GROW = 0.20
OPS = ("pixelate_boxes",)


def _wanted(box, op):
    classes = [str(c).lower() for c in op.get("classes") or []]
    label = str(box.get("label") or "").lower()
    if classes and not any(c == label or c in label for c in classes):
        return False
    return not (op.get("source") and box.get("source") and box["source"] != op["source"])


def region(box, kind, width, height):
    x, y, w, h = (float(box[k]) for k in ("left", "top", "width", "height"))
    if kind == "head":
        h = h * HEAD_SHARE
        x, y, w, h = x - w * HEAD_GROW / 2, y - h * HEAD_GROW / 2, w * (1 + HEAD_GROW), h * (1 + HEAD_GROW)
    # Even edges, so the chroma plane covers exactly the same pixels.
    x0, y0 = max(0, int(np.floor(x / 2)) * 2), max(0, int(np.floor(y / 2)) * 2)
    x1, y1 = min(width, int(np.ceil((x + w) / 2)) * 2), min(height, int(np.ceil((y + h) / 2)) * 2)
    return (x0, y0, x1, y1) if x1 - x0 >= 2 and y1 - y0 >= 2 else None


def _hide(plane, x0, y0, x1, y1, mode, cell):
    block = plane[y0:y1, x0:x1]
    h, w = block.shape[:2]
    if mode == "blur":
        k = max(3, (min(w, h) // 3) | 1)
        block[...] = cv2.blur(np.ascontiguousarray(block), (k, k))
        return
    small = cv2.resize(np.ascontiguousarray(block), (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_AREA)
    block[...] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST).reshape(block.shape)


def process(y, uv, boxes, params):
    # Returns how many regions were hidden on this frame.
    height, width = y.shape[:2]
    hidden = 0
    for op in params.get("ops") or []:
        if op.get("op") not in OPS:
            raise ValueError(f"unknown op {op.get('op')!r}")
        kind, mode, cell = op.get("region") or "person", op.get("mode") or "pixelate", max(2, int(op.get("cell") or 16))
        for box in boxes or []:
            if not _wanted(box, op):
                continue
            area = region(box, kind, width, height)
            if area is None:
                continue
            x0, y0, x1, y1 = area
            _hide(y, x0, y0, x1, y1, mode, cell)
            _hide(uv, x0 // 2, y0 // 2, x1 // 2, y1 // 2, mode, max(1, cell // 2))
            hidden += 1
    return hidden
