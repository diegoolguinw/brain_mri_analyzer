"""
Inference engine — ONNX Runtime version for low-memory deployment.

Loads an ONNX model + metadata once and provides analysis functions.
No PyTorch dependency at runtime — saves ~250 MB of RAM.
"""

import io
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.special import expit as sigmoid  # numpy sigmoid

# ---------------------------------------------------------------------------
# Singleton model holder (loaded once on first request)
# ---------------------------------------------------------------------------

_SESSION: Optional[ort.InferenceSession] = None
_THRESHOLD = 0.5
_IMG_SIZE = 128
_MODEL_NAME = ""


def load_model(onnx_path: Optional[str] = None):
    """Load the ONNX model.  Called lazily on first request."""
    global _SESSION, _THRESHOLD, _IMG_SIZE, _MODEL_NAME

    if onnx_path is None:
        onnx_path = os.environ.get("ONNX_MODEL_PATH")

    if onnx_path is None:
        # Fallback: look relative to this file
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(base, "checkpoints", "model.onnx")
        if not os.path.exists(candidate):
            candidate = os.path.join(base, "..", "checkpoints", "model.onnx")
        onnx_path = os.path.abspath(candidate)

    # Load companion metadata
    meta_path = onnx_path.replace(".onnx", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        _IMG_SIZE = meta.get("img_size", 128)
        _THRESHOLD = meta.get("threshold", 0.5)
        model_class = meta.get("model_class", "SmallUNet")
        base_ch = meta.get("base_ch", 48)
        _MODEL_NAME = f"{model_class} (base_ch={base_ch}) [ONNX]"
    else:
        _MODEL_NAME = "UNet [ONNX]"

    # Create ONNX Runtime session with minimal memory settings
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Minimize memory arena
    sess_options.enable_cpu_mem_arena = False

    _SESSION = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    print(f"[inference] Loaded {_MODEL_NAME} (threshold={_THRESHOLD:.2f}, img_size={_IMG_SIZE})")


def _ensure_loaded():
    if _SESSION is None:
        load_model()


# ---------------------------------------------------------------------------
# Inference (single pass — no TTA to save memory)
# ---------------------------------------------------------------------------


def _predict_prob(image_hw: np.ndarray) -> np.ndarray:
    """Run a single forward pass and return probability map [H, W]."""
    _ensure_loaded()
    # Shape: [1, 1, H, W] float32
    x = image_hw[np.newaxis, np.newaxis, :, :].astype(np.float32)
    input_name = _SESSION.get_inputs()[0].name
    logits = _SESSION.run(None, {input_name: x})[0]  # [1, 1, H, W]
    prob = sigmoid(logits[0, 0])  # numpy sigmoid — no torch needed
    return np.ascontiguousarray(prob.astype(np.float32))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Everything needed to display results and build a report."""
    tumor_detected: bool = False
    tumor_pixel_fraction: float = 0.0
    tumor_area_mm2: float = 0.0
    centroid_x_frac: float = 0.0
    centroid_y_frac: float = 0.0
    quadrant: str = ""
    max_prob: float = 0.0
    mean_prob_in_mask: float = 0.0
    threshold: float = 0.5
    model_name: str = ""
    img_size: int = 128
    original_size: tuple = (0, 0)
    image_gray: Optional[np.ndarray] = None
    prob_map: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None
    overlay_png_bytes: bytes = b""


def _quadrant_label(cx: float, cy: float) -> str:
    v = "superior" if cy < 0.5 else "inferior"
    h = "left" if cx < 0.5 else "right"
    return f"{v}-{h}"


def _build_overlay_png(image_gray: np.ndarray, mask: np.ndarray) -> bytes:
    """Create an RGB overlay PNG as bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_gray = np.ascontiguousarray(image_gray, dtype=np.float64)
    mask = np.ascontiguousarray(mask)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    ax.imshow(image_gray, cmap="gray")
    if mask.any():
        mask_float = np.ascontiguousarray(mask.astype(np.float64))
        mask_uint8 = np.ascontiguousarray(mask.astype(np.uint8))
        ax.imshow(np.ma.masked_where(~mask, mask_float), cmap="autumn", alpha=0.55)
        ax.contour(mask_uint8, levels=[0.5], colors="lime", linewidths=1.2)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def analyze_image(image_file) -> AnalysisResult:
    """
    Analyse an uploaded MRI image file.
    Returns an AnalysisResult with tumour detection info + visualisation bytes.
    """
    _ensure_loaded()

    if isinstance(image_file, (str, os.PathLike)):
        pil_img = Image.open(image_file)
    else:
        pil_img = Image.open(image_file)

    original_size = pil_img.size
    gray = pil_img.convert("L").resize((_IMG_SIZE, _IMG_SIZE), Image.BILINEAR)
    img_np = np.array(gray, dtype=np.float32) / 255.0

    # Single forward pass (no TTA — saves memory)
    prob_map = _predict_prob(img_np)
    binary_mask = prob_map > _THRESHOLD

    tumor_pixels = int(binary_mask.sum())
    total_pixels = binary_mask.size
    tumor_frac = tumor_pixels / total_pixels
    tumor_detected = tumor_frac > 0.001

    cx_frac, cy_frac = 0.5, 0.5
    if tumor_detected:
        ys, xs = np.where(binary_mask)
        cy_frac = float(ys.mean()) / _IMG_SIZE
        cx_frac = float(xs.mean()) / _IMG_SIZE

    fov_mm = 240.0
    pixel_mm = fov_mm / _IMG_SIZE
    tumor_area_mm2 = tumor_pixels * pixel_mm * pixel_mm

    overlay_bytes = _build_overlay_png(img_np, binary_mask)

    return AnalysisResult(
        tumor_detected=tumor_detected,
        tumor_pixel_fraction=tumor_frac,
        tumor_area_mm2=tumor_area_mm2,
        centroid_x_frac=cx_frac,
        centroid_y_frac=cy_frac,
        quadrant=_quadrant_label(cx_frac, cy_frac) if tumor_detected else "N/A",
        max_prob=float(prob_map.max()),
        mean_prob_in_mask=float(prob_map[binary_mask].mean()) if tumor_detected else 0.0,
        threshold=_THRESHOLD,
        model_name=_MODEL_NAME,
        img_size=_IMG_SIZE,
        original_size=original_size,
        image_gray=img_np,
        prob_map=prob_map,
        binary_mask=binary_mask,
        overlay_png_bytes=overlay_bytes,
    )
