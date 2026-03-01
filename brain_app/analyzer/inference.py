"""
Inference engine — loads a checkpoint once and provides analysis functions.
"""

import io
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .nn_models import SmallUNet, AttentionUNet

# ---------------------------------------------------------------------------
# Singleton model holder (loaded once on first request)
# ---------------------------------------------------------------------------

_MODEL = None
_DEVICE = None
_THRESHOLD = 0.5
_IMG_SIZE = 128
_MODEL_NAME = ""


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: Optional[str] = None):
    """Load the model from a .pt checkpoint.  Called lazily on first request."""
    global _MODEL, _DEVICE, _THRESHOLD, _IMG_SIZE, _MODEL_NAME

    if checkpoint_path is None:
        # Use CHECKPOINT_PATH env var, or Django setting, or find relative to this file
        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
        if checkpoint_path is None:
            try:
                from django.conf import settings as django_settings
                checkpoint_path = getattr(django_settings, "CHECKPOINT_PATH", None)
            except Exception:
                pass
        if checkpoint_path is None:
            # Fallback: walk up from this file to find checkpoints/
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(base, "checkpoints", "unet_resse_best.pt")
            if not os.path.exists(candidate):
                # Docker layout: checkpoints/ is a sibling of the app code
                candidate = os.path.join(base, "..", "checkpoints", "unet_resse_best.pt")
            checkpoint_path = os.path.abspath(candidate)

    _DEVICE = _get_device()
    ckpt = torch.load(checkpoint_path, map_location=_DEVICE, weights_only=False)

    model_class = ckpt.get("model_class", "SmallUNet")
    base_ch = ckpt.get("base_ch", 48)
    _IMG_SIZE = ckpt.get("img_size", 128)
    _THRESHOLD = ckpt.get("threshold", 0.5)

    if model_class == "AttentionUNet":
        model = AttentionUNet(base_ch=base_ch, deep_supervision=False)
        _MODEL_NAME = f"Attention U-Net (base_ch={base_ch})"
    else:
        model = SmallUNet(base_ch=base_ch)
        _MODEL_NAME = f"Residual SE U-Net (base_ch={base_ch})"

    # Prefer EMA weights if available
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    model.to(_DEVICE).eval()
    _MODEL = model
    print(f"[inference] Loaded {_MODEL_NAME} on {_DEVICE} (threshold={_THRESHOLD:.2f}, img_size={_IMG_SIZE})")


def _ensure_loaded():
    if _MODEL is None:
        load_model()


# ---------------------------------------------------------------------------
# TTA inference
# ---------------------------------------------------------------------------


def _predict_prob_tta(image_chw: torch.Tensor) -> np.ndarray:
    """Run test-time augmentation (4 flips) and return probability map [H, W]."""
    _ensure_loaded()
    x = image_chw.unsqueeze(0).to(_DEVICE)
    tta_dims = [None, (-1,), (-2,), (-1, -2)]
    probs = []
    with torch.no_grad():
        for dims in tta_dims:
            x_aug = torch.flip(x, dims=dims) if dims else x
            logits = _MODEL(x_aug)
            prob = torch.sigmoid(logits)
            if dims:
                prob = torch.flip(prob, dims=dims)
            probs.append(prob)
    result = torch.stack(probs).mean(0)[0, 0].cpu().numpy()
    return np.ascontiguousarray(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Everything needed to display results and build a report."""
    tumor_detected: bool = False
    tumor_pixel_fraction: float = 0.0
    tumor_area_mm2: float = 0.0  # approximate
    centroid_x_frac: float = 0.0  # 0-1 fraction in image width
    centroid_y_frac: float = 0.0  # 0-1 fraction in image height
    quadrant: str = ""
    max_prob: float = 0.0
    mean_prob_in_mask: float = 0.0
    threshold: float = 0.5
    model_name: str = ""
    img_size: int = 128
    original_size: tuple = (0, 0)
    # Raw arrays for visualisation (not serialised)
    image_gray: Optional[np.ndarray] = None
    prob_map: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None
    overlay_png_bytes: bytes = b""


def _quadrant_label(cx: float, cy: float) -> str:
    """Rough anatomical quadrant from centroid fractions."""
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
    Analyse an uploaded MRI image file (file-like or path).

    Returns an AnalysisResult with tumour detection info + visualisation bytes.
    """
    _ensure_loaded()

    # Load & preprocess
    if isinstance(image_file, (str, os.PathLike)):
        pil_img = Image.open(image_file)
    else:
        pil_img = Image.open(image_file)

    original_size = pil_img.size  # (W, H)
    gray = pil_img.convert("L").resize((_IMG_SIZE, _IMG_SIZE), Image.BILINEAR)
    img_np = np.array(gray, dtype=np.float32) / 255.0
    img_t = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

    # Predict
    prob_map = _predict_prob_tta(img_t)
    binary_mask = prob_map > _THRESHOLD

    # Compute statistics
    tumor_pixels = int(binary_mask.sum())
    total_pixels = binary_mask.size
    tumor_frac = tumor_pixels / total_pixels
    tumor_detected = tumor_frac > 0.001  # at least 0.1 % of pixels

    cx_frac, cy_frac = 0.5, 0.5
    if tumor_detected:
        ys, xs = np.where(binary_mask)
        cy_frac = float(ys.mean()) / _IMG_SIZE
        cx_frac = float(xs.mean()) / _IMG_SIZE

    # Very rough area estimate: assume a 240 mm FOV for a brain MRI
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
