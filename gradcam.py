"""
DeepShield — Grad-CAM Explainability Engine
Gradient-weighted Class Activation Mapping for DenseNet-121 and ResNet-50.
Uses forward hook + tensor retain_grad for 100% stable gradient backpropagation.
Includes fast vectorized colormap generation, alpha blending, and transparent visual reporting.
"""

import io
import base64
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None

from models import CLASS_AUTHENTIC, CLASS_AI


# ─────────────────────────────────────────────────────────────────────────────
# 1. FAST JET COLORMAP LOOKUP TABLE (Avoids slow matplotlib figure overhead)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_jet_lut() -> np.ndarray:
    """Generates a 256-entry RGB lookup table for the Jet colormap."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return lut

JET_LUT = _generate_jet_lut()


# ─────────────────────────────────────────────────────────────────────────────
# 2. GRAD-CAM GENERATOR CLASS (Using Tensor retain_grad for rock-solid stability)
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping on vision models.
    Hooks into the final convolutional layer to extract feature activations and gradients.
    """
    def __init__(self, model: Any, target_layer: Any):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[Any] = None
        self.hook_handle: Optional[Any] = None
        if TORCH_AVAILABLE and target_layer is not None:
            self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input_t, output_t):
            self.activations = output_t
            if hasattr(self.activations, 'requires_grad') and self.activations.requires_grad:
                self.activations.retain_grad()

        if hasattr(self.target_layer, 'register_forward_hook'):
            self.hook_handle = self.target_layer.register_forward_hook(forward_hook)

    def generate(self, input_tensor: Any, target_class: Optional[int] = None) -> np.ndarray:
        """
        Executes forward + backward pass to generate normalized 2D Grad-CAM heatmap.
        """
        if not TORCH_AVAILABLE or self.model is None or input_tensor is None:
            return np.zeros((7, 7), dtype=np.float32)

        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        else:
            target_class = max(0, min(output.shape[1] - 1, target_class))

        # Backward for target class score
        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.activations is None or not hasattr(self.activations, 'grad') or self.activations.grad is None:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h // 32, w // 32), dtype=np.float32)

        grads = self.activations.grad.detach()
        activations = self.activations.detach()

        # Global average pooling on gradients across spatial dimensions (H, W)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # Apply ReLU to retain features that positively correlate with the target class
        cam = F.relu(cam)
        cam_np = cam.squeeze().cpu().numpy()

        # Robust min-max normalization
        cam_min, cam_max = float(np.min(cam_np)), float(np.max(cam_np))
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm.astype(np.float32)

    def remove_hooks(self):
        """Cleanly unregisters hooks to prevent memory leaks."""
        if self.hook_handle is not None:
            try:
                self.hook_handle.remove()
            except Exception:
                pass
            self.hook_handle = None


def compute_forensic_gradient_map(image: Image.Image) -> np.ndarray:
    """
    Computes spatial optical gradient energy when PyTorch is not available.
    """
    gray = image.convert('L').resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(gray, dtype=np.float32) / 255.0
    gx = np.abs(arr[:, 2:] - arr[:, :-2])
    gy = np.abs(arr[2:, :] - arr[:-2, :])
    grad = np.zeros_like(arr)
    grad[1:-1, 1:-1] = np.sqrt(gx[1:-1, :]**2 + gy[:, 1:-1]**2)
    min_v, max_v = float(np.min(grad)), float(np.max(grad))
    if max_v - min_v > 1e-8:
        grad = (grad - min_v) / (max_v - min_v)
    return grad.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAD-CAM VISUALIZATION & EXPLAINABILITY PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────
def generate_gradcam_payload(
    original_img: Image.Image,
    cam_map: np.ndarray,
    target_class: int,
    target_layer_name: str,
    alpha: float = 0.55
) -> Dict[str, Any]:
    """
    Renders pure heatmap, blended overlay, and detected hotspot regions.
    """
    orig_w, orig_h = original_img.size

    # 1. Resize cam_map to match original image dimensions
    cam_uint8 = np.clip(cam_map * 255.0, 0, 255).astype(np.uint8)
    cam_pil = Image.fromarray(cam_uint8, mode='L')
    cam_resized = cam_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    cam_full = np.array(cam_resized, dtype=np.float32) / 255.0

    # 2. Apply fast Jet colormap lookup table
    cam_indices = np.clip(cam_full * 255.0, 0, 255).astype(np.uint8)
    colored_cam = JET_LUT[cam_indices]  # Shape (H, W, 3)

    # 3. Create pure heatmap PIL image
    pure_heatmap = Image.fromarray(colored_cam)

    # 4. Create blended overlay image
    orig_rgb = original_img.convert('RGB')
    orig_arr = np.array(orig_rgb, dtype=np.float32)
    blended_arr = (1.0 - alpha) * orig_arr + alpha * colored_cam.astype(np.float32)
    blended_arr = np.clip(blended_arr, 0, 255).astype(np.uint8)
    blended_overlay = Image.fromarray(blended_arr)

    # Helper: Convert PIL Image to Base64 data URL
    def to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    # 5. Extract hotspot regions for explainability
    hot_mask = cam_full > 0.60
    hot_regions = []

    if np.any(hot_mask):
        y_indices, x_indices = np.where(hot_mask)
        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
        peak_intensity = float(np.max(cam_full))

        center_x_norm = (min_x + max_x) / (2.0 * orig_w)
        center_y_norm = (min_y + max_y) / (2.0 * orig_h)
        region_desc = "Central Subject Focus" if (0.25 <= center_x_norm <= 0.75 and 0.25 <= center_y_norm <= 0.75) else "Peripheral Context / Edge Boundary"

        if target_class == CLASS_AI:
            hot_regions.append({
                "region": region_desc,
                "box": [min_x, min_y, max_x, max_y],
                "peak_activation": f"{int(peak_intensity * 100)}%",
                "feature_type": "Synthetic Attention Peak",
                "explanation": (
                    f"Neural network attention is elevated in the {region_desc.lower()}. "
                    "Features in this region drove the synthetic class score (e.g. generative micro-textures or synthesis blending boundaries)."
                )
            })
        else:
            hot_regions.append({
                "region": region_desc,
                "box": [min_x, min_y, max_x, max_y],
                "peak_activation": f"{int(peak_intensity * 100)}%",
                "feature_type": "Authentic Feature Alignment",
                "explanation": (
                    f"Neural network attention aligns with natural scene contours in the {region_desc.lower()}. "
                    "Consistent physical optical depth and gradient responses supported the authentic class score."
                )
            })
    else:
        hot_regions.append({
            "region": "Uniform Gradient Distribution",
            "box": [0, 0, orig_w, orig_h],
            "peak_activation": "< 50%",
            "feature_type": "Diffuse Activation",
            "explanation": "No localized extreme activation peaks detected. Activation energy is distributed broadly across the image frame."
        })

    return {
        "overlay_base64": to_b64(blended_overlay),
        "heatmap_base64": to_b64(pure_heatmap),
        "hot_regions": hot_regions,
        "target_layer": target_layer_name,
        "target_class": "AI-Generated" if target_class == CLASS_AI else "Authentic",
        "disclaimer": "Grad-CAM visualizes regions that influenced the model's prediction score. It does not independently verify whether an image is synthetic or authentic."
    }
