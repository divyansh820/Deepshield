"""
DeepShield — AI Fake Image Detection & Grad-CAM Forensics Engine
Backend Application Server (Flask + PyTorch + Advanced Vision Pipeline)
"""

import io
import os
import sys
import time
import json
import logging
from typing import Dict, Any, List

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None

# DeepShield Modular Components
from models import (
    model_manager, CLASS_AUTHENTIC, CLASS_AI, CLASS_NAMES,
    DENSENET_CKPT_PATH, RESNET_CKPT_PATH
)
from gradcam import GradCAM, generate_gradcam_payload
from forensics import (
    extract_metadata,
    compute_frequency_analysis,
    compute_ela,
    compute_sensor_noise_and_correlation,
    compute_vectorized_block_entropy
)
from ensemble import evaluate_ensemble

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("DeepShield.Server")

app = Flask(__name__)
CORS(app)

class VercelPathMiddleware:
    """
    Ensures WSGI PATH_INFO reflects the original client request path on Vercel,
    restoring it from X-Matched-Path or X-Forwarded-Uri if Vercel edge rewrite modified it to /api/index.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched_path = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_FORWARDED_URI')
        if matched_path:
            path = matched_path.split('?')[0]
            if path and path not in ('/api/index', '/api'):
                environ['PATH_INFO'] = path
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathMiddleware(app.wsgi_app)

# Configuration & Limits
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB max payload
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "TIFF", "BMP"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. STATIC & DOCUMENTATION ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    """Returns server operational status, active mode, and checkpoint readiness."""
    diagnostics_data = model_manager.get_diagnostics()
    return jsonify({
        "status": "online",
        "service": "DeepShield AI Image Forensics & Grad-CAM Engine",
        "version": "2.1.0",
        "device": diagnostics_data["device"],
        "active_mode": diagnostics_data["active_mode"],
        "is_trained": diagnostics_data["densenet_trained"] and diagnostics_data["resnet_trained"],
        "densenet_ready": diagnostics_data["densenet_ready"],
        "resnet_ready": diagnostics_data["resnet_ready"],
        "checkpoints": diagnostics_data["checkpoints"]
    })


@app.route('/api', methods=['GET'])
@app.route('/api/', methods=['GET'])
@app.route('/api/index', methods=['GET'])
def api_root():
    """Vercel function entrypoint status."""
    return health()


@app.route('/diagnostics', methods=['GET'])
@app.route('/api/diagnostics', methods=['GET'])
def diagnostics():
    """Returns deep system diagnostics, device info, memory, and checkpoint paths."""
    diag = model_manager.get_diagnostics()
    diag["torch_version"] = torch.__version__ if (TORCH_AVAILABLE and torch is not None) else "Not installed (Serverless Mode)"
    diag["python_version"] = sys.version.split()[0]
    return jsonify(diag)


@app.route('/benchmarks', methods=['GET'])
@app.route('/api/benchmarks', methods=['GET'])
def benchmarks():
    """Returns verified model evaluation metrics from held-out test data."""
    results_path = os.path.join(BASE_DIR, "checkpoints", "evaluation_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed reading benchmark data: {e}"}), 500
    return jsonify({
        "status": "pending_test_dataset",
        "message": "Evaluation test dataset has not been evaluated yet. Run 'python evaluate.py' with test data."
    })


@app.route('/models', methods=['GET'])
@app.route('/api/models', methods=['GET'])
def list_models():
    """Lists available model architectures and their forensic specialties."""
    diag = model_manager.get_diagnostics()
    return jsonify({
        "active_mode": diag["active_mode"],

        "available_models": [
            {
                "id": "ensemble",
                "name": "DeepShield Ensemble (Dual-Backbone + Forensics)",
                "description": "Fuses DenseNet-121, ResNet-50, and Spectral/ELA forensics for calibrated uncertainty estimation.",
                "target_layer": "Dual Backbone"
            },
            {
                "id": "densenet121",
                "name": "DenseNet-121",
                "description": "Exploits multi-layer dense feature reuse to identify micro-texture artifacts.",
                "target_layer": "features.norm5"
            },
            {
                "id": "resnet50",
                "name": "ResNet-50",
                "description": "Deep residual architecture specializing in multi-scale structural coherence.",
                "target_layer": "layer4[-1]"
            }
        ]
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMAGE FORENSIC & NEURAL PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def process_single_image(image_bytes: bytes, selected_model: str = "ensemble") -> Dict[str, Any]:
    """
    Executes staged progressive analysis:
    Stage 1: Validation and preprocessing
    Stage 2: Neural network inference
    Stage 3: Forensic vision pipeline
    Stage 4: Grad-CAM generation
    Stage 5: Ensemble synthesis and packaging
    """
    t_start = time.perf_counter()

    # Stage 1: Validate and decode
    t_stage1 = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_format = image.format or "UNKNOWN"
        image_size = image.size
        # Convert to RGB for all operations
        image_rgb = image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image format: {e}")

    # Preprocess tensor for deep vision
    tensor = model_manager.preprocess_image(image_rgb)
    dt_preprocess_ms = (time.perf_counter() - t_stage1) * 1000.0

    # Stage 2: Deep Learning Inference
    t_stage2 = time.perf_counter()
    dense_real, dense_ai = model_manager.predict_densenet(tensor)
    res_real, res_ai = model_manager.predict_resnet(tensor)
    dt_inference_ms = (time.perf_counter() - t_stage2) * 1000.0

    # Stage 3: Forensic Vision Algorithms
    t_stage3 = time.perf_counter()
    metadata = extract_metadata(image)
    freq_data = compute_frequency_analysis(image_rgb)
    ela_data = compute_ela(image_rgb)
    sensor_data = compute_sensor_noise_and_correlation(image_rgb)
    texture_data = compute_vectorized_block_entropy(image_rgb)
    dt_forensics_ms = (time.perf_counter() - t_stage3) * 1000.0

    forensic_payload = {
        "metadata": metadata,
        "frequency": freq_data,
        "ela": ela_data,
        "sensor": sensor_data,
        "texture": texture_data
    }

    # Stage 4: Ensemble Evaluation & Decision Policy
    t_stage4 = time.perf_counter()
    is_demo = (model_manager.active_mode == "demo_forensics")
    ensemble_result = evaluate_ensemble(
        densenet_probs=(dense_real, dense_ai),
        resnet_probs=(res_real, res_ai),
        forensic_data=forensic_payload,
        is_demo_mode=is_demo
    )

    # Stage 5: Grad-CAM Visual Heatmap
    t_stage5 = time.perf_counter()
    model_choice = selected_model.lower().strip()
    target_class = CLASS_AI if ensemble_result["verdict_code"] == "ai" else CLASS_AUTHENTIC

    gradcam_engine = None
    target_layer_name = "features.norm5"

    try:
        if tensor is None or model_manager.densenet is None:
            from gradcam import compute_forensic_gradient_map
            cam_map = compute_forensic_gradient_map(image_rgb)
            target_layer_name = "Forensic Gradient Activation Map"
        elif "resnet" in model_choice:
            target_layer_name = "ResNet-50 layer4 (Bottleneck Conv)"
            gradcam_engine = GradCAM(model_manager.resnet, model_manager.resnet.target_layer)
            cam_map = gradcam_engine.generate(tensor, target_class=target_class)
            gradcam_engine.remove_hooks()
        elif "dense" in model_choice:
            target_layer_name = "DenseNet-121 features.norm5 (Dense Feature Reuse)"
            gradcam_engine = GradCAM(model_manager.densenet, model_manager.densenet.target_layer)
            cam_map = gradcam_engine.generate(tensor, target_class=target_class)
            gradcam_engine.remove_hooks()
        else: # Ensemble
            target_layer_name = "DeepShield Dual-Backbone Composite (DenseNet + ResNet)"
            cam_d = GradCAM(model_manager.densenet, model_manager.densenet.target_layer)
            cam_map_d = cam_d.generate(tensor, target_class=target_class)
            cam_d.remove_hooks()

            cam_r = GradCAM(model_manager.resnet, model_manager.resnet.target_layer)
            cam_map_r = cam_r.generate(tensor, target_class=target_class)
            cam_r.remove_hooks()

            cam_map = 0.50 * cam_map_d + 0.50 * cam_map_r

        gradcam_visuals = generate_gradcam_payload(
            original_img=image_rgb,
            cam_map=cam_map,
            target_class=target_class,
            target_layer_name=target_layer_name
        )
    except Exception as e:
        logger.error(f"Grad-CAM generation error: {e}")
        gradcam_visuals = {
            "overlay_base64": "",
            "heatmap_base64": "",
            "hot_regions": [],
            "target_layer": target_layer_name,
            "error": str(e)
        }

    dt_gradcam_ms = (time.perf_counter() - t_stage5) * 1000.0
    dt_total_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "result": ensemble_result["verdict"],
        "verdict_code": ensemble_result["verdict_code"],
        "confidence": ensemble_result["confidence"],
        "ai_probability": ensemble_result["ai_probability"],
        "authentic_probability": ensemble_result["authentic_probability"],
        "is_demo_mode": is_demo,
        "model_used": (
            "DenseNet-121 Binary Classifier" if "dense" in model_choice
            else "ResNet-50 Binary Classifier" if "resnet" in model_choice
            else "DeepShield Hybrid Ensemble"
        ),
        "model_agreement": {
            "agree": ensemble_result["models_agree"],
            "disagreement_delta": ensemble_result["disagreement_delta"],
            "disagreement_warning": ensemble_result["disagreement_warning"],
            "explanation": ensemble_result["explanation"]
        },
        "model_scores": ensemble_result["model_scores"],
        "gradcam": gradcam_visuals,
        "forensics": {
            "ela_mean": ela_data["mean_residual"],
            "ela_std": ela_data["std_residual"],
            "edge_flat_ratio": ela_data["edge_flat_ratio"],
            "dct_hf_ratio": freq_data["hf_ratio"],
            "spectral_slope": freq_data["spectral_slope"],
            "grid_spike_score": freq_data["grid_spike_score"],
            "channel_corr": sensor_data["channel_corr"],
            "noise_var": sensor_data["noise_var"],
            "flat_block_ratio": texture_data["flat_block_ratio"],
            "has_exif": metadata["has_exif"],
            "camera_model": metadata["camera_model"],
            "software": metadata["software"],
            "ai_software_detected": metadata["ai_software_detected"],
            "status_description": metadata["status_description"]
        },
        "image_info": {
            "width": image_size[0],
            "height": image_size[1],
            "format": image_format,
            "mode": image.mode
        },
        "timing_ms": {
            "preprocess": round(dt_preprocess_ms, 1),
            "inference": round(dt_inference_ms, 1),
            "forensics": round(dt_forensics_ms, 1),
            "gradcam": round(dt_gradcam_ms, 1),
            "total": round(dt_total_ms, 1)
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANALYSIS API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/analyze', methods=['POST', 'OPTIONS'])
@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Handles single image upload and executes the deep forensic pipeline."""
    if request.method == 'OPTIONS':
        return '', 200

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in multipart/form-data"}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({"error": "Empty filename provided"}), 400

    selected_model = request.form.get('model', 'ensemble')

    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Uploaded file is 0 bytes"}), 400
        if len(image_bytes) > MAX_CONTENT_LENGTH:
            return jsonify({"error": f"Image exceeds maximum allowable size of {MAX_CONTENT_LENGTH // (1024*1024)}MB"}), 413

        response_data = process_single_image(image_bytes, selected_model=selected_model)
        return jsonify(response_data)

    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except Exception as e:
        logger.exception(f"Unhandled exception during image analysis: {e}")
        return jsonify({"error": f"Analysis pipeline failed: {str(e)}"}), 500


@app.route('/analyze_batch', methods=['POST', 'OPTIONS'])
@app.route('/api/analyze_batch', methods=['POST', 'OPTIONS'])
def analyze_batch():
    """
    Handles batch upload of multiple images.
    Returns individual forensic evaluations and summary statistics.
    """
    if request.method == 'OPTIONS':
        return '', 200

    files = request.files.getlist('images')
    if not files or len(files) == 0:
        return jsonify({"error": "No images provided in 'images' field"}), 400

    selected_model = request.form.get('model', 'ensemble')
    max_batch = 10
    if len(files) > max_batch:
        return jsonify({"error": f"Batch limit exceeded. Maximum {max_batch} images per request."}), 400

    results = []
    summary = {"total": len(files), "authentic": 0, "ai": 0, "uncertain": 0, "failed": 0}

    for file in files:
        fname = file.filename
        try:
            b = file.read()
            if len(b) == 0 or len(b) > MAX_CONTENT_LENGTH:
                results.append({"filename": fname, "status": "error", "error": "Invalid file size"})
                summary["failed"] += 1
                continue

            item_res = process_single_image(b, selected_model=selected_model)
            # Remove heavy base64 strings from batch overview to conserve bandwidth
            item_res.pop("gradcam", None)
            item_res["filename"] = fname
            item_res["status"] = "success"
            results.append(item_res)

            code = item_res["verdict_code"]
            if code == "real":
                summary["authentic"] += 1
            elif code == "ai":
                summary["ai"] += 1
            else:
                summary["uncertain"] += 1

        except Exception as e:
            results.append({"filename": fname, "status": "error", "error": str(e)})
            summary["failed"] += 1

    return jsonify({
        "summary": summary,
        "results": results
    })


@app.route('/<path:filename>', methods=['GET'])
def serve_static(filename):
    """Fallback static asset server for local development."""
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(BASE_DIR, filename)
    return jsonify({"error": "Resource not found"}), 404



# ─────────────────────────────────────────────────────────────────────────────
# 4. APPLICATION LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    diag = model_manager.get_diagnostics()
    print("=" * 65)
    print("  🛡️  DEEPSHIELD AI FORENSIC & GRAD-CAM EXPLAINABILITY SERVER")
    print(f"  Operating Mode : {diag['active_mode'].upper()}")
    print(f"  Device         : {diag['device']}")
    print(f"  Trained Models : {'ACTIVE (Binary Classifiers)' if diag['densenet_trained'] else 'DEMO/HEURISTICS (Weights Pending)'}")
    print("  Listening on   : http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host='0.0.0.0', port=5000, debug=False)
