"""
DeepShield — ML Models & Neural Network Architecture
Handles DenseNet-121 and ResNet-50 binary classifiers (Authentic vs. AI-Generated).
Supports both local PyTorch GPU/CPU inference and lightweight serverless fallback (Vercel).
"""

import os
import sys
import logging
from typing import Dict, Tuple, Optional, Any
from PIL import Image

logger = logging.getLogger("DeepShield.Models")

# ─────────────────────────────────────────────────────────────────────────────
# 1. OPTIONAL PYTORCH IMPORT (Enables serverless deployment under 250MB)
# ─────────────────────────────────────────────────────────────────────────────
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError as e:
    DEVICE = "cpu"
    torch = None
    nn = None
    F = None
    models = None
    transforms = None
    logger.info(f"PyTorch not available in current environment ({e}). Running in serverless forensic mode.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTANTS & CLASS INDEX DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
CLASS_AUTHENTIC = 0
CLASS_AI = 1
CLASS_NAMES = ["Authentic", "AI-Generated"]

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
DENSENET_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "densenet121_deepshield.pth")
RESNET_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "resnet50_deepshield.pth")

# Standard Vision normalization for 224x224 input
if TORCH_AVAILABLE:
    INFERENCE_TRANSFORM = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
else:
    INFERENCE_TRANSFORM = None


# ─────────────────────────────────────────────────────────────────────────────
# 3. BINARY CLASSIFIER ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class DenseNet121Binary(nn.Module):
        """
        DenseNet-121 adapted for binary image forensics (Authentic vs AI-Generated).
        Replaces the 1000-class head with a 2-class dropout+linear head.
        Target Grad-CAM layer: features.norm5
        """
        def __init__(self, pretrained_backbone: bool = True):
            super().__init__()
            weights = models.DenseNet121_Weights.DEFAULT if pretrained_backbone else None
            base = models.densenet121(weights=weights)
            self.features = base.features
            in_features = base.classifier.in_features
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.25),
                nn.Linear(in_features, 2)
            )
            self.target_layer = self.features.norm5

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.features(x)
            out = F.relu(features, inplace=False)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            return out


    class ResNet50Binary(nn.Module):
        """
        ResNet-50 adapted for binary image forensics (Authentic vs AI-Generated).
        Replaces the 1000-class head with a 2-class dropout+linear head.
        Target Grad-CAM layer: layer4[-1]
        """
        def __init__(self, pretrained_backbone: bool = True):
            super().__init__()
            weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
            base = models.resnet50(weights=weights)
            self.conv1 = base.conv1
            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.avgpool = base.avgpool
            in_features = base.fc.in_features
            self.fc = nn.Sequential(
                nn.Dropout(p=0.25),
                nn.Linear(in_features, 2)
            )
            self.target_layer = self.layer4[-1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
else:
    class DenseNet121Binary:
        pass
    class ResNet50Binary:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL MANAGER WITH CHECKPOINT VERIFICATION & DEMO FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class DeepShieldModelManager:
    """
    Manages model initialization, checkpoint loading, device placement,
    and diagnostic status tracking.
    """
    def __init__(self):
        self.device = DEVICE
        self.transform = INFERENCE_TRANSFORM
        self.densenet: Optional[Any] = None
        self.resnet: Optional[Any] = None
        
        self.densenet_trained: bool = False
        self.resnet_trained: bool = False
        self.checkpoint_status: Dict[str, Any] = {}
        self.active_mode: str = "demo_forensics"  # 'production_binary' or 'demo_forensics'
        
        if TORCH_AVAILABLE:
            self.initialize_models()
        else:
            self.checkpoint_status["densenet121"] = {
                "loaded": False,
                "note": "PyTorch not available in this serverless environment. Running in lightweight forensic mode."
            }
            self.checkpoint_status["resnet50"] = {
                "loaded": False,
                "note": "PyTorch not available in this serverless environment. Running in lightweight forensic mode."
            }

    def initialize_models(self):
        """Initializes architectures and attempts to load fine-tuned binary weights."""
        if not TORCH_AVAILABLE:
            return

        logger.info(f"Initializing DeepShield models on device: {self.device}")
        
        # 1. Initialize DenseNet-121
        try:
            self.densenet = DenseNet121Binary(pretrained_backbone=True)
            self.densenet.to(self.device)
            self.densenet.eval()
            
            if os.path.exists(DENSENET_CKPT_PATH):
                ckpt = torch.load(DENSENET_CKPT_PATH, map_location=self.device)
                state_dict = ckpt.get("state_dict", ckpt)
                self.densenet.load_state_dict(state_dict, strict=False)
                self.densenet_trained = True
                self.checkpoint_status["densenet121"] = {
                    "loaded": True,
                    "path": DENSENET_CKPT_PATH,
                    "type": "Trained Binary Classifier",
                    "version": ckpt.get("version", "2.1.0") if isinstance(ckpt, dict) else "custom"
                }
                logger.info(f"Loaded trained DenseNet-121 checkpoint from {DENSENET_CKPT_PATH}")
            else:
                self.densenet_trained = False
                self.checkpoint_status["densenet121"] = {
                    "loaded": False,
                    "path": DENSENET_CKPT_PATH,
                    "type": "Untrained Binary Head (ImageNet Backbone)",
                    "note": "Fine-tuned binary weights missing. Place checkpoint at checkpoints/densenet121_deepshield.pth"
                }
        except Exception as e:
            logger.error(f"Error initializing DenseNet-121: {e}")
            self.densenet = None

        # 2. Initialize ResNet-50
        try:
            self.resnet = ResNet50Binary(pretrained_backbone=True)
            self.resnet.to(self.device)
            self.resnet.eval()
            
            if os.path.exists(RESNET_CKPT_PATH):
                ckpt = torch.load(RESNET_CKPT_PATH, map_location=self.device)
                state_dict = ckpt.get("state_dict", ckpt)
                self.resnet.load_state_dict(state_dict, strict=False)
                self.resnet_trained = True
                self.checkpoint_status["resnet50"] = {
                    "loaded": True,
                    "path": RESNET_CKPT_PATH,
                    "type": "Trained Binary Classifier",
                    "version": ckpt.get("version", "2.1.0") if isinstance(ckpt, dict) else "custom"
                }
                logger.info(f"Loaded trained ResNet-50 checkpoint from {RESNET_CKPT_PATH}")
            else:
                self.resnet_trained = False
                self.checkpoint_status["resnet50"] = {
                    "loaded": False,
                    "path": RESNET_CKPT_PATH,
                    "type": "Untrained Binary Head (ImageNet Backbone)",
                    "note": "Fine-tuned binary weights missing. Place checkpoint at checkpoints/resnet50_deepshield.pth"
                }
        except Exception as e:
            logger.error(f"Error initializing ResNet-50: {e}")
            self.resnet = None

        # Determine overall mode
        if self.densenet_trained and self.resnet_trained:
            self.active_mode = "production_binary"
        else:
            self.active_mode = "demo_forensics"

    def preprocess_image(self, image: Image.Image) -> Optional[Any]:
        """Converts PIL image to standard normalized tensor on active device if PyTorch is available."""
        if not TORCH_AVAILABLE or self.transform is None:
            return None
        rgb_img = image.convert("RGB")
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
        return tensor

    def predict_densenet(self, tensor: Optional[Any]) -> Tuple[float, float]:
        """
        Runs DenseNet-121 forward inference if PyTorch is available.
        Returns: (prob_authentic, prob_ai) in [0.0, 1.0].
        """
        if not TORCH_AVAILABLE or self.densenet is None or tensor is None:
            return 0.5, 0.5
        with torch.inference_mode():
            logits = self.densenet(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            return float(probs[CLASS_AUTHENTIC].item()), float(probs[CLASS_AI].item())

    def predict_resnet(self, tensor: Optional[Any]) -> Tuple[float, float]:
        """
        Runs ResNet-50 forward inference if PyTorch is available.
        Returns: (prob_authentic, prob_ai) in [0.0, 1.0].
        """
        if not TORCH_AVAILABLE or self.resnet is None or tensor is None:
            return 0.5, 0.5
        with torch.inference_mode():
            logits = self.resnet(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            return float(probs[CLASS_AUTHENTIC].item()), float(probs[CLASS_AI].item())

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns hardware, model checkpoint status, and operating mode."""
        return {
            "device": str(self.device),
            "torch_available": TORCH_AVAILABLE,
            "cuda_available": torch.cuda.is_available() if TORCH_AVAILABLE else False,
            "active_mode": self.active_mode,
            "densenet_ready": self.densenet is not None,
            "densenet_trained": self.densenet_trained,
            "resnet_ready": self.resnet is not None,
            "resnet_trained": self.resnet_trained,
            "checkpoints": self.checkpoint_status,
            "checkpoint_directory": CHECKPOINT_DIR
        }

# Global singleton instance
model_manager = DeepShieldModelManager()
