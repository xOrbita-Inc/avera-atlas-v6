"""
Physics-Based Orbital Object Classifier
========================================
ONNX runtime inference for DINOv2 ViT-Small fine-tuned
on physics-based synthetic spectrograms from orbital-pbsdg.

Input:  256x256 grayscale PNG (EM/RF or Thermal spectrogram)
Output: 5-class orbital object classification

Classes (match orbital-finetune CLASS_TO_IDX):
  0 ACTIVE_SAT   — Operational satellite
  1 DEAD_SAT     — Non-maneuvering, residual thermal mass
  2 DEBRIS_SMALL — Sub-10cm fragments, fast tumble
  3 DEBRIS_LARGE — 10cm+ fragments
  4 MANEUVERING  — Executing propulsion burn

Preprocessing matches get_val_transforms() from
orbital-finetune/core/transforms.py exactly:
  Resize to 224x224 bilinear
  Grayscale → RGB (3-channel repeat)
  Normalize to [0,1]
  Apply ImageNet mean/std normalization
  Shape: (1, 3, 224, 224) float32
"""

import time
import numpy as np
from PIL import Image
import onnxruntime as ort

CLASS_NAMES = [
    "ACTIVE_SAT",
    "DEAD_SAT",
    "DEBRIS_SMALL",
    "DEBRIS_LARGE",
    "MANEUVERING",
]

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
)
IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
)

RISK_MAP = {
    "ACTIVE_SAT":   "satellite",
    "DEAD_SAT":     "debris",
    "DEBRIS_SMALL": "debris",
    "DEBRIS_LARGE": "debris",
    "MANEUVERING":  "satellite",
}


class PhysicsClassifier:
    def __init__(self, model_path: str):
        print(
            f"[SYSTEM] Loading physics classifier: "
            f"{model_path}"
        )
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = (
                self.session.get_inputs()[0].name
            )
            self.output_name = (
                self.session.get_outputs()[0].name
            )
            print(
                f"[SYSTEM] Physics classifier ready. "
                f"Input: {self.input_name}"
            )
            self._ready = True
        except Exception as e:
            print(
                f"[CRITICAL] Failed to load model: {e}"
            )
            self.session = None
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def preprocess(
        self, image: Image.Image
    ) -> np.ndarray:
        img = image.resize((224, 224), Image.BILINEAR)
        if img.mode != "RGB":
            img = img.convert("L")
            arr = np.array(img, dtype=np.float32)
            arr = np.stack([arr] * 3, axis=-1)
        else:
            arr = np.array(img, dtype=np.float32)
        arr /= 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        return arr.astype(np.float32)

    def softmax(
        self, logits: np.ndarray
    ) -> np.ndarray:
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    def classify(self, image: Image.Image) -> dict:
        if not self._ready:
            return {
                "error": "Model not loaded",
                "class_label": "UNKNOWN",
                "confidence": 0.0,
            }
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        logits = self.session.run(
            [self.output_name],
            {self.input_name: tensor}
        )[0][0]
        probs = self.softmax(logits)
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])
        inference_ms = (
            time.perf_counter() - t0
        ) * 1000
        return {
            "class_label":  pred_class,
            "confidence":   pred_conf,
            "all_probs": {
                CLASS_NAMES[i]: float(probs[i])
                for i in range(len(CLASS_NAMES))
            },
            "risk_class":   RISK_MAP[pred_class],
            "inference_ms": round(inference_ms, 2),
        }
