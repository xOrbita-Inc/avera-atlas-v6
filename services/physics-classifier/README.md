# physics-classifier

Physics-based orbital object classifier microservice.
Part of AVERA-ATLAS v6.

## Responsibility

Single responsibility: classify 256x256 physics-based
spectrogram PNGs using DINOv2 ViT-Small ONNX model.

Parallel to detector service (SWIR/YOLOv8).
Both feed the tracker for multi-modal data fusion.

## Classes

ACTIVE_SAT, DEAD_SAT, DEBRIS_SMALL,
DEBRIS_LARGE, MANEUVERING

## API

POST /predict  — classify a spectrogram
GET  /health   — service health

## Model

orbital_classifier.onnx — DINOv2 ViT-Small
Trained on orbital-pbsdg 100K sample dataset.
ADR-009 Phase 1. Macro F1: 0.8811, ECE: 0.0213.

## License

Proprietary — Avera Enterprises Inc.
