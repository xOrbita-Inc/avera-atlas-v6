"""
YOLOv8 Spacecraft Detector (Ultralytics Backend)
================================================
Uses Ultralytics YOLO for inference, ensuring compatibility
with models exported from the same library.
"""

import time
from typing import Dict, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


CLASS_NAMES = [
    'AcrimSat', 'Aquarius', 'Aura', 'Calipso', 'Cloudsat',
    'CubeSat', 'Debris', 'Jason', 'Sentinel-6', 'TRMM', 'Terra'
]


class YOLODetector:
    """
    YOLOv8 detector for spacecraft detection using Ultralytics.
    
    Provides both classification and accurate bounding box localization.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLOv8 model (.pt or .onnx)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = CLASS_NAMES
        
        # Load model via Ultralytics (handles .pt, .onnx, .torchscript, etc.)
        self.model = YOLO(model_path)
        
        print(f"✅ YOLOv8 detector initialized (Ultralytics backend)")
        print(f"   Model: {model_path}")
        print(f"   Classes: {len(self.class_names)}")
    
    def detect(self, image: Image.Image) -> Dict:
        """
        Run detection on image.
        
        Args:
            image: PIL Image
            
        Returns:
            Detection result with best detection, annotated image, etc.
        """
        start_time = time.perf_counter()
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run inference
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Parse results
        detections = []
        result = results[0]  # Single image
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get bbox in xyxy format
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Map class_id to name (use model's names if available, fallback to ours)
                if hasattr(result, 'names') and result.names:
                    class_name = result.names.get(class_id, f"Class_{class_id}")
                else:
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                
                detections.append({
                    'bbox': bbox,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
        
        # Build response
        if detections:
            # Sort by confidence, get best
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            best = detections[0]
            
            # Create all_probs format for compatibility
            all_probs = [(det['class_name'], det['confidence']) for det in detections]
            detected_classes = {det['class_name'] for det in detections}
            for cls in self.class_names:
                if cls not in detected_classes:
                    all_probs.append((cls, 0.0))
            all_probs = sorted(all_probs, key=lambda x: x[1], reverse=True)
            
            # Annotate image
            annotated = self.draw_annotations(
                image, best['bbox'], best['class_name'],
                best['confidence'], detections
            )
            
            return {
                'class_name': best['class_name'],
                'confidence': best['confidence'],
                'bbox': best['bbox'],
                'all_probs': all_probs,
                'all_detections': detections,
                'annotated_image': annotated,
                'processing_time_ms': processing_time
            }
        else:
            # No detection
            return {
                'class_name': 'Unknown',
                'confidence': 0.0,
                'bbox': [0, 0, image.width, image.height],
                'all_probs': [(cls, 0.0) for cls in self.class_names],
                'all_detections': [],
                'annotated_image': image.copy(),
                'processing_time_ms': processing_time
            }
    
    def draw_annotations(self, image: Image.Image, bbox: List[int], class_name: str,
                         confidence: float, all_detections: List[Dict] = None) -> Image.Image:
        """Draw bounding boxes and labels on image."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Draw all detections (secondary ones in lighter color)
        if all_detections and len(all_detections) > 1:
            for det in all_detections:
                if det['bbox'] != bbox:  # Skip primary detection
                    x1, y1, x2, y2 = det['bbox']
                    draw.rectangle([x1, y1, x2, y2], outline='#666666', width=1)
        
        # Draw primary detection
        x1, y1, x2, y2 = bbox
        box_color = '#ef4444' if class_name == 'Debris' else '#22c55e'
        line_width = max(2, int(min(img.size) * 0.003))
        
        # Corner style box
        corner_length = max(int(min(x2 - x1, y2 - y1) * 0.15), 15)
        
        # Draw corners
        for (cx, cy, dx, dy) in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            draw.line([(cx, cy), (cx + dx * corner_length, cy)], fill=box_color, width=line_width)
            draw.line([(cx, cy), (cx, cy + dy * corner_length)], fill=box_color, width=line_width)
        
        # Dashed lines
        dash, gap = 8, 6
        for x in range(x1, x2, dash + gap):
            draw.line([(x, y1), (min(x + dash, x2), y1)], fill=box_color, width=1)
            draw.line([(x, y2), (min(x + dash, x2), y2)], fill=box_color, width=1)
        for y in range(y1, y2, dash + gap):
            draw.line([(x1, y), (x1, min(y + dash, y2))], fill=box_color, width=1)
            draw.line([(x2, y), (x2, min(y + dash, y2))], fill=box_color, width=1)
        
        # Label
        label = f"{class_name} {confidence*100:.1f}%"
        font_size = max(16, int(min(img.size) * 0.022))
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        label_x = x1
        label_y = y1 - text_h - 8 if y1 - text_h - 10 > 0 else y2 + 5
        
        draw.rectangle([label_x - 4, label_y - 4, label_x + text_w + 4, label_y + text_h + 4], fill=box_color)
        draw.text((label_x, label_y), label, fill='white', font=font)
        
        # Crosshair at center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.line([(cx - 8, cy), (cx + 8, cy)], fill=box_color, width=1)
        draw.line([(cx, cy - 8), (cx, cy + 8)], fill=box_color, width=1)
        
        return img


# Backward compatibility alias
SpacecraftDetector = YOLODetector


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python detector_yolo.py <model.onnx> <image>")
        sys.exit(1)
    
    detector = YOLODetector(sys.argv[1])
    image = Image.open(sys.argv[2]).convert('RGB')
    
    result = detector.detect(image)
    
    print(f"\n🎯 Detection Result:")
    print(f"   Class: {result['class_name']}")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    print(f"   Bbox: {result['bbox']}")
    print(f"   Time: {result['processing_time_ms']:.1f}ms")
    print(f"   All detections: {len(result['all_detections'])}")
    
    result['annotated_image'].save("yolo_detection_result.png")
    print(f"\n📷 Saved: yolo_detection_result.png")