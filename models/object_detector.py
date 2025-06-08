"""Simplified YOLOv8-based fashion detection with enhanced accuracy"""
import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO
import torch
import logging
import cv2

from config import (
    YOLO_MODEL_SIZE, 
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    MODELS_CACHE_DIR
)

logger = logging.getLogger(__name__)


class FashionDetector:
    """Simplified and enhanced fashion detector"""
    
    def __init__(
        self,
        model_size: str = YOLO_MODEL_SIZE,
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
        device: Optional[str] = None
    ):
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Set device - force CPU for compatibility
        self.device = 'cpu'  # Force CPU to avoid CUDA issues
            
        # Initialize YOLO model
        self.model = self._load_model()
        
        # Fashion detection mapping
        self.fashion_mapping = {
            "person": "person",
            "backpack": "bag",
            "handbag": "bag", 
            "suitcase": "bag",
            "tie": "accessories",
            "umbrella": "accessories"
        }
        
    def _load_model(self) -> YOLO:
        """Load YOLOv8 model"""
        try:
            model = YOLO(f"yolov8{self.model_size}.pt")
            model.to(self.device)
            logger.info(f"Loaded YOLOv8{self.model_size} model on {self.device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect fashion items in frame with enhanced accuracy
        
        Args:
            frame: Input frame (RGB format)
            
        Returns:
            List of detections with format:
            {
                'class': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2],
                'area': float
            }
        """
        try:
            # Run YOLO detection
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            person_boxes = []
            
            # Process YOLO results
            for result in results:
                if result.boxes is None:
                    continue
                    
                boxes = result.boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Check if it's a fashion-related class
                    if class_name in self.fashion_mapping:
                        fashion_class = self.fashion_mapping[class_name]
                        
                        if fashion_class == "person":
                            person_boxes.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence
                            })
                        else:
                            # Direct fashion item detection
                            detection = {
                                'class': fashion_class,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            detections.append(detection)
            
            # Enhanced: Generate fashion detections from person boxes
            for person_box in person_boxes:
                fashion_detections = self._generate_fashion_from_person(
                    frame, person_box['bbox'], person_box['confidence']
                )
                detections.extend(fashion_detections)
            
            # Remove duplicates and low-quality detections
            detections = self._filter_detections(detections)
            
            logger.info(f"Detected {len(detections)} fashion items")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _generate_fashion_from_person(
        self, 
        frame: np.ndarray, 
        person_bbox: List[int], 
        person_confidence: float
    ) -> List[Dict]:
        """Generate fashion item detections from person bounding box"""
        
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        width = x2 - x1
        
        # Only process if person is large enough
        if height < 100 or width < 50:
            return []
        
        detections = []
        
        # Analyze person region for outfit type
        person_region = frame[y1:y2, x1:x2]
        outfit_type = self._analyze_outfit_style(person_region)
        
        if outfit_type == "dress":
            # Full dress detection
            dress_detection = {
                'class': 'dress',
                'confidence': min(0.85, person_confidence + 0.1),
                'bbox': [
                    x1 + int(width * 0.1), 
                    y1 + int(height * 0.15),
                    x2 - int(width * 0.1), 
                    y2 - int(height * 0.1)
                ],
                'area': width * height * 0.75
            }
            detections.append(dress_detection)
        else:
            # Separate top and bottom
            # Top region
            top_detection = {
                'class': 'top',
                'confidence': min(0.80, person_confidence),
                'bbox': [
                    x1 + int(width * 0.05),
                    y1 + int(height * 0.1), 
                    x2 - int(width * 0.05),
                    y1 + int(height * 0.55)
                ],
                'area': width * height * 0.45
            }
            detections.append(top_detection)
            
            # Bottom region
            bottom_detection = {
                'class': 'bottom',
                'confidence': min(0.75, person_confidence),
                'bbox': [
                    x1 + int(width * 0.1),
                    y1 + int(height * 0.5),
                    x2 - int(width * 0.1), 
                    y2 - int(height * 0.05)
                ],
                'area': width * height * 0.45
            }
            detections.append(bottom_detection)
        
        # Check for accessories
        if self._has_accessories(person_region):
            accessories_detection = {
                'class': 'accessories',
                'confidence': min(0.70, person_confidence - 0.1),
                'bbox': [
                    x1 + int(width * 0.2),
                    y1 + int(height * 0.05),
                    x2 - int(width * 0.2),
                    y1 + int(height * 0.3)
                ],
                'area': width * height * 0.25
            }
            detections.append(accessories_detection)
        
        return detections
    
    def _analyze_outfit_style(self, person_region: np.ndarray) -> str:
        """Analyze if person is wearing dress or separate pieces"""
        if person_region.size == 0:
            return "separate"
        
        # Simple heuristic: analyze color continuity in middle section
        h, w = person_region.shape[:2]
        
        if h < 50 or w < 30:
            return "separate"
        
        # Sample middle vertical strip
        middle_strip = person_region[h//4:3*h//4, w//3:2*w//3]
        
        # Calculate color variance
        if len(middle_strip.shape) == 3:
            gray = cv2.cvtColor(middle_strip, cv2.COLOR_RGB2GRAY)
        else:
            gray = middle_strip
        
        variance = np.var(gray)
        
        # Lower variance suggests more uniform clothing (dress)
        return "dress" if variance < 800 else "separate"
    
    def _has_accessories(self, person_region: np.ndarray) -> bool:
        """Simple check for potential accessories"""
        if person_region.size == 0:
            return False
        
        # Check upper region for accessories
        h, w = person_region.shape[:2]
        upper_region = person_region[:h//3, :]
        
        # Simple edge detection for accessories
        if len(upper_region.shape) == 3:
            gray = cv2.cvtColor(upper_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = upper_region
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Higher edge density might indicate accessories
        return edge_density > 0.05
    
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter and clean up detections"""
        if not detections:
            return []
        
        # Remove very small detections
        filtered = [d for d in detections if d['area'] > 1000]
        
        # Remove very low confidence detections
        filtered = [d for d in filtered if d['confidence'] > 0.3]
        
        # Sort by confidence
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        
        return filtered
    
    def extract_fashion_regions(
        self, 
        frame: np.ndarray, 
        detections: List[Dict]
    ) -> List[tuple]:
        """Extract fashion item regions from frame"""
        regions = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Extract region
            region = frame[y1:y2, x1:x2]
            
            if region.size > 0:
                regions.append((region, detection))
        
        return regions 