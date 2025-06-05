"""YOLOv8-based object detection for fashion items"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import torch
import logging
from pathlib import Path
import cv2

from config import (
    YOLO_MODEL_SIZE, 
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    FASHION_CLASSES,
    MODELS_CACHE_DIR
)

logger = logging.getLogger(__name__)


class FashionDetector:
    """YOLOv8-based fashion item detector"""
    
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
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize YOLO model
        self.model = self._load_model()
        
        # Map YOLO classes to fashion categories
        self.fashion_class_mapping = self._create_class_mapping()
        
    def _load_model(self) -> YOLO:
        """Load YOLOv8 model"""
        model_path = MODELS_CACHE_DIR / f"yolov8{self.model_size}.pt"
        
        if not model_path.exists():
            logger.info(f"Downloading YOLOv8{self.model_size} model...")
            model = YOLO(f"yolov8{self.model_size}.pt")
            # Save to cache directory
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
        else:
            logger.info(f"Loading YOLOv8{self.model_size} from cache...")
            model = YOLO(str(model_path))
            
        model.to(self.device)
        return model
    
    def _create_class_mapping(self) -> Dict[str, str]:
        """Map YOLO COCO classes to fashion categories"""
        # COCO class names that correspond to fashion items
        mapping = {
            # Accessories
            "backpack": "bag",
            "handbag": "bag",
            "suitcase": "bag",
            "tie": "accessories",
            "umbrella": "accessories",
            
            # General person detection (will need post-processing)
            "person": "person"
        }
        return mapping
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect fashion items in a frame
        
        Args:
            frame: Input frame (RGB)
            
        Returns:
            List of detections with format:
            {
                'class': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2],
                'area': float
            }
        """
        # Run YOLO detection
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                class_name = self.model.names[class_id]
                
                # Check if it's a fashion-related class
                if class_name in self.fashion_class_mapping:
                    fashion_class = self.fashion_class_mapping[class_name]
                    
                    detection = {
                        'class': fashion_class,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    detections.append(detection)
                    
                # For person detection, we'll use a custom fashion detector
                elif class_name == "person":
                    # Extract person region for further analysis
                    person_detections = self._detect_fashion_on_person(
                        frame, [int(x1), int(y1), int(x2), int(y2)]
                    )
                    detections.extend(person_detections)
                    
        return detections
    
    def _detect_fashion_on_person(self, frame: np.ndarray, person_bbox: List[int]) -> List[Dict]:
        """
        Detect fashion items on a detected person
        This is a simplified approach - in production, you'd use a specialized fashion detection model
        """
        x1, y1, x2, y2 = person_bbox
        person_region = frame[y1:y2, x1:x2]
        
        # Simple heuristic-based detection based on regions
        height = y2 - y1
        width = x2 - x1
        
        detections = []
        
        # Upper body region (top/jacket)
        upper_region = {
            'class': 'top',
            'confidence': 0.7,  # Lower confidence for heuristic detection
            'bbox': [x1, y1, x2, y1 + int(height * 0.4)],
            'area': width * (height * 0.4)
        }
        detections.append(upper_region)
        
        # Lower body region (bottom/dress)
        lower_region = {
            'class': 'bottom',
            'confidence': 0.7,
            'bbox': [x1, y1 + int(height * 0.4), x2, y2],
            'area': width * (height * 0.6)
        }
        detections.append(lower_region)
        
        # Potential bag detection (side regions)
        if width > 100:  # Only if person is large enough
            # Left side
            bag_region_left = {
                'class': 'bag',
                'confidence': 0.5,
                'bbox': [x1 - int(width * 0.1), y1 + int(height * 0.3), 
                        x1 + int(width * 0.2), y1 + int(height * 0.7)],
                'area': (width * 0.3) * (height * 0.4)
            }
            # Right side
            bag_region_right = {
                'class': 'bag',
                'confidence': 0.5,
                'bbox': [x2 - int(width * 0.2), y1 + int(height * 0.3),
                        x2 + int(width * 0.1), y1 + int(height * 0.7)],
                'area': (width * 0.3) * (height * 0.4)
            }
            
            # Check if regions are within frame bounds
            h, w = frame.shape[:2]
            for region in [bag_region_left, bag_region_right]:
                bbox = region['bbox']
                if bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] <= w and bbox[3] <= h:
                    detections.append(region)
                    
        return detections
    
    def extract_fashion_regions(
        self, 
        frame: np.ndarray, 
        detections: List[Dict]
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Extract cropped regions for each detection
        
        Args:
            frame: Original frame
            detections: List of detection dictionaries
            
        Returns:
            List of (cropped_image, detection_info) tuples
        """
        regions = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Extract region
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size > 0:  # Ensure valid crop
                regions.append((cropped, detection))
                
        return regions
    
    def visualize_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detections on frame
        
        Args:
            frame: Original frame
            detections: List of detections
            save_path: Optional path to save visualization
            
        Returns:
            Frame with drawn detections
        """
        vis_frame = frame.copy()
        
        # Define colors for different classes
        colors = {
            'top': (255, 0, 0),      # Red
            'bottom': (0, 255, 0),   # Green
            'dress': (0, 0, 255),    # Blue
            'bag': (255, 255, 0),    # Yellow
            'shoes': (255, 0, 255),  # Magenta
            'accessories': (0, 255, 255),  # Cyan
            'person': (128, 128, 128)  # Gray
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Get color
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(
                vis_frame,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            
        return vis_frame 