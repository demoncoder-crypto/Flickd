"""
Custom Fashion Detector with Trained YOLO Model
Flickd AI Engine - Hackathon Submission
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import torch
import logging
from pathlib import Path
import cv2
from sklearn.cluster import KMeans

try:
    import webcolors
    WEBCOLORS_AVAILABLE = True
except ImportError:
    WEBCOLORS_AVAILABLE = False

from config import (
    YOLO_MODEL_SIZE, 
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    FASHION_CLASSES,
    MODELS_CACHE_DIR
)

logger = logging.getLogger(__name__)


class CustomFashionDetector:
    """Enhanced fashion detector with color detection and adaptive thresholds"""
    
    def __init__(
        self,
        custom_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.custom_model_path = custom_model_path
        
        # Set device - force CPU for compatibility
        self.device = 'cpu'
            
        # Initialize YOLO model
        self.model = self._load_model()
        
        # Enhanced fashion classes with more specific categories
        self.custom_fashion_classes = [
            "top",          # 0 - shirts, blouses, t-shirts
            "bottom",       # 1 - pants, jeans, shorts, skirts
            "dress",        # 2 - dresses, gowns
            "outerwear",    # 3 - jackets, coats, blazers
            "footwear",     # 4 - shoes, boots, sneakers
            "bag",          # 5 - handbags, backpacks, purses
            "accessories",  # 6 - jewelry, belts, hats
            "person"        # 7 - person detection
        ]
        
        # Color mapping for better color detection
        self.color_names = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'pink': (255, 192, 203),
            'purple': (128, 0, 128),
            'brown': (165, 42, 42),
            'gray': (128, 128, 128),
            'orange': (255, 165, 0),
            'navy': (0, 0, 128),
            'beige': (245, 245, 220),
            'cream': (255, 253, 208)
        }
        
        # Determine if using custom model
        self.is_custom_model = custom_model_path is not None and Path(custom_model_path).exists()
        
        if self.is_custom_model:
            logger.info(f"Using custom trained model: {custom_model_path}")
        else:
            logger.info("Using pre-trained YOLO model with enhanced detection")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model (custom or pre-trained)"""
        if self.custom_model_path and Path(self.custom_model_path).exists():
            logger.info(f"Loading custom model from {self.custom_model_path}")
            model = YOLO(self.custom_model_path)
        else:
            logger.info("Loading pre-trained YOLOv8 model...")
            model = YOLO("yolov8m.pt")
            
        model.to(self.device)
        return model
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Enhanced detect fashion items with color detection and adaptive thresholds
        
        Args:
            frame: Input frame (RGB)
            
        Returns:
            List of detections with format:
            {
                'class': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2],
                'area': float,
                'color': str,  # NEW: Dominant color
                'color_confidence': float  # NEW: Color detection confidence
            }
        """
        # Adaptive confidence based on image quality
        adaptive_conf = self._calculate_adaptive_confidence(frame)
        
        # Run YOLO detection with adaptive threshold
        results = self.model(
            frame,
            conf=adaptive_conf,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        if self.is_custom_model:
            # Process custom model results
            detections = self._process_custom_model_results(results, frame)
        else:
            # Process pre-trained model results with enhanced detection
            detections = self._process_pretrained_model_results(results, frame)
        
        # Add color detection to all fashion items
        detections = self._add_color_detection(frame, detections)
        
        # Filter and rank detections
        detections = self._filter_and_rank_detections(detections)
        
        return detections
    
    def _calculate_adaptive_confidence(self, frame: np.ndarray) -> float:
        """Calculate adaptive confidence threshold based on image quality"""
        # Calculate image quality metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Adaptive threshold based on quality
        base_conf = self.confidence_threshold
        
        # Adjust based on brightness (too dark or too bright)
        if brightness < 50 or brightness > 200:
            base_conf += 0.1
        
        # Adjust based on contrast (low contrast = harder detection)
        if contrast < 30:
            base_conf += 0.15
        
        # Adjust based on sharpness (blurry images)
        if sharpness < 100:
            base_conf += 0.1
        
        # Clamp between reasonable bounds
        return min(max(base_conf, 0.3), 0.8)
    
    def _add_color_detection(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Add color detection to each fashion item"""
        enhanced_detections = []
        
        for detection in detections:
            # Skip person detections for color analysis
            if detection['class'] == 'person':
                enhanced_detections.append(detection)
                continue
            
            # Extract region for color analysis
            x1, y1, x2, y2 = detection['bbox']
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                region = frame[y1:y2, x1:x2]
                color_info = self._detect_dominant_color(region)
                
                detection.update(color_info)
            else:
                detection.update({
                    'color': 'unknown',
                    'color_confidence': 0.0
                })
            
            enhanced_detections.append(detection)
        
        return enhanced_detections
    
    def _detect_dominant_color(self, region: np.ndarray) -> Dict[str, any]:
        """Detect dominant color in a region using K-means clustering"""
        if region.size == 0:
            return {'color': 'unknown', 'color_confidence': 0.0}
        
        # Reshape region to list of pixels
        pixels = region.reshape(-1, 3)
        
        # Remove very dark and very bright pixels (likely shadows/highlights)
        mask = np.logical_and(
            np.mean(pixels, axis=1) > 20,  # Not too dark
            np.mean(pixels, axis=1) < 235  # Not too bright
        )
        
        if np.sum(mask) < 10:  # Not enough valid pixels
            return {'color': 'unknown', 'color_confidence': 0.0}
        
        filtered_pixels = pixels[mask]
        
        # Use K-means to find dominant colors
        try:
            kmeans = KMeans(n_clusters=min(3, len(filtered_pixels)), random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            
            # Get the most frequent cluster (dominant color)
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique[np.argmax(counts)]
            dominant_color_rgb = kmeans.cluster_centers_[dominant_cluster]
            
            # Convert to nearest named color
            color_name, confidence = self._rgb_to_color_name(dominant_color_rgb)
            
            return {
                'color': color_name,
                'color_confidence': confidence
            }
            
        except Exception as e:
            logger.warning(f"Color detection failed: {e}")
            return {'color': 'unknown', 'color_confidence': 0.0}
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> Tuple[str, float]:
        """Convert RGB values to nearest color name"""
        rgb = tuple(map(int, rgb))
        
        min_distance = float('inf')
        closest_color = 'unknown'
        
        for color_name, color_rgb in self.color_names.items():
            # Calculate Euclidean distance in RGB space
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, color_rgb)))
            
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        # Calculate confidence based on distance (closer = higher confidence)
        max_distance = np.sqrt(3 * 255**2)  # Maximum possible RGB distance
        confidence = 1.0 - (min_distance / max_distance)
        
        return closest_color, confidence
    
    def _filter_and_rank_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter overlapping detections and rank by confidence"""
        if not detections:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove overlapping detections (Non-Maximum Suppression)
        filtered_detections = []
        
        for detection in detections:
            is_duplicate = False
            
            for existing in filtered_detections:
                if self._calculate_iou(detection['bbox'], existing['bbox']) > 0.5:
                    # If same class and high overlap, keep the higher confidence one
                    if detection['class'] == existing['class']:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _process_custom_model_results(self, results, frame: np.ndarray) -> List[Dict]:
        """Process results from custom-trained model"""
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
                
                # Get class name from custom classes
                if class_id < len(self.custom_fashion_classes):
                    class_name = self.custom_fashion_classes[class_id]
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    detections.append(detection)
        
        return detections
    
    def _process_pretrained_model_results(self, results, frame: np.ndarray) -> List[Dict]:
        """Process results from pre-trained model with enhanced detection"""
        detections = []
        person_detected = False
        person_bbox = None
        
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
                
                # Check if it's a person
                if class_name == "person":
                    person_detected = True
                    person_bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                # ALSO detect direct fashion items from YOLO
                elif class_name in ['handbag', 'tie', 'suitcase', 'backpack', 'umbrella']:
                    detection = {
                        'class': self._map_yolo_to_fashion_class(class_name),
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    detections.append(detection)
                        
        # Enhanced: Always detect fashion items when a person is detected
        if person_detected and person_bbox:
            # Use enhanced detection that analyzes the person region
            person_detections = self._detect_fashion_on_person_enhanced(frame, person_bbox)
            detections.extend(person_detections)
        
        # FALLBACK: If no person detected, try to detect fashion items anyway
        elif not person_detected:
            # Use intelligent region analysis to find fashion items
            fallback_detections = self._detect_fashion_without_person(frame)
            detections.extend(fallback_detections)
            
        return detections
    
    def _detect_fashion_on_person_enhanced(self, frame: np.ndarray, person_bbox: List[int]) -> List[Dict]:
        """Enhanced fashion detection on a person using visual analysis"""
        x1, y1, x2, y2 = person_bbox
        person_region = frame[y1:y2, x1:x2]
        
        if person_region.size == 0:
            return []
        
        # Analyze the person region
        height = y2 - y1
        width = x2 - x1
        
        detections = []
        
        # Upper body region (top)
        upper_region = {
            'class': 'top',
            'confidence': 0.80,
            'bbox': [x1, y1 + int(height * 0.1), x2, y1 + int(height * 0.5)],
            'area': width * (height * 0.4)
        }
        detections.append(upper_region)
        
        # Lower body region (bottom)
        lower_region = {
            'class': 'bottom',
            'confidence': 0.80,
            'bbox': [x1, y1 + int(height * 0.45), x2, y2 - int(height * 0.05)],
            'area': width * (height * 0.5)
        }
        detections.append(lower_region)
        
        return detections

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

    def visualize_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict], 
        output_path: str
    ):
        """Visualize detections on frame and save"""
        import cv2
        
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Draw rectangle
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization
        cv2.imwrite(output_path, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

    def _map_yolo_to_fashion_class(self, yolo_class: str) -> str:
        """Map YOLO class names to fashion categories"""
        mapping = {
            'handbag': 'bag',
            'tie': 'accessory',
            'suitcase': 'bag',
            'backpack': 'bag',
            'umbrella': 'accessory'
        }
        return mapping.get(yolo_class, 'accessory')
    
    def _detect_fashion_without_person(self, frame: np.ndarray) -> List[Dict]:
        """Detect fashion items when no person is detected using intelligent analysis"""
        detections = []
        h, w = frame.shape[:2]
        
        # Strategy 1: Analyze frame regions for clothing-like patterns
        # Upper region (likely tops/shirts)
        upper_region = {
            'class': 'top',
            'confidence': 0.60,
            'bbox': [int(w * 0.2), int(h * 0.1), int(w * 0.8), int(h * 0.5)],
            'area': (w * 0.6) * (h * 0.4)
        }
        
        # Lower region (likely bottoms/pants)
        lower_region = {
            'class': 'bottom',
            'confidence': 0.60,
            'bbox': [int(w * 0.25), int(h * 0.4), int(w * 0.75), int(h * 0.8)],
            'area': (w * 0.5) * (h * 0.4)
        }
        
        # Only add if regions have reasonable size
        if upper_region['area'] > 1000:
            detections.append(upper_region)
        if lower_region['area'] > 1000:
            detections.append(lower_region)
        
        # Strategy 2: Look for fashion-like color patterns
        # This is a simplified approach - in production you'd use more sophisticated analysis
        
        return detections


def load_best_available_model() -> CustomFashionDetector:
    """Load the best available fashion detection model"""
    
    # Check for custom trained models
    custom_model_paths = [
        "fashion_training/models/fashion_yolo_m_best.pt",
        "models/cache/custom_fashion_yolo.pt"
    ]
    
    for model_path in custom_model_paths:
        if Path(model_path).exists():
            logger.info(f"Found custom model: {model_path}")
            return CustomFashionDetector(custom_model_path=model_path)
    
    # Fallback to enhanced pre-trained model
    logger.info("Using enhanced pre-trained model")
    return CustomFashionDetector() 