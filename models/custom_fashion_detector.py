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
from collections import Counter

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


class EnhancedFashionDetector:
    """Enhanced YOLO-based fashion detection with validation and visualization"""
    
    def __init__(self, model_path: str = "yolov8m.pt", confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.detection_stats = {
            'total_detections': 0,
            'confidence_distribution': [],
            'class_distribution': {},
            'false_positive_indicators': []
        }
        
        # Enhanced fashion class mapping with validation
        self.fashion_classes = {
            # Person-based fashion
            'person': ['top', 'bottom', 'dress', 'outerwear'],
            # Direct fashion items
            'handbag': 'bag',
            'tie': 'accessory', 
            'suitcase': 'bag',
            'backpack': 'bag',
            'umbrella': 'accessory'
        }
        
        # Enhanced color palette with fashion-specific colors
        self.fashion_colors = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'pink': (255, 192, 203),
            'purple': (128, 0, 128),
            'orange': (255, 165, 0),
            'brown': (165, 42, 42),
            'gray': (128, 128, 128),
            'navy': (0, 0, 128),
            'beige': (245, 245, 220),
            'cream': (255, 253, 208),
            'burgundy': (128, 0, 32),
            'olive': (128, 128, 0),
            'coral': (255, 127, 80),
            'turquoise': (64, 224, 208),
            'lavender': (230, 230, 250),
            'gold': (255, 215, 0),
            'silver': (192, 192, 192),
            'denim': (72, 118, 165),
            'khaki': (240, 230, 140),
            'maroon': (128, 0, 0),
            'teal': (0, 128, 128)
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with validation"""
        try:
            logger.info("Loading enhanced YOLO model with validation...")
            self.model = YOLO(self.model_path)
            logger.info("✅ YOLO model loaded successfully")
            
            # Validate model capabilities
            self._validate_model()
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            raise
    
    def _validate_model(self):
        """Validate YOLO model capabilities"""
        try:
            # Test with dummy image
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = self.model(dummy_image, verbose=False)
            
            # Check if model can detect relevant classes
            available_classes = self.model.names
            fashion_relevant = []
            
            for class_id, class_name in available_classes.items():
                if any(keyword in class_name.lower() for keyword in 
                      ['person', 'handbag', 'tie', 'suitcase', 'backpack', 'umbrella']):
                    fashion_relevant.append(class_name)
            
            logger.info(f"✅ Model validation complete. Fashion-relevant classes: {fashion_relevant}")
            
        except Exception as e:
            logger.warning(f"⚠️ Model validation failed: {e}")
    
    def detect_with_validation(self, image: np.ndarray, save_visualization: bool = False, 
                             output_path: Optional[str] = None) -> Tuple[List[Dict], Dict]:
        """Enhanced detection with comprehensive validation"""
        
        if self.model is None:
            logger.error("❌ Model not loaded")
            return [], {}
        
        try:
            # Run YOLO detection
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            validation_metrics = {
                'total_detections': 0,
                'confidence_stats': {},
                'class_distribution': {},
                'quality_indicators': {},
                'suspicious_detections': []
            }
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Extract detection info
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # Validate detection quality
                        quality_score = self._validate_detection_quality(
                            image, bbox, confidence, class_name
                        )
                        
                        # Process fashion-relevant detections
                        if self._is_fashion_relevant(class_name):
                            detection_info = self._process_fashion_detection(
                                image, bbox, class_name, confidence, quality_score
                            )
                            
                            if detection_info:
                                detections.append(detection_info)
                                
                                # Update validation metrics
                                validation_metrics['total_detections'] += 1
                                validation_metrics['class_distribution'][class_name] = \
                                    validation_metrics['class_distribution'].get(class_name, 0) + 1
                                
                                # Track suspicious detections
                                if quality_score < 0.6:
                                    validation_metrics['suspicious_detections'].append({
                                        'class': class_name,
                                        'confidence': confidence,
                                        'quality_score': quality_score,
                                        'bbox': bbox.tolist()
                                    })
            
            # Calculate validation metrics
            validation_metrics['confidence_stats'] = self._calculate_confidence_stats(detections)
            validation_metrics['quality_indicators'] = self._calculate_quality_indicators(detections)
            
            # Save visualization if requested
            if save_visualization and output_path:
                self._save_detection_visualization(image, detections, output_path, validation_metrics)
            
            # Update global stats
            self._update_detection_stats(detections, validation_metrics)
            
            logger.info(f"✅ Detection complete: {len(detections)} fashion items, "
                       f"avg confidence: {validation_metrics['confidence_stats'].get('mean', 0):.3f}")
            
            return detections, validation_metrics
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            return [], {}
    
    def _validate_detection_quality(self, image: np.ndarray, bbox: np.ndarray, 
                                  confidence: float, class_name: str) -> float:
        """Validate individual detection quality"""
        quality_score = 0.0
        
        # 1. Confidence validation (30% weight)
        confidence_score = min(confidence / 0.9, 1.0)  # Normalize to 0.9 max
        quality_score += confidence_score * 0.3
        
        # 2. Bounding box validation (25% weight)
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        image_area = image.shape[0] * image.shape[1]
        
        # Check reasonable size (not too small or too large)
        area_ratio = bbox_area / image_area
        if 0.01 <= area_ratio <= 0.8:  # 1% to 80% of image
            bbox_score = 1.0
        elif area_ratio < 0.01:
            bbox_score = area_ratio / 0.01  # Penalize tiny detections
        else:
            bbox_score = 0.8 / area_ratio  # Penalize huge detections
        
        quality_score += bbox_score * 0.25
        
        # 3. Aspect ratio validation (20% weight)
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
        
        # Fashion items should have reasonable aspect ratios
        if class_name == 'person':
            # People are typically taller than wide
            ideal_ratio = 0.5  # Height > Width
            ratio_score = 1.0 - abs(aspect_ratio - ideal_ratio) / 2.0
        else:
            # Other fashion items can vary more
            ratio_score = 1.0 if 0.2 <= aspect_ratio <= 5.0 else 0.5
        
        quality_score += max(0, ratio_score) * 0.2
        
        # 4. Image region validation (25% weight)
        region_score = self._validate_image_region(image, bbox)
        quality_score += region_score * 0.25
        
        return max(0.0, min(1.0, quality_score))
    
    def _validate_image_region(self, image: np.ndarray, bbox: np.ndarray) -> float:
        """Validate the image region for fashion content"""
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                return 0.0
            
            # Check color diversity (fashion items should have some color variation)
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            color_std = np.std(gray_region)
            color_score = min(color_std / 50.0, 1.0)  # Normalize
            
            # Check edge density (fashion items should have defined edges)
            edges = cv2.Canny(gray_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(edge_density * 10, 1.0)  # Normalize
            
            return (color_score + edge_score) / 2.0
            
        except Exception:
            return 0.5  # Default score if validation fails
    
    def _process_fashion_detection(self, image: np.ndarray, bbox: np.ndarray, 
                                 class_name: str, confidence: float, quality_score: float) -> Optional[Dict]:
        """Process and validate fashion detection"""
        
        # Skip low-quality detections
        if quality_score < 0.4:
            logger.debug(f"⚠️ Skipping low-quality detection: {class_name} (quality: {quality_score:.3f})")
            return None
        
        try:
            # Extract region for color analysis
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            region = image[y1:y2, x1:x2]
            
            # Enhanced color detection
            detected_color = self._detect_enhanced_color(region)
            
            # Map to fashion category
            fashion_type = self._map_to_fashion_type(class_name, bbox, image.shape)
            
            detection_info = {
                'class': fashion_type,
                'confidence': confidence,
                'quality_score': quality_score,
                'bbox': bbox.tolist(),
                'color': detected_color,
                'yolo_class': class_name,
                'region_size': region.shape[:2] if region.size > 0 else (0, 0)
            }
            
            return detection_info
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to process detection: {e}")
            return None
    
    def _detect_enhanced_color(self, region: np.ndarray) -> str:
        """Enhanced color detection with fashion-specific logic"""
        
        if region.size == 0:
            return 'unknown'
        
        try:
            # Resize region for faster processing
            if region.shape[0] > 100 or region.shape[1] > 100:
                region = cv2.resize(region, (100, 100))
            
            # Convert to RGB
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # Flatten for clustering
            pixels = region_rgb.reshape(-1, 3)
            
            # Remove very dark/light pixels (likely shadows/highlights)
            brightness = np.mean(pixels, axis=1)
            valid_pixels = pixels[(brightness > 30) & (brightness < 225)]
            
            if len(valid_pixels) < 10:
                return 'unknown'
            
            # Use multiple clustering approaches
            dominant_colors = []
            
            # Method 1: K-means clustering
            try:
                kmeans = KMeans(n_clusters=min(5, len(valid_pixels)//10), random_state=42, n_init=10)
                kmeans.fit(valid_pixels)
                
                # Get cluster centers and their sizes
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                
                for i, center in enumerate(centers):
                    cluster_size = np.sum(labels == i)
                    if cluster_size > len(valid_pixels) * 0.1:  # At least 10% of pixels
                        color_name = self._rgb_to_fashion_color(center)
                        dominant_colors.append((color_name, cluster_size))
                        
            except Exception:
                pass
            
            # Method 2: Histogram-based approach
            try:
                hist_colors = self._histogram_color_analysis(valid_pixels)
                dominant_colors.extend(hist_colors)
            except Exception:
                pass
            
            # Select best color
            if dominant_colors:
                # Sort by frequency and filter out 'unknown'
                valid_colors = [(color, freq) for color, freq in dominant_colors if color != 'unknown']
                if valid_colors:
                    return max(valid_colors, key=lambda x: x[1])[0]
            
            # Fallback: simple average
            avg_color = np.mean(valid_pixels, axis=0)
            return self._rgb_to_fashion_color(avg_color)
            
        except Exception as e:
            logger.debug(f"Color detection failed: {e}")
            return 'unknown'
    
    def _histogram_color_analysis(self, pixels: np.ndarray) -> List[Tuple[str, int]]:
        """Histogram-based color analysis"""
        colors = []
        
        # Analyze each channel
        for channel in range(3):
            hist, bins = np.histogram(pixels[:, channel], bins=16, range=(0, 256))
            dominant_bin = np.argmax(hist)
            
            # Create representative color
            channel_value = bins[dominant_bin] + (bins[1] - bins[0]) / 2
            
            if channel == 0:  # Red dominant
                test_color = [channel_value, 100, 100]
            elif channel == 1:  # Green dominant  
                test_color = [100, channel_value, 100]
            else:  # Blue dominant
                test_color = [100, 100, channel_value]
            
            color_name = self._rgb_to_fashion_color(test_color)
            colors.append((color_name, hist[dominant_bin]))
        
        return colors
    
    def _rgb_to_fashion_color(self, rgb: np.ndarray) -> str:
        """Convert RGB to fashion color name with enhanced logic"""
        
        try:
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            
            # Handle grayscale
            if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
                avg = (r + g + b) / 3
                if avg < 50:
                    return 'black'
                elif avg > 200:
                    return 'white'
                elif avg < 100:
                    return 'gray'
                else:
                    return 'gray'
            
            # Find closest fashion color
            min_distance = float('inf')
            closest_color = 'unknown'
            
            for color_name, color_rgb in self.fashion_colors.items():
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip((r, g, b), color_rgb)))
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name
            
            # Only return if reasonably close
            if min_distance < 100:  # Threshold for color similarity
                return closest_color
            
            # Fallback to basic color analysis
            if r > g and r > b:
                return 'red' if r > 150 else 'brown'
            elif g > r and g > b:
                return 'green'
            elif b > r and b > g:
                return 'blue' if b > 150 else 'navy'
            
            return 'unknown'
            
        except Exception:
            return 'unknown'
    
    def _save_detection_visualization(self, image: np.ndarray, detections: List[Dict], 
                                    output_path: str, validation_metrics: Dict):
        """Save detection visualization with validation info"""
        
        try:
            vis_image = image.copy()
            
            # Draw detections
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Color code by quality
                quality = detection['quality_score']
                if quality > 0.8:
                    color = (0, 255, 0)  # Green - high quality
                elif quality > 0.6:
                    color = (0, 255, 255)  # Yellow - medium quality
                else:
                    color = (0, 0, 255)  # Red - low quality
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Add labels
                label = f"{detection['class']} ({detection['confidence']:.2f})"
                color_label = f"Color: {detection['color']}"
                quality_label = f"Q: {quality:.2f}"
                
                cv2.putText(vis_image, label, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(vis_image, color_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(vis_image, quality_label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add validation summary
            summary_text = [
                f"Total: {validation_metrics['total_detections']}",
                f"Avg Conf: {validation_metrics['confidence_stats'].get('mean', 0):.2f}",
                f"Suspicious: {len(validation_metrics['suspicious_detections'])}"
            ]
            
            for i, text in enumerate(summary_text):
                cv2.putText(vis_image, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save image
            cv2.imwrite(output_path, vis_image)
            logger.info(f"✅ Visualization saved: {output_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save visualization: {e}")
    
    def _calculate_confidence_stats(self, detections: List[Dict]) -> Dict:
        """Calculate confidence statistics"""
        if not detections:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        confidences = [d['confidence'] for d in detections]
        return {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'count': len(confidences)
        }
    
    def _calculate_quality_indicators(self, detections: List[Dict]) -> Dict:
        """Calculate quality indicators"""
        if not detections:
            return {}
        
        quality_scores = [d['quality_score'] for d in detections]
        colors = [d['color'] for d in detections]
        
        return {
            'avg_quality': np.mean(quality_scores),
            'high_quality_count': sum(1 for q in quality_scores if q > 0.8),
            'low_quality_count': sum(1 for q in quality_scores if q < 0.6),
            'unknown_color_ratio': colors.count('unknown') / len(colors),
            'color_diversity': len(set(colors))
        }
    
    def _update_detection_stats(self, detections: List[Dict], validation_metrics: Dict):
        """Update global detection statistics"""
        self.detection_stats['total_detections'] += len(detections)
        
        for detection in detections:
            self.detection_stats['confidence_distribution'].append(detection['confidence'])
            
            class_name = detection['class']
            self.detection_stats['class_distribution'][class_name] = \
                self.detection_stats['class_distribution'].get(class_name, 0) + 1
        
        # Track potential false positives
        suspicious = validation_metrics.get('suspicious_detections', [])
        self.detection_stats['false_positive_indicators'].extend(suspicious)
    
    def get_detection_report(self) -> Dict:
        """Generate comprehensive detection quality report"""
        stats = self.detection_stats
        
        if stats['total_detections'] == 0:
            return {'status': 'no_detections', 'message': 'No detections to analyze'}
        
        confidences = stats['confidence_distribution']
        
        report = {
            'total_detections': stats['total_detections'],
            'confidence_analysis': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'below_threshold': sum(1 for c in confidences if c < 0.7),
                'high_confidence': sum(1 for c in confidences if c > 0.8)
            },
            'class_distribution': stats['class_distribution'],
            'quality_concerns': {
                'suspicious_detections': len(stats['false_positive_indicators']),
                'false_positive_rate_estimate': len(stats['false_positive_indicators']) / stats['total_detections']
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if report['confidence_analysis']['mean'] < 0.7:
            report['recommendations'].append("Consider increasing confidence threshold")
        
        if report['quality_concerns']['false_positive_rate_estimate'] > 0.2:
            report['recommendations'].append("High false positive rate detected - review detection logic")
        
        if stats['class_distribution'].get('unknown', 0) > stats['total_detections'] * 0.3:
            report['recommendations'].append("Many unknown classifications - improve class mapping")
        
        return report
    
    # Legacy methods for compatibility
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Legacy detect method for compatibility"""
        detections, _ = self.detect_with_validation(image)
        return detections
    
    def _is_fashion_relevant(self, class_name: str) -> bool:
        """Check if detected class is fashion-relevant"""
        return class_name.lower() in ['person', 'handbag', 'tie', 'suitcase', 'backpack', 'umbrella']
    
    def _map_to_fashion_type(self, class_name: str, bbox: np.ndarray, image_shape: Tuple) -> str:
        """Map YOLO class to fashion type"""
        if class_name == 'person':
            # Analyze person region to determine clothing type
            return self._analyze_person_region(bbox, image_shape)
        else:
            return self.fashion_classes.get(class_name, 'accessory')
    
    def _analyze_person_region(self, bbox: np.ndarray, image_shape: Tuple) -> str:
        """Analyze person bounding box to determine clothing type"""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        
        # Simple heuristic based on aspect ratio and position
        aspect_ratio = height / width if width > 0 else 1
        
        if aspect_ratio > 2:  # Very tall - likely full body
            return 'dress'
        elif aspect_ratio > 1.5:  # Tall - likely top + bottom
            return 'top'
        else:  # Wide - likely partial view
            return 'top'


def load_best_available_model() -> EnhancedFashionDetector:
    """Load the best available fashion detection model"""
    
    # Check for custom trained models
    custom_model_paths = [
        "fashion_training/models/fashion_yolo_m_best.pt",
        "models/cache/custom_fashion_yolo.pt"
    ]
    
    for model_path in custom_model_paths:
        if Path(model_path).exists():
            logger.info(f"Found custom model: {model_path}")
            return EnhancedFashionDetector(model_path=model_path)
    
    # Fallback to enhanced pre-trained model
    logger.info("Using enhanced pre-trained model")
    return EnhancedFashionDetector() 