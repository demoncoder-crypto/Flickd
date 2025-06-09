"""Detection Validation and Enhanced Color Analysis"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.cluster import KMeans
import webcolors
from collections import Counter

logger = logging.getLogger(__name__)

class DetectionValidator:
    """Validates YOLO detections and provides enhanced color analysis"""
    
    def __init__(self):
        self.detection_stats = {
            'total_detections': 0,
            'confidence_distribution': [],
            'quality_scores': [],
            'false_positive_indicators': [],
            'color_accuracy_samples': []
        }
        
        # Enhanced fashion color palette (25 colors vs 14)
        self.fashion_colors = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 128, 0),
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
    
    def validate_detection_quality(self, image: np.ndarray, bbox: np.ndarray, 
                                 confidence: float, class_name: str) -> Dict:
        """Comprehensive detection quality validation"""
        
        validation_result = {
            'quality_score': 0.0,
            'confidence_score': 0.0,
            'bbox_score': 0.0,
            'region_score': 0.0,
            'aspect_ratio_score': 0.0,
            'issues': [],
            'is_valid': False
        }
        
        try:
            # 1. Confidence validation (25% weight)
            conf_score = self._validate_confidence(confidence)
            validation_result['confidence_score'] = conf_score
            
            if confidence < 0.5:
                validation_result['issues'].append(f"Low confidence: {confidence:.3f}")
            
            # 2. Bounding box validation (25% weight)
            bbox_score = self._validate_bbox(image, bbox)
            validation_result['bbox_score'] = bbox_score
            
            # 3. Image region validation (25% weight)
            region_score = self._validate_image_region(image, bbox)
            validation_result['region_score'] = region_score
            
            # 4. Aspect ratio validation (25% weight)
            aspect_score = self._validate_aspect_ratio(bbox, class_name)
            validation_result['aspect_ratio_score'] = aspect_score
            
            # Calculate overall quality score
            quality_score = (conf_score + bbox_score + region_score + aspect_score) / 4.0
            validation_result['quality_score'] = quality_score
            
            # Determine if detection is valid
            validation_result['is_valid'] = quality_score > 0.6 and confidence > 0.5
            
            # Update statistics
            self._update_validation_stats(validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def _validate_confidence(self, confidence: float) -> float:
        """Validate detection confidence"""
        # Normalize confidence score
        if confidence >= 0.9:
            return 1.0
        elif confidence >= 0.7:
            return 0.8
        elif confidence >= 0.5:
            return 0.6
        else:
            return confidence / 0.5  # Linear scaling below 0.5
    
    def _validate_bbox(self, image: np.ndarray, bbox: np.ndarray) -> float:
        """Validate bounding box properties"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Check bbox is within image bounds
            h, w = image.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return 0.3  # Partially out of bounds
            
            # Check bbox size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            image_area = h * w
            
            area_ratio = bbox_area / image_area
            
            # Ideal size range: 1% to 60% of image
            if 0.01 <= area_ratio <= 0.6:
                return 1.0
            elif area_ratio < 0.01:
                return max(0.2, area_ratio / 0.01)  # Too small
            else:
                return max(0.2, 0.6 / area_ratio)  # Too large
                
        except Exception:
            return 0.0
    
    def _validate_image_region(self, image: np.ndarray, bbox: np.ndarray) -> float:
        """Validate the actual image region content"""
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                return 0.0
            
            # Convert to grayscale for analysis
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # 1. Color diversity check (fashion items should have some variation)
            color_std = np.std(gray_region)
            color_diversity_score = min(color_std / 30.0, 1.0)
            
            # 2. Edge density check (fashion items should have defined edges)
            edges = cv2.Canny(gray_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(edge_density * 5, 1.0)
            
            # 3. Texture analysis (avoid uniform regions)
            texture_score = self._analyze_texture(gray_region)
            
            # Combine scores
            region_score = (color_diversity_score + edge_score + texture_score) / 3.0
            
            return region_score
            
        except Exception:
            return 0.5  # Default if analysis fails
    
    def _analyze_texture(self, gray_region: np.ndarray) -> float:
        """Analyze texture complexity of region"""
        try:
            # Calculate local binary pattern-like measure
            if gray_region.shape[0] < 3 or gray_region.shape[1] < 3:
                return 0.5
            
            # Simple texture measure using gradient magnitude
            grad_x = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            texture_measure = np.mean(gradient_magnitude)
            return min(texture_measure / 20.0, 1.0)
            
        except Exception:
            return 0.5
    
    def _validate_aspect_ratio(self, bbox: np.ndarray, class_name: str) -> float:
        """Validate aspect ratio for different object types"""
        try:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            if height == 0:
                return 0.0
            
            aspect_ratio = width / height
            
            # Expected aspect ratios for different classes
            if class_name == 'person':
                # People are typically taller than wide (0.3 to 0.8)
                if 0.3 <= aspect_ratio <= 0.8:
                    return 1.0
                else:
                    return max(0.3, 1.0 - abs(aspect_ratio - 0.5) / 2.0)
            
            elif class_name in ['handbag', 'suitcase']:
                # Bags can vary but usually not extremely tall/wide (0.5 to 2.0)
                if 0.5 <= aspect_ratio <= 2.0:
                    return 1.0
                else:
                    return max(0.4, 1.0 - abs(aspect_ratio - 1.0) / 3.0)
            
            else:
                # Other items - more flexible (0.2 to 5.0)
                if 0.2 <= aspect_ratio <= 5.0:
                    return 1.0
                else:
                    return 0.6
                    
        except Exception:
            return 0.5
    
    def detect_enhanced_color(self, image_region: np.ndarray) -> Dict:
        """Enhanced color detection with validation"""
        
        color_result = {
            'primary_color': 'unknown',
            'secondary_color': 'unknown',
            'confidence': 0.0,
            'color_distribution': {},
            'analysis_method': 'none',
            'issues': []
        }
        
        if image_region.size == 0:
            color_result['issues'].append("Empty image region")
            return color_result
        
        try:
            # Resize for faster processing
            if image_region.shape[0] > 100 or image_region.shape[1] > 100:
                region = cv2.resize(image_region, (100, 100))
            else:
                region = image_region.copy()
            
            # Convert to RGB
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # Method 1: Advanced K-means clustering
            kmeans_result = self._kmeans_color_analysis(region_rgb)
            
            # Method 2: Histogram-based analysis
            histogram_result = self._histogram_color_analysis(region_rgb)
            
            # Method 3: Dominant color analysis
            dominant_result = self._dominant_color_analysis(region_rgb)
            
            # Combine results and select best
            all_results = [kmeans_result, histogram_result, dominant_result]
            valid_results = [r for r in all_results if r['confidence'] > 0.3]
            
            if valid_results:
                # Select result with highest confidence
                best_result = max(valid_results, key=lambda x: x['confidence'])
                color_result.update(best_result)
            else:
                # Fallback to simple average
                fallback_result = self._fallback_color_analysis(region_rgb)
                color_result.update(fallback_result)
                color_result['issues'].append("Low confidence in all methods")
            
            return color_result
            
        except Exception as e:
            color_result['issues'].append(f"Color analysis failed: {str(e)}")
            return color_result
    
    def _kmeans_color_analysis(self, region_rgb: np.ndarray) -> Dict:
        """K-means clustering color analysis"""
        try:
            pixels = region_rgb.reshape(-1, 3)
            
            # Filter out extreme values (shadows/highlights)
            brightness = np.mean(pixels, axis=1)
            valid_pixels = pixels[(brightness > 20) & (brightness < 235)]
            
            if len(valid_pixels) < 20:
                return {'confidence': 0.0}
            
            # Use adaptive number of clusters
            n_clusters = min(5, max(2, len(valid_pixels) // 50))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            
            # Analyze clusters
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            cluster_info = []
            for i, center in enumerate(centers):
                cluster_size = np.sum(labels == i)
                percentage = cluster_size / len(valid_pixels)
                
                if percentage > 0.1:  # At least 10% of pixels
                    color_name = self._rgb_to_fashion_color(center)
                    cluster_info.append((color_name, percentage, center))
            
            if not cluster_info:
                return {'confidence': 0.0}
            
            # Sort by percentage
            cluster_info.sort(key=lambda x: x[1], reverse=True)
            
            primary_color = cluster_info[0][0]
            secondary_color = cluster_info[1][0] if len(cluster_info) > 1 else 'none'
            
            # Calculate confidence based on cluster separation
            confidence = min(cluster_info[0][1] * 2, 1.0)  # Primary cluster dominance
            
            return {
                'primary_color': primary_color,
                'secondary_color': secondary_color,
                'confidence': confidence,
                'analysis_method': 'kmeans',
                'color_distribution': {info[0]: info[1] for info in cluster_info}
            }
            
        except Exception:
            return {'confidence': 0.0}
    
    def _histogram_color_analysis(self, region_rgb: np.ndarray) -> Dict:
        """Histogram-based color analysis"""
        try:
            # Analyze color distribution in HSV space
            region_hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
            
            # Focus on hue channel for color identification
            hue_channel = region_hsv[:, :, 0]
            saturation_channel = region_hsv[:, :, 1]
            value_channel = region_hsv[:, :, 2]
            
            # Filter out low saturation (grayscale) pixels
            high_sat_mask = saturation_channel > 30
            high_val_mask = value_channel > 30
            
            if np.sum(high_sat_mask & high_val_mask) < region_rgb.size * 0.1:
                # Mostly grayscale image
                avg_value = np.mean(value_channel)
                if avg_value < 60:
                    return {
                        'primary_color': 'black',
                        'confidence': 0.8,
                        'analysis_method': 'histogram_grayscale'
                    }
                elif avg_value > 200:
                    return {
                        'primary_color': 'white',
                        'confidence': 0.8,
                        'analysis_method': 'histogram_grayscale'
                    }
                else:
                    return {
                        'primary_color': 'gray',
                        'confidence': 0.7,
                        'analysis_method': 'histogram_grayscale'
                    }
            
            # Analyze hue distribution for colored pixels
            colored_hues = hue_channel[high_sat_mask & high_val_mask]
            
            if len(colored_hues) == 0:
                return {'confidence': 0.0}
            
            # Create hue histogram
            hist, bins = np.histogram(colored_hues, bins=18, range=(0, 180))
            
            # Find dominant hue
            dominant_bin = np.argmax(hist)
            dominant_hue = bins[dominant_bin] + (bins[1] - bins[0]) / 2
            
            # Map hue to color name
            color_name = self._hue_to_color_name(dominant_hue)
            
            # Calculate confidence
            confidence = hist[dominant_bin] / len(colored_hues)
            
            return {
                'primary_color': color_name,
                'confidence': confidence,
                'analysis_method': 'histogram_hsv'
            }
            
        except Exception:
            return {'confidence': 0.0}
    
    def _dominant_color_analysis(self, region_rgb: np.ndarray) -> Dict:
        """Simple dominant color analysis"""
        try:
            # Calculate average color
            avg_color = np.mean(region_rgb.reshape(-1, 3), axis=0)
            
            # Map to fashion color
            color_name = self._rgb_to_fashion_color(avg_color)
            
            # Calculate color uniformity as confidence measure
            pixels = region_rgb.reshape(-1, 3)
            color_distances = np.sqrt(np.sum((pixels - avg_color) ** 2, axis=1))
            uniformity = 1.0 - (np.std(color_distances) / 100.0)
            confidence = max(0.3, min(uniformity, 0.8))
            
            return {
                'primary_color': color_name,
                'confidence': confidence,
                'analysis_method': 'dominant_average'
            }
            
        except Exception:
            return {'confidence': 0.0}
    
    def _fallback_color_analysis(self, region_rgb: np.ndarray) -> Dict:
        """Fallback color analysis when other methods fail"""
        try:
            avg_color = np.mean(region_rgb.reshape(-1, 3), axis=0)
            color_name = self._rgb_to_fashion_color(avg_color)
            
            return {
                'primary_color': color_name,
                'confidence': 0.4,
                'analysis_method': 'fallback'
            }
        except Exception:
            return {
                'primary_color': 'unknown',
                'confidence': 0.0,
                'analysis_method': 'failed'
            }
    
    def _rgb_to_fashion_color(self, rgb: np.ndarray) -> str:
        """Convert RGB to fashion color name"""
        try:
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            
            # Handle grayscale first
            if abs(r - g) < 25 and abs(g - b) < 25 and abs(r - b) < 25:
                avg = (r + g + b) / 3
                if avg < 40:
                    return 'black'
                elif avg > 215:
                    return 'white'
                elif avg < 80:
                    return 'gray'
                elif avg < 140:
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
            
            # Only return if reasonably close (threshold: 80)
            if min_distance < 80:
                return closest_color
            
            # Fallback to basic color analysis
            max_channel = max(r, g, b)
            if max_channel == r and r > g + 30 and r > b + 30:
                return 'red'
            elif max_channel == g and g > r + 30 and g > b + 30:
                return 'green'
            elif max_channel == b and b > r + 30 and b > g + 30:
                return 'blue'
            
            return 'unknown'
            
        except Exception:
            return 'unknown'
    
    def _hue_to_color_name(self, hue: float) -> str:
        """Convert HSV hue to color name"""
        # HSV hue ranges (0-180 in OpenCV)
        if hue < 10 or hue > 170:
            return 'red'
        elif hue < 25:
            return 'orange'
        elif hue < 35:
            return 'yellow'
        elif hue < 85:
            return 'green'
        elif hue < 125:
            return 'blue'
        elif hue < 155:
            return 'purple'
        else:
            return 'pink'
    
    def _update_validation_stats(self, validation_result: Dict):
        """Update validation statistics"""
        self.detection_stats['total_detections'] += 1
        self.detection_stats['quality_scores'].append(validation_result['quality_score'])
        
        if validation_result['quality_score'] < 0.6:
            self.detection_stats['false_positive_indicators'].append(validation_result)
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        stats = self.detection_stats
        
        if stats['total_detections'] == 0:
            return {
                'status': 'no_data',
                'message': 'No detections to analyze'
            }
        
        quality_scores = stats['quality_scores']
        
        report = {
            'total_detections': stats['total_detections'],
            'quality_analysis': {
                'average_quality': np.mean(quality_scores),
                'quality_std': np.std(quality_scores),
                'high_quality_count': sum(1 for q in quality_scores if q > 0.8),
                'medium_quality_count': sum(1 for q in quality_scores if 0.6 <= q <= 0.8),
                'low_quality_count': sum(1 for q in quality_scores if q < 0.6),
                'quality_distribution': {
                    'excellent': sum(1 for q in quality_scores if q > 0.9),
                    'good': sum(1 for q in quality_scores if 0.8 <= q <= 0.9),
                    'fair': sum(1 for q in quality_scores if 0.6 <= q < 0.8),
                    'poor': sum(1 for q in quality_scores if q < 0.6)
                }
            },
            'false_positive_analysis': {
                'suspected_false_positives': len(stats['false_positive_indicators']),
                'false_positive_rate': len(stats['false_positive_indicators']) / stats['total_detections']
            },
            'recommendations': []
        }
        
        # Generate actionable recommendations
        avg_quality = report['quality_analysis']['average_quality']
        fp_rate = report['false_positive_analysis']['false_positive_rate']
        
        if avg_quality < 0.7:
            report['recommendations'].append("⚠️ Low average detection quality - consider tuning detection parameters")
        
        if fp_rate > 0.3:
            report['recommendations'].append("🚨 High false positive rate - review detection validation logic")
        
        if report['quality_analysis']['low_quality_count'] > stats['total_detections'] * 0.4:
            report['recommendations'].append("📊 Many low-quality detections - increase confidence threshold")
        
        return report

# Global validator instance
detection_validator = DetectionValidator() 