"""Enhanced vibe classification with both text and visual analysis"""
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config import SUPPORTED_VIBES, VIBE_KEYWORDS

logger = logging.getLogger(__name__)


class VibeClassifier:
    """Enhanced vibe classifier with text and visual analysis"""
    
    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.supported_vibes = SUPPORTED_VIBES
        self.vibe_keywords = VIBE_KEYWORDS
        
        # Load spaCy model for text processing
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            logger.warning("spaCy not available. Using basic text processing.")
            self.nlp = None
            
        # Initialize transformer model if requested
        self.transformer_pipeline = None
        if self.use_transformer and TRANSFORMERS_AVAILABLE:
            self._initialize_transformer()
    
    def _initialize_transformer(self):
        """Initialize transformer model for text classification"""
        try:
            self.transformer_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Initialized transformer model for vibe classification")
        except Exception as e:
            logger.error(f"Failed to initialize transformer model: {e}")
            self.transformer_pipeline = None
    
    def classify_vibes(self, text: str, max_vibes: int = 3, confidence_threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Classify vibes from text"""
        if not text or not text.strip():
            return []
            
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        # Get classifications
        rule_based_vibes = self._rule_based_classification(text)
        
        if self.use_transformer and self.transformer_pipeline:
            transformer_vibes = self._transformer_classification(text)
            vibes = self._combine_classifications(rule_based_vibes, transformer_vibes)
        else:
            vibes = rule_based_vibes
            
        # Filter and sort
        vibes = [(vibe, score) for vibe, score in vibes if score >= confidence_threshold]
        vibes = sorted(vibes, key=lambda x: x[1], reverse=True)[:max_vibes]
        
        return vibes
    
    def classify_vibes_from_image(self, image: np.ndarray, max_vibes: int = 3) -> List[str]:
        """
        Classify vibes from image using visual analysis
        
        Args:
            image: Input image as numpy array
            max_vibes: Maximum number of vibes to return
            
        Returns:
            List of detected vibes
        """
        # Analyze visual features
        color_features = self._analyze_colors(image)
        pattern_features = self._analyze_patterns(image)
        style_features = self._analyze_style(image)
        
        # Score each vibe based on visual features
        vibe_scores = {}
        
        for vibe in self.supported_vibes:
            score = self._calculate_visual_vibe_score(vibe, color_features, pattern_features, style_features)
            if score > 0:
                vibe_scores[vibe] = score
        
        # Sort by score and return top vibes
        sorted_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)
        return [vibe for vibe, score in sorted_vibes[:max_vibes] if score > 0.3]
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze color distribution in image"""
        import cv2
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        color_ranges = {
            'pink': [(150, 50, 50), (170, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'black': [(0, 0, 0), (180, 255, 30)],
            'beige': [(15, 30, 100), (25, 100, 200)],
            'brown': [(10, 50, 50), (20, 255, 150)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'red': [(0, 50, 50), (10, 255, 255)],
            'gold': [(20, 100, 100), (30, 255, 255)],
            'silver': [(0, 0, 150), (180, 20, 200)]
        }
        
        # Calculate color percentages
        color_percentages = {}
        total_pixels = image.shape[0] * image.shape[1]
        
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            percentage = np.sum(mask > 0) / total_pixels
            color_percentages[color] = percentage
        
        # Calculate overall brightness and saturation
        brightness = np.mean(hsv[:, :, 2]) / 255.0
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        color_percentages['brightness'] = brightness
        color_percentages['saturation'] = saturation
        
        return color_percentages
    
    def _analyze_patterns(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze patterns and textures in image"""
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features
        features = {}
        
        # Edge density (for pattern complexity)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Variance (for texture)
        features['texture_variance'] = np.var(gray) / 255.0
        
        # Pattern score
        features['pattern_score'] = features['edge_density'] * features['texture_variance']
        
        return features
    
    def _analyze_style(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze overall style features"""
        features = {}
        
        # Calculate image statistics
        features['contrast'] = np.std(image) / 255.0
        features['complexity'] = 1.0 - (np.sum(image == image[0, 0]) / image.size)
        
        return features
    
    def _calculate_visual_vibe_score(self, vibe: str, colors: Dict[str, float], 
                                   patterns: Dict[str, float], style: Dict[str, float]) -> float:
        """Calculate vibe score based on visual features"""
        
        # Define visual characteristics for each vibe
        vibe_characteristics = {
            "Coquette": {
                'colors': {'pink': 0.8, 'white': 0.6, 'beige': 0.4},
                'brightness': (0.7, 1.0),
                'saturation': (0.3, 0.7),
                'pattern_score': (0.0, 0.3)
            },
            "Clean Girl": {
                'colors': {'white': 0.8, 'beige': 0.7, 'brown': 0.5, 'black': 0.3},
                'brightness': (0.6, 0.9),
                'saturation': (0.0, 0.4),
                'pattern_score': (0.0, 0.2)
            },
            "Cottagecore": {
                'colors': {'brown': 0.7, 'beige': 0.6, 'green': 0.5, 'white': 0.4},
                'brightness': (0.5, 0.8),
                'saturation': (0.3, 0.6),
                'pattern_score': (0.2, 0.5)
            },
            "Streetcore": {
                'colors': {'black': 0.9, 'white': 0.5, 'red': 0.4, 'blue': 0.3},
                'brightness': (0.2, 0.6),
                'saturation': (0.0, 0.5),
                'pattern_score': (0.1, 0.4)
            },
            "Y2K": {
                'colors': {'silver': 0.7, 'pink': 0.6, 'blue': 0.5, 'white': 0.4},
                'brightness': (0.6, 1.0),
                'saturation': (0.5, 1.0),
                'pattern_score': (0.2, 0.6)
            },
            "Boho": {
                'colors': {'brown': 0.8, 'beige': 0.7, 'red': 0.4, 'gold': 0.3},
                'brightness': (0.4, 0.7),
                'saturation': (0.4, 0.8),
                'pattern_score': (0.3, 0.7)
            },
            "Party Glam": {
                'colors': {'gold': 0.8, 'silver': 0.7, 'black': 0.6, 'red': 0.5},
                'brightness': (0.5, 1.0),
                'saturation': (0.6, 1.0),
                'pattern_score': (0.1, 0.5)
            }
        }
        
        if vibe not in vibe_characteristics:
            return 0.0
        
        chars = vibe_characteristics[vibe]
        
        # Score based on color matches
        color_score = 0.0
        for color, weight in chars.get('colors', {}).items():
            color_score += colors.get(color, 0) * weight
        
        # Score based on brightness and saturation ranges
        brightness = colors.get('brightness', 0.5)
        saturation = colors.get('saturation', 0.5)
        
        bright_range = chars.get('brightness', (0, 1))
        sat_range = chars.get('saturation', (0, 1))
        
        brightness_score = 1.0 if bright_range[0] <= brightness <= bright_range[1] else 0.0
        saturation_score = 1.0 if sat_range[0] <= saturation <= sat_range[1] else 0.0
        
        # Score based on pattern
        pattern_range = chars.get('pattern_score', (0, 1))
        pattern_val = patterns.get('pattern_score', 0)
        pattern_score = 1.0 if pattern_range[0] <= pattern_val <= pattern_range[1] else 0.0
        
        # Combine scores
        total_score = (color_score * 0.5 + 
                      brightness_score * 0.2 + 
                      saturation_score * 0.2 + 
                      pattern_score * 0.1)
        
        return total_score
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for classification"""
        text = text.lower()
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', text)
        hashtag_text = ' '.join([tag[1:] for tag in hashtags])
        
        # Clean text
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return f"{text} {hashtag_text}".strip()
    
    def _rule_based_classification(self, text: str) -> List[Tuple[str, float]]:
        """Rule-based vibe classification using keywords"""
        vibe_scores = {}
        tokens = text.split()
        
        for vibe, keywords in self.vibe_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in tokens:
                    score += 1.0
                elif any(keyword in token for token in tokens):
                    score += 0.5
                    
            if score > 0:
                normalized_score = min(1.0, score / len(keywords))
                vibe_scores[vibe] = normalized_score
                
        return [(vibe, score) for vibe, score in vibe_scores.items()]
    
    def _transformer_classification(self, text: str) -> List[Tuple[str, float]]:
        """Transformer-based vibe classification"""
        if not self.transformer_pipeline:
            return []
            
        try:
            candidate_labels = list(self.supported_vibes)
            result = self.transformer_pipeline(
                text,
                candidate_labels=candidate_labels,
                multi_label=True
            )
            
            return list(zip(result['labels'], result['scores']))
            
        except Exception as e:
            logger.error(f"Transformer classification failed: {e}")
            return []
    
    def _combine_classifications(self, rule_based: List[Tuple[str, float]], 
                               transformer: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Combine rule-based and transformer classifications"""
        rule_dict = dict(rule_based)
        trans_dict = dict(transformer)
        
        combined = {}
        all_vibes = set(rule_dict.keys()) | set(trans_dict.keys())
        
        for vibe in all_vibes:
            rule_score = rule_dict.get(vibe, 0)
            trans_score = trans_dict.get(vibe, 0)
            
            if trans_score > 0:
                combined_score = 0.3 * rule_score + 0.7 * trans_score
            else:
                combined_score = rule_score
                
            combined[vibe] = combined_score
            
        return [(vibe, score) for vibe, score in combined.items()] 