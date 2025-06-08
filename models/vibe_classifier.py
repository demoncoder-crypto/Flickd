"""Enhanced vibe classification with audio, text and visual analysis"""
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import subprocess
import tempfile

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

import whisper

from config import SUPPORTED_VIBES, VIBE_KEYWORDS

logger = logging.getLogger(__name__)


class VibeClassifier:
    """Enhanced vibe classifier with audio transcription, text and visual analysis"""
    
    def __init__(self, use_transformer: bool = True, use_audio: bool = True):
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.use_audio = use_audio
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
            
        # Initialize Whisper model for audio transcription
        self.whisper_model = None
        if self.use_audio:
            self._initialize_whisper()
    
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
    
    def _initialize_whisper(self):
        """Initialize Whisper model for audio transcription"""
        try:
            # Use base model for good balance of speed/accuracy
            self.whisper_model = whisper.load_model("base")
            logger.info("Initialized Whisper model for audio transcription")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            self.whisper_model = None
    
    def classify_vibes_from_video(
        self, 
        video_path: str, 
        caption: str = None, 
        hashtags: List[str] = None,
        max_vibes: int = 3,
        video_frame: np.ndarray = None
    ) -> List[Tuple[str, float]]:
        """
        Enhanced vibe classification from video using audio, text, and visual analysis
        
        Args:
            video_path: Path to video file
            caption: Optional video caption
            hashtags: Optional list of hashtags
            max_vibes: Maximum number of vibes to return
            video_frame: Optional video frame for visual analysis
            
        Returns:
            List of (vibe, confidence) tuples
        """
        all_text = []
        
        # Add caption if provided
        if caption:
            all_text.append(caption)
        
        # Add hashtags if provided
        if hashtags:
            hashtag_text = " ".join(hashtags)
            all_text.append(hashtag_text)
        
        # Extract audio and transcribe if Whisper is available
        if self.use_audio and self.whisper_model:
            try:
                audio_transcript = self._transcribe_video_audio(video_path)
                if audio_transcript:
                    all_text.append(audio_transcript)
                    logger.info(f"Audio transcript: {audio_transcript[:100]}...")
            except Exception as e:
                logger.warning(f"Audio transcription failed: {e}")
        
        # Combine all text sources
        combined_text = " ".join(all_text)
        
        # Get text-based vibes with very low threshold for multi-vibe detection
        text_vibes = []
        if combined_text.strip():
            text_vibes = self.classify_vibes(combined_text, max_vibes=max_vibes, confidence_threshold=0.05)
        
        # Get visual vibes if frame is provided
        visual_vibes = []
        if video_frame is not None:
            try:
                visual_vibe_names = self.classify_vibes_from_image(video_frame, max_vibes=max_vibes)
                # Convert to (vibe, confidence) format with moderate confidence
                visual_vibes = [(vibe, 0.6) for vibe in visual_vibe_names]
            except Exception as e:
                logger.warning(f"Visual vibe classification failed: {e}")
        
        # Combine text and visual vibes
        combined_vibes = self._combine_text_and_visual_vibes(text_vibes, visual_vibes, max_vibes)
        
        # If still no vibes, use fallback detection
        if not combined_vibes:
            logger.info("No vibes detected, using fallback detection")
            combined_vibes = self._fallback_vibe_detection(combined_text, max_vibes)
        
        return combined_vibes
    
    def _transcribe_video_audio(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video and transcribe using Whisper
        
        Args:
            video_path: Path to video file
            
        Returns:
            Transcribed text or None if failed
        """
        if not self.whisper_model:
            return None
        
        try:
            # Extract audio using ffmpeg to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg", "-i", video_path, 
                "-vn", "-acodec", "pcm_s16le", 
                "-ar", "16000", "-ac", "1", 
                temp_audio_path, "-y"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg failed: {result.stderr}")
                return None
            
            # Transcribe audio with Whisper
            result = self.whisper_model.transcribe(temp_audio_path)
            transcript = result.get("text", "").strip()
            
            # Clean up temporary file
            Path(temp_audio_path).unlink(missing_ok=True)
            
            return transcript if transcript else None
            
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return None
    
    def classify_vibes(self, text: str, max_vibes: int = 3, confidence_threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Enhanced vibe classification from text with multiple methods"""
        if not text or not text.strip():
            logger.debug("Empty text provided to classify_vibes")
            return []
            
        # Clean and preprocess text
        original_text = text
        text = self._preprocess_text(text)
        logger.debug(f"Original text: '{original_text[:100]}...'")
        logger.debug(f"Preprocessed text: '{text[:100]}...'")
        
        if not text or not text.strip():
            logger.warning("Text became empty after preprocessing")
            return []
        
        # Get classifications from multiple methods
        rule_based_vibes = self._rule_based_classification(text)
        semantic_vibes = self._semantic_classification(text)
        
        if self.use_transformer and self.transformer_pipeline:
            transformer_vibes = self._transformer_classification(text)
            vibes = self._combine_multiple_classifications(rule_based_vibes, semantic_vibes, transformer_vibes)
        else:
            # Combine rule-based and semantic without transformer
            combined_scores = {}
            
            # Combine rule-based and semantic scores
            for vibe, score in rule_based_vibes:
                combined_scores[vibe] = score * 0.6  # Weight rule-based higher when no transformer
            
            for vibe, score in semantic_vibes:
                if vibe in combined_scores:
                    combined_scores[vibe] += score * 0.4
                else:
                    combined_scores[vibe] = score * 0.4
            
            vibes = list(combined_scores.items())
            
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
        
        # Check if text is valid and not empty
        if not text or not text.strip():
            logger.warning("Empty text provided to transformer classification")
            return []
            
        try:
            candidate_labels = list(self.supported_vibes)
            
            # Ensure we have valid text and labels
            if len(candidate_labels) == 0:
                logger.warning("No candidate labels available")
                return []
            
            result = self.transformer_pipeline(
                text.strip(),
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
    
    def _semantic_classification(self, text: str) -> List[Tuple[str, float]]:
        """Enhanced semantic analysis using word embeddings and context"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        vibe_scores = {}
        
        # Enhanced keyword matching with context
        for vibe, keywords in self.vibe_keywords.items():
            score = 0.0
            
            # Direct keyword matching with weights
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    score += 1.0
            
            # Semantic similarity using spaCy word vectors
            if doc.has_vector:
                for token in doc:
                    if token.has_vector and not token.is_stop and not token.is_punct:
                        for keyword in keywords:
                            keyword_doc = self.nlp(keyword)
                            if keyword_doc.has_vector:
                                similarity = token.similarity(keyword_doc[0])
                                if similarity > 0.6:  # High similarity threshold
                                    score += similarity * 0.5
            
            # Context-based scoring
            score += self._analyze_context_for_vibe(doc, vibe)
            
            # Normalize score
            if score > 0:
                vibe_scores[vibe] = min(score / len(keywords), 1.0)
        
        return list(vibe_scores.items())
    
    def _analyze_context_for_vibe(self, doc, vibe: str) -> float:
        """Analyze context and sentiment for specific vibe"""
        context_score = 0.0
        
        # Analyze sentiment and mood
        positive_words = ["beautiful", "gorgeous", "stunning", "amazing", "perfect", "love"]
        negative_words = ["ugly", "bad", "terrible", "awful", "hate"]
        
        text_lower = doc.text.lower()
        
        # Positive sentiment bonus
        for word in positive_words:
            if word in text_lower:
                context_score += 0.1
        
        # Negative sentiment penalty
        for word in negative_words:
            if word in text_lower:
                context_score -= 0.2
        
        # Vibe-specific context analysis
        if vibe == "Coquette":
            romantic_words = ["romantic", "feminine", "soft", "delicate", "sweet"]
            for word in romantic_words:
                if word in text_lower:
                    context_score += 0.15
        
        elif vibe == "Clean Girl":
            minimal_words = ["minimal", "simple", "natural", "effortless", "clean"]
            for word in minimal_words:
                if word in text_lower:
                    context_score += 0.15
        
        elif vibe == "Streetcore":
            urban_words = ["street", "urban", "city", "edgy", "cool"]
            for word in urban_words:
                if word in text_lower:
                    context_score += 0.15
        
        return context_score
    
    def _combine_multiple_classifications(
        self, 
        rule_based: List[Tuple[str, float]], 
        semantic: List[Tuple[str, float]],
        transformer: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Combine three classification methods with weighted averaging"""
        combined_scores = {}
        
        # Weights for different methods
        weights = {
            'rule_based': 0.4,
            'semantic': 0.3,
            'transformer': 0.3
        }
        
        # Combine scores
        for vibe in self.supported_vibes:
            total_score = 0.0
            
            # Rule-based score
            rule_score = next((score for v, score in rule_based if v == vibe), 0.0)
            total_score += rule_score * weights['rule_based']
            
            # Semantic score
            semantic_score = next((score for v, score in semantic if v == vibe), 0.0)
            total_score += semantic_score * weights['semantic']
            
            # Transformer score
            transformer_score = next((score for v, score in transformer if v == vibe), 0.0)
            total_score += transformer_score * weights['transformer']
            
            if total_score > 0:
                combined_scores[vibe] = total_score
        
        return list(combined_scores.items()) 
    
    def _combine_text_and_visual_vibes(
        self, 
        text_vibes: List[Tuple[str, float]], 
        visual_vibes: List[Tuple[str, float]], 
        max_vibes: int
    ) -> List[Tuple[str, float]]:
        """Combine text and visual vibe classifications"""
        combined_scores = {}
        
        # Add text vibes with higher weight
        for vibe, score in text_vibes:
            combined_scores[vibe] = score * 0.7
        
        # Add visual vibes with lower weight, but boost if already detected
        for vibe, score in visual_vibes:
            if vibe in combined_scores:
                # Boost existing vibes
                combined_scores[vibe] += score * 0.5
            else:
                # Add new visual vibes with moderate confidence
                combined_scores[vibe] = score * 0.3
        
        # Sort and return top vibes
        sorted_vibes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_vibes[:max_vibes]
    
    def _fallback_vibe_detection(self, text: str, max_vibes: int) -> List[Tuple[str, float]]:
        """Fallback vibe detection when no vibes are found"""
        # Default vibes based on common fashion content
        fallback_vibes = [
            ("Streetcore", 0.4),  # Most common in fashion videos
            ("Clean Girl", 0.3),  # Very popular trend
            ("Coquette", 0.25)    # Growing trend
        ]
        
        # If we have text, try to match at least one keyword
        if text:
            text_lower = text.lower()
            
            # Check for any fashion-related keywords
            if any(word in text_lower for word in ["outfit", "style", "fashion", "look", "wear"]):
                # Boost confidence if fashion-related
                fallback_vibes = [(vibe, score + 0.2) for vibe, score in fallback_vibes]
            
            # Check for specific vibe indicators
            if any(word in text_lower for word in ["street", "urban", "cool", "edgy"]):
                fallback_vibes[0] = ("Streetcore", 0.7)
            elif any(word in text_lower for word in ["clean", "minimal", "simple", "natural"]):
                fallback_vibes[0] = ("Clean Girl", 0.7)
            elif any(word in text_lower for word in ["cute", "sweet", "feminine", "soft"]):
                fallback_vibes[0] = ("Coquette", 0.7)
        
        return fallback_vibes[:max_vibes] 