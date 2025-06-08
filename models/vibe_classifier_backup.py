"""NLP-based vibe classification for fashion content"""
import re
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    
from typing import List, Dict, Optional, Tuple
import logging
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
import numpy as np
from collections import Counter

from config import SUPPORTED_VIBES, VIBE_KEYWORDS

logger = logging.getLogger(__name__)


class VibeClassifier:
    """Classify fashion vibes from text using NLP"""
    
    def __init__(self, use_transformer: bool = True):
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
            # Use a zero-shot classification model
            self.transformer_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Initialized transformer model for vibe classification")
        except Exception as e:
            logger.error(f"Failed to initialize transformer model: {e}")
            self.transformer_pipeline = None
            
    def classify_vibes(
        self, 
        text: str, 
        max_vibes: int = 3,
        confidence_threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Classify vibes from text
        
        Args:
            text: Input text (caption, hashtags, transcript)
            max_vibes: Maximum number of vibes to return
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of (vibe, confidence) tuples
        """
        if not text or not text.strip():
            return []
            
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        # Get classifications from both methods
        rule_based_vibes = self._rule_based_classification(text)
        
        if self.use_transformer and self.transformer_pipeline:
            transformer_vibes = self._transformer_classification(text)
            # Combine results
            vibes = self._combine_classifications(rule_based_vibes, transformer_vibes)
        else:
            vibes = rule_based_vibes
            
        # Filter by confidence and limit number
        vibes = [(vibe, score) for vibe, score in vibes if score >= confidence_threshold]
        vibes = sorted(vibes, key=lambda x: x[1], reverse=True)[:max_vibes]
        
        return vibes
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for classification"""
        # Convert to lowercase
        text = text.lower()
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', text)
        hashtag_text = ' '.join([tag[1:] for tag in hashtags])  # Remove # symbol
        
        # Clean text
        text = re.sub(r'#\w+', '', text)  # Remove hashtags from main text
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        # Combine cleaned text with hashtags
        full_text = f"{text} {hashtag_text}".strip()
        
        return full_text
    
    def _rule_based_classification(self, text: str) -> List[Tuple[str, float]]:
        """Rule-based vibe classification using keywords"""
        vibe_scores = {}
        
        # Tokenize text
        tokens = text.split()
        
        # Count keyword matches for each vibe
        for vibe, keywords in self.vibe_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                # Check for exact matches
                if keyword in tokens:
                    score += 1.0
                    matched_keywords.append(keyword)
                # Check for partial matches
                elif any(keyword in token for token in tokens):
                    score += 0.5
                    matched_keywords.append(keyword)
                    
            # Normalize score
            if score > 0:
                normalized_score = min(1.0, score / len(keywords))
                vibe_scores[vibe] = normalized_score
                
                if matched_keywords:
                    logger.debug(f"Vibe '{vibe}' matched keywords: {matched_keywords}")
                    
        # Convert to list of tuples
        results = [(vibe, score) for vibe, score in vibe_scores.items()]
        return results
    
    def _transformer_classification(self, text: str) -> List[Tuple[str, float]]:
        """Transformer-based vibe classification"""
        if not self.transformer_pipeline:
            return []
            
        try:
            # Prepare candidate labels with descriptions
            candidate_labels = []
            label_mapping = {}
            
            for vibe in self.supported_vibes:
                # Create descriptive labels for better zero-shot performance
                descriptions = self._get_vibe_descriptions(vibe)
                for desc in descriptions:
                    candidate_labels.append(desc)
                    label_mapping[desc] = vibe
                    
            # Run zero-shot classification
            result = self.transformer_pipeline(
                text,
                candidate_labels=candidate_labels,
                multi_label=True
            )
            
            # Aggregate scores by vibe
            vibe_scores = {}
            for label, score in zip(result['labels'], result['scores']):
                vibe = label_mapping[label]
                if vibe not in vibe_scores:
                    vibe_scores[vibe] = []
                vibe_scores[vibe].append(score)
                
            # Average scores for each vibe
            results = []
            for vibe, scores in vibe_scores.items():
                avg_score = np.mean(scores)
                results.append((vibe, float(avg_score)))
                
            return results
            
        except Exception as e:
            logger.error(f"Transformer classification failed: {e}")
            return []
            
    def _get_vibe_descriptions(self, vibe: str) -> List[str]:
        """Get descriptive labels for each vibe"""
        descriptions = {
            "Coquette": [
                "coquette fashion style",
                "feminine and romantic aesthetic",
                "soft pink and pastel colors with bows and ribbons"
            ],
            "Clean Girl": [
                "clean girl aesthetic",
                "minimal and effortless style",
                "natural and fresh look with neutral colors"
            ],
            "Cottagecore": [
                "cottagecore aesthetic",
                "rural and vintage inspired fashion",
                "floral patterns and countryside style"
            ],
            "Streetcore": [
                "streetwear fashion",
                "urban and edgy style",
                "street style with grunge elements"
            ],
            "Y2K": [
                "Y2K fashion style",
                "2000s inspired aesthetic",
                "retro futuristic with metallic and cyber elements"
            ],
            "Boho": [
                "bohemian fashion style",
                "boho chic aesthetic",
                "free-spirited with ethnic and tribal patterns"
            ],
            "Party Glam": [
                "party glamour style",
                "glamorous evening wear",
                "sparkly and elegant party fashion"
            ]
        }
        
        return descriptions.get(vibe, [vibe])
    
    def _combine_classifications(
        self, 
        rule_based: List[Tuple[str, float]], 
        transformer: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Combine rule-based and transformer classifications"""
        # Convert to dictionaries
        rule_dict = dict(rule_based)
        trans_dict = dict(transformer)
        
        # Combine scores with weights
        combined = {}
        all_vibes = set(rule_dict.keys()) | set(trans_dict.keys())
        
        for vibe in all_vibes:
            rule_score = rule_dict.get(vibe, 0)
            trans_score = trans_dict.get(vibe, 0)
            
            # Weighted average (giving more weight to transformer if available)
            if trans_score > 0:
                combined_score = 0.3 * rule_score + 0.7 * trans_score
            else:
                combined_score = rule_score
                
            combined[vibe] = combined_score
            
        # Convert back to list of tuples
        return [(vibe, score) for vibe, score in combined.items()]
    
    def extract_fashion_terms(self, text: str) -> List[str]:
        """Extract fashion-related terms from text"""
        if not self.nlp:
            return []
            
        doc = self.nlp(text.lower())
        
        # Fashion-related terms to look for
        fashion_terms = []
        
        # Extract nouns and adjectives that might be fashion-related
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 2:
                # Check if it's fashion-related
                if any(keyword in token.text for vibe_keywords in self.vibe_keywords.values() 
                      for keyword in vibe_keywords):
                    fashion_terms.append(token.text)
                    
        # Extract named entities that might be brands
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                fashion_terms.append(ent.text)
                
        return list(set(fashion_terms))
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the text"""
        if not self.nlp:
            return {"positive": 0.5, "negative": 0.5, "neutral": 0.0}
            
        doc = self.nlp(text)
        
        # Simple sentiment analysis based on adjectives
        positive_words = ['beautiful', 'amazing', 'love', 'perfect', 'gorgeous', 'stunning', 'cute', 'pretty']
        negative_words = ['ugly', 'bad', 'hate', 'awful', 'terrible', 'boring', 'basic']
        
        positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
        negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
        total_count = len(doc)
        
        if total_count == 0:
            return {"positive": 0.5, "negative": 0.5, "neutral": 0.0}
            
        positive_score = positive_count / total_count
        negative_score = negative_count / total_count
        neutral_score = 1.0 - positive_score - negative_score
        
        return {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": neutral_score
        } 