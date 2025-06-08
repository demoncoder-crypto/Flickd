#!/usr/bin/env python3
"""Debug vibe classification"""

import logging
logging.basicConfig(level=logging.DEBUG)

from models.vibe_classifier import VibeClassifier

# Test with sample audio transcript
test_texts = [
    "Peace out",
    "You said me free Summer love in you, let's have love Keep her back, got a lot of love",
    "Hi my name is Zeneca Zuecian, I am a doubts syndrome advocate, model and co-founder",
    "🎵",
    "Music",
    "fashion style outfit dress beautiful"
]

classifier = VibeClassifier(use_transformer=True, use_audio=True)

for text in test_texts:
    print(f"\n--- Testing text: '{text}' ---")
    vibes = classifier.classify_vibes(text, max_vibes=3, confidence_threshold=0.1)
    print(f"Detected vibes: {vibes}")
    
    # Test individual methods
    preprocessed = classifier._preprocess_text(text)
    print(f"Preprocessed: '{preprocessed}'")
    
    rule_based = classifier._rule_based_classification(preprocessed)
    print(f"Rule-based: {rule_based}")
    
    semantic = classifier._semantic_classification(preprocessed)
    print(f"Semantic: {semantic}")
    
    if classifier.transformer_pipeline:
        transformer = classifier._transformer_classification(preprocessed)
        print(f"Transformer: {transformer}") 