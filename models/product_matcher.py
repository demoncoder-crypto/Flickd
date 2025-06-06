"""CLIP-based product matching with FAISS similarity search"""
import numpy as np
import torch
import faiss
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import pandas as pd
import pickle

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    CLIPModel = Any  # Type hint placeholder
    CLIPProcessor = Any  # Type hint placeholder
    
import requests
from io import BytesIO
from tqdm import tqdm

from config import (
    CLIP_MODEL_NAME,
    CLIP_DEVICE,
    FAISS_INDEX_PATH,
    MATCH_THRESHOLD_EXACT,
    MATCH_THRESHOLD_SIMILAR,
    PRODUCT_CATALOG_PATH,
    DATA_DIR
)

logger = logging.getLogger(__name__)


class ProductMatcher:
    """CLIP + FAISS based product matching system"""
    
    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        device: Optional[str] = None,
        catalog_path: Optional[Path] = None
    ):
        self.model_name = model_name
        self.device = device or CLIP_DEVICE
        
        # Initialize CLIP model if available
        if CLIP_AVAILABLE:
            self.model, self.processor = self._load_clip_model()
        else:
            self.model = None
            self.processor = None
            logger.warning("Running without CLIP models - product matching disabled")
        
        # Load product catalog
        self.catalog_path = catalog_path or PRODUCT_CATALOG_PATH
        self.catalog = None
        self.product_embeddings = None
        self.faiss_index = None
        
        # Load or create index only if CLIP is available
        if CLIP_AVAILABLE:
            self._initialize_index()
        
    def _load_clip_model(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Load CLIP model and processor"""
        if not CLIP_AVAILABLE:
            return None, None
            
        logger.info(f"Loading CLIP model: {self.model_name}")
        
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(self.model_name)
        processor = CLIPProcessor.from_pretrained(self.model_name)
        
        model = model.to(self.device)
        model.eval()
        
        return model, processor
    
    def _initialize_index(self):
        """Initialize FAISS index with product embeddings"""
        index_path = Path(FAISS_INDEX_PATH)
        embeddings_path = index_path.with_suffix('.pkl')
        
        if index_path.exists() and embeddings_path.exists():
            # Load existing index
            logger.info("Loading existing FAISS index and embeddings...")
            self.faiss_index = faiss.read_index(str(index_path))
            
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.catalog = data['catalog']
                self.product_embeddings = data['embeddings']
        else:
            # Create new index
            logger.info("Creating new FAISS index...")
            self._build_index()
    
    def _build_index(self):
        """Build FAISS index from product catalog"""
        if not CLIP_AVAILABLE:
            logger.warning("Cannot build index without CLIP models")
            return
            
        # Load product catalog
        if not self.catalog_path.exists():
            logger.warning(f"Product catalog not found at {self.catalog_path}")
            # Create dummy catalog for testing
            self._create_dummy_catalog()
            
        self.catalog = pd.read_csv(self.catalog_path)
        logger.info(f"Loaded {len(self.catalog)} products from catalog")
        
        # Generate embeddings for all products
        embeddings = []
        
        for idx, row in tqdm(self.catalog.iterrows(), total=len(self.catalog), desc="Generating product embeddings"):
            try:
                # Load image from URL
                image = self._load_image_from_url(row['image_url'])
                if image is None:
                    # Use placeholder embedding
                    embedding = np.zeros(512)  # CLIP embedding dimension
                else:
                    # Generate embedding
                    embedding = self._generate_image_embedding(image)
                    
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error processing product {row['product_id']}: {e}")
                embeddings.append(np.zeros(512))
                
        self.product_embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = self.product_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.product_embeddings)
        self.faiss_index.add(self.product_embeddings)
        
        # Save index and embeddings
        self._save_index()
        
    def _save_index(self):
        """Save FAISS index and embeddings to disk"""
        index_path = Path(FAISS_INDEX_PATH)
        embeddings_path = index_path.with_suffix('.pkl')
        
        # Create directory if needed
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(index_path))
        
        # Save embeddings and catalog
        with open(embeddings_path, 'wb') as f:
            pickle.dump({
                'catalog': self.catalog,
                'embeddings': self.product_embeddings
            }, f)
            
        logger.info(f"Saved FAISS index to {index_path}")
        
    def _create_dummy_catalog(self):
        """Create dummy product catalog for testing"""
        dummy_products = [
            {
                'product_id': 'prod_001',
                'product_name': 'White Summer Dress',
                'image_url': 'https://via.placeholder.com/300x400/ffffff/000000?text=White+Dress',
                'category': 'dress',
                'color': 'white'
            },
            {
                'product_id': 'prod_002',
                'product_name': 'Black Leather Jacket',
                'image_url': 'https://via.placeholder.com/300x400/000000/ffffff?text=Black+Jacket',
                'category': 'jacket',
                'color': 'black'
            },
            {
                'product_id': 'prod_003',
                'product_name': 'Blue Denim Jeans',
                'image_url': 'https://via.placeholder.com/300x400/4169e1/ffffff?text=Blue+Jeans',
                'category': 'bottom',
                'color': 'blue'
            },
            {
                'product_id': 'prod_004',
                'product_name': 'Pink Handbag',
                'image_url': 'https://via.placeholder.com/300x400/ffc0cb/000000?text=Pink+Bag',
                'category': 'bag',
                'color': 'pink'
            },
            {
                'product_id': 'prod_005',
                'product_name': 'Floral Print Top',
                'image_url': 'https://via.placeholder.com/300x400/98fb98/000000?text=Floral+Top',
                'category': 'top',
                'color': 'multicolor'
            }
        ]
        
        df = pd.DataFrame(dummy_products)
        df.to_csv(self.catalog_path, index=False)
        logger.info(f"Created dummy catalog with {len(df)} products")
        
    def _load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Load image from URL"""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image from {url}: {e}")
            return None
            
    def _generate_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate CLIP embedding for an image"""
        if not CLIP_AVAILABLE or self.model is None:
            return np.zeros(512)
            
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().squeeze()
            
    def match_product(
        self, 
        image: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Match detected fashion item to products in catalog
        
        Args:
            image: Cropped fashion item image (numpy array)
            top_k: Number of top matches to return
            
        Returns:
            List of matches with format:
            {
                'product_id': str,
                'product_name': str,
                'similarity': float,
                'match_type': str ('exact', 'similar', 'no_match'),
                'rank': int
            }
        """
        if not CLIP_AVAILABLE or self.faiss_index is None:
            logger.warning("Product matching not available without CLIP models")
            return []
            
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Generate embedding
        query_embedding = self._generate_image_embedding(pil_image)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Process results
        matches = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No match found
                continue
                
            similarity = float(dist)  # Cosine similarity
            product = self.catalog.iloc[idx]
            
            # Determine match type
            if similarity >= MATCH_THRESHOLD_EXACT:
                match_type = "exact"
            elif similarity >= MATCH_THRESHOLD_SIMILAR:
                match_type = "similar"
            else:
                match_type = "no_match"
                
            match = {
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'similarity': similarity,
                'match_type': match_type,
                'rank': rank + 1,
                'category': product.get('category', 'unknown'),
                'color': product.get('color', 'unknown'),
                'image_url': product.get('image_url', '')
            }
            
            matches.append(match)
            
        return matches
    
    def update_catalog(self, new_catalog_path: Path):
        """Update product catalog and rebuild index"""
        logger.info(f"Updating catalog from {new_catalog_path}")
        self.catalog_path = new_catalog_path
        self._build_index()
        
    def add_products(self, products: List[Dict]):
        """Add new products to the catalog"""
        if not CLIP_AVAILABLE:
            logger.warning("Cannot add products without CLIP models")
            return
            
        # Convert to DataFrame
        new_products_df = pd.DataFrame(products)
        
        # Append to existing catalog
        self.catalog = pd.concat([self.catalog, new_products_df], ignore_index=True)
        
        # Generate embeddings for new products
        new_embeddings = []
        for product in products:
            image = self._load_image_from_url(product['image_url'])
            if image:
                embedding = self._generate_image_embedding(image)
            else:
                embedding = np.zeros(512)
            new_embeddings.append(embedding)
            
        new_embeddings = np.array(new_embeddings).astype('float32')
        faiss.normalize_L2(new_embeddings)
        
        # Add to FAISS index
        self.faiss_index.add(new_embeddings)
        
        # Update stored embeddings
        self.product_embeddings = np.vstack([self.product_embeddings, new_embeddings])
        
        # Save updated index
        self._save_index()
        
        logger.info(f"Added {len(products)} new products to catalog") 