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
        self.device = 'cpu'  # Force CPU for compatibility
        
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
                # Load image from URL - handle both catalog formats
                image_url = row.get('shopify_cdn_url', row.get('image_url', ''))
                image = self._load_image_from_url(image_url)
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
        """Generate CLIP embedding for an image with high-quality preprocessing"""
        if not CLIP_AVAILABLE or self.model is None:
            return np.zeros(512)

        with torch.no_grad():
            # HIGH-QUALITY PREPROCESSING PIPELINE
            # 1. Preserve aspect ratio and use smart resizing
            original_size = image.size
            
            # Use larger target size to preserve detail
            target_size = 224  # Match CLIP model expectations
            
            # Smart resize that preserves aspect ratio
            if original_size[0] != original_size[1]:
                # For non-square images, pad to square first
                max_dim = max(original_size)
                padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
                
                # Center the image
                paste_x = (max_dim - original_size[0]) // 2
                paste_y = (max_dim - original_size[1]) // 2
                padded_image.paste(image, (paste_x, paste_y))
                
                # Now resize the square image
                processed_image = padded_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            else:
                # For square images, direct high-quality resize
                processed_image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # 2. Use custom preprocessing instead of default processor
            # Convert to tensor manually for better control
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            
            # Apply transform
            image_tensor = transform(processed_image).unsqueeze(0).to(self.device)
            
            # Generate features
            image_features = self.model.get_image_features(pixel_values=image_tensor)

            # Normalize embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().squeeze()
            
    def match_product(
        self, 
        image: np.ndarray,
        detected_color: str = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Enhanced product matching with color awareness and improved similarity
        
        Args:
            image: Cropped fashion item image (RGB)
            detected_color: Color detected from YOLO (optional)
            top_k: Number of top matches to return
            
        Returns:
            List of product matches with enhanced scoring
        """
        if not CLIP_AVAILABLE or self.faiss_index is None:
            logger.warning("CLIP not available or index not loaded")
            return []
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image.astype('uint8'))
            else:
                image_pil = image
            
            # Generate embedding for the query image
            query_embedding = self._generate_image_embedding(image_pil)
            
            if query_embedding is None:
                return []
            
            # Normalize for cosine similarity
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            similarities, indices = self.faiss_index.search(query_embedding, min(top_k * 2, len(self.catalog)))
            
            matches = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= len(self.catalog):
                    continue
                    
                product = self.catalog.iloc[idx]
                
                # Enhanced similarity calculation with color bonus
                enhanced_similarity = self._calculate_enhanced_similarity(
                    similarity, product, detected_color
                )
                
                # Classify match type using Flickd requirements
                match_type = self._classify_match_type(enhanced_similarity)
                
                # Handle both 'title' (real catalog) and 'product_name' (demo catalog)
                product_name = product.get('title', product.get('product_name', 'Unknown Product'))
                
                match = {
                    'product_id': product['product_id'],
                    'product_name': product_name,
                    'similarity': float(enhanced_similarity),
                    'match_type': match_type,
                    'confidence': float(enhanced_similarity),
                    'rank': i + 1,
                    'category': product.get('category', 'unknown'),
                    'color': product.get('color', 'unknown'),
                    'image_url': product.get('shopify_cdn_url', product.get('image_url', ''))
                }
                
                # Only include matches above minimum threshold
                if enhanced_similarity >= 0.5:  # Minimum viable similarity
                    matches.append(match)
            
            # Sort by enhanced similarity and return top_k
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Error in product matching: {e}")
            return []
    
    def _calculate_enhanced_similarity(
        self, 
        base_similarity: float, 
        product: pd.Series, 
        detected_color: str = None
    ) -> float:
        """
        Calculate enhanced similarity with color and category bonuses
        
        Args:
            base_similarity: Base CLIP similarity score
            product: Product information from catalog
            detected_color: Color detected from the fashion item
            
        Returns:
            Enhanced similarity score
        """
        enhanced_score = base_similarity
        
        # Color matching bonus (up to +0.1)
        if detected_color and detected_color != 'unknown':
            product_color = product.get('color', '').lower()
            detected_color_lower = detected_color.lower()
            
            if product_color == detected_color_lower:
                enhanced_score += 0.1  # Exact color match bonus
            elif self._are_similar_colors(detected_color_lower, product_color):
                enhanced_score += 0.05  # Similar color bonus
        
        # Category consistency bonus (up to +0.05)
        # This would require category detection from YOLO, which we can add
        
        # Popularity/quality bonus based on product metadata
        if hasattr(product, 'rating') and product.get('rating', 0) > 4.0:
            enhanced_score += 0.02
        
        # Ensure score doesn't exceed 1.0
        return min(enhanced_score, 1.0)
    
    def _are_similar_colors(self, color1: str, color2: str) -> bool:
        """Check if two colors are similar"""
        similar_color_groups = [
            ['black', 'dark', 'charcoal'],
            ['white', 'cream', 'ivory', 'beige'],
            ['red', 'crimson', 'burgundy'],
            ['blue', 'navy', 'royal'],
            ['green', 'forest', 'olive'],
            ['pink', 'rose', 'blush'],
            ['brown', 'tan', 'camel'],
            ['gray', 'grey', 'silver']
        ]
        
        for group in similar_color_groups:
            if color1 in group and color2 in group:
                return True
        return False
    
    def _classify_match_type(self, similarity: float) -> str:
        """
        Classify match type according to Flickd requirements
        
        Args:
            similarity: Similarity score (0-1)
            
        Returns:
            Match type: 'exact', 'similar', or 'no_match'
        """
        if similarity > MATCH_THRESHOLD_EXACT:  # > 0.9
            return 'exact'
        elif similarity >= MATCH_THRESHOLD_SIMILAR:  # 0.75-0.9
            return 'similar'
        else:  # < 0.75
            return 'no_match'
    
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