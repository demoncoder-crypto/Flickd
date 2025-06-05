"""Utility script to update product catalog and rebuild embeddings"""
import argparse
import pandas as pd
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.product_matcher import ProductMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_catalog(df: pd.DataFrame) -> bool:
    """Validate catalog has required columns"""
    required_columns = ['product_id', 'product_name', 'image_url']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for duplicates
    duplicates = df[df.duplicated(subset=['product_id'], keep=False)]
    if not duplicates.empty:
        logger.warning(f"Found {len(duplicates)} duplicate product IDs")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Update Flickd product catalog')
    parser.add_argument('--csv', type=str, required=True, help='Path to product catalog CSV')
    parser.add_argument('--validate-only', action='store_true', help='Only validate, don\'t update')
    parser.add_argument('--sample', type=int, help='Only process first N products (for testing)')
    
    args = parser.parse_args()
    
    # Check if file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return 1
    
    try:
        # Load CSV
        logger.info(f"Loading catalog from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Sample if requested
        if args.sample:
            df = df.head(args.sample)
            logger.info(f"Using first {args.sample} products for testing")
        
        logger.info(f"Loaded {len(df)} products")
        
        # Validate
        if not validate_catalog(df):
            return 1
        
        # Print summary
        print("\nCatalog Summary:")
        print(f"Total products: {len(df)}")
        
        if 'category' in df.columns:
            print("\nCategories:")
            for category, count in df['category'].value_counts().items():
                print(f"  {category}: {count}")
        
        if 'color' in df.columns:
            print("\nTop colors:")
            for color, count in df['color'].value_counts().head(10).items():
                print(f"  {color}: {count}")
        
        if args.validate_only:
            print("\nValidation complete. Use without --validate-only to update catalog.")
            return 0
        
        # Update catalog
        print("\nUpdating product catalog and building embeddings...")
        print("This may take a while depending on the number of products...")
        
        matcher = ProductMatcher()
        matcher.catalog_path = csv_path
        matcher._build_index()
        
        print("\nCatalog updated successfully!")
        print(f"Index saved to: {matcher.faiss_index}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to update catalog: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 