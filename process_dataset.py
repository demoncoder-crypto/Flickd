#!/usr/bin/env python3
"""
Process hackathon dataset files and prepare for video analysis
Converts Excel to CSV and creates proper catalog format
"""
import pandas as pd
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_excel_to_catalog():
    """Convert product_data.xlsx to required catalog.csv format"""
    
    # File paths
    excel_path = Path("data/product_data.xlsx")
    images_path = Path("data/images.csv")
    output_path = Path("data/catalog.csv")
    
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        logger.info("Please save your product_data.xlsx to: data/product_data.xlsx")
        return False
    
    logger.info(f"Loading Excel file: {excel_path}")
    
    try:
        # Read Excel file
        df_products = pd.read_excel(excel_path)
        logger.info(f"Loaded {len(df_products)} products from Excel")
        
        # Read images CSV
        df_images = pd.read_csv(images_path)
        logger.info(f"Loaded {len(df_images)} image URLs")
        
        # Show first few columns to understand structure
        logger.info("Excel columns: " + ", ".join(df_products.columns.tolist()))
        logger.info("Sample data:")
        print(df_products.head())
        
        # Merge with images (assuming 'id' column exists in Excel)
        if 'id' in df_products.columns:
            # Merge products with first image for each product
            df_first_images = df_images.groupby('id').first().reset_index()
            df_merged = df_products.merge(df_first_images, on='id', how='left')
        else:
            logger.warning("No 'id' column found in Excel. Using row index.")
            # If no ID column, create mapping based on order
            df_products['id'] = range(len(df_products))
            df_first_images = df_images.groupby('id').first().reset_index()
            df_merged = df_products.merge(df_first_images, on='id', how='left')
        
        # Create catalog in required format
        catalog_data = []
        
        for idx, row in df_merged.iterrows():
            # Map your columns to required format
            product_entry = {
                'product_id': f"prod_{row.get('id', idx):03d}",
                'title': row.get('title', row.get('name', f"Product {idx}")),
                'shopify_cdn_url': row.get('image_url', ''),
                'category': row.get('category', row.get('type', 'unknown')),
                'color': row.get('color', row.get('colour', 'unknown'))
            }
            catalog_data.append(product_entry)
        
        # Create DataFrame and save
        catalog_df = pd.DataFrame(catalog_data)
        catalog_df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Created catalog.csv with {len(catalog_df)} products")
        logger.info(f"Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        return False

def validate_vibes_list():
    """Validate the vibes list format"""
    vibes_path = Path("data/vibeslist.json")
    
    if not vibes_path.exists():
        logger.error(f"Vibes list not found: {vibes_path}")
        return False
    
    try:
        with open(vibes_path, 'r') as f:
            vibes = json.load(f)
        
        logger.info(f"✅ Vibes list loaded: {vibes}")
        logger.info(f"Total vibes: {len(vibes)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error reading vibes list: {e}")
        return False

def show_dataset_summary():
    """Show summary of the complete dataset"""
    
    logger.info("\n" + "="*50)
    logger.info("HACKATHON DATASET SUMMARY")
    logger.info("="*50)
    
    # Videos
    videos_dir = Path("videos")
    if videos_dir.exists():
        videos = list(videos_dir.glob("*.mp4"))
        logger.info(f"📹 Videos: {len(videos)} MP4 files")
        for video in videos[:5]:  # Show first 5
            logger.info(f"  - {video.name}")
        if len(videos) > 5:
            logger.info(f"  ... and {len(videos) - 5} more")
    
    # Catalog
    catalog_path = Path("data/catalog.csv")
    if catalog_path.exists():
        df = pd.read_csv(catalog_path)
        logger.info(f"📋 Catalog: {len(df)} products")
        logger.info(f"  Categories: {', '.join(df['category'].unique()[:10])}")
    
    # Vibes
    vibes_path = Path("data/vibeslist.json")
    if vibes_path.exists():
        with open(vibes_path, 'r') as f:
            vibes = json.load(f)
        logger.info(f"🧠 Vibes: {vibes}")
    
    # Output directory
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        existing_outputs = list(outputs_dir.glob("*.json"))
        logger.info(f"📤 Outputs: {len(existing_outputs)} existing results")
    
    logger.info("="*50)

def main():
    """Main processing function"""
    
    logger.info("🚀 Processing Hackathon Dataset")
    
    # Step 1: Process Excel to catalog
    if process_excel_to_catalog():
        logger.info("✅ Excel processed successfully")
    else:
        logger.error("❌ Excel processing failed")
        return
    
    # Step 2: Validate vibes
    if validate_vibes_list():
        logger.info("✅ Vibes list validated")
    else:
        logger.error("❌ Vibes list validation failed")
        return
    
    # Step 3: Show summary
    show_dataset_summary()
    
    logger.info("\n🎯 Ready for video processing!")
    logger.info("Next steps:")
    logger.info("1. Run: python run_video_analysis.py")
    logger.info("2. Check outputs/ folder for JSON results")

if __name__ == "__main__":
    main() 