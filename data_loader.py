import os
import pandas as pd
from typing import Dict, Any, List, Tuple


class DataLoader:
    """Utility class for loading and processing the product and FAQ data."""
    
    def __init__(self, product_path: str, faq_path: str):
        """
        Initialize DataLoader with paths to data files.
        
        Args:
            product_path: Path to the product statistics CSV file
            faq_path: Path to the FAQ CSV file
        """
        self.product_path = product_path
        self.faq_path = faq_path
        self.products_df = None
        self.faq_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the product and FAQ data from CSV files.
        
        Returns:
            Tuple containing the products DataFrame and FAQ DataFrame
        """
        try:
            # Load product data
            self.products_df = pd.read_csv(self.product_path)
            
            # Load FAQ data
            self.faq_df = pd.read_csv(self.faq_path)
            
            # Convert column names to lowercase for consistency
            self.products_df.columns = [col.lower().replace(' ', '_') for col in self.products_df.columns]
            self.faq_df.columns = [col.lower().replace(' ', '_') for col in self.faq_df.columns]
            
            # Ensure required columns exist
            required_product_cols = ['product_id', 'name', 'category', 'price', 'sales_count', 'rating', 'stock_level']
            for col in required_product_cols:
                if col not in self.products_df.columns:
                    print(f"Warning: Expected column '{col}' not found in product data")
            
            required_faq_cols = ['question', 'answer']
            for col in required_faq_cols:
                if col not in self.faq_df.columns:
                    print(f"Warning: Expected column '{col}' not found in FAQ data")
            
            # Convert price to float if it exists
            if 'price' in self.products_df.columns:
                self.products_df['price'] = self.products_df['price'].astype(float)
            
            # Convert rating to float if it exists
            if 'rating' in self.products_df.columns:
                self.products_df['rating'] = self.products_df['rating'].astype(float)
                
            return self.products_df, self.faq_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_product_categories(self) -> List[str]:
        """
        Get the list of unique product categories.
        
        Returns:
            List of unique product categories
        """
        if self.products_df is None:
            self.load_data()
        
        return self.products_df['category'].unique().tolist()
    
    def get_price_range(self) -> Tuple[float, float]:
        """
        Get the minimum and maximum product price.
        
        Returns:
            Tuple containing the minimum and maximum price
        """
        if self.products_df is None:
            self.load_data()
        
        return self.products_df['price'].min(), self.products_df['price'].max()
    
    def get_rating_range(self) -> Tuple[float, float]:
        """
        Get the minimum and maximum product rating.
        
        Returns:
            Tuple containing the minimum and maximum rating
        """
        if self.products_df is None:
            self.load_data()
        
        return self.products_df['rating'].min(), self.products_df['rating'].max()
    
    def get_stock_levels(self) -> List[str]:
        """
        Get the list of unique stock levels.
        
        Returns:
            List of unique stock levels
        """
        if self.products_df is None:
            self.load_data()
        
        return self.products_df['stock_level'].unique().tolist()
