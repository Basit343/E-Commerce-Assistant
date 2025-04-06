import pandas as pd
import re
from typing import Dict, List, Any, Optional, Union, Literal, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class ProductQueryInput(BaseModel):
    query: str = Field(
        description="Natural language query about products with optional filters for category, price range, stock level, and rating"
    )


class ProductTool(BaseTool):
    name: Literal["product_query_tool"] = "product_query_tool"
    description: str = """
    Use this tool when the user is asking about product information, prices, availability, ratings, 
    or wants to filter products by any criteria such as category, price range, stock status, or ratings.
    
    Examples queries this tool can handle:
    - "Show me all electronics products"
    - "What are the top-rated kitchen appliances?"
    - "List products under $50 with rating above 4"
    - "Are there any gaming laptops in stock?"
    - "What's the price range for smartphones?"
    """
    
    args_schema: Type[BaseModel] = ProductQueryInput
    products_df: pd.DataFrame
    
    def __init__(self, products_df: pd.DataFrame):
        super().__init__(products_df=products_df)
        self.products_df = products_df
    
    def _run(self, query: str) -> str:
        try:
            filters = self._extract_filters(query)
            filtered_df = self._apply_filters(filters)
            return self._format_response(filtered_df, filters)
            
        except Exception as e:
            return f"Error processing product query: {str(e)}"
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        filters = {
            "category": None,
            "min_price": None,
            "max_price": None,
            "min_rating": None,
            "max_rating": None,
            "in_stock": None,
            "sort_by": None,
            "limit": 10,
            "original_query": query
        }
        
        categories = self.products_df['category'].unique()
        for category in categories:
            if category.lower() in query.lower():
                filters["category"] = category
                break
        
        min_price_match = re.search(r'(above|over|more than|>)\s*\$?(\d+)', query, re.IGNORECASE)
        if min_price_match:
            filters["min_price"] = float(min_price_match.group(2))
        
        max_price_match = re.search(r'(below|under|less than|<)\s*\$?(\d+)', query, re.IGNORECASE)
        if max_price_match:
            filters["max_price"] = float(max_price_match.group(2))
        
        price_range_match = re.search(r'\$(\d+)\s*(-|to)\s*\$?(\d+)', query, re.IGNORECASE)
        if price_range_match:
            filters["min_price"] = float(price_range_match.group(1))
            filters["max_price"] = float(price_range_match.group(3))
        
        rating_match = re.search(r'rating\s*(above|over|more than|>)\s*(\d+\.?\d*)', query, re.IGNORECASE)
        if rating_match:
            filters["min_rating"] = float(rating_match.group(2))
        
        max_rating_match = re.search(r'rating\s*(below|under|less than|<)\s*(\d+\.?\d*)', query, re.IGNORECASE)
        if max_rating_match:
            filters["max_rating"] = float(max_rating_match.group(2))
        
        if re.search(r'in\s*stock', query, re.IGNORECASE):
            filters["in_stock"] = True
        elif re.search(r'out\s*of\s*stock', query, re.IGNORECASE):
            filters["in_stock"] = False
        
        if re.search(r'(highest|best|top)\s*rated', query, re.IGNORECASE):
            filters["sort_by"] = ("rating", False)
        elif re.search(r'(lowest|worst)\s*rated', query, re.IGNORECASE):
            filters["sort_by"] = ("rating", True)
        elif re.search(r'(highest|most expensive|priciest)', query, re.IGNORECASE):
            filters["sort_by"] = ("price", False)
        elif re.search(r'(lowest|cheapest|least expensive)', query, re.IGNORECASE):
            filters["sort_by"] = ("price", True)
        elif re.search(r'(best|top|most|highest)\s*selling', query, re.IGNORECASE):
            filters["sort_by"] = ("sales_count", False)
        
        limit_match = re.search(r'(show|list|display)\s*(\d+)', query, re.IGNORECASE)
        if limit_match:
            filters["limit"] = int(limit_match.group(2))
        elif re.search(r'(all|every)', query, re.IGNORECASE):
            filters["limit"] = None
        
        return filters
    
    def _apply_filters(self, filters: Dict[str, Any]) -> pd.DataFrame:
        df = self.products_df.copy()
        
        if filters["category"] is not None:
            df = df[df['category'].str.lower() == filters["category"].lower()]
        
        if filters["min_price"] is not None:
            df = df[df['price'] >= filters["min_price"]]
        
        if filters["max_price"] is not None:
            df = df[df['price'] <= filters["max_price"]]
        
        if filters["min_rating"] is not None:
            df = df[df['rating'] >= filters["min_rating"]]
        
        if filters["max_rating"] is not None:
            df = df[df['rating'] <= filters["max_rating"]]
        
        if filters["in_stock"] is not None:
            if filters["in_stock"]:
                df = df[df['stock_level'].str.lower() != 'out of stock']
            else:
                df = df[df['stock_level'].str.lower() == 'out of stock']
        
        if filters["sort_by"] is not None:
            column, ascending = filters["sort_by"]
            df = df.sort_values(by=column, ascending=ascending)
        
        if filters["limit"] is not None:
            df = df.head(filters["limit"])
        
        return df
    
    def _format_response(self, df: pd.DataFrame, filters: Dict[str, Any]) -> str:
        if df.empty:
            return "No products found matching your criteria."
        
        parts = []
        
        if filters["category"] is not None:
            parts.append(f"category '{filters['category']}'")
        
        if filters["min_price"] is not None and filters["max_price"] is not None:
            parts.append(f"price between ${filters['min_price']} and ${filters['max_price']}")
        elif filters["min_price"] is not None:
            parts.append(f"price above ${filters['min_price']}")
        elif filters["max_price"] is not None:
            parts.append(f"price below ${filters['max_price']}")
        
        if filters["min_rating"] is not None:
            parts.append(f"rating above {filters['min_rating']}")
        
        if filters["in_stock"] is not None:
            stock_status = "in stock" if filters["in_stock"] else "out of stock"
            parts.append(stock_status)
        
        filter_description = ""
        if parts:
            filter_description = " for " + ", ".join(parts)
        
        header = f"Found {len(df)} products{filter_description}:\n\n"
        
        product_entries = []
        for _, row in df.iterrows():
            product_entry = (
                f"- {row['name']} (ID: {row['product_id']})\n"
                f"  Category: {row['category']}\n"
                f"  Price: ${row['price']:.2f}\n"
                f"  Rating: {row['rating']:.1f}/5.0\n"
                f"  Stock: {row['stock_level']}\n"
                f"  Sales: {row['sales_count']}\n"
            )
            product_entries.append(product_entry)
        
        return header + "\n".join(product_entries)