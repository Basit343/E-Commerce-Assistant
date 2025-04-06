import pandas as pd
from typing import Dict, List, Any, Optional, Type, Literal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class FAQQueryInput(BaseModel):
    query: str = Field(
        description="Natural language question that should be matched against the FAQ database"
    )


class FAQTool(BaseTool):
    name: Literal["faq_query_tool"] = "faq_query_tool"
    description: str = """
    Use this tool when the user is asking general questions about shipping, returns, warranties, 
    company policies, account management, or any other customer service related questions.
    
    Examples of queries this tool can handle:
    - "How do I return a product?"
    - "What is your shipping policy?"
    - "How long is the warranty period?"
    - "Can I change my shipping address after ordering?"
    - "How do I track my order?"
    """
    
    args_schema: Type[BaseModel] = FAQQueryInput
    faq_df: pd.DataFrame = Field(default_factory=pd.DataFrame)
    similarity_threshold: float = Field(default=0.3)
    vectorizer: TfidfVectorizer = Field(default_factory=lambda: TfidfVectorizer(stop_words='english'))
    question_vectors: Any = Field(default=None)
    
    def __init__(self, faq_df: pd.DataFrame, similarity_threshold: float = 0.3):
        super().__init__()
        self.faq_df = faq_df
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.question_vectors = self.vectorizer.fit_transform(faq_df['question'].tolist())
    
    def _initialize_vectorizer(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.question_vectors = self.vectorizer.fit_transform(self.faq_df['question'].tolist())
    
    def _find_most_similar_question(self, query: str) -> Optional[Dict[str, Any]]:
        query_vector = self.vectorizer.transform([query])
        
        similarity_scores = cosine_similarity(query_vector, self.question_vectors)[0]
        
        max_index = similarity_scores.argmax()
        max_similarity = similarity_scores[max_index]
        
        if max_similarity >= self.similarity_threshold:
            return {
                'question': self.faq_df.iloc[max_index]['question'],
                'answer': self.faq_df.iloc[max_index]['answer'],
                'similarity': float(max_similarity)
            }
        else:
            return None

    def _run(self, query: str) -> str:
        try:
            match = self._find_most_similar_question(query)
            
            if match:
                return (
                    f"I found a similar question in our FAQ:\n\n"
                    f"Q: {match['question']}\n\n"
                    f"A: {match['answer']}"
                )
            else:
                return (
                    "No relevant FAQ found for this query."
                    " Please try rephrasing your question or contact support."
                )
                
        except Exception as e:
            return f"Error processing FAQ query: {str(e)}"
