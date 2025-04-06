import os
from typing import List, Dict, Any
import pandas as pd
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

from product_tool import ProductTool
from faq_tool import FAQTool


class ECommerceAgent:
    def __init__(
        self,
        products_df: pd.DataFrame,
        faq_df: pd.DataFrame,
        model_name: str = "gpt-3.5-turbo-0125",
        temperature: float = 0.0,
        api_key: str = None
    ):
        self.products_df = products_df
        self.faq_df = faq_df
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")
        
        self.product_tool = ProductTool(products_df)
        self.faq_tool = FAQTool(faq_df)
        self.tools = [self.product_tool, self.faq_tool]
        
        self._initialize_agent()
    
    def _initialize_agent(self):
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key
        )
        
        system_prompt = """You are a helpful e-commerce assistant designed to help customers with product information and questions.

Your primary responsibilities are:
1. Provide accurate product information using the product query tool
2. Answer general questions using the FAQ tool

For product-related queries, use the product_query_tool to search for products based on filters like category, price range, rating, and stock status.
For general questions about shipping, returns, warranties, policies, etc., use the faq_query_tool to find the most relevant answer.

Always try to be helpful and provide complete information to the user. If the user's query is ambiguous, ask for clarification.

Route the query to the appropriate tool based on the user's intent:
- If they're asking about products, prices, or availability, use the product_query_tool
- If they're asking general questions about policies or services, use the faq_query_tool

Only use the tools provided to you. If you can't answer a question with the available tools, politely explain that you don't have that information.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("human", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(llm, self.tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    def process_query(self, query: str) -> str:
        try:
            if not self.api_key:
                return "Error: OpenAI API key not configured"
                
            response = self.agent_executor.invoke({"input": query})
            if isinstance(response, dict) and "output" in response:
                return response["output"]
            return "No valid response generated"
            
        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg:
                return f"Error: Invalid OpenAI model '{self.model_name}'. Please check your model configuration."
            elif "invalid_request_error" in error_msg:
                return "Error: Invalid API request. Please check your OpenAI API key and configuration."
            else:
                return f"Error processing query: {error_msg}"