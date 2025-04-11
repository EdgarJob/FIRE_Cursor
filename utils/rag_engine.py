"""
RAG (Retrieval-Augmented Generation) module.
Provides functionality to combine document retrieval with AI-powered responses.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
import logging

from utils.document_processing import document_processor
from utils.nlp_processing import client as openai_client  # Use the existing OpenAI client with OpenRouter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Class for RAG (Retrieval-Augmented Generation) functionality.
    Combines document search with AI-powered responses.
    """
    
    def __init__(self, api_key=None):
        """Initialize the RAG engine."""
        self.document_processor = document_processor
        self.openai_client = openai_client
        
        # Check if we have a valid API key
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.api_available = self.api_key is not None
    
    def query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a query using RAG approach.
        
        Args:
            query: The user's query
            context: Optional context information (such as the current dataframe)
            
        Returns:
            Dict with response, sources, and other metadata
        """
        # Search for relevant documents
        search_results = self.document_processor.search_documents(query, top_k=3)
        
        # Prepare context from search results
        context_text = ""
        sources = []
        
        for result in search_results:
            context_text += f"\n---\nSource: {result['metadata']['filename']}\n"
            context_text += result['text_snippet'] + "\n"
            sources.append({
                "document_id": result["document_id"],
                "filename": result["metadata"]["filename"],
                "score": result["score"]
            })
        
        # If we have dataframe context, add it
        df_context = ""
        if context and 'dataframe' in context:
            df = context['dataframe']
            df_context = f"\nDataFrame information:\n"
            df_context += f"Columns: {', '.join(df.columns.tolist())}\n"
            df_context += f"Rows: {len(df)}\n"
            
            # Add sample data
            try:
                sample_rows = df.head(3).to_string()
                df_context += f"Sample data:\n{sample_rows}\n"
            except:
                pass
        
        # Combine document context with dataframe context
        full_context = df_context + context_text
        
        # Generate AI response
        ai_response = self._generate_response(query, full_context)
        
        return {
            "query": query,
            "response": ai_response,
            "sources": sources,
            "context_used": bool(full_context.strip())
        }
    
    def _generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using OpenRouter API.
        
        Args:
            query: The user's query
            context: Context information from documents and data
            
        Returns:
            Generated response
        """
        if not self.api_available:
            logger.warning("API key not available for RAG response generation")
            return "AI response generation is not available. Please check your OpenRouter API key."
        
        try:
            # Create prompt with context and query
            system_message = """You are a helpful assistant that provides accurate information based on the provided context.
            Answer the user's query using only the information in the context.
            If the context doesn't contain enough information to answer, state that clearly.
            Do not make up information. Cite sources when possible.
            """
            
            user_message = f"""Context information:
            {context}
            
            Based only on the above context, answer this question: {query}
            """
            
            logger.info("Making RAG API call to OpenRouter...")
            
            # Make the API call to OpenRouter
            try:
                response = openai_client.chat.completions.create(
                    model="meta-llama/llama-4-scout:free",  # Using Llama 4 Scout model
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3  # Lower temperature for more factual responses
                )
                
                # Log the response for debugging
                logger.info(f"RAG API response: {response}")
                
                # Check if we have a valid response
                if not hasattr(response, 'choices') or not response.choices or not hasattr(response.choices[0], 'message'):
                    logger.error(f"Invalid RAG response structure: {response}")
                    return "Error: Received invalid response from AI service."
                
                content = response.choices[0].message.content
                if not content or not content.strip():
                    logger.error("Empty RAG response content")
                    return "Error: Received empty response from AI service."
                
                return content
                
            except Exception as api_error:
                logger.error(f"RAG API call error: {str(api_error)}")
                return f"Error calling AI service. Please try again. (Error: {str(api_error)})"
            
        except Exception as e:
            logger.error(f"Error generating RAG AI response: {str(e)}")
            return f"Error generating AI response. Please try again. (Error: {str(e)})"
    
    def query_with_data(self, query: str, df) -> Dict[str, Any]:
        """
        Process a query using both document context and dataframe context.
        
        Args:
            query: The user's query
            df: DataFrame to use as context
            
        Returns:
            Dict with response, sources, and other metadata
        """
        context = {"dataframe": df}
        return self.query(query, context)

# Initialize global RAG engine instance
rag_engine = RAGEngine()