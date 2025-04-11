"""
Document processing module for handling text, CSV, and Excel files.
Provides basic functionality for RAG (Retrieval-Augmented Generation).
"""

import os
import io
import re
import json
import tempfile
import csv
from typing import List, Dict, Any, Optional

# For now, we'll focus on plain text and CSV/Excel files that don't require additional libraries
TEXT_AVAILABLE = True
CSV_AVAILABLE = True
EXCEL_AVAILABLE = True  # We already have openpyxl installed

# Define types of documents supported based on available libraries
SUPPORTED_DOCUMENT_TYPES = [".txt", ".csv", ".xlsx", ".xls"]

class DocumentProcessor:
    """
    Class for handling document processing, including text extraction and chunking.
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.document_store = {}  # Store processed documents
        self.document_chunks = {}  # Store document chunks
        self.document_metadata = {}  # Store document metadata
        
        # Create temporary directory for storing documents
        os.makedirs("tmp/documents", exist_ok=True)
    
    def process_document(self, file_obj, filename: str) -> Dict[str, Any]:
        """
        Process a document file and extract text content.
        
        Args:
            file_obj: File object from Streamlit upload
            filename: Name of the uploaded file
        
        Returns:
            Dict with document_id, text_content, and metadata
        """
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension not in SUPPORTED_DOCUMENT_TYPES:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: {SUPPORTED_DOCUMENT_TYPES}")
        
        # Save file to temp location
        temp_file_path = os.path.join("tmp/documents", filename)
        with open(temp_file_path, "wb") as f:
            f.write(file_obj.getvalue())
        
        # Extract text based on file type
        if file_extension == ".txt":
            # Plain text file
            with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
        elif file_extension == ".csv":
            # CSV file - try with utf-8 encoding (no chardet)
            encoding = "utf-8"
            
            try:
                with open(temp_file_path, "r", encoding=encoding, errors="ignore") as f:
                    # Read CSV as text
                    text_content = f.read()
                
                # Add a more structured format for search
                try:
                    import pandas as pd
                    df = pd.read_csv(temp_file_path, encoding=encoding)
                    text_content += "\n\nStructured data summary:\n"
                    text_content += f"Columns: {', '.join(df.columns.tolist())}\n"
                    text_content += f"Rows: {len(df)}\n"
                    
                    # Add sample data
                    sample_rows = df.head(5).to_string()
                    text_content += f"\nSample data:\n{sample_rows}\n"
                except:
                    # If pandas fails, just use the raw text
                    pass
            except Exception as e:
                text_content = f"Error reading CSV file: {str(e)}"
        elif file_extension in [".xlsx", ".xls"]:
            # Excel file
            try:
                import pandas as pd
                df = pd.read_excel(temp_file_path)
                text_content = f"Excel file contents:\n\n"
                text_content += f"Sheets: Sheet1\n"  # We're reading only the first sheet
                text_content += f"Columns: {', '.join(df.columns.tolist())}\n"
                text_content += f"Rows: {len(df)}\n\n"
                
                # Add sample data
                sample_rows = df.head(5).to_string()
                text_content += f"Sample data:\n{sample_rows}\n"
            except Exception as e:
                text_content = f"Error reading Excel file: {str(e)}"
        else:
            text_content = "Could not extract text from document. Unsupported file type."
        
        # Generate document ID and store
        document_id = f"doc_{len(self.document_store) + 1}"
        metadata = {
            "filename": filename,
            "file_type": file_extension,
            "path": temp_file_path,
            "word_count": len(text_content.split())
        }
        
        self.document_store[document_id] = text_content
        self.document_metadata[document_id] = metadata
        
        # Create chunks for more efficient retrieval
        chunks = self._chunk_text(text_content)
        self.document_chunks[document_id] = chunks
        
        return {
            "document_id": document_id,
            "text_content": text_content,
            "metadata": metadata,
            "chunks": chunks
        }
    
    # PDF and DOCX extraction functionality removed due to library limitations
    # We'll focus on text, CSV, and Excel files for now
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for more efficient retrieval.
        
        Args:
            text: The text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        chunks = []
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if end < len(text) and end - start < chunk_size:
                    # Find the last period or newline to make cleaner chunks
                    last_period = text.rfind(".", start, end)
                    last_newline = text.rfind("\n", start, end)
                    if last_period > start + chunk_size // 2:
                        end = last_period + 1
                    elif last_newline > start + chunk_size // 2:
                        end = last_newline + 1
                
                chunks.append(text[start:end])
                start = end - overlap
        
        return chunks
    
    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            Dict with text_content and metadata
        """
        if document_id not in self.document_store:
            raise ValueError(f"Document ID {document_id} not found")
        
        return {
            "text_content": self.document_store[document_id],
            "metadata": self.document_metadata[document_id]
        }
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get a list of all processed documents.
        
        Returns:
            List of document metadata
        """
        return [
            {"document_id": doc_id, "metadata": metadata}
            for doc_id, metadata in self.document_metadata.items()
        ]
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents that match the query using simple keyword matching.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of document matches with relevance scores
        """
        results = []
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        for doc_id, chunks in self.document_chunks.items():
            best_chunk_score = 0
            best_chunk = ""
            
            for chunk in chunks:
                chunk_terms = set(re.findall(r'\w+', chunk.lower()))
                intersection = query_terms.intersection(chunk_terms)
                if intersection:
                    score = len(intersection) / len(query_terms)
                    if score > best_chunk_score:
                        best_chunk_score = score
                        best_chunk = chunk
            
            if best_chunk_score > 0:
                results.append({
                    "document_id": doc_id,
                    "metadata": self.document_metadata[doc_id],
                    "score": best_chunk_score,
                    "text_snippet": best_chunk[:300] + "..." if len(best_chunk) > 300 else best_chunk
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# Initialize global document processor instance
document_processor = DocumentProcessor()