"""
SmartDataFrame module.
Provides AI-powered functionality to analyze pandas DataFrames using natural language.
Inspired by PandasAI but using OpenRouter API directly.
"""

import os
import json
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
import re

from utils.nlp_processing import client as openai_client  # Use the existing OpenAI client with OpenRouter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartDataFrame:
    """
    Class that wraps a pandas DataFrame with AI-powered natural language querying.
    """
    
    def __init__(self, df: pd.DataFrame, api_key=None):
        """
        Initialize the SmartDataFrame.
        
        Args:
            df: Pandas DataFrame to wrap
            api_key: Optional API key for OpenRouter
        """
        self.df = df
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.api_available = self.api_key is not None
        self.openai_client = openai_client
        
        # Store DataFrame metadata
        self._compute_metadata()
    
    def _compute_metadata(self):
        """Compute metadata about the DataFrame."""
        self.metadata = {
            "columns": self.df.columns.tolist(),
            "shape": self.df.shape,
            "dtypes": {col: str(dtype) for col, dtype in zip(self.df.columns, self.df.dtypes)},
            "description": self.df.describe().to_dict() if self._has_numeric_columns() else {}
        }
    
    def _has_numeric_columns(self) -> bool:
        """Check if the DataFrame has any numeric columns."""
        return any(pd.api.types.is_numeric_dtype(self.df[col]) for col in self.df.columns)
    
    def _get_dataframe_sample(self, rows: int = 5) -> str:
        """Get a sample of the DataFrame as a string."""
        return self.df.head(rows).to_string()
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from an AI response."""
        # Look for python code blocks
        code_pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # Look for any code blocks
        code_pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # If no code blocks, look for lines that appear to be code
        lines = text.split("\n")
        code_lines = []
        for line in lines:
            # Simple heuristic: lines that contain .something() or = are likely code
            if (".(" in line.replace(" ", "") or "=" in line) and "df" in line:
                code_lines.append(line)
        
        return "\n".join(code_lines) if code_lines else ""
    
    def chat(self, query: str) -> Dict[str, Any]:
        """
        Query the DataFrame using natural language.
        
        Args:
            query: Natural language query
            
        Returns:
            Dict with result DataFrame, explanation, and other metadata
        """
        if not self.api_available:
            return {
                "success": False,
                "error": "API key not available. Please check your OpenRouter API key."
            }
        
        try:
            # Create prompt with DataFrame info and query
            system_message = """You are a data analysis assistant specialized in pandas operations.
            Given a pandas DataFrame, translate the user's natural language query into Python pandas code.
            Output your answer in the following format:
            
            ```python
            # Pandas code to solve the query
            import pandas as pd
            
            # Your solution code here
            # The input DataFrame is available as 'df'
            # The last line should produce the result
            ```
            
            After the code block, give a brief explanation of what the code does.
            Be concise but thorough. Use pandas functions that are efficient and readable.
            """
            
            user_message = f"""
            I have a pandas DataFrame 'df' with the following characteristics:
            
            Columns: {', '.join(self.metadata['columns'])}
            Shape: {self.metadata['shape'][0]} rows, {self.metadata['shape'][1]} columns
            Data types: {json.dumps(self.metadata['dtypes'])}
            
            Here's a sample of the data:
            {self._get_dataframe_sample()}
            
            Query: {query}
            
            Provide Python pandas code that answers this query.
            """
            
            # Make the API call to OpenRouter
            logger.info("Making SmartDataFrame API call to OpenRouter...")
            try:
                response = openai_client.chat.completions.create(
                    model="meta-llama/llama-4-scout:free",  # Using Llama 4 Scout model
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1  # Low temperature for more deterministic results
                )
                
                # Log the response for debugging
                logger.info(f"SmartDataFrame API response received")
                
                # Check if we have a valid response
                if not hasattr(response, 'choices') or not response.choices or not hasattr(response.choices[0], 'message'):
                    logger.error(f"Invalid SmartDataFrame response structure: {response}")
                    return {
                        "success": False,
                        "error": "Received invalid response from AI service"
                    }
                
                ai_response = response.choices[0].message.content
                if not ai_response or not ai_response.strip():
                    logger.error("Empty SmartDataFrame response content")
                    return {
                        "success": False,
                        "error": "Received empty response from AI service"
                    }
                
                logger.info(f"SmartDataFrame raw content length: {len(ai_response)}")
            except Exception as api_error:
                logger.error(f"SmartDataFrame API call error: {str(api_error)}")
                return {
                    "success": False,
                    "error": f"Error calling AI service: {str(api_error)}"
                }
            
            # Extract code from the response
            code = self._extract_code(ai_response)
            
            if not code:
                return {
                    "success": False,
                    "error": "Could not extract valid code from AI response",
                    "ai_response": ai_response
                }
            
            # Execute the code and get the result
            result_df, error = self._execute_code(code)
            
            if error:
                return {
                    "success": False,
                    "error": error,
                    "code": code,
                    "ai_response": ai_response
                }
            
            # Get explanation (text after the code block)
            explanation = ai_response.split("```")[-1].strip()
            
            return {
                "success": True,
                "result": result_df,
                "code": code,
                "explanation": explanation,
                "ai_response": ai_response
            }
            
        except Exception as e:
            logger.error(f"Error in SmartDataFrame chat: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating or executing code: {str(e)}"
            }
    
    def _execute_code(self, code: str) -> tuple:
        """
        Execute pandas code safely.
        
        Args:
            code: Python code to execute
            
        Returns:
            Tuple of (result_df, error)
        """
        # Create a safe local environment with common data science libraries
        import numpy as np
        try:
            import matplotlib.pyplot as plt
            import plotly.express as px
            local_vars = {
                "df": self.df.copy(), 
                "pd": pd, 
                "np": np, 
                "plt": plt,
                "px": px
            }
        except ImportError:
            # Fallback if some visualization libraries aren't available
            local_vars = {
                "df": self.df.copy(), 
                "pd": pd,
                "np": np
            }
        
        try:
            # Add safety checks but allow common data science libraries
            safe_imports = ["import pandas as pd", "import numpy as np", "import matplotlib.pyplot as plt", 
                            "from pandas import", "from numpy import", "import plotly.express as px"]
                            
            if "import" in code and not any(code.startswith(safe_import) for safe_import in safe_imports):
                # Check if it's a multiline import with allowed libraries
                lines = code.split("\n")
                for line in lines:
                    if "import" in line and not any(safe_import in line for safe_import in safe_imports):
                        if not line.strip().startswith("#"):  # Ignore comments
                            return None, "Only pandas, numpy, matplotlib and plotly imports are allowed"
            
            if any(unsafe_keyword in code for unsafe_keyword in ["os.", "sys.", "subprocess", "eval(", "exec("]):
                return None, "Unsafe code detected"
            
            # Execute the code
            lines = code.strip().split("\n")
            non_comment_lines = [line for line in lines if not line.strip().startswith("#")]
            
            # Get last expression to return as result
            if non_comment_lines:
                exec_code = "\n".join(lines[:-1])
                result_code = lines[-1]
                
                if exec_code:
                    exec(exec_code, {}, local_vars)
                
                result = eval(result_code, {}, local_vars)
                
                # Check if result is a DataFrame
                if isinstance(result, pd.DataFrame):
                    return result, None
                elif isinstance(result, pd.Series):
                    return result.to_frame(), None
                else:
                    # Convert other types to DataFrame
                    try:
                        return pd.DataFrame({"result": [result]}), None
                    except:
                        return None, f"Result type {type(result)} cannot be converted to DataFrame"
            
            return None, "No executable code found"
            
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            logger.error(f"Code: {code}")
            return None, f"Error executing code: {str(e)}"