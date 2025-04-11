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
        
        # Add column specific information including data types
        self.metadata["column_info"] = []
        for col in self.df.columns:
            col_info = {"name": col, "dtype": str(self.df[col].dtype)}
            
            # Add basic statistics based on data type
            if pd.api.types.is_numeric_dtype(self.df[col]):
                try:
                    col_info["min"] = float(self.df[col].min())
                    col_info["max"] = float(self.df[col].max())
                    col_info["mean"] = float(self.df[col].mean())
                    col_info["null_count"] = int(self.df[col].isna().sum())
                    col_info["unique_count"] = int(self.df[col].nunique())
                except:
                    pass
            elif pd.api.types.is_string_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                try:
                    col_info["null_count"] = int(self.df[col].isna().sum())
                    col_info["unique_count"] = int(self.df[col].nunique())
                    # Get top 3 most common values
                    top_values = self.df[col].value_counts().head(3).to_dict()
                    col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                except:
                    pass
            
            self.metadata["column_info"].append(col_info)
    
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
            system_message = """You are a data analysis assistant for humanitarian and development programs. 
            You analyze field data and create insights for non-technical users.
            
            Given a pandas DataFrame with field data from humanitarian programs, translate the user's natural language query into Python pandas code.
            First, think about what the user is asking for and what pandas operations would best accomplish that task.
            
            Consider:
            1. What columns are needed for the analysis?
            2. What filtering or data transformations are required?
            3. What grouping, aggregation, or calculations should be performed?
            4. What's the most appropriate way to present the results?
            
            Output your answer in the following format:
            
            ```python
            # Pandas code to solve the query
            import pandas as pd
            import numpy as np
            
            # Your solution code here
            # The input DataFrame is available as 'df'
            # The last line should produce the result
            ```
            
            After the code block, give a brief explanation in simple language that would help a non-technical user understand:
            1. What this analysis shows
            2. Any key insights or patterns in the data
            3. What decisions or actions could be informed by these results
            
            Make your explanation clear, concise, and actionable for field program staff.
            """
            
            # Create field-specific examples based on typical humanitarian data
            examples = """
            Example 1:
            Query: "Show me the distribution of beneficiaries by gender and age group"
            Answer:
            ```python
            import pandas as pd
            import numpy as np
            
            # Create age groups
            df['age_group'] = pd.cut(df['age'], bins=[0, 5, 18, 35, 60, 100], labels=['0-5', '6-18', '19-35', '36-60', '60+'])
            
            # Group by gender and age group and count beneficiaries
            result = df.groupby(['gender', 'age_group']).size().reset_index(name='count')
            
            # Pivot the table for better visualization
            pivot_result = result.pivot(index='age_group', columns='gender', values='count').fillna(0)
            
            pivot_result
            ```
            
            This analysis shows how beneficiaries are distributed across different age groups and genders. You can see which demographic groups have the highest representation in your program. This information can help you assess if your targeting criteria are being met and identify any under-served populations that might need additional outreach.
            
            Example 2:
            Query: "What is the average monthly income by region, and which region has the highest poverty rate?"
            Answer:
            ```python
            import pandas as pd
            import numpy as np
            
            # Calculate average monthly income by region
            avg_income = df.groupby('region')['monthly_income'].mean().reset_index()
            
            # Calculate poverty rate by region (assuming poverty line of $1.90 per day or ~$57 per month)
            poverty_line = 57
            df['is_poor'] = df['monthly_income'] < poverty_line
            poverty_rate = df.groupby('region')['is_poor'].mean().reset_index()
            poverty_rate['poverty_percentage'] = poverty_rate['is_poor'] * 100
            
            # Combine the results
            result = pd.merge(avg_income, poverty_rate[['region', 'poverty_percentage']], on='region')
            result = result.sort_values('poverty_percentage', ascending=False)
            
            result
            ```
            
            This analysis presents two key metrics by region: the average monthly income and the poverty rate (percentage of individuals below the $1.90/day international poverty line). The regions are ranked from highest to lowest poverty rate. This information can help program managers prioritize resource allocation to regions with the highest need and design appropriate interventions based on the economic situation in each area.
            """
            
            user_message = f"""
            I have a pandas DataFrame 'df' with the following characteristics:
            
            Columns: {', '.join(self.metadata['columns'])}
            Shape: {self.metadata['shape'][0]} rows, {self.metadata['shape'][1]} columns
            
            Column details:
            {json.dumps(self.metadata['column_info'], indent=2)}
            
            Here's a sample of the first few rows:
            {self._get_dataframe_sample()}
            
            Query from a field program staff: {query}
            
            Please provide Python pandas code that answers this query and explain the results in simple terms for non-technical program staff.
            """
            
            # Make the API call to OpenRouter
            logger.info("Making SmartDataFrame API call to OpenRouter...")
            try:
                response = openai_client.chat.completions.create(
                    model="meta-llama/llama-4-scout:free",  # Using Llama 4 Scout model
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": examples},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.2,  # Slightly higher temperature for creative solutions
                    max_tokens=2000   # Allow for longer responses with code and explanation
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
                # Try a second attempt with a simpler approach if the first one failed
                simplified_message = f"""
                The previous code had an error: {error}
                
                Please provide a simpler solution using only basic pandas operations.
                Focus on just answering the core question: "{query}"
                
                Use only these basic operations:
                - Filtering with df[condition]
                - Basic aggregations like .mean(), .sum(), .count()
                - Simple groupby operations
                - Avoid complex transformations or creating new calculated columns
                
                The DataFrame has these columns: {', '.join(self.metadata['columns'])}
                """
                
                try:
                    retry_response = openai_client.chat.completions.create(
                        model="meta-llama/llama-4-scout:free",
                        messages=[
                            {"role": "system", "content": "You are a data analysis assistant. Fix the pandas code to make it simpler and error-free."},
                            {"role": "user", "content": simplified_message}
                        ],
                        temperature=0.1
                    )
                    
                    retry_content = retry_response.choices[0].message.content
                    retry_code = self._extract_code(retry_content)
                    
                    if retry_code:
                        result_df, error = self._execute_code(retry_code)
                        if not error:
                            code = retry_code
                            # Extract explanation
                            explanation_parts = retry_content.split("```")
                            explanation = explanation_parts[-1].strip() if len(explanation_parts) > 1 else ""
                            
                            return {
                                "success": True,
                                "result": result_df,
                                "code": code,
                                "explanation": explanation,
                                "retry_used": True
                            }
                except Exception as retry_error:
                    logger.error(f"Error in retry attempt: {str(retry_error)}")
                
                # If retry also failed, return the original error
                return {
                    "success": False,
                    "error": error,
                    "code": code,
                    "ai_response": ai_response
                }
            
            # Get explanation (text after the code block)
            explanation_parts = ai_response.split("```")
            explanation = explanation_parts[-1].strip() if len(explanation_parts) > 1 else ""
            
            # Process the result to make it more presentable
            if isinstance(result_df, pd.DataFrame):
                # If it's a large DataFrame, show a summary instead
                if result_df.shape[0] > 50:
                    result_summary = {
                        "shape": result_df.shape,
                        "columns": result_df.columns.tolist(),
                        "head": result_df.head(10).to_dict(orient="records"),
                        "too_large": True
                    }
                    result_to_return = result_df.head(50)  # Return just the first 50 rows
                else:
                    result_summary = None
                    result_to_return = result_df
            else:
                result_summary = None
                result_to_return = result_df
            
            return {
                "success": True,
                "result": result_to_return,
                "summary": result_summary,
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
            has_viz_libs = True
        except ImportError:
            has_viz_libs = False
        
        # Build the local variables dictionary based on available libraries
        local_vars = {
            "df": self.df.copy(), 
            "pd": pd, 
            "np": np
        }
        
        if has_viz_libs:
            local_vars.update({
                "plt": plt,
                "px": px
            })
        
        try:
            # Add safety checks but allow common data science libraries
            safe_imports = ["import pandas as pd", "import numpy as np", 
                           "from pandas import", "from numpy import"]
            
            if has_viz_libs:
                safe_imports.extend(["import matplotlib.pyplot as plt", 
                                     "import plotly.express as px"])
                                     
            if "import" in code and not any(code.startswith(safe_import) for safe_import in safe_imports):
                # Check if it's a multiline import with allowed libraries
                lines = code.split("\n")
                for line in lines:
                    if line.strip().startswith("import") and not any(safe_import in line for safe_import in safe_imports):
                        return None, "Unauthorized import detected: " + line.strip()
            
            # Add extra safety for file operations, system calls, etc.
            dangerous_terms = ["os.", "subprocess", "eval(", "exec(", 
                             "read_", "write_", "open(", "save", "load", "file"]
            
            for term in dangerous_terms:
                if term in code:
                    # Check if it's a false positive (like "read_csv" in a comment)
                    lines = code.split("\n")
                    for line in lines:
                        if term in line and not line.strip().startswith("#"):
                            # Simple check if it's part of a legitimate pandas operation with safeguards
                            if term == "read_" and "read_csv" in line and "'http" not in line and '"http' not in line:
                                continue  # Allow DataFrame.read_csv but not with URLs
                            return None, f"Unauthorized operation detected: {term}"
            
            # Execute the code
            exec(code, {}, local_vars)
            
            # Find the result in the local variables
            # Strategy 1: Look for variables defined at the end of the code
            lines = code.strip().split("\n")
            last_var = None
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    last_var = line.split("=")[0].strip()
                    break
                elif line and not line.startswith("#") and any(x in line for x in ["display(", "print("]):
                    # Match display(var) or print(var) pattern
                    var_match = re.search(r"(?:display|print)\s*\(\s*([a-zA-Z0-9_]+)\s*\)", line)
                    if var_match:
                        last_var = var_match.group(1)
                        break
            
            # Strategy 2: Look for the last line that could be an expression
            last_expr = None
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith("#") and "=" not in line:
                    last_expr = line
                    break
            
            # Check if we found potential results and try to extract them
            result = None
            
            # Try last variable first
            if last_var and last_var in local_vars:
                result = local_vars[last_var]
            
            # If no result yet, try evaluating the last expression
            if result is None and last_expr:
                try:
                    # Avoid evaluating function calls or complex expressions
                    if not any(x in last_expr for x in ["(", ")", "="]):
                        result = local_vars.get(last_expr)
                except:
                    pass
            
            # If still no result, look for common result variable names
            if result is None:
                for var_name in ["result", "df_result", "output", "summary", "final"]:
                    if var_name in local_vars:
                        result = local_vars[var_name]
                        break
            
            # If still no result, just use the modified dataframe if it was changed
            if result is None:
                df_in_locals = local_vars.get("df")
                if df_in_locals is not None and not df_in_locals.equals(self.df):
                    result = df_in_locals
            
            # If we still have no result, check for any DataFrame or Series in the locals
            if result is None:
                for var_name, var_value in local_vars.items():
                    if var_name != "df" and (isinstance(var_value, pd.DataFrame) or isinstance(var_value, pd.Series)):
                        result = var_value
                        break
            
            return result, None
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg