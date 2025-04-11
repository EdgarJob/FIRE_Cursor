import pandas as pd
import numpy as np
import re
import streamlit as st
import os
import json
import requests
from openai import OpenAI
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Log the API key status (without revealing the key)
if OPENROUTER_API_KEY:
    logger.info("OpenRouter API key found in environment variables")
else:
    logger.warning("OpenRouter API key not found in environment variables")

# Initialize the OpenAI client with OpenRouter base URL
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://fire.replit.app",
        "X-Title": "FIRE: Field Insight & Reporting Engine"
    }
)

# Dictionary of keywords to look for in queries
AGGREGATION_KEYWORDS = {
    'average': 'mean',
    'avg': 'mean',
    'mean': 'mean',
    'sum': 'sum',
    'total': 'sum',
    'count': 'count',
    'min': 'min',
    'minimum': 'min',
    'max': 'max',
    'maximum': 'max',
    'median': 'median',
    'std': 'std',
    'standard deviation': 'std',
    'variance': 'var',
    'var': 'var',
    'distribution': 'value_counts',
    'breakdown': 'value_counts',
    'frequency': 'value_counts',
    'occurrences': 'value_counts',
    'ranking': 'rank'
}

GROUPBY_KEYWORDS = ['by', 'per', 'across', 'for each', 'grouped by']

VISUALIZATION_KEYWORDS = {
    'bar': ['bar', 'column', 'histogram', 'frequency'],
    'line': ['line', 'trend', 'over time', 'time series'],
    'pie': ['pie', 'proportion', 'percentage', 'share'],
    'scatter': ['scatter', 'correlation', 'relationship'],
    'box': ['box', 'boxplot', 'distribution', 'quartiles'],
    'heatmap': ['heatmap', 'heat map', 'correlation matrix']
}

def parse_query_with_ai(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Use OpenRouter API to parse a natural language query into structured components
    
    Args:
        query: Natural language query string
        df: Pandas DataFrame to provide context about available columns
        
    Returns:
        Dict with parsed components: target_columns, groupby_columns, agg_function, viz_type
    """
    if not OPENROUTER_API_KEY:
        # Fallback to rule-based parsing if no API key
        logger.warning("OpenRouter API key not found. Using rule-based parsing instead.")
        return parse_query_rule_based(query, df)
    
    try:
        # Prepare DataFrame information for the prompt
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            # Fix for empty dataframes or columns - use head instead of sample
            if len(df[col].dropna()) > 0:
                sample_values = df[col].dropna().head(3).tolist()
            else:
                sample_values = ["N/A"]  # Use placeholder for empty columns
                
            column_info.append({
                "name": col,
                "type": dtype,
                "sample_values": sample_values
            })
        
        # Create comprehensive prompt for the API
        system_message = """You are a data analysis assistant specialized in converting natural language queries into structured database queries. 
        Your task is to analyze the user's query and extract:
        1. Target columns to analyze
        2. Columns to group by (if any)
        3. Aggregation function to apply (e.g., mean, sum, count, value_counts)
        4. Appropriate visualization type (bar, line, pie, scatter, box, heatmap)
        
        Output your answer as a JSON object with these keys:
        - target_columns: array of column names
        - groupby_columns: array of column names
        - agg_function: string (one of: mean, sum, count, min, max, median, std, var, value_counts, or null)
        - viz_type: string (one of: bar, line, pie, scatter, box, heatmap)
        - explanation: brief explanation of your interpretation
        
        Make sure all column names exactly match the available columns.
        """
        
        user_message = f"""Query: {query}
        
        Available columns in the DataFrame:
        {json.dumps(column_info, indent=2)}
        
        Common aggregation functions: mean, sum, count, min, max, median, std, var, value_counts
        Common visualization types: bar, line, pie, scatter, box, heatmap
        
        Parse this query into components for executing against the DataFrame."""
        
        # Make the API call to OpenRouter using Llama 4 Scout model
        logger.info("Making OpenRouter API call...")
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout:free",  # Using Llama 4 Scout model as requested
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # Low temperature for more deterministic results
            )
            
            # Log the raw response for debugging
            logger.info(f"API response received: {response}")
            
            # Check if we have a valid response with content
            if not hasattr(response, 'choices') or not response.choices or not hasattr(response.choices[0], 'message'):
                logger.error(f"Invalid response structure: {response}")
                return parse_query_rule_based(query, df)
                
            # Extract the content and ensure it's valid JSON
            content = response.choices[0].message.content
            if not content or not content.strip():
                logger.error("Empty response content")
                return parse_query_rule_based(query, df)
                
            logger.info(f"Raw content: {content}")
            
            # Try to parse the JSON response - Extract JSON from markdown code blocks if present
            try:
                # Check if the response is wrapped in markdown code blocks
                if "```json" in content and "```" in content:
                    # Extract content between ```json and ```
                    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                        result = json.loads(json_content)
                    else:
                        # Try with just ``` (no json specifier)
                        json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
                        if json_match:
                            json_content = json_match.group(1)
                            result = json.loads(json_content)
                        else:
                            result = json.loads(content)
                else:
                    result = json.loads(content)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON, trying to extract JSON structure manually")
                # Try to find JSON-like structure with regex
                json_pattern = r'\{[^}]*"target_columns"[^}]*\}'
                match = re.search(json_pattern, content, re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group(0))
                    except:
                        logger.error("Manual JSON extraction failed")
                        return parse_query_rule_based(query, df)
                else:
                    logger.error("No JSON structure found in response")
                    return parse_query_rule_based(query, df)
        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            return parse_query_rule_based(query, df)
        
        # Validate that the columns actually exist in the dataframe
        validated_result = {
            "target_columns": [col for col in result.get("target_columns", []) if col in df.columns],
            "groupby_columns": [col for col in result.get("groupby_columns", []) if col in df.columns],
            "agg_function": result.get("agg_function"),
            "viz_type": result.get("viz_type", "bar"),
            "explanation": result.get("explanation", "")
        }
        
        # Log the interpretation for debugging
        logger.info(f"AI interpretation: {validated_result['explanation']}")
        
        # If the AI couldn't identify valid columns, fall back to rule-based
        if not validated_result["target_columns"] and not validated_result["groupby_columns"]:
            logger.warning("AI parsing didn't identify valid columns - falling back to rule-based")
            return parse_query_rule_based(query, df)
        
        return validated_result
        
    except Exception as e:
        logger.error(f"Error with OpenRouter API: {str(e)}")
        # Fallback to rule-based parsing
        return parse_query_rule_based(query, df)

def parse_query_rule_based(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Parse a natural language query into structured components using rule-based methods
    
    Args:
        query: Natural language query string
        df: Pandas DataFrame to query against
        
    Returns:
        Dict with parsed components: target_columns, groupby_columns, agg_function, viz_type
    """
    query = query.lower().strip()
    
    # Determine aggregation type
    agg_function = None
    for keyword, func in AGGREGATION_KEYWORDS.items():
        if keyword in query:
            agg_function = func
            break
    
    # If no explicit aggregation is mentioned, try to infer one
    if not agg_function:
        if 'show' in query or 'display' in query or 'list' in query:
            agg_function = None  # Just show data
        else:
            agg_function = 'value_counts'  # Default to showing counts
    
    # Determine columns to analyze
    target_columns = []
    for col in df.columns:
        if col.lower() in query:
            target_columns.append(col)
    
    # If no specific columns mentioned, try to infer sensible columns
    if not target_columns:
        # For numeric aggregations, use numeric columns
        if agg_function in ['mean', 'sum', 'min', 'max', 'median', 'std', 'var']:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                target_columns = [numeric_cols[0]]  # Use first numeric column by default
        else:
            # For counts/distributions, use categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                target_columns = [categorical_cols[0]]  # Use first categorical column by default
    
    # Determine groupby columns
    groupby_columns = []
    for keyword in GROUPBY_KEYWORDS:
        if keyword in query:
            # Find what comes after the keyword
            pattern = rf"{keyword}\s+(\w+)"
            matches = re.findall(pattern, query)
            for match in matches:
                # Find a column that closely matches this term
                for col in df.columns:
                    if match in col.lower():
                        groupby_columns.append(col)
                        break
    
    # Determine visualization type
    viz_type = None
    for viz, keywords in VISUALIZATION_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            viz_type = viz
            break
    
    # If no specific visualization mentioned, infer based on query and data
    if not viz_type:
        if agg_function == 'value_counts' or 'count' in query:
            viz_type = 'bar'
        elif groupby_columns and len(groupby_columns) == 1:
            viz_type = 'bar'
        elif groupby_columns and len(groupby_columns) > 1:
            viz_type = 'heatmap'
        elif len(target_columns) >= 2 and all(df[col].dtype.kind in 'ifc' for col in target_columns[:2]):
            viz_type = 'scatter'
        else:
            viz_type = 'bar'  # Default
    
    return {
        "target_columns": target_columns,
        "groupby_columns": groupby_columns,
        "agg_function": agg_function,
        "viz_type": viz_type,
        "explanation": "Query parsed using rule-based method"
    }

def process_query(query: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Process a natural language query and execute it against the dataframe.
    
    Args:
        query: Natural language query string
        df: Pandas DataFrame to query against
        
    Returns:
        tuple: (result_df, visualization_type)
    """
    # Try to parse with AI first, fall back to rule-based method if needed
    parsed_query = parse_query_with_ai(query, df)
    
    # Extract components from the parsed query
    target_columns = parsed_query["target_columns"]
    groupby_columns = parsed_query["groupby_columns"]
    agg_function = parsed_query["agg_function"]
    viz_type = parsed_query["viz_type"]
    
    # Store the explanation for the user
    if "explanation" in parsed_query and parsed_query["explanation"]:
        st.session_state.query_explanation = parsed_query["explanation"]
    
    # Execute the query
    result = execute_query(df, target_columns, groupby_columns, agg_function)
    
    return result, viz_type

def execute_query(df, target_columns, groupby_columns, agg_function):
    """
    Execute a query against the dataframe based on parsed components.
    
    Args:
        df: Pandas DataFrame to query
        target_columns: List of columns to analyze
        groupby_columns: List of columns to group by
        agg_function: Aggregation function to apply
        
    Returns:
        pandas.DataFrame: Result of the query
    """
    # Handle empty selections
    if not target_columns and not groupby_columns:
        # Return sample of dataframe if no specific analysis is requested
        return df.head(10)
    
    # If we have target columns but no aggregation
    if target_columns and not agg_function:
        if groupby_columns:
            # Return the selected columns grouped by the groupby columns
            return df.groupby(groupby_columns)[target_columns].head(20).reset_index()
        else:
            # Just return the selected columns
            return df[target_columns].head(20)
    
    # If we have an aggregation but no target columns
    if agg_function and not target_columns:
        if agg_function == 'value_counts':
            # Get distribution of the first categorical column
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                result = df[cat_cols[0]].value_counts().reset_index()
                result.columns = [cat_cols[0], 'count']
                return result
        else:
            # Get aggregation of the first numeric column
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                if groupby_columns:
                    result = df.groupby(groupby_columns)[num_cols[0]].agg(agg_function).reset_index()
                    result.columns = groupby_columns + [f"{agg_function}_{num_cols[0]}"]
                    return result
                else:
                    # Just aggregate the numeric column
                    agg_value = getattr(df[num_cols[0]], agg_function)()
                    return pd.DataFrame({
                        'Metric': [f"{agg_function.capitalize()} of {num_cols[0]}"],
                        'Value': [agg_value]
                    })
    
    # Standard case: we have target columns and aggregation
    if target_columns and agg_function:
        if agg_function == 'value_counts':
            # Handle value_counts specially since it's a Series method, not a GroupBy method
            if len(target_columns) == 1:
                result = df[target_columns[0]].value_counts().reset_index()
                result.columns = [target_columns[0], 'count']
                return result
            else:
                # For multiple columns, we'll create a combined column to count
                df_copy = df.copy()
                df_copy['combined'] = df_copy[target_columns].astype(str).agg(' - '.join, axis=1)
                result = df_copy['combined'].value_counts().reset_index()
                result.columns = ['combined', 'count']
                return result
                
        elif groupby_columns:
            # Group by the specified columns and aggregate the target columns
            result = df.groupby(groupby_columns)[target_columns].agg(agg_function).reset_index()
            return result
        else:
            # Just aggregate the target columns
            result = pd.DataFrame()
            for col in target_columns:
                if df[col].dtype.kind in 'ifc':  # If column is numeric
                    agg_value = getattr(df[col], agg_function)()
                    new_row = pd.DataFrame({
                        'Metric': [f"{agg_function.capitalize()} of {col}"],
                        'Value': [agg_value]
                    })
                    result = pd.concat([result, new_row])
            return result
    
    # Fallback - return a sample of the dataframe
    return df.head(10)
