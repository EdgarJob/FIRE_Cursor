"""
Advanced data cleaning and preprocessing module.
Provides functions for detecting and handling missing values, outliers, and other data quality issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report for the DataFrame.
    
    Args:
        df: Pandas DataFrame to analyze
        
    Returns:
        Dict with various data quality metrics
    """
    if df is None or df.empty:
        return {
            "error": "DataFrame is empty or None",
            "status": "error"
        }
    
    # Initialize report dictionary
    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_stats": {},
        "missing_values": {},
        "outliers": {},
        "duplicates": {
            "count": df.duplicated().sum(),
            "percentage": (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
        },
        "status": "success"
    }
    
    # Process each column
    for column in df.columns:
        # Skip processing if column is completely empty
        if df[column].isna().all():
            report["column_stats"][column] = {
                "dtype": str(df[column].dtype),
                "is_numeric": False,
                "unique_count": 0,
                "is_empty": True
            }
            continue
            
        # Basic column stats
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        unique_count = df[column].nunique()
        
        report["column_stats"][column] = {
            "dtype": str(df[column].dtype),
            "is_numeric": is_numeric,
            "unique_count": unique_count,
            "is_empty": False
        }
        
        # Missing values
        missing_count = df[column].isna().sum()
        missing_percentage = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        
        if missing_count > 0:
            report["missing_values"][column] = {
                "count": int(missing_count),
                "percentage": float(missing_percentage)
            }
        
        # Outliers (for numeric columns only)
        if is_numeric:
            try:
                # Convert to float to ensure consistency with PyArrow
                series = df[column].astype(float)
                
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(df)) * 100 if len(df) > 0 else 0
                
                if outlier_count > 0:
                    report["outliers"][column] = {
                        "count": int(outlier_count),
                        "percentage": float(outlier_percentage),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    }
            except Exception as e:
                logger.warning(f"Could not calculate outliers for column {column}: {str(e)}")
    
    return report


def fix_missing_values(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
    """
    Fix missing values in the DataFrame using specified strategies.
    
    Args:
        df: Pandas DataFrame to process
        strategy: Dict mapping column names to strategies
                  ('drop', 'mean', 'median', 'mode', 'zero', 'value:X')
        
    Returns:
        Processed DataFrame with fixed missing values
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    for column, strat in strategy.items():
        if column not in result_df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            continue
        
        if strat == 'drop':
            # Drop rows with missing values in this column
            result_df = result_df.dropna(subset=[column])
        elif strat == 'mean' and pd.api.types.is_numeric_dtype(result_df[column]):
            # Fill with mean (numeric columns only)
            result_df[column] = result_df[column].fillna(result_df[column].mean())
        elif strat == 'median' and pd.api.types.is_numeric_dtype(result_df[column]):
            # Fill with median (numeric columns only)
            result_df[column] = result_df[column].fillna(result_df[column].median())
        elif strat == 'mode':
            # Fill with mode (most frequent value)
            mode_value = result_df[column].mode()
            if not mode_value.empty:
                result_df[column] = result_df[column].fillna(mode_value[0])
        elif strat == 'zero' and pd.api.types.is_numeric_dtype(result_df[column]):
            # Fill with zero (numeric columns only)
            result_df[column] = result_df[column].fillna(0)
        elif strat.startswith('value:'):
            # Fill with specific value
            fill_value = strat.split(':', 1)[1]
            # Convert fill_value to appropriate type
            if pd.api.types.is_numeric_dtype(result_df[column]):
                try:
                    fill_value = float(fill_value)
                except ValueError:
                    logger.warning(f"Could not convert {fill_value} to numeric for column {column}")
                    continue
            result_df[column] = result_df[column].fillna(fill_value)
    
    return result_df


def handle_outliers(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
    """
    Handle outliers in the DataFrame using specified strategies.
    
    Args:
        df: Pandas DataFrame to process
        strategy: Dict mapping column names to strategies
                  ('clip', 'remove', 'iqr')
        
    Returns:
        Processed DataFrame with handled outliers
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    for column, strat in strategy.items():
        if column not in result_df.columns or not pd.api.types.is_numeric_dtype(result_df[column]):
            continue
        
        # Convert to float for consistency with PyArrow
        series = result_df[column].astype(float)
        
        # Calculate IQR bounds
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        if strat == 'clip':
            # Clip values to IQR bounds
            result_df[column] = series.clip(lower_bound, upper_bound)
        elif strat == 'remove':
            # Remove rows with outliers
            result_df = result_df[(series >= lower_bound) & (series <= upper_bound)]
        elif strat == 'iqr':
            # Set outliers to NaN
            result_df.loc[(series < lower_bound) | (series > upper_bound), column] = np.nan
    
    return result_df


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    
    Args:
        df: Pandas DataFrame to process
        subset: Optional list of columns to consider for identifying duplicates
        
    Returns:
        DataFrame with duplicates removed
    """
    if df is None or df.empty:
        return df
    
    return df.drop_duplicates(subset=subset)


def fix_data_types(df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Pandas DataFrame to process
        type_map: Dict mapping column names to data types
                  ('int', 'float', 'str', 'datetime', 'category')
        
    Returns:
        DataFrame with corrected data types
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    for column, data_type in type_map.items():
        if column not in result_df.columns:
            continue
        
        try:
            if data_type == 'int':
                # Handle NaN values for integer conversion
                result_df[column] = pd.to_numeric(result_df[column], errors='coerce')
                result_df[column] = result_df[column].fillna(0).astype(int)
            elif data_type == 'float':
                result_df[column] = pd.to_numeric(result_df[column], errors='coerce')
            elif data_type == 'str':
                result_df[column] = result_df[column].astype(str)
            elif data_type == 'datetime':
                # Convert to datetime but handle PyArrow compatibility issues
                result_df[column] = pd.to_datetime(result_df[column], errors='coerce')
                # Convert to string representation to avoid PyArrow conversion issues
                if pd.api.types.is_datetime64_dtype(result_df[column]):
                    result_df[column] = result_df[column].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif data_type == 'category':
                result_df[column] = result_df[column].astype('category')
        except Exception as e:
            logger.warning(f"Could not convert column {column} to {data_type}: {str(e)}")
    
    return result_df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names by removing special characters, 
    converting to lowercase, and replacing spaces with underscores.
    
    Args:
        df: Pandas DataFrame to process
        
    Returns:
        DataFrame with standardized column names
    """
    if df is None or df.empty:
        return df
    
    import re
    
    # Define cleaning function for column names
    def clean_column_name(name):
        # Convert to lowercase
        name = name.lower()
        # Replace spaces and special characters with underscore
        name = re.sub(r'[^a-z0-9]', '_', name)
        # Replace multiple underscores with single underscore
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name
    
    # Create a new DataFrame with cleaned column names
    result_df = df.copy()
    result_df.columns = [clean_column_name(col) for col in result_df.columns]
    
    return result_df


def apply_column_transformations(df: pd.DataFrame, transformations: Dict[str, str]) -> pd.DataFrame:
    """
    Apply custom transformations to specific columns.
    
    Args:
        df: Pandas DataFrame to process
        transformations: Dict mapping column names to transformation expressions
                         (e.g., 'column': 'log', 'column': 'normalize', 'column': 'uppercase')
        
    Returns:
        DataFrame with transformed columns
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    for column, transform in transformations.items():
        if column not in result_df.columns:
            continue
        
        try:
            if transform == 'log' and pd.api.types.is_numeric_dtype(result_df[column]):
                # Apply logarithmic transformation (handle zeros by adding small constant)
                result_df[column] = np.log1p(result_df[column])
            elif transform == 'sqrt' and pd.api.types.is_numeric_dtype(result_df[column]):
                # Apply square root transformation (handle negative values)
                result_df[column] = np.sqrt(np.maximum(result_df[column], 0))
            elif transform == 'normalize' and pd.api.types.is_numeric_dtype(result_df[column]):
                # Normalize to [0,1] range
                min_val = result_df[column].min()
                max_val = result_df[column].max()
                if max_val > min_val:
                    result_df[column] = (result_df[column] - min_val) / (max_val - min_val)
            elif transform == 'standardize' and pd.api.types.is_numeric_dtype(result_df[column]):
                # Standardize to mean=0, std=1
                mean = result_df[column].mean()
                std = result_df[column].std()
                if std > 0:
                    result_df[column] = (result_df[column] - mean) / std
            elif transform == 'uppercase' and pd.api.types.is_string_dtype(result_df[column]):
                # Convert to uppercase
                result_df[column] = result_df[column].str.upper()
            elif transform == 'lowercase' and pd.api.types.is_string_dtype(result_df[column]):
                # Convert to lowercase
                result_df[column] = result_df[column].str.lower()
            elif transform == 'title' and pd.api.types.is_string_dtype(result_df[column]):
                # Convert to title case
                result_df[column] = result_df[column].str.title()
            elif transform == 'trim' and pd.api.types.is_string_dtype(result_df[column]):
                # Trim whitespace
                result_df[column] = result_df[column].str.strip()
        except Exception as e:
            logger.warning(f"Could not apply transformation {transform} to column {column}: {str(e)}")
    
    return result_df


def create_derived_features(df: pd.DataFrame, derivations: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create new derived features based on existing columns.
    
    Args:
        df: Pandas DataFrame to process
        derivations: Dict mapping new column names to derivation specifications
                     (e.g., {'age_group': {'type': 'bin', 'source': 'age', 'bins': [0, 18, 65, 100]}})
        
    Returns:
        DataFrame with added derived features
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    for new_col, spec in derivations.items():
        try:
            if 'source' not in spec or spec['source'] not in result_df.columns:
                continue
                
            source_col = spec['source']
            
            if spec.get('type') == 'bin' and pd.api.types.is_numeric_dtype(result_df[source_col]):
                # Create binned categories
                bins = spec.get('bins', [])
                labels = spec.get('labels', None)
                if len(bins) > 1:
                    result_df[new_col] = pd.cut(result_df[source_col], bins=bins, labels=labels)
            
            elif spec.get('type') == 'extract_year' and pd.api.types.is_datetime64_dtype(result_df[source_col]):
                # Extract year from datetime and convert to int to avoid PyArrow issues
                result_df[new_col] = result_df[source_col].dt.year.astype(int)
                
            elif spec.get('type') == 'extract_month' and pd.api.types.is_datetime64_dtype(result_df[source_col]):
                # Extract month from datetime and convert to int to avoid PyArrow issues
                result_df[new_col] = result_df[source_col].dt.month.astype(int)
                
            elif spec.get('type') == 'extract_day' and pd.api.types.is_datetime64_dtype(result_df[source_col]):
                # Extract day from datetime and convert to int to avoid PyArrow issues
                result_df[new_col] = result_df[source_col].dt.day.astype(int)
                
            elif spec.get('type') == 'combine' and 'sources' in spec:
                # Combine multiple columns
                separator = spec.get('separator', '_')
                sources = spec.get('sources', [])
                if all(col in result_df.columns for col in sources):
                    result_df[new_col] = result_df[sources].astype(str).agg(separator.join, axis=1)
                    
            elif spec.get('type') == 'math' and 'operation' in spec:
                # Mathematical operations
                operation = spec.get('operation')
                sources = spec.get('sources', [])
                
                if operation == 'add' and len(sources) == 2 and all(col in result_df.columns for col in sources):
                    result_df[new_col] = result_df[sources[0]] + result_df[sources[1]]
                elif operation == 'subtract' and len(sources) == 2 and all(col in result_df.columns for col in sources):
                    result_df[new_col] = result_df[sources[0]] - result_df[sources[1]]
                elif operation == 'multiply' and len(sources) == 2 and all(col in result_df.columns for col in sources):
                    result_df[new_col] = result_df[sources[0]] * result_df[sources[1]]
                elif operation == 'divide' and len(sources) == 2 and all(col in result_df.columns for col in sources):
                    # Handle division by zero
                    result_df[new_col] = result_df[sources[0]] / result_df[sources[1]].replace(0, np.nan)
                
        except Exception as e:
            logger.warning(f"Could not create derived feature {new_col}: {str(e)}")
    
    return result_df


def filter_rows(df: pd.DataFrame, filters: Dict[str, Dict]) -> pd.DataFrame:
    """
    Filter rows based on specified conditions.
    
    Args:
        df: Pandas DataFrame to process
        filters: Dict mapping column names to filter specifications
                 (e.g., {'age': {'operator': '>', 'value': 18}})
        
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty or not filters:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Build a mask for all filter conditions
    mask = pd.Series(True, index=result_df.index)
    
    for column, condition in filters.items():
        if column not in result_df.columns:
            continue
        
        operator = condition.get('operator', '==')
        value = condition.get('value')
        
        if value is None:
            continue
        
        try:
            if operator == '==':
                mask &= result_df[column] == value
            elif operator == '!=':
                mask &= result_df[column] != value
            elif operator == '>':
                mask &= result_df[column] > value
            elif operator == '>=':
                mask &= result_df[column] >= value
            elif operator == '<':
                mask &= result_df[column] < value
            elif operator == '<=':
                mask &= result_df[column] <= value
            elif operator == 'in':
                if isinstance(value, list):
                    mask &= result_df[column].isin(value)
            elif operator == 'not in':
                if isinstance(value, list):
                    mask &= ~result_df[column].isin(value)
            elif operator == 'contains' and pd.api.types.is_string_dtype(result_df[column]):
                mask &= result_df[column].str.contains(str(value), na=False)
            elif operator == 'starts with' and pd.api.types.is_string_dtype(result_df[column]):
                mask &= result_df[column].str.startswith(str(value), na=False)
            elif operator == 'ends with' and pd.api.types.is_string_dtype(result_df[column]):
                mask &= result_df[column].str.endswith(str(value), na=False)
        except Exception as e:
            logger.warning(f"Could not apply filter on column {column}: {str(e)}")
    
    return result_df[mask]