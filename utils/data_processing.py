import pandas as pd
import streamlit as st
import io
import re

def process_uploaded_files(uploaded_file):
    """
    Process uploaded CSV or Excel files and return a pandas DataFrame.
    
    Args:
        uploaded_file: The file uploaded via Streamlit's file_uploader
        
    Returns:
        pandas.DataFrame: The processed DataFrame
    """
    try:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Process based on file type
        if file_extension == 'csv':
            # Try different encodings and delimiters
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                except Exception:
                    df = pd.read_csv(uploaded_file, encoding='cp1252')
            except Exception as e:
                # Try different delimiters if standard comma fails
                try:
                    df = pd.read_csv(uploaded_file, sep=';')
                except Exception:
                    raise Exception(f"Failed to parse CSV file: {str(e)}")
                
        elif file_extension in ['xlsx', 'xls']:
            # Read Excel file
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                raise Exception(f"Failed to parse Excel file: {str(e)}")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Clean up the column names
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Basic data cleaning
        df = clean_dataframe(df)
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return None

def clean_column_name(column_name):
    """
    Clean up column names to make them more consistent and usable.
    
    Args:
        column_name: The original column name
        
    Returns:
        str: The cleaned column name
    """
    # Convert to string in case it's not already
    col = str(column_name).strip()
    
    # Replace spaces with underscores and remove special characters
    col = re.sub(r'[^\w\s]', '', col)
    col = col.replace(' ', '_').lower()
    
    # Remove duplicate underscores
    col = re.sub(r'_+', '_', col)
    
    return col

def clean_dataframe(df):
    """
    Perform basic data cleaning on the DataFrame.
    
    Args:
        df: The pandas DataFrame to clean
        
    Returns:
        pandas.DataFrame: The cleaned DataFrame
    """
    # Make a copy to avoid warnings
    cleaned_df = df.copy()
    
    # Remove completely empty rows
    cleaned_df = cleaned_df.dropna(how='all')
    
    # Remove completely empty columns
    cleaned_df = cleaned_df.dropna(axis=1, how='all')
    
    # Try to convert appropriate columns to numeric
    for col in cleaned_df.columns:
        # Only try to convert if the column is not already numeric
        if cleaned_df[col].dtype == 'object':
            # Check if it might be a numeric column with some non-numeric values
            try:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            except:
                pass  # Keep as is if conversion fails
    
    # Handle date columns - try to convert any columns with 'date' in the name
    date_cols = [col for col in cleaned_df.columns if 'date' in col.lower()]
    for col in date_cols:
        try:
            # First try to convert to datetime
            date_series = pd.to_datetime(cleaned_df[col], errors='coerce')
            
            # Convert datetime columns to string format for PyArrow compatibility
            # This prevents the PyArrow error with datetime64 dtype
            if date_series.notna().any():
                # If successful, convert to string format with ISO format for compatibility
                cleaned_df[col] = date_series.dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                # If conversion failed (all NaT), keep as original
                pass
        except Exception as e:
            print(f"Failed to convert column {col} to datetime: {str(e)}")
            pass  # Keep as is if conversion fails
    
    # Handle any existing datetime columns (for PyArrow compatibility)
    for col in cleaned_df.columns:
        if pd.api.types.is_datetime64_dtype(cleaned_df[col]):
            try:
                # Convert to string to avoid PyArrow errors
                cleaned_df[col] = cleaned_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                # If conversion fails, convert to string representation
                cleaned_df[col] = cleaned_df[col].astype(str)
    
    return cleaned_df

def combine_dataframes(dataframes):
    """
    Combine multiple DataFrames into one, handling different columns and structures.
    
    Args:
        dataframes: List of pandas DataFrames to combine
        
    Returns:
        pandas.DataFrame: The combined DataFrame
    """
    if not dataframes:
        return pd.DataFrame()
    
    if len(dataframes) == 1:
        return dataframes[0]
    
    # Check if DataFrames have the same structure
    all_columns = set()
    for df in dataframes:
        all_columns.update(df.columns)
    
    # If all DataFrames have the same columns, simple concatenation
    if all(set(df.columns) == all_columns for df in dataframes):
        return pd.concat(dataframes, ignore_index=True)
    
    # If DataFrames have different columns, add missing columns with NaN values
    combined_dfs = []
    for df in dataframes:
        missing_cols = all_columns - set(df.columns)
        temp_df = df.copy()
        for col in missing_cols:
            temp_df[col] = None
        combined_dfs.append(temp_df)
    
    # Combine the DataFrames
    return pd.concat(combined_dfs, ignore_index=True)
