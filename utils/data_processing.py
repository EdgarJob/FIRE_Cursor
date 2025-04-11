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
        
        # Add a debug message
        st.info(f"Processing {uploaded_file.name} (Format: {file_extension})")
        
        # Process based on file type
        if file_extension == 'csv':
            # Try different encodings and delimiters
            for encoding in ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']:
                for delimiter in [',', ';', '\t', '|']:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding, sep=delimiter)
                        # If successful, display info and break
                        st.success(f"Successfully parsed CSV with encoding: {encoding}, delimiter: '{delimiter}'")
                        break
                    except Exception as e:
                        # Reset file pointer to start for next attempt
                        uploaded_file.seek(0)
                        continue
                
                # If we got a DataFrame, break out of the encoding loop too
                if 'df' in locals():
                    break
            
            if 'df' not in locals():
                # If all attempts failed, try one more time with pandas' auto-detection
                try:
                    # Read a small sample to detect delimiter
                    content_sample = uploaded_file.read(1024).decode('utf-8', errors='replace')
                    uploaded_file.seek(0)
                    
                    # Count potential delimiters
                    delimiters = [',', ';', '\t', '|']
                    counts = {d: content_sample.count(d) for d in delimiters}
                    most_likely = max(counts, key=counts.get)
                    
                    df = pd.read_csv(uploaded_file, sep=most_likely, encoding='utf-8', error_bad_lines=False)
                    st.warning(f"Used auto-detection to parse file. Detected delimiter: '{most_likely}'")
                except Exception as final_e:
                    raise Exception(f"Failed to parse CSV file after multiple attempts: {str(final_e)}")
                
        elif file_extension in ['xlsx', 'xls']:
            # Read Excel file
            try:
                # Try to get sheet names
                with pd.ExcelFile(uploaded_file) as xls:
                    sheet_names = xls.sheet_names
                    
                if len(sheet_names) > 1:
                    # If multiple sheets, let user select
                    selected_sheet = st.selectbox(
                        f"Multiple sheets found in {uploaded_file.name}. Select a sheet:",
                        options=sheet_names
                    )
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    st.success(f"Loaded sheet: {selected_sheet}")
                else:
                    # Just one sheet, load it
                    df = pd.read_excel(uploaded_file)
                    st.success(f"Loaded Excel file with sheet: {sheet_names[0]}")
            except Exception as e:
                # Try with engine specification
                try:
                    engine = 'openpyxl' if file_extension == 'xlsx' else 'xlrd'
                    df = pd.read_excel(uploaded_file, engine=engine)
                    st.success(f"Loaded Excel file using {engine} engine")
                except Exception as e2:
                    raise Exception(f"Failed to parse Excel file: {str(e)}. Second attempt error: {str(e2)}")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Check if DataFrame is empty
        if df.empty:
            st.warning("The uploaded file contains no data.")
            return pd.DataFrame()
            
        # Display basic file info
        st.info(f"File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Clean up the column names
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Basic data cleaning
        df = clean_dataframe(df)
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        # Show a more detailed error message for debugging
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
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
    
    # Handle duplicate columns by renaming
    if len(cleaned_df.columns) != len(set(cleaned_df.columns)):
        # Find duplicates
        seen = {}
        new_columns = []
        for col in cleaned_df.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        
        # Rename columns
        cleaned_df.columns = new_columns
    
    # Try to convert appropriate columns to numeric
    for col in cleaned_df.columns:
        # Only try to convert if the column is not already numeric
        if cleaned_df[col].dtype == 'object':
            # Check if the column might be numeric (at least 60% of non-null values are numeric)
            sample = cleaned_df[col].dropna().head(100)
            if len(sample) == 0:
                continue
                
            try:
                # Count how many values can be converted to numeric
                numeric_count = sum(pd.to_numeric(sample, errors='coerce').notna())
                if numeric_count / len(sample) >= 0.6:  # If at least 60% can be numeric
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            except:
                pass  # Keep as is if the test fails
    
    # Handle date columns - look for patterns beyond just column names
    date_patterns = [
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
        r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # DD-MM-YYYY or DD/MM/YYYY
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2}',  # DD-MM-YY or DD/MM/YY
    ]
    
    # Find potential date columns by either name or content
    potential_date_cols = []
    
    # First, check by column name
    date_name_indicators = ['date', 'day', 'month', 'year', 'time', 'when']
    for col in cleaned_df.columns:
        if any(indicator in col.lower() for indicator in date_name_indicators):
            potential_date_cols.append(col)
    
    # Then check for date patterns in string columns
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            # Skip columns already identified as potential dates
            if col in potential_date_cols:
                continue
                
            # Take a sample of values to check for date patterns
            sample = cleaned_df[col].dropna().head(20).astype(str)
            if len(sample) == 0:
                continue
                
            # Check if column values match date patterns
            matches = 0
            for val in sample:
                if any(re.search(pattern, val) for pattern in date_patterns):
                    matches += 1
            
            if matches / len(sample) >= 0.5:  # If at least 50% match date patterns
                potential_date_cols.append(col)
    
    # Now try to convert potential date columns
    for col in potential_date_cols:
        try:
            # First try pandas' auto-detection
            date_series = pd.to_datetime(cleaned_df[col], errors='coerce')
            
            # If most values converted successfully (at least 70%)
            success_rate = date_series.notna().mean()
            if success_rate >= 0.7:
                # Format dates consistently
                cleaned_df[col] = date_series.dt.strftime('%Y-%m-%d')
            
        except Exception as e:
            print(f"Failed to convert column {col} to datetime: {str(e)}")
            pass  # Keep as is if conversion fails
    
    # Handle any existing datetime columns (for PyArrow compatibility)
    for col in cleaned_df.columns:
        if pd.api.types.is_datetime64_dtype(cleaned_df[col]):
            try:
                # Convert to string to avoid PyArrow errors
                cleaned_df[col] = cleaned_df[col].dt.strftime('%Y-%m-%d')
            except:
                # If conversion fails, convert to string representation
                cleaned_df[col] = cleaned_df[col].astype(str)
    
    # Replace problematic values that might cause issues
    # Replace inf/-inf with NaN
    for col in cleaned_df.select_dtypes(include=['float64', 'float32']).columns:
        cleaned_df[col] = cleaned_df[col].replace([float('inf'), float('-inf')], None)
    
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
