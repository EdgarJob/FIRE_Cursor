import streamlit as st
import pandas as pd
import os
import time
import json
from utils.data_processing import process_uploaded_files, combine_dataframes
from utils.nlp_processing import process_query
from utils.visualization import create_visualization
from utils.export import export_to_excel, export_to_csv, export_to_pdf
from utils.data_cleaning import (get_data_quality_report, fix_missing_values, 
                                handle_outliers, remove_duplicates, fix_data_types,
                                standardize_column_names, apply_column_transformations,
                                create_derived_features, filter_rows)
from templates.analysis_templates import get_templates, apply_template
from utils.database import (
    save_file_to_db, get_file_from_db, get_all_files, load_dataframe_from_file,
    save_query_to_db, get_query_history, save_visualization_to_db, 
    get_visualization_history, save_user_preference, get_user_preference
)
from utils.document_processing import document_processor
from utils.rag_engine import rag_engine
from utils.smart_dataframe import SmartDataFrame

# Ensure temp directory exists
os.makedirs("tmp", exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="FIRE - Field Insight & Reporting Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = {}  # Dictionary to store DataFrames
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'visualization_history' not in st.session_state:
    st.session_state.visualization_history = []
if 'selected_files' not in st.session_state:
    st.session_state.selected_files = []
if 'query_explanation' not in st.session_state:
    st.session_state.query_explanation = ""

def main():
    # Header
    st.title("FIRE: Field Insight & Reporting Engine")
    st.markdown("Upload field data, analyze with natural language queries, and export insights")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Data Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload CSV or Excel files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                newly_uploaded = [f for f in uploaded_files if f.name not in st.session_state.selected_files]
                if newly_uploaded:
                    for file in newly_uploaded:
                        df = process_uploaded_files(file)
                        if df is not None:
                            # Save file to database
                            try:
                                file_id = save_file_to_db(file, file.name)
                                # Store file metadata in session state
                                if 'file_ids' not in st.session_state:
                                    st.session_state.file_ids = {}
                                st.session_state.file_ids[file.name] = file_id
                            except Exception as e:
                                st.warning(f"File saved in memory but could not be stored in database: {str(e)}")
                            
                            # Store in memory for current session
                            st.session_state.data[file.name] = df
                            if file.name not in st.session_state.selected_files:
                                st.session_state.selected_files.append(file.name)
                    
                    # Set the current dataframe to the last uploaded file if it's the first upload
                    if st.session_state.current_df is None and st.session_state.selected_files:
                        st.session_state.current_df = st.session_state.selected_files[0]
            
            # Load previously uploaded files from database if not already loaded
            if 'db_files_loaded' not in st.session_state:
                try:
                    # Get all files from database
                    db_files = get_all_files()
                    for file_info in db_files:
                        filename = file_info['filename']
                        # If file not already in session, load it
                        if filename not in st.session_state.selected_files:
                            df = load_dataframe_from_file(file_info['id'])
                            if df is not None:
                                st.session_state.data[filename] = df
                                st.session_state.selected_files.append(filename)
                                if 'file_ids' not in st.session_state:
                                    st.session_state.file_ids = {}
                                st.session_state.file_ids[filename] = file_info['id']
                    st.session_state.db_files_loaded = True
                except Exception as e:
                    st.warning(f"Could not load previously uploaded files from database: {str(e)}")
        
        # File selector
        if st.session_state.selected_files:
            st.subheader("Select Data Source")
            selected_option = st.radio(
                "Choose files to analyze:",
                ["Single File", "Multiple Files (Combined)"]
            )
            
            if selected_option == "Single File":
                selected_file = st.selectbox(
                    "Select a file:",
                    st.session_state.selected_files
                )
                st.session_state.current_df = selected_file
            else:
                files_to_combine = st.multiselect(
                    "Select files to combine:",
                    st.session_state.selected_files,
                    default=st.session_state.selected_files
                )
                
                if files_to_combine and st.button("Combine Selected Files"):
                    with st.spinner("Combining files..."):
                        dfs = [st.session_state.data[file] for file in files_to_combine]
                        combined_df = combine_dataframes(dfs)
                        st.session_state.data["Combined Data"] = combined_df
                        st.session_state.current_df = "Combined Data"
                        if "Combined Data" not in st.session_state.selected_files:
                            st.session_state.selected_files.append("Combined Data")
                        st.success("Files combined successfully!")
                        st.rerun()
        
        # Help section
        with st.expander("Help & Tips"):
            st.markdown("""
            **How to use FIRE:**
            1. Upload your CSV or Excel files
            2. Select a single file or combine multiple files
            3. Ask questions in natural language or use templates
            4. Explore visualizations
            5. Export your insights
            
            **Example queries:**
            - "Show me the age distribution"
            - "What is the average income by region?"
            - "Count beneficiaries by gender"
            """)

    # Main content area
    if not st.session_state.selected_files:
        st.info("👈 Please upload your data files from the sidebar to get started")
        
        # Show demo section
        st.subheader("What can FIRE do for you?")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Analyze Data")
            st.markdown("Upload field data and analyze it using simple, natural language queries")
        
        with col2:
            st.markdown("### 📈 Visualize Insights")
            st.markdown("Create charts and graphs to better understand your data")
        
        with col3:
            st.markdown("### 📑 Export Reports")
            st.markdown("Export your findings as Excel, CSV, or PDF reports")
    else:
        # Show selected dataset info
        if st.session_state.current_df:
            current_data = st.session_state.data[st.session_state.current_df]
            
            st.subheader(f"Currently analyzing: {st.session_state.current_df}")
            
            # Data overview
            with st.expander("Data Overview", expanded=True):
                st.write(f"Rows: {current_data.shape[0]}, Columns: {current_data.shape[1]}")
                
                # Convert the dataframe to a safer format for display
                try:
                    # Create a safe display copy 
                    display_df = current_data.head(5).copy()
                    
                    # Convert datetime columns to string to avoid PyArrow errors
                    for col in display_df.columns:
                        # Handle datetime types
                        if pd.api.types.is_datetime64_dtype(display_df[col]):
                            display_df[col] = display_df[col].astype(str)
                        # Handle any other problematic types
                        elif not pd.api.types.is_numeric_dtype(display_df[col]) and not pd.api.types.is_string_dtype(display_df[col]):
                            display_df[col] = display_df[col].astype(str)
                    
                    # Use a safer method to display
                    st.write(display_df)
                except Exception as e:
                    st.warning(f"Error displaying data preview: {str(e)}")
                    # Fallback to displaying as HTML
                    st.write("Data Preview (first 5 rows):")
                    st.write(current_data.head(5).to_html(index=False), unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Column Names:")
                    st.write(", ".join(current_data.columns.tolist()))
                with col2:
                    st.write("Data Types:")
                    # Convert dtypes to strings for display
                    dtypes_list = [f"{col}: {dtype}" for col, dtype in zip(current_data.columns, current_data.dtypes)]
                    st.write("\n".join(dtypes_list))
        
            # Analysis section with tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Natural Language Queries", "Analysis Templates", "Visualizations", "Data Cleaning", "Document Analysis", "Export"])
            
            # Tab 1: Natural Language Queries
            with tab1:
                st.subheader("Ask Questions About Your Data")
                
                query = st.text_input("Enter your question in natural language:", 
                                    placeholder="E.g., 'Show me the average age by gender'")
                
                if st.button("Submit Query"):
                    if query:
                        with st.spinner("Processing your query..."):
                            try:
                                # Add a message to let the user know it's connecting to AI
                                ai_status = st.empty()
                                ai_status.info("Connecting to AI service... This may take a few moments.")
                                
                                result, viz_type = process_query(query, current_data)
                                
                                # Clear the connecting message
                                ai_status.empty()
                                
                                # Store query in memory
                                st.session_state.query_history.append({"query": query, "result": result})
                                
                                # Store query in database
                                try:
                                    # Get the current file ID from session state
                                    file_id = None
                                    if 'file_ids' in st.session_state and st.session_state.current_df in st.session_state.file_ids:
                                        file_id = st.session_state.file_ids[st.session_state.current_df]
                                    
                                    # Save query to database
                                    query_id = save_query_to_db(query, result, file_id, viz_type)
                                    
                                    # Save this query_id for visualization linking
                                    if 'query_ids' not in st.session_state:
                                        st.session_state.query_ids = {}
                                    st.session_state.query_ids[query] = query_id
                                except Exception as e:
                                    st.warning(f"Query saved in memory but could not be stored in database: {str(e)}")
                                
                                st.subheader("Results")
                                
                                # Display AI-generated explanation of the query interpretation if available
                                if hasattr(st.session_state, 'query_explanation') and st.session_state.query_explanation:
                                    with st.expander("How AI understood your query", expanded=True):
                                        st.info(st.session_state.query_explanation)
                                    # Reset for next query
                                    st.session_state.query_explanation = ""
                                
                                # Check if result is empty or None
                                if result is None or (hasattr(result, 'empty') and result.empty):
                                    st.warning("No data found that matches your query. Try rephrasing or selecting different columns.")
                                else:
                                    st.write(result)
                                    
                                    # Create visualization based on the query result
                                    if viz_type:
                                        try:
                                            fig = create_visualization(result, viz_type, query)
                                            st.plotly_chart(fig, use_container_width=True, key=f"nlq_viz_{len(query)}")
                                            
                                            # Store visualization in memory
                                            st.session_state.visualization_history.append({
                                                "query": query,
                                                "viz_type": viz_type,
                                                "result": result
                                            })
                                            
                                            # Store visualization in database
                                            try:
                                                query_id = st.session_state.query_ids.get(query)
                                                if query_id:
                                                    save_visualization_to_db(query_id, viz_type, result)
                                            except Exception as e:
                                                st.warning(f"Visualization saved in memory but could not be stored in database: {str(e)}")
                                        except Exception as viz_err:
                                            st.error(f"Error creating visualization: {str(viz_err)}")
                                            st.info("The data was processed successfully, but could not be visualized due to an error.")
                            except Exception as e:
                                st.error(f"Error processing query: {str(e)}")
                                if "API" in str(e) or "OpenRouter" in str(e) or "openai" in str(e).lower():
                                    st.warning("There was an issue with the AI service connection. The service may be temporarily unavailable, or there may be an issue with your API key. Using rule-based processing as a fallback.")
                                st.info("Try simplifying your query or using different keywords. You can also use the Analysis Templates for pre-built analyses.")
                    else:
                        st.warning("Please enter a query first.")
                
                # Query history
                if st.session_state.query_history:
                    with st.expander("Query History", expanded=False):
                        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                            st.markdown(f"**Query {len(st.session_state.query_history)-i}:** {item['query']}")
                            st.write(item['result'])
                            st.divider()
            
            # Tab 2: Analysis Templates
            with tab2:
                st.subheader("Analysis Templates")
                
                templates = get_templates()
                selected_template = st.selectbox(
                    "Choose an analysis template:",
                    list(templates.keys())
                )
                
                if st.button("Apply Template"):
                    with st.spinner("Applying template..."):
                        try:
                            result, viz_type, description = apply_template(selected_template, current_data)
                            
                            st.markdown(f"**{description}**")
                            st.write(result)
                            
                            # Create visualization
                            fig = create_visualization(result, viz_type, selected_template)
                            st.plotly_chart(fig, use_container_width=True, key=f"template_viz_{selected_template}")
                            
                            # Store in visualization history
                            st.session_state.visualization_history.append({
                                "query": f"Template: {selected_template}",
                                "viz_type": viz_type,
                                "result": result
                            })
                        except Exception as e:
                            st.error(f"Error applying template: {str(e)}")
            
            # Tab 3: Visualizations
            with tab3:
                st.subheader("Visualization Gallery")
                
                if not st.session_state.visualization_history:
                    st.info("No visualizations yet. Use the Natural Language Queries or Analysis Templates to create visualizations.")
                else:
                    # Display the most recent visualizations
                    for i, viz in enumerate(reversed(st.session_state.visualization_history[-5:])):
                        with st.expander(f"{viz['query']}", expanded=(i==0)):
                            fig = create_visualization(viz['result'], viz['viz_type'], viz['query'])
                            st.plotly_chart(fig, use_container_width=True, key=f"gallery_viz_{i}")
                            
                            # Options to modify the visualization
                            col1, col2 = st.columns(2)
                            with col1:
                                new_viz_type = st.selectbox(
                                    "Change visualization type:",
                                    ["bar", "line", "pie", "scatter", "histogram"],
                                    key=f"viz_type_{i}"
                                )
                            with col2:
                                if st.button("Update Visualization", key=f"update_viz_{i}"):
                                    fig = create_visualization(viz['result'], new_viz_type, viz['query'])
                                    st.session_state.visualization_history[-(i+1)]['viz_type'] = new_viz_type
                                    st.plotly_chart(fig, use_container_width=True, key=f"updated_viz_{i}")
            
            # Tab 4: Data Cleaning
            with tab4:
                st.subheader("Advanced Data Cleaning and Preprocessing")
                
                # Initialize session state for cleaned dataframe
                if 'cleaned_df' not in st.session_state:
                    st.session_state.cleaned_df = current_data.copy()
                
                # Show data quality report
                with st.expander("Data Quality Report", expanded=True):
                    if st.button("Generate Data Quality Report"):
                        with st.spinner("Analyzing data quality..."):
                            quality_report = get_data_quality_report(current_data)
                            
                            if quality_report["status"] == "error":
                                st.error(quality_report["error"])
                            else:
                                # General statistics
                                st.markdown("### General Statistics")
                                st.write(f"- Total rows: {quality_report['row_count']}")
                                st.write(f"- Total columns: {quality_report['column_count']}")
                                st.write(f"- Duplicate rows: {quality_report['duplicates']['count']} ({quality_report['duplicates']['percentage']:.2f}%)")
                                
                                # Missing values
                                if quality_report["missing_values"]:
                                    st.markdown("### Missing Values")
                                    missing_df = pd.DataFrame({
                                        "Column": list(quality_report["missing_values"].keys()),
                                        "Count": [m["count"] for m in quality_report["missing_values"].values()],
                                        "Percentage": [f"{m['percentage']:.2f}%" for m in quality_report["missing_values"].values()]
                                    })
                                    st.dataframe(missing_df)
                                
                                # Outliers
                                if quality_report["outliers"]:
                                    st.markdown("### Outliers")
                                    outlier_df = pd.DataFrame({
                                        "Column": list(quality_report["outliers"].keys()),
                                        "Count": [o["count"] for o in quality_report["outliers"].values()],
                                        "Percentage": [f"{o['percentage']:.2f}%" for o in quality_report["outliers"].values()],
                                        "Lower Bound": [o["lower_bound"] for o in quality_report["outliers"].values()],
                                        "Upper Bound": [o["upper_bound"] for o in quality_report["outliers"].values()]
                                    })
                                    st.dataframe(outlier_df)
                
                # Missing value handling
                with st.expander("Handle Missing Values", expanded=False):
                    st.markdown("### Fix Missing Values")
                    st.write("Select columns and strategies to handle missing values")
                    
                    # Get columns with missing values
                    missing_cols = [col for col in current_data.columns if current_data[col].isna().any()]
                    
                    if not missing_cols:
                        st.success("No missing values detected in the dataset!")
                    else:
                        # Create multiselect for columns
                        selected_missing_cols = st.multiselect(
                            "Select columns with missing values to fix:",
                            missing_cols,
                            default=missing_cols[:min(3, len(missing_cols))]
                        )
                        
                        # Define strategies for each selected column
                        missing_strategies = {}
                        for col in selected_missing_cols:
                            is_numeric = pd.api.types.is_numeric_dtype(current_data[col])
                            
                            strategy_options = ["drop"]
                            if is_numeric:
                                strategy_options.extend(["mean", "median", "zero"])
                            strategy_options.extend(["mode", "custom value"])
                            
                            strategy = st.selectbox(
                                f"Strategy for {col}:",
                                strategy_options,
                                key=f"missing_strategy_{col}"
                            )
                            
                            if strategy == "custom value":
                                custom_value = st.text_input(
                                    f"Enter custom value for {col}:",
                                    key=f"custom_value_{col}"
                                )
                                missing_strategies[col] = f"value:{custom_value}"
                            else:
                                missing_strategies[col] = strategy
                        
                        if st.button("Apply Missing Value Fixes"):
                            with st.spinner("Fixing missing values..."):
                                try:
                                    fixed_df = fix_missing_values(current_data, missing_strategies)
                                    st.session_state.cleaned_df = fixed_df
                                    st.success(f"Successfully fixed missing values in {len(selected_missing_cols)} column(s)")
                                    st.write("Preview of cleaned data:")
                                    st.dataframe(fixed_df.head())
                                except Exception as e:
                                    st.error(f"Error fixing missing values: {str(e)}")
                
                # Outlier handling
                with st.expander("Handle Outliers", expanded=False):
                    st.markdown("### Fix Outliers")
                    st.write("Select columns and strategies to handle outliers")
                    
                    # Get numeric columns
                    numeric_cols = [col for col in current_data.columns if pd.api.types.is_numeric_dtype(current_data[col])]
                    
                    if not numeric_cols:
                        st.info("No numeric columns detected for outlier handling.")
                    else:
                        # Create multiselect for columns
                        selected_outlier_cols = st.multiselect(
                            "Select numeric columns to check for outliers:",
                            numeric_cols,
                            default=numeric_cols[:min(3, len(numeric_cols))]
                        )
                        
                        # Define strategies for each selected column
                        outlier_strategies = {}
                        for col in selected_outlier_cols:
                            strategy = st.selectbox(
                                f"Strategy for {col}:",
                                ["clip", "remove", "iqr"],
                                key=f"outlier_strategy_{col}"
                            )
                            outlier_strategies[col] = strategy
                        
                        if st.button("Apply Outlier Fixes"):
                            with st.spinner("Handling outliers..."):
                                try:
                                    fixed_df = handle_outliers(
                                        st.session_state.cleaned_df 
                                        if 'cleaned_df' in st.session_state else current_data, 
                                        outlier_strategies
                                    )
                                    st.session_state.cleaned_df = fixed_df
                                    st.success(f"Successfully handled outliers in {len(selected_outlier_cols)} column(s)")
                                    st.write("Preview of cleaned data:")
                                    st.dataframe(fixed_df.head())
                                except Exception as e:
                                    st.error(f"Error handling outliers: {str(e)}")
                
                # Data type conversion
                with st.expander("Fix Data Types", expanded=False):
                    st.markdown("### Convert Data Types")
                    st.write("Select columns and target data types")
                    
                    # Create multiselect for columns
                    selected_type_cols = st.multiselect(
                        "Select columns to convert:",
                        current_data.columns.tolist(),
                        default=[]
                    )
                    
                    # Define target types for each selected column
                    type_conversions = {}
                    for col in selected_type_cols:
                        target_type = st.selectbox(
                            f"Target type for {col}:",
                            ["int", "float", "str", "datetime", "category"],
                            key=f"type_conversion_{col}"
                        )
                        type_conversions[col] = target_type
                    
                    if st.button("Apply Type Conversions"):
                        with st.spinner("Converting data types..."):
                            try:
                                converted_df = fix_data_types(
                                    st.session_state.cleaned_df 
                                    if 'cleaned_df' in st.session_state else current_data, 
                                    type_conversions
                                )
                                st.session_state.cleaned_df = converted_df
                                st.success(f"Successfully converted data types for {len(selected_type_cols)} column(s)")
                                st.write("Preview of converted data:")
                                st.dataframe(converted_df.head())
                                st.write("New data types:")
                                st.write(converted_df.dtypes)
                            except Exception as e:
                                st.error(f"Error converting data types: {str(e)}")
                
                # Column transformations
                with st.expander("Apply Transformations", expanded=False):
                    st.markdown("### Apply Column Transformations")
                    st.write("Select columns and transformations to apply")
                    
                    # Create multiselect for columns
                    selected_transform_cols = st.multiselect(
                        "Select columns to transform:",
                        current_data.columns.tolist(),
                        default=[]
                    )
                    
                    # Define transformations for each selected column
                    transformations = {}
                    for col in selected_transform_cols:
                        is_numeric = pd.api.types.is_numeric_dtype(current_data[col])
                        is_string = pd.api.types.is_string_dtype(current_data[col])
                        
                        transform_options = []
                        if is_numeric:
                            transform_options.extend(["log", "sqrt", "normalize", "standardize"])
                        if is_string:
                            transform_options.extend(["uppercase", "lowercase", "title", "trim"])
                        
                        if transform_options:
                            transform = st.selectbox(
                                f"Transformation for {col}:",
                                transform_options,
                                key=f"transform_{col}"
                            )
                            transformations[col] = transform
                        else:
                            st.info(f"No applicable transformations for column {col}")
                    
                    if st.button("Apply Transformations"):
                        with st.spinner("Applying transformations..."):
                            try:
                                transformed_df = apply_column_transformations(
                                    st.session_state.cleaned_df 
                                    if 'cleaned_df' in st.session_state else current_data, 
                                    transformations
                                )
                                st.session_state.cleaned_df = transformed_df
                                st.success(f"Successfully applied transformations to {len(selected_transform_cols)} column(s)")
                                st.write("Preview of transformed data:")
                                st.dataframe(transformed_df.head())
                            except Exception as e:
                                st.error(f"Error applying transformations: {str(e)}")
                
                # Remove duplicates
                with st.expander("Remove Duplicates", expanded=False):
                    st.markdown("### Remove Duplicate Rows")
                    
                    # Check if there are any duplicates
                    duplicate_count = current_data.duplicated().sum()
                    
                    if duplicate_count == 0:
                        st.success("No duplicate rows detected in the dataset!")
                    else:
                        st.write(f"Detected {duplicate_count} duplicate rows ({(duplicate_count/len(current_data))*100:.2f}%)")
                        
                        # Option to select columns for duplicate detection
                        use_subset = st.checkbox("Consider only specific columns for duplicate detection")
                        
                        subset_cols = None
                        if use_subset:
                            subset_cols = st.multiselect(
                                "Select columns to consider for duplicates:",
                                current_data.columns.tolist()
                            )
                        
                        if st.button("Remove Duplicates"):
                            with st.spinner("Removing duplicates..."):
                                try:
                                    deduped_df = remove_duplicates(
                                        st.session_state.cleaned_df 
                                        if 'cleaned_df' in st.session_state else current_data, 
                                        subset_cols
                                    )
                                    rows_removed = len(current_data) - len(deduped_df)
                                    st.session_state.cleaned_df = deduped_df
                                    st.success(f"Successfully removed {rows_removed} duplicate rows")
                                    st.write("Preview of de-duplicated data:")
                                    st.dataframe(deduped_df.head())
                                except Exception as e:
                                    st.error(f"Error removing duplicates: {str(e)}")
                
                # Standardize column names
                with st.expander("Standardize Column Names", expanded=False):
                    st.markdown("### Standardize Column Names")
                    st.write("Convert column names to lowercase, replace spaces with underscores, and remove special characters")
                    
                    if st.button("Standardize Names"):
                        with st.spinner("Standardizing column names..."):
                            try:
                                standardized_df = standardize_column_names(
                                    st.session_state.cleaned_df 
                                    if 'cleaned_df' in st.session_state else current_data
                                )
                                
                                # Show before/after comparison
                                comparison = pd.DataFrame({
                                    "Original Names": current_data.columns.tolist(),
                                    "Standardized Names": standardized_df.columns.tolist()
                                })
                                st.write("Column name changes:")
                                st.dataframe(comparison)
                                
                                st.session_state.cleaned_df = standardized_df
                                st.success("Successfully standardized column names")
                            except Exception as e:
                                st.error(f"Error standardizing column names: {str(e)}")
                
                # Save cleaned data
                st.markdown("### Save Cleaned Data")
                save_option = st.radio(
                    "What would you like to do with the cleaned data?",
                    ["Replace current dataset", "Create new dataset"]
                )
                
                if st.button("Save Changes"):
                    if 'cleaned_df' in st.session_state and not st.session_state.cleaned_df.empty:
                        try:
                            if save_option == "Replace current dataset":
                                st.session_state.data[st.session_state.current_df] = st.session_state.cleaned_df
                                st.success(f"Successfully updated dataset: {st.session_state.current_df}")
                            else:
                                new_name = f"{st.session_state.current_df}_cleaned"
                                st.session_state.data[new_name] = st.session_state.cleaned_df
                                st.session_state.selected_files.append(new_name)
                                st.session_state.current_df = new_name
                                st.success(f"Created new dataset: {new_name}")
                            # Clear the cleaned_df from session state
                            st.session_state.cleaned_df = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving cleaned data: {str(e)}")
                    else:
                        st.warning("No cleaned data to save. Please apply some data cleaning operations first.")
            
            # Tab 5: Document Analysis
            with tab5:
                st.subheader("Document Analysis & AI-Powered Insights")
                
                # Initialize session state variables for document storage
                if 'document_files' not in st.session_state:
                    st.session_state.document_files = []
                if 'document_queries' not in st.session_state:
                    st.session_state.document_queries = []
                    
                # Document uploader
                st.markdown("Upload supporting documents (text, CSV, Excel files) for AI-powered analysis:")
                uploaded_documents = st.file_uploader(
                    "Upload supporting documents",
                    type=["txt", "csv", "xlsx", "xls"], 
                    accept_multiple_files=True,
                    key="doc_analysis_tab5_uploader"
                )
                
                if uploaded_documents:
                    with st.spinner("Processing documents..."):
                        for doc in uploaded_documents:
                            if doc.name not in [d["name"] for d in st.session_state.document_files]:
                                try:
                                    # Process document using our document processor
                                    doc_info = document_processor.process_document(doc, doc.name)
                                    
                                    # Store processed document info
                                    st.session_state.document_files.append({
                                        "name": doc.name,
                                        "document_id": doc_info["document_id"],
                                        "word_count": doc_info["metadata"]["word_count"],
                                        "file_type": doc_info["metadata"]["file_type"]
                                    })
                                    
                                    st.success(f"Successfully processed: {doc.name}")
                                except Exception as e:
                                    st.error(f"Error processing document {doc.name}: {str(e)}")
                
                # Document list and preview
                if st.session_state.document_files:
                    st.subheader("Processed Documents")
                    docs_df = pd.DataFrame(st.session_state.document_files)
                    st.dataframe(docs_df)
                    
                    # Document content preview
                    if len(st.session_state.document_files) > 0:
                        selected_doc = st.selectbox(
                            "Select a document to preview:",
                            [doc["name"] for doc in st.session_state.document_files]
                        )
                        
                        selected_doc_id = next((doc["document_id"] for doc in st.session_state.document_files if doc["name"] == selected_doc), None)
                        
                        if selected_doc_id:
                            try:
                                doc_content = document_processor.get_document_by_id(selected_doc_id)
                                with st.expander("Document Content Preview", expanded=False):
                                    st.text_area("Content", value=doc_content["text_content"][:1000] + "...", height=200, disabled=True)
                            except Exception as e:
                                st.error(f"Error loading document preview: {str(e)}")
                    
                    # AI-powered document queries section
                    st.subheader("Ask Questions About Documents")
                    
                    document_query = st.text_input(
                        "Enter your question about the documents:",
                        placeholder="E.g., 'What are the key findings in the report?'",
                        key="document_query_input"
                    )
                    
                    # Check if we should include the dataframe context
                    include_data_context = st.checkbox("Include current dataset context", value=True)
                    
                    if st.button("Submit Document Query", key="submit_doc_query"):
                        if document_query:
                            with st.spinner("Analyzing documents and generating response..."):
                                try:
                                    # Show a message about connecting to AI service
                                    rag_status = st.empty()
                                    rag_status.info("Connecting to AI service... This may take a few moments.")
                                    
                                    # Use RAG to get answer
                                    if include_data_context and 'current_df' in st.session_state and st.session_state.current_df:
                                        current_data = st.session_state.data[st.session_state.current_df]
                                        rag_response = rag_engine.query_with_data(document_query, current_data)
                                    else:
                                        rag_response = rag_engine.query(document_query)
                                    
                                    # Clear the connecting message
                                    rag_status.empty()
                                    
                                    # Check if we have a valid response
                                    if not rag_response or "response" not in rag_response:
                                        st.error("Received an invalid response from the AI service. Please try again.")
                                        st.info("Try asking a simpler question or check that your documents are properly uploaded.")
                                        return
                                    
                                    # Store query and response
                                    st.session_state.document_queries.append({
                                        "query": document_query,
                                        "response": rag_response["response"],
                                        "sources": rag_response.get("sources", [])
                                    })
                                    
                                    # Display the response
                                    st.subheader("AI Response")
                                    st.markdown(rag_response["response"])
                                    
                                    # Display sources if available
                                    if rag_response.get("sources"):
                                        with st.expander("Sources Used", expanded=True):
                                            for source in rag_response["sources"]:
                                                st.markdown(f"- **{source['filename']}** (Relevance: {source['score']:.2f})")
                                except Exception as e:
                                    st.error(f"Error processing document query: {str(e)}")
                                    if "API" in str(e) or "OpenRouter" in str(e) or "openai" in str(e).lower():
                                        st.warning("There was an issue with the AI service connection. The OpenRouter API may be temporarily unavailable, or there may be an issue with your API key.")
                                    st.info("Please check that your documents are properly uploaded and try again later.")
                        else:
                            st.warning("Please enter a query first.")
                    
                    # Document query history
                    if st.session_state.document_queries:
                        with st.expander("Document Query History", expanded=False):
                            for i, item in enumerate(reversed(st.session_state.document_queries[-5:])):
                                st.markdown(f"**Query {len(st.session_state.document_queries)-i}:** {item['query']}")
                                st.markdown(item['response'])
                                st.divider()
                
                # Advanced Analysis with SmartDataFrame
                if 'current_df' in st.session_state and st.session_state.current_df:
                    st.subheader("Advanced Data Analysis")
                    st.markdown("Use AI to perform complex data analysis on your current dataset:")
                    
                    advanced_query = st.text_area(
                        "Describe the analysis you want to perform:",
                        placeholder="E.g., 'Calculate the correlation between age and income, then create a scatter plot'",
                        height=100,
                        key="advanced_query_input"
                    )
                    
                    if st.button("Analyze", key="run_advanced_analysis"):
                        if advanced_query:
                            with st.spinner("Performing AI-powered analysis..."):
                                try:
                                    # Show a message about connecting to AI service
                                    smart_status = st.empty()
                                    smart_status.info("Connecting to AI service... This may take a few moments.")
                                    
                                    # Create SmartDataFrame and run the query
                                    current_data = st.session_state.data[st.session_state.current_df]
                                    smart_df = SmartDataFrame(current_data)
                                    result = smart_df.chat(advanced_query)
                                    
                                    # Clear the connecting message
                                    smart_status.empty()
                                    
                                    if result["success"]:
                                        st.success("Analysis completed successfully!")
                                        
                                        # Show the explanation
                                        with st.expander("Explanation", expanded=True):
                                            st.markdown(result["explanation"])
                                        
                                        # Show the generated code
                                        with st.expander("Generated Python Code", expanded=False):
                                            st.code(result["code"], language="python")
                                        
                                        # Show the result
                                        st.subheader("Analysis Result")
                                        if result["result"] is None or (hasattr(result["result"], 'empty') and result["result"].empty):
                                            st.warning("The analysis was completed but returned no data. Try adjusting your query.")
                                        else:
                                            st.dataframe(result["result"])
                                    else:
                                        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                                        if "API" in str(result.get('error', '')) or "response" in str(result.get('error', '')):
                                            st.warning("There may be an issue with the AI service connection. Try a simpler analysis request.")
                                        
                                        # If there's an AI response but the code execution failed, show it
                                        if "ai_response" in result:
                                            with st.expander("AI Response (Debug Information)", expanded=False):
                                                st.markdown(result["ai_response"])
                                except Exception as e:
                                    st.error(f"Error during advanced analysis: {str(e)}")
                                    if "API" in str(e) or "OpenRouter" in str(e) or "openai" in str(e).lower():
                                        st.warning("There was an issue with the AI service connection. The service may be temporarily unavailable or there may be an issue with your API key.")
                                    st.info("Try a simpler analysis request or use the Natural Language Queries tab instead.")
                        else:
                            st.warning("Please enter an analysis query first.")
                else:
                    st.info("Please upload and select a dataset to use advanced analysis features.")
            
            # Tab 5: Document Analysis
            with tab6:
                st.subheader("Export Data and Insights")
                
                export_options = st.radio(
                    "What would you like to export?",
                    ["Raw Data", "Query Results", "Current Visualization"]
                )
                
                col1, col2, col3 = st.columns(3)
                
                if export_options == "Raw Data":
                    data_to_export = current_data
                    filename = f"{st.session_state.current_df.split('.')[0]}_export"
                    
                    with col1:
                        if st.button("Export to Excel"):
                            with st.spinner("Generating Excel file..."):
                                export_path = export_to_excel(data_to_export, filename)
                                time.sleep(1)  # Simulate processing time
                                st.success("Excel file generated successfully!")
                                
                                # In a real app, you would provide a download link here
                                st.download_button(
                                    label="Download Excel File",
                                    data=open(export_path, "rb").read(),
                                    file_name=f"{filename}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                                # Clean up the temporary file
                                os.remove(export_path)
                    
                    with col2:
                        if st.button("Export to CSV"):
                            with st.spinner("Generating CSV file..."):
                                export_path = export_to_csv(data_to_export, filename)
                                time.sleep(1)  # Simulate processing time
                                st.success("CSV file generated successfully!")
                                
                                # In a real app, you would provide a download link here
                                st.download_button(
                                    label="Download CSV File",
                                    data=open(export_path, "rb").read(),
                                    file_name=f"{filename}.csv",
                                    mime="text/csv"
                                )
                                # Clean up the temporary file
                                os.remove(export_path)
                
                elif export_options == "Query Results" and st.session_state.query_history:
                    query_options = [item["query"] for item in st.session_state.query_history]
                    selected_query = st.selectbox("Select a query result to export:", query_options)
                    
                    selected_result = next((item["result"] for item in st.session_state.query_history if item["query"] == selected_query), None)
                    
                    if selected_result is not None:
                        filename = f"query_result_{int(time.time())}"
                        
                        with col1:
                            if st.button("Export to Excel"):
                                with st.spinner("Generating Excel file..."):
                                    export_path = export_to_excel(selected_result, filename)
                                    time.sleep(1)  # Simulate processing time
                                    st.success("Excel file generated successfully!")
                                    
                                    st.download_button(
                                        label="Download Excel File",
                                        data=open(export_path, "rb").read(),
                                        file_name=f"{filename}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                    # Clean up the temporary file
                                    os.remove(export_path)
                        
                        with col2:
                            if st.button("Export to CSV"):
                                with st.spinner("Generating CSV file..."):
                                    export_path = export_to_csv(selected_result, filename)
                                    time.sleep(1)  # Simulate processing time
                                    st.success("CSV file generated successfully!")
                                    
                                    st.download_button(
                                        label="Download CSV File",
                                        data=open(export_path, "rb").read(),
                                        file_name=f"{filename}.csv",
                                        mime="text/csv"
                                    )
                                    # Clean up the temporary file
                                    os.remove(export_path)
                
                elif export_options == "Current Visualization" and st.session_state.visualization_history:
                    viz_options = [item["query"] for item in st.session_state.visualization_history]
                    selected_viz = st.selectbox("Select a visualization to export:", viz_options)
                    
                    selected_viz_data = next((item for item in st.session_state.visualization_history if item["query"] == selected_viz), None)
                    
                    if selected_viz_data is not None:
                        filename = f"visualization_{int(time.time())}"
                        
                        with col1:
                            if st.button("Export to PDF"):
                                with st.spinner("Generating PDF file..."):
                                    fig = create_visualization(selected_viz_data["result"], selected_viz_data["viz_type"], selected_viz_data["query"])
                                    export_path = export_to_pdf(fig, selected_viz_data["query"], filename)
                                    time.sleep(1)  # Simulate processing time
                                    st.success("PDF report generated successfully!")
                                    
                                    st.download_button(
                                        label="Download PDF Report",
                                        data=open(export_path, "rb").read(),
                                        file_name=f"{filename}.pdf",
                                        mime="application/pdf"
                                    )
                                    # Clean up the temporary file
                                    os.remove(export_path)
                else:
                    st.info("No data available for export. Run some queries or visualizations first.")

if __name__ == "__main__":
    main()
