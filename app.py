import streamlit as st
import pandas as pd
import os
import time
import json
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

from utils.data_processing import process_uploaded_files, combine_dataframes
from utils.nlp_processing import process_query
from utils.visualization import create_visualization
from utils.export import export_to_excel, export_to_csv, export_to_pdf, export_to_html_report
from utils.data_cleaning import (get_data_quality_report, fix_missing_values, 
                                handle_outliers, remove_duplicates, fix_data_types,
                                standardize_column_names, apply_column_transformations,
                                create_derived_features, filter_rows)
from templates.analysis_templates import get_templates, get_flat_templates, apply_template
from utils.database import (
    save_file_to_db, get_file_from_db, get_all_files, load_dataframe_from_file,
    save_query_to_db, get_query_history, save_visualization_to_db, 
    get_visualization_history, save_user_preference, get_user_preference
)
from utils.document_processing import document_processor
from utils.rag_engine import rag_engine
from utils.smart_dataframe import SmartDataFrame
from utils.template_generator import (
    generate_custom_template, save_custom_template, 
    load_custom_templates, get_custom_templates_for_ui,
    execute_custom_template
)

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
    
    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        st.warning("""
        ⚠️ **OpenRouter API key not configured.** AI features will use backup methods with limited capabilities.
        
        To fix this issue:
        1. Sign up for a free account at [OpenRouter](https://openrouter.ai/keys)
        2. Create a new API key
        3. Add your API key to the `.env` file:
        ```
        OPENROUTER_API_KEY=your_api_key_here
        ```
        4. Restart the application
        """)
    
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
                
                # Create template tabs for built-in and custom templates
                template_tabs = st.tabs(["Built-in Templates", "Custom Templates", "Create Template"])
                
                with template_tabs[0]:  # Built-in Templates tab
                    templates = get_templates()
                    
                    # Create a two-level selection system for templates
                    category = st.selectbox(
                        "Select a template category:",
                        list(templates.keys()),
                        key="builtin_template_category"
                    )
                    
                    # Get templates for the selected category
                    category_templates = templates[category]
                    
                    # Select a specific template from the category
                    template_name = st.selectbox(
                        "Choose a template:",
                        list(category_templates.keys()),
                        format_func=lambda x: category_templates[x],
                        key="builtin_template_name"
                    )
                    
                    if st.button("Apply Template", key="apply_builtin_template"):
                        with st.spinner("Applying template..."):
                            try:
                                result, viz_type, description = apply_template(template_name, current_data)
                                
                                st.markdown(f"**{description}**")
                                st.write(result)
                                
                                # Create visualization
                                fig = create_visualization(result, viz_type, template_name)
                                st.plotly_chart(fig, use_container_width=True, key=f"template_viz_{template_name}")
                                
                                # Store in visualization history
                                st.session_state.visualization_history.append({
                                    "query": f"Template: {template_name}",
                                    "viz_type": viz_type,
                                    "result": result
                                })
                            except Exception as e:
                                st.error(f"Error applying template: {str(e)}")
                
                with template_tabs[1]:  # Custom Templates tab
                    # Load custom templates
                    if 'custom_templates' not in st.session_state:
                        st.session_state.custom_templates = load_custom_templates()
                    
                    if not st.session_state.custom_templates:
                        st.info("No custom templates found. Create a template in the 'Create Template' tab.")
                    else:
                        # Add an auto-fix all button
                        if st.button("Auto-Fix All Templates", key="fix_all_templates"):
                            fixed_count = 0
                            with st.spinner("Fixing templates..."):
                                for template_name, template_data in st.session_state.custom_templates.items():
                                    original_code = template_data["code"]
                                    fixed_code = fix_generated_code(original_code)
                                    
                                    if fixed_code != original_code:
                                        # Template needed fixing
                                        template_data["code"] = fixed_code
                                        template_data["auto_fixed"] = True
                                        
                                        # Save the fixed template
                                        save_custom_template(template_data)
                                        fixed_count += 1
                            
                            if fixed_count > 0:
                                st.success(f"Fixed {fixed_count} templates! Try running them now.")
                            else:
                                st.info("No templates needed fixing.")
                        
                        # Show available custom templates
                        custom_template_names = list(st.session_state.custom_templates.keys())
                        selected_custom_template = st.selectbox(
                            "Select a custom template:",
                            custom_template_names,
                            key="custom_template_select"
                        )
                        
                        # Show template description
                        if selected_custom_template in st.session_state.custom_templates:
                            template_info = st.session_state.custom_templates[selected_custom_template]
                            st.caption(f"Description: {template_info.get('description', 'No description available')}")
                            st.caption(f"Created: {template_info.get('timestamp', 'Unknown date')}")
                            
                            # Option to view the code
                            with st.expander("View Template Code"):
                                st.code(template_info.get('code', 'Code not available'), language="python")
                            
                            # Apply the custom template
                            if st.button("Apply Custom Template", key="apply_custom_template"):
                                with st.spinner("Applying custom template..."):
                                    try:
                                        result, viz_type, description = execute_custom_template(selected_custom_template, current_data)
                                        
                                        st.markdown(f"**{description}**")
                                        st.write(result)
                                        
                                        # Create visualization
                                        fig = create_visualization(result, viz_type, selected_custom_template)
                                        st.plotly_chart(fig, use_container_width=True, key=f"custom_viz_{selected_custom_template}")
                                        
                                        # Store in visualization history
                                        st.session_state.visualization_history.append({
                                            "query": f"Custom Template: {selected_custom_template}",
                                            "viz_type": viz_type,
                                            "result": result
                                        })
                                    except Exception as e:
                                        error_msg = str(e)
                                        st.error(f"Error applying custom template: {error_msg}")
                                        
                                        # Show detailed error information and fix options
                                        with st.expander("Troubleshoot Template", expanded=True):
                                            st.markdown("### Template Error Details")
                                            st.markdown("""
                                            The template execution failed. This could be due to:
                                            1. The 'col' variable issue - a common error in AI-generated code
                                            2. Missing or differently named columns in your dataset
                                            3. Incorrect data types or operations
                                            """)
                                            
                                            st.subheader("Edit Template Code")
                                            if "code" in st.session_state.custom_templates[selected_custom_template]:
                                                template_code = st.session_state.custom_templates[selected_custom_template]["code"]
                                                edited_code = st.text_area("Template Code", 
                                                                        value=template_code, 
                                                                        height=400,
                                                                        key=f"edit_code_{selected_custom_template}")
                                                
                                                if st.button("Save and Retry", key="save_edited_code"):
                                                    # Update the template with edited code
                                                    st.session_state.custom_templates[selected_custom_template]["code"] = edited_code
                                                    
                                                    # Save to disk
                                                    save_custom_template(st.session_state.custom_templates[selected_custom_template])
                                                    
                                                    st.success("Template updated. Try applying it again.")
                                                    st.experimental_rerun()
                            
                            # Fix template button for errors
                            if "Error" in st.session_state and selected_custom_template in st.session_state["Error"]:
                                with st.expander("Fix Template", expanded=True):
                                    st.markdown(f"**Error with template: {st.session_state['Error'][selected_custom_template]}**")
                                    
                                    if st.button("Auto-Fix Template", key="auto_fix_template"):
                                        try:
                                            # Apply auto-fix to the template
                                            template = st.session_state.custom_templates[selected_custom_template]
                                            template["code"] = fix_generated_code(template["code"])
                                            
                                            # Save fixed template
                                            save_custom_template(template)
                                            
                                            st.success("Template auto-fixed. Try applying it again.")
                                            if "Error" in st.session_state:
                                                if selected_custom_template in st.session_state["Error"]:
                                                    del st.session_state["Error"][selected_custom_template]
                                            st.experimental_rerun()
                                        except Exception as fix_error:
                                            st.error(f"Error fixing template: {str(fix_error)}")
                            
                            # Delete template option
                            if st.button("Delete Template", key="delete_custom_template"):
                                try:
                                    # Remove from session state
                                    template = st.session_state.custom_templates.pop(selected_custom_template)
                                    
                                    # Remove from disk
                                    template_path = os.path.join("custom_templates", f"{template['name'].replace(' ', '_')}.json")
                                    if os.path.exists(template_path):
                                        os.remove(template_path)
                                    
                                    st.success(f"Template '{selected_custom_template}' deleted successfully.")
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Error deleting template: {str(e)}")
                
                with template_tabs[2]:  # Create Template tab
                    st.markdown("### Create Custom Analysis Template")
                    st.markdown("""
                    Use AI to generate a custom analysis template for your data. Describe what analysis you want to perform,
                    and the AI will create a reusable template that you can apply to your data.
                    """)
                    
                    # Template information
                    template_name = st.text_input("Template Name", placeholder="Give your template a unique name")
                    template_description = st.text_area(
                        "Analysis Description", 
                        placeholder="Describe the analysis you want the template to perform. For example: 'Compare the distribution of program completion rates by gender and age group, showing the top performers in each category.'"
                    )
                    
                    # Generate button
                    if st.button("Generate Template", key="generate_custom_template"):
                        if not template_name or not template_description:
                            st.error("Please provide both a template name and description.")
                        else:
                            with st.spinner("Generating template using AI... This may take a moment."):
                                try:
                                    # Check if current data is available
                                    if current_data is None or current_data.empty:
                                        st.error("Please load data before creating a template.")
                                    else:
                                        # Generate the template
                                        template = generate_custom_template(
                                            description=template_description,
                                            df=current_data,
                                            template_name=template_name
                                        )
                                        
                                        # Save the template
                                        if save_custom_template(template):
                                            st.success(f"Template '{template_name}' created successfully!")
                                            
                                            # Show preview of the generated code
                                            with st.expander("Template Code", expanded=True):
                                                st.code(template["code"], language="python")
                                                
                                            # Option to apply immediately
                                            if st.button("Apply New Template Now", key="apply_new_template"):
                                                with st.spinner("Applying template..."):
                                                    try:
                                                        result, viz_type, description = execute_custom_template(template_name, current_data)
                                                        
                                                        st.markdown(f"**{description}**")
                                                        st.write(result)
                                                        
                                                        # Create visualization
                                                        fig = create_visualization(result, viz_type, template_name)
                                                        st.plotly_chart(fig, use_container_width=True, key=f"new_template_viz")
                                                        
                                                        # Store in visualization history
                                                        st.session_state.visualization_history.append({
                                                            "query": f"Custom Template: {template_name}",
                                                            "viz_type": viz_type,
                                                            "result": result
                                                        })
                                                    except Exception as e:
                                                        st.error(f"Error applying template: {str(e)}")
                                        else:
                                            st.error("Failed to save the template.")
                                except Exception as e:
                                    st.error(f"Error generating template: {str(e)}")
                    
                    # Example section
                    with st.expander("See Example Descriptions"):
                        st.markdown("""
                        Here are some examples of good template descriptions:
                        
                        1. **Program Impact Analysis**: "Create a template that calculates the average impact scores across different program types, and identifies which demographics show the highest improvement rates."
                        
                        2. **Regional Comparison**: "Compare key performance metrics across different regions, highlighting outliers and identifying regions that need additional support."
                        
                        3. **Beneficiary Progression**: "Track how beneficiaries progress through program stages over time, showing completion rates and identifying potential bottlenecks."
                        
                        4. **Funding Efficiency**: "Analyze the relationship between funding amounts and outcome measures, calculating cost per beneficiary and identifying the most cost-effective programs."
                        
                        5. **Vulnerability Index**: "Create a vulnerability score based on multiple indicators (like income, health status, and housing), then show distribution of vulnerable groups across regions."
                        """)
            
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
                st.subheader("Data Cleaning and Preprocessing")
                
                # Initialize session state for cleaned dataframe
                if 'cleaned_df' not in st.session_state:
                    st.session_state.cleaned_df = current_data.copy()
                
                # Add quick actions for common cleaning tasks
                st.markdown("### Quick Actions")
                quick_action_cols = st.columns(4)
                
                with quick_action_cols[0]:
                    if st.button("Fix Missing Values", key="quick_fix_missing"):
                        with st.spinner("Automatically fixing missing values..."):
                            try:
                                # Auto-detect strategies based on column types
                                auto_strategies = {}
                                for col in current_data.columns:
                                    missing_count = current_data[col].isna().sum()
                                    if missing_count > 0:
                                        if pd.api.types.is_numeric_dtype(current_data[col]):
                                            auto_strategies[col] = "median"  # Safe choice for numeric
                                        else:
                                            auto_strategies[col] = "mode"  # Safe choice for categorical
                                
                                if auto_strategies:
                                    fixed_df = fix_missing_values(current_data, auto_strategies)
                                    st.session_state.cleaned_df = fixed_df
                                    st.success(f"Fixed missing values in {len(auto_strategies)} columns using automatic strategies")
                                else:
                                    st.info("No missing values detected in the dataset")
                            except Exception as e:
                                st.error(f"Error fixing missing values: {str(e)}")
                
                with quick_action_cols[1]:
                    if st.button("Fix Outliers", key="quick_fix_outliers"):
                        with st.spinner("Automatically handling outliers..."):
                            try:
                                # Auto-detect numeric columns for outlier handling
                                numeric_cols = [col for col in current_data.columns 
                                              if pd.api.types.is_numeric_dtype(current_data[col])]
                                
                                if numeric_cols:
                                    auto_strategies = {col: "clip" for col in numeric_cols}  # Use clipping as safest choice
                                    fixed_df = handle_outliers(
                                        st.session_state.cleaned_df 
                                        if 'cleaned_df' in st.session_state else current_data,
                                        auto_strategies
                                    )
                                    st.session_state.cleaned_df = fixed_df
                                    st.success(f"Handled outliers in {len(numeric_cols)} numeric columns")
                                else:
                                    st.info("No numeric columns detected for outlier handling")
                            except Exception as e:
                                st.error(f"Error handling outliers: {str(e)}")
                
                with quick_action_cols[2]:
                    if st.button("Remove Duplicates", key="quick_remove_dupes"):
                        with st.spinner("Removing duplicate rows..."):
                            try:
                                df_to_clean = st.session_state.cleaned_df if 'cleaned_df' in st.session_state else current_data
                                duplicate_count = df_to_clean.duplicated().sum()
                                
                                if duplicate_count > 0:
                                    deduped_df = remove_duplicates(df_to_clean, None)
                                    rows_removed = len(df_to_clean) - len(deduped_df)
                                    st.session_state.cleaned_df = deduped_df
                                    st.success(f"Removed {rows_removed} duplicate rows")
                                else:
                                    st.info("No duplicate rows detected in the dataset")
                            except Exception as e:
                                st.error(f"Error removing duplicates: {str(e)}")
                
                with quick_action_cols[3]:
                    if st.button("Standardize Names", key="quick_standardize"):
                        with st.spinner("Standardizing column names..."):
                            try:
                                df_to_clean = st.session_state.cleaned_df if 'cleaned_df' in st.session_state else current_data
                                standardized_df = standardize_column_names(df_to_clean)
                                st.session_state.cleaned_df = standardized_df
                                st.success("Column names standardized")
                                st.write("New column names:")
                                st.write(", ".join(standardized_df.columns.tolist()))
                            except Exception as e:
                                st.error(f"Error standardizing column names: {str(e)}")
                
                # Apply button for all quick actions
                if st.button("Run All Quick Fixes", key="run_all_fixes"):
                    with st.spinner("Applying all quick fixes..."):
                        try:
                            # Start with current data or already cleaned data
                            df_to_clean = st.session_state.cleaned_df if 'cleaned_df' in st.session_state else current_data
                            
                            # 1. Fix missing values
                            auto_strategies = {}
                            for col in df_to_clean.columns:
                                missing_count = df_to_clean[col].isna().sum()
                                if missing_count > 0:
                                    if pd.api.types.is_numeric_dtype(df_to_clean[col]):
                                        auto_strategies[col] = "median"
                                    else:
                                        auto_strategies[col] = "mode"
                            
                            if auto_strategies:
                                df_to_clean = fix_missing_values(df_to_clean, auto_strategies)
                            
                            # 2. Handle outliers
                            numeric_cols = [col for col in df_to_clean.columns 
                                          if pd.api.types.is_numeric_dtype(df_to_clean[col])]
                            
                            if numeric_cols:
                                auto_strategies = {col: "clip" for col in numeric_cols}
                                df_to_clean = handle_outliers(df_to_clean, auto_strategies)
                            
                            # 3. Remove duplicates
                            df_to_clean = remove_duplicates(df_to_clean, None)
                            
                            # 4. Standardize column names
                            df_to_clean = standardize_column_names(df_to_clean)
                            
                            # Update the cleaned dataframe
                            st.session_state.cleaned_df = df_to_clean
                            st.success("All quick fixes applied successfully!")
                            
                            # Show data quality comparison
                            old_quality = get_data_quality_report(current_data)
                            new_quality = get_data_quality_report(df_to_clean)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Before Cleaning:**")
                                st.write(f"- Rows: {old_quality['row_count']}")
                                st.write(f"- Missing values: {sum(v['count'] for v in old_quality['missing_values'].values()) if 'missing_values' in old_quality else 0}")
                                st.write(f"- Duplicates: {old_quality['duplicates']['count']}")
                                st.write(f"- Outliers: {sum(v['count'] for v in old_quality['outliers'].values()) if 'outliers' in old_quality else 0}")
                            
                            with col2:
                                st.markdown("**After Cleaning:**")
                                st.write(f"- Rows: {new_quality['row_count']}")
                                st.write(f"- Missing values: {sum(v['count'] for v in new_quality['missing_values'].values()) if 'missing_values' in new_quality else 0}")
                                st.write(f"- Duplicates: {new_quality['duplicates']['count']}")
                                st.write(f"- Outliers: {sum(v['count'] for v in new_quality['outliers'].values()) if 'outliers' in new_quality else 0}")
                        
                        except Exception as e:
                            st.error(f"Error applying all fixes: {str(e)}")
                
                # Show data quality report with toggle
                data_quality_toggle = st.checkbox("Show Data Quality Report", value=False)
                
                if data_quality_toggle:
                    st.markdown("### Data Quality Report")
                    with st.spinner("Generating data quality report..."):
                        df_to_analyze = st.session_state.cleaned_df if 'cleaned_df' in st.session_state else current_data
                        quality_report = get_data_quality_report(df_to_analyze)
                        
                        if quality_report['status'] == 'success':
                            # Summary statistics
                            st.markdown("#### Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Rows", quality_report['row_count'])
                            with col2:
                                st.metric("Total Columns", quality_report['column_count'])
                            with col3:
                                duplicate_pct = quality_report['duplicates']['percentage']
                                st.metric("Duplicate Rows", f"{quality_report['duplicates']['count']} ({duplicate_pct:.1f}%)")
                            
                            # Missing values summary
                            if quality_report['missing_values']:
                                st.markdown("#### Missing Values")
                                missing_df = pd.DataFrame([
                                    {"Column": col, "Missing Count": data['count'], "Missing %": f"{data['percentage']:.1f}%"}
                                    for col, data in quality_report['missing_values'].items()
                                ])
                                st.dataframe(missing_df)
                            else:
                                st.success("No missing values found in the dataset!")
                            
                            # Outliers summary
                            if quality_report['outliers']:
                                st.markdown("#### Outliers")
                                outlier_df = pd.DataFrame([
                                    {"Column": col, "Outlier Count": data['count'], "Outlier %": f"{data['percentage']:.1f}%"}
                                    for col, data in quality_report['outliers'].items()
                                ])
                                st.dataframe(outlier_df)
                            else:
                                st.success("No outliers detected in numeric columns!")
                        else:
                            st.error(quality_report['error'])
                
                # Show advanced options with toggle
                st.markdown("### Advanced Options")
                st.info("Use these options to have more control over the cleaning process")
                
                # Use existing expanders for advanced options
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
                
                # Add a save button to replace or save as new dataset
                if 'cleaned_df' in st.session_state and not st.session_state.cleaned_df.empty:
                    st.markdown("### Save Cleaned Data")
                    save_col1, save_col2 = st.columns([3, 1])
                    
                    with save_col1:
                        save_option = st.radio(
                            "What would you like to do with the cleaned data?",
                            ["Replace current dataset", "Create new dataset"],
                            horizontal=True
                        )
                        
                        if save_option == "Create new dataset":
                            new_name = st.text_input(
                                "Enter name for the new dataset:",
                                value=f"{st.session_state.current_df}_cleaned"
                            )
                    
                    with save_col2:
                        if st.button("Save Changes", key="save_cleaned_data"):
                            try:
                                if save_option == "Replace current dataset":
                                    st.session_state.data[st.session_state.current_df] = st.session_state.cleaned_df
                                    st.success(f"Successfully updated dataset: {st.session_state.current_df}")
                                else:
                                    # Create new dataset
                                    st.session_state.data[new_name] = st.session_state.cleaned_df
                                    if new_name not in st.session_state.selected_files:
                                        st.session_state.selected_files.append(new_name)
                                    st.session_state.current_df = new_name
                                    st.success(f"Created new dataset: {new_name}")
                                
                                # Clear the cleaned_df to start fresh
                                st.session_state.cleaned_df = None
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error saving cleaned data: {str(e)}")
            
            # Tab 5: Document Analysis
            with tab5:
                st.subheader("Document Analysis & AI-Powered Insights")
                
                # Check for OpenRouter API key
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key or api_key == "your_openrouter_api_key_here":
                    st.error("API key not configured. Document analysis requires a valid OpenRouter API key.")
                    st.info("""
                    To enable document analysis:
                    1. Sign up for a free account at [OpenRouter](https://openrouter.ai/keys)
                    2. Create a new API key
                    3. Add your API key to the `.env` file
                    4. Restart the application
                    """)
                    st.markdown("---")
                
                # Initialize document uploads in session state if needed
                if 'uploaded_documents' not in st.session_state:
                    st.session_state.uploaded_documents = []
                if 'document_queries' not in st.session_state:
                    st.session_state.document_queries = []
                
                # Upload section
                st.markdown("### Upload Supporting Documents")
                st.markdown("Upload supporting documents (text, CSV, Excel files) for AI-powered analysis:")
                
                uploaded_docs = st.file_uploader(
                    "Upload supporting documents",
                    type=["txt", "pdf", "docx", "csv", "xlsx"],
                    accept_multiple_files=True,
                    key="document_uploader"
                )
                
                if uploaded_docs:
                    # Process the new document uploads
                    with st.spinner("Processing document uploads..."):
                        newly_uploaded = [doc for doc in uploaded_docs 
                                        if doc.name not in [d["filename"] for d in st.session_state.uploaded_documents]]
                        
                        if newly_uploaded:
                            for doc in newly_uploaded:
                                try:
                                    # Process the document
                                    doc_info = document_processor.process_document(doc)
                                    
                                    # Add to session state
                                    st.session_state.uploaded_documents.append({
                                        "filename": doc.name,
                                        "id": doc_info["id"],
                                        "file_type": doc_info["file_type"],
                                        "upload_date": doc_info["upload_date"],
                                        "size": doc_info["size"]
                                    })
                                    
                                    st.success(f"Document '{doc.name}' processed successfully!")
                                except Exception as e:
                                    st.error(f"Error processing document '{doc.name}': {str(e)}")
                
                # Show currently uploaded documents
                if st.session_state.uploaded_documents:
                    st.markdown("### Available Documents")
                    
                    col1, col2, col3 = st.columns([3, 2, 2])
                    with col1:
                        st.markdown("**Filename**")
                    with col2:
                        st.markdown("**Type**")
                    with col3:
                        st.markdown("**Upload Date**")
                    
                    for doc in st.session_state.uploaded_documents:
                        col1, col2, col3 = st.columns([3, 2, 2])
                        with col1:
                            st.markdown(doc["filename"])
                        with col2:
                            st.markdown(doc["file_type"])
                        with col3:
                            if isinstance(doc["upload_date"], str):
                                st.markdown(doc["upload_date"])
                            else:
                                st.markdown(doc["upload_date"].strftime("%Y-%m-%d %H:%M"))
                    
                    # Document viewer section
                    st.subheader("Document Viewer")
                    
                    doc_ids = {doc["filename"]: doc["id"] for doc in st.session_state.uploaded_documents}
                    selected_doc = st.selectbox("Select a document to view:", list(doc_ids.keys()))
                    
                    if selected_doc:
                        selected_doc_id = doc_ids[selected_doc]
                        try:
                            doc_content = document_processor.get_document_by_id(selected_doc_id)
                            with st.expander("Document Content Preview", expanded=False):
                                st.text_area("Content", value=doc_content["text_content"][:1000] + "...", height=200, disabled=True)
                        except Exception as e:
                            st.error(f"Error loading document preview: {str(e)}")
                
                # AI-powered document queries section
                st.subheader("Ask Questions About Documents")
                
                # Check again for OpenRouter API key before allowing document queries
                if not api_key or api_key == "your_openrouter_api_key_here":
                    st.warning("AI-powered document queries require a valid OpenRouter API key.")
                else:
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
                                    if "API" in str(e) or "OpenRouter" in str(e) or "openai" in str(e).lower() or "401" in str(e):
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
                    
                    # Check for OpenRouter API key again 
                    if not api_key or api_key == "your_openrouter_api_key_here":
                        st.warning("AI-powered advanced analysis requires a valid OpenRouter API key.")
                    else:
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
                                            if "API" in str(result.get('error', '')) or "401" in str(result.get('error', '')):
                                                st.warning("There may be an issue with the AI service connection. Make sure your API key is valid.")
                                            
                                            # If there's an AI response but the code execution failed, show it
                                            if "ai_response" in result:
                                                with st.expander("AI Response (Debug Information)", expanded=False):
                                                    st.markdown(result["ai_response"])
                                    except Exception as e:
                                        st.error(f"Error during advanced analysis: {str(e)}")
                                        if "API" in str(e) or "OpenRouter" in str(e) or "openai" in str(e).lower() or "401" in str(e):
                                            st.warning("There was an issue with the AI service connection. Make sure your API key is valid.")
                                        st.info("Try a simpler analysis request or use the Natural Language Queries tab instead.")
                            else:
                                st.warning("Please enter an analysis query first.")
                else:
                    st.info("Please upload and select a dataset to use advanced analysis features.")
            
            # Tab 6: Export Data and Insights
            with tab6:
                st.subheader("Export Data and Insights")
                
                export_options = st.radio(
                    "What would you like to export?",
                    ["Raw Data", "Query Results", "Current Visualization", "Full Report"]
                )
                
                col1, col2, col3 = st.columns(3)
                
                if export_options == "Raw Data":
                    data_to_export = current_data
                    filename = f"{st.session_state.current_df.split('.')[0]}_export"
                    
                    with col1:
                        if st.button("Export to Excel"):
                            with st.spinner("Generating Excel file..."):
                                # Use the enhanced Excel export with metadata
                                include_metadata = st.checkbox("Include metadata sheet", value=True)
                                export_path = export_to_excel(data_to_export, filename, include_metadata)
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
                        
                        with col2:
                            if st.button("Export to HTML Report"):
                                with st.spinner("Generating interactive HTML report..."):
                                    fig = create_visualization(selected_viz_data["result"], selected_viz_data["viz_type"], selected_viz_data["query"])
                                    description = f"This report shows the results of the query: '{selected_viz_data['query']}'"
                                    export_path = export_to_html_report(
                                        selected_viz_data["result"], 
                                        fig, 
                                        f"FIRE Report: {selected_viz_data['query']}", 
                                        description, 
                                        filename
                                    )
                                    time.sleep(1)  # Simulate processing time
                                    st.success("HTML report generated and opened in browser!")
                                    
                                    st.download_button(
                                        label="Download HTML Report",
                                        data=open(export_path, "rb").read(),
                                        file_name=f"{filename}.html",
                                        mime="text/html"
                                    )
                                    
                                    # Show file path
                                    st.info(f"Report saved to: {export_path}")
                
                elif export_options == "Full Report" and st.session_state.visualization_history:
                    st.markdown("### Create a comprehensive report with data, visualizations, and insights")
                    
                    # Select title and description
                    report_title = st.text_input("Report Title:", value=f"FIRE Analysis Report - {st.session_state.current_df}")
                    report_description = st.text_area("Report Description:", value="This report was generated using FIRE (Field Insight & Reporting Engine).")
                    
                    # Select visualizations to include
                    available_viz = [item["query"] for item in st.session_state.visualization_history]
                    selected_vizs = st.multiselect("Select visualizations to include:", available_viz)
                    
                    if st.button("Generate Full Report") and selected_vizs:
                        with st.spinner("Generating comprehensive report..."):
                            # Get the first selected visualization to use as the main one
                            main_viz_data = next((item for item in st.session_state.visualization_history if item["query"] == selected_vizs[0]), None)
                            
                            if main_viz_data:
                                # Create the main figure
                                fig = create_visualization(main_viz_data["result"], main_viz_data["viz_type"], main_viz_data["query"])
                                
                                # Create a comprehensive description including all selected viz
                                comprehensive_desc = f"{report_description}\n\nThis report includes analysis of:\n"
                                for viz_name in selected_vizs:
                                    comprehensive_desc += f"- {viz_name}\n"
                                
                                # Generate filename
                                filename = f"full_report_{int(time.time())}"
                                
                                # Create the HTML report
                                export_path = export_to_html_report(
                                    current_data.head(50),  # We just show a sample of the main data 
                                    fig, 
                                    report_title, 
                                    comprehensive_desc, 
                                    filename
                                )
                                
                                st.success("Full HTML report generated and opened in browser!")
                                
                                st.download_button(
                                    label="Download Full Report",
                                    data=open(export_path, "rb").read(),
                                    file_name=f"{filename}.html",
                                    mime="text/html"
                                )
                                
                                # Show file path
                                st.info(f"Report saved to: {export_path}")
                else:
                    st.info("No data available for export. Run some queries or visualizations first.")

if __name__ == "__main__":
    main()
