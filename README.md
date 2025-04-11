# FIRE: Field Insight & Reporting Engine

FIRE (Field Insight & Reporting Engine) is a Streamlit-based data analysis platform that leverages AI to transform complex data exploration into an intuitive, user-friendly experience for non-technical users working with field beneficiary information.

## Features

- **AI-Powered Analysis**: Ask questions about your data in plain language, powered by OpenRouter API and LLM models
- **Data Upload**: Support for CSV and Excel files, with single and multiple file uploads
- **Natural Language Queries**: Ask questions about your data in natural human language
- **Analysis Templates**: Pre-built templates for common analysis needs
- **Advanced Data Visualization**: Interactive charts and graphs including bar, line, scatter, pie and heatmaps
- **Document Analysis**: Support for analyzing related documents using RAG (Retrieval-Augmented Generation)
- **Data Cleaning Tools**: Fix missing values, outliers, and data type issues
- **Export Capabilities**: Export insights as Excel, CSV, or PDF reports

## Getting Started

1. Install the required dependencies:
   ```
   pip install streamlit pandas numpy plotly xlsxwriter openpyxl docx2txt faiss-cpu langchain langchain-openai openai pandasai psycopg2-binary pypdf requests sqlalchemy tiktoken unstructured
   ```

2. Set up environment variables for API keys:
   ```
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   ```

3. Run the application:
   ```
   streamlit run app.py --server.port 8501
   ```

4. Open your browser and navigate to http://localhost:8501

## Using FIRE

1. **Upload Data**: Use the sidebar to upload CSV or Excel files containing field data
2. **Analyze Data**: 
   - Ask natural language questions (e.g., "Show me the average age by gender")
   - Use pre-built analysis templates
3. **Visualize Results**: View and customize visualizations of your data
4. **Export Insights**: Export your findings as Excel, CSV, or PDF reports

## Project Structure

- **app.py**: Main application file
- **utils/**: Utility functions
  - **data_processing.py**: Functions for processing uploaded data
  - **data_cleaning.py**: Advanced data cleaning and preprocessing
  - **database.py**: Database storage and retrieval functions
  - **document_processing.py**: Document extraction and handling
  - **export.py**: Export functionality for various formats
  - **nlp_processing.py**: Natural language query processing with AI
  - **rag_engine.py**: Retrieval-Augmented Generation engine
  - **smart_dataframe.py**: AI-powered dataframe analysis
  - **visualization.py**: Data visualization functions
- **templates/**: Analysis templates
  - **analysis_templates.py**: Pre-built analysis templates

## Support

For questions or support, please contact the development team.
