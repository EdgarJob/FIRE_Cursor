# FIRE: Field Insight & Reporting Engine

## Executive Summary

FIRE (Field Insight & Reporting Engine) is a web-based data analysis and reporting application designed specifically for non-technical field staff and program managers. The application enables users to upload field data (CSV/Excel files), ask natural language questions, and receive both textual insights and interactive visualizations. Powered by AI (via OpenRouter API) and built on Streamlit, FIRE democratizes data analysis and reporting for users who may not have deep technical expertise.

## Objectives

- **Ease of Use**: Allow non-technical users to upload data and get insights using natural language queries without needing to understand underlying data processes.
- **Rapid Insight Generation**: Enable quick responses to data queries by integrating AI capabilities that transform natural language requests into data analysis operations.
- **Interactive Reporting**: Provide clear visualizations that help users quickly understand trends and insights in their data.
- **Modular & Scalable Architecture**: Built using modular components so that it can be easily maintained, scaled, and extended with new features.
- **Enhanced Context through RAG**: Integrate supporting documents to augment the analysis results with additional context.

## Target Users

- **Field Program Staff**: Individuals working in the field who collect data on beneficiaries, services, and outcomes.
- **Program Managers**: Managers who require consolidated reports and visualizations to monitor performance across regions or programs.
- **Data Analysts (Non-Technical)**: Users who can leverage natural language to run data queries and gain insights without writing code.

## Key Features

- **AI-Powered Analysis**: Ask questions about your data in plain language
- **Data Upload**: Support for CSV and Excel files, with single and multiple file uploads
- **Natural Language Queries**: Ask questions about your data in natural human language
- **Analysis Templates**: Pre-built templates for common analysis needs
- **Advanced Data Visualization**: Interactive charts and graphs including bar, line, scatter, pie and heatmaps
- **Document Analysis**: Support for analyzing related documents using RAG (Retrieval-Augmented Generation)
- **Data Cleaning Tools**: Fix missing values, outliers, and data type issues
- **Export Capabilities**: Export insights as Excel, CSV, or HTML reports

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables for API keys:
   ```
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   ```
   Alternatively, create a `.env` file in the project root with your API key.

3. Run the application:
   ```
   streamlit run app.py
   ```

4. Open your browser and navigate to http://localhost:8501

## User Journey

1. **Data Upload**: Upload your CSV or Excel files containing field data
2. **Data Cleaning**: Use the built-in tools to clean and preprocess your data
3. **Analyze Data**: 
   - Ask natural language questions (e.g., "Show me the average age by gender")
   - Use pre-built analysis templates
4. **Visualize Results**: View and customize visualizations of your data
5. **Export Insights**: Export your findings as Excel, CSV, or HTML reports

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
