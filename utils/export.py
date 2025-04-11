import pandas as pd
import os
import tempfile
import plotly
import plotly.io as pio
from datetime import datetime
import time
import webbrowser

def export_to_excel(data, filename, include_metadata=True):
    """
    Export data to Excel format.
    
    Args:
        data: DataFrame to export
        filename: Base filename without extension
        include_metadata: Whether to include metadata (timestamp, column info)
        
    Returns:
        str: Path to the exported file
    """
    # Create temporary file
    temp_dir = "tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(temp_dir, f"{filename}_{timestamp}.xlsx")
    
    # Convert to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Write to Excel
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        data.to_excel(writer, sheet_name='Data', index=False)
        
        # Access the workbook and the worksheet
        workbook = writer.book
        worksheet = writer.sheets['Data']
        
        # Add some formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#D9E1F2',
            'border': 1
        })
        
        # Write the column headers with the defined format
        for col_num, value in enumerate(data.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Adjust column widths
        for i, col in enumerate(data.columns):
            max_len = max(data[col].astype(str).apply(len).max(), len(str(col))) + 2
            worksheet.set_column(i, i, min(max_len, 30))
        
        # Add metadata if requested
        if include_metadata:
            # Create a metadata sheet
            metadata_sheet = workbook.add_worksheet('Metadata')
            
            metadata_sheet.write(0, 0, 'Export Information', header_format)
            metadata_sheet.write(1, 0, 'Export Date:')
            metadata_sheet.write(1, 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            metadata_sheet.write(2, 0, 'Total Rows:')
            metadata_sheet.write(2, 1, len(data))
            metadata_sheet.write(3, 0, 'Total Columns:')
            metadata_sheet.write(3, 1, len(data.columns))
            
            # Add column information
            metadata_sheet.write(5, 0, 'Column Information', header_format)
            metadata_sheet.write(6, 0, 'Column Name', header_format)
            metadata_sheet.write(6, 1, 'Data Type', header_format)
            metadata_sheet.write(6, 2, 'Missing Values', header_format)
            metadata_sheet.write(6, 3, 'Unique Values', header_format)
            
            for i, col in enumerate(data.columns):
                metadata_sheet.write(7+i, 0, col)
                metadata_sheet.write(7+i, 1, str(data[col].dtype))
                metadata_sheet.write(7+i, 2, data[col].isna().sum())
                metadata_sheet.write(7+i, 3, data[col].nunique())
            
            # Set column widths for metadata
            metadata_sheet.set_column(0, 0, 25)
            metadata_sheet.set_column(1, 3, 15)
    
    return file_path

def export_to_csv(data, filename):
    """
    Export data to CSV format.
    
    Args:
        data: DataFrame to export
        filename: Base filename without extension
        
    Returns:
        str: Path to the exported file
    """
    # Create temporary file
    temp_dir = "tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(temp_dir, f"{filename}_{timestamp}.csv")
    
    # Convert to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Write to CSV
    data.to_csv(file_path, index=False)
    
    return file_path

def export_to_pdf(fig, title, filename):
    """
    Export a Plotly figure to PDF format.
    
    Args:
        fig: Plotly figure to export
        title: Title for the report
        filename: Base filename without extension
        
    Returns:
        str: Path to the exported file
    """
    # Create temporary file
    temp_dir = "tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(temp_dir, f"{filename}_{timestamp}.pdf")
    
    # Update figure layout for PDF
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=80, b=40, l=40, r=40),
        template='plotly_white',
        width=800,
        height=600
    )
    
    # Write to PDF
    pio.write_image(fig, file_path, format='pdf')
    
    return file_path

def export_to_html_report(data, fig, title, description, filename):
    """
    Export data and visualization to an interactive HTML report.
    
    Args:
        data: DataFrame to include in the report
        fig: Plotly figure to include in the report
        title: Title for the report
        description: Text description for the report
        filename: Base filename without extension
        
    Returns:
        str: Path to the exported file
    """
    # Create temporary file
    temp_dir = "tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(temp_dir, f"{filename}_{timestamp}.html")
    
    # Convert figure to HTML
    plot_div = plotly.offline.plot(fig, include_plotlyjs=True, output_type='div')
    
    # Convert DataFrame to HTML table
    table_html = data.head(100).to_html(classes='dataframe', index=False)
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                padding: 0;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #3498db;
                margin-top: 30px;
            }}
            .visualization {{
                margin: 20px 0;
                border: 1px solid #eee;
                padding: 15px;
                border-radius: 5px;
            }}
            .description {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .dataframe {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            .dataframe th {{
                background-color: #3498db;
                color: white;
                padding: 8px;
                text-align: left;
            }}
            .dataframe td {{
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
            .dataframe tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .metadata {{
                font-size: 0.8em;
                color: #7f8c8d;
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            
            <div class="description">
                <h2>Description</h2>
                <p>{description}</p>
            </div>
            
            <div class="visualization">
                <h2>Visualization</h2>
                {plot_div}
            </div>
            
            <h2>Data Table (showing first 100 rows)</h2>
            {table_html}
            
            <div class="metadata">
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Total rows: {len(data)}, Total columns: {len(data.columns)}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(file_path, 'w') as f:
        f.write(html_content)
    
    # Try to open the HTML file in a browser
    try:
        webbrowser.open('file://' + os.path.abspath(file_path))
    except:
        pass  # If browser opening fails, just return the file path
    
    return file_path
