import pandas as pd
import os
import tempfile
import plotly
import plotly.io as pio
from datetime import datetime

def export_to_excel(data, filename):
    """
    Export data to Excel format.
    
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
