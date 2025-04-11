import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import re

def create_visualization(data, viz_type, title=None):
    """
    Create a plotly visualization based on the data and visualization type.
    
    Args:
        data: DataFrame containing the data to visualize
        viz_type: Type of visualization to create (bar, line, pie, scatter, etc.)
        title: Title for the visualization
        
    Returns:
        plotly.graph_objects.Figure: The created visualization
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No data to visualize",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="No data available for visualization",
                showarrow=False,
                font=dict(size=20)
            )]
        )
        return fig
    
    # Clean the title if provided
    chart_title = title if title else "Data Visualization"
    chart_title = re.sub(r'[^\w\s\-\,\.]', '', chart_title)
    
    # Determine x and y columns based on data structure
    if 'Metric' in data.columns and 'Value' in data.columns:
        # This is a metric-value pair format
        x_col = 'Metric'
        y_col = 'Value'
    elif len(data.columns) >= 2:
        # Assume last column is the value to plot
        y_col = data.columns[-1]
        
        # First column is usually the category/grouping
        x_col = data.columns[0]
        
        # If x_col has too many unique values, try to find a better column
        if len(data[x_col].unique()) > 20 and len(data.columns) > 2:
            for col in data.columns[:-1]:
                if len(data[col].unique()) <= 20:
                    x_col = col
                    break
    else:
        # Handle single column case
        x_col = data.index.name if data.index.name else 'Index'
        y_col = data.columns[0]
    
    # Create visualization based on type
    if viz_type == 'bar':
        # Handle cases with too many categories
        if x_col in data.columns and len(data[x_col].unique()) > 15:
            # Sort and take top 15
            sorted_data = data.sort_values(by=y_col, ascending=False).head(15)
        else:
            sorted_data = data
        
        fig = px.bar(
            sorted_data, 
            x=x_col, 
            y=y_col,
            title=chart_title,
            labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
            color_discrete_sequence=['#3B82F6']
        )
    
    elif viz_type == 'line':
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col,
            title=chart_title,
            labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
            markers=True
        )
    
    elif viz_type == 'pie':
        fig = px.pie(
            data, 
            names=x_col, 
            values=y_col,
            title=chart_title,
            labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()}
        )
    
    elif viz_type == 'scatter':
        # For scatter plots, try to find two numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            
            # If we have a third categorical column, use it for color
            color_col = None
            if len(data.columns) > 2:
                for col in data.columns:
                    if col not in [x_col, y_col] and len(data[col].unique()) <= 10:
                        color_col = col
                        break
            
            fig = px.scatter(
                data, 
                x=x_col, 
                y=y_col,
                color=color_col,
                title=chart_title,
                labels={
                    x_col: x_col.replace('_', ' ').title(), 
                    y_col: y_col.replace('_', ' ').title(),
                    color_col: color_col.replace('_', ' ').title() if color_col else None
                }
            )
        else:
            # Fall back to bar chart if we don't have enough numeric columns
            fig = px.bar(
                data, 
                x=x_col, 
                y=y_col,
                title=chart_title,
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()}
            )
    
    elif viz_type == 'histogram':
        # Find the first numeric column
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        if len(numeric_cols) > 0:
            x_col = numeric_cols[0]
            fig = px.histogram(
                data, 
                x=x_col,
                title=chart_title,
                labels={x_col: x_col.replace('_', ' ').title()}
            )
        else:
            # Fall back to bar chart for categorical data
            fig = px.bar(
                data, 
                x=x_col, 
                y=y_col,
                title=chart_title,
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()}
            )
    
    elif viz_type == 'box':
        # Find numeric columns for box plot
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        if len(numeric_cols) > 0:
            # Find a categorical column for grouping
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            
            if len(cat_cols) > 0:
                fig = px.box(
                    data, 
                    x=cat_cols[0], 
                    y=numeric_cols[0],
                    title=chart_title,
                    labels={
                        cat_cols[0]: cat_cols[0].replace('_', ' ').title(), 
                        numeric_cols[0]: numeric_cols[0].replace('_', ' ').title()
                    }
                )
            else:
                fig = px.box(
                    data, 
                    y=numeric_cols[0],
                    title=chart_title,
                    labels={numeric_cols[0]: numeric_cols[0].replace('_', ' ').title()}
                )
        else:
            # Fall back to bar chart if no numeric columns
            fig = px.bar(
                data, 
                x=x_col, 
                y=y_col,
                title=chart_title,
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()}
            )
    
    elif viz_type == 'heatmap':
        # Check if we have a pivot table or need to create one
        if len(data.columns) >= 3:
            # Create a pivot table using the first two columns as index and columns
            pivot_cols = data.columns[:-1]  # All but the last column
            value_col = data.columns[-1]   # Last column contains the values
            
            if len(pivot_cols) >= 2:
                try:
                    pivot_data = data.pivot_table(
                        index=pivot_cols[0], 
                        columns=pivot_cols[1], 
                        values=value_col,
                        aggfunc='mean'
                    )
                    
                    fig = px.imshow(
                        pivot_data,
                        title=chart_title,
                        labels=dict(
                            x=pivot_cols[1].replace('_', ' ').title(),
                            y=pivot_cols[0].replace('_', ' ').title(),
                            color=value_col.replace('_', ' ').title()
                        ),
                        color_continuous_scale='Blues'
                    )
                except:
                    # Fall back to correlation matrix if pivot fails
                    numeric_data = data.select_dtypes(include=np.number)
                    if len(numeric_data.columns) >= 2:
                        corr = numeric_data.corr()
                        fig = px.imshow(
                            corr,
                            title=f"Correlation Matrix: {chart_title}",
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1
                        )
                    else:
                        # If all else fails, do a bar chart
                        fig = px.bar(
                            data, 
                            x=x_col, 
                            y=y_col,
                            title=chart_title,
                            labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()}
                        )
            else:
                # Not enough columns for a proper pivot
                numeric_data = data.select_dtypes(include=np.number)
                if len(numeric_data.columns) >= 2:
                    corr = numeric_data.corr()
                    fig = px.imshow(
                        corr,
                        title=f"Correlation Matrix: {chart_title}",
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1
                    )
                else:
                    # If all else fails, do a bar chart
                    fig = px.bar(
                        data, 
                        x=x_col, 
                        y=y_col,
                        title=chart_title,
                        labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()}
                    )
        else:
            # Not enough columns for heatmap, do correlation if numeric
            numeric_data = data.select_dtypes(include=np.number)
            if len(numeric_data.columns) >= 2:
                corr = numeric_data.corr()
                fig = px.imshow(
                    corr,
                    title=f"Correlation Matrix: {chart_title}",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1
                )
            else:
                # If all else fails, do a bar chart
                fig = px.bar(
                    data, 
                    x=x_col, 
                    y=y_col,
                    title=chart_title,
                    labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()}
                )
    
    else:
        # Default to bar chart if unknown visualization type
        fig = px.bar(
            data, 
            x=x_col, 
            y=y_col,
            title=chart_title,
            labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()}
        )
    
    # Apply common layout settings
    fig.update_layout(
        title={
            'text': chart_title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=80, b=40, l=40, r=40),
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig
