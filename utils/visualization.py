import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import re
from plotly.subplots import make_subplots

def create_visualization(data, viz_type, title=None):
    """
    Create an infographic-style plotly visualization based on the data and visualization type.
    
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
    
    # Define a modern color palette for infographic style
    color_palette = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', 
                     '#EC4899', '#6366F1', '#D97706', '#059669', '#7C3AED']
                     
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
    
    # Common figure layout adjustments for infographic style
    def apply_infographic_style(fig):
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            title=dict(
                text=f"<b>{chart_title}</b>",
                font=dict(size=24, family="Arial, sans-serif", color="#333"),
                x=0.5,
                y=0.95
            ),
            plot_bgcolor='rgba(240,240,240,0.2)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=80, t=100, b=80),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255,255,255,0.6)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            )
        )
        
        # Add a subtle grid for readability
        fig.update_xaxes(
            showgrid=True, 
            gridwidth=0.5, 
            gridcolor='rgba(211,211,211,0.5)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.3)'
        )
        fig.update_yaxes(
            showgrid=True, 
            gridwidth=0.5, 
            gridcolor='rgba(211,211,211,0.5)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.3)'
        )
        
        return fig
    
    # Add insights and highlights
    def add_insights(fig, data, x_col, y_col, viz_type):
        insights = []
        
        # Skip for some chart types where annotations may clutter
        if viz_type in ['pie', 'heatmap']:
            return fig
            
        try:
            if viz_type in ['bar', 'line'] and y_col in data.columns:
                # Find max value and its position
                if x_col in data.columns:
                    max_idx = data[y_col].idxmax()
                    max_val = data.iloc[max_idx][y_col]
                    max_label = str(data.iloc[max_idx][x_col])
                    
                    fig.add_annotation(
                        x=max_label if viz_type == 'bar' else data.iloc[max_idx][x_col],
                        y=max_val,
                        text=f"Peak: {max_val:.2f}",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor="#FF5722",
                        font=dict(size=11, color="#333"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#FF5722",
                        borderwidth=1,
                        borderpad=4,
                        ay=-40
                    )
                    
                    # Calculate the average for a reference line
                    avg = data[y_col].mean()
                    
                    # Add average line
                    fig.add_shape(
                        type="line",
                        x0=0,
                        y0=avg,
                        x1=1,
                        y1=avg,
                        xref="paper",
                        line=dict(
                            color="rgba(102, 102, 102, 0.8)",
                            width=2,
                            dash="dash",
                        ),
                    )
                    
                    # Add avg label
                    fig.add_annotation(
                        x=0.01,
                        y=avg,
                        xref="paper",
                        text=f"Avg: {avg:.2f}",
                        showarrow=False,
                        font=dict(size=10, color="#666"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#666",
                        borderwidth=1,
                        borderpad=3
                    )
                    
                    # Add percent of average annotation for the max value
                    pct_of_avg = (max_val / avg - 1) * 100
                    insights.append(f"{max_label} is {abs(pct_of_avg):.1f}% {'above' if pct_of_avg > 0 else 'below'} average")
        except Exception as e:
            # If any error occurs during insight generation, continue without insights
            pass
            
        # Add insights as a textbox annotation if we have any
        if insights:
            insight_text = "<br>".join([f"• {insight}" for insight in insights])
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.15,
                text=f"<b>Key Insights:</b><br>{insight_text}",
                showarrow=False,
                font=dict(size=12, color="#333"),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                borderpad=10,
                width=500
            )
            
        return fig
    
    # Create visualization based on type
    if viz_type == 'bar':
        # Handle cases with too many categories
        if x_col in data.columns and len(data[x_col].unique()) > 15:
            # Sort and take top 15
            sorted_data = data.sort_values(by=y_col, ascending=False).head(15)
        else:
            sorted_data = data
        
        # Create figure with a subplot to add title space for infographic elements
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=[""]  # Empty title, we'll add a custom one
        )
        
        # Add the bar chart
        fig.add_trace(
            go.Bar(
                x=sorted_data[x_col],
                y=sorted_data[y_col],
                marker_color=color_palette,
                text=sorted_data[y_col].round(2),
                textposition='auto',
                hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y:.2f}}<extra></extra>"
            )
        )
        
        # Update layout for infographic style
        fig = apply_infographic_style(fig)
        
        # Update axes labels
        fig.update_xaxes(title_text=x_col.replace('_', ' ').title())
        fig.update_yaxes(title_text=y_col.replace('_', ' ').title())
        
        # Add insights
        fig = add_insights(fig, sorted_data, x_col, y_col, viz_type)
    
    elif viz_type == 'line':
        # Create figure with subplot
        fig = make_subplots(
            rows=1, cols=1, 
            subplot_titles=[""]
        )
        
        # Add line trace
        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                marker=dict(size=8, color=color_palette[0]),
                line=dict(width=3, color=color_palette[0]),
                hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y:.2f}}<extra></extra>"
            )
        )
        
        # Update layout for infographic style
        fig = apply_infographic_style(fig)
        
        # Update axes labels
        fig.update_xaxes(title_text=x_col.replace('_', ' ').title())
        fig.update_yaxes(title_text=y_col.replace('_', ' ').title())
        
        # Add insights
        fig = add_insights(fig, data, x_col, y_col, viz_type)
    
    elif viz_type == 'pie':
        # Create a more visually appealing pie chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Pie(
                labels=data[x_col],
                values=data[y_col],
                hole=0.4,  # Make it a donut chart for modern look
                marker=dict(
                    colors=color_palette[:len(data[x_col].unique())],
                    line=dict(color='#fff', width=2)
                ),
                textinfo='label+percent',
                insidetextfont=dict(color='white'),
                hoverinfo='label+value+percent',
                textposition='inside'
            )
        )
        
        # Calculate total for center annotation
        total = data[y_col].sum()
        
        # Add center annotation with total
        fig.add_annotation(
            text=f"<b>Total<br>{total:.1f}</b>",
            x=0.5, y=0.5,
            font=dict(size=16, color='#333', family='Arial, sans-serif'),
            showarrow=False
        )
        
        # Update layout for infographic style
        fig = apply_infographic_style(fig)
        
        # Add distribution insights to pie chart
        top_category = data.sort_values(by=y_col, ascending=False).iloc[0]
        percentage = (top_category[y_col] / total) * 100
        
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            text=f"<b>Key Insight:</b><br>• {top_category[x_col]} represents the largest portion at {percentage:.1f}% of the total",
            showarrow=False,
            font=dict(size=12, color="#333"),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            borderpad=10,
            width=500
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
            
            # Create infographic-style scatter plot
            fig = go.Figure()
            
            if color_col:
                # Create a colored scatter plot with trendline for each category
                for category in data[color_col].unique():
                    subset = data[data[color_col] == category]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=subset[x_col],
                            y=subset[y_col],
                            mode='markers',
                            marker=dict(
                                size=10,
                                opacity=0.7,
                                line=dict(width=1, color='white')
                            ),
                            name=str(category),
                            hovertemplate=f"{color_col}: {category}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                        )
                    )
                    
                    # Add trendline if we have enough points
                    if len(subset) > 2:
                        try:
                            z = np.polyfit(subset[x_col], subset[y_col], 1)
                            p = np.poly1d(z)
                            
                            x_range = np.linspace(subset[x_col].min(), subset[x_col].max(), 100)
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=x_range,
                                    y=p(x_range),
                                    mode='lines',
                                    line=dict(dash='dash', width=2),
                                    showlegend=False,
                                    hoverinfo='skip'
                                )
                            )
                        except:
                            pass
            else:
                # Create a simple scatter plot with trendline
                fig.add_trace(
                    go.Scatter(
                        x=data[x_col],
                        y=data[y_col],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=color_palette[0],
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                    )
                )
                
                # Add trendline
                try:
                    z = np.polyfit(data[x_col], data[y_col], 1)
                    p = np.poly1d(z)
                    
                    x_range = np.linspace(data[x_col].min(), data[x_col].max(), 100)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=p(x_range),
                            mode='lines',
                            name='Trend',
                            line=dict(color='rgba(0,0,0,0.5)', dash='dash', width=2)
                        )
                    )
                    
                    # Add correlation coefficient
                    corr = np.corrcoef(data[x_col], data[y_col])[0,1]
                    
                    fig.add_annotation(
                        xref="paper",
                        yref="paper",
                        x=0.05,
                        y=0.95,
                        text=f"Correlation: {corr:.2f}",
                        showarrow=False,
                        font=dict(size=12, color="#333"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="rgba(0,0,0,0.1)",
                        borderwidth=1,
                        borderpad=4
                    )
                except:
                    pass
            
            # Apply infographic styling
            fig = apply_infographic_style(fig)
            
            # Update axes labels
            fig.update_xaxes(title_text=x_col.replace('_', ' ').title())
            fig.update_yaxes(title_text=y_col.replace('_', ' ').title())
            
        else:
            # Fall back to bar chart if we don't have enough numeric columns
            fig = px.bar(
                data, 
                x=x_col, 
                y=y_col,
                title=chart_title,
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
                color_discrete_sequence=color_palette
            )
            fig = apply_infographic_style(fig)
            fig = add_insights(fig, data, x_col, y_col, 'bar')
    
    elif viz_type == 'histogram':
        # Find the first numeric column
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        if len(numeric_cols) > 0:
            x_col = numeric_cols[0]
            
            # Create histogram with more infographic style
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=data[x_col],
                    marker=dict(
                        color=color_palette[0],
                        line=dict(color='white', width=1)
                    ),
                    opacity=0.8,
                    hovertemplate=f"{x_col}: %{{x}}<br>Count: %{{y}}<extra></extra>"
                )
            )
            
            # Add distribution stats
            mean = data[x_col].mean()
            median = data[x_col].median()
            
            # Add mean and median lines
            fig.add_vline(x=mean, line_dash="solid", line_color="#EF4444", 
                          annotation_text="Mean", annotation_position="top right")
            fig.add_vline(x=median, line_dash="dash", line_color="#3B82F6", 
                          annotation_text="Median", annotation_position="top left")
            
            # Apply infographic styling
            fig = apply_infographic_style(fig)
            
            # Update axes labels
            fig.update_xaxes(title_text=x_col.replace('_', ' ').title())
            fig.update_yaxes(title_text="Frequency")
            
            # Add distribution insights
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.15,
                text=f"<b>Distribution Stats:</b><br>• Mean: {mean:.2f}<br>• Median: {median:.2f}<br>• Range: {data[x_col].min():.2f} to {data[x_col].max():.2f}",
                showarrow=False,
                font=dict(size=12, color="#333"),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                borderpad=10,
                width=500
            )
            
        else:
            # Fall back to bar chart for categorical data
            fig = px.bar(
                data, 
                x=x_col, 
                y=y_col,
                title=chart_title,
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
                color_discrete_sequence=color_palette
            )
            fig = apply_infographic_style(fig)
            fig = add_insights(fig, data, x_col, y_col, 'bar')
    
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
    
    return fig
