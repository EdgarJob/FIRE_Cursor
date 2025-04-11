import pandas as pd
import numpy as np

def get_templates():
    """
    Returns a dictionary of available analysis templates organized by categories.
    
    Returns:
        dict: Dictionary of template categories, names and descriptions
    """
    return {
        "Demographic Analysis": {
            "demographic_analysis": "Basic demographic breakdown (age, gender, location)",
            "demographic_summary_report": "Comprehensive summary of gender, age groups, and other categories"
        },
        "Program Performance": {
            "program_participation": "Analysis of program participation rates",
            "service_utilization": "Analysis of service usage patterns",
            "outcome_analysis": "Analysis of program outcomes and effectiveness",
            "outcome_impact_summary": "Aggregated report of key outcomes (completion status, satisfaction)"
        },
        "Geographic Insights": {
            "geographic_distribution": "Geographic distribution of beneficiaries",
            "regional_service_distribution": "Services delivered or beneficiaries served per region/district"
        },
        "Temporal Analysis": {
            "temporal_trends": "Analysis of changes over time",
            "time_based_trends": "Monthly or quarterly trends in services or beneficiary reach"
        },
        "Data Quality": {
            "missing_data_quality_check": "Identify columns with missing values, duplicates, or inconsistent formats",
            "duplicate_beneficiary_check": "Detect repeated entries using unique identifiers"
        },
        "Cross-Analysis": {
            "beneficiary_status": "Summary of beneficiary status categories",
            "gender_age_disaggregated": "Cross-tab showing service access or outcomes by age group and gender",
            "comparative_analysis": "Comparison between different groups or time periods",
            "vulnerability_assessment": "Assessment of vulnerability factors"
        },
        "Resource Analysis": {
            "funding_allocation": "Analysis of funding distribution",
            "top_n_analysis": "Top regions with most activity, services, or highest outcomes",
            "pivot_table_explorer": "Custom pivot tables for combinations like Services per Region per Month"
        }
    }

def get_flat_templates():
    """
    Returns a flattened dictionary of all available templates.
    
    Returns:
        dict: Dictionary of template names and descriptions
    """
    templates = get_templates()
    flat_templates = {}
    
    for category, category_templates in templates.items():
        flat_templates.update(category_templates)
    
    return flat_templates

def apply_template(template_name, df):
    """
    Apply a predefined analysis template to a dataframe.
    
    Args:
        template_name: The name of the template to apply
        df: DataFrame to analyze
        
    Returns:
        tuple: (result_df, visualization_type, description)
    """
    # Get column names (lowercase for case-insensitive matching)
    columns_lower = [col.lower() for col in df.columns]
    columns_dict = {col.lower(): col for col in df.columns}
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if template_name == "demographic_analysis":
        description = "Demographic Analysis of Beneficiaries"
        
        # Look for demographic columns
        gender_col = find_column(columns_lower, columns_dict, ['gender', 'sex'])
        age_col = find_column(columns_lower, columns_dict, ['age', 'years'])
        location_col = find_column(columns_lower, columns_dict, ['location', 'district', 'area', 'region', 'province', 'county'])
        
        # Prepare result based on available data
        result = pd.DataFrame()
        
        if gender_col:
            gender_counts = df[gender_col].value_counts().reset_index()
            gender_counts.columns = [gender_col, 'Count']
            result = gender_counts
            viz_type = 'pie'
        elif age_col:
            # Create age groups if age is available
            if df[age_col].dtype in (int, float, np.number):
                df['Age Group'] = pd.cut(df[age_col], bins=[0, 18, 35, 50, 65, 100], 
                                        labels=['0-18', '19-35', '36-50', '51-65', '65+'])
                age_counts = df['Age Group'].value_counts().reset_index()
                age_counts.columns = ['Age Group', 'Count']
                result = age_counts
                viz_type = 'bar'
            else:
                # If age is categorical already
                age_counts = df[age_col].value_counts().reset_index()
                age_counts.columns = [age_col, 'Count']
                result = age_counts
                viz_type = 'bar'
        elif location_col:
            location_counts = df[location_col].value_counts().head(15).reset_index()
            location_counts.columns = [location_col, 'Count']
            result = location_counts
            viz_type = 'bar'
        else:
            # Fallback to first categorical column
            if len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                cat_counts = df[cat_col].value_counts().head(15).reset_index()
                cat_counts.columns = [cat_col, 'Count']
                result = cat_counts
                viz_type = 'bar'
            else:
                # Very basic summary if no suitable columns found
                result = pd.DataFrame({
                    'Metric': ['Total Beneficiaries'],
                    'Value': [len(df)]
                })
                viz_type = 'bar'
        
        return result, viz_type, description
    
    elif template_name == "program_participation":
        description = "Program Participation Analysis"
        
        # Look for program/participation-related columns
        program_col = find_column(columns_lower, columns_dict, ['program', 'intervention', 'service', 'activity'])
        status_col = find_column(columns_lower, columns_dict, ['status', 'participation', 'enrolled', 'attendance'])
        date_col = find_column(columns_lower, columns_dict, ['date', 'start_date', 'enrollment_date'])
        
        if program_col:
            program_counts = df[program_col].value_counts().reset_index()
            program_counts.columns = [program_col, 'Participants']
            viz_type = 'bar'
            return program_counts, viz_type, description
        elif status_col:
            status_counts = df[status_col].value_counts().reset_index()
            status_counts.columns = [status_col, 'Count']
            viz_type = 'pie'
            return status_counts, viz_type, description
        elif date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            # Create time series if date column is available
            df['Year-Month'] = df[date_col].dt.strftime('%Y-%m')
            enrollment_trend = df.groupby('Year-Month').size().reset_index()
            enrollment_trend.columns = ['Time Period', 'New Enrollments']
            viz_type = 'line'
            return enrollment_trend, viz_type, description
        else:
            # Fallback to participation count by a categorical column
            if len(categorical_cols) > 1:
                main_cat = categorical_cols[0]
                second_cat = categorical_cols[1]
                cross_tab = pd.crosstab(df[main_cat], df[second_cat])
                cross_tab_reset = cross_tab.reset_index().melt(id_vars=main_cat)
                cross_tab_reset.columns = [main_cat, 'Category', 'Count']
                viz_type = 'bar'
                return cross_tab_reset, viz_type, description
            else:
                # Basic count if no suitable columns
                result = pd.DataFrame({
                    'Metric': ['Total Participants'],
                    'Value': [len(df)]
                })
                viz_type = 'bar'
                return result, viz_type, description
    
    elif template_name == "beneficiary_status":
        description = "Beneficiary Status Analysis"
        
        # Look for status-related columns
        status_col = find_column(columns_lower, columns_dict, ['status', 'state', 'condition', 'category'])
        type_col = find_column(columns_lower, columns_dict, ['type', 'beneficiary_type', 'group'])
        
        if status_col:
            status_counts = df[status_col].value_counts().reset_index()
            status_counts.columns = [status_col, 'Count']
            viz_type = 'pie'
            return status_counts, viz_type, description
        elif type_col:
            type_counts = df[type_col].value_counts().reset_index()
            type_counts.columns = [type_col, 'Count']
            viz_type = 'pie'
            return type_counts, viz_type, description
        else:
            # Try to find another categorical column
            for col in categorical_cols:
                if len(df[col].unique()) <= 10:  # Only if it has a reasonable number of categories
                    cat_counts = df[col].value_counts().reset_index()
                    cat_counts.columns = [col, 'Count']
                    viz_type = 'pie'
                    return cat_counts, viz_type, description
            
            # Very basic summary if no suitable columns found
            result = pd.DataFrame({
                'Metric': ['Total Beneficiaries'],
                'Value': [len(df)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "temporal_trends":
        description = "Temporal Trends Analysis"
        
        # Look for date/time columns
        date_col = find_column(columns_lower, columns_dict, ['date', 'year', 'month', 'time'])
        
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            # Create a time series by year-month
            df['Year-Month'] = df[date_col].dt.strftime('%Y-%m')
            time_series = df.groupby('Year-Month').size().reset_index()
            time_series.columns = ['Time Period', 'Count']
            viz_type = 'line'
            return time_series, viz_type, description
        elif date_col and 'year' in date_col.lower():
            # If it's a year column
            year_counts = df[date_col].value_counts().reset_index()
            year_counts.columns = [date_col, 'Count']
            # Sort by year if possible
            try:
                year_counts = year_counts.sort_values(date_col)
            except:
                pass
            viz_type = 'line'
            return year_counts, viz_type, description
        else:
            # Try to find any column that might represent time periods
            for col in categorical_cols:
                if any(period in col.lower() for period in ['year', 'month', 'quarter', 'period', 'phase']):
                    period_counts = df[col].value_counts().reset_index()
                    period_counts.columns = [col, 'Count']
                    viz_type = 'line'
                    return period_counts, viz_type, description
            
            # If we have numeric columns, show their trends
            if len(numeric_cols) > 0:
                # Take the first numeric column and show its average by first categorical column
                num_col = numeric_cols[0]
                if len(categorical_cols) > 0:
                    cat_col = categorical_cols[0]
                    agg_data = df.groupby(cat_col)[num_col].mean().reset_index()
                    agg_data.columns = [cat_col, f'Average {num_col}']
                    viz_type = 'bar'
                    return agg_data, viz_type, description
            
            # Fallback to simple count
            result = pd.DataFrame({
                'Metric': ['Data points', 'Unique categories'],
                'Value': [len(df), sum(len(df[col].unique()) for col in df.columns if col in categorical_cols)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "service_utilization":
        description = "Service Utilization Analysis"
        
        # Look for service/utilization related columns
        service_col = find_column(columns_lower, columns_dict, ['service', 'program', 'assistance', 'aid', 'support'])
        count_col = find_column(columns_lower, columns_dict, ['count', 'frequency', 'visits', 'times', 'sessions'])
        
        if service_col:
            service_counts = df[service_col].value_counts().reset_index()
            service_counts.columns = [service_col, 'Utilization Count']
            viz_type = 'bar'
            return service_counts, viz_type, description
        elif count_col and df[count_col].dtype in (int, float, np.number):
            # If we have a frequency/count column
            if len(categorical_cols) > 0:
                # Group by the first categorical column
                cat_col = categorical_cols[0]
                utilization_by_cat = df.groupby(cat_col)[count_col].sum().reset_index()
                utilization_by_cat.columns = [cat_col, 'Utilization Count']
                viz_type = 'bar'
                return utilization_by_cat, viz_type, description
            else:
                # Just summarize the count column
                result = pd.DataFrame({
                    'Metric': ['Total Utilization', 'Average per Beneficiary'],
                    'Value': [df[count_col].sum(), df[count_col].mean()]
                })
                viz_type = 'bar'
                return result, viz_type, description
        else:
            # Try to find patterns in categorical data
            if len(categorical_cols) >= 2:
                cat1 = categorical_cols[0]
                cat2 = categorical_cols[1]
                cross_tab = pd.crosstab(df[cat1], df[cat2])
                # Convert to long format for visualization
                cross_tab_reset = cross_tab.reset_index().melt(id_vars=cat1)
                cross_tab_reset.columns = [cat1, cat2, 'Count']
                viz_type = 'bar'
                return cross_tab_reset, viz_type, description
            
            # Basic summary stats
            result = pd.DataFrame({
                'Metric': ['Total Records'],
                'Value': [len(df)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "funding_allocation":
        description = "Funding Allocation Analysis"
        
        # Look for funding/financial columns
        amount_col = find_column(columns_lower, columns_dict, ['amount', 'funding', 'budget', 'cost', 'expense', 'grant', 'money'])
        category_col = find_column(columns_lower, columns_dict, ['category', 'type', 'purpose', 'allocation', 'project'])
        
        if amount_col and df[amount_col].dtype in (int, float, np.number):
            if category_col:
                # Group by category and sum amounts
                funding_by_cat = df.groupby(category_col)[amount_col].sum().reset_index()
                funding_by_cat.columns = [category_col, f'Total {amount_col}']
                # Sort by amount
                funding_by_cat = funding_by_cat.sort_values(f'Total {amount_col}', ascending=False)
                viz_type = 'pie'  # Pie chart is good for showing proportions of a whole
                return funding_by_cat, viz_type, description
            else:
                # Find another suitable categorical column
                for col in categorical_cols:
                    if len(df[col].unique()) <= 15:  # Reasonable number of categories
                        funding_by_cat = df.groupby(col)[amount_col].sum().reset_index()
                        funding_by_cat.columns = [col, f'Total {amount_col}']
                        funding_by_cat = funding_by_cat.sort_values(f'Total {amount_col}', ascending=False)
                        viz_type = 'pie'
                        return funding_by_cat, viz_type, description
                
                # Just summarize the amount column
                result = pd.DataFrame({
                    'Metric': ['Total Funding', 'Average per Entry', 'Maximum', 'Minimum'],
                    'Value': [
                        df[amount_col].sum(),
                        df[amount_col].mean(),
                        df[amount_col].max(),
                        df[amount_col].min()
                    ]
                })
                viz_type = 'bar'
                return result, viz_type, description
        else:
            # Try to find any numeric column that might represent funding
            for col in numeric_cols:
                funding_by_cat = df.groupby(categorical_cols[0])[col].sum().reset_index() if len(categorical_cols) > 0 else None
                if funding_by_cat is not None:
                    funding_by_cat.columns = [categorical_cols[0], f'Total {col}']
                    funding_by_cat = funding_by_cat.sort_values(f'Total {col}', ascending=False)
                    viz_type = 'bar'
                    return funding_by_cat, viz_type, description
            
            # If no suitable data, return basic summary
            result = pd.DataFrame({
                'Metric': ['Total Entries'],
                'Value': [len(df)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "outcome_analysis":
        description = "Program Outcome Analysis"
        
        # Look for outcome-related columns
        outcome_col = find_column(columns_lower, columns_dict, ['outcome', 'result', 'impact', 'success', 'achievement', 'status'])
        score_col = find_column(columns_lower, columns_dict, ['score', 'rating', 'level', 'grade', 'assessment', 'evaluation'])
        
        if outcome_col:
            outcome_counts = df[outcome_col].value_counts().reset_index()
            outcome_counts.columns = [outcome_col, 'Count']
            viz_type = 'pie'
            return outcome_counts, viz_type, description
        elif score_col and df[score_col].dtype in (int, float, np.number):
            # If we have a numeric score column
            
            # Create score distribution
            if len(categorical_cols) > 0:
                # Group by a categorical column and show average scores
                cat_col = categorical_cols[0]
                scores_by_cat = df.groupby(cat_col)[score_col].mean().reset_index()
                scores_by_cat.columns = [cat_col, f'Average {score_col}']
                scores_by_cat = scores_by_cat.sort_values(f'Average {score_col}', ascending=False)
                viz_type = 'bar'
                return scores_by_cat, viz_type, description
            else:
                # Create a histogram of scores
                score_hist = pd.cut(df[score_col], bins=5).value_counts().reset_index()
                score_hist.columns = ['Score Range', 'Count']
                viz_type = 'bar'
                return score_hist, viz_type, description
        else:
            # Try to find any categorical column that might represent outcomes
            for col in categorical_cols:
                if len(df[col].unique()) <= 10:  # Reasonable number of categories
                    outcome_counts = df[col].value_counts().reset_index()
                    outcome_counts.columns = [col, 'Count']
                    viz_type = 'pie'
                    return outcome_counts, viz_type, description
            
            # Basic summary if no suitable columns
            result = pd.DataFrame({
                'Metric': ['Total Records'],
                'Value': [len(df)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "geographic_distribution":
        description = "Geographic Distribution of Beneficiaries"
        
        # Look for geographic columns
        geo_col = find_column(columns_lower, columns_dict, ['location', 'district', 'region', 'province', 'county', 'city', 'state', 'country', 'area'])
        
        if geo_col:
            geo_counts = df[geo_col].value_counts().head(15).reset_index()
            geo_counts.columns = [geo_col, 'Count']
            viz_type = 'bar'
            return geo_counts, viz_type, description
        else:
            # Try to find any column that might represent location
            for col in categorical_cols:
                if len(df[col].unique()) <= 20:  # Not too many unique values
                    location_counts = df[col].value_counts().head(15).reset_index()
                    location_counts.columns = [col, 'Count']
                    viz_type = 'bar'
                    return location_counts, viz_type, description
            
            # Basic summary if no suitable columns
            result = pd.DataFrame({
                'Metric': ['Total Records'],
                'Value': [len(df)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "vulnerability_assessment":
        description = "Vulnerability Assessment Analysis"
        
        # Look for vulnerability-related columns
        vulnerability_col = find_column(columns_lower, columns_dict, ['vulnerability', 'risk', 'need', 'priority', 'severity'])
        factor_col = find_column(columns_lower, columns_dict, ['factor', 'indicator', 'condition', 'criteria'])
        
        if vulnerability_col:
            if df[vulnerability_col].dtype in (int, float, np.number):
                # Create vulnerability score distribution
                vuln_hist = pd.cut(df[vulnerability_col], bins=5).value_counts().reset_index()
                vuln_hist.columns = ['Vulnerability Level', 'Count']
                viz_type = 'bar'
                return vuln_hist, viz_type, description
            else:
                # Categorical vulnerability
                vuln_counts = df[vulnerability_col].value_counts().reset_index()
                vuln_counts.columns = [vulnerability_col, 'Count']
                viz_type = 'pie'
                return vuln_counts, viz_type, description
        elif factor_col:
            factor_counts = df[factor_col].value_counts().reset_index()
            factor_counts.columns = [factor_col, 'Count']
            viz_type = 'bar'
            return factor_counts, viz_type, description
        else:
            # Try to identify vulnerability factors from available columns
            result_data = []
            
            # Look for potential vulnerability indicators
            for col in categorical_cols:
                # Skip columns with too many unique values
                if len(df[col].unique()) <= 10:
                    col_counts = df[col].value_counts()
                    # Add the most common category
                    most_common = col_counts.index[0]
                    result_data.append({
                        'Factor': col,
                        'Most Common': most_common,
                        'Count': col_counts[most_common],
                        'Percentage': f"{col_counts[most_common] / len(df) * 100:.1f}%"
                    })
            
            if result_data:
                result = pd.DataFrame(result_data)
                viz_type = 'bar'
                return result, viz_type, description
            
            # Basic summary if no suitable columns
            result = pd.DataFrame({
                'Metric': ['Total Records'],
                'Value': [len(df)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "comparative_analysis":
        description = "Comparative Analysis between Groups"
        
        # Look for columns to compare
        comparison_col = find_column(columns_lower, columns_dict, ['group', 'category', 'type', 'status', 'condition'])
        
        if comparison_col and len(df[comparison_col].unique()) <= 10:
            # Find a good metric to compare across groups
            if len(numeric_cols) > 0:
                # Use the first numeric column as the metric
                metric_col = numeric_cols[0]
                
                # Calculate mean, count, and sum by group
                comparison = df.groupby(comparison_col).agg({
                    metric_col: ['mean', 'count', 'sum']
                }).reset_index()
                
                # Flatten the column names
                comparison.columns = [comparison_col, f'Average {metric_col}', 'Count', f'Total {metric_col}']
                
                viz_type = 'bar'
                return comparison, viz_type, description
            else:
                # If no numeric columns, find another categorical column to cross-tabulate
                for col in categorical_cols:
                    if col != comparison_col and len(df[col].unique()) <= 10:
                        cross_tab = pd.crosstab(df[comparison_col], df[col])
                        # Convert to long format for visualization
                        cross_tab_reset = cross_tab.reset_index().melt(id_vars=comparison_col)
                        cross_tab_reset.columns = [comparison_col, 'Category', 'Count']
                        viz_type = 'bar'
                        return cross_tab_reset, viz_type, description
                
                # If no good second categorical column, just count by the comparison column
                counts = df[comparison_col].value_counts().reset_index()
                counts.columns = [comparison_col, 'Count']
                viz_type = 'bar'
                return counts, viz_type, description
        else:
            # Try to find a good categorical column with a reasonable number of categories
            for col in categorical_cols:
                if len(df[col].unique()) <= 7:  # Not too many groups for comparison
                    if len(numeric_cols) > 0:
                        # Use the first numeric column as the metric
                        metric_col = numeric_cols[0]
                        comparison = df.groupby(col)[metric_col].mean().reset_index()
                        comparison.columns = [col, f'Average {metric_col}']
                        viz_type = 'bar'
                        return comparison, viz_type, description
                    else:
                        # Just count by the categorical column
                        counts = df[col].value_counts().reset_index()
                        counts.columns = [col, 'Count']
                        viz_type = 'bar'
                        return counts, viz_type, description
            
            # If no suitable columns for comparison
            if len(numeric_cols) >= 2:
                # Compare two numeric columns
                scatter_data = df[[numeric_cols[0], numeric_cols[1]]].copy()
                viz_type = 'scatter'
                return scatter_data, viz_type, description
            
            # Basic summary if no suitable columns for comparison
            result = pd.DataFrame({
                'Metric': ['Total Records'],
                'Value': [len(df)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "demographic_summary_report":
        description = "Comprehensive Demographic Summary Report"
        
        # Look for demographic columns
        gender_col = find_column(columns_lower, columns_dict, ['gender', 'sex'])
        age_col = find_column(columns_lower, columns_dict, ['age', 'years'])
        education_col = find_column(columns_lower, columns_dict, ['education', 'edu', 'qualification'])
        occupation_col = find_column(columns_lower, columns_dict, ['occupation', 'job', 'profession', 'work'])
        
        # Create a list to store DataFrames for each demographic dimension
        demographic_dfs = []
        
        # Process gender if available
        if gender_col:
            gender_counts = df[gender_col].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            gender_counts['Percentage'] = (gender_counts['Count'] / gender_counts['Count'].sum() * 100).round(1)
            gender_counts['Category'] = 'Gender'
            demographic_dfs.append(gender_counts)
        
        # Process age groups if available
        if age_col and df[age_col].dtype in (int, float, np.number):
            # Create age bins
            df['Age Group'] = pd.cut(df[age_col], 
                                    bins=[0, 12, 18, 25, 35, 50, 65, 100], 
                                    labels=['0-12', '13-18', '19-25', '26-35', '36-50', '51-65', '65+'])
            age_counts = df['Age Group'].value_counts().reset_index()
            age_counts.columns = ['Age Group', 'Count']
            age_counts['Percentage'] = (age_counts['Count'] / age_counts['Count'].sum() * 100).round(1)
            age_counts['Category'] = 'Age'
            demographic_dfs.append(age_counts)
        
        # Process education if available
        if education_col:
            edu_counts = df[education_col].value_counts().reset_index()
            edu_counts.columns = ['Education', 'Count']
            edu_counts['Percentage'] = (edu_counts['Count'] / edu_counts['Count'].sum() * 100).round(1)
            edu_counts['Category'] = 'Education'
            demographic_dfs.append(edu_counts)
        
        # Process occupation if available
        if occupation_col:
            # Limit to top 10 occupations
            occ_counts = df[occupation_col].value_counts().head(10).reset_index()
            occ_counts.columns = ['Occupation', 'Count']
            occ_counts['Percentage'] = (occ_counts['Count'] / df.shape[0] * 100).round(1)
            occ_counts['Category'] = 'Occupation'
            demographic_dfs.append(occ_counts)
        
        # If we have demographic data, combine and return
        if demographic_dfs:
            # Combine all dimensions
            result = pd.concat(demographic_dfs, ignore_index=True)
            viz_type = 'bar'
            return result, viz_type, description
        else:
            # Fallback if no demographic columns found
            result = pd.DataFrame({
                'Metric': ['Total Beneficiaries', 'Number of Categories'],
                'Value': [len(df), sum(len(df[col].unique()) for col in categorical_cols)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "regional_service_distribution":
        description = "Regional Service Distribution Analysis"
        
        # Look for region/location columns
        region_col = find_column(columns_lower, columns_dict, ['region', 'location', 'district', 'province', 'county', 'site', 'area'])
        service_col = find_column(columns_lower, columns_dict, ['service', 'program', 'assistance', 'intervention', 'aid'])
        
        if region_col and service_col:
            # Distribution of services by region
            cross_tab = pd.crosstab(df[region_col], df[service_col])
            
            # Get total services by region
            cross_tab['Total'] = cross_tab.sum(axis=1)
            
            # Convert to long format for visualization
            result = cross_tab.reset_index().melt(id_vars=[region_col, 'Total'], 
                                                var_name='Service', 
                                                value_name='Count')
            
            # Sort by total count descending
            result = result.sort_values('Total', ascending=False)
            
            viz_type = 'bar'
            return result, viz_type, description
        
        elif region_col:
            # Just a count of beneficiaries by region
            region_counts = df[region_col].value_counts().reset_index()
            region_counts.columns = ['Region', 'Beneficiaries']
            
            # Limit to top 15 regions
            region_counts = region_counts.head(15)
            
            viz_type = 'bar'
            return region_counts, viz_type, description
        
        else:
            # Try to find any columns that might indicate location
            for col in categorical_cols:
                if len(df[col].unique()) <= 30:  # Only use columns with reasonable number of categories
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, 'Count']
                    viz_type = 'bar'
                    return counts, viz_type, description
            
            # Fallback to basic summary
            result = pd.DataFrame({
                'Metric': ['Total Records'],
                'Value': [len(df)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "time_based_trends":
        description = "Time-Based Trends Analysis"
        
        # Look for date columns
        date_col = find_column(columns_lower, columns_dict, ['date', 'datetime', 'created', 'registered', 'start_date'])
        
        if date_col:
            # Try to convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                except:
                    pass
            
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                # Create a copy to avoid modifying original dataframe
                trend_df = df.copy()
                
                # Extract month and year
                trend_df['Year'] = trend_df[date_col].dt.year
                trend_df['Month'] = trend_df[date_col].dt.month
                trend_df['Year-Month'] = trend_df[date_col].dt.strftime('%Y-%m')
                
                # Count records by month
                monthly_counts = trend_df.groupby('Year-Month').size().reset_index()
                monthly_counts.columns = ['Period', 'Count']
                
                # Sort chronologically
                monthly_counts = monthly_counts.sort_values('Period')
                
                # Calculate month-over-month change
                monthly_counts['Previous'] = monthly_counts['Count'].shift(1)
                monthly_counts['Change %'] = ((monthly_counts['Count'] / monthly_counts['Previous'] - 1) * 100).round(1)
                
                viz_type = 'line'
                return monthly_counts, viz_type, description
        
        # Look for year or month columns
        year_col = find_column(columns_lower, columns_dict, ['year', 'yr'])
        month_col = find_column(columns_lower, columns_dict, ['month', 'mo'])
        
        if year_col and month_col:
            # Create a combined year-month field
            try:
                # Attempt to create year-month string
                df['Year-Month'] = df[year_col].astype(str) + '-' + df[month_col].astype(str).str.zfill(2)
                
                # Count by year-month
                period_counts = df.groupby('Year-Month').size().reset_index()
                period_counts.columns = ['Period', 'Count']
                
                # Try to sort chronologically
                period_counts = period_counts.sort_values('Period')
                
                viz_type = 'line'
                return period_counts, viz_type, description
            except:
                pass
        
        elif year_col:
            # Just use year column
            year_counts = df.groupby(year_col).size().reset_index()
            year_counts.columns = ['Year', 'Count']
            
            # Sort by year
            try:
                year_counts = year_counts.sort_values('Year')
            except:
                pass
            
            viz_type = 'line'
            return year_counts, viz_type, description
            
        # Fallback to temporal analysis template
        else:
            # Try to use any categorical column that might indicate time periods
            for col in categorical_cols:
                if any(period in col.lower() for period in ['period', 'quarter', 'phase', 'term']):
                    period_counts = df.groupby(col).size().reset_index()
                    period_counts.columns = [col, 'Count']
                    viz_type = 'line'
                    return period_counts, viz_type, description
            
            # Basic summary if no suitable columns
            result = pd.DataFrame({
                'Metric': ['Total Records'],
                'Value': [len(df)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "missing_data_quality_check":
        description = "Missing Data & Quality Check Report"
        
        # Calculate missing values for each column
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        
        # Calculate percentage of missing values
        missing_data['Missing %'] = (missing_data['Missing Values'] / len(df) * 100).round(1)
        
        # Sort by missing values (descending)
        missing_data = missing_data.sort_values('Missing Values', ascending=False)
        
        # Add data type information
        missing_data['Data Type'] = missing_data['Column'].map(lambda col: str(df[col].dtype))
        
        # Add unique value counts to assess cardinality issues
        missing_data['Unique Values'] = missing_data['Column'].map(lambda col: df[col].nunique())
        
        # Count duplicate rows
        duplicate_count = df.duplicated().sum()
        
        # Find problematic columns
        problematic_cols = []
        for col in df.columns:
            # Check for mixed data types (potential inconsistency)
            if df[col].dtype == 'object':
                numeric_count = pd.to_numeric(df[col], errors='coerce').notnull().sum()
                if 0 < numeric_count < len(df):  # Mixed types
                    problematic_cols.append({'Column': col, 'Issue': 'Mixed data types'})
            
            # Check for outliers in numeric columns
            if df[col].dtype in (int, float, np.number):
                if len(df) > 10:  # Only check if we have enough data
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    outlier_count = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)].shape[0]
                    if outlier_count > 0:
                        problematic_cols.append({'Column': col, 'Issue': f'{outlier_count} outliers detected'})
        
        # Add summary stats to the result
        summary_data = [
            {'Column': 'SUMMARY', 'Missing Values': '', 'Missing %': '', 'Data Type': '', 'Unique Values': ''},
            {'Column': 'Total Rows', 'Missing Values': len(df), 'Missing %': 100.0, 'Data Type': '', 'Unique Values': ''},
            {'Column': 'Duplicate Rows', 'Missing Values': duplicate_count, 'Missing %': round(duplicate_count/len(df)*100, 1) if len(df) > 0 else 0, 'Data Type': '', 'Unique Values': ''},
            {'Column': 'Columns with Missing Data', 'Missing Values': (missing_data['Missing Values'] > 0).sum(), 'Missing %': '', 'Data Type': '', 'Unique Values': ''}
        ]
        
        # Combine the results
        result = pd.concat([pd.DataFrame(summary_data), missing_data])
        
        # Add problematic columns if any were found
        if problematic_cols:
            problematic_df = pd.DataFrame(problematic_cols)
            # Format for display
            problematic_df['Column'] = 'ISSUE: ' + problematic_df['Column']
            problematic_df['Missing Values'] = ''
            problematic_df['Missing %'] = ''
            problematic_df['Data Type'] = problematic_df['Issue']
            problematic_df['Unique Values'] = ''
            problematic_df = problematic_df.drop('Issue', axis=1)
            
            # Add to result
            result = pd.concat([result, problematic_df])
        
        viz_type = 'bar'
        return result, viz_type, description
    
    elif template_name == "duplicate_beneficiary_check":
        description = "Duplicate Beneficiary Check"
        
        # Look for ID columns
        id_col = find_column(columns_lower, columns_dict, ['id', 'identifier', 'beneficiary_id', 'person_id', 'case_id'])
        name_col = find_column(columns_lower, columns_dict, ['name', 'beneficiary', 'person', 'client'])
        
        # Initialize results
        duplicate_summary = []
        duplicates_found = False
        
        # Check for duplicates by ID if available
        if id_col:
            # Count occurrences of each ID
            id_counts = df[id_col].value_counts().reset_index()
            id_counts.columns = [id_col, 'Occurrences']
            
            # Filter for duplicates (count > 1)
            duplicate_ids = id_counts[id_counts['Occurrences'] > 1]
            
            if len(duplicate_ids) > 0:
                duplicates_found = True
                duplicate_summary.append({
                    'Check Type': f'Duplicate {id_col}',
                    'Count': len(duplicate_ids),
                    'Details': f'{len(duplicate_ids)} IDs appear multiple times'
                })
                
                # Add the actual duplicate IDs to the results
                for i, row in duplicate_ids.head(10).iterrows():  # Limit to first 10
                    duplicate_summary.append({
                        'Check Type': f'Duplicate ID',
                        'Count': row['Occurrences'],
                        'Details': f"{id_col}: {row[id_col]}"
                    })
        
        # Check for duplicates by name if available
        if name_col:
            # Count occurrences of each name
            name_counts = df[name_col].value_counts().reset_index()
            name_counts.columns = [name_col, 'Occurrences']
            
            # Filter for duplicates (count > 1)
            duplicate_names = name_counts[name_counts['Occurrences'] > 1]
            
            if len(duplicate_names) > 0:
                duplicates_found = True
                duplicate_summary.append({
                    'Check Type': f'Duplicate {name_col}',
                    'Count': len(duplicate_names),
                    'Details': f'{len(duplicate_names)} names appear multiple times'
                })
                
                # Add the actual duplicate names to the results
                for i, row in duplicate_names.head(10).iterrows():  # Limit to first 10
                    duplicate_summary.append({
                        'Check Type': f'Duplicate Name',
                        'Count': row['Occurrences'],
                        'Details': f"{name_col}: {row[name_col]}"
                    })
        
        # Check for duplicates based on multiple columns
        if not duplicates_found:
            # Try to find combinations of columns that might identify a unique beneficiary
            potential_id_cols = []
            
            # Look for columns that might help identify unique records
            for col in df.columns:
                if (df[col].dtype == 'object' or 'id' in col.lower()) and df[col].nunique() > len(df) / 2:
                    potential_id_cols.append(col)
            
            # If we have potential columns, check for duplicates using combinations
            if len(potential_id_cols) >= 2:
                # Use first 2-3 columns with high cardinality
                id_columns = potential_id_cols[:min(3, len(potential_id_cols))]
                
                # Find duplicates
                duplicate_mask = df.duplicated(subset=id_columns, keep=False)
                duplicates = df[duplicate_mask]
                
                if len(duplicates) > 0:
                    duplicates_found = True
                    col_names = ', '.join(id_columns)
                    duplicate_summary.append({
                        'Check Type': f'Duplicate Combinations',
                        'Count': len(duplicates),
                        'Details': f'{len(duplicates)} rows have duplicate values for {col_names}'
                    })
        
        # If no specific duplicate checks found anything, check for exact duplicates
        if not duplicates_found:
            exact_duplicates = df.duplicated().sum()
            
            if exact_duplicates > 0:
                duplicate_summary.append({
                    'Check Type': 'Exact Duplicates',
                    'Count': exact_duplicates,
                    'Details': f'{exact_duplicates} rows are exact duplicates'
                })
            else:
                duplicate_summary.append({
                    'Check Type': 'No Duplicates Found',
                    'Count': 0,
                    'Details': 'No duplicate records detected'
                })
        
        # Create result DataFrame
        result = pd.DataFrame(duplicate_summary)
        
        viz_type = 'bar'
        return result, viz_type, description
    
    elif template_name == "outcome_impact_summary":
        description = "Outcome & Impact Summary"
        
        # Look for outcome related columns
        outcome_col = find_column(columns_lower, columns_dict, ['outcome', 'result', 'impact', 'effect', 'achievement'])
        status_col = find_column(columns_lower, columns_dict, ['status', 'completion', 'state', 'progress'])
        satisfaction_col = find_column(columns_lower, columns_dict, ['satisfaction', 'rating', 'feedback', 'score'])
        health_outcome_col = find_column(columns_lower, columns_dict, ['health', 'wellness', 'medical', 'condition'])
        
        # Initialize results storage
        outcome_results = []
        
        # Process outcome column if available
        if outcome_col:
            outcome_counts = df[outcome_col].value_counts().reset_index()
            outcome_counts.columns = ['Outcome', 'Count']
            outcome_counts['Percentage'] = (outcome_counts['Count'] / outcome_counts['Count'].sum() * 100).round(1)
            outcome_counts['Category'] = 'Outcome Type'
            outcome_results.append(outcome_counts)
        
        # Process status/completion if available
        if status_col:
            status_counts = df[status_col].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            status_counts['Percentage'] = (status_counts['Count'] / status_counts['Count'].sum() * 100).round(1)
            status_counts['Category'] = 'Completion Status'
            outcome_results.append(status_counts)
        
        # Process satisfaction if available
        if satisfaction_col:
            # Check if satisfaction is numeric
            if df[satisfaction_col].dtype in (int, float, np.number):
                # Create satisfaction categories
                if df[satisfaction_col].max() <= 5:  # Likely a 1-5 scale
                    df['Satisfaction Level'] = pd.cut(df[satisfaction_col], 
                                                    bins=[0, 1, 2, 3, 4, 5], 
                                                    labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
                    sat_counts = df['Satisfaction Level'].value_counts().reset_index()
                else:  # Custom scale
                    df['Satisfaction Level'] = pd.qcut(df[satisfaction_col], 
                                                     q=5, 
                                                     labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
                    sat_counts = df['Satisfaction Level'].value_counts().reset_index()
            else:
                # Use categorical values directly
                sat_counts = df[satisfaction_col].value_counts().reset_index()
            
            sat_counts.columns = ['Satisfaction', 'Count']
            sat_counts['Percentage'] = (sat_counts['Count'] / sat_counts['Count'].sum() * 100).round(1)
            sat_counts['Category'] = 'Satisfaction Level'
            outcome_results.append(sat_counts)
        
        # Process health outcome if available
        if health_outcome_col:
            health_counts = df[health_outcome_col].value_counts().reset_index()
            health_counts.columns = ['Health Outcome', 'Count']
            health_counts['Percentage'] = (health_counts['Count'] / health_counts['Count'].sum() * 100).round(1)
            health_counts['Category'] = 'Health Impact'
            outcome_results.append(health_counts)
        
        # If we found outcome data, combine and return
        if outcome_results:
            # Combine all outcome dimensions
            result = pd.concat(outcome_results, ignore_index=True)
            viz_type = 'bar'
            return result, viz_type, description
        
        # If no specific outcome columns, look for numeric indicators
        numeric_indicator_cols = []
        for col in numeric_cols:
            if any(indicator in col.lower() for indicator in ['score', 'rate', 'change', 'improvement', 'index']):
                numeric_indicator_cols.append(col)
        
        if numeric_indicator_cols:
            # Summarize numeric indicators
            summary_stats = []
            for col in numeric_indicator_cols[:5]:  # Limit to first 5 indicators
                summary_stats.append({
                    'Indicator': col,
                    'Average': df[col].mean().round(2),
                    'Minimum': df[col].min().round(2),
                    'Maximum': df[col].max().round(2),
                    'Median': df[col].median().round(2)
                })
            
            result = pd.DataFrame(summary_stats)
            viz_type = 'bar'
            return result, viz_type, description
        
        # Fallback to basic summary if no outcome indicators found
        result = pd.DataFrame({
            'Metric': ['Total Records', 'Categories Available'],
            'Value': [len(df), len(categorical_cols)]
        })
        viz_type = 'bar'
        return result, viz_type, description
    
    elif template_name == "gender_age_disaggregated":
        description = "Gender and Age Disaggregated Insights"
        
        # Look for demographic columns
        gender_col = find_column(columns_lower, columns_dict, ['gender', 'sex'])
        age_col = find_column(columns_lower, columns_dict, ['age', 'years'])
        service_col = find_column(columns_lower, columns_dict, ['service', 'program', 'assistance', 'intervention'])
        outcome_col = find_column(columns_lower, columns_dict, ['outcome', 'result', 'status', 'completion'])
        
        # Check if we have both gender and age columns
        if gender_col and age_col and df[age_col].dtype in (int, float, np.number):
            # Create age groups
            df['Age Group'] = pd.cut(df[age_col], 
                                   bins=[0, 12, 18, 25, 35, 50, 65, 100], 
                                   labels=['0-12', '13-18', '19-25', '26-35', '36-50', '51-65', '65+'])
            
            # If we have service or outcome column, create cross-tab with that
            if service_col:
                # Cross tabulation of gender, age group, and service
                cross_tab = pd.crosstab(
                    [df[gender_col], df['Age Group']], 
                    df[service_col],
                    margins=True,
                    margins_name='Total'
                )
                
                # Reset index for easier handling
                result = cross_tab.reset_index()
                
                viz_type = 'heatmap'
                return result, viz_type, description
                
            elif outcome_col:
                # Cross tabulation of gender, age group, and outcome
                cross_tab = pd.crosstab(
                    [df[gender_col], df['Age Group']], 
                    df[outcome_col],
                    margins=True,
                    margins_name='Total'
                )
                
                # Reset index for easier handling
                result = cross_tab.reset_index()
                
                viz_type = 'heatmap'
                return result, viz_type, description
                
            else:
                # Simple count by gender and age group
                cross_tab = pd.crosstab(
                    df[gender_col], 
                    df['Age Group'],
                    margins=True,
                    margins_name='Total'
                )
                
                # Reset index for easier handling
                result = cross_tab.reset_index()
                
                viz_type = 'heatmap'
                return result, viz_type, description
                
        elif gender_col:
            # If we only have gender but not age
            if service_col:
                # Cross-tab gender and service
                cross_tab = pd.crosstab(
                    df[gender_col], 
                    df[service_col],
                    margins=True,
                    margins_name='Total'
                )
                
                result = cross_tab.reset_index()
                viz_type = 'bar'
                return result, viz_type, description
                
            elif outcome_col:
                # Cross-tab gender and outcome
                cross_tab = pd.crosstab(
                    df[gender_col], 
                    df[outcome_col],
                    margins=True,
                    margins_name='Total'
                )
                
                result = cross_tab.reset_index()
                viz_type = 'bar'
                return result, viz_type, description
                
            else:
                # Simple gender distribution
                gender_counts = df[gender_col].value_counts().reset_index()
                gender_counts.columns = [gender_col, 'Count']
                
                viz_type = 'pie'
                return gender_counts, viz_type, description
                
        elif age_col and df[age_col].dtype in (int, float, np.number):
            # If we only have age but not gender
            df['Age Group'] = pd.cut(df[age_col], 
                                   bins=[0, 12, 18, 25, 35, 50, 65, 100], 
                                   labels=['0-12', '13-18', '19-25', '26-35', '36-50', '51-65', '65+'])
            
            if service_col:
                # Cross-tab age group and service
                cross_tab = pd.crosstab(
                    df['Age Group'], 
                    df[service_col],
                    margins=True,
                    margins_name='Total'
                )
                
                result = cross_tab.reset_index()
                viz_type = 'bar'
                return result, viz_type, description
                
            elif outcome_col:
                # Cross-tab age group and outcome
                cross_tab = pd.crosstab(
                    df['Age Group'], 
                    df[outcome_col],
                    margins=True,
                    margins_name='Total'
                )
                
                result = cross_tab.reset_index()
                viz_type = 'bar'
                return result, viz_type, description
                
            else:
                # Simple age distribution
                age_counts = df['Age Group'].value_counts().reset_index()
                age_counts.columns = ['Age Group', 'Count']
                
                viz_type = 'bar'
                return age_counts, viz_type, description
        
        # Fallback to general demographic analysis
        else:
            # Let's try to find any two categorical columns to cross-tabulate
            if len(categorical_cols) >= 2:
                cat1 = categorical_cols[0]
                cat2 = categorical_cols[1]
                
                # Limit categories if we have too many
                if df[cat1].nunique() > 10 or df[cat2].nunique() > 10:
                    # Get top categories
                    top_cat1 = df[cat1].value_counts().head(5).index.tolist()
                    top_cat2 = df[cat2].value_counts().head(5).index.tolist()
                    
                    # Filter data to top categories
                    filtered_df = df[df[cat1].isin(top_cat1) & df[cat2].isin(top_cat2)]
                    
                    # Create cross-tab
                    cross_tab = pd.crosstab(
                        filtered_df[cat1], 
                        filtered_df[cat2],
                        margins=True,
                        margins_name='Total'
                    )
                else:
                    # Create cross-tab of all categories
                    cross_tab = pd.crosstab(
                        df[cat1], 
                        df[cat2],
                        margins=True,
                        margins_name='Total'
                    )
                
                result = cross_tab.reset_index()
                viz_type = 'heatmap'
                return result, viz_type, description
            
            # Basic summary if we can't create cross-tabulation
            result = pd.DataFrame({
                'Metric': ['Total Records'],
                'Value': [len(df)]
            })
            viz_type = 'bar'
            return result, viz_type, description
    
    elif template_name == "top_n_analysis":
        description = "Top N Analysis"
        
        # Look for columns to analyze
        region_col = find_column(columns_lower, columns_dict, ['region', 'location', 'district', 'province', 'county', 'site'])
        service_col = find_column(columns_lower, columns_dict, ['service', 'program', 'assistance', 'intervention'])
        outcome_col = find_column(columns_lower, columns_dict, ['outcome', 'result', 'impact', 'achievement'])
        count_col = find_column(columns_lower, columns_dict, ['count', 'frequency', 'number', 'quantity'])
        
        # Set the number of top items to show
        top_n = 5
        
        # Results storage
        top_results = []
        
        # Top regions by count
        if region_col:
            top_regions = df[region_col].value_counts().head(top_n).reset_index()
            top_regions.columns = ['Region', 'Count']
            top_regions['Percentage'] = (top_regions['Count'] / len(df) * 100).round(1)
            top_regions['Category'] = 'Top Regions'
            top_results.append(top_regions)
        
        # Top services
        if service_col:
            top_services = df[service_col].value_counts().head(top_n).reset_index()
            top_services.columns = ['Service', 'Count']
            top_services['Percentage'] = (top_services['Count'] / len(df) * 100).round(1)
            top_services['Category'] = 'Top Services'
            top_results.append(top_services)
        
        # Top outcomes
        if outcome_col:
            top_outcomes = df[outcome_col].value_counts().head(top_n).reset_index()
            top_outcomes.columns = ['Outcome', 'Count']
            top_outcomes['Percentage'] = (top_outcomes['Count'] / len(df) * 100).round(1)
            top_outcomes['Category'] = 'Top Outcomes'
            top_results.append(top_outcomes)
        
        # If we have both region and count columns, show top regions by some metric
        if region_col and count_col and df[count_col].dtype in (int, float, np.number):
            # Aggregate by region
            region_metrics = df.groupby(region_col)[count_col].sum().reset_index()
            region_metrics.columns = ['Region', 'Total']
            
            # Get top regions by sum of count
            top_region_metrics = region_metrics.sort_values('Total', ascending=False).head(top_n)
            top_region_metrics['Percentage'] = (top_region_metrics['Total'] / top_region_metrics['Total'].sum() * 100).round(1)
            top_region_metrics['Category'] = f'Top Regions by {count_col}'
            top_results.append(top_region_metrics)
        
        # If we found any top analysis, combine and return
        if top_results:
            # Combine all top analyses
            result = pd.concat(top_results, ignore_index=True)
            viz_type = 'bar'
            return result, viz_type, description
        
        # If no specific columns for top analysis, use any categorical columns
        if len(categorical_cols) > 0:
            # Get top values for the first categorical column
            cat_col = categorical_cols[0]
            top_values = df[cat_col].value_counts().head(top_n).reset_index()
            top_values.columns = [cat_col, 'Count']
            top_values['Percentage'] = (top_values['Count'] / len(df) * 100).round(1)
            
            viz_type = 'bar'
            return top_values, viz_type, description
        
        # Numeric columns if no categorical
        if len(numeric_cols) > 0:
            # Sort by first numeric column and get top values
            num_col = numeric_cols[0]
            top_values = df.sort_values(num_col, ascending=False).head(top_n)
            
            viz_type = 'bar'
            return top_values, viz_type, description
        
        # Fallback
        result = pd.DataFrame({
            'Metric': ['Total Records'],
            'Value': [len(df)]
        })
        viz_type = 'bar'
        return result, viz_type, description
    
    elif template_name == "pivot_table_explorer":
        description = "Pivot Table Explorer"
        
        # Look for key columns to use in pivot
        region_col = find_column(columns_lower, columns_dict, ['region', 'location', 'district', 'province', 'county', 'site'])
        service_col = find_column(columns_lower, columns_dict, ['service', 'program', 'assistance', 'intervention'])
        date_col = find_column(columns_lower, columns_dict, ['date', 'datetime', 'created', 'registered'])
        gender_col = find_column(columns_lower, columns_dict, ['gender', 'sex'])
        count_col = find_column(columns_lower, columns_dict, ['count', 'frequency', 'number', 'quantity'])
        
        # If we have a date column, try to extract month
        month_col = None
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df['Month'] = df[date_col].dt.strftime('%Y-%m')
            month_col = 'Month'
        elif date_col:
            try:
                # Try to convert to datetime
                df['Month'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m')
                month_col = 'Month'
            except:
                pass
        
        # If we have region and service columns, create a pivot of Services per Region
        if region_col and service_col:
            # If we also have month, create a more complex pivot
            if month_col:
                # Limit to top regions and services if we have too many
                if df[region_col].nunique() > 10 or df[service_col].nunique() > 10:
                    top_regions = df[region_col].value_counts().head(8).index
                    top_services = df[service_col].value_counts().head(6).index
                    pivot_df = df[df[region_col].isin(top_regions) & df[service_col].isin(top_services)]
                else:
                    pivot_df = df
                
                # Create pivot table of Services per Region per Month
                if count_col and df[count_col].dtype in (int, float, np.number):
                    # Use the count column as values
                    pivot = pd.pivot_table(
                        pivot_df,
                        values=count_col,
                        index=[region_col],
                        columns=[month_col, service_col],
                        aggfunc='sum',
                        fill_value=0
                    )
                else:
                    # Count occurrences
                    pivot = pd.pivot_table(
                        pivot_df,
                        values=service_col,
                        index=[region_col],
                        columns=[month_col],
                        aggfunc='count',
                        fill_value=0
                    )
                
                # Reset index for visualization
                result = pivot.reset_index()
                
                viz_type = 'heatmap'
                return result, viz_type, description
            
            else:
                # Simple Service per Region pivot
                # Limit to top categories if needed
                if df[region_col].nunique() > 15 or df[service_col].nunique() > 10:
                    top_regions = df[region_col].value_counts().head(15).index
                    top_services = df[service_col].value_counts().head(10).index
                    pivot_df = df[df[region_col].isin(top_regions) & df[service_col].isin(top_services)]
                else:
                    pivot_df = df
                
                # Create pivot with service as columns
                if count_col and df[count_col].dtype in (int, float, np.number):
                    # Use the count column as values
                    pivot = pd.pivot_table(
                        pivot_df,
                        values=count_col,
                        index=[region_col],
                        columns=[service_col],
                        aggfunc='sum',
                        fill_value=0
                    )
                else:
                    # Count occurrences
                    pivot = pd.pivot_table(
                        pivot_df,
                        values=service_col,
                        index=[region_col],
                        columns=[service_col],
                        aggfunc='count',
                        fill_value=0
                    )
                
                # Reset index for visualization
                result = pivot.reset_index()
                
                viz_type = 'heatmap'
                return result, viz_type, description
        
        # If we have gender and service columns, create a pivot
        elif gender_col and service_col:
            # Create gender by service pivot
            if count_col and df[count_col].dtype in (int, float, np.number):
                # Use the count column as values
                pivot = pd.pivot_table(
                    df,
                    values=count_col,
                    index=[gender_col],
                    columns=[service_col],
                    aggfunc='sum',
                    fill_value=0
                )
            else:
                # Count occurrences
                pivot = pd.pivot_table(
                    df,
                    values=service_col,
                    index=[gender_col],
                    columns=[service_col],
                    aggfunc='count',
                    fill_value=0
                )
            
            # Reset index for visualization
            result = pivot.reset_index()
            
            viz_type = 'heatmap'
            return result, viz_type, description
        
        # Try to find any two categorical columns to create a pivot
        elif len(categorical_cols) >= 2:
            cat1 = categorical_cols[0]
            cat2 = categorical_cols[1]
            
            # Limit to top categories if needed
            if df[cat1].nunique() > 15 or df[cat2].nunique() > 10:
                top_cat1 = df[cat1].value_counts().head(15).index
                top_cat2 = df[cat2].value_counts().head(10).index
                pivot_df = df[df[cat1].isin(top_cat1) & df[cat2].isin(top_cat2)]
            else:
                pivot_df = df
            
            # Create pivot
            if len(numeric_cols) > 0:
                # Use first numeric column as values
                num_col = numeric_cols[0]
                pivot = pd.pivot_table(
                    pivot_df,
                    values=num_col,
                    index=[cat1],
                    columns=[cat2],
                    aggfunc='mean',
                    fill_value=0
                )
            else:
                # Count occurrences
                pivot = pd.pivot_table(
                    pivot_df,
                    values=cat2,
                    index=[cat1],
                    columns=[cat2],
                    aggfunc='count',
                    fill_value=0
                )
            
            # Reset index for visualization
            result = pivot.reset_index()
            
            viz_type = 'heatmap'
            return result, viz_type, description
        
        # Fallback
        result = pd.DataFrame({
            'Metric': ['Total Records', 'Categorical Columns', 'Numeric Columns'],
            'Value': [len(df), len(categorical_cols), len(numeric_cols)]
        })
        viz_type = 'bar'
        return result, viz_type, description
    
    else:
        # Default basic analysis if template not found
        description = "Basic Data Summary"
        
        # Gather basic stats
        result_data = [
            {'Metric': 'Total Records', 'Value': len(df)},
            {'Metric': 'Number of Columns', 'Value': len(df.columns)},
            {'Metric': 'Numeric Columns', 'Value': len(numeric_cols)},
            {'Metric': 'Categorical Columns', 'Value': len(categorical_cols)}
        ]
        
        # Add stats for some key columns if available
        if len(numeric_cols) > 0:
            first_num = numeric_cols[0]
            result_data.extend([
                {'Metric': f'Average {first_num}', 'Value': df[first_num].mean()},
                {'Metric': f'Maximum {first_num}', 'Value': df[first_num].max()},
                {'Metric': f'Minimum {first_num}', 'Value': df[first_num].min()}
            ])
        
        if len(categorical_cols) > 0:
            first_cat = categorical_cols[0]
            result_data.append({
                'Metric': f'Unique values in {first_cat}', 
                'Value': len(df[first_cat].unique())
            })
        
        result = pd.DataFrame(result_data)
        viz_type = 'bar'
        return result, viz_type, description

def find_column(columns_lower, columns_dict, keywords):
    """
    Find a column that matches any of the keywords.
    
    Args:
        columns_lower: List of column names in lowercase
        columns_dict: Dictionary mapping lowercase column names to original column names
        keywords: List of keywords to search for
        
    Returns:
        str: Original column name if found, None otherwise
    """
    # Common abbreviations and alternative terms
    abbrev_map = {
        'gender': ['gen', 'sex', 'gndr', 'm/f', 'male/female'],
        'age': ['yrs', 'years', 'yr', 'años', 'âge', 'alter'],
        'location': ['loc', 'place', 'addr', 'city', 'town', 'village', 'district', 'geo'],
        'date': ['dt', 'time', 'when', 'fecha', 'datum', 'day'],
        'program': ['prog', 'prg', 'project', 'proj', 'initiative', 'course', 'training'],
        'status': ['stat', 'condition', 'state', 'sts', 'situation', 'category'],
        'region': ['area', 'zone', 'sector', 'territory', 'locality'],
        'income': ['earnings', 'salary', 'wage', 'revenue', 'money'],
        'education': ['edu', 'school', 'academic', 'qualification', 'study', 'grad'],
        'occupation': ['job', 'work', 'profession', 'career', 'employment'],
        'household': ['hh', 'family', 'home', 'house', 'residence']
    }
    
    # Expand keywords with their abbreviations
    expanded_keywords = list(keywords)  # Start with original keywords
    for keyword in keywords:
        if keyword in abbrev_map:
            expanded_keywords.extend(abbrev_map[keyword])
    
    # Scoring system for matches (higher is better)
    scored_matches = []
    
    for col in columns_lower:
        score = 0
        matched_term = None
        
        # Exact match - highest priority
        for kw in expanded_keywords:
            if col == kw:
                score = 100
                matched_term = kw
                break
        
        # Column starts with keyword
        if score == 0:
            for kw in expanded_keywords:
                if col.startswith(kw):
                    score = 80
                    matched_term = kw
                    break
        
        # Keyword is contained in column name
        if score == 0:
            for kw in expanded_keywords:
                if kw in col:
                    score = 60
                    matched_term = kw
                    break
        
        # Parts of column contain keyword (like 'gndr_cd' for 'gender')
        if score == 0:
            col_parts = col.split('_')
            for part in col_parts:
                for kw in expanded_keywords:
                    # Check for partial match at beginning of part
                    if len(kw) >= 3 and part.startswith(kw[:3]):
                        score = 40
                        matched_term = kw
                        break
                if score > 0:
                    break
        
        # Column contains first few letters of a longer keyword
        if score == 0:
            for kw in expanded_keywords:
                if len(kw) >= 4 and kw[:3] in col:
                    score = 30
                    matched_term = kw
                    break
        
        if score > 0:
            scored_matches.append((col, score, matched_term))
    
    # Sort by score descending
    scored_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Return the highest scoring match if any
    if scored_matches:
        best_match = scored_matches[0][0]
        return columns_dict[best_match]
    
    return None
