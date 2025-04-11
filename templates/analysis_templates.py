import pandas as pd
import numpy as np

def get_templates():
    """
    Returns a dictionary of available analysis templates.
    
    Returns:
        dict: Dictionary of template names and descriptions
    """
    return {
        "demographic_analysis": "Basic demographic breakdown (age, gender, location)",
        "program_participation": "Analysis of program participation rates",
        "beneficiary_status": "Summary of beneficiary status categories",
        "temporal_trends": "Analysis of changes over time",
        "service_utilization": "Analysis of service usage patterns",
        "funding_allocation": "Analysis of funding distribution",
        "outcome_analysis": "Analysis of program outcomes and effectiveness",
        "geographic_distribution": "Geographic distribution of beneficiaries",
        "vulnerability_assessment": "Assessment of vulnerability factors",
        "comparative_analysis": "Comparison between different groups or time periods"
    }

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
