"""
Utility functions for the policy recommendation.
"""

import os
import pandas as pd
import logging
from pathlib import Path

# Get logger
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory):
    """
    Ensure that a directory exists, create if it doesn't.
    
    Parameters:
    -----------
    directory : str or Path
        Directory path to check/create
        
    Returns:
    --------
    Path
        Path object of the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def format_recommendation_output(recommendations, customer):
    """
    Format the recommendation output for display.
    
    Parameters:
    -----------
    recommendations : pandas.DataFrame
        Recommended policies
    customer : pandas.Series or dict
        Customer information
        
    Returns:
    --------
    str
        Formatted output string
    """
    output = []
    output.append(f"=== Recommendations for {customer['name']} (ID: {customer['company_id']}) ===")
    output.append(f"Sectors: {customer['sector']}")
    output.append(f"Operating jurisdictions: {customer['operating_jurisdiction']}")
    output.append("")
    
    if recommendations.empty:
        output.append("No relevant policies found for this customer.")
    else:
        # Check if these are fallback recommendations
        if 'is_direct_match' in recommendations.columns and not recommendations['is_direct_match'].iloc[0]:
            if 'fallback_reason' in recommendations.columns:
                output.append(f"NOTE: {recommendations['fallback_reason'].iloc[0]}")
            else:
                output.append("NOTE: No direct matches found in your jurisdiction. Using alternative recommendations.")
            output.append("The following recommendations are based on sector relevance and recency.")
            output.append("")
        
        output.append("Top recommended policies:")
        for i, (_, rec) in enumerate(recommendations.iterrows()):
            output.append(f"{i+1}. {rec['name']} ({rec['geography']})")
            output.append(f"   ID: {rec['id']}")
            output.append(f"   Updated: {rec['updated_date']}")
            output.append(f"   Relevance score: {rec['relevance_score']:.2f}")
            output.append(f"   Sector match score: {rec.get('sector_score', 'N/A')}")
            output.append(f"   Recency score: {rec.get('recency_score', 'N/A')}")
            
            # Add jurisdiction match information
            if 'is_direct_match' in rec:
                if rec['is_direct_match']:
                    output.append(f"   Jurisdiction: Direct match")
                else:
                    output.append(f"   Jurisdiction: No direct match")
            
            # Add a short excerpt from the description if available
            if 'description' in rec and pd.notna(rec['description']):
                excerpt = rec['description'][:100] + '...' if len(rec['description']) > 100 else rec['description']
                output.append(f"   Description: {excerpt}")
            
            output.append("")
    
    output.append("=" * 80)
    return "\n".join(output)