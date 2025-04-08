"""
Personalization scoring for policy recommendations.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Get logger
logger = logging.getLogger(__name__)


def calculate_recency_score(days_since_update, max_days=30):
    """
    Calculate recency score with exponential decay.
    
    Parameters:
    -----------
    days_since_update : int or float
        Days since the policy was last updated
    max_days : int, default=30
        Maximum days to consider for recency scoring
        
    Returns:
    --------
    float
        Recency score between 0 and 1
    """
    if pd.isna(days_since_update):
        return 0.0
    
    # Cap days at max_days
    days = min(float(days_since_update), float(max_days))
    
    # Exponential decay function (decays faster at the beginning)
    # Score = e^(-λ * days/max_days)
    decay_rate = 3.0  # Adjust for desired decay speed
    score = np.exp(-decay_rate * days / max_days)
    
    return score


def get_fallback_recommendations(conn, customer, top_n=3):
    """
    Get recommendations when no direct matches are found.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    customer : pandas.Series or dict
        Customer information including sector
    top_n : int, default=3
        Number of top recommendations to return
        
    Returns:
    --------
    pandas.DataFrame
        Top N fallback policy recommendations
    """
    from src.db import pd  # Avoid circular import
    
    # Extract customer sector
    customer_sector = customer['sector']
    
    logger.info(f"Looking for fallback recommendations for customer {customer['name']} with sector: {customer_sector}")
    
    # Find policies that match by sector instead of geography
    query = """
    SELECT 
        p.id, 
        p.name, 
        p.geography, 
        p.sectors,
        p.topics,
        p.status,
        p.description,
        p.updated_date
    FROM 
        policies p
    WHERE 
        p.status = 'active'
    ORDER BY 
        p.updated_date DESC
    LIMIT 50
    """
    
    recent_policies = pd.read_sql_query(query, conn)
    
    if recent_policies.empty:
        logger.warning("No active policies found for fallback recommendations")
        return pd.DataFrame()
    
    # Ensure updated_date is in datetime format
    recent_policies['updated_date'] = pd.to_datetime(recent_policies['updated_date'], errors='coerce', utc=True)
    
    # Calculate days since update for recency scoring
    now = pd.Timestamp.now(tz='UTC')
    recent_policies['days_since_update'] = recent_policies['updated_date'].apply(
        lambda x: (now - x).days if pd.notna(x) else 30
    )
    
    # Calculate recency score using our exponential decay function
    recent_policies['recency_score'] = recent_policies['days_since_update'].apply(
        lambda days: calculate_recency_score(days)
    )
    
    # Simple sector matching (basic version)
    def basic_sector_match(policy_sectors, customer_sector):
        if pd.isna(policy_sectors) or pd.isna(customer_sector):
            return 0.0
            
        if isinstance(policy_sectors, str) and isinstance(customer_sector, str):
            policy_sectors_list = [s.strip().lower() for s in policy_sectors.split(',')]
            customer_sectors_list = [s.strip().lower() for s in customer_sector.split(',')]
            
            # Check for any overlap between policy sectors and customer sectors
            for cs in customer_sectors_list:
                for ps in policy_sectors_list:
                    if cs in ps or ps in cs:
                        return 1.0
        
        return 0.0
    
    # Calculate sector score
    recent_policies['sector_score'] = recent_policies['sectors'].apply(
        lambda x: basic_sector_match(x, customer_sector)
    )
    
    # Calculate combined relevance score (20% recency, 80% sector match)
    recent_policies['relevance_score'] = 0.2 * recent_policies['recency_score'] + 0.8 * recent_policies['sector_score']
    
    # Return top matches
    result = recent_policies.sort_values('relevance_score', ascending=False).head(top_n)
    logger.info(f"Found {len(result)} fallback recommendations")
    return result


def calculate_relevance_score(policies_df, customer, max_days=30):
    """
    Calculate a relevance score for each policy with improved recency scoring.
    
    Parameters:
    -----------
    policies_df : pandas.DataFrame
        Dataframe containing relevant policies
    customer : pandas.Series or dict
        Customer information including sectors
    max_days : int, default=30
        Maximum days to consider for recency scoring
        
    Returns:
    --------
    pandas.DataFrame
        Policies with added relevance scores
    """
    if policies_df.empty:
        return policies_df
    
    # Create a copy to avoid modifying the original dataframe
    scored_policies = policies_df.copy()
    
    # Ensure updated_date is in datetime format with explicit conversion
    scored_policies['updated_date'] = pd.to_datetime(scored_policies['updated_date'], errors='coerce', utc=True)
    
    # Calculate days since update
    now = pd.Timestamp.now(tz='UTC')
    scored_policies['days_since_update'] = scored_policies['updated_date'].apply(
        lambda x: (now - x).days if pd.notna(x) else max_days
    )
    
    # 1. Calculate recency score
    scored_policies['recency_score'] = scored_policies['days_since_update'].apply(
        lambda days: calculate_recency_score(days, max_days)
    )
    
    # 2. Geography match (exact match = 1.0)
    # This is already filtered in the SQL query, so all policies have a match - could go more complex with some new data labels
    scored_policies['geography_score'] = 1.0
    
    # 3. Simple sector score (0-1): checking if customer sector appears in policy sectors.( Sector match is binary as well for now )
    customer_sectors = customer['sector']
    
    def basic_sector_match(policy_sectors):
        if pd.isna(policy_sectors) or pd.isna(customer_sectors):
            return 0.0
            
        if isinstance(policy_sectors, str) and isinstance(customer_sectors, str):
            policy_sectors_list = [s.strip().lower() for s in policy_sectors.split(',')]
            customer_sectors_list = [s.strip().lower() for s in customer_sectors.split(',')]
            
            # Check for any overlap
            for cs in customer_sectors_list:
                for ps in policy_sectors_list:
                    if cs in ps or ps in cs:
                        return 1.0
        
        return 0.0
    
    scored_policies['sector_score'] = scored_policies['sectors'].apply(basic_sector_match)
    
    # Calculate final weighted score
    weights = {
        'recency': 0.2,
        'geography': 0,
        'sector': 0.8
    }
    
    scored_policies['relevance_score'] = (
        weights['recency'] * scored_policies['recency_score'] +
        weights['geography'] * scored_policies['geography_score'] +
        weights['sector'] * scored_policies['sector_score']
    )
    
    # Sort by relevance score descending
    return scored_policies.sort_values('relevance_score', ascending=False)


def get_top_recommendations(conn, customer, top_n=3, days_window=7):
    """
    Get top policy recommendations for a customer with improved scoring and fallback.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    customer : pandas.Series or dict
        Customer information including operating_jurisdictions
    top_n : int, default=3
        Number of top recommendations to return
    days_window : int, default=7
        Time window for policy updates in days
        
    Returns:
    --------
    tuple
        (policies_df, is_fallback): Policies DataFrame and boolean indicating if fallback was used
    """
    from src.db import get_relevant_policies  # Avoid circular import
    
    # Get customer jurisdictions (split if comma-separated)
    jurisdictions = customer['operating_jurisdiction'].split(',') if isinstance(customer['operating_jurisdiction'], str) else [customer['operating_jurisdiction']]
    jurisdictions = [j.strip() for j in jurisdictions]
    
    logger.info(f"Getting recommendations for customer: {customer['name']} with jurisdictions: {jurisdictions}")
    
    # Get relevant policies for each jurisdiction
    all_relevant_policies = pd.DataFrame()
    for jurisdiction in jurisdictions:
        relevant_policies = get_relevant_policies(conn, jurisdiction, days_window)
        all_relevant_policies = pd.concat([all_relevant_policies, relevant_policies])
    
    # Remove duplicates if a policy appears in multiple jurisdictions
    all_relevant_policies = all_relevant_policies.drop_duplicates(subset=['id'])
    
    # If no policies found with the time window, try fallback recommendations
    if all_relevant_policies.empty:
        logger.warning(f"No direct matches found for {customer['name']}'s jurisdictions in the past {days_window} days.")
        logger.info("Looking for alternative recommendations...")
        fallback_recommendations = get_fallback_recommendations(conn, customer, top_n)
        
        # Add explanation that these are fallback recommendations
        if not fallback_recommendations.empty:
            logger.info("Using fallback recommendations based on sector relevance and recency")
            
            # Add a column to indicate these are fallback recommendations
            fallback_recommendations['is_direct_match'] = False
            fallback_recommendations['fallback_reason'] = f"No policies found in jurisdiction(s) {', '.join(jurisdictions)} updated in the last {days_window} days"
            
            return fallback_recommendations
        else:
            logger.warning("No alternative recommendations found.")
            return pd.DataFrame()
    
    # Score and rank policies with our improved algorithm
    scored_policies = calculate_relevance_score(all_relevant_policies, customer, max_days=days_window)
    
    # Add a column to indicate these are direct matches
    scored_policies['is_direct_match'] = True
    
    # Return top N recommendations
    result = scored_policies.head(top_n)
    logger.info(f"Returning {len(result)} top recommendations")
    return result