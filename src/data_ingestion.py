"""
Data ingestion and cleaning module for the policy recommendation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging

# Get logger
logger = logging.getLogger(__name__)

def load_datasets(policies_path, companies_path):
    """
    Load the policies and companies datasets from CSV files.
    
    Parameters:
    -----------
    policies_path : str or Path
        Path to the policies CSV file
    companies_path : str or Path
        Path to the companies CSV file
        
    Returns:
    --------
    tuple
        (policies_df, companies_df): Pandas DataFrames containing the raw data
    """
    try:
        policies_df = pd.read_csv(policies_path)
        logger.info(f"Loaded {len(policies_df)} policies from {policies_path}")
    except Exception as e:
        logger.error(f"Failed to load policies: {e}")
        raise
        
    try:
        companies_df = pd.read_csv(companies_path)
        logger.info(f"Loaded {len(companies_df)} companies from {companies_path}")
    except Exception as e:
        logger.error(f"Failed to load companies: {e}")
        raise
    
    return policies_df, companies_df


def clean_policies_data(policies_df):
    """
    Clean the policies dataset.
    
    Parameters:
    -----------
    policies_df : pandas.DataFrame
        Raw policies DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned policies DataFrame
    """
    logger.info("Starting policy data cleaning")
    
    # Create a copy to avoid modifying the original
    df = policies_df.copy()
    
    # Convert date columns to datetime with UTC timezone
    for date_field in ['published_date', 'updated_date']:
        if date_field in df.columns:
            original_nulls = df[date_field].isna().sum()
            df[date_field] = pd.to_datetime(df[date_field], errors='coerce', utc=True)
            new_nulls = df[date_field].isna().sum()
            if new_nulls > original_nulls:
                logger.warning(f"Date conversion created {new_nulls - original_nulls} new null values in {date_field}")
    
    # Standardize status field (lowercase)
    if 'status' in df.columns:
        df['status'] = df['status'].str.lower()
        # Log unique status values
        unique_statuses = df['status'].dropna().unique()
        logger.info(f"Unique status values: {unique_statuses}")
    
    # Clean HTML content from description using simple regex
    if 'description' in df.columns:
        logger.info("Cleaning HTML content from descriptions")
        df['description'] = df['description'].apply(
            lambda x: re.sub(r'<.*?>', '', str(x)) if isinstance(x, str) else x
        )
    
    # Convert geography to lowercase for case-insensitive matching
    if 'geography' in df.columns:
        logger.info("Converting geography to lowercase for consistent matching")
        df['geography'] = df['geography'].str.lower()
    
    # Standardize sectors and topics (lowercase)
    for field in ['sectors', 'topics']:
        if field in df.columns:
            logger.info(f"Standardizing {field} field")
            df[field] = df[field].str.lower()
    
    # Check for duplicates in id
    duplicate_count = df.duplicated(subset=['id']).sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate policy IDs. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['id'], keep='first')
    
    # Check for missing values in critical fields
    critical_fields = ['id', 'name', 'geography', 'status', 'updated_date']
    for field in critical_fields:
        if field in df.columns and df[field].isna().sum() > 0:
            logger.warning(f"Found {df[field].isna().sum()} missing values in '{field}'")
    
    logger.info(f"Policies data cleaning complete. {len(df)} records after cleaning.")
    return df


def clean_companies_data(companies_df):
    """
    Clean the companies dataset.
    
    Parameters:
    -----------
    companies_df : pandas.DataFrame
        Raw companies DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned companies DataFrame
    """
    logger.info("Starting company data cleaning")
    
    # Create a copy to avoid modifying the original
    df = companies_df.copy()
    
    # Convert date columns to datetime with UTC timezone
    if 'last_login' in df.columns:
        df['last_login'] = pd.to_datetime(df['last_login'], errors='coerce', utc=True)
    
    # Check for duplicates in company_id
    duplicate_count = df.duplicated(subset=['company_id']).sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate company IDs. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['company_id'], keep='first')
    
    # Convert operating_jurisdiction to lowercase for case-insensitive matching
    if 'operating_jurisdiction' in df.columns:
        logger.info("Converting operating_jurisdiction to lowercase for consistent matching")
        df['operating_jurisdiction'] = df['operating_jurisdiction'].str.lower()
    
    # Standardize sector (lowercase)
    if 'sector' in df.columns:
        logger.info("Standardizing sector field")
        df['sector'] = df['sector'].str.lower()
    
    # Check for missing values in critical fields
    critical_fields = ['company_id', 'name', 'operating_jurisdiction', 'sector']
    for field in critical_fields:
        if field in df.columns and df[field].isna().sum() > 0:
            logger.warning(f"Found {df[field].isna().sum()} missing values in '{field}'")
    
    logger.info(f"Companies data cleaning complete. {len(df)} records after cleaning.")
    return df


def save_cleaned_data(policies_df, companies_df, output_dir):
    """
    Save cleaned datasets to CSV files.
    
    Parameters:
    -----------
    policies_df : pandas.DataFrame
        Cleaned policies DataFrame
    companies_df : pandas.DataFrame
        Cleaned companies DataFrame
    output_dir : str or Path
        Directory to save the cleaned files
        
    Returns:
    --------
    tuple
        (policies_path, companies_path): Paths to the saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    policies_path = output_dir / 'policies_cleaned.csv'
    companies_path = output_dir / 'companies_cleaned.csv'
    
    try:
        policies_df.to_csv(policies_path, index=False)
        logger.info(f"Saved cleaned policies to {policies_path}")
    except Exception as e:
        logger.error(f"Failed to save cleaned policies: {e}")
        raise
        
    try:
        companies_df.to_csv(companies_path, index=False)
        logger.info(f"Saved cleaned companies to {companies_path}")
    except Exception as e:
        logger.error(f"Failed to save cleaned companies: {e}")
        raise
    
    return policies_path, companies_path