"""
Database operations for the policy recommendation.
"""

import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta

# Get logger
logger = logging.getLogger(__name__)

def create_database_connection(db_path):
    """
    Create a connection to an SQLite database.
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database file
        
    Returns:
    --------
    sqlite3.Connection
        Database connection object
    """
    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise


def load_data_to_db(conn, policies_df, companies_df):
    """
    Load cleaned data into the database.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    policies_df : pandas.DataFrame
        Cleaned policies DataFrame
    companies_df : pandas.DataFrame
        Cleaned companies DataFrame
    """
    try:
        # Load policies table
        policies_df.to_sql('policies', conn, if_exists='replace', index=False)
        logger.info(f"Loaded {len(policies_df)} policies to database")
        
        # Load companies table
        companies_df.to_sql('companies', conn, if_exists='replace', index=False)
        logger.info(f"Loaded {len(companies_df)} companies to database")
        
    except Exception as e:
        logger.error(f"Database load error: {e}")
        raise


def get_relevant_policies(conn, customer_jurisdiction, days_window=7):
    """
    Retrieve policies that match the requirements for a given customer jurisdiction.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    customer_jurisdiction : str
        The jurisdiction to match with policy geography
    days_window : int, default=7
        Number of days to look back for recent updates
        
    Returns:
    --------
    pandas.DataFrame
        Relevant policies matching the criteria
    """
    # Calculate the date X days ago
    x_days_ago = (datetime.now() - timedelta(days=days_window)).strftime('%Y-%m-%d')
    
    # Calculate the date 1 year ago for sector average calculation
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    logger.info(f"Querying relevant policies for jurisdiction '{customer_jurisdiction}' updated in the last {days_window} days")
    
    # Query with improved sector matching
    query = """
    SELECT 
        p.id, 
        p.name, 
        p.geography, 
        p.topics,
        p.updated_date,
        p.sectors,
        p.status,
        p.description,
        (SELECT AVG(JULIANDAY('now') - JULIANDAY(p2.updated_date)) 
         FROM policies p2 
         WHERE p2.sectors LIKE '%' || p.sectors || '%' 
             OR p.sectors LIKE '%' || p2.sectors || '%'
         AND p2.updated_date >= ?) AS avg_days_since_update_in_sector
    FROM 
        policies p
    WHERE 
        p.status = 'active'
        AND p.geography = ?
        AND p.updated_date >= ?
    ORDER BY 
        p.updated_date DESC
    """
    
    try:
        result = pd.read_sql_query(query, conn, params=[one_year_ago, customer_jurisdiction, x_days_ago])
        logger.info(f"Found {len(result)} relevant policies")
        return result
    except Exception as e:
        logger.error(f"Error querying policies: {e}")
        return pd.DataFrame()


def get_all_customers(conn):
    """
    Retrieve all customers from the database.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
        
    Returns:
    --------
    pandas.DataFrame
        All customers
    """
    try:
        query = "SELECT * FROM companies"
        result = pd.read_sql_query(query, conn)
        logger.info(f"Retrieved {len(result)} customers from database")
        return result
    except Exception as e:
        logger.error(f"Error retrieving customers: {e}")
        return pd.DataFrame()