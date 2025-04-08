"""
Main script for the policy recommendation system.
"""

import os
import argparse
import sys
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

from src.data_ingestion import load_datasets, clean_policies_data, clean_companies_data, save_cleaned_data
from src.db import create_database_connection, load_data_to_db, get_all_customers
from src.scoring import get_top_recommendations
from src.utils import ensure_directory_exists, format_recommendation_output


def setup_logging(log_level=logging.INFO):
    """Configure logging settings."""
    log_dir = ensure_directory_exists('logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/policy_recommendation_{timestamp}.log'
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Reduce verbosity of third-party libraries
    logging.getLogger('pandas').setLevel(logging.WARNING)
    
    logging.info(f"Logging configured. Log file: {log_file}")
    return log_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Policy Recommendation System')
    
    parser.add_argument('--policies', type=str, default='data/raw/random_policies_1743755330437.csv',
                        help='Path to the policies CSV file')
    parser.add_argument('--companies', type=str, default='data/raw/company_data_single_jurisdiction_1743592075371.csv',
                        help='Path to the companies CSV file')
    parser.add_argument('--db', type=str, default='policy_recommendation.db',
                        help='Path to the SQLite database file')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to look back for recent updates')
    parser.add_argument('--top-n', type=int, default=3,
                        help='Number of top recommendations to return')
    parser.add_argument('--customer-id', type=int,
                        help='Specific customer ID to generate recommendations for')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='Logging level')
    
    return parser.parse_args()


def main():
    """Main function for the policy recommendation system."""
    args = parse_arguments()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    logging.info("\n=== Policy Recommendation System ===")
    logging.info(f"Running at: {datetime.now()}")
    logging.info(f"Looking for policies updated in the last {args.days} days")
    logging.info(f"Returning top {args.top_n} recommendations per customer")
    logging.info("=" * 40)
    
    conn = None
    
    try:
        # Step 1: Data Ingestion & Cleaning
        logging.info("\n=== Step 1: Data Ingestion & Cleaning ===")
        policies_df, companies_df = load_datasets(args.policies, args.companies)
        
        # Log available geographies for debugging
        if 'geography' in policies_df.columns and 'operating_jurisdiction' in companies_df.columns:
            policy_geos = sorted(policies_df['geography'].dropna().unique())
            company_geos = sorted(companies_df['operating_jurisdiction'].dropna().unique())
            logging.info(f"Policy geographies before cleaning: {policy_geos}")
            logging.info(f"Company jurisdictions before cleaning: {company_geos}")
        
        cleaned_policies = clean_policies_data(policies_df)
        cleaned_companies = clean_companies_data(companies_df)
        
        # Log cleaned geographies for debugging
        if 'geography' in cleaned_policies.columns and 'operating_jurisdiction' in cleaned_companies.columns:
            policy_geos = sorted(cleaned_policies['geography'].dropna().unique())
            company_geos = sorted(cleaned_companies['operating_jurisdiction'].dropna().unique())
            logging.info(f"Policy geographies after cleaning: {policy_geos}")
            logging.info(f"Company jurisdictions after cleaning: {company_geos}")
        
        processed_dir = ensure_directory_exists('data/processed')
        save_cleaned_data(cleaned_policies, cleaned_companies, processed_dir)
        
        # Step 2: Data Integration with SQL
        logging.info("\n=== Step 2: Data Integration with SQL ===")
        conn = create_database_connection(args.db)
        load_data_to_db(conn, cleaned_policies, cleaned_companies)
        
        # Step 3: Personalization Scoring
        logging.info("\n=== Step 3: Personalization Scoring ===")
        
        # Get customer data
        if args.customer_id:
            customers = pd.read_sql_query(f"SELECT * FROM companies WHERE company_id = {args.customer_id}", conn)
            if customers.empty:
                logging.error(f"Error: Customer with ID {args.customer_id} not found")
                return
        else:
            customers = get_all_customers(conn)
        
        # Generate recommendations for each customer
        all_outputs = []
        for _, customer in customers.iterrows():
            logging.info(f"Processing recommendations for: {customer['name']} (ID: {customer['company_id']})")
            recommendations = get_top_recommendations(
                conn, customer, top_n=args.top_n, days_window=args.days
            )
            
            output = format_recommendation_output(recommendations, customer)
            print(output)
            all_outputs.append(output)
        
        # Save outputs to a file
        output_dir = ensure_directory_exists('output')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'recommendations_{timestamp}.txt'
        
        with open(output_file, 'w') as f:
            f.write('\n\n'.join(all_outputs))
        
        logging.info(f"\nAll recommendations saved to: {output_file}")
        logging.info("\nDone!")
    
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed")


if __name__ == '__main__':
    main()