from ucimlrepo import fetch_ucirepo
import pandas as pd
import sqlite3
import os

def ingest_dataset():
    print("Fetching Online Retail dataset from UCI...")
    # fetch dataset (ID 352 is Online Retail)
    online_retail = fetch_ucirepo(id=352) 
      
    # Extract data as pandas dataframe
    features = online_retail.data.features.copy()
    ids = online_retail.data.ids.copy()
    
    # Combine IDs and Features
    df = pd.concat([ids, features], axis=1)
    
    print(f"Dataset downloaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    print("Cleaning data...")
    # Basic cleaning
    # Drop rows without CustomerID
    df = df.dropna(subset=['CustomerID'])
    
    # Convert InvoiceDate to datetime string for SQLite
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert CustomerID to integer
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Database connection
    db_path = 'retail_dw.db'
    print(f"Saving to SQLite database: {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Ingest raw data
    df.to_sql('raw_transactions', conn, if_exists='replace', index=False)
    
    # Now let's execute the DDL to create the Star Schema
    print("Executing DDL to create Star Schema...")
    with open('../../sql/01_ddl_tables.sql', 'r') as f:
        ddl_script = f.read()
    conn.executescript(ddl_script)
    
    # Populate the Star Schema from raw_transactions
    print("Populating Star Schema...")
    populate_script = """
    BEGIN TRANSACTION;
    
    -- Populate Customers
    INSERT OR IGNORE INTO dim_customers (CustomerID, Country)
    SELECT DISTINCT CustomerID, Country
    FROM raw_transactions
    WHERE CustomerID IS NOT NULL;
    
    -- Populate Products
    INSERT OR IGNORE INTO dim_products (StockCode, Description)
    SELECT DISTINCT StockCode, Description
    FROM raw_transactions
    WHERE StockCode IS NOT NULL;
    
    -- Populate Fact Sales
    INSERT INTO fact_sales (InvoiceNo, StockCode, CustomerID, Quantity, InvoiceDate, UnitPrice)
    SELECT InvoiceNo, StockCode, CustomerID, Quantity, InvoiceDate, UnitPrice
    FROM raw_transactions;
    
    COMMIT;
    """
    try:
        conn.executescript(populate_script)
        print("Star Schema populated successfully.")
    except Exception as e:
        print(f"Error populating Star Schema. Rolling back. Details: {e}")
        conn.rollback()
        raise
    
    # Create BI Views
    print("Creating BI Views...")
    with open('../../sql/02_bi_views.sql', 'r') as f:
        views_script = f.read()
    conn.executescript(views_script)
    
    conn.close()
    print("Ingestion and modeling complete!")

if __name__ == '__main__':
    # Ensure working directory is the script's directory for relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ingest_dataset()
