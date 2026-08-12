import sqlite3
import pandas as pd
import os

def export_datasets():
    print("Connecting to Data Warehouse...")
    db_path = '../data/retail_dw.db'
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    output_dir = '../../bi_exports'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Exporting Sales Summary Data Mart...")
    df_sales = pd.read_sql_query("SELECT * FROM v_sales_summary", conn)
    df_sales.to_csv(os.path.join(output_dir, 'sales_summary.csv'), index=False)
    
    print("Exporting Cohort Analysis Data Mart...")
    df_cohort = pd.read_sql_query("SELECT * FROM v_cohort_base", conn)
    df_cohort.to_csv(os.path.join(output_dir, 'cohort_analysis.csv'), index=False)
    
    print("Exporting Customer Clusters...")
    try:
        df_clusters = pd.read_sql_query("SELECT * FROM customer_clusters", conn)
        df_clusters.to_csv(os.path.join(output_dir, 'customer_clusters.csv'), index=False)
    except Exception as e:
        print("Could not export customer_clusters (did you run multivariate_analysis.py?):", e)
    
    conn.close()
    print(f"Export complete. BI Datasets saved to {output_dir}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    export_datasets()
