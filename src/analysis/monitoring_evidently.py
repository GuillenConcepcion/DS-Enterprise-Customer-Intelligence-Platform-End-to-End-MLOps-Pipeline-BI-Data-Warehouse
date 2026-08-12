import sqlite3
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

def run_monitoring():
    print("Connecting to Data Warehouse...")
    db_path = '../data/retail_dw.db'
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return
        
    conn = sqlite3.connect(db_path)
    
    # Load RFM Data
    print("Loading RFM data for Drift Analysis...")
    df_rfm = pd.read_sql_query("SELECT * FROM v_customer_rfm_base", conn)
    
    # Simulate Reference vs Current by splitting dataset
    # In a real scenario, this would be historical vs recent data
    reference = df_rfm.sample(frac=0.5, random_state=42).copy()
    current = df_rfm.drop(reference.index).copy()
    
    # Injecting artificial drift into Current dataset to demonstrate Evidently's capabilities
    # E.g. Simulating an aggressive marketing campaign that increased frequency and monetary value
    current['MonetaryValue'] = current['MonetaryValue'] * 1.25
    current['Frequency'] = current['Frequency'] + 2
    
    print("Generating Evidently Data Drift Report...")
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    data_drift_report.run(reference_data=reference, current_data=current)
    
    # Save Report
    output_dir = '../../bi_exports'
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'data_drift_report.html')
    data_drift_report.save_html(report_path)
    
    print(f"Monitoring complete! Data Drift Report saved to: {report_path}")
    conn.close()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_monitoring()
