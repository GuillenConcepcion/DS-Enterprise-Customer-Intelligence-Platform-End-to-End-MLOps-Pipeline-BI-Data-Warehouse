import sqlite3
import pandas as pd
import great_expectations as gx
import os

def run_data_quality():
    print("Connecting to Data Warehouse...")
    db_path = 'retail_dw.db'
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    
    print("Loading RFM Data Mart...")
    df_rfm = pd.read_sql_query("SELECT * FROM v_customer_rfm_base", conn)
    conn.close()

    print("Initializing Great Expectations Ephemeral Data Context...")
    context = gx.get_context()

    # Define Data Source, Data Asset and Batch Definition using GX 1.x
    datasource_name = "rfm_data_source"
    datasource = context.data_sources.add_pandas(name=datasource_name)
    
    asset_name = "rfm_dataframe"
    data_asset = datasource.add_dataframe_asset(name=asset_name)
    
    batch_definition = data_asset.add_batch_definition_whole_dataframe("rfm_batch_definition")

    # Define Expectation Suite
    suite_name = "rfm_expectations"
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    # Add expectations
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="CustomerID"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="CustomerID"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="Frequency", min_value=1, max_value=None))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="MonetaryValue", min_value=0.0, max_value=None))

    # Define Validation Definition and Run
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="rfm_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    print("Running Data Quality validation via Great Expectations...")
    validation_result = validation_definition.run(batch_parameters={"dataframe": df_rfm})

    print("\n--- GREAT EXPECTATIONS DATA QUALITY REPORT ---")
    print(f"Overall Success: {validation_result.success}")
    print("-----------------------------------------------")
    
    all_passed = validation_result.success
    
    for idx, result in enumerate(validation_result.results):
        expect_type = result.expectation_config.type
        col = result.expectation_config.kwargs.get("column", "")
        success = result.success
        status = "[PASSED]" if success else "[FAILED]"
        print(f"Expectation {idx+1}: {expect_type} (col: {col}) -> {status}")
        if not success:
            print(f"  Details: {result.result}")
            
    print("-----------------------------------------------")
    if all_passed:
        print("SUCCESS: Data Contract validated successfully! Safe for ML training.")
    else:
        raise ValueError("CRITICAL WARNING: Data Contract violations detected! Stop training.")

if __name__ == '__main__':
    # Ensure working directory is the script's directory for relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_data_quality()

