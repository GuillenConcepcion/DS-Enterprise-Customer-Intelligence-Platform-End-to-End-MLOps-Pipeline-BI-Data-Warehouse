import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.xgboost
import os
import shap
import matplotlib.pyplot as plt

def run_churn_prediction():
    print("Connecting to Data Warehouse...")
    db_path = '../data/retail_dw.db'
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return
        
    conn = sqlite3.connect(db_path)
    
    print("Extracting Temporal Data for Feature Engineering...")
    # Load all sales
    query = "SELECT CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice FROM fact_sales WHERE Quantity > 0 AND CustomerID IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Define split date (e.g., 2011-09-01) for out-of-time validation
    split_date = pd.to_datetime('2011-09-01')
    
    # Feature window (Before Sept 2011)
    df_feat = df[df['InvoiceDate'] < split_date]
    
    # Target window (Sept 2011 onwards)
    df_targ = df[df['InvoiceDate'] >= split_date]
    
    # Feature Engineering (RFM)
    features = df_feat.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (split_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique')
    )
    
    # Calculate MonetaryValue separately for clarity
    df_feat = df_feat.copy()
    df_feat['TotalAmount'] = df_feat['Quantity'] * df_feat['UnitPrice']
    monetary = df_feat.groupby('CustomerID')['TotalAmount'].sum()
    features = features.join(monetary).rename(columns={'TotalAmount': 'MonetaryValue'}).reset_index()
    
    # Target Engineering (1 = Churned, i.e., didn't buy in target window)
    active_customers = df_targ['CustomerID'].unique()
    features['is_churned'] = np.where(features['CustomerID'].isin(active_customers), 0, 1)
    
    print(f"Dataset generated: {features.shape[0]} customers. Churn rate: {features['is_churned'].mean():.2%}")
    
    # ML Pipeline
    X = features[['Recency', 'Frequency', 'MonetaryValue']]
    y = features['is_churned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Orchestrating Hyperparameter Tuning via GridSearchCV...")
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [50, 100]
    }
    
    # We will optimize for ROC-AUC
    xgb_base = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=42)
    grid_search = GridSearchCV(estimator=xgb_base, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    clf = grid_search.best_estimator_
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    
    print("Training best XGBoost Classifier with MLflow tracking...")
    mlflow.set_experiment("Churn_Prediction_XGBoost")
    
    with mlflow.start_run(run_name="XGB_Tuned_Model"):
        # Log grid search best params
        mlflow.log_params(grid_search.best_params_)
        
        # Evaluation
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        print("\n--- CLASSIFICATION REPORT ---")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {auc:.4f}")
        
        # Log and Register model in Registry
        mlflow.xgboost.log_model(
            clf, 
            artifact_path="xgboost_churn_model",
            registered_model_name="ChurnPredictionXGBoost"
        )
        print("Model logged and registered in MLflow Model Registry successfully!")

        # 1. Confusion Matrix Plot
        print("Generating Confusion Matrix...")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
        disp.plot(cmap='Blues', ax=ax_cm, values_format='d')
        ax_cm.set_title('Confusion Matrix')
        plt.tight_layout()
        mlflow.log_figure(fig_cm, "plots/confusion_matrix.png")
        os.makedirs("../../bi_exports", exist_ok=True)
        fig_cm.savefig("../../bi_exports/confusion_matrix.png")
        plt.close(fig_cm)

        # 2. SHAP Summary Plot (Explainability)
        print("Generating SHAP Explainer...")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        
        fig_shap, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("SHAP Summary Plot - Churn Drivers")
        plt.tight_layout()
        mlflow.log_figure(fig_shap, "plots/shap_summary.png")
        fig_shap.savefig("../../bi_exports/shap_summary.png") # Also save locally for easy viewing
        plt.close(fig_shap)

        # 3. Cumulative Gains / Lift Curve
        print("Generating Cumulative Gains (Lift) Curve...")
        df_lift = pd.DataFrame({'y_true': y_test, 'y_prob': y_proba})
        df_lift = df_lift.sort_values(by='y_prob', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative capture
        df_lift['cumulative_true'] = df_lift['y_true'].cumsum()
        total_true = df_lift['y_true'].sum()
        df_lift['cumulative_gains'] = df_lift['cumulative_true'] / total_true
        
        # Percentage of sample targeted
        df_lift['percentage_targeted'] = (df_lift.index + 1) / len(df_lift)
        
        fig_lift, ax_lift = plt.subplots(figsize=(8, 6))
        ax_lift.plot(df_lift['percentage_targeted'], df_lift['cumulative_gains'], label='XGBoost Gains', color='darkblue', linewidth=2)
        ax_lift.plot([0, 1], [0, 1], label='Random Target', color='red', linestyle='--')
        ax_lift.set_xlabel('Percentage of Customers Contacted')
        ax_lift.set_ylabel('Percentage of Churners Captured')
        ax_lift.set_title('Cumulative Gains Curve (ROI Translation)')
        ax_lift.legend(loc='lower right')
        ax_lift.grid(True, alpha=0.3)
        plt.tight_layout()
        
        mlflow.log_figure(fig_lift, "plots/cumulative_gains.png")
        fig_lift.savefig("../../bi_exports/cumulative_gains.png") # Also save locally
        plt.close(fig_lift)
        print("XAI and Lift plots logged to MLflow and saved to bi_exports!")

    conn.close()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_churn_prediction()
