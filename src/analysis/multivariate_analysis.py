import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_analysis():
    print("Connecting to database...")
    db_path = '../data/retail_dw.db'
    
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found. Run ingest script first.")
        return

    conn = sqlite3.connect(db_path)
    
    print("Loading RFM data...")
    df_rfm = pd.read_sql_query("SELECT * FROM v_customer_rfm_base", conn)
    
    # Calculate Recency in days
    max_date = pd.to_datetime(df_rfm['LastPurchaseDate']).max()
    df_rfm['Recency'] = (max_date - pd.to_datetime(df_rfm['LastPurchaseDate'])).dt.days
    
    # Features for clustering
    features = ['Recency', 'Frequency', 'MonetaryValue']
    X = df_rfm[features]
    
    # Handle skewness with Log Transformation
    X_log = np.log1p(X)
    
    # Run Elbow and Silhouette analysis
    print("Running Elbow and Silhouette analysis for K in [2, 6]...")
    k_range = range(2, 7)
    inertias = []
    sil_scores = []
    
    # Temp scaling for tuning
    scaler_temp = StandardScaler()
    X_scaled_temp = scaler_temp.fit_transform(X_log)
    
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_temp.fit_predict(X_scaled_temp)
        inertias.append(kmeans_temp.inertia_)
        sil_scores.append(silhouette_score(X_scaled_temp, labels))
        print(f"K = {k} | Inertia = {kmeans_temp.inertia_:.2f} | Silhouette Score = {sil_scores[-1]:.4f}")
        
    # Generate tuning plot
    fig_tuning, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(k_range, inertias, 'o-', color='darkblue', linewidth=2)
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(k_range, sil_scores, 'o-', color='darkgreen', linewidth=2)
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('K-Means Hyperparameter Tuning')
    plt.tight_layout()
    os.makedirs('../../bi_exports', exist_ok=True)
    tuning_plot_path = '../../bi_exports/kmeans_tuning.png'
    plt.savefig(tuning_plot_path)
    print(f"Tuning plot saved to {tuning_plot_path}")
    
    # Configure MLflow
    mlflow.set_experiment("RFM_Customer_Segmentation")
    
    print("Applying K-Means Clustering Pipeline with MLflow tracking...")
    with mlflow.start_run(run_name="KMeans_RFM_Base"):
        # Selected Parameters (K=4 justifies customer segmentations)
        n_clusters = 4
        n_init = 10
        random_state = 42
        
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("n_init", n_init)
        mlflow.log_param("random_state", random_state)
        mlflow.log_figure(fig_tuning, "plots/kmeans_tuning.png")
        plt.close(fig_tuning)
        
        # Build standard Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init))
        ])
        
        pipeline.fit(X_log)
        cluster_labels = pipeline.named_steps['kmeans'].labels_
        df_rfm['Cluster'] = cluster_labels
        
        # Metrics
        inertia = pipeline.named_steps['kmeans'].inertia_
        X_scaled = pipeline.named_steps['scaler'].transform(X_log)
        sil_score = silhouette_score(X_scaled, cluster_labels)
        
        mlflow.log_metric("inertia", inertia)
        mlflow.log_metric("silhouette_score", sil_score)
        
        print(f"Selected K={n_clusters} - Inertia: {inertia:.2f}, Silhouette Score: {sil_score:.4f}")
        
        # Log Pipeline Model to MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="kmeans_rfm_pipeline")
        print("KMeans Pipeline logged to MLflow successfully!")
        
        print("Applying PCA for 2D visualization...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df_rfm['PCA1'] = pca_result[:, 0]
        df_rfm['PCA2'] = pca_result[:, 1]
        
        # Generate PCA plot
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_rfm, palette='viridis', ax=ax_scatter)
        ax_scatter.set_title('Customer Segmentation based on RFM (PCA reduced)')
        plt.tight_layout()
        
        # Log PCA plot to MLflow
        mlflow.log_figure(fig_scatter, "plots/cluster_plot.png")
        
        # Save plots locally for dashboard consumption
        os.makedirs('../bi', exist_ok=True)
        fig_scatter.savefig('../bi/cluster_plot.png')
        fig_scatter.savefig('../../bi_exports/cluster_plot.png')
        plt.close(fig_scatter)
    
    # Save the clustered data back to the database as a new table for BI
    print("Saving cluster assignments to database...")
    df_rfm.to_sql('customer_clusters', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Analysis complete! Cluster data saved to 'customer_clusters' table.")

if __name__ == '__main__':
    # Ensure working directory is the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_analysis()

