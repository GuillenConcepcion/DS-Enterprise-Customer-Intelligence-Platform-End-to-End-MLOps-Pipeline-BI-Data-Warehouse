# Enterprise Customer Intelligence Platform: End-to-End MLOps Pipeline & BI Data Warehouse

<p align="center">
  <img src="images/guillen_logo.png" alt="Guillén Concepción Logo" width="220"/>
</p>

<p align="center">
  <strong>A production-ready data platform integrating SQL/SQLite DW, XGBoost Churn Analytics, MLflow Model Registry, Evidently AI Observability, and Streamlit Web Interface.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"/>
  <img src="https://img.shields.io/badge/streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/mlflow-%23d9efff.svg?style=for-the-badge&logo=mlflow&logoColor=black" alt="MLflow"/>
  <img src="https://img.shields.io/badge/Evidently%20AI-orange?style=for-the-badge" alt="Evidently AI"/>
  <img src="https://img.shields.io/badge/XGBoost-orange.svg?style=for-the-badge" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/terraform-%235835CC.svg?style=for-the-badge&logo=terraform&logoColor=white" alt="Terraform"/>
  <img src="https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS"/>
</p>

---


## 🎯 Project Overview

This repository contains a comprehensive **Data Analysis and Business Intelligence (BI)** project using the famous **UCI Online Retail Dataset**. It is designed to demonstrate advanced capabilities in data modeling, statistical/multivariate analysis, and preparation of semantic models for BI tools like Tableau and PowerBI.

### Core Technologies
- **Data Management:** SQL, SQLite (Data Warehouse simulation)
- **Data Analysis & ML:** Python (`pandas`, `scikit-learn`, `xgboost`)
- **Environment & MLOps:** `uv`, `mlflow`, `evidently`
- **Visualization:** `matplotlib`, `seaborn`

## 🏗️ Architecture & Workflow

```mermaid
flowchart TD
    %% Define Styles
    classDef source fill:#f9d0c4,stroke:#333,stroke-width:1px;
    classDef database fill:#c4e1f9,stroke:#333,stroke-width:1px;
    classDef process fill:#f9f2c4,stroke:#333,stroke-width:1px;
    classDef ml fill:#c4f9d0,stroke:#333,stroke-width:1px;
    classDef output fill:#e2c4f9,stroke:#333,stroke-width:1px;

    subgraph Ingestion
        A[UCI API / Raw Data]:::source -->|uv run| B(Ingest Script):::process
    end

    subgraph Data Warehouse
        B -->|Transactions| C[(SQLite Star Schema)]:::database
        C --> D[(BI Views - RFM)]:::database
    end

    subgraph Data Governance
        D --> E{Data Quality\nChecks}:::process
    end

    subgraph Advanced ML & Observability
        E -->|Valid| F[K-Means & PCA]:::ml
        E -->|Valid| G[XGBoost Churn]:::ml
        F -.-> H[MLflow Tracking]:::ml
        G -.-> H
        F --> I[AI Profiling\nMock LLM]:::ml
        E --> J[Evidently AI\nData Drift]:::ml
    end

    subgraph Business Intelligence
        D --> K[BI Export CSVs]:::output
        F --> K
        K --> L[PowerBI / Tableau]:::output
    end
```

### ☁️ Enterprise Cloud Deployment (AWS Topology)
While this project runs locally using SQLite and `uv` for ease of evaluation, it is designed with a **Cloud-Native mindset**. Below is the architectural blueprint for deploying this MLOps pipeline to a production AWS environment:

```mermaid
flowchart LR
    %% Styles
    classDef aws fill:#FF9900,stroke:#232F3E,stroke-width:2px,color:white;
    classDef storage fill:#3F8624,stroke:#232F3E,stroke-width:2px,color:white;
    classDef compute fill:#D86613,stroke:#232F3E,stroke-width:2px,color:white;
    classDef ml fill:#01A99D,stroke:#232F3E,stroke-width:2px,color:white;
    classDef bi fill:#0052CC,stroke:#232F3E,stroke-width:2px,color:white;

    subgraph AWS [AWS Cloud Infrastructure]
        direction TB
        
        subgraph Data Lake & DWH
            S3[(Amazon S3\nRaw Data)]:::storage
            RDS[(Amazon Redshift\nStar Schema)]:::storage
        end

        subgraph Data Engineering Pipeline
            GLUE[AWS Glue\nETL & Ingestion]:::compute
            DQ[Data Quality\nChecks]:::compute
        end

        subgraph MLOps & Advanced Analytics
            SM[Amazon SageMaker\nXGBoost & K-Means]:::ml
            MLF[MLflow on EKS\nModel Registry]:::ml
            EVI[Evidently AI\nData Drift Monitor]:::ml
            LLM[AWS Bedrock\nAI Profiling]:::ml
        end

        subgraph Serving & BI
            APP[Streamlit App\nAWS App Runner]:::bi
            PBI[Tableau / PowerBI\nSemantic Models]:::bi
        end
    end

    %% Flow
    UCI([External API]) --> S3
    S3 --> GLUE
    GLUE --> RDS
    RDS --> DQ
    DQ -->|Pass| SM
    SM -->|Log Models| MLF
    SM --> LLM
    RDS --> EVI
    MLF -.->|Load Model| APP
    RDS --> APP
    RDS --> PBI
```

### 1. Ingestion & ETL (Python + SQL)
- The raw data (over 500k transactions) is ingested directly from the UCI Machine Learning Repository via API.
- We utilize SQL DDL (`sql/01_ddl_tables.sql`) to construct a robust **Star Schema** (Fact and Dimension tables) inside SQLite, ensuring referential integrity and optimized querying.

### 2. Data Modeling & BI Views (SQL)
- Complex SQL views (`sql/02_bi_views.sql`) are implemented to prepare Data Marts.
- This includes **Common Table Expressions (CTEs)** and **Window Functions** to build base tables for Cohort Analysis and RFM (Recency, Frequency, Monetary) calculations.

### 3. Data Governance & Quality (MLOps)
- **Data Contracts:** A native Pandas assertion layer validates data integrity before modeling (`src/data/data_quality_checks.py`), ensuring no nulls and valid RFM ranges.

### 4. Multivariate Statistical Analysis (Python)
- **Clustering (K-Means):** The RFM base view is scaled with a log transformation and processed to segment customers.
- **AI Cluster Profiling:** A script (`src/analysis/ai_cluster_profiling.py`) analyzes cluster centroids to generate automated semantic marketing personas.
- **Dimensionality Reduction (PCA):** Applied to reduce the RFM space into 2D for intuitive visualization.

### 5. Advanced ML: Churn Prediction (XGBoost)
- **Supervised Learning:** Features (Recency, Frequency, Monetary) are extracted chronologically. We train an **XGBoost Classifier** (`src/analysis/churn_prediction.py`) to predict the probability of a customer churning.
- **Experiment Tracking:** **MLflow** tracks hyperparameters, logs the final model, and visualizes metrics like ROC-AUC.

### 6. Observability & BI Export
- **Data Drift:** **Evidently AI** evaluates dataset shifts generating an HTML report (`src/analysis/monitoring_evidently.py`).
- **Semantic Models:** The final modeled data is exported into CSVs, formatted for immediate ingestion into BI tools.

## 🖼️ Interactive Model Insights & Artifacts Gallery

To view the advanced data science and MLOps visualizations generated by this platform, expand the slides below:

<details>
<summary><strong>📊 Slide 1: Customer Segmentation Clusters (PCA + K-Means)</strong></summary>
<br/>
<p align="center">
  <img src="bi_exports/cluster_plot.png" alt="Customer Clusters" width="70%"/>
</p>
<p align="center">
  <em>Figure 1: RFM metrics compressed into 2D via Principal Component Analysis (PCA) and segmented into optimized cohorts using K-Means clustering. This allows for automated semantic customer profiling.</em>
</p>
</details>

<details>
<summary><strong>📈 Slide 2: K-Means Optimization (Elbow Method)</strong></summary>
<br/>
<p align="center">
  <img src="bi_exports/kmeans_tuning.png" alt="K-Means Tuning" width="70%"/>
</p>
<p align="center">
  <em>Figure 2: Silhouette analysis and Within-Cluster Sum of Squares (WCSS) to determine the mathematical optimal number of clusters (K) for customer segmentation.</em>
</p>
</details>

<details>
<summary><strong>🔍 Slide 3: Model Explainability (SHAP Feature Importance)</strong></summary>
<br/>
<p align="center">
  <img src="bi_exports/shap_summary.png" alt="SHAP Summary" width="70%"/>
</p>
<p align="center">
  <em>Figure 3: SHAP (SHapley Additive exPlanations) values highlighting the global feature importance and their directional impact on the XGBoost Churn Prediction model.</em>
</p>
</details>

<details>
<summary><strong>📉 Slide 4: XGBoost Churn Model Performance (Confusion Matrix)</strong></summary>
<br/>
<p align="center">
  <img src="bi_exports/confusion_matrix.png" alt="Confusion Matrix" width="60%"/>
</p>
<p align="center">
  <em>Figure 4: Confusion matrix detailing the precision, recall, and classification efficiency of the XGBoost classifier on the test dataset.</em>
</p>
</details>

<details>
<summary><strong>🎯 Slide 5: Business Impact (Cumulative Gains Chart)</strong></summary>
<br/>
<p align="center">
  <img src="bi_exports/cumulative_gains.png" alt="Cumulative Gains" width="70%"/>
</p>
<p align="center">
  <em>Figure 5: Cumulative Gains chart showing how much better the XGBoost model performs compared to a random selection, directly translating model accuracy into targeting efficiency for retention campaigns.</em>
</p>
</details>

---

## 🚀 How to Run

1. **Install dependencies:**
   Run `uv sync` or `uv install` to set up the virtual environment.
2. **Execute the Pipeline:**
   - Ingest & Model: `uv run src/data/ingest_uci_dataset.py`
   - Data Quality Check: `uv run src/data/data_quality_checks.py`
   - K-Means & PCA: `uv run src/analysis/multivariate_analysis.py`
   - AI Profiling: `uv run src/analysis/ai_cluster_profiling.py`
   - Churn Prediction (XGBoost): `uv run src/analysis/churn_prediction.py`
   - Monitoring & Drift: `uv run src/analysis/monitoring_evidently.py`
3. **MLflow UI:**
   Run `uv run mlflow ui` to view the model tracking dashboard.

## 📊 Analytics & MLOps Highlights
- **End-to-End Architecture:** Combines Data Engineering (SQL), Governance, and Advanced ML.
- **Predictive Analytics:** Implementation of XGBoost to solve tangible business problems (Customer Churn).
- **MLOps Best Practices:** Robust logging with MLflow and model observability with Evidently AI.

## 👤 Author Profile

<table style="border-collapse: collapse; border: none; width: 100%;">
  <tr style="border: none;">
    <td width="150" align="center" style="border: none; vertical-align: middle;">
      <img src="images/guillen_logo.png" alt="Guillén Concepción" width="130" style="border-radius: 8px;"/>
    </td>
    <td style="border: none; vertical-align: middle; padding-left: 20px;">
      <strong>Guillén Concepción</strong><br/>
      <em>Senior Data Scientist & MLOps Engineer</em>
      <p style="margin-top: 8px; margin-bottom: 12px; font-size: 0.95em; color: #4A4A4A;">
        Experto en diseño, desarrollo y despliegue de soluciones integrales de Inteligencia Artificial. Pragmático y centrado en el valor de negocio, abarcando desde la fase de investigación (CRISP-DM) hasta sistemas de producción escalables, resilientes y auditables utilizando arquitecturas Cloud-Native y prácticas MLOps.
      </p>
      <div>
        <a href="mailto:guillenconcepcion@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Email-guillenconcepcion%40gmail.com-blue?style=flat-square&logo=gmail&logoColor=white" alt="Email"/></a>
        <a href="https://www.linkedin.com/in/guillen-concepcion-25266b127" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-Guill%C3%A9n%20Concepci%C3%B3n-blue?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
        <a href="https://github.com/GuillenConcepcion" target="_blank"><img src="https://img.shields.io/badge/GitHub-GuillenConcepcion-black?style=flat-square&logo=github&logoColor=white" alt="GitHub"/></a>
      </div>
    </td>
  </tr>
</table>

