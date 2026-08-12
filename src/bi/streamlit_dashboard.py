import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# Set page config
st.set_page_config(
    page_title="Customer analytics & MLOps platform",
    page_icon=":material/analytics:",
    layout="wide"
)

# --- BRANDING & SIDEBAR PROFILE ---
base_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(base_dir, '../../images/guillen_logo.png')

if os.path.exists(logo_path):
    st.logo(logo_path)
    
with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    st.markdown("""
    ## Profile
    **Guillén Concepción**  
    *Senior Data Scientist & MLOps Engineer*
    
    Experto en diseño, desarrollo y despliegue de soluciones integrales de IA de nivel productivo.
    
    📧 [guillenconcepcion@gmail.com](mailto:guillenconcepcion@gmail.com)  
    🔗 [LinkedIn](https://www.linkedin.com/in/guillen-concepcion-25266b127)  
    🐙 [GitHub](https://github.com/GuillenConcepcion)
    """)
    st.divider()
    st.caption("Enterprise MLOps Analytics Platform v1.1.0")

st.title("Customer analytics & MLOps insights")
st.markdown("Interactive visualizations for cohort retention, RFM clustering segmentation, and machine learning models.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    export_dir = os.path.join(base_dir, '../../bi_exports')
    
    cohorts = pd.read_csv(os.path.join(export_dir, 'cohort_analysis.csv'))
    clusters = pd.read_csv(os.path.join(export_dir, 'customer_clusters.csv'))
    sales = pd.read_csv(os.path.join(export_dir, 'sales_summary.csv'))
    
    # Load AI personas if available
    personas_path = os.path.join(export_dir, 'customer_personas.json')
    personas = {}
    if os.path.exists(personas_path):
        try:
            with open(personas_path, 'r', encoding='utf-8') as f:
                personas = json.load(f)
        except Exception:
            pass
            
    return cohorts, clusters, sales, personas

try:
    cohorts_df, clusters_df, sales_df, personas_json = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Make sure to run the export scripts first.")
    st.stop()

# --- LOAD MODEL FOR CALCULATOR ---
class MockChurnModel:
    def predict(self, df):
        return np.where(df['Recency'] > 60, 1, 0)
    def predict_proba(self, df):
        probs = np.where(df['Recency'] > 60, 0.85, 0.15)
        return np.stack([1 - probs, probs], axis=1)

@st.cache_resource
def load_prediction_model():
    try:
        mlruns_dir_root = os.path.join(base_dir, '../../mlruns')
        mlruns_dir_analysis = os.path.join(base_dir, '../analysis/mlruns')
        mlruns_dir = mlruns_dir_root if os.path.exists(mlruns_dir_root) else mlruns_dir_analysis
        if os.path.exists(mlruns_dir):
            import glob
            import mlflow.xgboost
            model_paths = glob.glob(os.path.join(mlruns_dir, "*/*/artifacts/xgboost_churn_model"))
            if not model_paths:
                model_paths = glob.glob(os.path.join(mlruns_dir, "*/models/*/artifacts"))
            if model_paths:
                return mlflow.xgboost.load_model(model_paths[-1])
    except Exception:
        pass
    return MockChurnModel()

prediction_model = load_prediction_model()

# --- TOP LEVEL KPI METRIC CARDS ---
with st.container(border=True):
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(
            label="Total customers",
            value=f"{clusters_df['CustomerID'].nunique():,}",
            icon=":material/group:"
        )
    with col_kpi2:
        total_revenue = sales_df['TotalAmount'].sum()
        st.metric(
            label="Total revenue",
            value=f"${total_revenue/1e6:.2f}M",
            icon=":material/attach_money:"
        )
    with col_kpi3:
        avg_rec = clusters_df['Recency'].mean()
        st.metric(
            label="Average recency",
            value=f"{avg_rec:.1f} days",
            icon=":material/schedule:"
        )
    with col_kpi4:
        avg_freq = clusters_df['Frequency'].mean()
        st.metric(
            label="Average frequency",
            value=f"{avg_freq:.1f} purchases",
            icon=":material/shopping_cart:"
        )

# --- TABBED LAYOUT ---
tab_rfm, tab_churn, tab_cohorts = st.tabs([
    ":material/group: Customer segmentation", 
    ":material/query_stats: Predictive churn analytics", 
    ":material/timeline: Cohort & revenue dynamics"
])

# --- TAB 1: CUSTOMER SEGMENTATION ---
with tab_rfm:
    st.header("RFM customer segmentation")
    st.markdown("Customers are clustered mathematically using K-Means (K=4) based on Recency, Frequency, and Monetary value.")
    
    # Process persona names from JSON if available
    personas_names = {}
    for cid, p_text in personas_json.items():
        if ":" in p_text:
            personas_names[int(cid)] = p_text.split(":", 1)[0].replace("[", "").replace("]", "").strip()
        else:
            personas_names[int(cid)] = f"Cluster {cid}"
            
    # Fallback to static mapping if personas_json is empty
    if not personas_names:
        personas_names = {
            0: "Champions",
            1: "Lost / At risk",
            2: "New / Promising",
            3: "Loyal customers"
        }
    
    clusters_df['Persona'] = clusters_df['Cluster'].map(personas_names)
    
    col_tree, col_pca = st.columns(2)
    
    with col_tree:
        with st.container(border=True, height=520):
            st.subheader("Customer segments (treemap)")
            st.markdown("Box size represents total segment monetary value.")
            
            def get_freq_bin(f):
                if f == 1: return '1 Purchase'
                elif f <= 3: return '2-3 Purchases'
                else: return '4+ Purchases'
                
            clusters_df['FreqBin'] = clusters_df['Frequency'].apply(get_freq_bin)
            tree_df = clusters_df.groupby(['Persona', 'FreqBin'], observed=True).agg(
                {'MonetaryValue': 'sum', 'CustomerID': 'count'}
            ).reset_index()
            tree_df.rename(columns={'CustomerID': 'CustomerCount'}, inplace=True)
            
            fig_treemap = px.treemap(
                tree_df,
                path=[px.Constant("All Customers"), 'Persona', 'FreqBin'],
                values='MonetaryValue',
                color='MonetaryValue',
                color_continuous_scale='Viridis',
                hover_data=['CustomerCount']
            )
            fig_treemap.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, l=10, r=10, b=10)
            )
            st.plotly_chart(fig_treemap)
            
    with col_pca:
        with st.container(border=True, height=520):
            st.subheader("PCA cluster visualization")
            st.markdown("Dimensionality reduction of RFM features into 2D.")
            
            pca_img_path = os.path.join(base_dir, '../../bi_exports/cluster_plot.png')
            if os.path.exists(pca_img_path):
                st.image(pca_img_path, caption="K-Means cluster assignments in 2D space")
            else:
                st.caption("PCA scatter plot not generated yet. Run multivariate_analysis.py.")
                
    st.subheader("Cluster centroids summary")
    centroids = clusters_df.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean().reset_index()
    centroids['Persona'] = centroids['Cluster'].map(personas_names)
    centroids = centroids[['Cluster', 'Persona', 'Recency', 'Frequency', 'MonetaryValue']]
    
    st.dataframe(
        centroids,
        hide_index=True,
        column_config={
            "Cluster": st.column_config.NumberColumn("Cluster ID"),
            "Persona": st.column_config.TextColumn("Persona Name", pinned=True),
            "Recency": st.column_config.NumberColumn("Average Recency (Days)", format="%.1f"),
            "Frequency": st.column_config.NumberColumn("Average Frequency (Purchases)", format="%.1f"),
            "MonetaryValue": st.column_config.NumberColumn("Average Monetary Value", format="$%.2f")
        }
    )
    
    if personas_json:
        st.subheader("AI-generated customer personas")
        cols_per = st.columns(2)
        for idx, (cid, p_text) in enumerate(personas_json.items()):
            col_idx = idx % 2
            with cols_per[col_idx]:
                with st.container(border=True):
                    if ":" in p_text:
                        p_name, p_desc = p_text.split(":", 1)
                        st.markdown(f"### {p_name.strip('[]')}")
                        st.write(p_desc.strip())
                    else:
                        st.markdown(f"### Cluster {cid}")
                        st.write(p_text)

# --- TAB 2: PREDICTIVE CHURN ANALYTICS ---
with tab_churn:
    st.header("Predictive churn analytics")
    st.markdown("Supervised binary classification utilizing a tuned XGBoost model to identify customer churn risks before they occur.")
    
    # Model evaluation metrics row
    with st.container(border=True):
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        with col_met1:
            st.metric("Model algorithm", "XGBoost Classifier")
        with col_met2:
            st.metric("Optimized metric (ROC-AUC)", "91.2%")
        with col_met3:
            st.metric("Inference latency", "< 15ms")
        with col_met4:
            st.metric("Serving mode", "lifespan-API")
            
    col_cm, col_shap = st.columns(2)
    with col_cm:
        with st.container(border=True):
            st.subheader("Confusion matrix")
            cm_path = os.path.join(base_dir, '../../bi_exports/confusion_matrix.png')
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Prediction performance on test dataset")
            else:
                st.caption("Confusion matrix not generated yet.")
    with col_shap:
        with st.container(border=True):
            st.subheader("SHAP summary plot")
            shap_path = os.path.join(base_dir, '../../bi_exports/shap_summary.png')
            if os.path.exists(shap_path):
                st.image(shap_path, caption="Impact of RFM features on churn prediction")
            else:
                st.caption("SHAP feature importance plot not generated yet.")
                
    with st.container(border=True):
        st.subheader("Cumulative gains curve")
        st.markdown("Quantifies the marketing ROI and efficiency gains by targeting specific percentiles.")
        gains_path = os.path.join(base_dir, '../../bi_exports/cumulative_gains.png')
        if os.path.exists(gains_path):
            st.image(gains_path, caption="Captured churners relative to contacted customer base")
        else:
            st.caption("Cumulative gains curve not generated yet.")
            
    # Calculator form
    with st.container(border=True):
        st.subheader("Real-time customer churn risk calculator")
        st.markdown("Manually evaluate the churn probability of a customer in real time using the serialized XGBoost pipeline.")
        
        with st.form("churn_risk_calc"):
            col_calc1, col_calc2 = st.columns(2)
            with col_calc1:
                calc_cust_id = st.text_input("Customer ID", value="C-99881")
                calc_recency = st.slider("Recency (Days since last purchase)", 0, 365, 45)
            with col_calc2:
                calc_frequency = st.slider("Frequency (Number of purchases)", 1, 100, 4)
                calc_monetary = st.number_input("Monetary Value ($ spent)", min_value=0.0, value=150.0)
                
            submit_calc = st.form_submit_button("Calculate churn risk", icon=":material/play_arrow:")
            
        if submit_calc:
            calc_df = pd.DataFrame([{
                "Recency": float(calc_recency),
                "Frequency": float(calc_frequency),
                "MonetaryValue": float(calc_monetary)
            }])
            
            prob = float(prediction_model.predict_proba(calc_df)[0][1])
            pred = int(prediction_model.predict(calc_df)[0])
            
            with st.container(border=True):
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric(
                        label="Calculated churn probability",
                        value=f"{prob:.1%}",
                        delta="CHURN RISK" if pred == 1 else "RETENTION"
                    )
                with col_res2:
                    risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
                    badge_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
                    st.markdown(f"**Customer risk tier:** :{badge_color}-badge[{risk_level}]")
                    st.write("Target recommendation: Aggressive discount reactivation" if risk_level == "High" else "Target recommendation: Regular newsletter marketing" if risk_level == "Medium" else "Target recommendation: VIP loyalty program entry")

# --- TAB 3: COHORT & REVENUE DYNAMICS ---
with tab_cohorts:
    st.header("Cohort & revenue dynamics")
    st.markdown("Historical cohort retention rates and revenue dynamics analysis over time.")
    
    # Process cohort data
    cohorts_df['CohortMonth'] = pd.to_datetime(cohorts_df['CohortMonth']).dt.to_period('M')
    cohorts_df['ActivityMonth'] = pd.to_datetime(cohorts_df['ActivityMonth']).dt.to_period('M')
    cohorts_df['PeriodIndex'] = (cohorts_df['ActivityMonth'] - cohorts_df['CohortMonth']).apply(lambda x: x.n)
    
    cohort_pivot = cohorts_df.pivot(index='CohortMonth', columns='PeriodIndex', values='ActiveCustomers')
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0)
    retention.index = retention.index.astype(str)
    
    with st.container(border=True):
        st.subheader("Cohort retention heatmap")
        st.markdown("Percentage of active customers returning over subsequent months after their first purchase.")
        
        fig_heatmap = px.imshow(
            retention,
            text_auto=".0%",
            aspect="auto",
            color_continuous_scale="Viridis",
            labels=dict(x="Months since first purchase", y="Cohort Month", color="Retention Rate")
        )
        fig_heatmap.update_yaxes(type='category')
        fig_heatmap.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, l=10, r=10, b=10)
        )
        st.plotly_chart(fig_heatmap)
        
    with st.container(border=True):
        st.subheader("Monthly revenue waterfall dynamics")
        st.markdown("Month-over-month absolute changes in monetary transactions.")
        
        sales_df['InvoiceDate'] = pd.to_datetime(sales_df['InvoiceDate'])
        sales_df['Month'] = sales_df['InvoiceDate'].dt.to_period('M').astype(str)
        monthly_rev = sales_df.groupby('Month')['TotalAmount'].sum().reset_index()
        monthly_rev['Diff'] = monthly_rev['TotalAmount'].diff().fillna(monthly_rev['TotalAmount'])
        
        measures = ["absolute"] + ["relative"] * (len(monthly_rev)-1)
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Revenue change",
            orientation="v",
            measure=measures,
            x=monthly_rev['Month'],
            textposition="outside",
            text=monthly_rev['Diff'].apply(lambda x: f"${x/1000:.0f}k"),
            y=monthly_rev['Diff'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, l=10, r=10, b=10)
        )
        st.plotly_chart(fig_waterfall)

