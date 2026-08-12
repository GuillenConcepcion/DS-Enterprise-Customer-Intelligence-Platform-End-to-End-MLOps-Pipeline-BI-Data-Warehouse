import sqlite3
import pandas as pd
import os
import json
import urllib.request
import urllib.error

def get_offline_fallback(cluster_id, recency, frequency, monetary_value):
    # Simulated persona logic (offline fallback)
    if recency < 50 and frequency > 4:
        return "[Champions]: Clientes muy recientes, frecuentes y con alto nivel de gasto. Recomendación: Incorporar a programa de fidelización VIP, ofrecer acceso exclusivo y preventas de nuevos lanzamientos."
    elif recency > 150:
        return "[At Risk / Lost]: Clientes inactivos que no han comprado en mucho tiempo. Recomendación: Campaña agresiva de reactivación mediante cupones de descuento personalizados y encuestas de satisfacción."
    elif recency < 50 and frequency <= 4:
        return "[New / Promising]: Clientes de reciente incorporación pero con pocas compras. Recomendación: Incentivar la segunda compra con correos post-venta de recomendación de productos complementarios (cross-selling)."
    else:
        return "[Loyal Customers]: Compradores regulares y consistentes con gasto moderado. Recomendación: Mantener el compromiso diario a través de recompensas acumulables por puntos y comunicaciones informativas periódicas."

def get_llm_profile(cluster_id, recency, frequency, monetary_value):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("INFO: GEMINI_API_KEY not found in environment variables. Using offline rule-based fallback.")
        return get_offline_fallback(cluster_id, recency, frequency, monetary_value)
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    prompt = f"""You are an expert marketing strategist. Analyze this customer segment centroid:
Cluster ID: {cluster_id}
Average Recency: {recency} days since last purchase
Average Frequency: {frequency} transactions
Average Monetary Value: ${monetary_value} spent

Based on this data:
1. Provide a short, catchy semantic name for this customer persona in Spanish (e.g. Champions, Clientes en Riesgo, Nuevos Compradores).
2. Write a 1-sentence description of their purchasing behavior in Spanish.
3. Write a 1-sentence marketing recommendation in Spanish.

Keep the output brief and formatted exactly as follows:
[Nombre Persona]: [Descripción]. [Recomendación]
Do not include any other markdown formatting, bold tags, or intro/outro text."""

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        # 10s timeout to prevent hanging the pipeline
        with urllib.request.urlopen(req, timeout=10) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            persona_text = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
            # Clean up potential markdown formatting that Gemini might output
            if persona_text.startswith("```"):
                persona_text = persona_text.replace("```", "")
            return persona_text
    except Exception as e:
        print(f"Warning: Gemini API call failed ({e}). Falling back to local offline profiling.")
        return get_offline_fallback(cluster_id, recency, frequency, monetary_value)

def run_ai_profiling():
    print("Connecting to Data Warehouse...")
    db_path = '../data/retail_dw.db'
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return
        
    conn = sqlite3.connect(db_path)
    
    try:
        print("Loading Customer Clusters...")
        df = pd.read_sql_query("SELECT * FROM customer_clusters", conn)
    except Exception:
        print("Error: 'customer_clusters' table not found. Run multivariate_analysis.py first.")
        conn.close()
        return
    
    # Calculate Centroids (Average RFM per Cluster)
    centroids = df.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean().round(2)
    
    print("\n--- RUNNING AI CLUSTER PROFILING ---")
    
    cluster_profiles = {}
    db_records = []
    
    for cluster_id, row in centroids.iterrows():
        print(f"Profiling Cluster {cluster_id}...")
        persona = get_llm_profile(cluster_id, row['Recency'], row['Frequency'], row['MonetaryValue'])
        print(f"Result: {persona}\n")
        
        cluster_profiles[str(cluster_id)] = persona
        db_records.append((int(cluster_id), persona, float(row['Recency']), float(row['Frequency']), float(row['MonetaryValue'])))
        
    # Save profiles back to SQLite database
    print("Saving profiles to SQLite table 'customer_cluster_profiles'...")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS customer_cluster_profiles")
    cursor.execute("""
        CREATE TABLE customer_cluster_profiles (
            Cluster INTEGER PRIMARY KEY,
            Persona TEXT,
            Recency REAL,
            Frequency REAL,
            MonetaryValue REAL
        )
    """)
    cursor.executemany("""
        INSERT INTO customer_cluster_profiles (Cluster, Persona, Recency, Frequency, MonetaryValue)
        VALUES (?, ?, ?, ?, ?)
    """, db_records)
    conn.commit()
    conn.close()
    
    # Export profiles to JSON for Streamlit dashboard consumption
    export_dir = '../../bi_exports'
    os.makedirs(export_dir, exist_ok=True)
    json_path = os.path.join(export_dir, 'customer_personas.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_profiles, f, ensure_ascii=False, indent=4)
        
    print(f"AI Profiling complete. Profiles exported to: {json_path}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_ai_profiling()

