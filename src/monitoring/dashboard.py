"""
Dashboard Streamlit - Page d'accueil.

Point d'entrée principal du dashboard de monitoring.
Affiche le statut du système et permet la navigation.

Usage:
    streamlit run src/monitoring/dashboard.py --server.port 8501
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from src.monitoring.storage import PredictionStorage
from src.api.config import settings
import requests
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring - Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh toutes les 30 secondes
REFRESH_INTERVAL_MS = 30000
count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="home_refresh")

# Titre principal
st.title("🏠 Credit Scoring - Monitoring")
st.markdown("---")

# Section 1 : Statut du système
st.header("📡 Statut du Système")

col1, col2 = st.columns(2)

with col1:
    st.subheader("API FastAPI")
    try:
        response = requests.get("http://localhost:8000/", timeout=3)
        if response.status_code == 200:
            data = response.json()
            st.success("✅ API opérationnelle")
            st.metric("Version API", data.get('model_version', 'N/A'))
            st.metric("Modèle chargé", "✅ Oui" if data.get('model_loaded') else "❌ Non")
        else:
            st.error(f"⚠️ API retourne code {response.status_code}")
    except Exception as e:
        st.error("❌ API inaccessible")
        st.caption(f"Erreur : {str(e)}")
        st.info("Lancez l'API : `uvicorn src.api.main:app --reload --port 8000`")

with col2:
    st.subheader("PostgreSQL")
    try:
        storage = PredictionStorage(
            database_url=settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow
        )
        stats = storage.get_stats()
        st.success("✅ Base de données connectée")
        st.metric("Prédictions totales", stats.get('total_predictions', 0))
        st.metric("Taux d'approbation", f"{stats.get('approval_rate', 0) * 100:.1f}%")
        storage.close()
    except Exception as e:
        st.error("❌ PostgreSQL inaccessible")
        st.caption(f"Erreur : {str(e)}")
        st.info("Vérifiez PostgreSQL : `brew services list`")

# Section 2 : Navigation rapide
st.markdown("---")
st.header("📊 Pages Disponibles")

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/overview.py", label="📈 Overview", icon="📈")
    st.caption("Vue d'ensemble - KPIs et tendances")

    st.page_link("pages/performance.py", label="⚡ Performance", icon="⚡")
    st.caption("Latence, throughput, erreurs")

with col2:
    st.page_link("pages/business.py", label="💼 Business", icon="💼")
    st.caption("Analyse métier - montants, profils")

    st.page_link("pages/drift.py", label="🔍 Data Drift", icon="🔍")
    st.caption("Détection dérive des données")

# Section 3 : Instructions
st.markdown("---")
st.header("📚 Guide d'Utilisation")

with st.expander("🚀 Démarrage rapide"):
    st.markdown("""
    1. **Lancez l'API** :
       ```bash
       ENVIRONMENT=local LOG_LEVEL=INFO uvicorn src.api.main:app --reload --port 8000
       ```

    2. **Lancez le dashboard** :
       ```bash
       streamlit run src/monitoring/dashboard.py --server.port 8501
       ```

    3. **Générez du trafic** (optionnel) :
       ```bash
       python src/scripts/simulate_traffic.py --num-predictions 50 --delay 0.5
       ```

    4. **Naviguez** entre les pages via la barre latérale ⬅️
    """)

with st.expander("⚙️ Configuration"):
    st.markdown(f"""
    - **Auto-refresh** : {REFRESH_INTERVAL_MS / 1000:.0f} secondes
    - **Base de données** : `credit_scoring_prod`
    - **Port API** : 8000
    - **Port Dashboard** : 8501
    """)

# Footer
st.markdown("---")
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"Refresh automatique dans {REFRESH_INTERVAL_MS / 1000:.0f}s")
