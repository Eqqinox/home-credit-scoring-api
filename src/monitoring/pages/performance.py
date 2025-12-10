"""
Page Performance - VERSION SIMPLE (Plan Original).

Affiche UNIQUEMENT :
- Boxplot des latences par endpoint
- Top 10 requêtes les plus lentes
- Logs d'erreurs récentes
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from src.monitoring.storage import PredictionStorage
from src.api.config import settings

# Configuration
st.set_page_config(
    page_title="Performance - Credit Scoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh
REFRESH_INTERVAL_MS = 30000
count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="perf_refresh")

# Titre
st.title("⚡ Performance & Latence")
st.markdown("---")

try:
    storage = PredictionStorage(
        database_url=settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow
    )

    # Charger données (toutes les prédictions)
    with storage.get_session() as session:
        query = text("""
            SELECT timestamp, endpoint, total_response_time_ms,
                   http_status_code, client_id
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 1000
        """)
        result = session.execute(query)
        df = pd.DataFrame(result.fetchall(),
                         columns=['timestamp', 'endpoint', 'total_response_time_ms',
                                 'http_status_code', 'client_id'])

    if df.empty:
        st.warning("⚠️ Aucune donnée disponible")
        storage.close()
        st.stop()

    # 1. Boxplot latences
    st.header("📦 Distribution des Latences")
    fig_box = px.box(
        df,
        x='endpoint',
        y='total_response_time_ms',
        title="Latence par Endpoint",
        labels={'total_response_time_ms': 'Latence (ms)'}
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # 2. Top 10 requêtes lentes
    st.header("🐌 Top 10 Requêtes les Plus Lentes")
    slowest = df.nlargest(10, 'total_response_time_ms')[
        ['timestamp', 'client_id', 'endpoint', 'total_response_time_ms']
    ]
    st.dataframe(slowest, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 3. Erreurs HTTP
    st.header("⚠️ Erreurs HTTP")
    errors = df[df['http_status_code'] != 200]

    if len(errors) > 0:
        st.error(f"⚠️ {len(errors)} erreurs détectées")
        st.dataframe(
            errors[['timestamp', 'client_id', 'endpoint', 'http_status_code']].head(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("✅ Aucune erreur")

    storage.close()

except Exception as e:
    st.error(f"❌ Erreur : {str(e)}")

# Footer
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
