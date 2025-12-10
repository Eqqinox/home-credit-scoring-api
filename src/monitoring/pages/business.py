"""
Page Business - VERSION SIMPLE (Plan Original).

Affiche UNIQUEMENT :
- Distribution montants de crédit
- Profils clients approuvés/refusés
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
    page_title="Business - Credit Scoring",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh
REFRESH_INTERVAL_MS = 30000
count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="business_refresh")

# Couleurs
COLORS = {
    'approve': '#22c55e',
    'refuse': '#ef4444',
}

# Titre
st.title("💼 Analyse Métier")
st.markdown("---")

try:
    storage = PredictionStorage(
        database_url=settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow
    )

    # Charger données
    with storage.get_session() as session:
        query = text("""
            SELECT decision, loan_amount
            FROM predictions
            WHERE loan_amount IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1000
        """)
        result = session.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=['decision', 'loan_amount'])

    if df.empty:
        st.warning("⚠️ Aucune donnée disponible")
        storage.close()
        st.stop()

    # 1. Profils clients (pie chart)
    st.header("🎯 Profils Clients")
    decision_counts = df['decision'].value_counts()
    fig_pie = px.pie(
        values=decision_counts.values,
        names=decision_counts.index,
        title="Répartition Approve / Refuse",
        color=decision_counts.index,
        color_discrete_map={'approve': COLORS['approve'], 'refuse': COLORS['refuse']}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # 2. Distribution montants
    st.header("💰 Distribution des Montants de Crédit")
    fig_hist = px.histogram(
        df,
        x='loan_amount',
        color='decision',
        nbins=30,
        title="Montants par Décision",
        labels={'loan_amount': 'Montant du crédit (€)'},
        color_discrete_map={'approve': COLORS['approve'], 'refuse': COLORS['refuse']}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    storage.close()

except Exception as e:
    st.error(f"❌ Erreur : {str(e)}")

# Footer
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
