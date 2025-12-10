"""
Page Overview - Vue d'ensemble du monitoring.

Affiche :
- KPIs clés (total prédictions, taux approbation, latence moyenne)
- Distribution des décisions (approve/refuse)
- Volume de prédictions par heure
- Distribution des scores de probabilité
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from src.monitoring.storage import PredictionStorage
from src.api.config import settings

# Configuration
st.set_page_config(
    page_title="Overview - Credit Scoring",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh
REFRESH_INTERVAL_MS = 30000
count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="overview_refresh")

# Couleurs
COLORS = {
    'approve': '#22c55e',
    'refuse': '#ef4444',
    'low': '#fbbf24',
    'medium': '#fb923c',
    'high': '#22c55e',
    'info': '#3b82f6',
}

# Titre
st.title("📈 Vue d'Ensemble")
st.markdown("---")

# Sidebar : Filtres temporels
st.sidebar.header("🔍 Filtres")
period = st.sidebar.selectbox(
    "Période",
    ["Dernières 24h", "7 derniers jours", "30 derniers jours", "Tout"],
    index=0
)

# Calculer start_date selon la période
now = datetime.now()
if period == "Dernières 24h":
    start_date = now - timedelta(hours=24)
elif period == "7 derniers jours":
    start_date = now - timedelta(days=7)
elif period == "30 derniers jours":
    start_date = now - timedelta(days=30)
else:
    start_date = None

# Charger les données
try:
    storage = PredictionStorage(
        database_url=settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow
    )

    # Stats globales
    stats = storage.get_stats(start_date=start_date)

    # Prédictions détaillées
    predictions = storage.get_predictions(
        limit=1000,
        start_date=start_date
    )

    if not predictions:
        st.warning("⚠️ Aucune prédiction disponible pour la période sélectionnée")
        st.info("Générez du trafic avec : `python src/scripts/simulate_traffic.py --num-predictions 50`")
        storage.close()
        st.stop()

    df = pd.DataFrame(predictions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Section 1 : KPIs
    st.header("📊 Indicateurs Clés")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Prédictions",
            value=f"{stats['total_predictions']:,}",
            delta=None
        )

    with col2:
        approval_rate = stats['approval_rate'] * 100
        st.metric(
            label="Taux d'Approbation",
            value=f"{approval_rate:.1f}%",
            delta=None
        )

    with col3:
        st.metric(
            label="Latence Moyenne",
            value=f"{stats['avg_inference_time_ms']:.2f} ms",
            delta=None
        )

    with col4:
        error_rate = (stats['error_count'] / stats['total_predictions'] * 100) if stats['total_predictions'] > 0 else 0
        st.metric(
            label="Taux d'Erreur",
            value=f"{error_rate:.1f}%",
            delta=None
        )

    st.markdown("---")

    # Section 2 : Distribution des décisions
    st.header("🎯 Distribution des Décisions")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Pie chart
        decision_counts = df['decision'].value_counts()
        fig_pie = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title="Répartition Approve / Refuse",
            color=decision_counts.index,
            color_discrete_map={'approve': COLORS['approve'], 'refuse': COLORS['refuse']},
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Détails")
        st.metric("Approuvées", stats['approve_count'])
        st.metric("Refusées", stats['refuse_count'])
        st.metric("Erreurs", stats['error_count'])

    st.markdown("---")

    # Section 3 : Volume de prédictions dans le temps
    st.header("📈 Volume de Prédictions")

    # Agréger par heure
    df['hour'] = df['timestamp'].dt.floor('H')
    hourly_counts = df.groupby('hour').size().reset_index(name='count')

    fig_volume = px.line(
        hourly_counts,
        x='hour',
        y='count',
        title="Nombre de Prédictions par Heure",
        labels={'hour': 'Heure', 'count': 'Nombre de prédictions'}
    )
    fig_volume.update_traces(line_color=COLORS['info'], line_width=2)
    fig_volume.update_layout(hovermode='x unified')
    st.plotly_chart(fig_volume, use_container_width=True)

    st.markdown("---")

    # Section 4 : Distribution des scores
    st.header("📊 Distribution des Scores de Probabilité")

    fig_hist = px.histogram(
        df,
        x='prediction_proba',
        nbins=50,
        title="Distribution des Probabilités de Défaut",
        labels={'prediction_proba': 'Probabilité de défaut', 'count': 'Nombre de clients'},
        color_discrete_sequence=[COLORS['info']]
    )
    fig_hist.add_vline(
        x=0.5225,
        line_dash="dash",
        line_color="red",
        annotation_text="Seuil (0.5225)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Section 5 : Confidence Level
    st.header("🎯 Niveaux de Confiance")

    confidence_counts = df['confidence_level'].value_counts()
    fig_confidence = px.bar(
        x=confidence_counts.index,
        y=confidence_counts.values,
        title="Répartition des Niveaux de Confiance",
        labels={'x': 'Niveau de confiance', 'y': 'Nombre de prédictions'},
        color=confidence_counts.index,
        color_discrete_map={
            'LOW': COLORS['low'],
            'MEDIUM': COLORS['medium'],
            'HIGH': COLORS['high']
        }
    )
    st.plotly_chart(fig_confidence, use_container_width=True)

    storage.close()

except Exception as e:
    st.error(f"❌ Erreur lors du chargement des données : {str(e)}")
    st.info("Vérifiez que PostgreSQL est actif et que l'API fonctionne")

# Footer
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
