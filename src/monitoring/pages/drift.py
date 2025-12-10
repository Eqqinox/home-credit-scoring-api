"""
Page Data Drift - Détection de la dérive des données.

Affiche :
- Dernier rapport Evidently AI (HTML embarqué)
- Historique des scores de drift
- Alertes si drift détecté
- Liste des features affectées
- Placeholder pour Phase 6 (génération automatique)
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from pathlib import Path
from src.monitoring.storage import PredictionStorage
from src.api.config import settings

# Configuration
st.set_page_config(
    page_title="Data Drift - Credit Scoring",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh
REFRESH_INTERVAL_MS = 30000
count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="drift_refresh")

# Titre
st.title("🔍 Détection Data Drift")
st.markdown("---")

try:
    storage = PredictionStorage(
        database_url=settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow
    )

    # Requête pour récupérer les rapports de drift
    with storage.get_session() as session:
        query = text("""
            SELECT
                id,
                generated_at,
                report_period_start,
                report_period_end,
                current_dataset_size,
                reference_dataset_size,
                drift_detected,
                drift_score,
                n_features_drifted,
                drifted_features,
                report_html_path
            FROM drift_reports
            ORDER BY generated_at DESC
            LIMIT 10
        """)

        result = session.execute(query)
        rows = result.fetchall()

    if not rows:
        # Pas de rapports disponibles
        st.info("ℹ️ Aucun rapport de drift disponible")

        st.markdown("""
        ### 📝 Phase 6 : Génération Automatique des Rapports

        Les rapports de drift seront générés dans la **Phase 6** avec Evidently AI.

        **Fonctionnalités prévues** :
        - Détection automatique du drift sur les features
        - Comparaison dataset de référence vs production
        - Rapports HTML interactifs
        - Alertes si drift critique détecté
        - Historique des scores de drift

        **Commande de génération** (Phase 6) :
        ```bash
        python src/scripts/generate_drift_report.py
        ```
        """)

        # Placeholder : Simulation visuelle
        st.markdown("---")
        st.subheader("📊 Aperçu (Données Simulées)")

        # Graphique exemple
        example_data = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=10, freq='D'),
            'drift_score': [0.05, 0.08, 0.12, 0.15, 0.22, 0.18, 0.25, 0.30, 0.28, 0.26]
        })

        fig_example = px.line(
            example_data,
            x='date',
            y='drift_score',
            title="Évolution du Score de Drift (Exemple)",
            labels={'date': 'Date', 'drift_score': 'Score de drift'}
        )
        fig_example.add_hline(
            y=0.2,
            line_dash="dash",
            annotation_text="Seuil d'alerte (0.2)",
            line_color="red"
        )
        st.plotly_chart(fig_example, use_container_width=True)

        storage.close()
        st.stop()

    # Des rapports existent
    df = pd.DataFrame(rows, columns=[
        'id', 'generated_at', 'report_period_start', 'report_period_end',
        'current_dataset_size', 'reference_dataset_size', 'drift_detected',
        'drift_score', 'n_features_drifted', 'drifted_features', 'report_html_path'
    ])
    df['generated_at'] = pd.to_datetime(df['generated_at'])

    # Section 1 : Dernier rapport
    st.header("📄 Dernier Rapport")

    latest = df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Drift Détecté", "⚠️ OUI" if latest['drift_detected'] else "✅ NON")

    with col2:
        score = latest['drift_score'] if latest['drift_score'] is not None else 0.0
        st.metric("Score de Drift", f"{score:.3f}")

    with col3:
        st.metric("Features Affectées", latest['n_features_drifted'] or 0)

    with col4:
        st.metric("Taille Dataset", latest['current_dataset_size'])

    if latest['drift_detected']:
        st.warning(f"⚠️ Drift détecté ! {latest['n_features_drifted']} features affectées")

        if latest['drifted_features']:
            st.subheader("Features avec Drift Significatif")
            for feature in latest['drifted_features']:
                st.markdown(f"- `{feature}`")
    else:
        st.success("✅ Pas de drift détecté sur la période")

    st.markdown("---")

    # Section 2 : Affichage du rapport HTML
    st.header("📊 Rapport Evidently AI")

    html_path = Path(latest['report_html_path'])

    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Afficher dans un iframe
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.error(f"⚠️ Rapport HTML introuvable : {html_path}")

    st.markdown("---")

    # Section 3 : Historique des scores
    st.header("📈 Historique des Scores de Drift")

    if len(df) > 1:
        fig_hist = px.line(
            df,
            x='generated_at',
            y='drift_score',
            title="Évolution du Score de Drift",
            labels={'generated_at': 'Date', 'drift_score': 'Score de drift'},
            markers=True
        )
        fig_hist.add_hline(
            y=0.2,
            line_dash="dash",
            annotation_text="Seuil d'alerte",
            line_color="red"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("ℹ️ Pas assez de rapports pour afficher l'historique")

    st.markdown("---")

    # Section 4 : Tableau des rapports
    st.header("📋 Historique des Rapports")

    df_display = df.copy()
    df_display['generated_at'] = df_display['generated_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_display['drift_detected'] = df_display['drift_detected'].apply(lambda x: "⚠️ OUI" if x else "✅ NON")

    st.dataframe(
        df_display[[
            'generated_at', 'drift_detected', 'drift_score',
            'n_features_drifted', 'current_dataset_size'
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "drift_score": st.column_config.NumberColumn(
                "Score Drift",
                format="%.3f"
            )
        }
    )

    storage.close()

except Exception as e:
    st.error(f"❌ Erreur lors du chargement des données : {str(e)}")

# Footer
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
