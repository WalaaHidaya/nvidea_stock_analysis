"""
Streamlit Application for NVIDIA Stock Analysis with LLM Integration
This application demonstrates the use of Generative AI (LLM) for:
1. Generating model hypotheses from descriptive statistics
2. Explaining results in simplified manner
3. Generating investment recommendations
4. Comparing human vs AI explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import json
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="NVDA Stock Analysis with LLM",
    page_icon="📈",
    layout="wide"
)

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.error("GROQ_API_KEY not found in .env file!")
        st.stop()
    return Groq(api_key=api_key)

client = get_groq_client()

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("NVDA.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[data['Date'] >= '2020-01-02'].reset_index(drop=True)
    return data

def call_llm(prompt, temperature=0.6):
    """Call Groq LLM with streaming"""
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def get_descriptive_statistics(data):
    """Calculate descriptive statistics for the stock data"""
    stats = {
        'mean': data['Adj Close'].mean(),
        'std': data['Adj Close'].std(),
        'min': data['Adj Close'].min(),
        'max': data['Adj Close'].max(),
        'median': data['Adj Close'].median(),
        'returns_mean': data['Adj Close'].pct_change().mean(),
        'returns_std': data['Adj Close'].pct_change().std(),
        'skewness': data['Adj Close'].skew(),
        'kurtosis': data['Adj Close'].kurtosis(),
        'total_points': len(data),
        'date_range': f"{data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}"
    }
    return stats

# Main app
def main():
    st.title("NVIDIA Stock Analysis with Generative AI (LLM)")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis:",
        ["Data Overview", 
         "LLM: Model Hypotheses", 
         "LLM: Explain Results", 
         "LLM: Investment Recommendations",
         "Human vs AI Comparison",
         "Notebook Results (LSTM/GRU)"]
    )
    
    # Load data
    data = load_data()
    stats = get_descriptive_statistics(data)
    
    if page == "Data Overview":
        show_data_overview(data, stats)
    elif page == "LLM: Model Hypotheses":
        show_model_hypotheses(data, stats)
    elif page == "LLM: Explain Results":
        show_results_explanation(data, stats)
    elif page == "LLM: Investment Recommendations":
        show_investment_recommendations(data, stats)
    elif page == "Human vs AI Comparison":
        show_comparison(data, stats)
    elif page == "Notebook Results (LSTM/GRU)":
        show_notebook_results(data, stats)

def show_data_overview(data, stats):
    """Display data overview and statistics"""
    st.header("NVIDIA Stock Data Overview")
    
    # Create two columns: left for metrics, right for chart
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("Key Metrics")
        st.metric("Average Price", f"${stats['mean']:.2f}")
        st.metric("Min Price", f"${stats['min']:.2f}")
        st.metric("Max Price", f"${stats['max']:.2f}")
        st.metric("Std Dev", f"${stats['std']:.2f}")
    
    with col_right:
        st.subheader("Price Evolution")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(data['Date'], data['Adj Close'], linewidth=2)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Adjusted Close Price ($)', fontsize=12)
        ax.set_title('NVIDIA Stock Price Evolution (2020-Present)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.subheader("Descriptive Statistics")
    stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', 'Skewness', 'Kurtosis', 
                   'Avg Daily Return', 'Return Volatility', 'Total Data Points', 'Date Range'],
        'Value': [
            f"${stats['mean']:.2f}",
            f"${stats['std']:.2f}",
            f"${stats['min']:.2f}",
            f"${stats['max']:.2f}",
            f"${stats['median']:.2f}",
            f"{stats['skewness']:.4f}",
            f"{stats['kurtosis']:.4f}",
            f"{stats['returns_mean']*100:.4f}%",
            f"{stats['returns_std']*100:.4f}%",
            f"{stats['total_points']}",
            stats['date_range']
        ]
    })
    st.dataframe(stats_df, use_container_width=True)
    
    st.subheader("Recent Data Sample")
    st.dataframe(data.tail(10), use_container_width=True)

def show_model_hypotheses(data, stats):
    """Generate model hypotheses using LLM"""
    st.header("Générer des Hypothèses de Modèles (LLM)")
    
    st.markdown("""
    Cette section utilise un Large Language Model (LLM) pour analyser les statistiques descriptives 
    et générer automatiquement des hypothèses de modèles à tester.
    """)
    
    if st.button("Générer les Hypothèses", type="primary"):
        with st.spinner("Le LLM analyse les données et génère des hypothèses..."):
            # Prepare prompt
            prompt = f"""
Tu es un expert en analyse de séries temporelles financières. Analyse les statistiques descriptives suivantes pour l'action NVIDIA (NVDA) et génère des hypothèses de modèles à tester.

STATISTIQUES DESCRIPTIVES:
- Prix moyen: ${stats['mean']:.2f}
- Écart-type: ${stats['std']:.2f}
- Prix minimum: ${stats['min']:.2f}
- Prix maximum: ${stats['max']:.2f}
- Médiane: ${stats['median']:.2f}
- Asymétrie (Skewness): {stats['skewness']:.4f}
- Aplatissement (Kurtosis): {stats['kurtosis']:.4f}
- Rendement quotidien moyen: {stats['returns_mean']*100:.4f}%
- Volatilité des rendements: {stats['returns_std']*100:.4f}%
- Nombre de points de données: {stats['total_points']}
- Période: {stats['date_range']}

TÂCHE:
Génère une analyse complète avec:
1. **Interprétation des statistiques**: Que nous disent ces chiffres sur le comportement de l'action NVDA?
2. **Hypothèses de modèles classiques**: Quels modèles ARIMA, SARIMA, VAR, ARCH/GARCH seraient appropriés? Justifie avec les statistiques.
3. **Hypothèses pour les réseaux de neurones**: Pourquoi LSTM et GRU seraient-ils adaptés? Quelle architecture recommandes-tu?
4. **Modèles hybrides**: Quelles combinaisons (ARIMA-LSTM, etc.) pourraient améliorer les prévisions?
5. **Variables exogènes**: Quelles variables externes pourraient être utiles?

Sois précis et technique, mais aussi accessible.
"""
            
            response = call_llm(prompt, temperature=0.7)
            
            st.success("Hypothèses générées avec succès!")
            st.markdown("### Analyse et Hypothèses du LLM")
            st.markdown(response)
            
            # Save to session state
            st.session_state['model_hypotheses'] = response
            
            # Download button
            st.download_button(
                label="Télécharger les hypothèses",
                data=response,
                file_name="hypotheses_modeles_llm.txt",
                mime="text/plain"
            )

def show_results_explanation(data, stats):
    """Explain model results using LLM"""
    st.header("Explication des Résultats (LLM)")
    
    st.markdown("""
    Cette section utilise le LLM pour expliquer les résultats des modèles LSTM et GRU 
    de manière vulgarisée et accessible.
    """)
    
    # Display actual results from notebook
    st.info("""
    **Résultats des modèles depuis le notebook `lstm_gry.ipynb`:**
    Les valeurs ci-dessous proviennent de l'exécution du notebook. Vous pouvez les modifier si nécessaire.
    """)
    
    # User can input actual results (with real values from notebook as defaults)
    st.subheader("Résultats des Modèles LSTM & GRU")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lstm_rmse = st.number_input("LSTM RMSE ($)", value=7.39, step=0.01)
        gru_rmse = st.number_input("GRU RMSE ($)", value=5.14, step=0.01)
    
    with col2:
        lstm_mae = st.number_input("LSTM MAE ($)", value=5.93, step=0.01)
        gru_mae = st.number_input("GRU MAE ($)", value=4.00, step=0.01)
    
    with col3:
        lstm_mape = st.number_input("LSTM MAPE (%)", value=4.64, step=0.01)
        gru_mape = st.number_input("GRU MAPE (%)", value=3.10, step=0.01)
    
    if st.button("Expliquer les Résultats", type="primary"):
        with st.spinner("Le LLM génère une explication vulgarisée..."):
            prompt = f"""
Tu es un expert en machine learning et finance qui doit expliquer des résultats techniques à un public non-expert.

CONTEXTE:
Nous avons entraîné deux modèles de réseaux de neurones récurrents (LSTM et GRU) pour prédire le prix de l'action NVIDIA.

RÉSULTATS:
**Modèle LSTM:**
- RMSE: ${lstm_rmse:.2f}
- MAE: ${lstm_mae:.2f}
- MAPE: {lstm_mape:.2f}%

**Modèle GRU:**
- RMSE: ${gru_rmse:.2f}
- MAE: ${gru_mae:.2f}
- MAPE: {gru_mape:.2f}%

TÂCHE:
Crée un rapport automatisé et vulgarisé qui explique:
1. **Qu'est-ce que LSTM et GRU?** (explication simple, avec analogies)
2. **Que signifient ces métriques?** (RMSE, MAE, MAPE) - explique en termes simples
3. **Quel modèle est le meilleur?** Pourquoi?
4. **Que signifie concrètement une erreur de ${lstm_rmse:.2f}?** 
5. **Ces résultats sont-ils bons?** Contextualise avec le prix moyen de ${stats['mean']:.2f}
6. **Limites et précautions**: Que doit-on comprendre sur ces prédictions?

Utilise des analogies, des exemples concrets, et évite le jargon technique autant que possible.
Formate avec des sections claires et des émojis pour la lisibilité.
"""
            
            response = call_llm(prompt, temperature=0.6)
            
            st.success("Explication générée avec succès!")
            st.markdown("### Explication Vulgarisée des Résultats")
            st.markdown(response)
            
            st.session_state['results_explanation'] = response
            
            st.download_button(
                label="Télécharger l'explication",
                data=response,
                file_name="explication_resultats_llm.txt",
                mime="text/plain"
            )

def show_investment_recommendations(data, stats):
    """Generate investment recommendations using LLM"""
    st.header("Recommandations d'Investissement (LLM)")
    
    st.markdown("""
    Cette section utilise le LLM pour générer des recommandations d'investissement simulées 
    basées sur l'analyse des modèles, avec une explicitation claire des limites et risques.
    """)
    
    st.warning("**AVERTISSEMENT**: Ces recommandations sont générées par une IA à des fins éducatives uniquement. Ne constituent pas des conseils financiers.")
    
    # Investment parameters
    st.subheader("Paramètres d'Investissement")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        investment_horizon = st.selectbox("Horizon d'investissement", 
                                         ["Court terme (1-3 mois)", 
                                          "Moyen terme (3-12 mois)", 
                                          "Long terme (1-5 ans)"])
    with col2:
        risk_profile = st.selectbox("Profil de risque", 
                                   ["Conservateur", "Modéré", "Agressif"])
    with col3:
        capital = st.number_input("Capital disponible ($)", 
                                 min_value=1000, value=10000, step=1000)
    
    if st.button("Générer les Recommandations", type="primary"):
        with st.spinner("Le LLM génère des recommandations d'investissement..."):
            current_price = data['Adj Close'].iloc[-1]
            price_change_1m = ((data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[-30]) - 1) * 100
            price_change_6m = ((data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[-180]) - 1) * 100
            
            prompt = f"""
Tu es un conseiller financier expert qui doit générer des recommandations d'investissement basées sur l'analyse de l'action NVIDIA.

DONNÉES DU MARCHÉ:
- Prix actuel: ${current_price:.2f}
- Variation 1 mois: {price_change_1m:.2f}%
- Variation 6 mois: {price_change_6m:.2f}%
- Prix moyen (période): ${stats['mean']:.2f}
- Volatilité: ${stats['std']:.2f}

PERFORMANCE DES MODÈLES:
- Les modèles LSTM et GRU ont été entraînés avec succès
- Erreur de prédiction moyenne: ~3-4%
- Les modèles montrent une bonne capacité à capturer les tendances

PROFIL CLIENT:
- Horizon d'investissement: {investment_horizon}
- Profil de risque: {risk_profile}
- Capital disponible: ${capital:,.2f}

TÂCHE:
Génère un rapport de recommandation complet incluant:

1. **ANALYSE DE LA SITUATION ACTUELLE**
   - Position actuelle de NVIDIA sur le marché
   - Tendances récentes observées
   
2. **RECOMMANDATIONS SIMULÉES**
   - Faut-il acheter, vendre, ou conserver?
   - Montant suggéré à investir (basé sur le profil de risque)
   - Points d'entrée suggérés
   - Objectifs de prix à court/moyen/long terme
   - Niveaux de stop-loss recommandés
   
3. **STRATÉGIE PROPOSÉE**
   - Allocation du capital
   - Horizon de temps recommandé
   - Stratégie de diversification
   
4. **LIMITES ET RISQUES (TRÈS IMPORTANT)**
   - Limites des modèles de prédiction
   - Risques spécifiques à NVIDIA
   - Risques du secteur technologique
   - Incertitudes macroéconomiques
   - Pourquoi ces recommandations ne sont pas des garanties
   
5. **DISCLAIMER**
   - Rappel que ce sont des simulations éducatives
   - Importance de consulter un vrai conseiller financier
   - Risque de perte en capital

Sois honnête sur les incertitudes et les limites. La section sur les risques doit être aussi développée que les recommandations elles-mêmes.
"""
            
            response = call_llm(prompt, temperature=0.6)
            
            st.success("Recommandations générées avec succès!")
            st.markdown("### Recommandations d'Investissement Simulées")
            st.markdown(response)
            
            st.session_state['investment_recommendations'] = response
            
            # Additional warning
            st.error("""
            **RAPPEL IMPORTANT**: 
            - Ces recommandations sont générées par une IA à des fins éducatives
            - Ne constituent en AUCUN CAS des conseils financiers professionnels
            - Les investissements comportent des risques de perte en capital
            - Consultez toujours un conseiller financier agréé avant d'investir
            """)
            
            st.download_button(
                label="Télécharger les recommandations",
                data=response,
                file_name="recommandations_investissement_llm.txt",
                mime="text/plain"
            )

def show_comparison(data, stats):
    """Compare human explanations vs AI-generated explanations"""
    st.header("Comparaison: Explications Humaines vs IA")
    
    st.markdown("""
    Cette section compare la cohérence entre les explications rédigées par des humains 
    et celles générées par l'IA pour évaluer la qualité et la fiabilité du LLM.
    """)
    
    # Human expert explanation
    st.subheader("Explication Humaine (Expert)")
    human_explanation = st.text_area(
        "Entrez l'explication d'un expert humain:",
        value="""Les modèles LSTM et GRU sont des réseaux de neurones récurrents conçus pour analyser des séquences de données temporelles. 
Pour NVIDIA, nous observons que:

1. **Performance des modèles**: Les deux modèles montrent des performances similaires avec des erreurs de prédiction autour de 3-4%. Le GRU est légèrement meilleur avec un RMSE plus faible.

2. **Interprétation**: Ces résultats sont satisfaisants étant donné la volatilité naturelle des marchés boursiers. Une erreur moyenne de $12-15 sur un prix moyen de $400+ représente une précision de ~96%.

3. **Limites**: Ces modèles ne capturent pas les événements imprévisibles (annonces d'entreprise, crises géopolitiques). Ils se basent uniquement sur les patterns historiques.

4. **Recommandations**: Utiliser ces prédictions comme un outil parmi d'autres dans une stratégie d'investissement diversifiée. Ne jamais se fier uniquement aux modèles algorithmiques.""",
        height=300
    )
    
    # Generate AI explanation
    if st.button("Générer l'Explication IA pour Comparaison", type="primary"):
        with st.spinner("Génération de l'explication IA..."):
            prompt = f"""
Tu es un expert en machine learning appliqué à la finance. Explique les résultats de modèles LSTM et GRU entraînés sur l'action NVIDIA.

CONTEXTE:
- Les modèles ont été entraînés sur des données de 2020 à aujourd'hui
- RMSE: ~$12-15
- MAE: ~$11-13  
- MAPE: ~3-4%
- Prix moyen de l'action: ${stats['mean']:.2f}

Explique en environ 200-250 mots:
1. La performance des modèles
2. Ce que signifient ces métriques
3. Les limites de ces prédictions
4. Comment utiliser ces résultats

Sois clair, précis et professionnel.
"""
            
            ai_explanation = call_llm(prompt, temperature=0.5)
            
            st.subheader("Explication Générée par l'IA")
            st.markdown(ai_explanation)
            
            # Analysis of coherence
            st.subheader("Analyse de Cohérence")
            
            analysis_prompt = f"""
Compare ces deux explications et analyse leur cohérence:

**Explication Humaine:**
{human_explanation}

**Explication IA:**
{ai_explanation}

Analyse:
1. **Points de convergence**: Quels éléments sont similaires?
2. **Différences notables**: Où les explications divergent-elles?
3. **Complémentarité**: Chaque explication apporte-t-elle des perspectives uniques?
4. **Qualité technique**: Laquelle est plus précise techniquement?
5. **Accessibilité**: Laquelle est plus facile à comprendre pour un non-expert?
6. **Score de cohérence**: Sur 10, quelle cohérence entre les deux?

Sois objectif et analytique.
"""
            
            with st.spinner("Analyse de la cohérence..."):
                coherence_analysis = call_llm(analysis_prompt, temperature=0.5)
            
            st.markdown("### Analyse Détaillée de la Cohérence")
            st.markdown(coherence_analysis)
            
            # Visualization
            st.subheader("Métriques de Comparaison")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Longueur Explication Humaine", f"{len(human_explanation.split())} mots")
            with col2:
                st.metric("Longueur Explication IA", f"{len(ai_explanation.split())} mots")
            with col3:
                # Simple similarity metric (word overlap)
                human_words = set(human_explanation.lower().split())
                ai_words = set(ai_explanation.lower().split())
                similarity = len(human_words & ai_words) / len(human_words | ai_words) * 100
                st.metric("Similarité Lexicale", f"{similarity:.1f}%")
            
            # Save comparison
            comparison_report = f"""
=== COMPARAISON EXPLICATIONS HUMAINES VS IA ===
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXPLICATION HUMAINE:
{human_explanation}

EXPLICATION IA:
{ai_explanation}

ANALYSE DE COHÉRENCE:
{coherence_analysis}
"""
            
            st.download_button(
                label="Télécharger le Rapport de Comparaison",
                data=comparison_report,
                file_name="comparaison_humain_vs_ia.txt",
                mime="text/plain"
            )

def show_notebook_results(data, stats):
    """Display outputs and visualizations from the Jupyter notebook"""
    st.header("Notebook Results: LSTM & GRU Models")
    
    st.markdown("""
    Cette section présente les résultats des modèles LSTM et GRU entraînés dans le notebook Jupyter.
    """)
    
    # Information about the notebook
    st.info("""
    **Note**: Les résultats ci-dessous proviennent de l'exécution du notebook `lstm_gry.ipynb`.  
    Pour voir les résultats actualisés, exécutez le notebook et entrez les métriques ici.
    """)
    
    # Model results input section
    st.subheader("Résultats des Modèles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Modèle LSTM**")
        lstm_rmse = st.number_input("LSTM RMSE ($)", value=0.0, step=0.01, key="lstm_rmse_nb")
        lstm_mae = st.number_input("LSTM MAE ($)", value=0.0, step=0.01, key="lstm_mae_nb")
        lstm_mape = st.number_input("LSTM MAPE (%)", value=0.0, step=0.01, key="lstm_mape_nb")
    
    with col2:
        st.markdown("**Modèle GRU**")
        gru_rmse = st.number_input("GRU RMSE ($)", value=0.0, step=0.01, key="gru_rmse_nb")
        gru_mae = st.number_input("GRU MAE ($)", value=0.0, step=0.01, key="gru_mae_nb")
        gru_mape = st.number_input("GRU MAPE (%)", value=0.0, step=0.01, key="gru_mape_nb")
    
    # Display comparison if values are entered
    if lstm_rmse > 0 or gru_rmse > 0:
        st.subheader("Comparaison des Modèles")
        
        comparison_df = pd.DataFrame({
            'Modèle': ['LSTM', 'GRU'],
            'RMSE ($)': [lstm_rmse, gru_rmse],
            'MAE ($)': [lstm_mae, gru_mae],
            'MAPE (%)': [lstm_mape, gru_mape]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Determine best model
        if lstm_rmse > 0 and gru_rmse > 0:
            best_model = 'LSTM' if lstm_rmse < gru_rmse else 'GRU'
            best_rmse = min(lstm_rmse, gru_rmse)
            st.success(f"**Meilleur modèle**: {best_model} (RMSE: ${best_rmse:.2f})")
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        metrics = ['RMSE ($)', 'MAE ($)', 'MAPE (%)']
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = comparison_df[metric].values
            bars = ax.bar(comparison_df['Modèle'], values, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
            ax.set_title(f'Comparaison par {metric}', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Training configuration section
    st.subheader("Configuration de l'Entraînement")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown("""
        **Données**
        - Période: 2020-01-02 à aujourd'hui
        - Split: 90% train / 10% test
        - Normalisation: MinMaxScaler (0, 1)
        """)
    
    with config_col2:
        st.markdown("""
        **Architecture**
        - Séquences: 30 jours
        - LSTM: 2 couches (50 unités)
        - GRU: 2 couches (50 unités)
        - Dropout: 20%
        """)
    
    with config_col3:
        st.markdown("""
        **Entraînement**
        - Epochs: 50 (max)
        - Batch size: 32
        - Validation: 10%
        - Early stopping: patience 10
        """)
    
    # Notebook link section
    st.subheader("Accéder au Notebook Complet")
    
    st.markdown("""
    Pour voir tous les détails, graphiques et analyses complètes:  
    **Fichier**: `lstm_gry.ipynb`
    
    Le notebook contient:
    - Préparation et normalisation des données
    - Construction et entraînement des modèles LSTM et GRU
    - Visualisations: 
      - Historique d'entraînement (loss)
      - Prédictions vs valeurs réelles
      - Distribution des erreurs
      - Graphiques de comparaison
    - Métriques détaillées (RMSE, MAE, MAPE)
    - Analyse comparative des performances
    """)
    
    # Instructions
    st.info("""
    **Pour exécuter le notebook**:  
    1. Ouvrez `lstm_gry.ipynb` dans Jupyter Notebook ou VS Code  
    2. Exécutez toutes les cellules  
    3. Copiez les valeurs RMSE, MAE, MAPE obtenues  
    4. Entrez-les dans les champs ci-dessus pour voir la comparaison
    """)
    
    # Summary statistics from the data
    st.subheader("Statistiques des Données NVIDIA")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Prix Moyen", f"${stats['mean']:.2f}")
    with col2:
        st.metric("Prix Min", f"${stats['min']:.2f}")
    with col3:
        st.metric("Prix Max", f"${stats['max']:.2f}")
    with col4:
        st.metric("Écart-type", f"${stats['std']:.2f}")
    with col5:
        st.metric("Total Points", stats['total_points'])
    
    # Recent price trend
    st.subheader("Tendance Récente des Prix")
    
    # Show last 60 days
    recent_data = data.tail(60)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent_data['Date'], recent_data['Adj Close'], linewidth=2, color='#2E86AB')
    ax.fill_between(recent_data['Date'], recent_data['Adj Close'], alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Prix Ajusté de Clôture ($)', fontsize=11)
    ax.set_title('Évolution du Prix NVIDIA (60 derniers jours)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
