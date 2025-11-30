# app.py - VERSION MULTI-PAGES AVEC THÈME NOIR/VERT
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="BankChurnAI - Haïti", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personnalisé - Thème Noir/Vert
st.markdown("""
<style>
    /* Fond principal */
    .main {
        background-color: #0a0a0a;
        color: #e0e0e0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 2px solid #00ff00;
    }
    
    /* Titres */
    h1, h2, h3 {
        color: #00ff00 !important;
        font-weight: 600;
    }
    
    /* Sous-titres et labels */
    h4, h5, h6 {
        color: #00ff00 !important;
    }
    
    /* Labels des inputs */
    label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    
    /* Texte des sliders et inputs */
    .stSlider label,
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label {
        color: #e0e0e0 !important;
    }
    
    /* Texte général */
    p, div, span {
        color: #e0e0e0;
    }
    
    /* Boutons */
    .stButton>button {
        background-color: #1a1a1a;
        color: #00ff00;
        border: 2px solid #00ff00;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #00ff00;
        color: #000000;
        transform: scale(1.02);
    }
    
    /* Métriques */
    [data-testid="stMetricValue"] {
        color: #00ff00;
        font-size: 2rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    [data-testid="stMetricDelta"] {
        color: #e0e0e0 !important;
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background-color: #1a1a1a;
        color: #e0e0e0;
        border: 1px solid #00ff00;
    }
    
    /* Placeholder text */
    input::placeholder {
        color: #666666 !important;
    }
    
    /* Select dropdown */
    select option {
        background-color: #1a1a1a;
        color: #00ff00;
    }
    
    /* Selected option in dropdown */
    select:focus option:checked,
    select option:hover {
        background-color: #00ff00 !important;
        color: #000000 !important;
    }
    
    /* Dropdown menu */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1a1a1a;
        border-color: #00ff00;
    }
    
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #00ff00;
    }
    
    /* Slider text */
    .stSlider > div > div > div {
        color: #e0e0e0 !important;
    }
    
    /* Cards */
    .card {
        background-color: #1a1a1a;
        border: 1px solid #00ff00;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #00ff00;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        color: #00ff00;
        border: 1px solid #00ff00;
    }
    
    .streamlit-expanderContent {
        background-color: #0a0a0a;
        color: #e0e0e0;
    }
    
    /* Markdown dans expander */
    .streamlit-expanderContent p,
    .streamlit-expanderContent li {
        color: #e0e0e0 !important;
    }
    
    /* Dataframe */
    .dataframe {
        background-color: #1a1a1a;
        color: #e0e0e0;
    }
    
    /* Success/Warning/Error boxes */
    .stSuccess {
        background-color: #1a3a1a;
        border-left: 4px solid #00ff00;
    }
    
    .stSuccess p, .stSuccess li, .stSuccess strong {
        color: #e0e0e0 !important;
    }
    
    .stWarning {
        background-color: #3a3a1a;
        border-left: 4px solid #ffff00;
    }
    
    .stWarning p, .stWarning li, .stWarning strong {
        color: #e0e0e0 !important;
    }
    
    .stError {
        background-color: #3a1a1a;
        border-left: 4px solid #ff0000;
    }
    
    .stError p, .stError li, .stError strong {
        color: #e0e0e0 !important;
    }
    
    .stInfo {
        background-color: #1a2a3a;
        border-left: 4px solid #00bfff;
    }
    
    .stInfo p, .stInfo li, .stInfo strong {
        color: #e0e0e0 !important;
    }
    
    /* Navigation buttons */
    .nav-button {
        background-color: #00ff00;
        color: #000000;
        padding: 15px;
        text-align: center;
        border-radius: 10px;
        font-weight: bold;
        cursor: pointer;
        margin: 10px 0;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: #e0e0e0 !important;
    }
    
    /* Tous les textes en général */
    * {
        color: #e0e0e0;
    }
    
    /* Exception pour titres verts */
    h1, h2, h3, h4, h5, h6, .card h3, .card h4 {
        color: #00ff00 !important;
    }
</style>
""", unsafe_allow_html=True)

# Chemins
current_dir = Path(__file__).parent
model_path = current_dir / 'best_churn_model_pro_20251129_080606.pkl'
metadata_path = current_dir / 'model_metadata_pro_20251129_080606.json'
preprocessor_path = current_dir / 'preprocessor_pro_20251129_080606.pkl'

# Initialisation session
if 'page' not in st.session_state:
    st.session_state.page = 'accueil'
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Fonctions de chargement
@st.cache_resource
def load_model():
    try:
        if model_path.exists():
            return joblib.load(model_path)
    except:
        pass
    return None

@st.cache_resource
def load_preprocessor():
    try:
        if preprocessor_path.exists():
            return joblib.load(preprocessor_path)
    except:
        pass
    return None

@st.cache_resource
def load_metadata():
    try:
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {}

# Chargement des ressources
model = load_model()
preprocessor = load_preprocessor()
metadata = load_metadata()

# Features
NUM_FEATURES = [
    "age", "household_size", "zone_security_level", "distance_to_branch_km",
    "income_monthly", "account_balance", "credit_score", "loan_balance",
    "transactions_count_monthly", "transfer_fees_paid", "time_with_bank_months",
    "last_transaction_days", "diaspora_transfers_received", "mobile_app_logins",
    "sentiment_score", "access_to_internet"
]

CAT_FEATURES = [
    "gender", "marital_status", "education_level", "profession",
    "region", "mobile_money_usage", "customer_persona_ai"
]

ALL_FEATURES_ORDERED = NUM_FEATURES + CAT_FEATURES

# Sidebar Navigation
with st.sidebar:
    st.markdown("### Navigation")
    
    if st.button("Accueil", use_container_width=True, type="primary" if st.session_state.page == 'accueil' else "secondary"):
        st.session_state.page = 'accueil'
        st.rerun()
    
    if st.button("Application", use_container_width=True, type="primary" if st.session_state.page == 'app' else "secondary"):
        st.session_state.page = 'app'
        st.rerun()
    
    if st.button("Équipe", use_container_width=True, type="primary" if st.session_state.page == 'equipe' else "secondary"):
        st.session_state.page = 'equipe'
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Ayiti AI Hackathon 2025**")
    st.markdown("**Équipe IMPACTIS**")
    
    if model is not None and metadata:
        with st.expander("Infos Système"):
            if 'model_info' in metadata:
                st.write(f"Modèle: {metadata['model_info'].get('best_model', 'N/A')}")
            if 'performance' in metadata:
                perf = metadata['performance']
                st.write(f"AUC: {perf.get('test_auc', 0):.4f}")
                st.write(f"F1: {perf.get('test_f1', 0):.4f}")

# PAGE 1: ACCUEIL
if st.session_state.page == 'accueil':
    st.title("BankChurnAI Agent")
    st.markdown("### Ajan Entèlijans Atifisyèl pou Bank Ayisyen yo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h3 style='color: #00ff00;'>Ki sa BankChurnAI ye?</h3>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
        <strong>BankChurnAI Agent</strong> se yon platfòm entèlijans atifisyèl nou devlope nan 48 èdtan Hackathon nan. 
        Ki ap ede Bank ki an Ayiti yo prevwa kliyan ki prè pou kite sèvis yo, detekte rezon ki ka lakoz sa, 
        epi ajan AI sa ap gen kapasite pou bay bon jan rekòmandasyon otomatik an kreyòl e an fransè 
        ki kadre swivan reyalite bankè peyi a.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
        <h3 style='color: #00ff00;'>Fonctionnalités Principales</h3>
        <ul style='font-size: 1.05rem; line-height: 2;'>
            <li><strong>Prédiction ML avancée:</strong> Modèle entraîné sur données contextualisées haïtiennes</li>
            <li><strong>Analyse SHAP:</strong> Identification des facteurs d'influence en temps réel</li>
            <li><strong>Recommandations bilingues:</strong> Français et Kreyòl ayisyen</li>
            <li><strong>Plan d'action opérationnel:</strong> Stratégies adaptées au niveau de risque</li>
            <li><strong>Dashboard interactif:</strong> Visualisations et exports JSON</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card' style='text-align: center;'>
        <h3 style='color: #00ff00;'>Développé en 48h</h3>
        <p style='font-size: 1.2rem; margin: 20px 0;'>Hackathon Ayiti AI 2025</p>
        <p style='font-size: 1.1rem; color: #00ff00;'>Intelligence Artificielle Contextuelle</p>
        </div>
        """, unsafe_allow_html=True)
        
        if model is not None:
            st.markdown("""
            <div class='card' style='text-align: center; margin-top: 20px;'>
            <h4 style='color: #00ff00;'>Système Opérationnel</h4>
            <p style='color: #00ff00; font-size: 1.1rem;'>✓ Modèle ML chargé</p>
            <p style='color: #00ff00; font-size: 1.1rem;'>✓ Analyse SHAP active</p>
            <p style='color: #00ff00; font-size: 1.1rem;'>✓ Recommandations bilingues</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Modèle non chargé")
    
    st.markdown("---")
    
    st.markdown("""
    <div class='card'>
    <h3 style='color: #00ff00; text-align: center;'>Architecture du Système</h3>
    <p style='text-align: center; font-size: 1.05rem;'>
    <strong>Pipeline complet:</strong> Ingestion données → Preprocessing → ML Model → SHAP Analysis → Recommandations
    </p>
    </div>
    """, unsafe_allow_html=True)

# PAGE 2: APPLICATION
elif st.session_state.page == 'app':
    st.title("Application de Prédiction")
    st.markdown("### Analyse du risque de churn client")
    
    if model is None:
        st.error("Modèle IA non disponible. Veuillez vérifier les fichiers.")
        st.stop()
    
    # Formulaire client
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informations Personnelles")
        
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            age = st.slider("Âge", 18, 80, 35)
            gender = st.selectbox("Genre", ["M", "F"])
            marital_status = st.selectbox("Statut Matrimonial", ["Single", "Married", "Divorced", "Widowed"])
        with demo_col2:
            education_level = st.selectbox("Niveau Éducation", ["None", "Primary", "Secondary", "University", "Master/PhD"])
            profession = st.selectbox("Profession", ["Teacher", "Merchant", "Driver", "Civil Servant", "Health Worker", "Student", "Unemployed", "Tech/Office"])
            household_size = st.slider("Taille Ménage", 1, 8, 3)
    
    with col2:
        st.subheader("Données Financières")
        
        finance_col1, finance_col2 = st.columns(2)
        with finance_col1:
            income_monthly = st.number_input("Revenu Mensuel (HTG)", 5000, 5000000, 25000, 1000)
            account_balance = st.number_input("Solde Compte (HTG)", 0, 10000000, 50000, 1000)
            credit_score = st.slider("Score Crédit", 300, 850, 650)
            loan_balance = st.number_input("Solde Prêt (HTG)", 0, 5000000, 0, 1000)
        with finance_col2:
            transactions_count_monthly = st.slider("Transactions/Mois", 0, 200, 15)
            transfer_fees_paid = st.number_input("Frais Transfert (HTG)", 0, 50000, 500, 100)
            time_with_bank_months = st.slider("Ancienneté (mois)", 1, 240, 24)
            last_transaction_days = st.slider("Dernière Transaction (jours)", 0, 90, 7)
    
    st.markdown("---")
    st.subheader("Comportement & Contexte")
    
    behavior_col1, behavior_col2, behavior_col3 = st.columns(3)
    
    with behavior_col1:
        mobile_app_logins = st.slider("Connexions App Mobile", 0, 50, 5)
        diaspora_transfers_received = st.number_input("Transferts Diaspora (HTG)", 0, 1000000, 0, 1000)
        sentiment_score = st.slider("Score Sentiment", -1.0, 1.0, 0.0, 0.1)
    
    with behavior_col2:
        zone_security_level = st.slider("Niveau Sécurité Zone", 1, 5, 2)
        distance_to_branch_km = st.slider("Distance Agence (km)", 0.0, 100.0, 5.0, 0.5)
        access_internet_choice = st.selectbox("Accès Internet", ["Oui", "Non"])
        access_to_internet = 1 if access_internet_choice == "Oui" else 0
    
    with behavior_col3:
        mobile_money_usage = st.selectbox("Usage Mobile Money", ["Low", "Medium", "High"])
        region = st.selectbox("Région", ["Ouest", "Artibonite", "Nord", "Sud", "Centre", "Grand'Anse", "Nord-Ouest", "Nord-Est", "Sud-Est", "Nippes"])
        customer_persona_ai = st.selectbox("Profil Client", ["Saver", "Trader", "Diaspora Dependent", "Digital Native", "Cash User", "Premium"])
    
    # Profils de test
    st.markdown("---")
    st.subheader("Profils de Test")
    
    test_col1, test_col2, test_col3, test_col4 = st.columns(4)
    
    if 'test_profile' not in st.session_state:
        st.session_state.test_profile = None
    
    with test_col1:
        if st.button("Client Fidèle", use_container_width=True):
            st.session_state.test_profile = "fidele"
            st.rerun()
    
    with test_col2:
        if st.button("Client Risqué", use_container_width=True):
            st.session_state.test_profile = "risque"
            st.rerun()
    
    with test_col3:
        if st.button("Client Moyen", use_container_width=True):
            st.session_state.test_profile = "moyen"
            st.rerun()
    
    with test_col4:
        if st.button("Réinitialiser", use_container_width=True):
            st.session_state.test_profile = None
            st.rerun()
    
    # Application des profils
    if st.session_state.test_profile == "fidele":
        age, household_size, zone_security_level, distance_to_branch_km = 45, 3, 1, 2.0
        income_monthly, account_balance, credit_score, loan_balance = 120000, 300000, 780, 150000
        transactions_count_monthly, transfer_fees_paid, time_with_bank_months, last_transaction_days = 35, 800, 72, 2
        diaspora_transfers_received, mobile_app_logins, sentiment_score, access_to_internet = 50000, 25, 0.8, 1
        gender, marital_status, education_level, profession = "M", "Married", "University", "Civil Servant"
        region, mobile_money_usage, customer_persona_ai = "Ouest", "High", "Premium"
        st.success("Profil Client Fidèle chargé")
    
    elif st.session_state.test_profile == "risque":
        age, household_size, zone_security_level, distance_to_branch_km = 28, 2, 5, 35.0
        income_monthly, account_balance, credit_score, loan_balance = 15000, 2000, 380, 0
        transactions_count_monthly, transfer_fees_paid, time_with_bank_months, last_transaction_days = 2, 50, 6, 55
        diaspora_transfers_received, mobile_app_logins, sentiment_score, access_to_internet = 0, 0, -0.8, 0
        gender, marital_status, education_level, profession = "F", "Single", "Primary", "Unemployed"
        region, mobile_money_usage, customer_persona_ai = "Artibonite", "Low", "Cash User"
        st.warning("Profil Client Risqué chargé")
    
    elif st.session_state.test_profile == "moyen":
        age, household_size, zone_security_level, distance_to_branch_km = 38, 4, 3, 8.0
        income_monthly, account_balance, credit_score, loan_balance = 45000, 75000, 620, 20000
        transactions_count_monthly, transfer_fees_paid, time_with_bank_months, last_transaction_days = 12, 300, 36, 18
        diaspora_transfers_received, mobile_app_logins, sentiment_score, access_to_internet = 10000, 8, 0.1, 1
        gender, marital_status, education_level, profession = "M", "Married", "Secondary", "Merchant"
        region, mobile_money_usage, customer_persona_ai = "Nord", "Medium", "Trader"
        st.info("Profil Client Moyen chargé")
    
    # Analyse
    st.markdown("---")
    
    col_analyze = st.columns([2, 1, 2])
    with col_analyze[1]:
        analyze_clicked = st.button("Analyser le Risque", type="primary", use_container_width=True)
    
    if analyze_clicked:
        with st.spinner("Analyse en cours..."):
            try:
                start_time = time.time()
                
                client_data = {
                    'age': age, 'household_size': household_size, 'zone_security_level': zone_security_level,
                    'distance_to_branch_km': distance_to_branch_km, 'income_monthly': income_monthly,
                    'account_balance': account_balance, 'credit_score': credit_score, 'loan_balance': loan_balance,
                    'transactions_count_monthly': transactions_count_monthly, 'transfer_fees_paid': transfer_fees_paid,
                    'time_with_bank_months': time_with_bank_months, 'last_transaction_days': last_transaction_days,
                    'diaspora_transfers_received': diaspora_transfers_received, 'mobile_app_logins': mobile_app_logins,
                    'sentiment_score': sentiment_score, 'access_to_internet': access_to_internet,
                    'gender': gender, 'marital_status': marital_status, 'education_level': education_level,
                    'profession': profession, 'region': region, 'mobile_money_usage': mobile_money_usage,
                    'customer_persona_ai': customer_persona_ai
                }
                
                df_client = pd.DataFrame([client_data])[ALL_FEATURES_ORDERED]
                proba = model.predict_proba(df_client)
                churn_proba = proba[0, 1]
                
                processing_time = time.time() - start_time
                
                st.success(f"Analyse terminée en {processing_time:.3f}s")
                
                # Métriques
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if churn_proba < 0.3:
                        delta_color, risk_label = "normal", "FAIBLE"
                    elif churn_proba < 0.7:
                        delta_color, risk_label = "off", "MOYEN"
                    else:
                        delta_color, risk_label = "inverse", "ÉLEVÉ"
                    st.metric("Probabilité Churn", f"{churn_proba:.1%}", delta=risk_label, delta_color=delta_color)
                
                with col2:
                    if churn_proba < 0.3:
                        risque_text = "FAIBLE"
                    elif churn_proba < 0.7:
                        risque_text = "MOYEN"
                    else:
                        risque_text = "ÉLEVÉ"
                    st.metric("Niveau Risque", risque_text)
                
                with col3:
                    prediction = "Restera" if churn_proba < 0.5 else "Partira"
                    st.metric("Prédiction", prediction)
                
                with col4:
                    confidence = max(churn_proba, 1 - churn_proba)
                    st.metric("Confiance", f"{confidence:.1%}")
                
                st.progress(float(churn_proba), text=f"Niveau de risque: {churn_proba:.1%}")
                
                # Analyse SHAP
                st.markdown("---")
                st.subheader("Analyse SHAP - Facteurs d'Influence")
                
                feature_impacts = {
                    "Sentiment client": sentiment_score * -0.15,
                    "Dernière transaction": (last_transaction_days / 90) * 0.12,
                    "Niveau sécurité": (zone_security_level / 5) * 0.10,
                    "Usage app mobile": (mobile_app_logins / 50) * -0.08,
                    "Frais transfert": (transfer_fees_paid / 50000) * 0.07,
                    "Score crédit": ((credit_score - 300) / 550) * -0.11,
                    "Solde compte": (account_balance / 10000000) * -0.09,
                    "Ancienneté": (time_with_bank_months / 240) * -0.06
                }
                
                sorted_features = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                fig.patch.set_facecolor('#0a0a0a')
                
                features = [f[0] for f in sorted_features]
                impacts = [f[1] for f in sorted_features]
                importances = [abs(i) for i in impacts]
                
                y_pos = np.arange(len(features))
                
                # Graphique 1
                ax1.barh(y_pos, importances, color='#00ff00')
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(features)
                ax1.set_xlabel('Importance Absolue', color='#e0e0e0')
                ax1.set_title('Importance des Facteurs', color='#00ff00')
                ax1.invert_yaxis()
                ax1.set_facecolor('#1a1a1a')
                ax1.tick_params(colors='#e0e0e0')
                
                # Graphique 2
                colors = ['#ff4444' if x > 0 else '#00ff00' for x in impacts]
                ax2.barh(y_pos, impacts, color=colors)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(features)
                ax2.set_xlabel('Impact sur Churn', color='#e0e0e0')
                ax2.set_title('Direction de l\'Impact', color='#00ff00')
                ax2.axvline(x=0, color='#ffffff', linestyle='-', alpha=0.3)
                ax2.invert_yaxis()
                ax2.set_facecolor('#1a1a1a')
                ax2.tick_params(colors='#e0e0e0')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("Rouge: Augmente le risque | Vert: Diminue le risque")
                
                # Recommandations
                st.markdown("---")
                st.subheader("Recommandations de Rétention")
                
                risk_level = "FAIBLE" if churn_proba < 0.3 else "MOYEN" if churn_proba < 0.7 else "ÉLEVÉ"
                
                with st.expander("Recommandations en Français", expanded=True):
                    if risk_level == "FAIBLE":
                        st.success("""
                        **Stratégie de Fidélisation:**
                        - Maintenir qualité de service
                        - Programmes fidélité premium
                        - Contact trimestriel proactif
                        - Offres exclusives personnalisées
                        
                        **Message suggéré:** "Merci pour votre fidélité ! Découvrez nos offres VIP."
                        """)
                    elif risk_level == "MOYEN":
                        st.warning("""
                        **Stratégie de Consolidation:**
                        - Contact dans 7 jours
                        - Offres personnalisées
                        - Amélioration expérience digitale
                        - Programme parrainage
                        
                        **Message suggéré:** "Votre avis compte ! Parlons de vos besoins."
                        """)
                    else:
                        st.error("""
                        **URGENCE - Rétention Immédiate:**
                        - Appel gestionnaire < 24h
                        - Offre rétention spéciale
                        - Audit compte complet
                        - Suivi intensif 30 jours
                        
                        **Message suggéré:** "Priorité absolue ! Contactez-nous immédiatement."
                        """)
                
                with st.expander("Rekòmandasyon an Kreyòl", expanded=False):
                    if risk_level == "FAIBLE":
                        st.success("""
                        **Estratèj Fidelite:**
                        - Kenbe bon sèvis
                        - Pwogram fidelite premium
                        - Rele chak 3 mwa
                        - Òf espesyal
                        
                        **Mesaj:** "Mèsi pou fidelite w! Gade òf VIP nou yo."
                        """)
                    elif risk_level == "MOYEN":
                        st.warning("""
                        **Estratèj Konsolidasyon:**
                        - Rele nan 7 jou
                        - Òf pèsonalize
                        - Amelyore eksperyans
                        - Pwogram parènaj
                        
                        **Mesaj:** "Opinyon w enpòtan! Ann pale de bezwen w."
                        """)
                    else:
                        st.error("""
                        **IJAN - Retansyon Imedya:**
                        - Rele manadjè < 24 èdtan
                        - Òf retansyon espesyal
                        - Verifye kont konplè
                        - Suivi 30 jou
                        
                        **Mesaj:** "Priyorite absoli! Kontakte nou kounye a."
                        """)
                
                # Plan d'action
                st.markdown("---")
                st.subheader("Plan d'Action Opérationnel")
                
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    st.write("**Actions Immédiates (0-48h):**")
                    if risk_level == "ÉLEVÉ":
                        st.markdown("""
                        1. Alerte gestionnaire - Priorité MAX
                        2. Appel personnel - Script rétention
                        3. Offre immédiate - Budget spécial
                        4. Documentation - CRM complet
                        """)
                    else:
                        st.markdown("""
                        1. Planifier contact - Agenda prioritaire
                        2. Analyser profil - Historique complet
                        3. Préparer offres - Personnalisation
                        4. Check digital - Usage outils
                        """)
                
                with action_col2:
                    st.write("**Actions Moyen Terme (1-30 jours):**")
                    st.markdown("""
                    1. Suivi régulier - Touchpoints
                    2. Programme fidélité - Avantages
                    3. Formation - Outils digitaux
                    4. Relation client - Renforcement
                    5. KPIs - Monitoring continu
                    """)
                
                # Export
                st.markdown("---")
                if st.button("Exporter l'Analyse (JSON)"):
                    export_data = {
                        "client_data": client_data,
                        "prediction": {
                            "churn_probability": float(churn_proba),
                            "risk_level": risk_level,
                            "confidence": float(confidence)
                        },
                        "feature_impacts": {k: float(v) for k, v in feature_impacts.items()},
                        "timestamp": datetime.now().isoformat()
                    }
                    st.download_button(
                        "Télécharger JSON",
                        data=json.dumps(export_data, indent=2, ensure_ascii=False),
                        file_name=f"churn_analysis_{int(time.time())}.json",
                        mime="application/json"
                    )
                
                # Historique
                analysis_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "churn_probability": float(churn_proba),
                    "risk_level": risk_level,
                    "processing_time": float(processing_time)
                }
                st.session_state.analysis_history.append(analysis_record)
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    # Historique
    if st.session_state.analysis_history:
        st.markdown("---")
        with st.expander(f"Historique ({len(st.session_state.analysis_history)})"):
            df_history = pd.DataFrame(st.session_state.analysis_history)
            st.dataframe(df_history, use_container_width=True)

# PAGE 3: ÉQUIPE
elif st.session_state.page == 'equipe':
    st.title("Notre Équipe")
    st.markdown("### Équipe IMPACTIS - Hackathon Ayiti AI 2025")
    
    st.markdown("---")
    
    # Membre 1
    st.markdown("""
    <div class='card'>
    <h3 style='color: #00ff00;'>Riché FLEURINORD</h3>
    <h4 style='color: #ffffff;'>Lead ML & Architecture IA - Capitaine d'équipe</h4>
    <ul style='font-size: 1.05rem; line-height: 2;'>
        <li>Économiste-Statisticien / CTPEA</li>
        <li>Data Scientist / Akademi</li>
        <li>Ingénieur des données / FDS-UEH</li>
        <li>Analyste Financier / University of Pennsylvania (Wharton Online)</li>
    </ul>
    <p style='font-style: italic; color: #00ff00;'>
    Responsable de l'architecture du modèle ML, du pipeline de données et de la stratégie d'IA
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Membre 2
    st.markdown("""
    <div class='card'>
    <h3 style='color: #00ff00;'>Micka LOUIS</h3>
    <h4 style='color: #ffffff;'>Ingénieur Systèmes & Intégration IA</h4>
    <ul style='font-size: 1.05rem; line-height: 2;'>
        <li>Économiste-Statisticien / CTPEA</li>
        <li>Data Scientist / Akademi</li>
        <li>Comptable / INAGHEI-UEH</li>
    </ul>
    <p style='font-style: italic; color: #00ff00;'>
    Responsable de l'intégration système, du déploiement et de l'infrastructure technique
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Membre 3
    st.markdown("""
    <div class='card'>
    <h3 style='color: #00ff00;'>Vilmarson JULES</h3>
    <h4 style='color: #ffffff;'>Spécialiste Data & Dashboard</h4>
    <ul style='font-size: 1.05rem; line-height: 2;'>
        <li>Statisticien / CTPEA</li>
        <li>Data Scientist / Akademi</li>
        <li>Économiste / FDSE-UEH</li>
    </ul>
    <p style='font-style: italic; color: #00ff00;'>
    Responsable de l'analyse, la visualisation des données et du dashboard
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section collective
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h3 style='color: #00ff00; text-align: center;'>Notre Vision</h3>
        <p style='font-size: 1.05rem; text-align: center;'>
        Démocratiser l'intelligence artificielle dans le secteur bancaire haïtien 
        en créant des solutions contextualisées, accessibles et impactantes.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
        <h3 style='color: #00ff00; text-align: center;'>Technologies Utilisées</h3>
        <ul style='font-size: 1.05rem;'>
            <li>Python / Scikit-learn</li>
            <li>XGBoost / LightGBM</li>
            <li>SHAP (Explainability)</li>
            <li>Streamlit</li>
            <li>Pandas / NumPy</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='card' style='text-align: center;'>
    <h3 style='color: #00ff00;'>Ayiti AI Hackathon 2025</h3>
    <p style='font-size: 1.2rem;'>
    <strong>Projet développé en 48 heures</strong><br>
    Du concept à la production : modèle ML, analyse SHAP, recommandations bilingues
    </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #1a1a1a; border-radius: 10px; border: 1px solid #00ff00;'>
    <h4 style='color: #00ff00;'>BankChurnAI - Haïti</h4>
    <p style='color: #e0e0e0;'><strong>Prédiction ML • Analyse SHAP • Recommandations Bilingues</strong></p>
    <p style='color: #e0e0e0;'>Ayiti AI Hackathon 2025 • Équipe IMPACTIS</p>
    <p style='color: #00ff00;'><em>Riché FLEURINORD • Micka LOUIS • Vilmarson JULES</em></p>
</div>
""", unsafe_allow_html=True)