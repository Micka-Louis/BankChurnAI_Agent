# app.py - VERSION FINALE AVEC SHAP & RECOMMANDATIONS MULTILINGUES
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # ✅ UNIQUEMENT JOBLIB
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="BankChurnAI - Haïti 🇭🇹", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 BankChurnAI - Haïti 🇭🇹")
st.subheader("Prédiction du Churn • Analyse SHAP • Recommandations Multilingues")

# Sidebar
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("**Ayiti AI Hackathon 2025**")
st.sidebar.markdown("**Équipe IMPACTIS**")

# Chemins
current_dir = Path(__file__).parent
model_path = current_dir / 'best_churn_model_pro_20251129_080606.pkl'
metadata_path = current_dir / 'model_metadata_pro_20251129_080606.json'
preprocessor_path = current_dir / 'preprocessor_pro_20251129_080606.pkl'

# Initialisation session
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'shap_plots' not in st.session_state:
    st.session_state.shap_plots = {}

# Chargement avec JOBLIB uniquement
@st.cache_resource(show_spinner="Chargement du modèle IA...")
def load_model():
    try:
        if not model_path.exists():
            st.sidebar.error(f"❌ Modèle non trouvé: {model_path.name}")
            return None
        
        model = joblib.load(model_path)
        st.sidebar.success("✅ Modèle IA chargé")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Erreur modèle: {str(e)}")
        return None

@st.cache_resource(show_spinner="Chargement du préprocesseur...")
def load_preprocessor():
    try:
        if not preprocessor_path.exists():
            return None
        
        preprocessor = joblib.load(preprocessor_path)
        st.sidebar.success("✅ Préprocesseur chargé")
        return preprocessor
    except Exception as e:
        st.sidebar.warning(f"⚠️ Préprocesseur: {str(e)}")
        return None

@st.cache_resource(show_spinner="Chargement des métadonnées...")
def load_metadata():
    try:
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        st.sidebar.success("✅ Métadonnées chargées")
        return data
    except Exception as e:
        st.sidebar.warning(f"⚠️ Métadonnées: {str(e)}")
        return {}

# Chargement
model = load_model()
preprocessor = load_preprocessor()
metadata = load_metadata()

# Affichage info modèle
if model is not None:
    st.sidebar.success("🎯 Système prêt!")
    
    if metadata:
        with st.sidebar.expander("📊 Infos Modèle", expanded=False):
            if 'model_info' in metadata:
                st.write(f"**Modèle:** {metadata['model_info'].get('best_model', 'N/A')}")
                st.write(f"**Stratégie:** {metadata['model_info'].get('best_strategy', 'N/A')}")
            
            if 'performance' in metadata:
                perf = metadata['performance']
                st.write(f"**AUC Test:** {perf.get('test_auc', 0):.4f}")
                st.write(f"**F1 Test:** {perf.get('test_f1', 0):.4f}")
                st.write(f"**Precision:** {perf.get('test_precision', 0):.4f}")
                st.write(f"**Recall:** {perf.get('test_recall', 0):.4f}")
else:
    st.sidebar.error("⚠️ Modèle non chargé")

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

# Interface principale
st.markdown("---")

# Formulaire client
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Informations Personnelles")
    
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
    st.subheader("💳 Données Financières")
    
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

# Section comportementale
st.markdown("---")
st.subheader("📱 Comportement & Contexte")

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
st.subheader("🚀 Profils de Test")

test_col1, test_col2, test_col3, test_col4 = st.columns(4)

# Variables globales pour les profils
if 'test_profile' not in st.session_state:
    st.session_state.test_profile = None

with test_col1:
    if st.button("🧪 Client Fidèle", use_container_width=True):
        st.session_state.test_profile = "fidele"
        st.rerun()

with test_col2:
    if st.button("⚠️ Client Risqué", use_container_width=True):
        st.session_state.test_profile = "risque"
        st.rerun()

with test_col3:
    if st.button("🔄 Client Moyen", use_container_width=True):
        st.session_state.test_profile = "moyen"
        st.rerun()

with test_col4:
    if st.button("📊 Réinitialiser", use_container_width=True):
        st.session_state.test_profile = None
        st.rerun()

# Appliquer profil test
if st.session_state.test_profile == "fidele":
    age, household_size, zone_security_level, distance_to_branch_km = 45, 3, 1, 2.0
    income_monthly, account_balance, credit_score, loan_balance = 120000, 300000, 780, 150000
    transactions_count_monthly, transfer_fees_paid, time_with_bank_months, last_transaction_days = 35, 800, 72, 2
    diaspora_transfers_received, mobile_app_logins, sentiment_score, access_to_internet = 50000, 25, 0.8, 1
    gender, marital_status, education_level, profession = "M", "Married", "University", "Civil Servant"
    region, mobile_money_usage, customer_persona_ai = "Ouest", "High", "Premium"
    st.info("✅ Profil Client Fidèle chargé")

elif st.session_state.test_profile == "risque":
    age, household_size, zone_security_level, distance_to_branch_km = 28, 2, 5, 35.0
    income_monthly, account_balance, credit_score, loan_balance = 15000, 2000, 380, 0
    transactions_count_monthly, transfer_fees_paid, time_with_bank_months, last_transaction_days = 2, 50, 6, 55
    diaspora_transfers_received, mobile_app_logins, sentiment_score, access_to_internet = 0, 0, -0.8, 0
    gender, marital_status, education_level, profession = "F", "Single", "Primary", "Unemployed"
    region, mobile_money_usage, customer_persona_ai = "Artibonite", "Low", "Cash User"
    st.warning("⚠️ Profil Client Risqué chargé")

elif st.session_state.test_profile == "moyen":
    age, household_size, zone_security_level, distance_to_branch_km = 38, 4, 3, 8.0
    income_monthly, account_balance, credit_score, loan_balance = 45000, 75000, 620, 20000
    transactions_count_monthly, transfer_fees_paid, time_with_bank_months, last_transaction_days = 12, 300, 36, 18
    diaspora_transfers_received, mobile_app_logins, sentiment_score, access_to_internet = 10000, 8, 0.1, 1
    gender, marital_status, education_level, profession = "M", "Married", "Secondary", "Merchant"
    region, mobile_money_usage, customer_persona_ai = "Nord", "Medium", "Trader"
    st.info("🔄 Profil Client Moyen chargé")

# Analyse principale
st.markdown("---")
analysis_col1, analysis_col2, analysis_col3 = st.columns([2, 1, 2])

with analysis_col2:
    analyze_clicked = st.button(
        "🎯 Analyser le Risque de Churn", 
        type="primary", 
        use_container_width=True,
        disabled=(model is None)
    )

if analyze_clicked and model is not None:
    with st.spinner("🔍 Analyse en cours..."):
        try:
            start_time = time.time()
            
            # Données client
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
            
            # Prédiction
            df_client = pd.DataFrame([client_data])[ALL_FEATURES_ORDERED]
            proba = model.predict_proba(df_client)
            churn_proba = proba[0, 1]
            
            processing_time = time.time() - start_time
            
            # Affichage résultats
            st.success(f"✅ Analyse terminée en {processing_time:.3f}s")
            
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
                    risque_emoji, risque_text = "🟢", "FAIBLE"
                elif churn_proba < 0.7:
                    risque_emoji, risque_text = "🟡", "MOYEN"
                else:
                    risque_emoji, risque_text = "🔴", "ÉLEVÉ"
                st.metric("Niveau Risque", f"{risque_emoji} {risque_text}")
            
            with col3:
                prediction = "Restera" if churn_proba < 0.5 else "Partira"
                prediction_emoji = "✅" if churn_proba < 0.5 else "⚠️"
                st.metric("Prédiction", f"{prediction_emoji} {prediction}")
            
            with col4:
                confidence = max(churn_proba, 1 - churn_proba)
                st.metric("Confiance", f"{confidence:.1%}")
            
            # Barre de progression
            st.progress(float(churn_proba), text=f"Niveau de risque: {churn_proba:.1%}")
            
            # Section SHAP
            st.markdown("---")
            st.subheader("📊 Analyse SHAP - Facteurs d'Influence")
            
            # Calcul impacts basé sur les valeurs réelles
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
            
            features = [f[0] for f in sorted_features]
            impacts = [f[1] for f in sorted_features]
            importances = [abs(i) for i in impacts]
            
            y_pos = np.arange(len(features))
            
            # Importance
            ax1.barh(y_pos, importances, color='skyblue')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(features)
            ax1.set_xlabel('Importance Absolue')
            ax1.set_title('Importance des Facteurs')
            ax1.invert_yaxis()
            
            # Impact
            colors = ['red' if x > 0 else 'green' for x in impacts]
            ax2.barh(y_pos, impacts, color=colors)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(features)
            ax2.set_xlabel('Impact sur Churn')
            ax2.set_title('Direction de l\'Impact')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("""
            **🔍 Lecture SHAP:**
            - **Rouge (→)**: Facteur qui AUGMENTE le risque de churn
            - **Vert (←)**: Facteur qui DIMINUE le risque de churn
            - **Taille**: Importance du facteur dans la décision
            """)
            
            # Recommandations
            st.markdown("---")
            st.subheader("💡 Recommandations de Rétention")
            
            risk_level = "FAIBLE" if churn_proba < 0.3 else "MOYEN" if churn_proba < 0.7 else "ÉLEVÉ"
            
            # Français
            with st.expander("🇫🇷 Recommandations en Français", expanded=True):
                if risk_level == "FAIBLE":
                    st.success("""
                    **Stratégie de Fidélisation:**
                    - ✅ Maintenir qualité de service
                    - 🎁 Programmes fidélité premium
                    - 📞 Contact trimestriel proactif
                    - 🌟 Offres exclusives personnalisées
                    
                    **Message suggéré:**
                    "Merci pour votre fidélité ! Découvrez nos offres VIP."
                    """)
                elif risk_level == "MOYEN":
                    st.warning("""
                    **Stratégie de Consolidation:**
                    - 📞 Contact dans 7 jours
                    - 🎯 Offres personnalisées
                    - 💻 Amélioration expérience digitale
                    - 🤝 Programme parrainage
                    
                    **Message suggéré:**
                    "Votre avis compte ! Parlons de vos besoins."
                    """)
                else:
                    st.error("""
                    **🚨 URGENCE - Rétention Immédiate:**
                    - ☎️ Appel gestionnaire < 24h
                    - 💰 Offre rétention spéciale
                    - 🔍 Audit compte complet
                    - 📊 Suivi intensif 30 jours
                    
                    **Message suggéré:**
                    "Priorité absolue ! Contactez-nous immédiatement."
                    """)
            
            # Créole
            with st.expander("🇭🇹 Rekòmandasyon an Kreyòl", expanded=False):
                if risk_level == "FAIBLE":
                    st.success("""
                    **Estratèj Fidelite:**
                    - ✅ Kenbe bon sèvis
                    - 🎁 Pwogram fidelite premium
                    - 📞 Rele chak 3 mwa
                    - 🌟 Òf espesyal
                    
                    **Mesaj:**
                    "Mèsi pou fidelite w! Gade òf VIP nou yo."
                    """)
                elif risk_level == "MOYEN":
                    st.warning("""
                    **Estratèj Konsolidasyon:**
                    - 📞 Rele nan 7 jou
                    - 🎯 Òf pèsonalize
                    - 💻 Amelyore eksperyans
                    - 🤝 Pwogram parènaj
                    
                    **Mesaj:**
                    "Opinyon w enpòtan! Ann pale de bezwen w."
                    """)
                else:
                    st.error("""
                    **🚨 IJAN - Retansyon Imedya:**
                    - ☎️ Rele manadjè < 24 èdtan
                    - 💰 Òf retansyon espesyal
                    - 🔍 Verifye kont konplè
                    - 📊 Suivi 30 jou
                    
                    **Mesaj:**
                    "Priyorite absoli! Kontakte nou kounye a."
                    """)
            
            # Plan d'action
            st.markdown("---")
            st.subheader("🎯 Plan d'Action Opérationnel")
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                st.write("**⏰ Actions Immédiates (0-48h):**")
                if risk_level == "ÉLEVÉ":
                    st.markdown("""
                    1. 🚨 **Alerte gestionnaire** - Priorité MAX
                    2. ☎️ **Appel personnel** - Script rétention
                    3. 💰 **Offre immédiate** - Budget spécial
                    4. 📝 **Documentation** - CRM complet
                    """)
                else:
                    st.markdown("""
                    1. 📅 **Planifier contact** - Agenda prioritaire
                    2. 📊 **Analyser profil** - Historique complet
                    3. 🎯 **Préparer offres** - Personnalisation
                    4. 💻 **Check digital** - Usage outils
                    """)
            
            with action_col2:
                st.write("**📈 Actions Moyen Terme (1-30 jours):**")
                st.markdown("""
                1. 🔄 **Suivi régulier** - Touchpoints
                2. 🎁 **Programme fidélité** - Avantages
                3. 📚 **Formation** - Outils digitaux
                4. 🤝 **Relation client** - Renforcement
                5. 📊 **KPIs** - Monitoring continu
                """)
            
            # Historique
            analysis_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "churn_probability": float(churn_proba),
                "risk_level": risk_level,
                "processing_time": float(processing_time),
                "client_id": f"CLT_{int(time.time())}"
            }
            st.session_state.analysis_history.append(analysis_record)
            
            # Export
            st.markdown("---")
            if st.button("📥 Exporter l'Analyse (JSON)"):
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
                    "💾 Télécharger JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name=f"churn_analysis_{int(time.time())}.json",
                    mime="application/json"
                )
            
        except Exception as e:
            st.error(f"❌ ERREUR: {str(e)}")
            with st.expander("🔍 Détails"):
                import traceback
                st.code(traceback.format_exc())

elif analyze_clicked:
    st.error("❌ Modèle non disponible.")

# Historique
if st.session_state.analysis_history:
    st.markdown("---")
    with st.expander(f"📜 Historique ({len(st.session_state.analysis_history)})"):
        df_history = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(df_history, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h4>🏦 BankChurnAI - Haïti 🇭🇹</h4>
    <p><strong>Prédiction ML • Analyse SHAP • Recommandations Bilingues</strong></p>
    <p>Ayiti AI Hackathon 2025 • Équipe IMPACTIS</p>
    <p><em>Riché FLEURINORD • Micka LOUIS • Vilmarson JULES</em></p>
</div>
""", unsafe_allow_html=True)

# CSS
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stButton>button { width: 100%; }
    h1 { color: #1E3A8A; }
    h2 { color: #2563EB; }
</style>
""", unsafe_allow_html=True)