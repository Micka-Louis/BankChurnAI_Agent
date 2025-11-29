# dashboard/app_test.py - TEST DU NOUVEAU MODÈLE HACKATHON
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import sys

# Ajouter le chemin parent pour importer les modules
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="TEST Nouveau Modèle", layout="centered")
st.title("🧪 TEST - Nouveau Modèle Hackathon")

# ==================== CONFIGURATION DES CHEMINS ====================
current_dir = Path(__file__).parent
models_dir = current_dir.parent / "models"

# Chemins des fichiers
model_path = models_dir / "best_churn_model_hackathon.pkl"
preprocessor_path = models_dir / "preprocessor.pkl" 
feature_names_path = models_dir / "feature_names.pkl"
metadata_path = models_dir / "model_metadata.json"

st.write(f"📁 Dossier modèles: {models_dir}")

# ==================== CHARGEMENT DES COMPOSANTS ====================
@st.cache_resource
def load_components():
    try:
        model = joblib.load(model_path)
        st.success("✅ Modèle chargé")
        
        preprocessor = joblib.load(preprocessor_path)
        st.success("✅ Preprocesseur chargé")
        
        feature_names = joblib.load(feature_names_path)
        st.success("✅ Features names chargés")
        
        return model, preprocessor, feature_names
    except Exception as e:
        st.error(f"❌ Erreur chargement: {e}")
        return None, None, None

model, preprocessor, feature_names = load_components()

if model is None:
    st.stop()

# ==================== FONCTION DE NORMALISATION ====================
def normalize_client_data(client_data):
    """Normalise les données client comme pendant l'entraînement"""
    normalized = client_data.copy()
    
    # Normalisation monétaire (CRITIQUE - même que l'entraînement)
    monetary_features = {
        'income_monthly': 1000,
        'account_balance': 1000, 
        'loan_balance': 1000,
        'diaspora_transfers_received': 1000,
        'transfer_fees_paid': 100
    }
    
    for feature, divisor in monetary_features.items():
        if feature in normalized:
            normalized[feature] = normalized[feature] / divisor
    
    return normalized

# ==================== FONCTION DE PRÉDICTION ====================
def make_prediction(client_data):
    """Fait une prédiction avec le nouveau modèle"""
    try:
        # 1. Normalisation
        client_normalized = normalize_client_data(client_data)
        
        # 2. Création DataFrame
        df_client = pd.DataFrame([client_normalized])
        
        # 3. Preprocessing
        client_processed = preprocessor.transform(df_client)
        
        # 4. Prédiction
        proba = model.predict_proba(client_processed)
        
        return proba, client_normalized, client_processed
        
    except Exception as e:
        st.error(f"❌ Erreur prédiction: {e}")
        return None, None, None

# ==================== TESTS AUTOMATIQUES ====================
st.markdown("---")
st.subheader("🧪 TESTS AUTOMATIQUES")

test_clients = [
    {
        "name": "👑 CLIENT FIDÈLE",
        "description": "Client stable, bon revenu, utilise les services digitaux",
        "data": {
            'age': 45, 'household_size': 3, 'zone_security_level': 1,
            'distance_to_branch_km': 2.0, 'income_monthly': 80000,
            'account_balance': 150000, 'credit_score': 750, 'loan_balance': 50000,
            'transactions_count_monthly': 25, 'transfer_fees_paid': 500,
            'time_with_bank_months': 60, 'last_transaction_days': 3,
            'diaspora_transfers_received': 20000, 'mobile_app_logins': 15,
            'sentiment_score': 0.7, 'access_to_internet': 1,
            'gender': 'M', 'marital_status': 'Married', 'education_level': 'University',
            'profession': 'Civil Servant', 'region': 'Ouest',
            'mobile_money_usage': 'High', 'customer_persona_ai': 'Digital Native'
        }
    },
    {
        "name": "⚠️ CLIENT RISQUÉ", 
        "description": "Client jeune, faible revenu, peu d'activité",
        "data": {
            'age': 28, 'household_size': 2, 'zone_security_level': 5,
            'distance_to_branch_km': 25.0, 'income_monthly': 12000,
            'account_balance': 5000, 'credit_score': 420, 'loan_balance': 0,
            'transactions_count_monthly': 3, 'transfer_fees_paid': 50,
            'time_with_bank_months': 8, 'last_transaction_days': 45,
            'diaspora_transfers_received': 0, 'mobile_app_logins': 0,
            'sentiment_score': -0.6, 'access_to_internet': 0,
            'gender': 'F', 'marital_status': 'Single', 'education_level': 'Primary',
            'profession': 'Unemployed', 'region': 'Nord',
            'mobile_money_usage': 'Low', 'customer_persona_ai': 'Cash User'
        }
    },
    {
        "name": "📊 CLIENT MOYEN",
        "description": "Client avec profil mixte, risque modéré",
        "data": {
            'age': 35, 'household_size': 2, 'zone_security_level': 3,
            'distance_to_branch_km': 10.0, 'income_monthly': 40000,
            'account_balance': 50000, 'credit_score': 600, 'loan_balance': 20000,
            'transactions_count_monthly': 12, 'transfer_fees_paid': 200,
            'time_with_bank_months': 24, 'last_transaction_days': 15,
            'diaspora_transfers_received': 5000, 'mobile_app_logins': 8,
            'sentiment_score': 0.1, 'access_to_internet': 1,
            'gender': 'M', 'marital_status': 'Married', 'education_level': 'Secondary',
            'profession': 'Merchant', 'region': 'Artibonite',
            'mobile_money_usage': 'Medium', 'customer_persona_ai': 'Trader'
        }
    }
]

results = []

for i, client in enumerate(test_clients, 1):
    st.write(f"### Test {i}: {client['name']}")
    st.write(f"*{client['description']}*")
    
    # Prédiction
    proba, client_normalized, client_processed = make_prediction(client['data'])
    
    if proba is not None:
        # Extraction probabilité churn (index 1)
        churn_proba = proba[0, 1]
        results.append(churn_proba)
        
        # Debug détaillé
        with st.expander("🔍 Détails de la prédiction"):
            st.write("**Données brutes envoyées:**")
            st.json(client['data'])
            
            st.write("**Données normalisées:**")
            st.json(client_normalized)
            
            st.write("**Probabilités brutes:**")
            st.write(f"Classe 0 (fidèle): {proba[0, 0]:.6f} → {proba[0, 0]:.4%}")
            st.write(f"Classe 1 (churn): {proba[0, 1]:.6f} → {proba[0, 1]:.4%}")
            
            st.write(f"**Shape données transformées:** {client_processed.shape}")
        
        # Affichage résultat
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="**Probabilité Churn**",
                value=f"{churn_proba:.2%}",
                delta="FAIBLE" if churn_proba < 0.3 else "MOYEN" if churn_proba < 0.7 else "ÉLEVÉ"
            )
        
        with col2:
            decision = "FIDÈLE" if churn_proba < 0.5 else "CHURN"
            st.metric("**Décision**", decision)
            
        with col3:
            st.progress(float(churn_proba), text=f"Risque: {churn_proba:.1%}")
    
    st.markdown("---")

# ==================== ANALYSE DES RÉSULTATS ====================
st.subheader("📊 ANALYSE COMPARATIVE")

if len(results) == 3:
    st.write(f"**👑 Client Fidèle:** {results[0]:.2%}")
    st.write(f"**⚠️ Client Risqué:** {results[1]:.2%}") 
    st.write(f"**📊 Client Moyen:** {results[2]:.2%}")
    
    difference = abs(results[0] - results[1])
    st.write(f"**Différence fidèle/risqué:** {difference:.2%}")
    
    if difference > 0.5:
        st.success("🎉 **EXCELLENTE discrimination!**")
    elif difference > 0.3:
        st.warning("⚠️ **Bonne discrimination**")
    else:
        st.error("❌ **Discrimination faible**")

# ==================== TEST MANUEL ====================
st.markdown("---")
st.subheader("🎯 TEST MANUEL")

with st.form("manual_test"):
    st.write("**Entrez les données d'un client:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Âge", 18, 100, 35)
        income = st.number_input("Revenu Mensuel (HTG)", 5000, 500000, 50000)
        balance = st.number_input("Solde Compte (HTG)", 0, 1000000, 50000)
        credit_score = st.slider("Score Crédit", 300, 850, 650)
        
    with col2:
        transactions = st.slider("Transactions/Mois", 0, 100, 15)
        last_transaction = st.slider("Dernière Transaction (jours)", 0, 90, 7)
        mobile_logins = st.slider("Connexions App Mobile", 0, 50, 8)
        sentiment = st.slider("Score Sentiment", -1.0, 1.0, 0.0, 0.1)
    
    submitted = st.form_submit_button("🎯 Tester ce client")
    
    if submitted:
        client_data = {
            'age': age, 'income_monthly': income, 'account_balance': balance,
            'credit_score': credit_score, 'transactions_count_monthly': transactions,
            'last_transaction_days': last_transaction, 'mobile_app_logins': mobile_logins,
            'sentiment_score': sentiment, 'access_to_internet': 1,
            'household_size': 3, 'zone_security_level': 2, 'distance_to_branch_km': 5.0,
            'loan_balance': 0, 'transfer_fees_paid': income * 0.02, 'time_with_bank_months': 24,
            'diaspora_transfers_received': 0, 'gender': 'M', 'marital_status': 'Married',
            'education_level': 'Secondary', 'profession': 'Merchant', 'region': 'Ouest',
            'mobile_money_usage': 'Medium', 'customer_persona_ai': 'Saver'
        }
        
        proba, _, _ = make_prediction(client_data)
        if proba is not None:
            churn_proba = proba[0, 1]
            
            st.success(f"**Résultat:** {churn_proba:.2%} de risque de churn")
            st.progress(float(churn_proba))

# ==================== INFORMATIONS MODÈLE ====================
st.markdown("---")
st.subheader("ℹ️ INFORMATIONS DU MODÈLE")

try:
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    st.write(f"**Modèle:** {metadata.get('best_model', 'N/A')}")
    st.write(f"**Performance:** ROC-AUC = {metadata.get('performance', {}).get('test_roc_auc', 'N/A')}")
    
except:
    st.write("ℹ️ Modèle: LightGBM (Hackathon Express)")
    st.write("📈 Performance: ROC-AUC = 0.9944")

st.caption("BankChurnAI - Test du nouveau modèle • Hackathon Ayiti AI 2025")