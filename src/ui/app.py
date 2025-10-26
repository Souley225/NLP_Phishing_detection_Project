"""
Interface utilisateur Streamlit pour la détection de phishing.

Cette interface permet aux utilisateurs de tester des URLs
et de visualiser les prédictions du modèle.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import os
from datetime import datetime

import requests
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# URL de l'API (depuis variable d'environnement ou localhost par défaut)
API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health() -> bool:
    """
    Vérifie que l'API est accessible et fonctionnelle.
    
    Returns:
        True si l'API est accessible, False sinon
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("model_loaded", False)
    except Exception:
        return False


def predict_url(url: str) -> dict | None:
    """
    Envoie une URL à l'API pour prédiction.
    
    Args:
        url: URL à analyser
    
    Returns:
        Résultat de la prédiction ou None en cas d'erreur
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"url": url},
            timeout=10,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de se connecter à l'API. Vérifiez qu'elle est lancée.")
        return None
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        return None


def display_prediction_result(result: dict) -> None:
    """
    Affiche le résultat de la prédiction de manière visuelle.
    
    Args:
        result: Dictionnaire contenant les résultats
    """
    # Déterminer la couleur selon la prédiction
    is_phishing = result["prediction"] == 1
    color = "red" if is_phishing else "green"
    emoji = "🚨" if is_phishing else "✅"
    
    # Afficher le résultat principal
    st.markdown("---")
    st.markdown(f"### {emoji} Résultat de l'analyse")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Classification",
            value=result["label"].upper(),
            delta=None,
        )
    
    with col2:
        st.metric(
            label="Confiance",
            value=f"{result['confidence']:.1%}",
            delta=None,
        )
    
    with col3:
        st.metric(
            label="Timestamp",
            value=datetime.fromisoformat(result["timestamp"].replace("Z", "")).strftime("%H:%M:%S"),
            delta=None,
        )
    
    # Afficher les probabilités détaillées
    st.markdown("#### 📊 Probabilités détaillées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.progress(result["proba_legitimate"])
        st.write(f"**Légitime**: {result['proba_legitimate']:.2%}")
    
    with col2:
        st.progress(result["proba_phishing"])
        st.write(f"**Phishing**: {result['proba_phishing']:.2%}")
    
    # Avertissement si phishing détecté
    if is_phishing:
        st.error(
            "⚠️ **ATTENTION**: Cette URL semble être du phishing. "
            "Ne partagez pas d'informations personnelles ou financières sur ce site."
        )
    else:
        st.success("✅ Cette URL semble légitime, mais restez toujours vigilant en ligne.")


def main() -> None:
    """
    Fonction principale de l'interface Streamlit.
    """
    # Header
    st.title("🛡️ Détection de Phishing par URL")
    st.markdown(
        """
        Analysez une URL pour déterminer si elle est **légitime** ou **suspecte de phishing**.
        
        **Auteur**: Souleymane Sall | **Email**: sallsouleymane2207@gmail.com
        """
    )
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Informations")
        
        # Vérifier le statut de l'API
        st.markdown("### Statut de l'API")
        if check_api_health():
            st.success("✅ API opérationnelle")
        else:
            st.error("❌ API inaccessible")
            st.info(f"URL de l'API: `{API_URL}`")
        
        st.markdown("---")
        
        # Informations sur le modèle
        st.markdown("### 📈 À propos du modèle")
        st.info(
            """
            Ce modèle utilise:
            - **TF-IDF** sur les tokens d'URL
            - **TF-IDF** sur les n-grams de caractères
            - **Features lexicales** (longueur, entropie, structure)
            - **Régression logistique** optimisée
            """
        )
        
        st.markdown("---")
        
        # Exemples d'URLs
        st.markdown("### 📝 Exemples d'URLs")
        
        st.markdown("**URLs légitimes:**")
        st.code("https://www.google.com")
        st.code("https://github.com")
        
        st.markdown("**URLs suspectes:**")
        st.code("http://paypal-secure.tk/login")
        st.code("http://192.168.1.1/bank-verify")
    
    # Zone principale
    st.markdown("---")
    st.header("🔍 Analyser une URL")
    
    # Input de l'utilisateur
    url_input = st.text_input(
        "Entrez l'URL à analyser:",
        placeholder="https://example.com",
        help="Saisissez l'URL complète (avec http:// ou https://)",
    )
    
    # Bouton d'analyse
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        analyze_button = st.button("🔎 Analyser", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Effacer", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    # Traitement de l'analyse
    if analyze_button:
        if not url_input:
            st.warning("⚠️ Veuillez entrer une URL")
        else:
            with st.spinner("🔄 Analyse en cours..."):
                result = predict_url(url_input)
                
                if result:
                    display_prediction_result(result)
                    
                    # Historique dans session state
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    
                    st.session_state.history.append({
                        "url": url_input,
                        "label": result["label"],
                        "confidence": result["confidence"],
                        "timestamp": result["timestamp"],
                    })
    
    # Afficher l'historique
    if "history" in st.session_state and st.session_state.history:
        st.markdown("---")
        st.header("📜 Historique des analyses")
        
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"{i+1}. {item['url'][:50]}... - {item['label'].upper()}"):
                st.write(f"**Label**: {item['label']}")
                st.write(f"**Confiance**: {item['confidence']:.2%}")
                st.write(f"**Timestamp**: {item['timestamp']}")
        
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Projet MLOps - NLP Phishing Detection</p>
            <p>© 2025 Souleymane Sall | Technologies: Hydra, Optuna, MLflow, FastAPI, Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()