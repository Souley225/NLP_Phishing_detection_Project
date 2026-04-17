"""
PhishGuard — Outil de détection de phishing par analyse d'URL.
Interface professionnelle en français, adaptée à un usage RH/entreprise.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="PhishGuard — Détection de phishing",
    page_icon="https://cdn-icons-png.flaticon.com/512/2716/2716652.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Source+Code+Pro:wght@400;500&display=swap');

    :root {
        --navy:      #1b2f4e;
        --navy2:     #243a5e;
        --blue:      #2563eb;
        --blue-l:    #eff6ff;
        --red:       #dc2626;
        --red-l:     #fef2f2;
        --red-b:     #fecaca;
        --green:     #16a34a;
        --green-l:   #f0fdf4;
        --green-b:   #bbf7d0;
        --amber:     #d97706;
        --bg:        #f1f5f9;
        --surface:   #ffffff;
        --border:    #e2e8f0;
        --border2:   #cbd5e1;
        --text:      #1e293b;
        --text2:     #64748b;
        --text3:     #94a3b8;
        --ff:        'Plus Jakarta Sans', sans-serif;
        --ff-m:      'Source Code Pro', monospace;
        --radius:    10px;
        --shadow:    0 1px 3px rgba(0,0,0,.07), 0 4px 16px rgba(0,0,0,.05);
        --shadow2:   0 2px 8px rgba(0,0,0,.10), 0 8px 32px rgba(0,0,0,.07);
    }

    html, body, .stApp {
        font-family: var(--ff) !important;
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    /* Masquer chrome Streamlit */
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }

    /* Conteneur principal */
    .main .block-container {
        padding: 2rem 2.5rem 4rem !important;
        max-width: 940px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--navy) !important;
    }
    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,.85) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: rgba(255,255,255,.7) !important;
        font-size: .84rem !important;
    }

    /* Titres */
    h1, h2, h3 {
        font-family: var(--ff) !important;
        font-weight: 700 !important;
        color: var(--navy) !important;
        letter-spacing: -.01em !important;
    }

    /* Input */
    .stTextInput label { display: none !important; }
    .stTextInput > div > div > input {
        background: var(--surface) !important;
        border: 1.5px solid var(--border2) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
        font-family: var(--ff-m) !important;
        font-size: .9rem !important;
        padding: .8rem 1rem !important;
        box-shadow: var(--shadow) !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,.12) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: var(--text3) !important;
        font-family: var(--ff) !important;
    }

    /* Boutons */
    .stButton > button, .stFormSubmitButton > button {
        font-family: var(--ff) !important;
        font-weight: 600 !important;
        font-size: .88rem !important;
        border-radius: var(--radius) !important;
        padding: .6rem 1.4rem !important;
        transition: all .2s !important;
    }
    .stFormSubmitButton > button[kind="primary"] {
        background: var(--navy) !important;
        color: #fff !important;
        border: none !important;
        box-shadow: var(--shadow) !important;
    }
    .stFormSubmitButton > button[kind="primary"]:hover {
        background: var(--navy2) !important;
        box-shadow: var(--shadow2) !important;
        transform: translateY(-1px) !important;
    }
    .stFormSubmitButton > button:not([kind="primary"]) {
        background: var(--surface) !important;
        color: var(--text2) !important;
        border: 1.5px solid var(--border2) !important;
    }
    .stFormSubmitButton > button:not([kind="primary"]):hover {
        background: var(--bg) !important;
        border-color: var(--text3) !important;
    }

    /* Alertes */
    .stWarning {
        background: #fffbeb !important;
        border: 1px solid #fde68a !important;
        border-radius: var(--radius) !important;
        color: #92400e !important;
        font-size: .88rem !important;
    }
    .stError {
        background: var(--red-l) !important;
        border: 1px solid var(--red-b) !important;
        border-radius: var(--radius) !important;
        font-size: .88rem !important;
    }

    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* ── Composants maison ── */

    /* En-tête */
    .pg-header {
        padding: 2rem 0 1.5rem;
        border-bottom: 2px solid var(--border);
        margin-bottom: 1.8rem;
    }
    .pg-logo-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: .4rem;
    }
    .pg-icon {
        width: 42px; height: 42px;
        background: var(--navy);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }
    .pg-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--navy);
        letter-spacing: -.02em;
        line-height: 1.1;
    }
    .pg-subtitle {
        font-size: .88rem;
        color: var(--text2);
        margin-top: .25rem;
        line-height: 1.5;
    }

    /* Badge statut */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .3rem .8rem;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .01em;
    }
    .status-badge.online {
        background: var(--green-l);
        color: var(--green);
        border: 1px solid var(--green-b);
    }
    .status-badge.offline {
        background: #fef3c7;
        color: var(--amber);
        border: 1px solid #fde68a;
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: currentColor;
    }

    /* Section label */
    .section-label {
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--text3);
        margin-bottom: .6rem;
    }

    /* Carte résultat */
    .result-card {
        background: var(--surface);
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow2);
        padding: 1.6rem;
        margin: 1rem 0 1.4rem;
    }
    .result-card.danger {
        border-left: 4px solid var(--red);
        background: linear-gradient(135deg, var(--red-l) 0%, var(--surface) 60%);
    }
    .result-card.success {
        border-left: 4px solid var(--green);
        background: linear-gradient(135deg, var(--green-l) 0%, var(--surface) 60%);
    }

    .result-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 1.2rem;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .result-verdict {
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .result-verdict.danger { color: var(--red); }
    .result-verdict.success { color: var(--green); }
    .result-verdict-sub {
        font-size: .82rem;
        color: var(--text2);
        font-weight: 400;
        margin-top: .2rem;
    }
    .result-confidence {
        text-align: right;
        flex-shrink: 0;
    }
    .result-conf-val {
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1;
    }
    .result-conf-val.danger { color: var(--red); }
    .result-conf-val.success { color: var(--green); }
    .result-conf-label {
        font-size: .72rem;
        color: var(--text3);
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    /* URL analysée */
    .result-url {
        font-family: var(--ff-m);
        font-size: .82rem;
        color: var(--text2);
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: .5rem .85rem;
        margin-bottom: 1.2rem;
        word-break: break-all;
    }

    /* Barres de probabilité */
    .prob-section { margin-bottom: 1.2rem; }
    .prob-row { margin-bottom: .65rem; }
    .prob-meta {
        display: flex;
        justify-content: space-between;
        font-size: .78rem;
        color: var(--text2);
        margin-bottom: .3rem;
        font-weight: 500;
    }
    .prob-track {
        height: 8px;
        background: var(--bg);
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        width: var(--w);
        transition: width 1s cubic-bezier(.4,0,.2,1);
    }
    .prob-fill.legit { background: var(--green); }
    .prob-fill.phish { background: var(--red); }

    /* Grille de stats */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .8rem;
        margin-bottom: 1rem;
    }
    .stat-cell {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: .7rem .9rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--navy);
        line-height: 1.2;
    }
    .stat-val.url-len { font-family: var(--ff-m); font-size: .9rem; }
    .stat-lbl {
        font-size: .66rem;
        color: var(--text3);
        text-transform: uppercase;
        letter-spacing: .07em;
        margin-top: .15rem;
    }

    /* Message d'alerte résultat */
    .result-msg {
        font-size: .85rem;
        border-radius: 8px;
        padding: .8rem 1rem;
        line-height: 1.55;
    }
    .result-msg.danger {
        background: var(--red-l);
        border: 1px solid var(--red-b);
        color: #991b1b;
    }
    .result-msg.success {
        background: var(--green-l);
        border: 1px solid var(--green-b);
        color: #166534;
    }

    /* Historique */
    .hist-table {
        width: 100%;
        border-collapse: collapse;
        font-size: .82rem;
    }
    .hist-table th {
        background: var(--bg);
        color: var(--text3);
        font-weight: 600;
        font-size: .7rem;
        text-transform: uppercase;
        letter-spacing: .07em;
        padding: .55rem .75rem;
        border-bottom: 1px solid var(--border);
        text-align: left;
    }
    .hist-table td {
        padding: .6rem .75rem;
        border-bottom: 1px solid var(--border);
        color: var(--text2);
        vertical-align: middle;
    }
    .hist-table tr:last-child td { border-bottom: none; }
    .hist-table tr:hover td { background: var(--bg); }
    .hist-url-cell {
        font-family: var(--ff-m);
        font-size: .78rem;
        max-width: 340px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: .3rem;
        padding: .18rem .6rem;
        border-radius: 999px;
        font-size: .72rem;
        font-weight: 600;
    }
    .pill.phishing {
        background: var(--red-l);
        color: var(--red);
        border: 1px solid var(--red-b);
    }
    .pill.legitimate {
        background: var(--green-l);
        color: var(--green);
        border: 1px solid var(--green-b);
    }
    .pill-dot {
        width: 5px; height: 5px;
        border-radius: 50%;
        background: currentColor;
        flex-shrink: 0;
    }

    /* Sidebar */
    .sb-section {
        margin-bottom: 1.4rem;
    }
    .sb-title {
        font-size: .68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .1em;
        color: rgba(255,255,255,.4) !important;
        margin-bottom: .6rem;
        padding-bottom: .4rem;
        border-bottom: 1px solid rgba(255,255,255,.1);
    }
    .sb-tag {
        display: inline-block;
        background: rgba(255,255,255,.1);
        border: 1px solid rgba(255,255,255,.15);
        border-radius: 4px;
        padding: .2rem .55rem;
        font-size: .68rem;
        color: rgba(255,255,255,.75) !important;
        margin: .15rem;
        font-family: var(--ff);
    }
    .sb-ex-url {
        font-family: var(--ff-m);
        font-size: .7rem;
        padding: .35rem .6rem;
        border-radius: 5px;
        margin-bottom: .3rem;
        word-break: break-all;
        color: rgba(255,255,255,.75) !important;
    }
    .sb-ex-url.ok  { background: rgba(22,163,74,.15);  border: 1px solid rgba(22,163,74,.3); }
    .sb-ex-url.bad { background: rgba(220,38,38,.15);  border: 1px solid rgba(220,38,38,.3); }
    .sb-stats-grid {
        display: grid;
        grid-template-columns: repeat(3,1fr);
        gap: .5rem;
    }
    .sb-stat-box {
        background: rgba(255,255,255,.07);
        border-radius: 7px;
        padding: .6rem .4rem;
        text-align: center;
    }
    .sb-stat-val {
        font-size: 1.1rem;
        font-weight: 700;
        color: #fff !important;
    }
    .sb-stat-lbl {
        font-size: .62rem;
        color: rgba(255,255,255,.45) !important;
        text-transform: uppercase;
        letter-spacing: .07em;
    }

    /* Pied de page */
    .pg-footer {
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--border);
        font-size: .75rem;
        color: var(--text3);
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: .5rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Chargement du modèle ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modèle de détection…")
def load_model():
    """Charge le modele, les vectorizers et le seuil de decision. Mis en cache."""
    import joblib
    from src.features.build_features import URLFeatureExtractor

    models_dir = ROOT / "models"
    cfg_path   = ROOT / "configs" / "config.yaml"

    with open(cfg_path) as f:
        feat_cfg = yaml.safe_load(f)["features"]

    model = joblib.load(models_dir / "best_model.pkl")

    extractor = URLFeatureExtractor(feat_cfg)
    if feat_cfg.get("tfidf_word", {}).get("use"):
        extractor.tfidf_word = joblib.load(models_dir / "tfidf_word.pkl")
    if feat_cfg.get("tfidf_char", {}).get("use"):
        extractor.tfidf_char = joblib.load(models_dir / "tfidf_char.pkl")
    if feat_cfg.get("lexical", {}).get("use"):
        extractor.scaler = joblib.load(models_dir / "scaler.pkl")
    extractor.is_fitted = True

    threshold_path = models_dir / "threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold = float(json.load(f).get("threshold", 0.5))
    else:
        threshold = 0.5

    return model, extractor, threshold


def is_model_ready() -> bool:
    try:
        load_model()
        return True
    except Exception:
        return False


def predict_url(url: str) -> dict | None:
    try:
        model, extractor, threshold = load_model()
        X = extractor.transform(pd.Series([url]))
        probabilities = model.predict_proba(X)[0]

        classes = list(model.classes_)
        if isinstance(classes[0], (int, np.integer)):
            idx_legit = classes.index(0) if 0 in classes else 0
            idx_phish = classes.index(1) if 1 in classes else 1
        else:
            idx_legit = classes.index("good") if "good" in classes else 0
            idx_phish = classes.index("bad")  if "bad"  in classes else 1

        proba_legit = float(probabilities[idx_legit])
        proba_phish = float(probabilities[idx_phish])

        # Seuil optimise pour limiter les faux positifs
        prediction  = 1 if proba_phish >= threshold else 0
        confidence  = proba_phish if prediction == 1 else proba_legit

        return {
            "url":              url,
            "prediction":       prediction,
            "label":            "phishing" if prediction == 1 else "legitimate",
            "confidence":       confidence,
            "proba_legitimate": proba_legit,
            "proba_phishing":   proba_phish,
            "timestamp":        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
        return None


# ── Rendu HTML ────────────────────────────────────────────────────────────────

def render_header(online: bool) -> None:
    dot_cls   = "online" if online else "offline"
    dot_label = "Modèle actif" if online else "Chargement en cours…"
    st.markdown(
        f'<div class="pg-header">'
        f'<div class="pg-logo-row">'
        f'<div class="pg-icon">&#x1F6E1;</div>'
        f'<div>'
        f'<div class="pg-title">PhishGuard</div>'
        f'<div class="pg-subtitle">Détection automatique de liens de phishing par analyse linguistique</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:.8rem">'
        f'<span class="status-badge {dot_cls}">'
        f'<span class="status-dot"></span>{dot_label}'
        f'</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_result(result: dict, url: str) -> None:
    is_phishing = result["prediction"] == 1
    cls         = "danger" if is_phishing else "success"
    verdict     = "Lien suspect — Phishing probable" if is_phishing else "Lien légitime"
    verdict_sub = (
        "Ce lien présente des caractéristiques typiques d'une tentative de phishing."
        if is_phishing else
        "Aucun indicateur de phishing détecté sur ce lien."
    )
    conf_pct    = f"{result['confidence']:.0%}"
    p_legit     = result["proba_legitimate"]
    p_phish     = result["proba_phishing"]
    w_legit     = f"{p_legit * 100:.1f}%"
    w_phish     = f"{p_phish * 100:.1f}%"
    ts          = datetime.fromisoformat(result["timestamp"].replace("Z", "")).strftime("%d/%m/%Y %H:%M UTC")
    msg = (
        "Attention : ne cliquez pas sur ce lien et ne saisissez aucune information personnelle "
        "(identifiant, mot de passe, coordonnées bancaires). Signalez-le à votre service informatique."
        if is_phishing else
        "Ce lien semble fiable. Restez néanmoins vigilant et vérifiez toujours l'expéditeur avant "
        "de saisir des informations sensibles."
    )

    st.markdown(
        f'<div class="result-card {cls}">'
        f'<div class="result-header">'
        f'<div>'
        f'<div class="result-verdict {cls}">{verdict}</div>'
        f'<div class="result-verdict-sub">{verdict_sub}</div>'
        f'</div>'
        f'<div class="result-confidence">'
        f'<div class="result-conf-label">Confiance</div>'
        f'<div class="result-conf-val {cls}">{conf_pct}</div>'
        f'</div>'
        f'</div>'
        f'<div class="result-url">{url}</div>'
        f'<div class="prob-section">'
        f'<div class="prob-row">'
        f'<div class="prob-meta"><span>Légitime</span><span>{p_legit:.1%}</span></div>'
        f'<div class="prob-track"><div class="prob-fill legit" style="--w:{w_legit}"></div></div>'
        f'</div>'
        f'<div class="prob-row">'
        f'<div class="prob-meta"><span>Phishing</span><span>{p_phish:.1%}</span></div>'
        f'<div class="prob-track"><div class="prob-fill phish" style="--w:{w_phish}"></div></div>'
        f'</div>'
        f'</div>'
        f'<div class="stats-grid">'
        f'<div class="stat-cell"><div class="stat-val">{ts}</div><div class="stat-lbl">Date d\'analyse</div></div>'
        f'<div class="stat-cell"><div class="stat-val">{"Phishing" if is_phishing else "Légitime"}</div><div class="stat-lbl">Classification</div></div>'
        f'<div class="stat-cell"><div class="stat-val url-len">{len(url)}</div><div class="stat-lbl">Longueur URL</div></div>'
        f'</div>'
        f'<div class="result-msg {cls}">{msg}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_history(history: list) -> None:
    items = list(reversed(history[-10:]))
    rows = ""
    for item in items:
        lbl   = item["label"]
        ts    = datetime.fromisoformat(item["timestamp"].replace("Z", "")).strftime("%H:%M")
        short = item["url"][:60] + ("…" if len(item["url"]) > 60 else "")
        conf  = f"{item['confidence']:.0%}"
        label_fr = "Phishing" if lbl == "phishing" else "Légitime"
        rows += (
            f'<tr>'
            f'<td class="hist-url-cell">{short}</td>'
            f'<td><span class="pill {lbl}"><span class="pill-dot"></span>{label_fr}</span></td>'
            f'<td style="font-weight:600;color:var(--text)">{conf}</td>'
            f'<td style="color:var(--text3)">{ts}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="hist-table">'
        f'<thead><tr>'
        f'<th>URL analysée</th><th>Verdict</th><th>Confiance</th><th>Heure</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>',
        unsafe_allow_html=True,
    )


def render_sidebar(history: list) -> None:
    total   = len(history)
    threats = sum(1 for h in history if h["label"] == "phishing")
    safe    = total - threats

    st.markdown(
        f'<div class="sb-section">'
        f'<div class="sb-title">Statistiques de session</div>'
        f'<div class="sb-stats-grid">'
        f'<div class="sb-stat-box"><div class="sb-stat-val">{total}</div><div class="sb-stat-lbl">Analyses</div></div>'
        f'<div class="sb-stat-box"><div class="sb-stat-val">{threats}</div><div class="sb-stat-lbl">Suspects</div></div>'
        f'<div class="sb-stat-box"><div class="sb-stat-val">{safe}</div><div class="sb-stat-lbl">Sûrs</div></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sb-section">'
        '<div class="sb-title">Modèle</div>',
        unsafe_allow_html=True,
    )
    for tag in ["TF-IDF mots", "TF-IDF n-grammes", "Features lexicales", "Régression logistique", "F1 = 0.93", "Précision = 96%"]:
        st.markdown(f'<span class="sb-tag">{tag}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sb-section" style="margin-top:.5rem">'
        '<div class="sb-title">Exemples de liens</div>'
        '<div style="font-size:.66rem;color:rgba(255,255,255,.4);text-transform:uppercase;letter-spacing:.07em;margin-bottom:.35rem">Légitimes</div>'
        '<div class="sb-ex-url ok">https://www.google.com</div>'
        '<div class="sb-ex-url ok">https://github.com/login</div>'
        '<div style="font-size:.66rem;color:rgba(255,255,255,.4);text-transform:uppercase;letter-spacing:.07em;margin:.5rem 0 .35rem">Suspects</div>'
        '<div class="sb-ex-url bad">http://paypal-secure.tk/login.php</div>'
        '<div class="sb-ex-url bad">http://192.168.1.1/bank-verify</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="margin-top:auto;padding-top:1rem;font-size:.68rem;'
        'color:rgba(255,255,255,.3);line-height:1.7">'
        'Souleymane Sall<br>NLP Phishing Detection &middot; 2025'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()

    online  = is_model_ready()
    history = st.session_state.get("history", [])

    with st.sidebar:
        render_sidebar(history)

    render_header(online)

    # Formulaire d'analyse
    st.markdown('<div class="section-label">Analyser une URL</div>', unsafe_allow_html=True)

    with st.form(key="scan_form", clear_on_submit=False):
        url_input = st.text_input(
            label="url",
            placeholder="Collez ici l'URL à vérifier — ex. https://example.com/page",
            key="url_field",
        )
        col1, col2, _ = st.columns([1.5, 1, 4])
        with col1:
            submitted = st.form_submit_button("Analyser", type="primary", use_container_width=True)
        with col2:
            clear = st.form_submit_button("Effacer", use_container_width=True)

    if clear:
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_url", None)
        st.rerun()

    if submitted:
        if not url_input.strip():
            st.warning("Veuillez saisir une URL avant de lancer l'analyse.")
        elif not online:
            st.warning("Le modèle est en cours de chargement. Veuillez patienter quelques instants.")
        else:
            with st.spinner("Analyse en cours…"):
                result = predict_url(url_input.strip())
            if result:
                st.session_state["last_result"] = result
                st.session_state["last_url"]    = url_input.strip()
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    "url":        url_input.strip(),
                    "label":      result["label"],
                    "confidence": result["confidence"],
                    "timestamp":  result["timestamp"],
                })
                st.rerun()

    # Résultat
    if "last_result" in st.session_state:
        st.markdown('<div class="section-label" style="margin-top:1.2rem">Résultat de l\'analyse</div>', unsafe_allow_html=True)
        render_result(st.session_state["last_result"], st.session_state["last_url"])

    # Historique
    history = st.session_state.get("history", [])
    if history:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Historique de la session</div>', unsafe_allow_html=True)
        render_history(history)
        st.markdown('<div style="margin-top:.6rem"></div>', unsafe_allow_html=True)
        if st.button("Effacer l'historique"):
            st.session_state.history = []
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_url", None)
            st.rerun()

    # Pied de page
    st.markdown(
        '<div class="pg-footer">'
        '<span>PhishGuard — Détection de phishing par NLP</span>'
        '<span>scikit-learn &middot; Streamlit &middot; '
        '<a href="mailto:sallsouleymane2207@gmail.com" style="color:inherit">Souleymane Sall</a>'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
