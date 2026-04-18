"""
PhishGuard — Outil de détection de phishing par analyse d'URL.
Interface professionnelle en français, adaptée à un usage entreprise.

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

# ── Constantes ────────────────────────────────────────────────────────────────

LINKEDIN_URL = "https://www.linkedin.com/in/souleymanes-sall/"
GITHUB_URL   = "https://github.com/Souley225/NLP_Phishing_detection_Project"

SVG_SHIELD = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'width="22" height="22" fill="white">'
    '<path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>'
    '</svg>'
)

SVG_LINKEDIN = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'width="16" height="16" fill="currentColor">'
    '<path d="M19 0H5C2.239 0 0 2.239 0 5v14c0 2.761 2.239 5 5 5h14c2.762 0 '
    '5-2.239 5-5V5c0-2.761-2.238-5-5-5zM8 19H5V8h3v11zM6.5 6.732c-.966 0-1.75'
    '-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 '
    '1.764zM20 19h-3v-5.604c0-3.368-4-3.113-4 0V19h-3V8h3v1.765C14.396 7.179 '
    '20 6.988 20 12.248V19z"/>'
    '</svg>'
)

SVG_GITHUB = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'width="16" height="16" fill="currentColor">'
    '<path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599'
    '.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546'
    '-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 '
    '1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418'
    '-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 '
    '1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23'
    '.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291'
    '-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235'
    ' 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823'
    ' 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627'
    '-5.373-12-12-12z"/>'
    '</svg>'
)


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Source+Code+Pro:wght@400;500&display=swap');

    :root {
        --navy:      #1b2f4e;
        --navy2:     #243a5e;
        --navy3:     #162540;
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

    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }

    .main .block-container {
        padding: 2rem 2.5rem 4rem !important;
        max-width: 960px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: var(--navy) !important; }
    [data-testid="stSidebar"] * { color: rgba(255,255,255,.85) !important; }
    [data-testid="stSidebar"] .stMarkdown p {
        color: rgba(255,255,255,.65) !important;
        font-size: .83rem !important;
        line-height: 1.6 !important;
    }

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
    }
    .stButton > button {
        background: var(--surface) !important;
        color: var(--text2) !important;
        border: 1.5px solid var(--border2) !important;
    }
    .stButton > button:hover {
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

    hr { border-color: var(--border) !important; margin: 1.8rem 0 !important; }

    /* ── Composants maison ── */

    .section-label {
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
        color: var(--text3);
        margin-bottom: .7rem;
    }

    /* En-tête principal */
    .pg-header {
        background: var(--navy);
        border-radius: 14px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.8rem;
        display: flex;
        align-items: flex-start;
        gap: 1.4rem;
    }
    .pg-icon-box {
        width: 48px; height: 48px;
        background: rgba(255,255,255,.12);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        margin-top: .1rem;
    }
    .pg-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #fff;
        letter-spacing: -.02em;
        line-height: 1.15;
        margin-bottom: .2rem;
    }
    .pg-tagline {
        font-size: .88rem;
        color: rgba(255,255,255,.65);
        line-height: 1.5;
        margin-bottom: .9rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        padding: .28rem .75rem;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 600;
    }
    .status-badge.online {
        background: rgba(22,163,74,.2);
        color: #86efac;
        border: 1px solid rgba(22,163,74,.35);
    }
    .status-badge.offline {
        background: rgba(217,119,6,.2);
        color: #fcd34d;
        border: 1px solid rgba(217,119,6,.35);
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

    /* Carte présentation projet */
    .about-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.6rem;
        box-shadow: var(--shadow);
    }
    .about-title {
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
        color: var(--navy);
        margin-bottom: .7rem;
        padding-bottom: .5rem;
        border-bottom: 1px solid var(--border);
    }
    .about-text {
        font-size: .875rem;
        color: var(--text2);
        line-height: 1.7;
        margin: 0;
    }
    .about-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .8rem;
        margin-top: 1.1rem;
    }
    .about-stat {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: .75rem 1rem;
        text-align: center;
    }
    .about-stat-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--navy);
        line-height: 1.1;
    }
    .about-stat-lbl {
        font-size: .68rem;
        color: var(--text3);
        text-transform: uppercase;
        letter-spacing: .07em;
        margin-top: .18rem;
    }

    /* Guide d'utilisation */
    .guide-card {
        background: var(--blue-l);
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.6rem;
    }
    .guide-title {
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
        color: var(--blue);
        margin-bottom: .9rem;
    }
    .guide-steps {
        display: flex;
        gap: 1.2rem;
        flex-wrap: wrap;
    }
    .guide-step {
        flex: 1;
        min-width: 160px;
        display: flex;
        gap: .75rem;
        align-items: flex-start;
    }
    .guide-num {
        width: 26px; height: 26px;
        border-radius: 50%;
        background: var(--blue);
        color: #fff;
        font-size: .78rem;
        font-weight: 700;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        margin-top: .05rem;
    }
    .guide-step-text {
        font-size: .83rem;
        color: #1e40af;
        line-height: 1.55;
    }
    .guide-step-title {
        font-weight: 700;
        display: block;
        margin-bottom: .1rem;
        color: #1e3a8a;
    }

    /* Zone de saisie */
    .input-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.4rem 1.6rem 1rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--shadow);
    }
    .input-label {
        font-size: .83rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: .4rem;
    }
    .input-hint {
        font-size: .76rem;
        color: var(--text3);
        margin-top: .35rem;
        line-height: 1.5;
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
    .result-card.danger { border-left: 4px solid var(--red); }
    .result-card.success { border-left: 4px solid var(--green); }
    .result-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 1.2rem;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .result-verdict { font-size: 1.1rem; font-weight: 700; line-height: 1.2; }
    .result-verdict.danger  { color: var(--red); }
    .result-verdict.success { color: var(--green); }
    .result-verdict-sub {
        font-size: .82rem;
        color: var(--text2);
        margin-top: .2rem;
    }
    .result-conf-block { text-align: right; flex-shrink: 0; }
    .result-conf-val { font-size: 1.55rem; font-weight: 800; line-height: 1; }
    .result-conf-val.danger  { color: var(--red); }
    .result-conf-val.success { color: var(--green); }
    .result-conf-label {
        font-size: .7rem;
        color: var(--text3);
        text-transform: uppercase;
        letter-spacing: .06em;
    }
    .result-url {
        font-family: var(--ff-m);
        font-size: .8rem;
        color: var(--text2);
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: .5rem .85rem;
        margin-bottom: 1.2rem;
        word-break: break-all;
    }
    .prob-section { margin-bottom: 1.1rem; }
    .prob-row { margin-bottom: .6rem; }
    .prob-meta {
        display: flex;
        justify-content: space-between;
        font-size: .78rem;
        color: var(--text2);
        margin-bottom: .28rem;
        font-weight: 500;
    }
    .prob-track {
        height: 7px;
        background: var(--bg);
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .prob-fill { height: 100%; border-radius: 4px; width: var(--w); }
    .prob-fill.legit { background: var(--green); }
    .prob-fill.phish { background: var(--red); }
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
        padding: .65rem .9rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1rem;
        font-weight: 700;
        color: var(--navy);
        line-height: 1.2;
        word-break: break-all;
    }
    .stat-lbl {
        font-size: .63rem;
        color: var(--text3);
        text-transform: uppercase;
        letter-spacing: .07em;
        margin-top: .15rem;
    }
    .result-msg {
        font-size: .85rem;
        border-radius: 8px;
        padding: .85rem 1rem;
        line-height: 1.6;
    }
    .result-msg.danger  { background: var(--red-l);   border: 1px solid var(--red-b);   color: #991b1b; }
    .result-msg.success { background: var(--green-l); border: 1px solid var(--green-b); color: #166534; }

    /* Historique */
    .hist-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
    .hist-table th {
        background: var(--bg);
        color: var(--text3);
        font-weight: 700;
        font-size: .68rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        padding: .55rem .75rem;
        border-bottom: 2px solid var(--border);
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
        font-size: .77rem;
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .pill {
        display: inline-flex; align-items: center; gap: .3rem;
        padding: .2rem .65rem;
        border-radius: 999px;
        font-size: .72rem;
        font-weight: 600;
    }
    .pill.phishing  { background: var(--red-l);   color: var(--red);   border: 1px solid var(--red-b);   }
    .pill.legitimate{ background: var(--green-l); color: var(--green); border: 1px solid var(--green-b); }
    .pill-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; flex-shrink: 0; }

    /* Sidebar */
    .sb-section { margin-bottom: 1.5rem; }
    .sb-divider {
        height: 1px;
        background: rgba(255,255,255,.08);
        margin: 1.2rem 0;
    }
    .sb-heading {
        font-size: .64rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .12em;
        color: rgba(255,255,255,.35) !important;
        margin-bottom: .65rem;
    }
    .sb-body {
        font-size: .82rem;
        color: rgba(255,255,255,.65) !important;
        line-height: 1.65;
    }
    .sb-tag {
        display: inline-block;
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 4px;
        padding: .2rem .55rem;
        font-size: .68rem;
        color: rgba(255,255,255,.7) !important;
        margin: .15rem;
    }
    .sb-ex-url {
        font-family: var(--ff-m);
        font-size: .69rem;
        padding: .35rem .6rem;
        border-radius: 5px;
        margin-bottom: .28rem;
        word-break: break-all;
        color: rgba(255,255,255,.72) !important;
    }
    .sb-ex-url.ok  { background: rgba(22,163,74,.15);  border: 1px solid rgba(22,163,74,.28); }
    .sb-ex-url.bad { background: rgba(220,38,38,.15);  border: 1px solid rgba(220,38,38,.28); }
    .sb-stats-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: .45rem; }
    .sb-stat-box {
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 7px;
        padding: .6rem .4rem;
        text-align: center;
    }
    .sb-stat-val  { font-size: 1.1rem; font-weight: 700; color: #fff !important; }
    .sb-stat-lbl  { font-size: .6rem; color: rgba(255,255,255,.4) !important; text-transform: uppercase; letter-spacing: .07em; }
    .sb-link-row {
        display: flex;
        align-items: center;
        gap: .55rem;
        padding: .45rem .6rem;
        border-radius: 7px;
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.1);
        margin-bottom: .4rem;
        text-decoration: none;
        transition: background .15s;
    }
    .sb-link-row:hover { background: rgba(255,255,255,.12); }
    .sb-link-text { font-size: .8rem; color: rgba(255,255,255,.8) !important; font-weight: 500; }

    /* Pied de page */
    .pg-footer {
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: .8rem;
    }
    .footer-left { font-size: .77rem; color: var(--text3); line-height: 1.6; }
    .footer-links { display: flex; gap: .6rem; }
    .footer-link {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        padding: .35rem .75rem;
        border-radius: 7px;
        background: var(--surface);
        border: 1px solid var(--border);
        font-size: .78rem;
        color: var(--text2) !important;
        text-decoration: none;
        font-weight: 500;
        transition: border-color .15s, color .15s;
    }
    .footer-link:hover { border-color: var(--navy); color: var(--navy) !important; }
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
    dot_label = "Modele actif" if online else "Chargement en cours..."
    st.markdown(
        f'<div class="pg-header">'
        f'<div class="pg-icon-box">{SVG_SHIELD}</div>'
        f'<div style="flex:1">'
        f'<div class="pg-title">PhishGuard</div>'
        f'<div class="pg-tagline">'
        f'Outil de detection automatique de liens de phishing par traitement du langage naturel'
        f'</div>'
        f'<span class="status-badge {dot_cls}">'
        f'<span class="status-dot"></span>{dot_label}'
        f'</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_about() -> None:
    st.markdown(
        '<div class="about-card">'
        '<div class="about-title">A propos du projet</div>'
        '<p class="about-text">'
        'PhishGuard est un outil d\'analyse automatique de liens web developpe dans le cadre d\'un '
        'projet de recherche en traitement du langage naturel (NLP). Il permet d\'identifier en '
        'quelques secondes si une URL presente des caracteristiques typiques d\'une tentative de '
        'phishing, c\'est-a-dire une usurpation d\'identite numerique visant a derober des '
        'informations confidentielles telles que des mots de passe ou des coordonnees bancaires.'
        '</p>'
        '<p class="about-text" style="margin-top:.7rem">'
        'Le modele analyse trois familles de caracteristiques : les representations TF-IDF au '
        'niveau des mots et des n-grammes de caracteres, ainsi que des indicateurs lexicaux '
        '(longueur de l\'URL, entropie, presence d\'une adresse IP, extension de domaine suspecte). '
        'Il a ete entraine sur plus de 450 000 URLs issues de sources reelles et labelisees.'
        '</p>'
        '<div class="about-stats">'
        '<div class="about-stat"><div class="about-stat-val">450 000+</div><div class="about-stat-lbl">URLs d\'entrainement</div></div>'
        '<div class="about-stat"><div class="about-stat-val">93 %</div><div class="about-stat-lbl">Score F1</div></div>'
        '<div class="about-stat"><div class="about-stat-val">96 %</div><div class="about-stat-lbl">Precision globale</div></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_guide() -> None:
    st.markdown(
        '<div class="guide-card">'
        '<div class="guide-title">Comment utiliser PhishGuard</div>'
        '<div class="guide-steps">'
        '<div class="guide-step">'
        '<div class="guide-num">1</div>'
        '<div class="guide-step-text">'
        '<span class="guide-step-title">Copiez le lien suspect</span>'
        'Recuperez l\'URL depuis un e-mail, un SMS ou toute autre source douteuse.'
        '</div>'
        '</div>'
        '<div class="guide-step">'
        '<div class="guide-num">2</div>'
        '<div class="guide-step-text">'
        '<span class="guide-step-title">Collez et analysez</span>'
        'Collez le lien dans le champ ci-dessous et cliquez sur Analyser.'
        '</div>'
        '</div>'
        '<div class="guide-step">'
        '<div class="guide-num">3</div>'
        '<div class="guide-step-text">'
        '<span class="guide-step-title">Lisez le verdict</span>'
        'Suivez les recommandations affichees et signalez tout lien suspect a votre '
        'service informatique.'
        '</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_result(result: dict, url: str) -> None:
    is_phishing = result["prediction"] == 1
    cls         = "danger" if is_phishing else "success"
    verdict     = "Lien suspect — Phishing probable" if is_phishing else "Lien legitime"
    verdict_sub = (
        "Ce lien presente des caracteristiques typiques d'une tentative de phishing."
        if is_phishing else
        "Aucun indicateur de phishing detecte sur ce lien."
    )
    conf_pct = f"{result['confidence']:.0%}"
    p_legit  = result["proba_legitimate"]
    p_phish  = result["proba_phishing"]
    w_legit  = f"{p_legit * 100:.1f}%"
    w_phish  = f"{p_phish * 100:.1f}%"
    ts       = datetime.fromisoformat(result["timestamp"].replace("Z", "")).strftime("%d/%m/%Y %H:%M")
    msg = (
        "Ne cliquez pas sur ce lien et ne saisissez aucune information personnelle "
        "(identifiant, mot de passe, coordonnees bancaires). Signalez-le immediatement "
        "a votre service informatique ou responsable de la securite."
        if is_phishing else
        "Ce lien semble fiable d'apres l'analyse du modele. Restez neanmoins vigilant : "
        "verifiez toujours l'expediteur avant de saisir des informations sensibles."
    )

    st.markdown(
        f'<div class="result-card {cls}">'
        f'<div class="result-header">'
        f'<div>'
        f'<div class="result-verdict {cls}">{verdict}</div>'
        f'<div class="result-verdict-sub">{verdict_sub}</div>'
        f'</div>'
        f'<div class="result-conf-block">'
        f'<div class="result-conf-label">Indice de confiance</div>'
        f'<div class="result-conf-val {cls}">{conf_pct}</div>'
        f'</div>'
        f'</div>'
        f'<div class="result-url">{url}</div>'
        f'<div class="prob-section">'
        f'<div class="prob-row">'
        f'<div class="prob-meta"><span>Legitime</span><span>{p_legit:.1%}</span></div>'
        f'<div class="prob-track"><div class="prob-fill legit" style="--w:{w_legit}"></div></div>'
        f'</div>'
        f'<div class="prob-row">'
        f'<div class="prob-meta"><span>Phishing</span><span>{p_phish:.1%}</span></div>'
        f'<div class="prob-track"><div class="prob-fill phish" style="--w:{w_phish}"></div></div>'
        f'</div>'
        f'</div>'
        f'<div class="stats-grid">'
        f'<div class="stat-cell"><div class="stat-val">{ts}</div><div class="stat-lbl">Date d\'analyse</div></div>'
        f'<div class="stat-cell"><div class="stat-val">{"Phishing" if is_phishing else "Legitime"}</div><div class="stat-lbl">Classification</div></div>'
        f'<div class="stat-cell"><div class="stat-val">{len(url)}</div><div class="stat-lbl">Longueur de l\'URL</div></div>'
        f'</div>'
        f'<div class="result-msg {cls}">{msg}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_history(history: list) -> None:
    items = list(reversed(history[-10:]))
    rows  = ""
    for item in items:
        lbl      = item["label"]
        ts       = datetime.fromisoformat(item["timestamp"].replace("Z", "")).strftime("%H:%M")
        short    = item["url"][:58] + ("..." if len(item["url"]) > 58 else "")
        conf     = f"{item['confidence']:.0%}"
        label_fr = "Phishing" if lbl == "phishing" else "Legitime"
        rows += (
            f'<tr>'
            f'<td class="hist-url-cell" title="{item["url"]}">{short}</td>'
            f'<td><span class="pill {lbl}"><span class="pill-dot"></span>{label_fr}</span></td>'
            f'<td style="font-weight:600;color:var(--text)">{conf}</td>'
            f'<td style="color:var(--text3)">{ts}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="hist-table"><thead><tr>'
        f'<th>URL analysee</th><th>Verdict</th><th>Confiance</th><th>Heure</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_sidebar(history: list) -> None:
    total   = len(history)
    threats = sum(1 for h in history if h["label"] == "phishing")
    safe    = total - threats

    # Stats de session
    st.markdown(
        f'<div class="sb-section">'
        f'<div class="sb-heading">Statistiques de session</div>'
        f'<div class="sb-stats-grid">'
        f'<div class="sb-stat-box"><div class="sb-stat-val">{total}</div><div class="sb-stat-lbl">Analyses</div></div>'
        f'<div class="sb-stat-box"><div class="sb-stat-val">{threats}</div><div class="sb-stat-lbl">Suspects</div></div>'
        f'<div class="sb-stat-box"><div class="sb-stat-val">{safe}</div><div class="sb-stat-lbl">Surs</div></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Modele
    st.markdown('<div class="sb-heading">Technologies du modele</div>', unsafe_allow_html=True)
    for tag in ["TF-IDF mots", "TF-IDF n-grammes", "Features lexicales", "Regression logistique"]:
        st.markdown(f'<span class="sb-tag">{tag}</span>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Exemples
    st.markdown(
        '<div class="sb-heading">Exemples d\'URLs</div>'
        '<div style="font-size:.66rem;color:rgba(255,255,255,.35);text-transform:uppercase;'
        'letter-spacing:.07em;margin-bottom:.3rem">Legitimes</div>'
        '<div class="sb-ex-url ok">https://www.google.com</div>'
        '<div class="sb-ex-url ok">https://github.com/login</div>'
        '<div style="font-size:.66rem;color:rgba(255,255,255,.35);text-transform:uppercase;'
        'letter-spacing:.07em;margin:.5rem 0 .3rem">Suspects</div>'
        '<div class="sb-ex-url bad">http://paypal-secure.tk/login.php</div>'
        '<div class="sb-ex-url bad">http://192.168.1.1/bank-verify</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Liens
    st.markdown(
        '<div class="sb-heading">Liens</div>'
        f'<a class="sb-link-row" href="{LINKEDIN_URL}" target="_blank">'
        f'{SVG_LINKEDIN}'
        f'<span class="sb-link-text">Souleymane Sall — LinkedIn</span>'
        f'</a>'
        f'<a class="sb-link-row" href="{GITHUB_URL}" target="_blank">'
        f'{SVG_GITHUB}'
        f'<span class="sb-link-text">Code source — GitHub</span>'
        f'</a>',
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
    render_about()
    render_guide()

    # Zone de saisie
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Analyser une URL</div>', unsafe_allow_html=True)

    with st.form(key="scan_form", clear_on_submit=False):
        url_input = st.text_input(
            label="url",
            placeholder="Collez ici l'URL complete a verifier — ex. https://example.com/page",
            key="url_field",
        )
        st.markdown(
            '<div class="input-hint">'
            'Incluez le protocole complet (http:// ou https://). '
            'L\'outil analyse la structure linguistique de l\'URL, sans y acceder.'
            '</div>',
            unsafe_allow_html=True,
        )
        col1, col2, _ = st.columns([1.4, 1, 4])
        with col1:
            submitted = st.form_submit_button("Analyser", type="primary", use_container_width=True)
        with col2:
            clear = st.form_submit_button("Effacer", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if clear:
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_url", None)
        st.rerun()

    if submitted:
        if not url_input.strip():
            st.warning("Veuillez saisir une URL avant de lancer l'analyse.")
        elif not online:
            st.warning("Le modele est en cours de chargement. Veuillez patienter quelques instants.")
        else:
            with st.spinner("Analyse en cours..."):
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
        st.markdown(
            '<div class="section-label" style="margin-top:1.4rem">Resultat de l\'analyse</div>',
            unsafe_allow_html=True,
        )
        render_result(st.session_state["last_result"], st.session_state["last_url"])

    # Historique
    history = st.session_state.get("history", [])
    if history:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Historique de la session</div>', unsafe_allow_html=True)
        render_history(history)
        st.markdown('<div style="margin-top:.7rem"></div>', unsafe_allow_html=True)
        if st.button("Effacer l'historique"):
            st.session_state.history = []
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_url", None)
            st.rerun()

    # Pied de page
    st.markdown(
        f'<div class="pg-footer">'
        f'<div class="footer-left">'
        f'PhishGuard — Detection de phishing par NLP<br>'
        f'Souleymane Sall &middot; scikit-learn &middot; Streamlit'
        f'</div>'
        f'<div class="footer-links">'
        f'<a class="footer-link" href="{LINKEDIN_URL}" target="_blank">'
        f'{SVG_LINKEDIN} LinkedIn'
        f'</a>'
        f'<a class="footer-link" href="{GITHUB_URL}" target="_blank">'
        f'{SVG_GITHUB} GitHub'
        f'</a>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
