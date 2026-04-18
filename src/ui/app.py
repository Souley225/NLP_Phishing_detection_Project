"""
PhishGuard — Outil de détection de phishing par analyse d'URL.
Interface mobile-first en français, adaptée à un usage entreprise.

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
    'width="20" height="20" fill="white" aria-hidden="true">'
    '<path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>'
    '</svg>'
)

SVG_LINKEDIN = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'width="15" height="15" fill="currentColor" aria-hidden="true">'
    '<path d="M19 0H5C2.239 0 0 2.239 0 5v14c0 2.761 2.239 5 5 5h14c2.762 0 '
    '5-2.239 5-5V5c0-2.761-2.238-5-5-5zM8 19H5V8h3v11zM6.5 6.732c-.966 0-1.75'
    '-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 '
    '1.764zM20 19h-3v-5.604c0-3.368-4-3.113-4 0V19h-3V8h3v1.765C14.396 7.179 '
    '20 6.988 20 12.248V19z"/>'
    '</svg>'
)

SVG_GITHUB = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'width="15" height="15" fill="currentColor" aria-hidden="true">'
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


# ── CSS mobile-first ──────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Source+Code+Pro:wght@400;500&display=swap');

    /* ── Variables ── */
    :root {
        --navy:       #1b2f4e;
        --navy-h:     #243a5e;
        --navy-d:     #0f1f35;
        --blue:       #2563eb;
        --blue-l:     #eff6ff;
        --blue-b:     #bfdbfe;
        --red:        #dc2626;
        --red-l:      #fff1f1;
        --red-b:      #fecaca;
        --red-t:      #fef2f2;
        --green:      #16a34a;
        --green-l:    #f0fdf4;
        --green-b:    #bbf7d0;
        --green-t:    #f0fdf4;
        --amber:      #d97706;
        --bg:         #f6f8fb;
        --surface:    #ffffff;
        --border:     #e4e9f0;
        --border2:    #c8d3e0;
        --text:       #0f172a;
        --text2:      #475569;
        --text3:      #94a3b8;
        --ff:         'Plus Jakarta Sans', sans-serif;
        --ff-m:       'Source Code Pro', monospace;
        --r:          12px;
        --r-sm:       8px;
        --sh:         0 1px 3px rgba(15,23,42,.06), 0 4px 16px rgba(15,23,42,.04);
        --sh2:        0 2px 8px rgba(15,23,42,.08), 0 12px 40px rgba(15,23,42,.07);
        --sh-danger:  0 4px 24px rgba(220,38,38,.12);
        --sh-success: 0 4px 24px rgba(22,163,74,.10);
    }

    /* ── Reset & base ── */
    html, body, .stApp {
        font-family: var(--ff) !important;
        background: var(--bg) !important;
        color: var(--text) !important;
        -webkit-text-size-adjust: 100%;
    }

    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }

    /* ── Container — mobile-first ── */
    .main .block-container {
        padding: 1rem 1rem 3rem !important;
        max-width: 100% !important;
    }
    @media (min-width: 640px) {
        .main .block-container {
            padding: 1.5rem 1.5rem 3rem !important;
        }
    }
    @media (min-width: 960px) {
        .main .block-container {
            padding: 2rem 2.5rem 4rem !important;
            max-width: 920px !important;
            margin: 0 auto !important;
        }
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--navy) !important;
    }
    [data-testid="stSidebar"] * { color: rgba(255,255,255,.85) !important; }
    [data-testid="stSidebar"] .stMarkdown p {
        color: rgba(255,255,255,.62) !important;
        font-size: .84rem !important;
        line-height: 1.65 !important;
    }

    /* ── Typography ── */
    h1, h2, h3 {
        font-family: var(--ff) !important;
        font-weight: 700 !important;
        color: var(--navy) !important;
        letter-spacing: -.015em !important;
    }

    /* ── Input — 16px prevents iOS zoom ── */
    .stTextInput label { display: none !important; }
    .stTextInput > div > div > input {
        background: var(--surface) !important;
        border: 1.5px solid var(--border2) !important;
        border-radius: var(--r) !important;
        color: var(--text) !important;
        font-family: var(--ff-m) !important;
        font-size: 16px !important;
        padding: .9rem 1rem !important;
        min-height: 52px !important;
        box-shadow: var(--sh) !important;
        transition: border-color .18s, box-shadow .18s !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,.14) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: var(--text3) !important;
        font-family: var(--ff) !important;
        font-size: .9rem !important;
    }

    /* ── Buttons — min 48px touch targets ── */
    .stButton > button,
    .stFormSubmitButton > button {
        font-family: var(--ff) !important;
        font-weight: 600 !important;
        font-size: .95rem !important;
        letter-spacing: -.01em !important;
        border-radius: var(--r) !important;
        min-height: 48px !important;
        padding: .7rem 1.2rem !important;
        transition: all .18s !important;
        width: 100% !important;
    }
    .stFormSubmitButton > button[kind="primary"] {
        background: var(--navy) !important;
        color: #fff !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(27,47,78,.25) !important;
    }
    .stFormSubmitButton > button[kind="primary"]:hover {
        background: var(--navy-h) !important;
        box-shadow: 0 4px 16px rgba(27,47,78,.3) !important;
        transform: translateY(-1px) !important;
    }
    .stFormSubmitButton > button[kind="primary"]:active {
        transform: translateY(0) !important;
        box-shadow: 0 1px 4px rgba(27,47,78,.2) !important;
    }
    .stFormSubmitButton > button:not([kind="primary"]) {
        background: var(--surface) !important;
        color: var(--text2) !important;
        border: 1.5px solid var(--border2) !important;
    }
    .stFormSubmitButton > button:not([kind="primary"]):hover {
        background: var(--bg) !important;
        border-color: var(--text2) !important;
        color: var(--text) !important;
    }
    .stButton > button {
        background: var(--surface) !important;
        color: var(--text2) !important;
        border: 1.5px solid var(--border2) !important;
    }
    .stButton > button:hover {
        background: var(--bg) !important;
        border-color: var(--navy) !important;
        color: var(--navy) !important;
    }

    /* ── Columns adapt on mobile ── */
    @media (max-width: 480px) {
        [data-testid="column"] {
            min-width: 0 !important;
        }
    }

    /* ── Alerts ── */
    .stWarning {
        background: #fffbeb !important;
        border: 1.5px solid #fde68a !important;
        border-radius: var(--r) !important;
        color: #78350f !important;
        font-size: .9rem !important;
        padding: .85rem 1rem !important;
    }
    .stError {
        background: var(--red-l) !important;
        border: 1.5px solid var(--red-b) !important;
        border-radius: var(--r) !important;
        font-size: .9rem !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.6rem 0 !important;
    }

    /* ════════════════════════════════
       COMPOSANTS MAISON
    ════════════════════════════════ */

    .section-label {
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: var(--text3);
        margin-bottom: .65rem;
    }

    /* ── En-tête ── */
    .pg-header {
        background: linear-gradient(135deg, var(--navy-d) 0%, var(--navy) 60%, var(--navy-h) 100%);
        border-radius: var(--r);
        padding: 1.4rem 1.3rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        position: relative;
        overflow: hidden;
    }
    .pg-header::after {
        content: '';
        position: absolute;
        right: -30px; top: -30px;
        width: 130px; height: 130px;
        border-radius: 50%;
        background: rgba(255,255,255,.04);
        pointer-events: none;
    }
    @media (min-width: 640px) {
        .pg-header { padding: 1.8rem 2rem; }
        .pg-header::after { width: 200px; height: 200px; right: -50px; top: -50px; }
    }
    .pg-icon-box {
        width: 44px; height: 44px;
        background: rgba(255,255,255,.13);
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    @media (min-width: 640px) {
        .pg-icon-box { width: 50px; height: 50px; }
    }
    .pg-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #fff;
        letter-spacing: -.025em;
        line-height: 1.1;
        margin-bottom: .22rem;
    }
    @media (min-width: 640px) { .pg-title { font-size: 1.6rem; } }
    .pg-tagline {
        font-size: .78rem;
        color: rgba(255,255,255,.58);
        line-height: 1.5;
        margin-bottom: .75rem;
    }
    @media (min-width: 640px) { .pg-tagline { font-size: .86rem; } }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: .38rem;
        padding: .3rem .75rem;
        border-radius: 999px;
        font-size: .72rem;
        font-weight: 600;
        letter-spacing: .01em;
    }
    .status-badge.online {
        background: rgba(22,163,74,.22);
        color: #86efac;
        border: 1px solid rgba(22,163,74,.35);
    }
    .status-badge.offline {
        background: rgba(217,119,6,.2);
        color: #fcd34d;
        border: 1px solid rgba(217,119,6,.3);
    }
    .status-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: currentColor;
        flex-shrink: 0;
    }

    /* ── Carte à propos ── */
    .about-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--r);
        padding: 1.25rem 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--sh);
    }
    @media (min-width: 640px) {
        .about-card { padding: 1.5rem 1.6rem; }
    }
    .card-title {
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: var(--navy);
        margin-bottom: .65rem;
        padding-bottom: .55rem;
        border-bottom: 1.5px solid var(--border);
    }
    .about-text {
        font-size: .875rem;
        color: var(--text2);
        line-height: 1.72;
        margin: 0 0 .7rem;
    }
    .about-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .6rem;
        margin-top: 1rem;
    }
    .about-stat {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--r-sm);
        padding: .65rem .5rem;
        text-align: center;
    }
    .about-stat-val {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--navy);
        line-height: 1.15;
    }
    @media (min-width: 640px) { .about-stat-val { font-size: 1.35rem; } }
    .about-stat-lbl {
        font-size: .62rem;
        color: var(--text3);
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-top: .15rem;
        line-height: 1.3;
    }

    /* ── Guide ── */
    .guide-card {
        background: var(--blue-l);
        border: 1px solid var(--blue-b);
        border-radius: var(--r);
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    @media (min-width: 640px) { .guide-card { padding: 1.35rem 1.5rem; } }
    .guide-title {
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: var(--blue);
        margin-bottom: .9rem;
    }
    .guide-steps {
        display: flex;
        flex-direction: column;
        gap: .85rem;
    }
    @media (min-width: 560px) {
        .guide-steps { flex-direction: row; gap: .8rem; }
    }
    .guide-step {
        display: flex;
        gap: .7rem;
        align-items: flex-start;
        flex: 1;
    }
    .guide-num {
        width: 28px; height: 28px;
        border-radius: 50%;
        background: var(--blue);
        color: #fff;
        font-size: .78rem;
        font-weight: 700;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .guide-step-text { font-size: .84rem; color: #1e40af; line-height: 1.55; }
    .guide-step-title {
        font-weight: 700;
        display: block;
        color: #1e3a8a;
        margin-bottom: .1rem;
    }

    /* ── Zone saisie ── */
    .input-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--r);
        padding: 1.2rem 1.2rem .8rem;
        margin-bottom: 1.1rem;
        box-shadow: var(--sh);
    }
    @media (min-width: 640px) { .input-card { padding: 1.4rem 1.5rem 1rem; } }
    .input-hint {
        font-size: .78rem;
        color: var(--text3);
        margin-top: .4rem;
        line-height: 1.55;
    }

    /* ── Carte résultat ── */
    @keyframes resultIn {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-card {
        background: var(--surface);
        border-radius: var(--r);
        border: 1px solid var(--border);
        overflow: hidden;
        margin: .8rem 0 1.2rem;
        animation: resultIn .35s ease both;
    }
    .result-card.danger {
        box-shadow: var(--sh-danger);
        border-color: var(--red-b);
    }
    .result-card.success {
        box-shadow: var(--sh-success);
        border-color: var(--green-b);
    }
    /* Barre de couleur pleine largeur en haut */
    .result-topbar {
        height: 5px;
        width: 100%;
    }
    .result-topbar.danger  { background: var(--red); }
    .result-topbar.success { background: var(--green); }

    .result-body { padding: 1.2rem 1.1rem 1.1rem; }
    @media (min-width: 640px) { .result-body { padding: 1.5rem 1.4rem 1.3rem; } }

    /* Verdict + confiance : colonne sur mobile, ligne sur desktop */
    .result-header {
        display: flex;
        flex-direction: column;
        gap: .65rem;
        margin-bottom: 1.1rem;
    }
    @media (min-width: 480px) {
        .result-header {
            flex-direction: row;
            align-items: flex-start;
            justify-content: space-between;
        }
    }
    .result-verdict { font-size: 1.15rem; font-weight: 800; line-height: 1.2; }
    @media (min-width: 640px) { .result-verdict { font-size: 1.25rem; } }
    .result-verdict.danger  { color: var(--red); }
    .result-verdict.success { color: var(--green); }
    .result-verdict-sub {
        font-size: .82rem;
        color: var(--text2);
        margin-top: .22rem;
        line-height: 1.45;
    }
    .result-conf-block {
        display: flex;
        align-items: center;
        gap: .55rem;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--r-sm);
        padding: .55rem .85rem;
        align-self: flex-start;
        flex-shrink: 0;
    }
    @media (min-width: 480px) { .result-conf-block { flex-direction: column; align-items: flex-end; gap: .1rem; padding: .5rem .8rem; } }
    .result-conf-val { font-size: 1.9rem; font-weight: 800; line-height: 1; }
    @media (min-width: 640px) { .result-conf-val { font-size: 2.2rem; } }
    .result-conf-val.danger  { color: var(--red); }
    .result-conf-val.success { color: var(--green); }
    .result-conf-label {
        font-size: .65rem;
        color: var(--text3);
        text-transform: uppercase;
        letter-spacing: .07em;
    }

    /* URL analysée */
    .result-url {
        font-family: var(--ff-m);
        font-size: .8rem;
        color: var(--text2);
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--r-sm);
        padding: .55rem .85rem;
        margin-bottom: 1.1rem;
        word-break: break-all;
        line-height: 1.5;
    }

    /* Barres de probabilité */
    .prob-section { margin-bottom: 1rem; }
    .prob-row { margin-bottom: .6rem; }
    .prob-meta {
        display: flex;
        justify-content: space-between;
        font-size: .8rem;
        color: var(--text2);
        margin-bottom: .3rem;
        font-weight: 500;
    }
    .prob-track {
        height: 10px;
        background: var(--bg);
        border-radius: 5px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .prob-fill { height: 100%; border-radius: 5px; width: var(--w); }
    .prob-fill.legit { background: linear-gradient(90deg, #16a34a, #4ade80); }
    .prob-fill.phish { background: linear-gradient(90deg, #dc2626, #f87171); }

    /* Grille stats résultat */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .55rem;
        margin-bottom: .9rem;
    }
    .stat-cell {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--r-sm);
        padding: .6rem .5rem;
        text-align: center;
    }
    .stat-val {
        font-size: .9rem;
        font-weight: 700;
        color: var(--navy);
        line-height: 1.25;
        word-break: break-word;
    }
    @media (min-width: 640px) { .stat-val { font-size: 1rem; } }
    .stat-lbl {
        font-size: .6rem;
        color: var(--text3);
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-top: .15rem;
    }

    /* Message conseil */
    .result-msg {
        font-size: .86rem;
        border-radius: var(--r-sm);
        padding: .9rem 1rem;
        line-height: 1.65;
    }
    .result-msg.danger {
        background: var(--red-t);
        border: 1px solid var(--red-b);
        color: #7f1d1d;
    }
    .result-msg.success {
        background: var(--green-t);
        border: 1px solid var(--green-b);
        color: #14532d;
    }

    /* ── Historique — cartes sur mobile ── */
    .hist-list { display: flex; flex-direction: column; gap: .55rem; }
    .hist-item {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--r-sm);
        padding: .75rem 1rem;
        display: flex;
        align-items: center;
        gap: .75rem;
        min-height: 52px;
    }
    .hist-item:hover { background: var(--bg); }
    .hist-verdict-bar {
        width: 3px;
        min-height: 36px;
        border-radius: 2px;
        flex-shrink: 0;
        align-self: stretch;
    }
    .hist-verdict-bar.phishing  { background: var(--red); }
    .hist-verdict-bar.legitimate{ background: var(--green); }
    .hist-main { flex: 1; min-width: 0; }
    .hist-url {
        font-family: var(--ff-m);
        font-size: .77rem;
        color: var(--text2);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: .2rem;
    }
    .hist-meta {
        display: flex;
        align-items: center;
        gap: .5rem;
        flex-wrap: wrap;
    }
    .pill {
        display: inline-flex; align-items: center; gap: .28rem;
        padding: .18rem .6rem;
        border-radius: 999px;
        font-size: .7rem;
        font-weight: 600;
    }
    .pill.phishing  { background: var(--red-l);   color: var(--red);   border: 1px solid var(--red-b);   }
    .pill.legitimate{ background: var(--green-l); color: var(--green); border: 1px solid var(--green-b); }
    .pill-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
    .hist-conf { font-size: .75rem; font-weight: 700; color: var(--text); }
    .hist-time { font-size: .72rem; color: var(--text3); }

    /* ── Sidebar ── */
    .sb-divider { height: 1px; background: rgba(255,255,255,.07); margin: 1.1rem 0; }
    .sb-heading {
        font-size: .63rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: .12em;
        color: rgba(255,255,255,.32) !important;
        margin-bottom: .65rem;
    }
    .sb-body { font-size: .82rem; color: rgba(255,255,255,.62) !important; line-height: 1.65; }

    /* Tags tech */
    .sb-tag {
        display: inline-block;
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 4px;
        padding: .22rem .55rem;
        font-size: .68rem;
        color: rgba(255,255,255,.72) !important;
        margin: .15rem;
    }

    /* Exemples d'URLs */
    .sb-ex-url {
        font-family: var(--ff-m);
        font-size: .68rem;
        padding: .38rem .65rem;
        border-radius: 6px;
        margin-bottom: .28rem;
        word-break: break-all;
        color: rgba(255,255,255,.7) !important;
    }
    .sb-ex-url.ok  { background: rgba(22,163,74,.14);  border: 1px solid rgba(22,163,74,.25); }
    .sb-ex-url.bad { background: rgba(220,38,38,.14);  border: 1px solid rgba(220,38,38,.25); }

    /* Grille stats session */
    .sb-stats-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: .4rem; }
    .sb-stat-box {
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 8px; padding: .6rem .35rem; text-align: center;
    }
    .sb-stat-val { font-size: 1.15rem; font-weight: 800; color: #fff !important; }
    .sb-stat-lbl { font-size: .58rem; color: rgba(255,255,255,.38) !important; text-transform: uppercase; letter-spacing: .07em; }

    /* Métriques modèle */
    .sb-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: .42rem 0;
        border-bottom: 1px solid rgba(255,255,255,.05);
    }
    .sb-metric-row:last-child { border-bottom: none; }
    .sb-metric-key { font-size: .77rem; color: rgba(255,255,255,.52) !important; }
    .sb-metric-val { font-size: .82rem; font-weight: 700; color: rgba(255,255,255,.9) !important; font-family: var(--ff-m); }
    .sb-metric-val.accent { color: #93c5fd !important; }

    /* Pipeline */
    .sb-pipeline { display: flex; flex-direction: column; gap: .32rem; }
    .sb-pipe-step {
        display: flex; align-items: center; gap: .6rem;
        padding: .42rem .6rem;
        border-radius: 6px;
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.07);
    }
    .sb-pipe-num {
        width: 18px; height: 18px;
        border-radius: 50%;
        background: rgba(147,197,253,.18);
        border: 1px solid rgba(147,197,253,.3);
        color: #93c5fd !important;
        font-size: .62rem; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .sb-pipe-label { font-size: .74rem; color: rgba(255,255,255,.68) !important; line-height: 1.3; }
    .sb-pipe-sub { font-size: .65rem; color: rgba(255,255,255,.35) !important; display: block; }

    /* Stack technique — items avec point */
    .sb-stack-list { display: flex; flex-direction: column; gap: .28rem; }
    .sb-stack-item {
        display: flex; align-items: flex-start; gap: .55rem;
        padding: .38rem .6rem;
        border-radius: 6px;
        background: rgba(255,255,255,.04);
    }
    .sb-stack-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #93c5fd;
        flex-shrink: 0;
        margin-top: .38rem;
    }
    .sb-stack-name { font-size: .76rem; font-weight: 600; color: rgba(255,255,255,.82) !important; }
    .sb-stack-desc { font-size: .67rem; color: rgba(255,255,255,.38) !important; display: block; }

    /* Liens */
    .sb-link-row {
        display: flex; align-items: center; gap: .55rem;
        padding: .6rem .75rem;
        border-radius: 8px;
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.1);
        margin-bottom: .4rem;
        text-decoration: none;
        min-height: 44px;
        transition: background .15s;
    }
    .sb-link-row:hover { background: rgba(255,255,255,.13); }
    .sb-link-text { font-size: .82rem; color: rgba(255,255,255,.8) !important; font-weight: 500; }

    /* ── Pied de page ── */
    .pg-footer {
        margin-top: 2.5rem;
        padding-top: 1.1rem;
        border-top: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        gap: .9rem;
        align-items: flex-start;
    }
    @media (min-width: 560px) {
        .pg-footer {
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
        }
    }
    .footer-left { font-size: .76rem; color: var(--text3); line-height: 1.65; }
    .footer-links { display: flex; gap: .55rem; }
    .footer-link {
        display: inline-flex; align-items: center; gap: .4rem;
        padding: .5rem .85rem;
        border-radius: var(--r-sm);
        background: var(--surface);
        border: 1.5px solid var(--border);
        font-size: .8rem;
        color: var(--text2) !important;
        text-decoration: none;
        font-weight: 500;
        min-height: 40px;
        transition: border-color .15s, color .15s, background .15s;
    }
    .footer-link:hover {
        border-color: var(--navy);
        color: var(--navy) !important;
        background: var(--blue-l);
    }
    </style>
    """, unsafe_allow_html=True)


# ── Chargement du modèle ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modele de detection...")
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


def _normalize_url(url: str) -> str:
    """Ajoute https:// si l'URL n'a pas de scheme — urlparse en a besoin pour parser correctement."""
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def predict_url(url: str) -> dict | None:
    try:
        url = _normalize_url(url)
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

        prediction = 1 if proba_phish >= threshold else 0
        confidence = proba_phish if prediction == 1 else proba_legit

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
    dot_label = "Modele actif" if online else "Chargement..."
    st.markdown(
        f'<div class="pg-header">'
        f'<div class="pg-icon-box">{SVG_SHIELD}</div>'
        f'<div style="flex:1;min-width:0">'
        f'<div class="pg-title">PhishGuard</div>'
        f'<div class="pg-tagline">Detection automatique de liens de phishing par NLP</div>'
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
        '<div class="card-title">A propos du projet</div>'
        '<p class="about-text">'
        'PhishGuard analyse automatiquement la structure d\'une URL pour determiner si elle '
        'presente des caracteristiques associees au phishing — une technique de fraude visant '
        'a usurper l\'identite d\'un service de confiance pour derober des informations '
        'personnelles (mots de passe, donnees bancaires).'
        '</p>'
        '<p class="about-text">'
        'Le modele s\'appuie sur le traitement du langage naturel : representations TF-IDF '
        'des mots et n-grammes de caracteres, indicateurs lexicaux (entropie, longueur, '
        'extension de domaine). Il ne necessite pas d\'acceder au contenu du lien.'
        '</p>'
        '<div class="about-stats">'
        '<div class="about-stat">'
        '<div class="about-stat-val">450k+</div>'
        '<div class="about-stat-lbl">URLs entrainement</div>'
        '</div>'
        '<div class="about-stat">'
        '<div class="about-stat-val">93&thinsp;%</div>'
        '<div class="about-stat-lbl">Score F1</div>'
        '</div>'
        '<div class="about-stat">'
        '<div class="about-stat-val">96&thinsp;%</div>'
        '<div class="about-stat-lbl">Precision</div>'
        '</div>'
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
        '<span class="guide-step-title">Copiez le lien</span>'
        'Recuperez l\'URL depuis un e-mail, un SMS ou tout autre message suspect.'
        '</div>'
        '</div>'
        '<div class="guide-step">'
        '<div class="guide-num">2</div>'
        '<div class="guide-step-text">'
        '<span class="guide-step-title">Collez et analysez</span>'
        'Collez le lien dans le champ ci-dessous, puis appuyez sur Analyser.'
        '</div>'
        '</div>'
        '<div class="guide-step">'
        '<div class="guide-num">3</div>'
        '<div class="guide-step-text">'
        '<span class="guide-step-title">Lisez le verdict</span>'
        'Suivez les recommandations. En cas de doute, contactez votre equipe informatique.'
        '</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_result(result: dict, url: str) -> None:
    is_phishing = result["prediction"] == 1
    cls         = "danger" if is_phishing else "success"
    verdict     = "Lien signale comme suspect" if is_phishing else "Aucun indicateur suspect detecte"
    verdict_sub = (
        "Le modele a releve des caracteristiques associees au phishing. Une verification manuelle est recommandee."
        if is_phishing else
        "Le modele n'a pas identifie de caracteristiques suspectes sur cette URL."
    )
    conf_pct = f"{result['confidence']:.0%}"
    p_legit  = result["proba_legitimate"]
    p_phish  = result["proba_phishing"]
    w_legit  = f"{p_legit * 100:.1f}%"
    w_phish  = f"{p_phish * 100:.1f}%"
    ts       = datetime.fromisoformat(result["timestamp"].replace("Z", "")).strftime("%d/%m %H:%M")
    msg = (
        "Par mesure de precaution, nous vous conseillons de verifier l'origine de ce lien "
        "avant de cliquer. Si vous l'avez recu de maniere inattendue, confirmez son authenticite "
        "aupres de l'expediteur par un autre canal. En cas de doute persistant, votre equipe "
        "informatique pourra vous aider. Aucun outil automatique n'est infaillible."
        if is_phishing else
        "Ce lien ne presente pas d'indicateurs de phishing connus. Gardez neanmoins le reflexe "
        "de verifier l'expediteur et le contexte avant de saisir des informations sensibles. "
        "Aucun outil automatique n'offre une garantie absolue."
    )

    st.markdown(
        f'<div class="result-card {cls}">'
        f'<div class="result-topbar {cls}"></div>'
        f'<div class="result-body">'
        f'<div class="result-header">'
        f'<div style="min-width:0">'
        f'<div class="result-verdict {cls}">{verdict}</div>'
        f'<div class="result-verdict-sub">{verdict_sub}</div>'
        f'</div>'
        f'<div class="result-conf-block">'
        f'<div class="result-conf-label">Confiance</div>'
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
        f'<div class="stat-cell"><div class="stat-val">{ts}</div><div class="stat-lbl">Date</div></div>'
        f'<div class="stat-cell"><div class="stat-val">{"Phishing" if is_phishing else "Legitime"}</div><div class="stat-lbl">Verdict</div></div>'
        f'<div class="stat-cell"><div class="stat-val">{len(url)}</div><div class="stat-lbl">Longueur URL</div></div>'
        f'</div>'
        f'<div class="result-msg {cls}">{msg}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_history(history: list) -> None:
    """Affiche l'historique sous forme de cartes — lisibles sur mobile."""
    items = list(reversed(history[-10:]))
    rows  = ""
    for item in items:
        lbl      = item["label"]
        ts       = datetime.fromisoformat(item["timestamp"].replace("Z", "")).strftime("%H:%M")
        short    = item["url"][:52] + ("..." if len(item["url"]) > 52 else "")
        conf     = f"{item['confidence']:.0%}"
        label_fr = "Phishing" if lbl == "phishing" else "Legitime"
        rows += (
            f'<div class="hist-item">'
            f'<div class="hist-verdict-bar {lbl}"></div>'
            f'<div class="hist-main">'
            f'<div class="hist-url" title="{item["url"]}">{short}</div>'
            f'<div class="hist-meta">'
            f'<span class="pill {lbl}"><span class="pill-dot"></span>{label_fr}</span>'
            f'<span class="hist-conf">{conf}</span>'
            f'<span class="hist-time">{ts}</span>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
    st.markdown(f'<div class="hist-list">{rows}</div>', unsafe_allow_html=True)


def render_sidebar(history: list) -> None:
    total   = len(history)
    threats = sum(1 for h in history if h["label"] == "phishing")
    safe    = total - threats

    # ── Session ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="sb-heading">Session en cours</div>'
        f'<div class="sb-stats-grid">'
        f'<div class="sb-stat-box"><div class="sb-stat-val">{total}</div><div class="sb-stat-lbl">Analyses</div></div>'
        f'<div class="sb-stat-box"><div class="sb-stat-val">{threats}</div><div class="sb-stat-lbl">Suspects</div></div>'
        f'<div class="sb-stat-box"><div class="sb-stat-val">{safe}</div><div class="sb-stat-lbl">Surs</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Modèle ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-heading">Modele de detection</div>'
        '<div class="sb-metric-row"><span class="sb-metric-key">Algorithme</span>'
        '<span class="sb-metric-val accent">Regression logistique</span></div>'
        '<div class="sb-metric-row"><span class="sb-metric-key">Seuil de decision</span>'
        '<span class="sb-metric-val">0.40</span></div>'
        '<div class="sb-metric-row"><span class="sb-metric-key">Score F1</span>'
        '<span class="sb-metric-val">92.7 %</span></div>'
        '<div class="sb-metric-row"><span class="sb-metric-key">Precision</span>'
        '<span class="sb-metric-val">93.6 %</span></div>'
        '<div class="sb-metric-row"><span class="sb-metric-key">Rappel</span>'
        '<span class="sb-metric-val">91.7 %</span></div>'
        '<div class="sb-metric-row"><span class="sb-metric-key">URLs d\'entrainement</span>'
        '<span class="sb-metric-val">450 000+</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Pipeline ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-heading">Pipeline NLP</div>'
        '<div class="sb-pipeline">'
        '<div class="sb-pipe-step"><div class="sb-pipe-num">1</div>'
        '<div class="sb-pipe-label">Tokenisation<span class="sb-pipe-sub">Decomposition de l\'URL en tokens</span></div></div>'
        '<div class="sb-pipe-step"><div class="sb-pipe-num">2</div>'
        '<div class="sb-pipe-label">TF-IDF mots<span class="sb-pipe-sub">Bigrammes de tokens (50 000 features)</span></div></div>'
        '<div class="sb-pipe-step"><div class="sb-pipe-num">3</div>'
        '<div class="sb-pipe-label">TF-IDF caracteres<span class="sb-pipe-sub">N-grammes 2–4 chars (100 000 features)</span></div></div>'
        '<div class="sb-pipe-step"><div class="sb-pipe-num">4</div>'
        '<div class="sb-pipe-label">Features lexicales<span class="sb-pipe-sub">Entropie, longueur, TLD, IP, tirets…</span></div></div>'
        '<div class="sb-pipe-step"><div class="sb-pipe-num">5</div>'
        '<div class="sb-pipe-label">Classification<span class="sb-pipe-sub">Score de probabilite + seuil 0.40</span></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Stack technique ───────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-heading">Stack technique</div>'
        '<div class="sb-stack-list">'
        '<div class="sb-stack-item"><div class="sb-stack-dot"></div><div>'
        '<div class="sb-stack-name">Python 3.11</div>'
        '<span class="sb-stack-desc">Langage principal</span></div></div>'
        '<div class="sb-stack-item"><div class="sb-stack-dot"></div><div>'
        '<div class="sb-stack-name">scikit-learn</div>'
        '<span class="sb-stack-desc">Modelisation et vectorisation TF-IDF</span></div></div>'
        '<div class="sb-stack-item"><div class="sb-stack-dot"></div><div>'
        '<div class="sb-stack-name">scipy.sparse</div>'
        '<span class="sb-stack-desc">Matrices creuses pour les features combinees</span></div></div>'
        '<div class="sb-stack-item"><div class="sb-stack-dot"></div><div>'
        '<div class="sb-stack-name">FastAPI</div>'
        '<span class="sb-stack-desc">API REST de prediction (port 8000)</span></div></div>'
        '<div class="sb-stack-item"><div class="sb-stack-dot"></div><div>'
        '<div class="sb-stack-name">Streamlit</div>'
        '<span class="sb-stack-desc">Interface utilisateur web</span></div></div>'
        '<div class="sb-stack-item"><div class="sb-stack-dot"></div><div>'
        '<div class="sb-stack-name">Hugging Face Spaces</div>'
        '<span class="sb-stack-desc">Hebergement cloud (Docker)</span></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Exemples ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-heading">Exemples d\'URLs</div>'
        '<div style="font-size:.63rem;color:rgba(255,255,255,.32);text-transform:uppercase;'
        'letter-spacing:.08em;margin-bottom:.3rem">Legitimes</div>'
        '<div class="sb-ex-url ok">https://www.google.com</div>'
        '<div class="sb-ex-url ok">https://github.com/login</div>'
        '<div style="font-size:.63rem;color:rgba(255,255,255,.32);text-transform:uppercase;'
        'letter-spacing:.08em;margin:.5rem 0 .3rem">Suspects</div>'
        '<div class="sb-ex-url bad">http://paypal-secure.tk/login.php</div>'
        '<div class="sb-ex-url bad">http://192.168.1.1/bank-verify</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Liens ─────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="sb-heading">Liens</div>'
        f'<a class="sb-link-row" href="{LINKEDIN_URL}" target="_blank" rel="noopener">'
        f'{SVG_LINKEDIN}'
        f'<span class="sb-link-text">Souleymane Sall — LinkedIn</span>'
        f'</a>'
        f'<a class="sb-link-row" href="{GITHUB_URL}" target="_blank" rel="noopener">'
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
            placeholder="https://example.com/page — collez l'URL complete ici",
            key="url_field",
        )
        st.markdown(
            '<div class="input-hint">'
            'Collez l\'URL complete (avec http:// ou https://). '
            'L\'outil analyse la structure du lien sans y acceder.'
            '</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
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

    if "last_result" in st.session_state:
        st.markdown(
            '<div class="section-label" style="margin-top:1.2rem">Resultat de l\'analyse</div>',
            unsafe_allow_html=True,
        )
        render_result(st.session_state["last_result"], st.session_state["last_url"])

    history = st.session_state.get("history", [])
    if history:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Historique de la session</div>', unsafe_allow_html=True)
        render_history(history)
        st.markdown('<div style="margin-top:.8rem"></div>', unsafe_allow_html=True)
        if st.button("Effacer l'historique"):
            st.session_state.history = []
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_url", None)
            st.rerun()

    st.markdown(
        f'<div class="pg-footer">'
        f'<div class="footer-left">'
        f'PhishGuard &mdash; Detection de phishing par NLP<br>'
        f'Souleymane Sall &middot; scikit-learn &middot; Streamlit'
        f'</div>'
        f'<div class="footer-links">'
        f'<a class="footer-link" href="{LINKEDIN_URL}" target="_blank" rel="noopener">'
        f'{SVG_LINKEDIN}&thinsp;LinkedIn'
        f'</a>'
        f'<a class="footer-link" href="{GITHUB_URL}" target="_blank" rel="noopener">'
        f'{SVG_GITHUB}&thinsp;GitHub'
        f'</a>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
