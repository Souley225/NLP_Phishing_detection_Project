"""
PhishGuard — Enhanced UI for phishing URL detection.
Dark cybersecurity aesthetic with terminal/scan effects.

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import os
from datetime import datetime

import requests
import streamlit as st

st.set_page_config(
    page_title="PhishGuard — URL Threat Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")


# ── CSS Injection ─────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Sora:wght@300;400;600&display=swap');

    /* ── Variables ── */
    :root {
        --bg:       #060b16;
        --bg2:      #0b1222;
        --bg3:      #0f1928;
        --surface:  rgba(0, 229, 255, 0.04);
        --surface2: rgba(0, 229, 255, 0.08);
        --border:   rgba(0, 229, 255, 0.14);
        --border2:  rgba(0, 229, 255, 0.28);
        --cyan:     #00e5ff;
        --cyan2:    rgba(0, 229, 255, 0.55);
        --red:      #ff3b3b;
        --red2:     rgba(255, 59, 59, 0.55);
        --green:    #00ff88;
        --green2:   rgba(0, 255, 136, 0.55);
        --amber:    #ffb300;
        --text:     #c4d6e8;
        --text2:    rgba(196, 214, 232, 0.45);
        --ff-d:     'Orbitron', monospace;
        --ff-m:     'DM Mono', monospace;
        --ff-b:     'Sora', sans-serif;
    }

    /* ── Global reset ── */
    .stApp {
        background: var(--bg) !important;
        font-family: var(--ff-b);
        color: var(--text);
    }

    /* animated dot-grid background */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(circle, rgba(0,229,255,0.07) 1px, transparent 1px);
        background-size: 28px 28px;
        pointer-events: none;
        z-index: 0;
    }

    /* sweeping scan line */
    @keyframes scan {
        0%   { top: -4px; opacity: 0; }
        5%   { opacity: 0.4; }
        95%  { opacity: 0.4; }
        100% { top: 100vh; opacity: 0; }
    }
    .stApp::after {
        content: '';
        position: fixed;
        left: 0; width: 100%;
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, var(--cyan) 50%, transparent 100%);
        animation: scan 10s linear infinite;
        pointer-events: none;
        z-index: 1;
    }

    /* ── Layout ── */
    .main .block-container {
        padding: 1.5rem 2.5rem 3rem !important;
        max-width: 1100px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg2) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: var(--text) !important; }

    /* ── Hide chrome ── */
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }

    /* ── Headings ── */
    h1, h2, h3 {
        font-family: var(--ff-d) !important;
        color: var(--cyan) !important;
        letter-spacing: 0.06em;
    }

    /* ── Text input ── */
    .stTextInput label { display: none !important; }
    .stTextInput > div > div > input {
        background: rgba(0,0,0,0.35) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 6px !important;
        color: var(--cyan) !important;
        font-family: var(--ff-m) !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.85rem 1.1rem !important;
        caret-color: var(--cyan);
        transition: box-shadow .3s, border-color .3s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 1px var(--cyan), 0 0 24px rgba(0,229,255,.18) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: var(--text2) !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--border2) !important;
        color: var(--cyan) !important;
        font-family: var(--ff-d) !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        border-radius: 5px !important;
        padding: 0.65rem 1.6rem !important;
        transition: all .25s ease !important;
        position: relative !important;
    }
    .stButton > button:hover {
        background: rgba(0,229,255,.1) !important;
        box-shadow: 0 0 22px rgba(0,229,255,.28) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"] {
        border-color: var(--cyan) !important;
        background: rgba(0,229,255,.1) !important;
        box-shadow: 0 0 16px rgba(0,229,255,.18) !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 1.1rem !important;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--text2) !important;
        font-family: var(--ff-m) !important;
        font-size: 0.68rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--cyan) !important;
        font-family: var(--ff-d) !important;
        font-size: 1.35rem !important;
    }

    /* ── Progress bars ── */
    .stProgress > div > div { background: rgba(255,255,255,.06) !important; border-radius: 2px !important; }
    .stProgress > div > div > div { border-radius: 2px !important; }

    /* ── Expander ── */
    details summary {
        background: var(--bg3) !important;
        border: 1px solid var(--border) !important;
        border-radius: 5px !important;
        color: var(--text) !important;
        font-family: var(--ff-m) !important;
        font-size: 0.8rem !important;
    }

    /* ── Alerts ── */
    .stSuccess { background: rgba(0,255,136,.07) !important; border: 1px solid rgba(0,255,136,.3) !important; color: var(--green) !important; }
    .stError   { background: rgba(255,59,59,.07) !important;  border: 1px solid rgba(255,59,59,.3) !important;  color: var(--red)   !important; }
    .stWarning { background: rgba(255,179,0,.07) !important;  border: 1px solid rgba(255,179,0,.3) !important; }
    .stInfo    { background: var(--surface) !important; border: 1px solid var(--border) !important; }

    /* ── Spinner ── */
    .stSpinner > div > div { border-top-color: var(--cyan) !important; }

    hr { border-color: var(--border) !important; }

    /* ════════════════════════════════════════
       CUSTOM COMPONENT STYLES
    ════════════════════════════════════════ */

    @keyframes fadeUp   { from { opacity:0; transform:translateY(18px) } to { opacity:1; transform:translateY(0) } }
    @keyframes glow     { 0%,100%{text-shadow:0 0 12px rgba(0,229,255,.5)} 50%{text-shadow:0 0 28px rgba(0,229,255,.85),0 0 50px rgba(0,229,255,.25)} }
    @keyframes blink    { 0%,100%{opacity:1} 50%{opacity:0} }
    @keyframes throb    { 0%,100%{box-shadow:0 0 18px rgba(255,59,59,.35)} 50%{box-shadow:0 0 40px rgba(255,59,59,.7),0 0 80px rgba(255,59,59,.2)} }
    @keyframes safeglow { 0%,100%{box-shadow:0 0 14px rgba(0,255,136,.2)} 50%{box-shadow:0 0 32px rgba(0,255,136,.45),0 0 60px rgba(0,255,136,.12)} }
    @keyframes barfill  { from{width:0} to{width:var(--w)} }
    @keyframes pulsedot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.6;transform:scale(1.4)} }
    @keyframes scanrev  {
        0%   { clip-path: inset(0 0 100% 0); opacity:.7; }
        60%  { clip-path: inset(0 0 0% 0);   opacity:1; }
        100% { clip-path: inset(0 0 0% 0);   opacity:1; }
    }

    /* Header */
    .pg-header {
        text-align: center;
        padding: 1.8rem 0 1.2rem;
        animation: fadeUp .7s ease both;
    }
    .pg-shield {
        font-size: 2.8rem;
        line-height: 1;
        filter: drop-shadow(0 0 18px rgba(0,229,255,.9));
    }
    .pg-title {
        font-family: var(--ff-d);
        font-size: clamp(1.6rem, 4vw, 2.8rem);
        font-weight: 900;
        color: var(--cyan);
        letter-spacing: .22em;
        text-transform: uppercase;
        margin: .4rem 0 0;
        animation: glow 3.5s ease-in-out infinite;
    }
    .pg-sub {
        font-family: var(--ff-m);
        font-size: .72rem;
        color: var(--text2);
        letter-spacing: .3em;
        text-transform: uppercase;
        margin-top: .35rem;
    }
    .pg-cursor { animation: blink 1.1s step-end infinite; }

    /* Input section */
    .input-prefix {
        font-family: var(--ff-m);
        font-size: .72rem;
        color: var(--cyan2);
        text-transform: uppercase;
        letter-spacing: .15em;
        margin-bottom: .35rem;
    }
    .input-prefix span { color: var(--cyan); }

    /* Result card */
    .res-card {
        border-radius: 10px;
        padding: 1.6rem;
        margin: 1.4rem 0;
        animation: scanrev .7s ease both;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    .res-card.threat {
        background: rgba(255,59,59,.05);
        border: 1px solid rgba(255,59,59,.45);
        animation: scanrev .7s ease both, throb 2.5s ease-in-out 1s infinite;
    }
    .res-card.safe {
        background: rgba(0,255,136,.03);
        border: 1px solid rgba(0,255,136,.4);
        animation: scanrev .7s ease both, safeglow 3s ease-in-out 1s infinite;
    }

    .res-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
        gap: .6rem;
    }
    .res-badge {
        font-family: var(--ff-d);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .18em;
        text-transform: uppercase;
        padding: .35rem 1rem;
        border-radius: 4px;
    }
    .res-badge.threat {
        color: var(--red);
        background: rgba(255,59,59,.12);
        border: 1px solid rgba(255,59,59,.5);
        box-shadow: 0 0 14px rgba(255,59,59,.25);
    }
    .res-badge.safe {
        color: var(--green);
        background: rgba(0,255,136,.1);
        border: 1px solid rgba(0,255,136,.45);
        box-shadow: 0 0 14px rgba(0,255,136,.18);
    }
    .res-confidence {
        font-family: var(--ff-d);
        font-size: 1.6rem;
        font-weight: 900;
    }
    .res-confidence.threat { color: var(--red); }
    .res-confidence.safe   { color: var(--green); }
    .res-conf-label {
        font-family: var(--ff-m);
        font-size: .62rem;
        color: var(--text2);
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .1rem;
    }

    /* URL pill */
    .res-url {
        font-family: var(--ff-m);
        font-size: .82rem;
        color: var(--text2);
        background: rgba(0,0,0,.3);
        border: 1px solid rgba(255,255,255,.05);
        border-radius: 5px;
        padding: .55rem .9rem;
        margin-bottom: 1.2rem;
        word-break: break-all;
    }
    .res-url .prompt { color: var(--cyan); margin-right: .4rem; }

    /* Probability bars */
    .prob-section { margin: .8rem 0; }
    .prob-row { margin-bottom: .75rem; }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-family: var(--ff-m);
        font-size: .7rem;
        color: var(--text2);
        text-transform: uppercase;
        letter-spacing: .1em;
        margin-bottom: .35rem;
    }
    .prob-header .pct { color: var(--text); font-weight: 500; }
    .prob-track {
        height: 7px;
        background: rgba(255,255,255,.06);
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        width: var(--w);
        animation: barfill .9s cubic-bezier(.4,0,.2,1) both;
    }
    .prob-fill.legit  { background: linear-gradient(90deg, #00ff88, rgba(0,255,136,.45)); box-shadow: 0 0 10px rgba(0,255,136,.5); }
    .prob-fill.phish  { background: linear-gradient(90deg, #ff3b3b, rgba(255,59,59,.45));  box-shadow: 0 0 10px rgba(255,59,59,.5);  }

    /* Stats row */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .9rem;
        margin: 1.2rem 0 .6rem;
    }
    .stat-box {
        background: rgba(0,0,0,.25);
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 7px;
        padding: .8rem 1rem;
        text-align: center;
    }
    .stat-val {
        font-family: var(--ff-d);
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.15;
    }
    .stat-val.cyan  { color: var(--cyan); }
    .stat-val.green { color: var(--green); }
    .stat-val.red   { color: var(--red); }
    .stat-lbl {
        font-family: var(--ff-m);
        font-size: .62rem;
        color: var(--text2);
        text-transform: uppercase;
        letter-spacing: .1em;
        margin-top: .2rem;
    }

    /* Warning / safe message */
    .res-msg {
        font-family: var(--ff-b);
        font-size: .83rem;
        border-radius: 5px;
        padding: .75rem 1rem;
        margin-top: 1rem;
        line-height: 1.5;
    }
    .res-msg.threat { background: rgba(255,59,59,.08); border: 1px solid rgba(255,59,59,.25); color: rgba(255,100,100,.9); }
    .res-msg.safe   { background: rgba(0,255,136,.06); border: 1px solid rgba(0,255,136,.22); color: rgba(0,220,110,.9); }

    /* History */
    .hist-row {
        display: flex;
        align-items: center;
        gap: .7rem;
        padding: .55rem .9rem;
        border-radius: 5px;
        margin-bottom: .35rem;
        background: rgba(0,0,0,.22);
        border: 1px solid rgba(255,255,255,.04);
        font-family: var(--ff-m);
        font-size: .75rem;
        animation: fadeUp .3s ease both;
    }
    .hist-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .hist-dot.phishing   { background: var(--red);   box-shadow: 0 0 6px var(--red);   }
    .hist-dot.legitimate { background: var(--green); box-shadow: 0 0 6px var(--green); }
    .hist-url  { flex: 1; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .hist-conf { color: var(--text2); font-size: .68rem; flex-shrink: 0; }
    .hist-lbl  { font-size: .65rem; text-transform: uppercase; letter-spacing: .1em; flex-shrink: 0; }
    .hist-lbl.phishing   { color: var(--red);   }
    .hist-lbl.legitimate { color: var(--green); }
    .hist-time { font-size: .65rem; color: var(--text2); flex-shrink: 0; }

    /* Section divider */
    .sec-title {
        font-family: var(--ff-d);
        font-size: .7rem;
        color: var(--cyan);
        letter-spacing: .2em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: .75rem;
        margin: 2rem 0 1rem;
    }
    .sec-title::after { content:''; flex:1; height:1px; background: var(--border); }

    /* Sidebar pieces */
    .sb-label {
        font-family: var(--ff-d);
        font-size: .6rem;
        color: var(--cyan);
        letter-spacing: .2em;
        text-transform: uppercase;
        margin-bottom: .6rem;
        padding-bottom: .35rem;
        border-bottom: 1px solid var(--border);
    }
    .sb-status {
        display: inline-flex;
        align-items: center;
        gap: .55rem;
        font-family: var(--ff-m);
        font-size: .78rem;
    }
    .sb-dot {
        width: 9px; height: 9px;
        border-radius: 50%;
    }
    .sb-dot.on  { background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulsedot 2s ease-in-out infinite; }
    .sb-dot.off { background: var(--red);   box-shadow: 0 0 8px var(--red); }
    .sb-tag {
        display: inline-block;
        background: rgba(0,229,255,.07);
        border: 1px solid var(--border);
        border-radius: 3px;
        padding: .22rem .55rem;
        font-family: var(--ff-m);
        font-size: .65rem;
        color: var(--cyan2);
        margin: .18rem;
    }
    .ex-url {
        font-family: var(--ff-m);
        font-size: .7rem;
        padding: .38rem .65rem;
        border-radius: 4px;
        margin-bottom: .28rem;
        word-break: break-all;
    }
    .ex-url.ok  { background: rgba(0,255,136,.05); border: 1px solid rgba(0,255,136,.18); color: rgba(0,220,110,.8); }
    .ex-url.bad { background: rgba(255,59,59,.05);  border: 1px solid rgba(255,59,59,.18);  color: rgba(255,100,100,.8); }

    /* Footer */
    .pg-footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
        font-family: var(--ff-m);
        font-size: .68rem;
        color: var(--text2);
        letter-spacing: .06em;
    }
    .pg-footer a { color: var(--cyan2); text-decoration: none; }
    </style>
    """, unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def predict_url(url: str) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json={"url": url}, timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure it is running.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# ── Custom HTML blocks ────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown("""
    <div class="pg-header">
        <div class="pg-shield">🛡</div>
        <div class="pg-title">PHISHGUARD</div>
        <div class="pg-sub">Real-time URL Threat Intelligence<span class="pg-cursor">█</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_result(result: dict, url: str) -> None:
    is_phishing   = result["prediction"] == 1
    cls           = "threat" if is_phishing else "safe"
    badge_txt     = "⚠ PHISHING DETECTED" if is_phishing else "✓ LEGITIMATE"
    conf_pct      = f"{result['confidence']:.1%}"
    p_legit       = result["proba_legitimate"]
    p_phish       = result["proba_phishing"]
    w_legit       = f"{p_legit*100:.1f}%"
    w_phish       = f"{p_phish*100:.1f}%"
    ts            = datetime.fromisoformat(result["timestamp"].replace("Z", "")).strftime("%H:%M:%S UTC")
    msg = (
        "⚠ This URL exhibits strong phishing characteristics. "
        "Do <strong>not</strong> enter credentials or personal data on this site."
        if is_phishing else
        "This URL appears legitimate. Stay vigilant — always verify sensitive sites independently."
    )

    st.markdown(f"""
    <div class="res-card {cls}">
        <div class="res-top">
            <span class="res-badge {cls}">{badge_txt}</span>
            <div style="text-align:right">
                <div class="res-conf-label">Confidence</div>
                <div class="res-confidence {cls}">{conf_pct}</div>
            </div>
        </div>

        <div class="res-url">
            <span class="prompt">⟩</span>{url}
        </div>

        <div class="prob-section">
            <div class="prob-row">
                <div class="prob-header">
                    <span>Legitimate</span>
                    <span class="pct">{p_legit:.2%}</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill legit" style="--w:{w_legit}"></div>
                </div>
            </div>
            <div class="prob-row">
                <div class="prob-header">
                    <span>Phishing</span>
                    <span class="pct">{p_phish:.2%}</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill phish" style="--w:{w_phish}"></div>
                </div>
            </div>
        </div>

        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-val cyan">{ts}</div>
                <div class="stat-lbl">Timestamp</div>
            </div>
            <div class="stat-box">
                <div class="stat-val {'red' if is_phishing else 'green'}">{result['label'].upper()}</div>
                <div class="stat-lbl">Classification</div>
            </div>
            <div class="stat-box">
                <div class="stat-val cyan">{len(url)}</div>
                <div class="stat-lbl">URL Length</div>
            </div>
        </div>

        <div class="res-msg {cls}">{msg}</div>
    </div>
    """, unsafe_allow_html=True)


def render_history(history: list) -> None:
    items = list(reversed(history[-10:]))
    rows_html = ""
    for item in items:
        lbl   = item["label"]
        ts    = datetime.fromisoformat(item["timestamp"].replace("Z", "")).strftime("%H:%M")
        short = item["url"][:55] + ("…" if len(item["url"]) > 55 else "")
        conf  = f"{item['confidence']:.0%}"
        rows_html += f"""
        <div class="hist-row">
            <span class="hist-dot {lbl}"></span>
            <span class="hist-url">{short}</span>
            <span class="hist-conf">{conf}</span>
            <span class="hist-lbl {lbl}">{lbl}</span>
            <span class="hist-time">{ts}</span>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)


def render_sidebar_status(online: bool) -> None:
    dot_cls = "on" if online else "off"
    label   = "ONLINE — Model Ready" if online else "OFFLINE"
    color   = "var(--green)" if online else "var(--red)"
    st.markdown(f"""
    <div class="sb-label">System Status</div>
    <div class="sb-status" style="color:{color}">
        <span class="sb-dot {dot_cls}"></span>{label}
    </div>
    """, unsafe_allow_html=True)


def render_model_info() -> None:
    st.markdown("""
    <div class="sb-label" style="margin-top:1.4rem">Model Stack</div>
    <div style="margin-top:.3rem; line-height:2">
        <span class="sb-tag">TF-IDF Word</span>
        <span class="sb-tag">TF-IDF Char n-grams</span>
        <span class="sb-tag">Lexical Features</span>
        <span class="sb-tag">Logistic Regression</span>
        <span class="sb-tag">F1 = 0.91</span>
        <span class="sb-tag">Acc = 96%</span>
    </div>
    """, unsafe_allow_html=True)


def render_examples() -> None:
    st.markdown("""
    <div class="sb-label" style="margin-top:1.4rem">Example URLs</div>
    <div style="margin-top:.4rem; font-size:.68rem; color:var(--green); font-family:var(--ff-m); margin-bottom:.3rem; text-transform:uppercase; letter-spacing:.1em;">✓ Legitimate</div>
    <div class="ex-url ok">https://www.google.com</div>
    <div class="ex-url ok">https://github.com/login</div>
    <div style="margin-top:.7rem; font-size:.68rem; color:var(--red); font-family:var(--ff-m); margin-bottom:.3rem; text-transform:uppercase; letter-spacing:.1em;">⚠ Phishing</div>
    <div class="ex-url bad">http://paypal-secure.tk/login.php</div>
    <div class="ex-url bad">http://192.168.1.1/bank-verify</div>
    """, unsafe_allow_html=True)


def render_session_stats() -> None:
    history = st.session_state.get("history", [])
    total   = len(history)
    threats = sum(1 for h in history if h["label"] == "phishing")
    safe    = total - threats
    st.markdown(f"""
    <div class="sb-label" style="margin-top:1.4rem">Session Stats</div>
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:.5rem; margin-top:.5rem">
        <div class="stat-box">
            <div class="stat-val cyan" style="font-size:1.1rem">{total}</div>
            <div class="stat-lbl">Scans</div>
        </div>
        <div class="stat-box">
            <div class="stat-val red" style="font-size:1.1rem">{threats}</div>
            <div class="stat-lbl">Threats</div>
        </div>
        <div class="stat-box">
            <div class="stat-val green" style="font-size:1.1rem">{safe}</div>
            <div class="stat-lbl">Safe</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        online = check_api_health()
        render_sidebar_status(online)
        render_model_info()
        render_examples()
        render_session_stats()
        st.markdown("""
        <div style="margin-top:2rem; font-family:var(--ff-m); font-size:.62rem; color:var(--text2); letter-spacing:.06em">
            Built by Souleymane Sall<br>
            NLP Phishing Detection · 2025
        </div>
        """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────
    render_header()

    # ── Input section ────────────────────────────────────────────
    st.markdown('<div class="input-prefix">⟩_ <span>ENTER TARGET URL FOR ANALYSIS</span></div>', unsafe_allow_html=True)

    with st.form(key="scan_form", clear_on_submit=False):
        url_input = st.text_input(
            label="url",
            placeholder="https://example.com/path?query=value",
            key="url_field",
        )
        col1, col2, col3 = st.columns([1.2, 1, 4])
        with col1:
            submitted = st.form_submit_button("⟩ SCAN URL", type="primary", use_container_width=True)
        with col2:
            clear = st.form_submit_button("CLEAR", use_container_width=True)

    if clear:
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_url", None)
        st.rerun()

    if submitted:
        if not url_input.strip():
            st.warning("Please enter a URL to scan.")
        elif not online:
            st.error("API is offline. Cannot perform scan.")
        else:
            with st.spinner("Scanning..."):
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

    # ── Result ───────────────────────────────────────────────────
    if "last_result" in st.session_state:
        st.markdown('<div class="sec-title">Scan Result</div>', unsafe_allow_html=True)
        render_result(st.session_state["last_result"], st.session_state["last_url"])

    # ── History ──────────────────────────────────────────────────
    history = st.session_state.get("history", [])
    if history:
        st.markdown('<div class="sec-title">Scan History</div>', unsafe_allow_html=True)
        render_history(history)
        if st.button("Clear History", use_container_width=False):
            st.session_state.history = []
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_url", None)
            st.rerun()

    # ── Footer ───────────────────────────────────────────────────
    st.markdown("""
    <div class="pg-footer">
        PhishGuard — NLP Phishing Detection &nbsp;·&nbsp;
        FastAPI + Streamlit + scikit-learn &nbsp;·&nbsp;
        <a href="mailto:sallsouleymane2207@gmail.com">Souleymane Sall</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
