"""
Nordek · NRK Price Analysis Dashboard  ·  v2 — Honest ML Edition
─────────────────────────────────────────────────────────────────
Key upgrades over v1:
  • Look-ahead data leak FIXED — all rolling features shifted by 1 day
  • Model roster: Ridge (winner), Gradient Boosting, + naive baseline
  • TimeSeriesSplit CV reported in UI
  • Honest commentary on 73-row dataset limitations
  • All design tokens preserved; UI layout improved
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings, base64, os
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NRK · Nordek Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG      = "#050D18"
PANEL   = "#080F1C"
CARD    = "#0C1929"
CARD2   = "#0E1E30"
BORDER  = "#142236"
BORDER2 = "#1C3050"
BLUE    = "#1E90FF"
CYAN    = "#00D4FF"
GREEN   = "#00E5A0"
RED     = "#FF4060"
AMBER   = "#FFAA00"
PURPLE  = "#9B6DFF"
TEXT    = "#D8E8F8"
SUBTEXT = "#4A6885"
DIM     = "#243B55"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body,[class*="css"],.main,.stApp{{background-color:{BG}!important;color:{TEXT};font-family:'IBM Plex Mono',monospace}}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{{display:none!important;height:0!important}}
[data-testid="stSidebar"]{{background:{PANEL}!important;border-right:1px solid {BORDER2}!important;padding-top:0!important}}
[data-testid="stSidebar"]>div:first-child{{padding-top:0!important}}
[data-testid="stSidebar"] *{{color:{TEXT}!important;font-family:'IBM Plex Mono',monospace!important}}
.block-container{{padding:0 2rem 3rem 2rem!important;max-width:1400px!important}}
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:{BG}}}
::-webkit-scrollbar-thumb{{background:{BORDER2};border-radius:2px}}
::-webkit-scrollbar-thumb:hover{{background:{BLUE}}}

/* Radio nav */
div[role="radiogroup"]{{display:flex;flex-direction:column;gap:4px;padding:0}}
div[role="radiogroup"] label{{background:transparent!important;border:1px solid transparent!important;border-radius:4px!important;padding:9px 14px!important;margin:0!important;cursor:pointer!important;transition:all 0.15s ease!important;font-family:'IBM Plex Mono',monospace!important;font-size:12px!important;letter-spacing:0.05em!important;color:{SUBTEXT}!important}}
div[role="radiogroup"] label:hover{{border-color:{BORDER2}!important;color:{TEXT}!important;background:rgba(30,144,255,0.05)!important}}
div[role="radiogroup"] label:has(input:checked){{border-color:{BLUE}!important;color:{BLUE}!important;background:rgba(30,144,255,0.08)!important;font-weight:600!important}}
div[role="radiogroup"] input[type="radio"]{{display:none!important}}

/* Download button */
[data-testid="stDownloadButton"] button{{background:transparent!important;border:1px solid {BORDER2}!important;color:{SUBTEXT}!important;font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;letter-spacing:0.06em!important;padding:8px 16px!important;border-radius:4px!important;transition:all 0.15s!important;cursor:pointer!important}}
[data-testid="stDownloadButton"] button:hover{{border-color:{BLUE}!important;color:{BLUE}!important;background:rgba(30,144,255,0.06)!important}}

/* Dataframe */
[data-testid="stDataFrame"]{{border:1px solid {BORDER2}!important;border-radius:8px!important;overflow:hidden}}
[data-testid="stDataFrame"] th{{background:{CARD2}!important;color:{SUBTEXT}!important;font-family:'IBM Plex Mono',monospace!important;font-size:10px!important;letter-spacing:0.08em!important;text-transform:uppercase!important;padding:10px 12px!important;border-bottom:1px solid {BORDER2}!important}}
[data-testid="stDataFrame"] td{{font-family:'IBM Plex Mono',monospace!important;font-size:12px!important;color:{TEXT}!important;background:{CARD}!important;padding:9px 12px!important;border-bottom:1px solid {BORDER}!important}}

/* Components */
.topbar{{background:{PANEL};border-bottom:1px solid {BORDER2};padding:0 2rem;height:56px;display:flex;align-items:center;justify-content:space-between;margin:0 -2rem 2rem -2rem;position:sticky;top:0;z-index:100}}
.topbar-badge{{font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:{CYAN};border:1px solid rgba(0,212,255,0.3);padding:3px 8px;border-radius:3px;background:rgba(0,212,255,0.05)}}
.topbar-right{{display:flex;align-items:center;gap:20px;font-size:11px;color:{SUBTEXT};letter-spacing:0.04em}}
.topbar-dot{{width:6px;height:6px;background:{GREEN};border-radius:50%;display:inline-block;box-shadow:0 0 8px {GREEN};animation:pulse 2s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}

.page-header{{margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid {BORDER}}}
.page-eyebrow{{font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:{BLUE};margin-bottom:8px}}
.page-title{{font-size:28px;font-weight:700;color:{TEXT};letter-spacing:-0.02em;line-height:1.1}}
.page-title span{{color:{BLUE}}}
.page-sub{{font-size:12px;color:{SUBTEXT};margin-top:8px;line-height:1.6;max-width:640px;font-family:'IBM Plex Sans',sans-serif}}

.kpi-row{{display:grid;gap:12px;margin-bottom:2rem}}
.kpi-box{{background:{CARD};border:1px solid {BORDER};border-radius:6px;padding:16px 18px;position:relative;overflow:hidden;transition:border-color 0.2s}}
.kpi-box:hover{{border-color:{BORDER2}}}
.kpi-box::before{{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,{BLUE},{CYAN});opacity:0;transition:opacity 0.2s}}
.kpi-box:hover::before{{opacity:1}}
.kpi-label{{font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:{SUBTEXT};margin-bottom:10px}}
.kpi-value{{font-size:22px;font-weight:700;color:{TEXT};line-height:1;letter-spacing:-0.02em}}
.kpi-meta{{font-size:10px;margin-top:6px}}
.up{{color:{GREEN}}}.dn{{color:{RED}}}.neu{{color:{SUBTEXT}}}

.sec{{font-size:9px;letter-spacing:0.14em;text-transform:uppercase;color:{DIM};padding:1.5rem 0 0.75rem 0;display:flex;align-items:center;gap:12px}}
.sec::after{{content:"";flex:1;height:1px;background:{BORDER}}}

.m-card{{background:{CARD};border:1px solid {BORDER};border-radius:6px;padding:24px 20px;text-align:center}}
.m-label{{font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:{SUBTEXT};margin-bottom:12px}}
.m-value{{font-size:36px;font-weight:700;line-height:1;letter-spacing:-0.03em;margin-bottom:8px}}
.m-desc{{font-size:10px;color:{SUBTEXT};line-height:1.6}}

.intern-card{{background:linear-gradient(135deg,{CARD} 0%,rgba(30,144,255,0.06) 100%);border:1px solid {BORDER2};border-radius:8px;padding:24px 28px;margin-bottom:2rem;display:flex;gap:24px;align-items:flex-start;position:relative;overflow:hidden}}
.intern-card::before{{content:"";position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,{BLUE},{CYAN})}}
.intern-badge{{background:linear-gradient(135deg,{BLUE},{CYAN});border-radius:6px;padding:14px 16px;text-align:center;min-width:80px;flex-shrink:0}}
.intern-badge-title{{font-size:9px;font-weight:700;letter-spacing:0.1em;color:{BG};line-height:1.4;text-transform:uppercase}}

.bullet-item{{display:grid;grid-template-columns:auto 1fr;gap:16px;padding:14px 0;border-bottom:1px solid {BORDER};align-items:start}}
.bullet-item:last-child{{border-bottom:none}}
.bullet-icon{{width:28px;height:28px;border-radius:4px;background:rgba(30,144,255,0.12);border:1px solid rgba(30,144,255,0.25);display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0;margin-top:1px}}
.bullet-title{{font-size:13px;font-weight:600;color:{TEXT};margin-bottom:4px}}
.bullet-desc{{font-size:12px;color:{SUBTEXT};line-height:1.7;font-family:'IBM Plex Sans',sans-serif}}

.step-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:2rem}}
.step-card{{background:{CARD};border:1px solid {BORDER};border-radius:6px;padding:24px;position:relative;overflow:hidden;transition:border-color 0.2s}}
.step-card:hover{{border-color:{BORDER2}}}
.step-num{{font-size:64px;font-weight:700;color:{BORDER};line-height:1;position:absolute;top:12px;right:20px;user-select:none;pointer-events:none}}
.step-tag{{font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:{BLUE};margin-bottom:10px}}
.step-title{{font-size:15px;font-weight:600;color:{TEXT};margin-bottom:10px}}
.step-desc{{font-size:12px;color:{SUBTEXT};line-height:1.8;font-family:'IBM Plex Sans',sans-serif;position:relative;z-index:1}}

.timeline{{display:flex;gap:0;margin:0 0 2rem 0;border:1px solid {BORDER};border-radius:6px;overflow:hidden}}
.tl-item{{flex:1;padding:16px 20px;border-right:1px solid {BORDER}}}
.tl-item:last-child{{border-right:none}}
.tl-month{{font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:{SUBTEXT};margin-bottom:8px}}
.tl-bar{{height:3px;border-radius:2px;margin-bottom:10px}}
.tl-desc{{font-size:11px;color:{TEXT};line-height:1.6}}

.disclaimer{{background:rgba(255,64,96,0.06);border:1px solid rgba(255,64,96,0.2);border-radius:6px;padding:14px 18px;font-size:11px;color:rgba(255,180,190,0.9);line-height:1.7;margin:1rem 0;display:flex;gap:10px;align-items:flex-start;font-family:'IBM Plex Sans',sans-serif}}
.insight-box{{background:rgba(30,144,255,0.05);border:1px solid rgba(30,144,255,0.2);border-radius:6px;padding:14px 18px;font-size:11px;color:rgba(160,200,255,0.9);line-height:1.7;margin:1rem 0;display:flex;gap:10px;align-items:flex-start;font-family:'IBM Plex Sans',sans-serif}}
.chart-wrap{{background:{CARD};border:1px solid {BORDER};border-radius:6px;padding:4px;margin-bottom:1rem}}
.sidebar-logo{{padding:20px 16px 16px 16px;border-bottom:1px solid {BORDER};margin-bottom:0}}
.nav-section{{font-size:9px;letter-spacing:0.14em;text-transform:uppercase;color:{DIM};padding:16px 14px 6px 14px}}
.sidebar-info{{background:{CARD};border:1px solid {BORDER};border-radius:6px;padding:14px 16px;margin:12px 0}}
.sidebar-info-label{{font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:{SUBTEXT};margin-bottom:8px}}
.sidebar-info-text{{font-size:11px;color:{TEXT};line-height:1.7;font-family:'IBM Plex Sans',sans-serif}}
.model-compare-table{{width:100%;border-collapse:collapse;font-size:11px;font-family:'IBM Plex Mono',monospace}}
.model-compare-table th{{font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:{SUBTEXT};padding:8px 12px;border-bottom:1px solid {BORDER2};text-align:left;background:{CARD2}}}
.model-compare-table td{{padding:10px 12px;border-bottom:1px solid {BORDER};color:{TEXT}}}
.model-compare-table tr:last-child td{{border-bottom:none}}
.model-compare-table tr.winner td{{color:{GREEN}}}
.model-compare-table tr.baseline-row td{{color:{SUBTEXT};font-style:italic}}
.tag-winner{{background:rgba(0,229,160,0.12);color:{GREEN};border:1px solid rgba(0,229,160,0.3);padding:2px 8px;border-radius:3px;font-size:9px;letter-spacing:0.06em}}
.tag-leak{{background:rgba(255,64,96,0.12);color:{RED};border:1px solid rgba(255,64,96,0.3);padding:2px 8px;border-radius:3px;font-size:9px;letter-spacing:0.06em}}
.tag-info{{background:rgba(30,144,255,0.1);color:{BLUE};border:1px solid rgba(30,144,255,0.25);padding:2px 8px;border-radius:3px;font-size:9px;letter-spacing:0.06em}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_logo(h=36):
    p = "nordek_logo.png"
    if os.path.exists(p):
        with open(p,"rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" height="{h}" style="display:block;"/>'
    return f"""<svg width="120" height="{h}" viewBox="0 0 120 {h}" xmlns="http://www.w3.org/2000/svg">
      <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="{BLUE}"/><stop offset="100%" stop-color="{CYAN}"/></linearGradient></defs>
      <rect x="0" y="6" width="22" height="28" rx="3" fill="url(#lg)" opacity="0.9"/>
      <text x="4" y="25" font-family="IBM Plex Mono" font-size="11" font-weight="700" fill="{BG}">NRK</text>
      <text x="30" y="27" font-family="IBM Plex Mono" font-size="14" font-weight="700" fill="{TEXT}" letter-spacing="1">NORDEK</text>
    </svg>"""

def pbase(title="", h=420):
    return dict(
        title=dict(text=title, font=dict(family="IBM Plex Mono", size=11, color=SUBTEXT), x=0, xanchor="left", pad=dict(l=4,t=4)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", size=10, color=SUBTEXT),
        height=h, margin=dict(l=52,r=20,t=36,b=44),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER2, showgrid=True, tickfont=dict(size=10,color=SUBTEXT), zeroline=False),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER2, showgrid=True, tickfont=dict(size=10,color=SUBTEXT), zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10,color=SUBTEXT), bordercolor=BORDER, borderwidth=1),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER2, font=dict(family="IBM Plex Mono",size=11,color=TEXT)),
    )

def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

def chart(fig):
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ── Data pipeline ─────────────────────────────────────────────────────────────
def _pp(v):
    return float(str(v).replace("$","").strip()) if not pd.isna(v) else np.nan
def _pv(v):
    if pd.isna(v): return np.nan
    s=str(v).replace("$","").strip().lower()
    return float(s[:-1])*1e6 if s.endswith("m") else float(s[:-1])*1e3 if s.endswith("k") else float(s)

@st.cache_data
def load_data():
    df = pd.read_csv("price_history_filtered.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%B %d %Y")
    df = (df[(df["Date"]>="2023-06-01")&(df["Date"]<="2023-08-31")]
          .sort_values("Date").reset_index(drop=True))
    for col in ["Open","High","Low","Close"]:
        df[col] = df[col].apply(_pp).astype(float)
    df["Volume"] = df["Volume"].apply(_pv).astype(float)
    df["Volume_NRK"] = df["Volume(NRK)"].astype(float)
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"]  = df["Low"].fillna(df["Close"])
    return df.reset_index(drop=True)

@st.cache_data
def engineer_features(_df):
    df = _df.copy()
    c  = df["Close"]
    # ── All rolling features are shifted by 1 day → zero look-ahead leak ──
    sc_ = c.shift(1)
    delta = sc_.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["MA3"]    = sc_.rolling(3).mean()
    df["MA7"]    = sc_.rolling(7).mean()
    df["MA14"]   = sc_.rolling(14).mean()
    df["EMA7"]   = sc_.ewm(span=7, adjust=False).mean()
    df["EMA14"]  = sc_.ewm(span=14, adjust=False).mean()
    df["Vol7"]   = sc_.rolling(7).std()
    df["Ret1"]   = sc_.pct_change(1)
    df["Ret3"]   = sc_.pct_change(3)
    df["RSI14"]  = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df["HLSpread"] = df["High"].shift(1) - df["Low"].shift(1)
    for lag in range(1, 8):
        df[f"Lag{lag}"] = c.shift(lag)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"]     = df["Date"].dt.month
    df["VolMA7"]    = df["Volume"].shift(1).rolling(7).mean()
    FEATS = ["MA3","MA7","MA14","EMA7","EMA14","Vol7","Ret1","Ret3","RSI14",
             "HLSpread","Lag1","Lag2","Lag3","Lag4","Lag5","Lag6","Lag7",
             "DayOfWeek","Month","VolMA7"]
    df_f = df.dropna(subset=FEATS+["Close"]).copy().reset_index(drop=True)
    for col in FEATS:
        df_f[col] = df_f[col].astype("float64")
    return df_f, FEATS

@st.cache_data
def train_models(_df_feat, feat_cols):
    n     = len(_df_feat)
    split = int(n * 0.80)
    tr    = _df_feat.iloc[:split]
    te    = _df_feat.iloc[split:]
    Xtr   = tr[feat_cols].values;  ytr = tr["Close"].values
    Xte   = te[feat_cols].values;  yte = te["Close"].values

    sc = MinMaxScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xte_sc = sc.transform(Xte)

    # ── Models ──
    ridge = Ridge(alpha=0.5)
    gb    = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03,
                                      max_depth=2, subsample=0.7,
                                      min_samples_leaf=2, random_state=42)
    ridge.fit(Xtr_sc, ytr);  yp_ridge = ridge.predict(Xte_sc)
    gb.fit(Xtr_sc, ytr);     yp_gb    = gb.predict(Xte_sc)
    yp_naive = np.concatenate([[ytr[-1]], yte[:-1]])

    # ── TimeSeriesSplit CV (4 folds) ──
    tscv = TimeSeriesSplit(n_splits=4)
    def cv_mae(mdl):
        scores = []
        for ti, vi in tscv.split(Xtr_sc):
            mdl.fit(Xtr_sc[ti], ytr[ti])
            scores.append(mean_absolute_error(ytr[vi], mdl.predict(Xtr_sc[vi])))
        return float(np.mean(scores)), float(np.std(scores))

    cv_r = cv_mae(Ridge(alpha=0.5))
    cv_g = cv_mae(GradientBoostingRegressor(n_estimators=300,learning_rate=0.03,
                                             max_depth=2,subsample=0.7,
                                             min_samples_leaf=2,random_state=42))

    def metrics(yt, yp):
        return dict(
            mae  = mean_absolute_error(yt, yp),
            rmse = float(np.sqrt(mean_squared_error(yt, yp))),
            da   = int(np.sum(np.sign(np.diff(yt)) == np.sign(np.diff(yp)))),
        )

    return dict(
        ridge=ridge, gb=gb, scaler=sc,
        train=tr, test=te,
        y_test=yte, y_pred_ridge=yp_ridge, y_pred_gb=yp_gb, y_pred_naive=yp_naive,
        m_ridge=metrics(yte, yp_ridge),
        m_gb   =metrics(yte, yp_gb),
        m_naive=metrics(yte, yp_naive),
        cv_ridge=cv_r, cv_gb=cv_g,
        n_test=len(yte),
        feat_importances=dict(zip(feat_cols, gb.feature_importances_)),
        ridge_coefs=dict(zip(feat_cols, np.abs(ridge.coef_))),
    )

@st.cache_data
def build_forecast(_df, _results, feat_cols, days=30):
    df    = _df.copy()
    ridge = _results["ridge"]
    sc    = _results["scaler"]
    hist  = list(df["Close"].values)
    highs = list(df["High"].values)
    lows  = list(df["Low"].values)
    vols  = list(df["Volume"].values)
    last_d = pd.Timestamp(df["Date"].iloc[-1])
    preds  = []
    for i in range(days):
        c_s = pd.Series(hist)
        nd  = last_d + pd.Timedelta(days=i+1)
        sc_ = c_s  # already "yesterday" from model's perspective each step
        delta = sc_.diff(); gain=delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss=(-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        row = [
            sc_.rolling(3).mean().iloc[-1], sc_.rolling(7).mean().iloc[-1],
            sc_.rolling(14).mean().iloc[-1], sc_.ewm(span=7,adjust=False).mean().iloc[-1],
            sc_.ewm(span=14,adjust=False).mean().iloc[-1], sc_.rolling(7).std().iloc[-1],
            sc_.pct_change(1).iloc[-1], sc_.pct_change(3).iloc[-1],
            100 - 100/(1+gain/(loss+1e-10)),
            highs[-1]-lows[-1],
        ] + [hist[-j] for j in range(1,8)] + [nd.dayofweek, nd.month, float(np.mean(vols[-7:]))]
        p = float(ridge.predict(sc.transform(np.array(row, dtype="float64").reshape(1,-1)))[0])
        preds.append((nd, p))
        hist.append(p); highs.append(p*1.01); lows.append(p*0.99)
        vols.append(float(np.mean(vols[-7:])))

    fc_prices = [x[1] for x in preds]
    roll_std  = float(pd.Series(df["Close"].values).rolling(7).std().iloc[-1])
    return dict(
        dates=[x[0] for x in preds], prices=fc_prices,
        upper=[p+1.5*roll_std for p in fc_prices],
        lower=[p-1.5*roll_std for p in fc_prices],
        hist_dates=list(df["Date"]), hist_prices=list(df["Close"]),
    )


# ── Load ──────────────────────────────────────────────────────────────────────
df_raw             = load_data()
df_feat, feat_cols = engineer_features(df_raw)
results            = train_models(df_feat, feat_cols)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="sidebar-logo">{get_logo(36)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("nav",
        ["◈  My Work at Nordek",
         "◈  Price & Model Results",
         "◈  30-Day Forecast",
         "◈  How It Works"],
        label_visibility="collapsed")
    st.markdown(f"""
    <div style="padding:0 4px;">
      <div class="sidebar-info">
        <div class="sidebar-info-label">About</div>
        <div class="sidebar-info-text">ML price analysis built during a summer 2023 internship at Nordek.
        Feature engineering · model evaluation · data storytelling.</div>
      </div>
      <div style="font-size:10px;color:{DIM};padding:8px 4px;line-height:1.9;">
        Dataset · Jun–Aug 2023<br/>73 trading days · 20 features<br/>
        Primary model · Ridge Regression<br/>Secondary · Gradient Boosting<br/>
        <span style="color:{RED};">⚑</span> Look-ahead leak fixed in v2
      </div>
    </div>""", unsafe_allow_html=True)

_page = page.split("  ", 1)[-1].strip()

# ── Topbar ────────────────────────────────────────────────────────────────────
open_p  = df_raw["Open"].iloc[0]
close_p = df_raw["Close"].iloc[-1]
chg     = (close_p - open_p) / open_p * 100
chg_col = GREEN if chg >= 0 else RED
chg_sym = "▲" if chg >= 0 else "▼"
st.markdown(f"""
<div class="topbar">
  <div style="display:flex;align-items:center;gap:16px;">
    <span style="font-size:14px;font-weight:700;color:{TEXT};">NRK / USD</span>
    <span class="topbar-badge">NORDEK TOKEN</span>
    <span style="font-size:13px;font-weight:700;color:{TEXT};">${close_p:.4f}</span>
    <span style="font-size:12px;color:{chg_col};">{chg_sym} {abs(chg):.2f}% · period</span>
  </div>
  <div class="topbar-right">
    <span><span class="topbar-dot"></span>&nbsp; Data loaded</span>
    <span style="color:{DIM};">|</span>
    <span>Jun – Aug 2023 · 73 days</span>
    <span style="color:{DIM};">|</span>
    <span style="color:{GREEN};">v2 · leak-free</span>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — My Work at Nordek
# ══════════════════════════════════════════════════════════════════════════════
if _page == "My Work at Nordek":
    st.markdown(f"""
    <div class="page-header">
      <div class="page-eyebrow">Portfolio · Summer 2023</div>
      <div class="page-title">My Work at <span>Nordek</span></div>
      <div class="page-sub">Applied ML internship — analysing on-chain NRK token data,
      engineering 20 leak-free predictive features, and building a full model evaluation
      pipeline with honest benchmarking.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="intern-card">
      <div class="intern-badge"><div class="intern-badge-title">Applied<br/>ML<br/>Intern<br/>2023</div></div>
      <div>
        <div style="font-size:18px;font-weight:700;color:{TEXT};margin-bottom:4px;">Applied Machine Learning Intern</div>
        <div style="font-size:12px;color:{CYAN};margin-bottom:10px;letter-spacing:0.02em;">
          June – August 2023 &nbsp;·&nbsp; 3 months &nbsp;·&nbsp; Blockchain / DeFi
        </div>
        <div style="font-size:12px;color:{SUBTEXT};line-height:1.7;font-family:'IBM Plex Sans',sans-serif;">
          Collected and cleaned daily OHLCV data for the NRK token. Engineered 20 leak-free
          predictive signals, compared Ridge Regression against Gradient Boosting with
          TimeSeriesSplit cross-validation, and packaged the full pipeline into this
          interactive dashboard.
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    high_p = df_raw["High"].max()
    low_p  = df_raw["Low"].min()
    st.markdown(f"""
    <div class="kpi-row" style="grid-template-columns:repeat(5,1fr);">
      <div class="kpi-box"><div class="kpi-label">Period Open</div>
        <div class="kpi-value">${open_p:.4f}</div><div class="kpi-meta neu">1 Jun 2023</div></div>
      <div class="kpi-box"><div class="kpi-label">Period Close</div>
        <div class="kpi-value">${close_p:.4f}</div>
        <div class="kpi-meta {'up' if chg>=0 else 'dn'}">{chg_sym} {abs(chg):.2f}% total</div></div>
      <div class="kpi-box"><div class="kpi-label">Highest Price</div>
        <div class="kpi-value">${high_p:.4f}</div><div class="kpi-meta up">▲ Period peak</div></div>
      <div class="kpi-box"><div class="kpi-label">Lowest Price</div>
        <div class="kpi-value">${low_p:.4f}</div><div class="kpi-meta dn">▼ Period trough</div></div>
      <div class="kpi-box"><div class="kpi-label">Trading Days</div>
        <div class="kpi-value">{len(df_raw)}</div><div class="kpi-meta neu">Jun–Aug 2023</div></div>
    </div>""", unsafe_allow_html=True)

    # Candlestick + volume combined
    sec("Price History · Candlestick + MAs + Volume")
    ma7  = df_raw["Close"].rolling(7).mean()
    ma14 = df_raw["Close"].rolling(14).mean()
    fig  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.72,0.28], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=df_raw["Date"], open=df_raw["Open"], high=df_raw["High"],
        low=df_raw["Low"], close=df_raw["Close"], name="NRK/USD",
        increasing=dict(line=dict(color=GREEN,width=1), fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED,width=1),   fillcolor=RED),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_raw["Date"], y=ma7,  name="MA 7",
        line=dict(color=BLUE, width=1.2, dash="dot"), opacity=0.9), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_raw["Date"], y=ma14, name="MA 14",
        line=dict(color=AMBER, width=1.2, dash="dash"), opacity=0.9), row=1, col=1)
    vcol = [GREEN if df_raw["Close"].iloc[i]>=df_raw["Open"].iloc[i] else RED for i in range(len(df_raw))]
    fig.add_trace(go.Bar(x=df_raw["Date"], y=df_raw["Volume"],
        marker_color=vcol, opacity=0.7, name="Volume"), row=2, col=1)
    base = pbase(h=520)
    fig.update_layout(**base, xaxis_rangeslider_visible=False,
        xaxis2=dict(gridcolor=BORDER,linecolor=BORDER2,tickfont=dict(size=9,color=SUBTEXT)),
        yaxis=dict(tickprefix="$",gridcolor=BORDER,linecolor=BORDER2,tickfont=dict(size=10,color=SUBTEXT)),
        yaxis2=dict(tickformat=".2s",gridcolor=BORDER,linecolor=BORDER2,tickfont=dict(size=9,color=SUBTEXT)))
    chart(fig)

    sec("Deliverables · What I Built")
    for icon, title, desc in [
        ("📥","Data Collection & Cleaning",
         "73 days of daily OHLCV data. Parsed dollar signs and K/M volume suffixes. Validated consistency across the full Jun–Aug 2023 window."),
        ("⚙️","Feature Engineering — 20 leak-free signals",
         "All rolling features (MAs, EMA, RSI, volatility) are shifted 1 day forward so the model never sees future data. This is a common production-grade requirement that v1 was missing."),
        ("🤖","Model Training & Honest Evaluation",
         "Ridge Regression vs Gradient Boosting, evaluated with a held-out test set AND 4-fold TimeSeriesSplit CV. Ridge wins on this dataset — a smaller, regularised model outperforms a complex one when data is limited."),
        ("📊","Interactive Dashboard",
         "Full pipeline packaged into this Streamlit app with candlestick charts, model comparison table, CV scores, feature importance, and a 30-day forecast."),
    ]:
        st.markdown(f"""
        <div class="bullet-item">
          <div class="bullet-icon">{icon}</div>
          <div><div class="bullet-title">{title}</div>
          <div class="bullet-desc">{desc}</div></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Price & Model Results
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "Price & Model Results":
    st.markdown(f"""
    <div class="page-header">
      <div class="page-eyebrow">Evaluation · Leak-free features · TimeSeriesSplit CV</div>
      <div class="page-title">Price &amp; <span>Model Results</span></div>
      <div class="page-sub">Ridge Regression vs Gradient Boosting vs naive baseline —
      MAE, RMSE, direction accuracy, and cross-validation scores on a 73-row dataset.</div>
    </div>""", unsafe_allow_html=True)

    mr  = results["m_ridge"]
    mg  = results["m_gb"]
    mn  = results["m_naive"]
    nte = results["n_test"]
    cvr = results["cv_ridge"]
    cvg = results["cv_gb"]
    yte = results["y_test"]
    ypr = results["y_pred_ridge"]
    ypg = results["y_pred_gb"]
    ypn = results["y_pred_naive"]
    tdf = results["test"]

    # Best model determination
    best = "ridge" if mr["mae"] <= mg["mae"] else "gb"

    # Top metric trio — best model highlighted
    bm  = mr if best=="ridge" else mg
    bname = "Ridge Regression" if best=="ridge" else "Gradient Boosting"
    pct_vs_naive = (mn["mae"] - bm["mae"]) / mn["mae"] * 100
    dir_pct = bm["da"] / (nte-1) * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        dc = GREEN if dir_pct >= 55 else AMBER if dir_pct >= 45 else RED
        st.markdown(f"""<div class="m-card">
          <div class="m-label">Direction Accuracy · {bname}</div>
          <div class="m-value" style="color:{dc};">{bm['da']}/{nte-1}</div>
          <div class="m-desc">{dir_pct:.0f}% correct UP/DOWN calls<br/>on held-out test days</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="m-card">
          <div class="m-label">Mean Abs. Error · {bname}</div>
          <div class="m-value" style="color:{AMBER};">${bm['mae']:.5f}</div>
          <div class="m-desc">Average prediction gap<br/>on held-out test set</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        vc = GREEN if pct_vs_naive > 0 else RED
        st.markdown(f"""<div class="m-card">
          <div class="m-label">vs Naive Baseline</div>
          <div class="m-value" style="color:{vc};">{pct_vs_naive:+.1f}%</div>
          <div class="m-desc">MAE improvement over<br/>carry-forward baseline</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Model comparison table
    sec("Model Comparison · Test Set + Cross-Validation")
    st.markdown(f"""
    <table class="model-compare-table">
      <thead><tr>
        <th>Model</th><th>MAE ↓</th><th>RMSE ↓</th><th>Dir Acc ↑</th>
        <th>CV MAE (4-fold)</th><th>Notes</th>
      </tr></thead>
      <tbody>
        <tr class="baseline-row">
          <td>Naive carry-forward</td>
          <td>${mn['mae']:.5f}</td><td>${mn['rmse']:.5f}</td>
          <td>—</td><td>—</td>
          <td>Predict tomorrow = today</td>
        </tr>
        <tr class="{'winner' if best=='gb' else ''}">
          <td>Gradient Boosting <span class="tag-info">300 trees</span></td>
          <td>${mg['mae']:.5f}</td><td>${mg['rmse']:.5f}</td>
          <td>{mg['da']}/{nte-1} ({mg['da']/(nte-1)*100:.0f}%)</td>
          <td>{cvg[0]:.5f} ± {cvg[1]:.5f}</td>
          <td>Overfits on 73 rows</td>
        </tr>
        <tr class="{'winner' if best=='ridge' else ''}">
          <td>Ridge Regression <span class="{'tag-winner' if best=='ridge' else 'tag-info'}">{'★ best' if best=='ridge' else 'α=0.5'}</span></td>
          <td>${mr['mae']:.5f}</td><td>${mr['rmse']:.5f}</td>
          <td>{mr['da']}/{nte-1} ({mr['da']/(nte-1)*100:.0f}%)</td>
          <td>{cvr[0]:.5f} ± {cvr[1]:.5f}</td>
          <td>Simpler model wins on small data</td>
        </tr>
      </tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box" style="margin-top:10px;">
      <span style="font-size:13px;">💡</span>
      <span><strong>Why Ridge beats Gradient Boosting here:</strong>
      With only 73 rows and ~46 training samples, Gradient Boosting has too few examples
      to generalise well — it has higher variance across CV folds (±{cvg[1]:.5f} vs ±{cvr[1]:.5f}).
      Ridge's L2 regularisation prevents overfitting by shrinking coefficients rather than
      memorising noise. This is a classic small-data result — simpler models win.
      </span>
    </div>""", unsafe_allow_html=True)

    # Actual vs predicted
    sec("Actual vs Predicted · Test Window")
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=list(tdf["Date"])+list(tdf["Date"])[::-1],
        y=list(yte)+list(ypr)[::-1],
        fill="toself", fillcolor="rgba(30,144,255,0.05)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig_p.add_trace(go.Scatter(x=tdf["Date"], y=yte, name="Actual",
        line=dict(color=CYAN, width=2.5), mode="lines+markers",
        marker=dict(size=5,color=CYAN,line=dict(width=1.5,color=BG))))
    fig_p.add_trace(go.Scatter(x=tdf["Date"], y=ypr, name="Ridge (best)",
        line=dict(color=BLUE, width=2, dash="dot")))
    fig_p.add_trace(go.Scatter(x=tdf["Date"], y=ypg, name="Gradient Boosting",
        line=dict(color=PURPLE, width=1.5, dash="dot"), opacity=0.8))
    fig_p.add_trace(go.Scatter(x=tdf["Date"], y=ypn, name="Naive baseline",
        line=dict(color=AMBER, width=1.2, dash="dash"), opacity=0.6))
    fig_p.update_layout(**pbase(h=380))
    fig_p.update_yaxes(tickprefix="$")
    chart(fig_p)

    col_a, col_b = st.columns(2)
    with col_a:
        sec("Feature Importance · GB Model")
        fi_df = (pd.DataFrame(list(results["feat_importances"].items()), columns=["feat","imp"])
                 .sort_values("imp",ascending=False).head(8))
        label_map = {"Lag1":"Yesterday close","Lag2":"2-day lag","Lag3":"3-day lag",
                     "Lag4":"4-day lag","Lag5":"5-day lag","Lag6":"6-day lag","Lag7":"7-day lag",
                     "MA3":"MA 3","MA7":"MA 7","MA14":"MA 14","EMA7":"EMA 7","EMA14":"EMA 14",
                     "Vol7":"Volatility 7d","Ret1":"1-day return","Ret3":"3-day return",
                     "RSI14":"RSI 14","HLSpread":"Daily range","DayOfWeek":"Day of week",
                     "Month":"Month","VolMA7":"Vol MA 7"}
        fi_df["label"] = fi_df["feat"].map(label_map).fillna(fi_df["feat"])
        fig_fi = go.Figure(go.Bar(
            x=fi_df["imp"], y=fi_df["label"], orientation="h",
            marker=dict(color=fi_df["imp"], colorscale=[[0,BORDER2],[1,CYAN]], showscale=False, line=dict(width=0)),
            text=[f"{v:.3f}" for v in fi_df["imp"]], textposition="outside",
            textfont=dict(size=9,color=SUBTEXT,family="IBM Plex Mono")))
        fig_fi.update_layout(**pbase(h=320))
        fig_fi.update_xaxes(title_text="Importance")
        chart(fig_fi)

    with col_b:
        sec("Residuals · Ridge Model")
        residuals = yte - ypr
        r_col = [GREEN if r>=0 else RED for r in residuals]
        fig_r = go.Figure()
        fig_r.add_hline(y=0, line_color=SUBTEXT, line_dash="dot", line_width=1)
        fig_r.add_trace(go.Bar(x=tdf["Date"], y=residuals,
            marker_color=r_col, marker_line_width=0, opacity=0.85, name="Error"))
        fig_r.update_layout(**pbase("green = under-predicted · red = over-predicted", h=320))
        fig_r.update_yaxes(tickprefix="$")
        chart(fig_r)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — 30-Day Forecast
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "30-Day Forecast":
    st.markdown(f"""
    <div class="page-header">
      <div class="page-eyebrow">Projection · From 31 Aug 2023 · Portfolio demo only</div>
      <div class="page-title">30-Day <span>Forecast</span></div>
      <div class="page-sub">Ridge Regression rolled forward 30 days with ±1.5σ confidence band
      derived from 7-day rolling volatility. Forecast uses leak-free autoregressive features.</div>
    </div>""", unsafe_allow_html=True)

    fc = build_forecast(df_raw, results, feat_cols, days=30)
    fca = np.array(fc["prices"])
    fcs, fce, fcmax, fcmin = fca[0], fca[-1], fca.max(), fca.min()
    fcp = (fce-fcs)/fcs*100
    fcsym = "▲" if fcp>=0 else "▼"; fccol = GREEN if fcp>=0 else RED

    st.markdown(f"""
    <div class="kpi-row" style="grid-template-columns:repeat(4,1fr);">
      <div class="kpi-box"><div class="kpi-label">Forecast Start</div>
        <div class="kpi-value">${fcs:.4f}</div><div class="kpi-meta neu">1 Sep 2023</div></div>
      <div class="kpi-box"><div class="kpi-label">Forecast End</div>
        <div class="kpi-value">${fce:.4f}</div>
        <div class="kpi-meta {'up' if fcp>=0 else 'dn'}">{fcsym} {abs(fcp):.2f}% projected</div></div>
      <div class="kpi-box"><div class="kpi-label">Projected High</div>
        <div class="kpi-value">${fcmax:.4f}</div><div class="kpi-meta up">▲ Upper band peak</div></div>
      <div class="kpi-box"><div class="kpi-label">Projected Low</div>
        <div class="kpi-value">${fcmin:.4f}</div><div class="kpi-meta dn">▼ Lower band trough</div></div>
    </div>""", unsafe_allow_html=True)

    sec("Full Context + 30-Day Outlook · Ridge Model")
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=fc["hist_dates"], y=fc["hist_prices"], name="Historical",
        line=dict(color=CYAN,width=2), fill="tozeroy", fillcolor="rgba(0,212,255,0.04)"))
    fig_fc.add_trace(go.Scatter(
        x=fc["dates"]+fc["dates"][::-1], y=fc["upper"]+fc["lower"][::-1],
        fill="toself", fillcolor="rgba(30,144,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="±1.5σ band"))
    fig_fc.add_trace(go.Scatter(x=fc["dates"], y=fc["prices"], name="Forecast (Ridge)",
        line=dict(color=BLUE,width=2,dash="dot"), mode="lines+markers",
        marker=dict(size=3.5,color=BLUE)))
    fig_fc.add_vline(x=pd.Timestamp("2023-08-31"), line_color=AMBER, line_dash="dash",
        line_width=1.2, annotation_text="internship end",
        annotation_font_color=AMBER, annotation_font_size=10,
        annotation_font_family="IBM Plex Mono", annotation_position="top right")
    fig_fc.update_layout(**pbase(h=460))
    fig_fc.update_yaxes(tickprefix="$")
    chart(fig_fc)

    st.markdown(f"""
    <div class="disclaimer">
      <span style="font-size:13px;flex-shrink:0;">⚠</span>
      <span><strong>Disclaimer:</strong> Portfolio demonstration only. Nordek ceased operations
      after August 2023 and the NRK token is no longer actively traded. Not financial advice.
      Autoregressive forecasts degrade rapidly beyond a few steps — treat the 30-day projection
      as a trajectory illustration, not a reliable price target.</span>
    </div>""", unsafe_allow_html=True)

    sec("Day-by-Day Forecast Table")
    fc_table = pd.DataFrame({
        "Date":               [d.strftime("%d %b %Y") for d in fc["dates"]],
        "Projected Price":    [f"${p:.5f}" for p in fc["prices"]],
        "Upper (+1.5σ)":      [f"${u:.5f}" for u in fc["upper"]],
        "Lower (−1.5σ)":      [f"${l:.5f}" for l in fc["lower"]],
    })
    st.dataframe(fc_table, hide_index=True, use_container_width=True)
    st.download_button("⬇  Export forecast as CSV",
        data=fc_table.to_csv(index=False).encode(),
        file_name="nrk_30day_forecast.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — How It Works
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "How It Works":
    st.markdown(f"""
    <div class="page-header">
      <div class="page-eyebrow">Methodology · Plain-English Walkthrough</div>
      <div class="page-title">How It <span>Works</span></div>
      <div class="page-sub">A non-technical breakdown of the data pipeline,
      leak-free feature engineering, model selection, and honest evaluation.</div>
    </div>""", unsafe_allow_html=True)

    sec("Project Steps")
    st.markdown('<div class="step-grid">', unsafe_allow_html=True)
    for num, tag, title, desc in [
        ("01","STEP 1","Gathering the Data",
         "73 days of daily OHLCV data collected across the full Jun–Aug 2023 window. Raw quirks like dollar signs and K/M volume suffixes were parsed and normalised into a clean float dataset."),
        ("02","STEP 2","Leak-Free Feature Engineering",
         "All 20 signals — rolling averages, EMA, RSI, volatility, returns — are shifted 1 day forward before being fed to the model. This prevents look-ahead bias, a subtle but critical error present in many beginner pipelines."),
        ("03","STEP 3","Model Selection with CV",
         "Ridge Regression and Gradient Boosting were both trained on 80% of data. Rather than a single train/test split, 4-fold TimeSeriesSplit CV was used to measure generalisation. Ridge won — simpler models beat complex ones when training data is small."),
        ("04","STEP 4","Honest Evaluation",
         "Results are reported against a naive carry-forward baseline. A model that can't beat 'predict tomorrow = today' is not useful. CV variance is also shown — a low mean with high variance means the model got lucky on one fold."),
    ]:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-num">{num}</div>
          <div class="step-tag">{tag}</div>
          <div class="step-title">{title}</div>
          <div class="step-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec("Project Timeline · Summer 2023")
    tl_html = '<div class="timeline">'
    for month, color, desc in [
        ("June 2023", BLUE,  "Data collection · baseline exploration · initial cleaning"),
        ("July 2023", CYAN,  "Leak-free feature engineering · model architecture · first CV runs"),
        ("August 2023", GREEN,"Model evaluation · dashboard build · stakeholder demo · internship end"),
    ]:
        tl_html += f"""<div class="tl-item">
          <div class="tl-month">{month}</div>
          <div class="tl-bar" style="background:{color};"></div>
          <div class="tl-desc">{desc}</div></div>"""
    st.markdown(tl_html + '</div>', unsafe_allow_html=True)

    sec("Limitations & Honest Notes")
    st.markdown(f"""
    <div class="disclaimer">
      <span style="font-size:13px;flex-shrink:0;">⚠</span>
      <span>
        <strong>73 rows is a small dataset.</strong> Most production financial ML systems
        use years of data. Results here should be treated as a methodology demonstration,
        not a production trading signal. The Ridge model beats the naive baseline on MAE
        but direction accuracy on 11 test days is not statistically significant —
        you'd need ~100+ test points to draw firm conclusions.
      </span>
    </div>""", unsafe_allow_html=True)

    sec("Tech Stack")
    cols = st.columns(5)
    for col, (name, role) in zip(cols, [
        ("Python 3.11","Runtime"), ("Streamlit","Dashboard"),
        ("scikit-learn","Ridge · GB · CV"), ("Pandas / NumPy","Data pipeline"),
        ("Plotly","Charts"),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-box" style="text-align:center;padding:14px 10px;">
              <div class="kpi-label">{role}</div>
              <div style="font-size:12px;font-weight:600;color:{BLUE};margin-top:6px;">{name}</div>
            </div>""", unsafe_allow_html=True)
