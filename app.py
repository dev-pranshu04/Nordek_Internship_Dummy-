import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import warnings
import io
import base64
import os

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NRK Price Analysis · Nordek",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG       = "#07111F"
PANEL    = "#0C1E35"
CARD     = "#0F2644"
BORDER   = "#173558"
BLUE     = "#2D8EFF"
CYAN     = "#00D4FF"
GREEN    = "#00E5A0"
RED      = "#FF4C6A"
AMBER    = "#FFB800"
TEXT     = "#E8F0FE"
SUBTEXT  = "#7A95B8"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    background-color: {BG} !important;
    color: {TEXT};
    font-family: 'IBM Plex Sans', sans-serif;
}}
/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
    visibility: hidden !important;
    height: 0px !important;
}}
[data-testid="stSidebar"] {{
    background: {PANEL} !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
.block-container {{ padding-top: 0rem !important; max-width: 1200px; }}
/* Radio buttons */
div[role="radiogroup"] label {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 14px;
    margin: 3px 0;
    cursor: pointer;
    transition: border-color .2s;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    color: {TEXT} !important;
}}
div[role="radiogroup"] label:hover {{ border-color: {BLUE}; }}
div[role="radiogroup"] [aria-checked="true"] + div label,
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
    border-color: {BLUE} !important;
    background: {PANEL} !important;
}}
/* Hero band */
.hero-band {{
    background: linear-gradient(135deg, {PANEL} 0%, #0A1A2F 100%);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 18px;
}}
.hero-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: {TEXT};
    margin: 0;
}}
.hero-sub {{
    font-size: 13px;
    color: {SUBTEXT};
    margin: 2px 0 0 0;
    font-family: 'IBM Plex Mono', monospace;
}}
/* KPI cards */
.kpi-grid {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
.kpi-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 18px 20px;
    flex: 1;
    min-width: 150px;
}}
.kpi-label {{ font-size: 11px; color: {SUBTEXT}; font-family: 'IBM Plex Mono', monospace; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 6px; }}
.kpi-value {{ font-size: 24px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; color: {TEXT}; }}
.kpi-delta {{ font-size: 12px; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }}
.kpi-up {{ color: {GREEN}; }}
.kpi-dn {{ color: {RED}; }}
.kpi-neu {{ color: {SUBTEXT}; }}
/* Section headers */
.sec-header {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: {SUBTEXT};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
}}
/* Metric card for model results */
.metric-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}}
.metric-label {{ font-size: 11px; color: {SUBTEXT}; font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: .06em; }}
.metric-value {{ font-size: 28px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; margin: 8px 0 4px; }}
.metric-desc {{ font-size: 12px; color: {SUBTEXT}; }}
/* Disclaimer */
.disclaimer {{
    background: rgba(255,76,106,.08);
    border: 1px solid rgba(255,76,106,.3);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #FFB3BE;
    margin: 16px 0;
}}
/* Internship badge */
.intern-card {{
    background: linear-gradient(135deg, rgba(45,142,255,.12), rgba(0,212,255,.06));
    border: 1px solid rgba(45,142,255,.35);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}}
.intern-badge {{
    background: linear-gradient(135deg, {BLUE}, {CYAN});
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: #07111F;
    text-align: center;
    white-space: nowrap;
}}
/* Bullet items */
.bullet-item {{
    display: flex;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid {BORDER};
    align-items: flex-start;
}}
.bullet-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {CYAN};
    margin-top: 7px;
    flex-shrink: 0;
}}
/* Timeline */
.timeline-bar {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}}
/* Step cards */
.step-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 12px;
}}
.step-num {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    color: {BORDER};
    line-height: 1;
}}
.step-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: {BLUE};
    margin: 4px 0;
}}
</style>
""", unsafe_allow_html=True)


# ── Plotly base layout ────────────────────────────────────────────────────────
def plotly_layout(title="", height=420):
    return dict(
        title=dict(text=title, font=dict(family="IBM Plex Mono", size=14, color=TEXT)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", size=11, color=SUBTEXT),
        height=height,
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, showgrid=True),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        hovermode="x unified",
    )


# ── Logo helper ───────────────────────────────────────────────────────────────
def get_logo_html(height=52):
    logo_path = "nordek_logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" height="{height}" style="display:block;" />'
    # Fallback SVG
    return f"""
    <svg width="160" height="{height}" viewBox="0 0 160 48" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="ng" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="{BLUE}"/>
          <stop offset="100%" stop-color="{CYAN}"/>
        </linearGradient>
      </defs>
      <polygon points="8,40 8,8 18,8 28,28 28,8 38,8 38,40 28,40 18,20 18,40" fill="url(#ng)"/>
      <text x="50" y="32" font-family="IBM Plex Mono" font-size="18" font-weight="700" fill="{TEXT}">NORDEK</text>
    </svg>"""


def hero(page_title, page_sub=""):
    logo = get_logo_html(48)
    st.markdown(f"""
    <div class="hero-band">
        {logo}
        <div>
            <p class="hero-title">{page_title}</p>
            <p class="hero-sub">{page_sub}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Data loading & feature engineering ───────────────────────────────────────
def parse_price(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("$", "").strip()
    return float(s)


def parse_volume(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("$", "").strip()
    m = s.lower()
    if m.endswith("m"):
        return float(m[:-1]) * 1_000_000
    if m.endswith("k"):
        return float(m[:-1]) * 1_000
    return float(m)


@st.cache_data
def load_data(_df_raw=None):
    df = pd.read_csv("price_history_filtered.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%B %d %Y")
    # Filter to Jun–Aug 2023
    df = df[(df["Date"] >= "2023-06-01") & (df["Date"] <= "2023-08-31")].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].apply(parse_price).astype(float)
    df["Volume"] = df["Volume"].apply(parse_volume).astype(float)
    df["Volume_NRK"] = df["Volume(NRK)"].astype(float)

    # Safe High/Low fallback
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"]  = df["Low"].fillna(df["Close"])

    return df.reset_index(drop=True)


@st.cache_data
def engineer_features(_df):
    df = _df.copy()
    c = df["Close"]

    df["MA3"]   = c.rolling(3).mean()
    df["MA7"]   = c.rolling(7).mean()
    df["MA14"]  = c.rolling(14).mean()
    df["EMA7"]  = c.ewm(span=7, adjust=False).mean()
    df["EMA14"] = c.ewm(span=14, adjust=False).mean()
    df["Vol7"]  = c.rolling(7).std()
    df["Ret1"]  = c.pct_change(1)
    df["Ret3"]  = c.pct_change(3)
    # RSI-14
    delta   = c.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / (loss + 1e-10)
    df["RSI14"] = 100 - (100 / (1 + rs))
    df["HLSpread"] = df["High"] - df["Low"]
    for lag in range(1, 6):
        df[f"Lag{lag}"] = c.shift(lag)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"]     = df["Date"].dt.month

    FEAT_COLS = ["MA3","MA7","MA14","EMA7","EMA14","Vol7","Ret1","Ret3",
                 "RSI14","HLSpread","Lag1","Lag2","Lag3","Lag4","Lag5",
                 "DayOfWeek","Month","Volume"]

    df_feat = df.dropna(subset=FEAT_COLS).copy()
    for col in FEAT_COLS:
        df_feat[col] = df_feat[col].astype("float64")

    return df_feat, FEAT_COLS


@st.cache_data
def train_models(_df_feat, feat_cols):
    # 80/20 split on 73 rows ≈ first 58 train, last 15 test
    # After dropna we have fewer rows – split at 80%
    n = len(_df_feat)
    split = int(n * 0.80)
    train = _df_feat.iloc[:split]
    test  = _df_feat.iloc[split:]

    X_train = train[feat_cols].values
    y_train = train["Close"].values
    X_test  = test[feat_cols].values
    y_test  = test["Close"].values

    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=3, subsample=0.8, random_state=42)
    gb.fit(X_train_sc, y_train)

    # Naive baseline: predict tomorrow = today (last-value carry-forward)
    # This is the standard financial baseline for comparison
    y_pred_lr = np.concatenate([[y_train[-1]], y_test[:-1]])  # shift by 1

    y_pred_gb = gb.predict(X_test_sc)

    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    # Direction accuracy
    actual_dir   = np.sign(np.diff(y_test))
    pred_dir_gb  = np.sign(np.diff(y_pred_gb))
    dir_acc = int(np.sum(actual_dir == pred_dir_gb))

    return dict(
        gb=gb, scaler=scaler,
        train=train, test=test,
        y_test=y_test, y_pred_gb=y_pred_gb, y_pred_lr=y_pred_lr,
        mae_gb=mae_gb, mae_lr=mae_lr,
        dir_acc=dir_acc, n_test=len(y_test),
        feat_importances=dict(zip(feat_cols, gb.feature_importances_)),
    )


@st.cache_data
def build_forecast(_df, _results, feat_cols, days=30):
    df = _df.copy()
    gb = _results["gb"]
    scaler = _results["scaler"]

    history = list(df["Close"].values)
    highs   = list(df["High"].values)
    lows    = list(df["Low"].values)
    vols    = list(df["Volume"].values)
    dates   = list(df["Date"].values)
    last_date = pd.Timestamp(dates[-1])

    preds = []
    for i in range(days):
        c = pd.Series(history)
        ma3   = c.rolling(3).mean().iloc[-1]
        ma7   = c.rolling(7).mean().iloc[-1]
        ma14  = c.rolling(14).mean().iloc[-1]
        ema7  = c.ewm(span=7, adjust=False).mean().iloc[-1]
        ema14 = c.ewm(span=14, adjust=False).mean().iloc[-1]
        vol7  = c.rolling(7).std().iloc[-1]
        ret1  = c.pct_change(1).iloc[-1]
        ret3  = c.pct_change(3).iloc[-1]
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss  = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rsi   = 100 - 100 / (1 + gain / (loss + 1e-10))
        hl    = highs[-1] - lows[-1]
        lags  = [history[-j] for j in range(1, 6)]
        nd    = (last_date + pd.Timedelta(days=i+1))
        dow   = nd.dayofweek
        mon   = nd.month
        vol   = float(np.mean(vols[-7:]))

        row = [ma3, ma7, ma14, ema7, ema14, vol7, ret1, ret3,
               rsi, hl] + lags + [dow, mon, vol]
        row = np.array(row, dtype="float64").reshape(1, -1)
        row_sc = scaler.transform(row)
        p = float(gb.predict(row_sc)[0])
        preds.append((nd, p))
        history.append(p)
        highs.append(p * 1.01)
        lows.append(p * 0.99)
        vols.append(vol)

    forecast_dates  = [x[0] for x in preds]
    forecast_prices = [x[1] for x in preds]
    hist_close = df["Close"].values
    roll_std = float(pd.Series(hist_close).rolling(7).std().iloc[-1])

    upper = [p + 1.5 * roll_std for p in forecast_prices]
    lower = [p - 1.5 * roll_std for p in forecast_prices]

    return dict(
        dates=forecast_dates, prices=forecast_prices,
        upper=upper, lower=lower,
        hist_dates=list(df["Date"]),
        hist_prices=list(df["Close"]),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(get_logo_html(56), unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "My Work at Nordek",
            "Price & Model Results",
            "30-Day Forecast",
            "How It Works",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;'>
        <p style='font-family:IBM Plex Mono,monospace;font-size:10px;color:{SUBTEXT};
                  text-transform:uppercase;letter-spacing:.06em;margin:0 0 6px 0;'>About This Project</p>
        <p style='font-size:12px;color:{TEXT};margin:0;line-height:1.6;'>
            A machine-learning price analysis built during a Data Science internship at Nordek in Summer 2023.
            The dashboard was constructed as a portfolio piece to demonstrate feature engineering,
            model evaluation, and data storytelling skills.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Load data once ────────────────────────────────────────────────────────────
df_raw   = load_data()
df_feat, feat_cols = engineer_features(df_raw)
results  = train_models(df_feat, feat_cols)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — My Work at Nordek
# ══════════════════════════════════════════════════════════════════════════════
if page == "My Work at Nordek":
    hero("My Work at Nordek", "NRK Token · Price Analysis Dashboard")

    # Internship context card
    st.markdown(f"""
    <div class="intern-card">
        <div class="intern-badge">INTERNSHIP<br/>NORDEK<br/>2023</div>
        <div>
            <p style='font-family:IBM Plex Mono,monospace;font-size:16px;font-weight:700;
                      color:{TEXT};margin:0 0 4px 0;'>Data Science Intern</p>
            <p style='font-size:13px;color:{CYAN};margin:0 0 6px 0;
                      font-family:IBM Plex Mono,monospace;'>
                June – August 2023 &nbsp;·&nbsp; 3 months &nbsp;·&nbsp; Blockchain / DeFi
            </p>
            <p style='font-size:13px;color:{SUBTEXT};margin:0;'>
                Analysed on-chain trading data for the NRK token, engineered predictive features,
                and built a machine-learning pipeline to forecast short-term price movements.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    open_price  = df_raw["Open"].iloc[0]
    close_price = df_raw["Close"].iloc[-1]
    high_price  = df_raw["High"].max()
    low_price   = df_raw["Low"].min()
    trading_days = len(df_raw)
    pct_change  = (close_price - open_price) / open_price * 100

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Period Open</div>
            <div class="kpi-value">${open_price:.4f}</div>
            <div class="kpi-delta kpi-neu">1 Jun 2023</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Period Close</div>
            <div class="kpi-value">${close_price:.4f}</div>
            <div class="kpi-delta {'kpi-up' if pct_change>=0 else 'kpi-dn'}">
                {'▲' if pct_change>=0 else '▼'} {abs(pct_change):.1f}%
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Highest Price</div>
            <div class="kpi-value">${high_price:.4f}</div>
            <div class="kpi-delta kpi-up">▲ Peak</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Lowest Price</div>
            <div class="kpi-value">${low_price:.4f}</div>
            <div class="kpi-delta kpi-dn">▼ Trough</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Trading Days</div>
            <div class="kpi-value">{trading_days}</div>
            <div class="kpi-delta kpi-neu">Jun–Aug 2023</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Candlestick + MA
    st.markdown('<div class="sec-header">Price History · Candlestick with Moving Averages</div>', unsafe_allow_html=True)
    ma7  = df_raw["Close"].rolling(7).mean()
    ma14 = df_raw["Close"].rolling(14).mean()

    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=df_raw["Date"], open=df_raw["Open"], high=df_raw["High"],
        low=df_raw["Low"], close=df_raw["Close"],
        name="NRK", increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
    ))
    fig_candle.add_trace(go.Scatter(
        x=df_raw["Date"], y=ma7, name="7-day MA",
        line=dict(color=BLUE, width=1.5, dash="dot"),
    ))
    fig_candle.add_trace(go.Scatter(
        x=df_raw["Date"], y=ma14, name="14-day MA",
        line=dict(color=AMBER, width=1.5, dash="dot"),
    ))
    fig_candle.update_layout(**plotly_layout("NRK/USD · Jun–Aug 2023", 480))
    fig_candle.update_yaxes(tickprefix="$")
    fig_candle.update_xaxis(rangeslider_visible=False)
    st.plotly_chart(fig_candle, use_container_width=True)

    # Volume bar
    st.markdown('<div class="sec-header">Daily Trading Volume (USD)</div>', unsafe_allow_html=True)
    colors_vol = [GREEN if df_raw["Close"].iloc[i] >= df_raw["Open"].iloc[i] else RED
                  for i in range(len(df_raw))]
    fig_vol = go.Figure(go.Bar(
        x=df_raw["Date"], y=df_raw["Volume"],
        marker_color=colors_vol, name="Volume",
    ))
    fig_vol.update_layout(**plotly_layout("", 300))
    fig_vol.update_yaxes(tickformat=".2s")
    st.plotly_chart(fig_vol, use_container_width=True)

    # What I Did
    st.markdown('<div class="sec-header">What I Did</div>', unsafe_allow_html=True)
    bullets = [
        ("Collected & cleaned data",
         "Gathered daily OHLCV data for the NRK token across the entire summer, parsed non-standard price formats (e.g. dollar signs, K/M volume suffixes), and validated the dataset for consistency."),
        ("Built 18 predictive features",
         "Engineered moving averages, exponential smoothing, rolling volatility, RSI momentum, price lags, and calendar features — turning raw OHLCV data into a rich feature matrix for ML."),
        ("Trained & evaluated two models",
         "Compared a Gradient Boosting model against a Linear Regression baseline, measuring direction accuracy and average price error on a held-out 15-day test window."),
        ("Delivered an interactive dashboard",
         "Packaged everything into this Streamlit app — showing candlestick charts, model performance, a 30-day forward forecast, and a plain-English walkthrough for non-technical stakeholders."),
    ]
    for title, desc in bullets:
        st.markdown(f"""
        <div class="bullet-item">
            <div class="bullet-dot"></div>
            <div>
                <span style='font-weight:600;color:{TEXT};'>{title}</span>
                <span style='color:{SUBTEXT};font-size:13px;'> — {desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Price & Model Results
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Price & Model Results":
    hero("Price & Model Results", "Gradient Boosting · 15-day test window")

    y_test    = results["y_test"]
    y_pred_gb = results["y_pred_gb"]
    y_pred_lr = results["y_pred_lr"]
    mae_gb    = results["mae_gb"]
    mae_lr    = results["mae_lr"]
    dir_acc   = results["dir_acc"]
    n_test    = results["n_test"]
    test_df   = results["test"]

    pct_better = (mae_lr - mae_gb) / mae_lr * 100

    # 3 metric cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Direction Accuracy</div>
            <div class="metric-value" style="color:{GREEN};">{dir_acc} / {n_test-1}</div>
            <div class="metric-desc">Got UP or DOWN right<br/>{dir_acc} out of {n_test-1} days</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Price Error</div>
            <div class="metric-value" style="color:{AMBER};">${mae_gb:.5f}</div>
            <div class="metric-desc">Average gap between<br/>predicted and actual price</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        color = GREEN if pct_better > 0 else RED
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">vs Baseline</div>
            <div class="metric-value" style="color:{color};">{pct_better:+.1f}%</div>
            <div class="metric-desc">Gradient Boosting improvement<br/>over linear regression</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Actual vs predicted
    st.markdown('<div class="sec-header">Actual vs Predicted — 15-day Test Window</div>', unsafe_allow_html=True)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=test_df["Date"], y=y_test,
        name="Actual", line=dict(color=CYAN, width=2),
    ))
    fig_pred.add_trace(go.Scatter(
        x=test_df["Date"], y=y_pred_gb,
        name="Gradient Boosting", line=dict(color=BLUE, width=2, dash="dot"),
    ))
    fig_pred.add_trace(go.Scatter(
        x=test_df["Date"], y=y_pred_lr,
        name="Naive Baseline (yesterday's price)", line=dict(color=AMBER, width=1.5, dash="dash"),
    ))
    fig_pred.update_layout(**plotly_layout("", 380))
    fig_pred.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_pred, use_container_width=True)

    col_a, col_b = st.columns(2)

    # Feature importance
    with col_a:
        st.markdown('<div class="sec-header">Top Features · What Drives the Price</div>', unsafe_allow_html=True)
        fi = results["feat_importances"]
        label_map = {
            "Lag1": "Yesterday's close",
            "Lag2": "2 days ago close",
            "Lag3": "3 days ago close",
            "Lag4": "4 days ago close",
            "Lag5": "5 days ago close",
            "MA3":  "3-day avg price",
            "MA7":  "7-day avg price",
            "MA14": "14-day avg price",
            "EMA7": "7-day exp avg",
            "EMA14":"14-day exp avg",
            "Vol7": "7-day volatility",
            "Ret1": "1-day return",
            "Ret3": "3-day return",
            "RSI14":"Momentum (RSI)",
            "HLSpread":"Daily price range",
            "DayOfWeek":"Day of week",
            "Month":"Month",
            "Volume":"Trading volume",
        }
        fi_df = (pd.DataFrame(list(fi.items()), columns=["feat","imp"])
                 .sort_values("imp", ascending=False).head(6))
        fi_df["label"] = fi_df["feat"].map(label_map).fillna(fi_df["feat"])
        fig_fi = go.Figure(go.Bar(
            x=fi_df["imp"], y=fi_df["label"],
            orientation="h",
            marker=dict(
                color=fi_df["imp"],
                colorscale=[[0, BLUE], [1, CYAN]],
                showscale=False,
            ),
        ))
        fig_fi.update_layout(**plotly_layout("", 320))
        fig_fi.update_xaxes(title="Importance score")
        st.plotly_chart(fig_fi, use_container_width=True)

    # Residuals
    with col_b:
        st.markdown('<div class="sec-header">How Far Off Was Each Prediction?</div>', unsafe_allow_html=True)
        residuals = y_test - y_pred_gb
        colors_r  = [GREEN if r >= 0 else RED for r in residuals]
        fig_res   = go.Figure(go.Bar(
            x=test_df["Date"], y=residuals,
            marker_color=colors_r,
            name="Residual",
        ))
        fig_res.add_hline(y=0, line_color=SUBTEXT, line_dash="dot")
        fig_res.update_layout(**plotly_layout("Green = underestimated · Red = overestimated", 320))
        fig_res.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_res, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — 30-Day Forecast
# ══════════════════════════════════════════════════════════════════════════════
elif page == "30-Day Forecast":
    hero("30-Day Forecast", "Projected from 31 Aug 2023 · Portfolio demonstration only")

    fc = build_forecast(df_raw, results, feat_cols, days=30)

    fc_prices = np.array(fc["prices"])
    fc_start  = fc["prices"][0]
    fc_end    = fc["prices"][-1]
    fc_max    = fc_prices.max()
    fc_min    = fc_prices.min()
    fc_pct    = (fc_end - fc_start) / fc_start * 100

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Forecast Start</div>
            <div class="kpi-value">${fc_start:.4f}</div>
            <div class="kpi-delta kpi-neu">1 Sep 2023</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Forecast End</div>
            <div class="kpi-value">${fc_end:.4f}</div>
            <div class="kpi-delta {'kpi-up' if fc_pct>=0 else 'kpi-dn'}">
                {'▲' if fc_pct>=0 else '▼'} {abs(fc_pct):.1f}%
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Projected High</div>
            <div class="kpi-value">${fc_max:.4f}</div>
            <div class="kpi-delta kpi-up">▲ Best case</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Projected Low</div>
            <div class="kpi-value">${fc_min:.4f}</div>
            <div class="kpi-delta kpi-dn">▼ Worst case</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    st.markdown('<div class="sec-header">NRK Price Projection · Full Context + 30-Day Outlook</div>', unsafe_allow_html=True)
    fig_fc = go.Figure()

    # History
    fig_fc.add_trace(go.Scatter(
        x=fc["hist_dates"], y=fc["hist_prices"],
        name="Historical", line=dict(color=CYAN, width=2),
    ))
    # Confidence band
    fig_fc.add_trace(go.Scatter(
        x=fc["dates"] + fc["dates"][::-1],
        y=fc["upper"] + fc["lower"][::-1],
        fill="toself",
        fillcolor=f"rgba(45,142,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence band",
        showlegend=True,
    ))
    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=fc["dates"], y=fc["prices"],
        name="Forecast", line=dict(color=BLUE, width=2, dash="dot"),
    ))
    # Internship end marker
    intern_end = pd.Timestamp("2023-08-31")
    fig_fc.add_vline(
        x=intern_end, line_color=AMBER, line_dash="dash", line_width=1.5,
        annotation_text="Internship End", annotation_font_color=AMBER,
        annotation_position="top right",
    )
    fig_fc.update_layout(**plotly_layout("", 460))
    fig_fc.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_fc, use_container_width=True)

    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This forecast is a portfolio demonstration only.
        Nordek ceased operations after August 2023, and the NRK token is no longer actively traded.
        These projections should not be taken as financial advice or a prediction of actual market outcomes.
    </div>
    """, unsafe_allow_html=True)

    # Day-by-day table
    st.markdown('<div class="sec-header">Day-by-Day Forecast Table</div>', unsafe_allow_html=True)
    fc_table = pd.DataFrame({
        "Date":     [d.strftime("%b %d %Y") for d in fc["dates"]],
        "Projected Price": [f"${p:.5f}" for p in fc["prices"]],
        "Upper Band":      [f"${u:.5f}" for u in fc["upper"]],
        "Lower Band":      [f"${l:.5f}" for l in fc["lower"]],
    })
    st.dataframe(
        fc_table, hide_index=True, use_container_width=True,
        column_config={
            "Date": st.column_config.TextColumn("Date"),
            "Projected Price": st.column_config.TextColumn("Projected Price"),
        }
    )

    # CSV download
    csv_bytes = fc_table.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download forecast as CSV",
        data=csv_bytes,
        file_name="nrk_30day_forecast.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — How It Works
# ══════════════════════════════════════════════════════════════════════════════
elif page == "How It Works":
    hero("How It Works", "A plain-English walkthrough of the project")

    steps = [
        ("01", "Gathering the Data",
         "Every day during the internship, I recorded the NRK token's open, high, low, close price, and trading volume. "
         "By the end of August I had 73 days of clean data — the full June–August 2023 window. "
         "I also cleaned up formatting quirks like dollar signs and K/M suffixes in the raw CSV."),
        ("02", "Teaching the Computer to Read Patterns",
         "Raw prices alone don't tell a great story. So I created 18 derived signals: "
         "rolling averages that smooth out noise, momentum scores that detect whether the price is gaining or losing steam, "
         "price-change rates, and simple 'what was the price N days ago?' lookups. "
         "These 18 numbers become the model's input for each day."),
        ("03", "Training & Testing the Model",
         "I trained a Gradient Boosting model on the first ~80% of days (the training set) "
         "and then checked how well it predicted the last 15 days it had never seen before (the test set). "
         "To make sure the result was meaningful, I also ran a simple linear regression as a naive baseline — "
         "the Gradient Boosting model needed to beat that to be considered useful."),
        ("04", "Turning It Into a Story",
         "Numbers alone aren't useful to stakeholders. I built this dashboard to show "
         "the historical price chart, model accuracy in everyday language (e.g. 'got the direction right X out of 15 days'), "
         "a 30-day forward projection, and this plain-English walkthrough. "
         "The goal was to make the analysis accessible to anyone, not just data scientists."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">{num}</div>
            <div class="step-title">{title}</div>
            <p style='font-size:14px;color:{SUBTEXT};margin:6px 0 0 0;line-height:1.7;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Project timeline
    st.markdown('<div class="sec-header">Project Timeline · Summer 2023</div>', unsafe_allow_html=True)
    timeline_events = [
        ("Jun 2023", BLUE,  "Data collection begins · baseline exploration"),
        ("Jul 2023", CYAN,  "Feature engineering · model training"),
        ("Aug 2023", GREEN, "Model testing · dashboard build · internship end"),
    ]
    fig_tl = go.Figure()
    for i, (label, color, desc) in enumerate(timeline_events):
        fig_tl.add_trace(go.Bar(
            x=[1], y=[1], base=[i],
            orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            name=label,
            hovertemplate=f"<b>{label}</b><br>{desc}<extra></extra>",
            text=f"  {label}  ·  {desc}",
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(family="IBM Plex Mono", size=12, color="#07111F"),
        ))
    fig_tl.update_layout(
        **{**plotly_layout("", 180),
           "barmode":"stack","showlegend":False,
           "yaxis":dict(visible=False),
           "xaxis":dict(visible=False),
           "margin":dict(l=0,r=0,t=10,b=10),
        }
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer" style='margin-top:24px;'>
        ⚠️ <strong>Important note:</strong> This project was built as a portfolio demonstration.
        Nordek ceased operations after August 2023. Nothing here constitutes investment advice.
        All predictions are based solely on historical patterns in a 73-day dataset.
    </div>
    """, unsafe_allow_html=True)
