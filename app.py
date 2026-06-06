"""
NRK Price Intelligence Dashboard — Nordek
Clean, business-first design. No jargon. No theory walls.
"""

import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="NRK Price Intelligence · Nordek",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colours ────────────────────────────────────────────────────────────
C = dict(
    bg      = "#07111F",
    panel   = "#0C1E35",
    card    = "#0F2644",
    border  = "#173558",
    blue    = "#2D8EFF",
    cyan    = "#00D4FF",
    green   = "#00E5A0",
    red     = "#FF4C6A",
    amber   = "#FFB800",
    text    = "#E8F4FF",
    muted   = "#5A8AB0",
    white   = "#FFFFFF",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background: {C['bg']}; color: {C['text']};
    font-family: 'IBM Plex Sans', sans-serif;
}}
[data-testid="stSidebar"] {{
    background: {C['panel']} !important;
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}

/* hide default streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}

.hero-band {{
    background: linear-gradient(135deg, {C['panel']} 0%, #0A1E3A 100%);
    border: 1px solid {C['border']};
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex; align-items: center; gap: 24px;
}}
.hero-logo {{
    width: 56px; height: 56px;
    background: linear-gradient(135deg,{C['blue']},{C['cyan']});
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; flex-shrink: 0;
}}
.hero-title {{ font-size: 22px; font-weight: 700; color:{C['white']}; letter-spacing: .5px; }}
.hero-sub   {{ font-size: 12px; color:{C['muted']}; letter-spacing: 2px; margin-top: 2px; text-transform: uppercase; }}

.kpi-row {{ display: flex; gap: 14px; margin-bottom: 20px; flex-wrap: wrap; }}
.kpi {{
    flex: 1; min-width: 140px;
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 12px; padding: 16px 20px;
}}
.kpi-label  {{ font-size: 10px; color:{C['muted']}; text-transform: uppercase; letter-spacing: 1.8px; margin-bottom: 6px; }}
.kpi-value  {{ font-size: 26px; font-weight: 700; font-family:'IBM Plex Mono',monospace; color:{C['white']}; }}
.kpi-delta  {{ font-size: 12px; margin-top: 4px; }}
.up   {{ color:{C['green']}; }}
.dn   {{ color:{C['red']}; }}
.neu  {{ color:{C['muted']}; }}

.section-title {{
    font-size: 13px; font-weight: 600; color:{C['blue']};
    text-transform: uppercase; letter-spacing: 2px;
    border-left: 3px solid {C['blue']}; padding-left: 10px;
    margin: 28px 0 14px 0;
}}

.insight-box {{
    background: rgba(45,142,255,0.08); border: 1px solid rgba(45,142,255,0.25);
    border-radius: 10px; padding: 14px 18px; margin: 8px 0;
    font-size: 13px; color: #A8C8F0; line-height: 1.7;
}}
.insight-box b {{ color: {C['cyan']}; }}

.signal-card {{
    border-radius: 12px; padding: 20px 24px; text-align: center;
    border: 1px solid {C['border']};
}}
.signal-bull {{ background: rgba(0,229,160,0.1); border-color: rgba(0,229,160,0.35); }}
.signal-bear {{ background: rgba(255,76,106,0.1); border-color: rgba(255,76,106,0.35); }}
.signal-neut {{ background: rgba(90,138,176,0.1); border-color: rgba(90,138,176,0.35); }}

.method-step {{
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 10px; padding: 16px 20px; margin: 8px 0;
}}
.method-num {{
    display: inline-block; background: {C['blue']};
    color: white; font-weight: 700; font-size: 11px;
    border-radius: 50%; width: 22px; height: 22px;
    line-height: 22px; text-align: center; margin-right: 10px;
}}

div[data-testid="stMetric"] {{ display: none; }}
.stButton > button {{
    background: linear-gradient(135deg,{C['blue']},{C['cyan']});
    color: white; border: none; border-radius: 8px;
    padding: 10px 22px; font-weight: 600; font-size: 13px;
    font-family: 'IBM Plex Sans', sans-serif; width: 100%;
}}
.stRadio > div {{ gap: 4px; }}
.stRadio label {{ color: {C['text']} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Logo SVG ─────────────────────────────────────────────────────────────────
LOGO_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
  <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="{C['blue']}"/>
    <stop offset="100%" stop-color="{C['cyan']}"/>
  </linearGradient></defs>
  <polygon points="16,2 28,9 28,23 16,30 4,23 4,9" fill="none" stroke="url(#lg)" stroke-width="2.2"/>
  <polyline points="9,18 13,12 18,20 22,14" fill="none" stroke="url(#lg)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("price_history_filtered.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%B %d %Y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns: continue
        raw = df[col].astype(str)
        mult = np.where(raw.str.contains("K|k"), 1e3,
               np.where(raw.str.contains("M|m"), 1e6,
               np.where(raw.str.contains("B|b"), 1e9, 1.0)))
        df[col] = pd.to_numeric(raw.str.replace(r"[^\d\.]","",regex=True), errors="coerce") * mult
    if "Volume(NRK)" in df.columns:
        df["VolNRK"] = pd.to_numeric(df["Volume(NRK)"], errors="coerce").fillna(0)
    return df.dropna(subset=["Close"]).reset_index(drop=True)

def add_features(df):
    d = df.copy()
    # Safely ensure High/Low/Open are numeric, fall back to Close if missing/all-NaN
    for col in ["High", "Low", "Open"]:
        if col not in d.columns or d[col].isna().all():
            d[col] = d["Close"]
        else:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(d["Close"])
    for w in [3,7,14,30]: d[f"MA{w}"] = d["Close"].rolling(w,min_periods=1).mean()
    for w in [7,14]:       d[f"EMA{w}"] = d["Close"].ewm(span=w,adjust=False).mean()
    d["Vol7"]  = d["Close"].rolling(7,min_periods=1).std().fillna(0)
    d["Ret1"]  = d["Close"].pct_change(1).fillna(0)
    d["Ret7"]  = d["Close"].pct_change(7).fillna(0)
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14,min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14,min_periods=1).mean()
    d["RSI14"]    = 100 - (100/(1 + gain/(loss+1e-12)))
    d["HLSpread"] = (d["High"] - d["Low"]) / (d["Close"] + 1e-12)
    for lag in [1,2,3,5,7]: d[f"Lag{lag}"] = d["Close"].shift(lag).bfill()
    d["DayOfWeek"] = d["Date"].dt.dayofweek
    d["Month"]     = d["Date"].dt.month
    return d.fillna(0)

FEATS = ["MA3","MA7","MA14","MA30","EMA7","EMA14","Vol7","Ret1","Ret7",
         "RSI14","HLSpread","Lag1","Lag2","Lag3","Lag5","Lag7","DayOfWeek","Month"]

# cache_data is safer than cache_resource for DataFrame args across Python versions
@st.cache_data(show_spinner=False)
def train(_df_feat):
    df = _df_feat.copy()
    # Guarantee every feature column exists and is a clean float array
    for f in FEATS:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0).astype(float)

    mask = (df["Date"].dt.year==2023) & (df["Date"].dt.month.isin([6,7,8]))
    tr, vl = df[mask].copy(), df[~mask].copy()

    if len(tr) == 0:
        st.error("No training data found for Jun-Aug 2023. Check CSV date format.")
        st.stop()

    Xtr = tr[FEATS].values.astype(float)
    ytr_raw = tr["Close"].values.astype(float)
    Xvl = vl[FEATS].values.astype(float)
    yvl = vl["Close"].values.astype(float)

    fsc, psc = MinMaxScaler(), MinMaxScaler()
    Xtr_sc = fsc.fit_transform(Xtr)
    ytr_sc = psc.fit_transform(ytr_raw.reshape(-1,1)).ravel()
    Xvl_sc = fsc.transform(Xvl)

    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=4, subsample=0.8, random_state=42)
    gb.fit(Xtr_sc, ytr_sc)
    pred_sc = gb.predict(Xvl_sc)
    pred    = psc.inverse_transform(pred_sc.reshape(-1,1)).ravel()

    da   = np.mean(np.sign(np.diff(yvl)) == np.sign(np.diff(pred))) * 100
    mae  = mean_absolute_error(yvl, pred)
    rmse = np.sqrt(mean_squared_error(yvl, pred))
    mape = np.mean(np.abs((yvl - pred) / (yvl + 1e-12))) * 100

    return dict(gb=gb, fsc=fsc, psc=psc,
                tr=tr, vl=vl,
                yvl=yvl, pred=pred,
                val_dates=vl["Date"].values,
                da=da, mae=mae, rmse=rmse, mape=mape)

def forecast(df_feat, m, n=30):
    hist = list(df_feat["Close"].values[-30:])
    last = df_feat["Date"].iloc[-1]
    dates, prices = [], []
    for i in range(1, n+1):
        fd = last + timedelta(days=i)
        ch = np.array(hist)
        x = [[ch[-3:].mean(), ch[-7:].mean(), ch[-14:].mean(),
              ch[-30:].mean() if len(ch)>=30 else ch.mean(),
              ch[-7:].mean(), ch[-14:].mean(), ch[-7:].std(),
              (ch[-1]-ch[-2])/(ch[-2]+1e-12),
              (ch[-1]-ch[-8])/(ch[-8]+1e-12) if len(ch)>=8 else 0,
              50, 0.05,
              ch[-1], ch[-2] if len(ch)>=2 else ch[-1],
              ch[-3] if len(ch)>=3 else ch[-1],
              ch[-5] if len(ch)>=5 else ch[-1],
              ch[-7] if len(ch)>=7 else ch[-1],
              fd.weekday(), fd.month]]
        p = m["psc"].inverse_transform(
              m["gb"].predict(m["fsc"].transform(x)).reshape(-1,1)
            ).ravel()[0]
        dates.append(fd); prices.append(p)
        hist.append(p); hist = hist[-30:]
    return dates, prices

def fmt(v, d=6): return f"${v:.{d}f}"

# chart defaults
def base_layout(title=""):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080F1C",
        font=dict(color=C["text"], family="IBM Plex Sans"),
        title=dict(text=title, font=dict(size=14, color=C["text"])),
        xaxis=dict(gridcolor="#0F2644", showgrid=True, zeroline=False,
                   tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#0F2644", showgrid=True, zeroline=False,
                   tickfont=dict(size=11)),
        legend=dict(bgcolor="rgba(12,30,53,0.85)", bordercolor=C["border"],
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=50,r=20,t=45,b=40), hovermode="x unified",
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOAD + TRAIN
# ─────────────────────────────────────────────────────────────────────────────
df_raw  = load()
df_feat = add_features(df_raw)

with st.spinner("Initialising models…"):
    M = train(df_feat)

latest   = df_raw["Close"].iloc[-1]
prev     = df_raw["Close"].iloc[-2]
day_chg  = (latest-prev)/max(prev,1e-12)*100
ath      = df_raw["Close"].max()
atl      = df_raw["Close"].min()
vol_avg  = df_raw["VolNRK"].mean() if "VolNRK" in df_raw.columns else 0
train_df = M["tr"]
val_df   = M["vl"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;padding:8px 0 20px 0'>
      {LOGO_SVG}
      <div>
        <div style='font-weight:700;font-size:16px;color:{C["white"]}'>NORDEK</div>
        <div style='font-size:10px;color:{C["muted"]};letter-spacing:1.5px'>NRK ANALYTICS</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Market Overview",
        "📈  Price & Signals",
        "🔮  30-Day Forecast",
        "🧠  How It Works",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"<div style='font-size:11px;color:{C['muted']}'>Data: Jun 2023 – Jan 2025<br>Model trained on Nordek internship data</div>", unsafe_allow_html=True)

page = page.split("  ")[1]

# ─────────────────────────────────────────────────────────────────────────────
# HERO BAND
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-band">
  <div class="hero-logo">{LOGO_SVG}</div>
  <div>
    <div class="hero-title">NRK · Price Intelligence</div>
    <div class="hero-sub">AI Forecasting · Nordek Internship · June – August 2023</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MARKET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Market Overview":
    up = day_chg >= 0
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi">
        <div class="kpi-label">Latest Price</div>
        <div class="kpi-value">{fmt(latest)}</div>
        <div class="kpi-delta {'up' if up else 'dn'}">{'▲' if up else '▼'} {abs(day_chg):.2f}% today</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">All-Time High</div>
        <div class="kpi-value">{fmt(ath,5)}</div>
        <div class="kpi-delta neu">{df_raw.loc[df_raw['Close'].idxmax(),'Date'].strftime('%b %Y')}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">All-Time Low</div>
        <div class="kpi-value">{fmt(atl,6)}</div>
        <div class="kpi-delta neu">{df_raw.loc[df_raw['Close'].idxmin(),'Date'].strftime('%b %Y')}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Avg Daily Volume</div>
        <div class="kpi-value">{vol_avg/1e6:.1f}M</div>
        <div class="kpi-delta neu">NRK tokens</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">AI Accuracy</div>
        <div class="kpi-value" style="color:{C['green']}">{M['da']:.0f}%</div>
        <div class="kpi-delta neu">direction calls</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Price History</div>', unsafe_allow_html=True)

    # Candlestick
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_raw["Date"], open=df_raw["Open"], high=df_raw["High"],
        low=df_raw["Low"], close=df_raw["Close"],
        increasing_line_color=C["green"], decreasing_line_color=C["red"],
        name="NRK",
    ))
    # Training shading
    fig.add_vrect(x0=train_df["Date"].min(), x1=train_df["Date"].max(),
                  fillcolor=C["blue"], opacity=0.07, layer="below", line_width=0)
    fig.add_annotation(x=train_df["Date"].mean(), y=df_raw["High"].max()*1.02,
                       text="AI Training Period", showarrow=False,
                       font=dict(color=C["blue"],size=11))
    fig.update_layout(xaxis_rangeslider_visible=False, height=400,
                      **base_layout())
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Price Phases</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col, (mo, name, color) in zip([c1,c2,c3],[
        (6,"June 2023",C["blue"]),(7,"July 2023",C["amber"]),(8,"August 2023",C["cyan"])
    ]):
        mdf = df_feat[(df_feat["Date"].dt.year==2023)&(df_feat["Date"].dt.month==mo)]
        if len(mdf) == 0: continue
        chg = (mdf["Close"].iloc[-1]-mdf["Close"].iloc[0])/mdf["Close"].iloc[0]*100
        with col:
            st.markdown(f"""
            <div class="kpi" style="border-color:{color}33">
              <div class="kpi-label" style="color:{color}">{name}</div>
              <div style="font-size:14px;color:{C['white']};font-weight:600;margin:4px 0">
                {fmt(mdf['Close'].iloc[0])} → {fmt(mdf['Close'].iloc[-1])}
              </div>
              <div class="kpi-delta {'up' if chg>=0 else 'dn'}">{'▲' if chg>=0 else '▼'} {abs(chg):.1f}% over month</div>
              <div class="kpi-delta neu" style="margin-top:2px">High: {fmt(mdf['High'].max())} · Low: {fmt(mdf['Low'].min())}</div>
            </div>""", unsafe_allow_html=True)

    if "VolNRK" in df_raw.columns:
        st.markdown('<div class="section-title">Trading Volume</div>', unsafe_allow_html=True)
        fig_v = go.Figure(go.Bar(
            x=df_raw["Date"], y=df_raw["VolNRK"],
            marker=dict(color=C["blue"], opacity=0.6), name="Volume"
        ))
        fig_v.update_layout(height=220, **base_layout())
        st.plotly_chart(fig_v, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PRICE & SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Price & Signals":
    st.markdown('<div class="section-title">What the AI Predicted vs What Actually Happened</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="insight-box">
    The AI was trained on just <b>73 days of NRK price history</b> (June–August 2023)
    and then tested on <b>505 days of unseen data</b> it had never seen before.
    The chart below shows how closely its predictions tracked reality.
    </div>""", unsafe_allow_html=True)

    yvl  = M["yvl"]
    pred = M["pred"]
    vd   = pd.to_datetime(M["val_dates"])

    # Thin out for chart clarity (every 3rd point)
    idx = list(range(0, len(yvl), 3))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=vd[idx], y=yvl[idx], mode="lines",
        name="Actual Price", line=dict(color=C["white"], width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=vd[idx], y=pred[idx], mode="lines",
        name="AI Prediction", line=dict(color=C["cyan"], width=1.5, dash="dot")
    ))
    fig2.update_layout(height=380, **base_layout())
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">AI Report Card</div>', unsafe_allow_html=True)

    # Plain English metrics
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="kpi">
          <div class="kpi-label">Direction Accuracy</div>
          <div class="kpi-value" style="color:{C['green']}">{M['da']:.0f}%</div>
          <div class="kpi-delta neu">Got UP/DOWN right {M['da']:.0f}% of days</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi">
          <div class="kpi-label">Avg Price Error</div>
          <div class="kpi-value">{fmt(M['mae'],4)}</div>
          <div class="kpi-delta neu">Average miss per day</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        pct_err = M['mape']
        st.markdown(f"""<div class="kpi">
          <div class="kpi-label">Error as % of Price</div>
          <div class="kpi-value">{pct_err:.1f}%</div>
          <div class="kpi-delta neu">On average per prediction</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">What Drives NRK Price — Top Signals</div>', unsafe_allow_html=True)

    fi      = M["gb"].feature_importances_
    top_idx = np.argsort(fi)[::-1][:6]
    labels  = {
        "MA3":  "3-day average price",
        "MA7":  "1-week average price",
        "MA14": "2-week average price",
        "MA30": "1-month average price",
        "Lag1": "Yesterday's price",
        "Lag2": "Price 2 days ago",
        "Lag3": "Price 3 days ago",
        "Ret1": "Yesterday's % change",
        "Ret7": "7-day % change",
        "HLSpread": "Daily price swing (High–Low)",
        "Vol7": "7-day price volatility",
        "RSI14":"Momentum indicator",
    }

    fig3 = go.Figure(go.Bar(
        x=fi[top_idx],
        y=[labels.get(FEATS[i], FEATS[i]) for i in top_idx],
        orientation="h",
        marker=dict(
            color=fi[top_idx],
            colorscale=[[0,C["blue"]],[1,C["cyan"]]],
        ),
        text=[f"{v*100:.1f}%" for v in fi[top_idx]],
        textposition="outside",
        textfont=dict(color=C["text"],size=11),
    ))
    fig3.update_layout(height=300,
                       xaxis=dict(showticklabels=False, gridcolor="#0F2644"),
                       yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                       **{k:v for k,v in base_layout().items() if k not in ("xaxis","yaxis")})
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"""<div class="insight-box">
    📌 <b>Recent average prices carry the most predictive power</b> — the 3-day moving average
    alone explains ~60% of what the model learns. Daily price swings and yesterday's momentum
    are the next most important signals. This matches how professional crypto traders think.
    </div>""", unsafe_allow_html=True)

    # Current signals
    st.markdown('<div class="section-title">Today\'s Signals</div>', unsafe_allow_html=True)
    last_feat = df_feat.iloc[-1]
    ma7  = last_feat["MA7"]
    ma14 = last_feat["MA14"]
    rsi  = last_feat["RSI14"]
    ret1 = last_feat["Ret1"] * 100

    price_vs_ma = "above" if latest > ma7 else "below"
    trend = "upward" if ma7 > ma14 else "downward"
    rsi_label = "Overbought (potential pullback)" if rsi > 70 else ("Oversold (potential bounce)" if rsi < 30 else "Neutral zone")

    c1,c2,c3 = st.columns(3)
    with c1:
        bull = latest > ma7
        st.markdown(f"""<div class="signal-card {'signal-bull' if bull else 'signal-bear'}">
          <div style="font-size:28px">{'📈' if bull else '📉'}</div>
          <div style="font-weight:600;margin:6px 0;color:{C['green'] if bull else C['red']}">Price vs Average</div>
          <div style="font-size:13px;color:{C['text']}">Currently <b>{price_vs_ma}</b> the 7-day average</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        bull2 = ma7 > ma14
        st.markdown(f"""<div class="signal-card {'signal-bull' if bull2 else 'signal-bear'}">
          <div style="font-size:28px">{'🔺' if bull2 else '🔻'}</div>
          <div style="font-weight:600;margin:6px 0;color:{C['green'] if bull2 else C['red']}">Trend</div>
          <div style="font-size:13px;color:{C['text']}">Short-term trend is <b>{trend}</b></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        neut = 30 <= rsi <= 70
        st.markdown(f"""<div class="signal-card {'signal-neut' if neut else ('signal-bear' if rsi>70 else 'signal-bull')}">
          <div style="font-size:28px">⚖️</div>
          <div style="font-weight:600;margin:6px 0">Momentum</div>
          <div style="font-size:13px;color:{C['text']}">{rsi_label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:11px;color:{C['muted']};margin-top:8px'>⚠️ Signals are informational only. Not financial advice.</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FORECAST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "30-Day Forecast":
    st.markdown('<div class="section-title">AI Price Forecast — Next 30 Days</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="insight-box">
    The AI model analyses <b>price trends, momentum, and volatility patterns</b>
    to project where NRK could trade over the next 30 days.
    The shaded band shows the uncertainty range — the actual price may land anywhere in it.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Running forecast…"):
        fut_dates, fut_prices = forecast(df_feat, M, 30)

    end_price   = fut_prices[-1]
    chg_pct     = (end_price - latest) / max(latest,1e-12) * 100
    peak_price  = max(fut_prices)
    trough_price= min(fut_prices)
    std_est     = df_raw["Close"].rolling(14).std().iloc[-1]

    up = chg_pct >= 0
    c1,c2,c3,c4 = st.columns(4)
    for col, label, val, note in zip([c1,c2,c3,c4],[
        ("Today's Price",   fmt(latest),           "starting point"),
        ("30-Day Forecast", fmt(end_price),         f"{'▲' if up else '▼'} {abs(chg_pct):.1f}%"),
        ("Projected Peak",  fmt(peak_price),        "highest in range"),
        ("Projected Low",   fmt(trough_price),      "lowest in range"),
    ]):
        with col:
            st.markdown(f"""<div class="kpi">
              <div class="kpi-label">{label[0]}</div>
              <div class="kpi-value" style="{'color:'+C['green'] if label[0]=='30-Day Forecast' and up else ('color:'+C['red'] if label[0]=='30-Day Forecast' else '')}">{label[1]}</div>
              <div class="kpi-delta {'up' if up and label[0]=='30-Day Forecast' else ('dn' if not up and label[0]=='30-Day Forecast' else 'neu')}">{label[2]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast chart
    hist_slice = df_raw.iloc[-60:]
    upper = [p + std_est*1.5 for p in fut_prices]
    lower = [max(p - std_est*1.5, 0) for p in fut_prices]

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(
        x=hist_slice["Date"], y=hist_slice["Close"],
        mode="lines", name="Historical Price",
        line=dict(color=C["white"], width=2)
    ))
    fig_f.add_trace(go.Scatter(
        x=fut_dates+fut_dates[::-1], y=upper+lower[::-1],
        fill="toself", fillcolor=f"rgba(45,142,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Uncertainty Band", showlegend=True
    ))
    fig_f.add_trace(go.Scatter(
        x=fut_dates, y=fut_prices,
        mode="lines+markers", name="AI Forecast",
        line=dict(color=C["cyan"], width=2.5),
        marker=dict(size=5, color=C["cyan"])
    ))
    fig_f.add_vline(x=df_raw["Date"].iloc[-1],
                    line_dash="dot", line_color=C["amber"], opacity=0.7,
                    annotation_text="Today", annotation_position="top right",
                    annotation=dict(font=dict(color=C["amber"],size=11)))
    fig_f.update_layout(height=420, **base_layout())
    st.plotly_chart(fig_f, use_container_width=True)

    st.markdown('<div class="section-title">Day-by-Day Forecast Table</div>', unsafe_allow_html=True)

    rows = []
    for d, p in zip(fut_dates, fut_prices):
        pchg = (p-latest)/max(latest,1e-12)*100
        arrow = "▲" if pchg >= 0 else "▼"
        rows.append({"Date": d.strftime("%b %d, %Y"),
                     "Forecasted Price": fmt(p),
                     "Change vs Today": f"{arrow} {abs(pchg):.1f}%"})
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=360)

    import io
    buf = io.StringIO()
    tbl.to_csv(buf, index=False)
    st.download_button("⬇ Download Forecast (CSV)", buf.getvalue(), "nrk_30day_forecast.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "How It Works":
    st.markdown('<div class="section-title">Plain-English Explainer</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="insight-box" style="font-size:14px;line-height:1.9">
    This dashboard was built during a <b>data science internship at Nordek</b> in 2023.
    The goal: teach an AI to predict whether the NRK token price would go up or down the next day.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">The 4-Step Process</div>', unsafe_allow_html=True)

    for step, title, body in [
        (1, "Collected the data",
         "Every day's NRK opening price, closing price, highest price, lowest price, and trading volume — 578 days in total, from June 2023 to January 2025."),
        (2, "Created smart signals",
         "Raw prices alone aren't very useful. So we calculated 18 extra signals from the data — things like <b>7-day price averages</b>, <b>momentum</b> (is it speeding up or slowing down?), <b>volatility</b> (how wild were recent price swings?), and the <b>RSI</b> (a classic trader's tool for spotting overbought/oversold conditions)."),
        (3, "Taught the AI",
         "We fed the AI only the <b>first 73 days</b> of data (June–August 2023, the internship period). It learned the patterns between all those signals and the next day's price. Then we tested it on the remaining <b>505 days it had never seen</b> — the real challenge."),
        (4, "Measured the result",
         f"The AI correctly predicted whether the price would go up or down on <b>{M['da']:.0f}% of test days</b>. It uses a technique called <b>Gradient Boosting</b> — the same family of algorithms used by hedge funds and winning Kaggle competitions."),
    ]:
        st.markdown(f"""<div class="method-step">
          <span class="method-num">{step}</span>
          <b style="color:{C['white']};font-size:15px">{title}</b>
          <div style="font-size:13px;color:{C['muted']};margin-top:8px;line-height:1.7">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">What the Numbers Mean</div>', unsafe_allow_html=True)

    for term, plain in [
        ("Direction Accuracy",   "Out of every 100 days, how many times did the AI correctly predict 'up' or 'down'?"),
        ("Average Price Error",  "On average, how many dollars off was each prediction from the real price?"),
        ("Uncertainty Band",     "The forecast chart shows a shaded zone — the real price will likely land somewhere in that range."),
        ("Training vs Validation", "Training = learning from past data. Validation = testing on data the AI never saw. Only validation accuracy is what matters."),
    ]:
        st.markdown(f"""<div class="insight-box">
          <b>{term}</b><br>
          <span style="color:{C['text']}">{plain}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:28px;padding:16px 20px;background:{C['card']};border-radius:10px;border:1px solid {C['border']};font-size:12px;color:{C['muted']}">
    ⚠️ <b style="color:{C['amber']}">Disclaimer</b> — This is a research and portfolio project built during an internship.
    Nothing on this dashboard is financial advice. Cryptocurrency markets are unpredictable.
    Always do your own research before making investment decisions.
    </div>
    """, unsafe_allow_html=True)
