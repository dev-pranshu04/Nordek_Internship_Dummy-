# 📈 Nordek (NRK) Price Prediction Dashboard

> **Internship Analytics Project · Nordek · June–August 2023**  
> A production-ready Streamlit dashboard combining LSTM deep learning and XGBoost gradient boosting for NRK cryptocurrency price prediction.

---

## 🖼️ Dashboard Preview

The dashboard features 7 interactive pages:
| Page | Description |
|---|---|
| 📊 Data Explorer | Candlestick chart, volume, train/val split visualization |
| 🔬 Feature Engineering | 20+ engineered features, correlation heatmap, technical indicators |
| 🤖 LSTM Model | Predictions, metrics, directional accuracy chart |
| ⚡ XGBoost Model | Feature importances, validation predictions |
| ⚔️ Model Comparison | Side-by-side metrics + radar scorecard |
| 🔮 Predict Next Close | Interactive input form → real-time prediction |
| 💡 Methodology | Architecture rationale, hyperparameters, future roadmap |

---

## 🚀 Deployment: GitHub → Streamlit Cloud (Step-by-Step)

### Step 1 — Prepare Your Repository

```bash
# Create project folder
mkdir nordek-dashboard && cd nordek-dashboard

# Copy these files into the folder:
# ├── app.py                    ← main dashboard
# ├── lstm_full.py              ← full Keras LSTM (local use)
# ├── requirements.txt
# ├── nordek_logo.png           ← place your logo here
# └── README.md
```

### Step 2 — Initialize Git and Push to GitHub

```bash
git init
git add .
git commit -m "feat: Nordek NRK price prediction dashboard"

# Create repo on GitHub (github.com → New Repository → name: nordek-dashboard)
git remote add origin https://github.com/YOUR_USERNAME/nordek-dashboard.git
git branch -M main
git push -u origin main
```

### Step 3 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/nordek-dashboard`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"** — Streamlit Cloud handles all package installation automatically
6. Your app will be live at: `https://YOUR_USERNAME-nordek-dashboard-app-XXXXX.streamlit.app`

> ⏱️ First deployment takes 2–5 minutes. Subsequent updates deploy in ~30 seconds on `git push`.

### Step 4 — Add Logo (Optional but Recommended)

Place `nordek_logo.png` in the root directory of your repo. The dashboard auto-detects it.  
If not found, an elegant text fallback is displayed instead.

---

## 💻 Local Development

### Prerequisites
- Python 3.9–3.11 recommended
- pip or conda

### Quick Setup

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/nordek-dashboard.git
cd nordek-dashboard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
# → Opens at http://localhost:8501
```

### Full LSTM with TensorFlow (local only)

```bash
# Install TensorFlow (not included in Streamlit Cloud requirements)
pip install tensorflow>=2.13.0

# Run the full LSTM training script
python lstm_full.py
```

---

## 🧠 Model Architecture

### LSTM (Long Short-Term Memory)
```
Input: (batch, window=7, features=15)
       ↓
LSTM(64 units, return_sequences=True)
       ↓
Dropout(0.2)
       ↓
LSTM(32 units)
       ↓
Dropout(0.2)
       ↓
Dense(16, activation='relu')
       ↓
Dense(1)  →  predicted Close price

Loss: Huber (robust to outliers)
Optimizer: Adam(lr=0.001)
Callbacks: EarlyStopping(patience=15), ReduceLROnPlateau
```

**Note:** The Streamlit Cloud version uses a Ridge Regression with exponential-weighted lookback (LSTM proxy) for zero-dependency deployment. The `lstm_full.py` file contains the true Keras LSTM — same features, same splits.

### XGBoost
```
n_estimators: 200
max_depth:    4
learning_rate: 0.05
subsample:    0.8
colsample_bytree: 0.8
reg_alpha:    0.1  (L1)
reg_lambda:   1.0  (L2)
```

---

## 📐 Feature Engineering (20+ Features)

| Category | Features |
|---|---|
| **Trend** | MA_3, MA_5, MA_7, MA_14, EMA_7, EMA_14 |
| **Momentum** | RSI_14, RSI_7, MACD, MACD_Signal, Momentum_5, Momentum_10, ROC_5 |
| **Volatility** | Volatility_7, Volatility_14, ATR_7, BB_width_14 |
| **Volume** | Volume_Ratio, OBV, VWAP_proxy, Volume_MA_7 |
| **Candlestick** | Price_Range, Body_Size, Upper_Shadow, Lower_Shadow, Position_in_Range |
| **Bands** | BB_upper_14, BB_lower_14, BB_%B_14 |
| **Lag** | Close_lag_1/2/3, Return_lag_1/2/3 |

---

## 📊 Data Split Strategy

```
73 raw rows → 56 after feature engineering (NaN rows removed)

Timeline: June 20, 2023 → August 31, 2023

TRAIN  (70%): ~39 rows  — model learns price patterns
VALID  (30%): ~17 rows  — model evaluated on unseen data

All metrics (MAE, RMSE, MAPE, Directional Accuracy) computed on validation only.
```

---

## 🗂 Project Structure

```
nordek-dashboard/
├── app.py                  ← Streamlit dashboard (deploy this)
├── lstm_full.py            ← Full Keras LSTM for local training
├── requirements.txt        ← Streamlit Cloud dependencies
├── nordek_logo.png         ← Place your logo here
└── README.md               ← This file
```

---

## 🔭 Future Improvements

1. **More data:** CoinGecko API for 1-2 years of daily OHLCV history
2. **Sentiment:** Twitter/Telegram NRK community sentiment scores
3. **On-chain:** Active wallets, transaction count, staking metrics
4. **Models:** Temporal Fusion Transformer, Prophet + ensemble
5. **Backtesting:** Paper trading simulation with Sharpe ratio evaluation
6. **Real-time:** WebSocket price feed + live prediction refresh

---

## 👨‍💻 About

Built as an internship analytics project at **Nordek** (June–August 2023).  
Demonstrates: data preprocessing → feature engineering → ML model training → deployment → interactive UI.

**Tech stack:** Python · Streamlit · Plotly · scikit-learn · XGBoost · (Keras/TF for full LSTM)

---

*⚠️ Disclaimer: This dashboard is for educational and portfolio purposes only. Not financial advice.*
