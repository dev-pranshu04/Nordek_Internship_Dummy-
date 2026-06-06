"""
lstm_full.py  —  Full Keras/TensorFlow LSTM for local execution
Run: python lstm_full.py
Requires: pip install tensorflow keras scikit-learn pandas numpy
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Paste your CSV path or use embedded data ─────────────────────────
CSV_PATH = "price_history_filtered.csv"   # or use embedded string

EMBEDDED_CSV = """Date,Open,High,Low,Close,Volume,Volume(NRK),Market Cap
August 31 2023,$0.0836,$0.0844,$0.0809,$0.0829,$2.8 m,33650722,$0
August 30 2023,$0.0846,$0.0847,$0.0833,$0.0837,$2.3 m,27860741,$0
August 29 2023,$0.0868,$0.0886,$0.0820,$0.0846,$2.3 m,26989697,$0
August 28 2023,$0.0871,$0.0893,$0.0834,$0.0871,$2.8 m,32024959,$0
August 27 2023,$0.0827,$0.0866,$0.0810,$0.0865,$1.8 m,21167556,$0
August 26 2023,$0.0833,$0.0838,$0.0794,$0.0826,$1.5 m,18126409,$0
August 25 2023,$0.0862,$0.0875,$0.0826,$0.0833,$1.4 m,16793091,$0
August 24 2023,$0.0870,$0.0900,$0.0853,$0.0862,$1.6 m,18636268,$0
August 23 2023,$0.0874,$0.0904,$0.0859,$0.0871,$1.7 m,18827756,$0
August 22 2023,$0.0855,$0.0935,$0.0844,$0.0875,$1.8 m,20205614,$0
August 21 2023,$0.0882,$0.0884,$0.0837,$0.0857,$1.6 m,19071105,$0
August 20 2023,$0.0857,$0.0909,$0.0848,$0.0883,$1.8 m,20133740,$0
August 19 2023,$0.0842,$0.0863,$0.0830,$0.0856,$1.4 m,16691717,$0
August 18 2023,$0.0849,$0.0850,$0.0820,$0.0842,$1.7 m,20841902,$0
August 17 2023,$0.0857,$0.0858,$0.0841,$0.0850,$1.6 m,19124900,$0
August 16 2023,$0.0854,$0.0871,$0.0853,$0.0857,$1.6 m,18338262,$0
August 15 2023,$0.0869,$0.0873,$0.0841,$0.0854,$1.5 m,17912363,$0
August 14 2023,$0.0861,$0.0890,$0.0852,$0.0868,$1.5 m,17247427,$0
August 13 2023,$0.0856,$0.0877,$0.0837,$0.0860,$1.9 m,22562067,$0
August 12 2023,$0.0881,$0.0900,$0.0839,$0.0857,$1.8 m,20501916,$0
August 11 2023,$0.0894,$0.0907,$0.0880,$0.0881,$1.6 m,18461987,$0
August 10 2023,$0.0895,$0.0929,$0.0885,$0.0895,$1.8 m,20220382,$0
August 09 2023,$0.0881,$0.0906,$0.0880,$0.0893,$1.4 m,15717946,$0
August 08 2023,$0.0905,$0.0911,$0.0880,$0.0881,$1.4 m,15474936,$0
August 07 2023,$0.0925,$0.0936,$0.0892,$0.0907,$1.6 m,17175582,$0
August 06 2023,$0.0925,$0.0940,$0.0866,$0.0924,$1.3 m,13850033,$0
August 05 2023,$0.0905,$0.0949,$0.0842,$0.0925,$1.4 m,15213537,$0
August 04 2023,$0.0871,$0.0932,$0.0863,$0.0906,$1.0 m,11512163,$0
August 03 2023,$0.0865,$0.0899,$0.0838,$0.0872,$1.2 m,13853181,$0
August 02 2023,$0.0859,$0.0878,$0.0854,$0.0865,$927.5 K,10747868,$0
August 01 2023,$0.0867,$0.0871,$0.0841,$0.0859,$799.1 K,9289140,$0
July 31 2023,$0.0865,$0.0880,$0.0850,$0.0865,$882.4 K,10217565,$0
July 30 2023,$0.0861,$0.0872,$0.0852,$0.0864,$1.0 m,11620410,$0
July 29 2023,$0.0860,$0.0866,$0.0850,$0.0861,$1.3 m,15452928,$0
July 28 2023,$0.0877,$0.0889,$0.0716,$0.0860,$768.3 K,8961163,$0
July 27 2023,$0.0852,$0.1000,$0.0851,$0.0877,$806.9 K,9000192,$0
July 26 2023,$0.0804,$0.0882,$0.0781,$0.0852,$1.2 m,14572748,$0
July 25 2023,$0.0800,$0.0805,$0.0736,$0.0804,$1.2 m,15515230,$0
July 24 2023,$0.0768,$0.0850,$0.0740,$0.0800,$1.1 m,13634055,$0
July 23 2023,$0.0751,$0.0834,$0.0746,$0.0767,$1.2 m,15947152,$0
July 22 2023,$0.0753,$0.0769,$0.0707,$0.0751,$1.2 m,16465082,$0
July 21 2023,$0.0757,$0.0780,$0.0749,$0.0753,$1.1 m,14788296,$0
July 20 2023,$0.0815,$0.0821,$0.0740,$0.0758,$952.3 K,11925035,$0
July 19 2023,$0.0794,$0.0821,$0.0787,$0.0814,$981.2 K,12167940,$0
July 18 2023,$0.0819,$0.0831,$0.0791,$0.0791,$1.2 m,15244252,$0
July 17 2023,$0.0843,$0.0849,$0.0811,$0.0820,$1.4 m,17281154,$0
July 16 2023,$0.0839,$0.0849,$0.0827,$0.0842,$1.1 m,13581706,$0
July 15 2023,$0.0849,$0.0849,$0.0825,$0.0839,$1.4 m,16415077,$0
July 14 2023,$0.0840,$0.0862,$0.0820,$0.0849,$1.1 m,13396554,$0
July 13 2023,$0.0855,$0.0877,$0.0830,$0.0840,$1.1 m,13102409,$0
July 12 2023,$0.0847,$0.0880,$0.0840,$0.0855,$1.4 m,16251299,$0
July 11 2023,$0.0858,$0.0869,$0.0842,$0.0847,$1.5 m,17293015,$0
July 10 2023,$0.0849,$0.0863,$0.0820,$0.0858,$1.6 m,19226128,$0
July 09 2023,$0.0859,$0.0864,$0.0829,$0.0849,$1.5 m,17159667,$0
July 08 2023,$0.0870,$0.0876,$0.0843,$0.0861,$1.8 m,21008640,$0
July 07 2023,$0.0878,$0.0879,$0.0845,$0.0871,$1.0 m,11748683,$0
July 06 2023,$0.0872,$0.0884,$0.0826,$0.0878,$1.6 m,18020778,$0
July 05 2023,$0.0863,$0.0890,$0.0858,$0.0873,$1.7 m,19322633,$0
July 04 2023,$0.0891,$0.0903,$0.0863,$0.0865,$1.5 m,16932105,$0
July 03 2023,$0.0887,$0.0915,$0.0873,$0.0894,$1.4 m,15882748,$0
July 02 2023,$0.0902,$0.0907,$0.0882,$0.0887,$1.6 m,17895683,$0
July 01 2023,$0.0905,$0.0910,$0.0879,$0.0902,$1.4 m,15279059,$0
June 30 2023,$0.0906,$0.0920,$0.0855,$0.0906,$1.4 m,15431406,$0
June 29 2023,$0.0899,$0.0943,$0.0890,$0.0907,$1.6 m,17359353,$0
June 28 2023,$0.0929,$0.0934,$0.0886,$0.0902,$1.7 m,18784592,$0
June 27 2023,$0.0899,$0.0950,$0.0884,$0.0930,$1.5 m,16337259,$0
June 26 2023,$0.0918,$0.0942,$0.0883,$0.0893,$1.6 m,17689537,$0
June 25 2023,$0.0931,$0.0951,$0.0910,$0.0921,$1.6 m,17261898,$0
June 24 2023,$0.0948,$0.0960,$0.0907,$0.0930,$1.6 m,17249867,$0
June 23 2023,$0.1098,$0.1180,$0.0940,$0.0948,$1.5 m,14405458,$0
June 22 2023,$0.0834,$0.1097,$0.0810,$0.1093,$1.9 m,22142535,$0
June 21 2023,$0.0897,$0.0901,$0.0807,$0.0830,$1.0 m,11802029,$0
June 20 2023,$0.0895,$0.0918,$0.0876,$0.0898,$1.4 m,15309419,$0"""


def parse_volume(v):
    v = str(v).strip().replace("$","").replace(",","")
    if "m" in v.lower(): return float(v.lower().replace("m","").strip()) * 1e6
    if "k" in v.lower(): return float(v.lower().replace("k","").strip()) * 1e3
    try: return float(v)
    except: return np.nan


def load_data():
    try:
        df = pd.read_csv(CSV_PATH)
    except:
        df = pd.read_csv(StringIO(EMBEDDED_CSV))
    df["Date"] = pd.to_datetime(df["Date"], format="%B %d %Y")
    df = df.sort_values("Date").reset_index(drop=True)
    for c in ["Open","High","Low","Close"]:
        df[c] = df[c].str.replace("$","",regex=False).astype(float)
    df["Volume_USD"] = df["Volume"].apply(parse_volume)
    return df


def engineer_features(df):
    df["Return"]      = df["Close"].pct_change()
    df["Price_Range"] = df["High"] - df["Low"]
    df["Body_Size"]   = (df["Close"] - df["Open"]).abs()
    for w in [3,5,7,14]:
        df[f"MA_{w}"]  = df["Close"].rolling(w).mean()
        df[f"EMA_{w}"] = df["Close"].ewm(span=w,adjust=False).mean()
    def rsi(s, p=14):
        d=s.diff(); g=d.clip(lower=0).rolling(p).mean(); l=(-d.clip(upper=0)).rolling(p).mean()
        return 100-(100/(1+g/l.replace(0,np.nan)))
    df["RSI_14"] = rsi(df["Close"])
    ema12 = df["Close"].ewm(span=12,adjust=False).mean()
    ema26 = df["Close"].ewm(span=26,adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9,adjust=False).mean()
    df["Volatility_7"]  = df["Return"].rolling(7).std()
    df["Volatility_14"] = df["Return"].rolling(14).std()
    df["Volume_MA_7"]   = df["Volume_USD"].rolling(7).mean()
    df["Volume_Ratio"]  = df["Volume_USD"] / df["Volume_MA_7"]
    df["Momentum_5"]    = df["Close"]/df["Close"].shift(5)-1
    mid = df["Close"].rolling(14).mean(); std = df["Close"].rolling(14).std()
    df["BB_upper"] = mid+2*std; df["BB_lower"] = mid-2*std
    df["BB_%B"]    = (df["Close"]-df["BB_lower"])/(df["BB_upper"]-df["BB_lower"])
    df["Position_in_Range"] = np.where(df["Price_Range"]>0,(df["Close"]-df["Low"])/df["Price_Range"],0.5)
    df["Close_lag_1"]  = df["Close"].shift(1)
    df["Return_lag_1"] = df["Return"].shift(1)
    df["Target_Close"] = df["Close"].shift(-1)
    return df.dropna().reset_index(drop=True)


def build_sequences(X_scaled, y_scaled, window=7):
    Xs, ys = [], []
    for i in range(window, len(X_scaled)):
        Xs.append(X_scaled[i-window:i])
        ys.append(y_scaled[i])
    return np.array(Xs), np.array(ys)


def build_lstm(input_shape):
    """Full Keras LSTM — requires tensorflow"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="huber")
    return model


def main():
    FEATURES = ["Open","High","Low","Close","Volume_USD","RSI_14",
                 "Volatility_7","MA_7","MA_14","MACD","Momentum_5",
                 "BB_%B","Volume_Ratio","Position_in_Range","Close_lag_1"]

    df = load_data()
    df = engineer_features(df)
    print(f"✅ Loaded {len(df)} rows after feature engineering")

    split = int(len(df) * 0.70)
    print(f"📊 Train: {split} rows | Val: {len(df)-split} rows")

    feat_cols = [c for c in FEATURES if c in df.columns]
    X = df[feat_cols].values
    y = df["Target_Close"].values

    sx = MinMaxScaler(); sy = MinMaxScaler()
    X_s = sx.fit_transform(X)
    y_s = sy.fit_transform(y.reshape(-1,1)).ravel()

    window = 7
    Xs, ys = build_sequences(X_s, y_s, window)
    split_w = split - window

    X_tr, y_tr = Xs[:split_w], ys[:split_w]
    X_vl, y_vl = Xs[split_w:], ys[split_w:]

    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        model = build_lstm((window, len(feat_cols)))
        model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=150, batch_size=16, verbose=1,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=8)
            ]
        )
        val_pred_s = model.predict(X_vl).ravel()
    except ImportError:
        print("⚠️  TensorFlow not found. Falling back to Ridge regression proxy.")
        from sklearn.linear_model import Ridge
        X_tr_f = X_tr.reshape(len(X_tr), -1)
        X_vl_f = X_vl.reshape(len(X_vl), -1)
        model = Ridge(0.5).fit(X_tr_f, y_tr)
        val_pred_s = model.predict(X_vl_f)

    val_pred = sy.inverse_transform(val_pred_s.reshape(-1,1)).ravel()
    val_true = sy.inverse_transform(y_vl.reshape(-1,1)).ravel()

    mae  = mean_absolute_error(val_true, val_pred)
    rmse = np.sqrt(mean_squared_error(val_true, val_pred))
    mape = np.mean(np.abs((val_true-val_pred)/val_true))*100
    da   = np.mean(np.sign(np.diff(val_true))==np.sign(np.diff(val_pred)))*100

    print(f"\n📈 LSTM Results:")
    print(f"   MAE:  ${mae:.6f}")
    print(f"   RMSE: ${rmse:.6f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Directional Accuracy: {da:.1f}%")


if __name__ == "__main__":
    main()
