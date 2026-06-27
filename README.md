<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a1a,50:001833,100:003366&height=220&section=header&text=Stock%20Price%20Predictor&fontSize=42&fontColor=4da6ff&fontAlignY=36&desc=POWERGRID.NS%20%7C%20LSTM%20Deep%20Learning%20%7C%20Time%20Series%20Forecasting&descAlignY=58&descColor=99ccff&animation=fadeIn" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-Live%20Data-brightgreen?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

<br/>

> **A deep learning LSTM model trained on 17 years of POWERGRID.NS historical stock data (2007–2024) to predict future closing prices — achieving a final training loss of 0.0021 after 50 epochs.**

<br/>

[![Portfolio](https://img.shields.io/badge/🌐%20Portfolio-anandsavarn.vercel.app-00C896?style=for-the-badge)](https://anandsavarn.vercel.app)
[![GitHub](https://img.shields.io/badge/GitHub-Anandsavarn-181717?style=for-the-badge&logo=github)](https://github.com/Anandsavarn)
[![Predictor.com](https://img.shields.io/badge/🚀%20Live%20Platform-predictor--65n3.onrender.com-4da6ff?style=for-the-badge)](https://predictor-65n3.onrender.com)

</div>

---

## 📘 Project Overview

This project builds a **multi-layer LSTM (Long Short-Term Memory)** neural network to predict the closing stock price of **POWERGRID Corporation of India (NSE: POWERGRID.NS)**.

Using **17 years of historical market data** (Oct 2007 – Nov 2024), the model learns temporal patterns in stock price movements and generates predictions on unseen test data — visualized against actual prices.

This notebook is part of the **Predictor.com** AI stock analytics platform.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Stock** | POWERGRID Corporation of India Ltd |
| **Ticker** | `POWERGRID.NS` (NSE India) |
| **Source** | `yfinance` Python library |
| **Period** | 01 Jan 2000 → 01 Nov 2024 |
| **Actual Data Start** | 05 Oct 2007 (first available) |
| **Total Records** | 4,208 trading days |
| **Features Used** | Close, Open, High, Low, Volume |
| **Target Variable** | Closing Price (₹) |

### Dataset Statistics

| Metric | Close Price (₹) | Volume |
|---|---|---|
| Mean | 76.94 | 11.89M |
| Std Dev | 62.45 | 20.62M |
| Min | 18.24 | 0 |
| Max | 348.71 | 855.22M |
| Median | 53.28 | 8.34M |

---

## 🔄 Project Pipeline

```
yfinance API
     │
     ▼
Raw OHLCV Data (4,208 rows)
     │
     ▼
Data Cleaning & EDA
  ├── Null check (0 missing values)
  ├── Shape verification (4208 × 5)
  ├── Descriptive statistics
  └── Export to powergrid.csv
     │
     ▼
Visualization
  ├── Closing Price trend (Matplotlib)
  ├── Opening Price trend
  ├── High Price trend
  ├── Volume trend
  ├── Candlestick chart (Plotly)
  ├── MA100 + MA200 (Simple Moving Average)
  └── EMA100 + EMA200 (Exponential Moving Average)
     │
     ▼
Train / Test Split  →  70% Train | 30% Test
                        (2,945 rows) | (1,263 rows)
     │
     ▼
MinMaxScaler  →  Normalize to [0, 1]
     │
     ▼
Sliding Window (100 days look-back)
  x_train: (2845, 100, 1)
  y_train: (2845,)
     │
     ▼
4-Layer LSTM Model (178,761 parameters)
     │
     ▼
Training: 50 Epochs | Adam | MSE Loss
  Final Loss: 0.0021
     │
     ▼
Prediction on Test Set
  x_test: (1263, 100, 1)
     │
     ▼
Inverse Scale → ₹ Price
     │
     ▼
Actual vs Predicted Plot + Model Saved (.keras)
```

---

## 🧠 LSTM Model Architecture

```python
model = Sequential()

# Layer 1
model.add(LSTM(units=50,  activation='relu', return_sequences=True, input_shape=(100, 1)))
model.add(Dropout(0.2))

# Layer 2
model.add(LSTM(units=60,  activation='relu', return_sequences=True))
model.add(Dropout(0.3))

# Layer 3
model.add(LSTM(units=80,  activation='relu', return_sequences=True))
model.add(Dropout(0.4))

# Layer 4
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

# Output
model.add(Dense(units=1))
```

### Model Summary

| Layer | Output Shape | Parameters |
|---|---|---|
| LSTM (50 units) | (None, 100, 50) | 10,400 |
| Dropout (0.2) | (None, 100, 50) | 0 |
| LSTM (60 units) | (None, 100, 60) | 26,640 |
| Dropout (0.3) | (None, 100, 60) | 0 |
| LSTM (80 units) | (None, 100, 80) | 45,120 |
| Dropout (0.4) | (None, 100, 80) | 0 |
| LSTM (120 units) | (None, 120) | 96,480 |
| Dropout (0.5) | (None, 120) | 0 |
| Dense (1 unit) | (None, 1) | 121 |
| **Total** | — | **178,761** |

**Optimizer:** Adam &nbsp;|&nbsp; **Loss:** Mean Squared Error &nbsp;|&nbsp; **Epochs:** 50

---

## 📉 Training Performance

| Epoch | Loss |
|---|---|
| 1 | 0.0388 |
| 5 | 0.0058 |
| 10 | 0.0042 |
| 20 | 0.0030 |
| 30 | 0.0024 |
| 40 | 0.0026 |
| **50** | **0.0021** ✅ |

Loss decreased steadily from **0.0388 → 0.0021** over 50 epochs — confirming the model is learning the temporal patterns in the data.

---

## 📈 Visualizations Generated

| Chart | Description |
|---|---|
| 📉 Closing Price Trend | 17-year POWERGRID closing price line chart |
| 📈 Opening Price Trend | Opening price over time |
| 🔺 High Price Trend | Daily highs over the full period |
| 📦 Volume Trend | Trading volume over time |
| 🕯️ Candlestick Chart | Interactive OHLC candlestick via Plotly |
| 〰️ MA100 + MA200 | Simple Moving Average overlay on closing price |
| 〰️ EMA100 + EMA200 | Exponential Moving Average overlay |
| 🎯 Actual vs Predicted | Final comparison of real vs LSTM-predicted price |

---

## 🛠️ Tech Stack

```
Language        →  Python 3.10+
Deep Learning   →  TensorFlow / Keras (LSTM, Dense, Dropout)
Data Source     →  yfinance (NSE live + historical)
Data Handling   →  Pandas, NumPy
Preprocessing   →  Scikit-learn (MinMaxScaler)
Visualization   →  Matplotlib, Plotly (Candlestick)
Notebook        →  Jupyter Notebook / Google Colab
Model Format    →  .keras (stock_model.keras)
```

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install yfinance pandas numpy matplotlib plotly scikit-learn tensorflow keras
```

### Run the Notebook

```bash
git clone https://github.com/Anandsavran/Stock-Price-Predictor-LSTM.git
cd Stock-Price-Predictor-LSTM
jupyter notebook stock_prediction.ipynb
```

### Change the Stock Symbol

```python
# Line 4 of the notebook — change to any NSE/BSE ticker
stock = "POWERGRID.NS"   # Change to "RELIANCE.NS", "TCS.NS", "INFY.NS" etc.
```

---

## 📁 Repository Structure

```
📦 Stock-Price-Predictor-LSTM
 ┣ 📓 stock_prediction.ipynb    ← Main Jupyter notebook (full pipeline)
 ┣ 📋 powergrid.csv             ← Exported historical data
 ┣ 🧠 stock_model.keras         ← Saved trained LSTM model
 ┣ 📄 requirements.txt          ← Python dependencies
 ┣ 📄 README.md                 ← Project documentation
 ┗ 📁 assets/
    ┣ 🖼️ closing_price.png      ← Closing price chart
    ┣ 🖼️ ma_ema_chart.png       ← Moving averages chart
    ┣ 🖼️ candlestick.png        ← Candlestick chart
    └── 🖼️ actual_vs_predicted.png ← Final prediction comparison
```

---

## 💡 Key Concepts Used

| Concept | Where Used |
|---|---|
| **LSTM** | Captures long-term dependencies in sequential stock data |
| **Dropout** | Prevents overfitting (rates: 0.2, 0.3, 0.4, 0.5) |
| **MinMaxScaler** | Normalizes prices to [0,1] for stable training |
| **Sliding Window (100 days)** | Each prediction uses the previous 100 trading days |
| **70/30 Train-Test Split** | Standard time-series split (no shuffle) |
| **MA100 / MA200** | Trend indicators using `rolling().mean()` |
| **EMA100 / EMA200** | Trend indicators using `ewm(span=).mean()` |
| **Candlestick Chart** | Interactive OHLC visualization via Plotly |

---

## 🔮 Future Improvements

- [ ] 📡 Real-time prediction endpoint via Flask API
- [ ] 📊 Add RSI, MACD, Bollinger Bands as input features
- [ ] 🧠 Experiment with Transformer / Attention models
- [ ] 🌐 Multi-stock support (Reliance, TCS, Infosys)
- [ ] 📱 Deploy prediction dashboard on Streamlit
- [ ] 📉 Add confidence interval bands to predictions
- [ ] 🔁 Walk-forward validation for more robust backtesting
- [ ] 🤖 Integrate with Predictor.com platform as the core ML engine

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**. Stock price predictions are inherently uncertain. Do **not** use this model to make real financial decisions. Past performance does not guarantee future results.

---

## 📖 References

- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [TensorFlow LSTM Guide](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Keras Sequential Model](https://keras.io/guides/sequential_model/)
- [NSE India — POWERGRID](https://www.nseindia.com/get-quotes/equity?symbol=POWERGRID)
- [Scikit-learn MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

---

<div align="center">

### 👨‍💻 Author

**Anand Kumar**
B.Tech – Computer Science Engineering (Data Science)
Lovely Professional University, Punjab

*This model powers the AI prediction engine inside **[Predictor.com](https://predictor-65n3.onrender.com)***

<br/>

[![Portfolio](https://img.shields.io/badge/🌐%20Portfolio-anandsavarn.vercel.app-00C896?style=for-the-badge)](https://anandsavarn.vercel.app)
[![Predictor.com](https://img.shields.io/badge/🚀%20Predictor.com-Live%20Demo-4da6ff?style=for-the-badge)](https://predictor-65n3.onrender.com)
[![GitHub](https://img.shields.io/badge/GitHub-Anandsavarn-181717?style=for-the-badge&logo=github)](https://github.com/Anandsavarn)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/)

<br/>

---

*⭐ Star this repo if you found the LSTM implementation useful!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:003366,50:001833,100:0a0a1a&height=100&section=footer" width="100%"/>

</div>
