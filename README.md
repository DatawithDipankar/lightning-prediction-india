# ⚡ India Lightning Prediction System

Multi-city lightning probability forecasting for Mumbai and Goa
using machine learning on real meteorological data.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)

## 🌐 Live Demo
[streamlit-link](https://lightning-prediction-india.streamlit.app/)

## 📌 Problem Statement
Given daily weather conditions (temperature, humidity, rainfall,
wind speed, solar radiation), predict the probability of lightning
occurrence for Mumbai and Goa.

## 📦 Data Sources
| Source | Purpose | Period |
|---|---|---|
| NASA LIS VHRMC (GHRC DAAC) | Lightning labels — satellite flash rate | 1998–2013 |
| Open-Meteo Historical API | Weather features — hourly aggregated to daily | 2015–2024 |

## 🔧 Approach
- Merged satellite lightning climatology with modern weather API data
- Engineered 32 meteorological features: dew point, CAPE proxy,
  7-day rolling averages, day-over-day gradients, dry thunderstorm flags
- Time-based train/test split (2015–2022 train | 2023–2024 test)
- Detected and fixed data leakage from month-encoded features
- Diagnosed Logistic Regression failure via coefficient inversion
  analysis switched Mumbai to Random Forest

## 📊 Model Performance
| City | Model | ROC-AUC | PR-AUC | Accuracy | ⚡ Recall |
|---|---|---|---|---|---|
| Mumbai | Random Forest | 0.9818 | 0.9942 | 94% | 95% |
| Goa | XGBoost | 0.9853 | 0.9952 | 96% | 98% |

## 💡 Key Findings
- **7-day rolling humidity** was the strongest predictor in both cities
- **Raw daily rainfall had near-zero importance** Mumbai experiences
  significant dry thunderstorms in pre-monsoon months where lightning
  occurs without rainfall, driven purely by convective instability
- **Goa's pattern is more complex** than Mumbai due to Western Ghats
  orographic lifting requiring XGBoost over simpler models
- **Logistic Regression failed** for Mumbai due to multicollinearity
  among correlated humidity features causing coefficient inversion

## ⚠️ Known Limitation
Labels are derived from NASA LIS monthly climatology averages.
The model overestimates risk during meteorologically ambiguous
transition months (March–April) where weather conditions overlap
with active lightning months. Production upgrade: replace with
daily WWLLN strike records for precise threshold learning.

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app_multicity.py
```

## 📁 Project Structure
```
├── app_multicity.py          # Streamlit app
├── mumbai_model.pkl          # Random Forest model
├── goa_model.pkl             # XGBoost model
├── *_data.pkl                # Processed feature data
├── *_features.pkl            # Feature column lists
├── *_scaler.pkl              # Scalers (placeholder for Mumbai)
└── requirements.txt
```

## 👤 Author
**Dipankar** — [GitHub](https://github.com/DatawithDipankar)
