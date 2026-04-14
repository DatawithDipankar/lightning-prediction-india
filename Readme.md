# ⚡ India Lightning Prediction System

Multi-city lightning probability forecasting for Mumbai and Goa 
using machine learning on real meteorological data.

## Live Demo
[Streamlit App Link here]

## Problem Statement
Given temperature, humidity, and rainfall data, predict the 
probability of lightning occurrence for a given day.

## Data Sources
| Source | What | Period |
|---|---|---|
| NASA LIS VHRMC (GHRC DAAC) | Lightning labels (satellite flash rate) | 1998–2013 |
| Open-Meteo Historical API | Weather features (hourly → daily) | 2015–2024 |

## Approach
- Merged satellite lightning climatology with modern weather data
- Engineered 32 features: dew point, CAPE proxy, 7-day rolling averages, 
  day-over-day gradients, dry thunderstorm flags
- Time-based train/test split (2015–2022 train, 2023–2024 test)
- Detected and resolved data leakage from month-encoded features
- Identified multicollinearity failure in Logistic Regression for Mumbai 
  via coefficient analysis switched to Random Forest

## Model Performance
| City | Model | ROC-AUC | PR-AUC | Accuracy | ⚡ Recall |
|---|---|---|---|---|---|
| Mumbai | Random Forest | 0.9818 | 0.9942 | 94% | 95% |
| Goa | XGBoost | 0.9853 | 0.9952 | 96% | 98% |

## Key Findings
- 7-day rolling humidity average was the strongest predictor in both cities
- Raw daily rainfall had near-zero importance Mumbai has significant 
  dry thunderstorm activity in pre-monsoon months where lightning occurs 
  without rainfall
- Goa's lightning pattern is more complex than Mumbai's due to Western 
  Ghats orographic lifting, requiring XGBoost over simpler models

## Known Limitation
Labels are monthly climatology averages the model overestimates risk 
in meteorologically ambiguous transition months (March–April). 
Production upgrade: replace with daily WWLLN strike records.

## How to Run

    >pip install streamlit xgboost scikit-learn joblib shap matplotlib pandas numpy

    >streamlit run app_multicity.py
