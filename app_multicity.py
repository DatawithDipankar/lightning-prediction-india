import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Page Config
st.set_page_config(
    page_title="India Lightning Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS — Dark industrial weather aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg:         #070b12;
    --surface:    #0d1420;
    --card:       #111c2e;
    --border:     #1a2840;
    --accent:     #f0b429;
    --blue:       #2d9cdb;
    --green:      #27ae60;
    --orange:     #e67e22;
    --red:        #e74c3c;
    --text:       #dce8f5;
    --muted:      #5a7a9a;
    --mumbai:     #f0b429;
    --goa:        #27ae60;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background-color: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero */
.hero {
    padding: 2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-left { flex: 1; }
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 0.04em;
    line-height: 0.95;
    color: #fff;
    margin: 0;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.5rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 500;
}
.hero-icon {
    font-size: 5rem;
    filter: drop-shadow(0 0 30px #f0b42966);
    animation: flicker 3s ease-in-out infinite;
}
@keyframes flicker {
    0%,100% { filter: drop-shadow(0 0 30px #f0b42966); opacity:1; }
    45%      { filter: drop-shadow(0 0 60px #f0b429cc); opacity:0.9; }
    50%      { filter: drop-shadow(0 0 10px #f0b42933); opacity:0.7; }
    55%      { filter: drop-shadow(0 0 60px #f0b429cc); opacity:0.95; }
}

/* ── City selector tabs */
.city-tabs {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
}
.city-tab {
    padding: 0.5rem 1.4rem;
    border-radius: 6px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.08em;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--muted);
    transition: all 0.2s;
}

/* ── Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.card-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.9rem;
}

/* ── Metric strip */
.metric-strip {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.2rem;
}
.metric-pill {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 0.5rem;
    text-align: center;
}
.metric-pill .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-pill .lbl {
    font-size: 0.62rem;
    color: var(--muted);
    margin-top: 0.2rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Risk display */
.risk-display {
    text-align: center;
    padding: 0.5rem 0;
}
.risk-pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    line-height: 1;
    letter-spacing: 0.05em;
}
.risk-level {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 0.12em;
    margin-top: -0.2rem;
}
.risk-desc {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* ── Derived metrics */
.derived-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.8rem;
}
.derived-item {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem;
    text-align: center;
}
.derived-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: var(--blue);
}
.derived-lbl {
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.15rem;
}

/* ── Alert box */
.alert {
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.83rem;
    line-height: 1.6;
    margin-top: 0.8rem;
    border-left: 3px solid;
}
.alert-low    { background:#0a1f12; border-color:var(--green);  color:#7ecda0; }
.alert-mod    { background:#1f1700; border-color:var(--accent); color:#f0c96a; }
.alert-high   { background:#1f1000; border-color:var(--orange); color:#f0a060; }
.alert-vhigh  { background:#1f0808; border-color:var(--red);    color:#f08080; }

/* ── City badge */
.city-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Comparison table */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.compare-table th {
    text-align: left;
    padding: 0.5rem 0.8rem;
    color: var(--muted);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
}
.compare-table td {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid #111c2e;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.compare-table tr:last-child td { border-bottom: none; }

.divider { height:1px; background:var(--border); margin:1rem 0; }
.footer {
    text-align:center; padding:1.5rem 0 0.5rem;
    color:var(--muted); font-size:0.75rem;
}
.footer a { color:var(--blue); text-decoration:none; }
</style>
""", unsafe_allow_html=True)


# ── City config
CITIES = {
    "Mumbai": {
        "emoji": "🏙️",
        "color": "#f0b429",
        "model_file":    "mumbai_model.pkl",
        "features_file": "mumbai_features.pkl",
        "data_file":     "mumbai_data.pkl",
        "algo":          "Random Forest",
        "roc":           "0.9818",
        "pr":            "0.9942",
        "acc":           "92%",
        "recall":        "93%",
        "desc":          "Coastal megacity · Arabian Sea influence · Pre-monsoon dry thunderstorms",
        "lat": 19.076, "lon": 72.877
    },
    "Goa": {
        "emoji": "🌴",
        "color": "#27ae60",
        "model_file":    "goa_model.pkl",
        "features_file": "goa_features.pkl",
        "data_file":     "goa_data.pkl",
        "algo":          "XGBoost",
        "roc":           "0.9853",
        "pr":            "0.9952",
        "acc":           "96%",
        "recall":        "98%",
        "desc":          "Western Ghats proximity · Early monsoon onset · High convective activity",
        "lat": 15.491, "lon": 73.828
    }
}

FEATURE_COLS = [
    'temp_mean','temp_max','temp_min','temp_range',
    'humidity','humidity_max','dew_point',
    'rainfall','windspeed_max','solar_radiation',
    'cape_proxy','heat_index',
    'temp_mean_3day','humidity_3day','rainfall_3day',
    'temp_mean_7day','humidity_7day','rainfall_7day',
    'windspeed_max_3day','windspeed_max_7day',
    'rainfall_3day_sum','rainfall_7day_sum',
    'temp_change','humidity_change','rainfall_change','wind_change',
    'humidity_x_wind','solar_anomaly',
    'rainfall_spike','dry_thunderstorm',
    'intense_hourly_rain','humidity_max_anom'
]

def load_city_assets(city):
    # No caching — load fresh per city to avoid cross-city model bleed
    cfg    = CITIES[city]
    model  = joblib.load(cfg['model_file'])
    feats  = joblib.load(cfg['features_file'])
    data   = joblib.load(cfg['data_file'])
    try:
        scaler = joblib.load(cfg['model_file'].replace('_model', '_scaler'))
    except Exception:
        scaler = None
    return model, feats, data, scaler

def compute_features(inputs, ref_data):
    t, h, r = inputs['temp_mean'], inputs['humidity'], inputs['rainfall']
    t7, h7  = inputs['temp_7day'], inputs['humidity_7day']
    w       = inputs['windspeed_max']
    sr      = inputs['solar_radiation']
    hmax    = inputs['humidity_max']
    r3      = inputs['rainfall_3day_sum']
    r7      = inputs['rainfall_7day_sum']

    dew   = inputs.get('dew_point', t - ((100 - h) / 5))
    if t >= 27 and h >= 40:
        hi = (-8.78469475556 + 1.61139411*t + 2.33854883889*h
              - 0.14611605*t*h - 0.012308094*t**2
              - 0.0164248277778*h**2 + 0.002211732*t**2*h
              + 0.00072546*t*h**2 - 0.000003582*t**2*h**2)
    else:
        hi = t
    cape  = t * h / 100
    t3    = (t + t7*2) / 3
    h3    = (h + h7*2) / 3
    r3d   = r3 / 3 if r3 > 0 else r
    q90   = ref_data['rainfall'].quantile(0.90)
    q85   = ref_data['hourly_rain_max'].quantile(0.85) if 'hourly_rain_max' in ref_data.columns else 5.0
    hm7   = ref_data['humidity_max'].rolling(7, min_periods=1).mean().iloc[-1]

    return pd.DataFrame([{
        'temp_mean':          t,
        'temp_max':           inputs['temp_max'],
        'temp_min':           inputs['temp_min'],
        'temp_range':         inputs['temp_max'] - inputs['temp_min'],
        'humidity':           h,
        'humidity_max':       hmax,
        'dew_point':          dew,
        'rainfall':           r,
        'windspeed_max':      w,
        'solar_radiation':    sr,
        'cape_proxy':         cape,
        'heat_index':         hi,
        'temp_mean_3day':     t3,
        'humidity_3day':      h3,
        'rainfall_3day':      r3d,
        'temp_mean_7day':     t7,
        'humidity_7day':      h7,
        'rainfall_7day':      r7/7 if r7 > 0 else r,
        'windspeed_max_3day': w,
        'windspeed_max_7day': w,
        'rainfall_3day_sum':  r3,
        'rainfall_7day_sum':  r7,
        'temp_change':        t - t7,
        'humidity_change':    h - h7,
        'rainfall_change':    r - r3d,
        'wind_change':        0.0,
        'humidity_x_wind':    h * w / 100,
        'solar_anomaly':      sr - ref_data['solar_radiation'].rolling(7, min_periods=1).mean().iloc[-1],
        'rainfall_spike':     int(r > q90),
        'dry_thunderstorm':   int(r < 0.1 and h > 70),
        'intense_hourly_rain':int(r > q85),
        'humidity_max_anom':  hmax - hm7,
    }])[FEATURE_COLS]

def risk_meta(pct):
    if pct < 40:   return "LOW",       "#27ae60", "alert-low",   "⬇ Atmospheric conditions stable. Low convective activity expected."
    elif pct < 62: return "MODERATE",  "#f0b429", "alert-mod",   "⚠ Some instability present. Possible isolated thunderstorm activity."
    elif pct < 80: return "HIGH",      "#e67e22", "alert-high",  "⚡ Significant convective potential. Lightning activity likely."
    else:          return "VERY HIGH", "#e74c3c", "alert-vhigh", "🔴 Severe storm conditions. High probability of lightning strikes."

def gauge_fig(prob, color, bg='#0d1420'):
    fig, ax = plt.subplots(figsize=(4.5, 2.6), facecolor=bg)
    ax.set_facecolor(bg)
    segs = [(0,.30,'#1a3a22'),(0.30,.55,'#3a3010'),(0.55,.75,'#3a1f08'),(0.75,1.0,'#3a0808')]
    for s,e,c in segs:
        t = np.linspace(np.pi - s*np.pi, np.pi - e*np.pi, 80)
        xo,yo = np.cos(t)*1.0, np.sin(t)*1.0
        xi,yi = np.cos(t)*0.60, np.sin(t)*0.60
        ax.fill(np.concatenate([xo,xi[::-1]]), np.concatenate([yo,yi[::-1]]),
                color=c, alpha=0.9, zorder=2)
    # Active fill
    t_active = np.linspace(np.pi, np.pi - prob*np.pi, 120)
    xo,yo = np.cos(t_active)*1.0, np.sin(t_active)*1.0
    xi,yi = np.cos(t_active)*0.60, np.sin(t_active)*0.60
    ax.fill(np.concatenate([xo,xi[::-1]]), np.concatenate([yo,yi[::-1]]),
            color=color, alpha=0.7, zorder=3)
    # Needle
    na = np.pi - prob*np.pi
    ax.annotate('', xy=(np.cos(na)*0.85, np.sin(na)*0.85), xytext=(0,0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2.2, mutation_scale=14))
    ax.plot(0,0,'o',color='white',markersize=6,zorder=6)
    # Labels
    for lbl, ang in [("0%",np.pi),("50%",np.pi/2),("100%",0)]:
        ax.text(np.cos(ang)*1.18, np.sin(ang)*1.18, lbl,
                ha='center',va='center',fontsize=7,color='#5a7a9a')
    ax.set_xlim(-1.35,1.35); ax.set_ylim(-0.55,1.25); ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def shap_fig(model, input_df, feature_cols, city_color, ref_data, bg='#0d1420'):
    # Use built-in feature importances scaled by input deviation from training mean
    # This gives stable, meaningful direction for each feature
    background = ref_data[feature_cols].dropna()
    train_mean = background.mean().values
    train_std  = background.std().values + 1e-8

    input_vals = input_df.iloc[0].values

    # Get model feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    # Deviation of input from training mean (in std units)
    deviation = (input_vals - train_mean) / train_std

    # Signed contribution = importance * direction of deviation
    sv = importances * deviation

    idx    = np.argsort(np.abs(sv))[::-1][:12]
    names  = [feature_cols[i].replace('_',' ') for i in idx]
    vals   = [sv[i] for i in idx]
    colors = [city_color if v > 0 else '#2d9cdb' for v in vals]

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=bg)
    ax.set_facecolor(bg)
    ax.barh(range(len(names)), vals[::-1], color=colors[::-1], height=0.55, edgecolor='none')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], color='#dce8f5', fontsize=8.5)
    ax.axvline(0, color='#1a2840', lw=1)
    ax.set_xlabel('SHAP value — impact on prediction', color='#5a7a9a', fontsize=8)
    ax.tick_params(colors='#5a7a9a', labelsize=7.5)
    for sp in ['top','right']:      ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']:    ax.spines[sp].set_color('#1a2840')
    p1 = mpatches.Patch(color=city_color, label='↑ Increases probability')
    p2 = mpatches.Patch(color='#2d9cdb',  label='↓ Decreases probability')
    ax.legend(handles=[p1,p2], framealpha=0, labelcolor='#5a7a9a', fontsize=8, loc='lower right')
    plt.tight_layout(pad=0.4)
    return fig


# ════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-left">
    <p class="hero-title">INDIA<br><span>LIGHTNING</span><br>FORECAST</p>
    <p class="hero-sub">Random Forest · XGBoost · NASA LIS + Open-Meteo · 2015–2024</p>
  </div>
  <div class="hero-icon">⚡</div>
</div>
""", unsafe_allow_html=True)

# City selector
city = st.radio("Select City", list(CITIES.keys()),
                horizontal=True, label_visibility="collapsed")
cfg  = CITIES[city]

try:
    model, feat_cols, ref_data, scaler = load_city_assets(city)
    assets_loaded = True
except Exception as e:
    st.error(f"Could not load {city} model files: {e}")
    assets_loaded = False

if assets_loaded:

    # City badge + metrics
    st.markdown(f"""
    <div class="city-badge" style="color:{cfg['color']};border-color:{cfg['color']}33">
        {cfg['emoji']} {city} &nbsp;·&nbsp; {cfg['algo']}
    </div>
    <p style="color:#5a7a9a;font-size:0.78rem;margin:-0.5rem 0 1rem">{cfg['desc']}</p>
    <div class="metric-strip">
        <div class="metric-pill"><div class="val">{cfg['roc']}</div><div class="lbl">ROC-AUC</div></div>
        <div class="metric-pill"><div class="val">{cfg['pr']}</div><div class="lbl">PR-AUC</div></div>
        <div class="metric-pill"><div class="val">{cfg['acc']}</div><div class="lbl">Accuracy</div></div>
        <div class="metric-pill"><div class="val">{cfg['recall']}</div><div class="lbl">⚡ Recall</div></div>
        <div class="metric-pill"><div class="val">3,653</div><div class="lbl">Train Days</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Layout
    col_in, col_pred = st.columns([1, 1.1], gap="large")

    # ── INPUTS
    with col_in:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">📡 Today\'s Conditions</div>', unsafe_allow_html=True)

        # key=city ensures sliders fully reset when switching cities
        temp_mean = st.slider("Mean Temperature (°C)",    17.0, 38.0, 28.0, 0.1, key=f"temp_mean_{city}")
        temp_max  = st.slider("Max Temperature (°C)",     temp_mean, 42.0, float(min(temp_mean+3, 42.0)), 0.1, key=f"temp_max_{city}")
        temp_min  = st.slider("Min Temperature (°C)",     10.0, float(temp_mean), float(max(temp_mean-5, 10.0)), 0.1, key=f"temp_min_{city}")
        humidity  = st.slider("Relative Humidity (%)",    10.0, 100.0, 75.0, 0.5, key=f"humidity_{city}")
        rainfall  = st.slider("Rainfall (mm)",            0.0,  150.0,  5.0, 0.5, key=f"rainfall_{city}")
        windspeed = st.slider("Wind Speed (km/h)",        0.0,   80.0, 15.0, 0.5, key=f"windspeed_{city}")
        solar     = st.slider("Solar Radiation (MJ/m²)",  0.0,   30.0, 14.0, 0.1, key=f"solar_{city}")

        st.markdown('<div class="card-label" style="margin-top:1rem">📅 7-Day Context</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            temp_7d  = st.number_input("Temp 7d avg (°C)",    17.0, 38.0, float(round(temp_mean-0.5, 1)), 0.1, key=f"temp7d_{city}")
            hum_7d   = st.number_input("Humidity 7d avg (%)", 10.0,100.0, float(round(humidity-3, 1)),    0.5, key=f"hum7d_{city}")
            hum_max  = st.number_input("Humidity max (%)",    float(humidity),100.0, float(min(humidity+10,100)), 0.5, key=f"hummax_{city}")
        with c2:
            rain_3d  = st.number_input("Rainfall 3d sum (mm)",  0.0,300.0, float(round(rainfall*2,1)),  0.5, key=f"rain3d_{city}")
            rain_7d  = st.number_input("Rainfall 7d sum (mm)",  0.0,600.0, float(round(rainfall*5,1)),  0.5, key=f"rain7d_{city}")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── PREDICTION
    with col_pred:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">⚡ Lightning Forecast</div>', unsafe_allow_html=True)

        inputs = {
            'temp_mean': temp_mean, 'temp_max': temp_max, 'temp_min': temp_min,
            'humidity': humidity, 'humidity_max': hum_max,
            'rainfall': rainfall, 'windspeed_max': windspeed,
            'solar_radiation': solar,
            'temp_7day': temp_7d, 'humidity_7day': hum_7d,
            'rainfall_3day_sum': rain_3d, 'rainfall_7day_sum': rain_7d,
        }

        input_df  = compute_features(inputs, ref_data)
        # Both models are now tree-based — no scaling needed
        prob      = model.predict_proba(input_df)[0][1]
        pct       = int(prob * 100)
        risk, risk_color, alert_cls, alert_msg = risk_meta(pct)

        # Gauge
        fig_g = gauge_fig(prob, risk_color)
        st.pyplot(fig_g, use_container_width=True)
        plt.close()

        # Risk label
        st.markdown(f"""
        <div class="risk-display">
            <div class="risk-pct" style="color:{risk_color}">{pct}%</div>
            <div class="risk-level" style="color:{risk_color}">{risk} RISK</div>
            <div class="risk-desc">Lightning probability for {city}</div>
        </div>
        """, unsafe_allow_html=True)

        # Derived values
        dew_pt   = temp_mean - ((100 - humidity) / 5)
        cape_val = temp_mean * humidity / 100
        if temp_mean >= 27 and humidity >= 40:
            hi_val = (-8.78469475556
                      + 1.61139411*temp_mean
                      + 2.33854883889*humidity
                      - 0.14611605*temp_mean*humidity
                      - 0.012308094*temp_mean**2
                      - 0.0164248277778*humidity**2
                      + 0.002211732*temp_mean**2*humidity
                      + 0.00072546*temp_mean*humidity**2
                      - 0.000003582*temp_mean**2*humidity**2)
        else:
            hi_val = None

        st.markdown(f"""
        <div class="derived-row">
            <div class="derived-item">
                <div class="derived-val">{dew_pt:.1f}°C</div>
                <div class="derived-lbl">Dew Point</div>
            </div>
            <div class="derived-item">
                <div class="derived-val">{'N/A' if hi_val is None else f'{hi_val:.1f}°C'}</div>
                <div class="derived-lbl">Heat Index</div>
            </div>
            <div class="derived-item">
                <div class="derived-val">{cape_val:.1f}</div>
                <div class="derived-lbl">CAPE Proxy</div>
            </div>
            <div class="derived-item">
                <div class="derived-val">{temp_max-temp_min:.1f}°C</div>
                <div class="derived-lbl">Temp Range</div>
            </div>
        </div>
        <div class="alert {alert_cls}">{alert_msg}</div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── SHAP
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">🔍 Feature Contributions — Why This Prediction?</div>', unsafe_allow_html=True)

    fig_s = shap_fig(model, input_df, FEATURE_COLS, cfg['color'], ref_data)
    st.pyplot(fig_s, use_container_width=True)
    plt.close()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── City Comparison (when both models available)
    try:
        other_city  = "Goa" if city == "Mumbai" else "Mumbai"
        other_model, other_feats, other_data, other_scaler = load_city_assets(other_city)
        other_df    = compute_features(inputs, other_data)
        other_prob  = other_model.predict_proba(other_df)[0][1]
        other_pct   = int(other_prob * 100)
        _, oc, _, _ = risk_meta(other_pct)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">🗺 City Comparison — Same Conditions</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <table class="compare-table">
            <thead>
                <tr>
                    <th>City</th><th>Model</th>
                    <th>Lightning Prob</th><th>Risk Level</th>
                    <th>ROC-AUC</th><th>Accuracy</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="color:{cfg['color']};font-weight:600">{cfg['emoji']} {city}</td>
                    <td>{cfg['algo']}</td>
                    <td style="color:{risk_color};font-weight:700">{pct}%</td>
                    <td style="color:{risk_color}">{risk}</td>
                    <td>{cfg['roc']}</td>
                    <td>{cfg['acc']}</td>
                </tr>
                <tr>
                    <td style="color:{CITIES[other_city]['color']};font-weight:600">{CITIES[other_city]['emoji']} {other_city}</td>
                    <td>{CITIES[other_city]['algo']}</td>
                    <td style="color:{oc};font-weight:700">{other_pct}%</td>
                    <td style="color:{oc}">{risk_meta(other_pct)[0]}</td>
                    <td>{CITIES[other_city]['roc']}</td>
                    <td>{CITIES[other_city]['acc']}</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception:
        pass

    # ── About
    with st.expander("📖 About This Project"):
        st.markdown(f"""
        **Data Sources**
        - Weather features: Open-Meteo Historical Archive API (2015–2024)
        - Lightning labels: NASA LIS VHRMC Monthly Climatology (1998–2013, satellite-derived)
        - Cities: Mumbai (19.07°N 72.88°E) · Goa (15.49°N 73.83°E)

        **Why Two Different Models?**
        Random Forest wins for Mumbai — Logistic Regression failed due to multicollinearity 
        among correlated humidity features, causing inverted coefficients. Random Forest 
        handles correlated features naturally by splitting on thresholds. Goa uses XGBoost 
        for its more complex non-linear lightning pattern driven by Western Ghats orographic lifting.

        **Model Performance**
        | City | Model | ROC-AUC | PR-AUC | Accuracy | ⚡ Recall |
        |---|---|---|---|---|---|
        | Mumbai | Random Forest       | 0.9818 | 0.9942 | 94% | 95% |
        | Goa | XGBoost | 0.9853 | 0.9952 | 96% | 98% |

        **Key Finding:** 7-day rolling humidity average dominates predictions in both cities.
        Mumbai shows significant dry thunderstorm activity (high humidity, zero rainfall)
        in pre-monsoon months — a pattern not seen as strongly in Goa.

        **GitHub:** [DatawithDipankar](https://github.com/DatawithDipankar)
        """)

# Footer
st.markdown("""
<div class="footer">
    Built by <b>Dipankar</b> &nbsp;·&nbsp; 
    Data: Open-Meteo + NASA LIS GHRC &nbsp;·&nbsp;
    <a href="https://github.com/DatawithDipankar">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)