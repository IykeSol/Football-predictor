import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
import re
import toml
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Football Predictor Pro",
    page_icon="⚽",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0e1117;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #444;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0e1117;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .ai-text {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# --- SECRETS & SETUP ---
try:
    secrets = toml.load(".streamlit/secrets.toml")
    GEMINI_API_KEY = secrets['google']['gemini_api_key']
    genai.configure(api_key=GEMINI_API_KEY)
except:
    st.error("❌ Error: Secrets not found. Please create .streamlit/secrets.toml")
    st.stop()

# --- 1. LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    try: return joblib.load("real_data_football_model.pkl")
    except: return None

@st.cache_data(ttl=3600)
def load_data():
    if os.path.exists("real_live_data_filled.csv"):
        df = pd.read_csv("real_live_data_filled.csv")
    elif os.path.exists("real_live_data_filled.xlsx"):
        df = pd.read_excel("real_live_data_filled.xlsx", engine='openpyxl')
    else: return pd.DataFrame()
    
    cols = [c for c in df.columns if not re.search(r'\.\d+$', c)]
    df = df[cols]
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')

model = load_model()
df = load_data()

# --- 2. AI ANALYSIS (UPDATED TO INCLUDE H2H STATS) ---
def get_gemini_analysis(home_team, away_team, prediction_text, confidence, home_recent_str, away_recent_str, h2h_str, h2h_avg_goals, h2h_home_win):
    model_name = "models/gemini-2.0-flash" 
    
    prompt = f"""
    You are a Senior Football Betting Analyst.
    **DATE:** December 2025.
    
    Match: **{home_team} vs {away_team}**.
    
    **DATA PROVIDED:**
    - Model Verdict: {prediction_text} ({confidence}% Confidence)
    - {home_team} Form:
    {home_recent_str}
    - {away_team} Form:
    {away_recent_str}
    
    **HEAD-TO-HEAD STATS (CRITICAL):**
    - Match History:
    {h2h_str}
    - **Average Goals per H2H Match:** {h2h_avg_goals}
    - **{home_team} H2H Win Rate:** {h2h_home_win}
    
    **INSTRUCTIONS:**
    1. **Analyze the Winner:** Do not blindly agree with the model. Look at xG vs Goals.
    2. **Predict Goals:** Use the "Average Goals per H2H Match" ({h2h_avg_goals}) to predict Over/Under.
    
    **FORMAT YOUR RESPONSE EXACTLY LIKE THIS:**
    
    ### 1. The Analyst's Verdict
    [State your independent winner prediction. Explain if you agree with the model.]

    ### 2. Goal Market Predictions
    *   **Main Pick:** [e.g. Over 2.5 Goals / Under 2.5 Goals]
    *   **Alternative:** [e.g. Both Teams to Score (YES/NO)]
    *   **Reasoning:** [Use the H2H Avg Goals ({h2h_avg_goals}) and recent form to justify this.]

    ### 3. Key Tactical Insights
    1. **[Tactical Title]:** [Detailed paragraph analyzing the match-up.]
    2. **[Form & xG Title]:** [Detailed paragraph analyzing if teams are over/underperforming their xG.]
    3. **[Historical Context]:** [Discuss the H2H. Mention explicitly that they average {h2h_avg_goals} goals per game and the win rate.]

    ### 4. Risk Factors
    *   **[Risk Title]:** [What could go wrong? e.g. "The low H2H goal average suggests a tight game..."]
    """
    
    try:
        model_ai = genai.GenerativeModel(model_name)
        response = model_ai.generate_content(prompt)
        return response.text
    except:
        return "⚠️ AI Analysis unavailable."

# --- 3. PREDICTION ENGINE ---
def make_prediction(home_team, away_team):
    home_matches = df[((df['home_team'] == home_team) | (df['away_team'] == home_team))].tail(5)
    away_matches = df[((df['home_team'] == away_team) | (df['away_team'] == away_team))].tail(5)
    h2h_matches = df[((df['home_team'] == home_team) & (df['away_team'] == away_team) | (df['home_team'] == away_team) & (df['away_team'] == home_team))].tail(10)
    
    if len(home_matches) == 0 or len(away_matches) == 0: return None, None, "Insufficient Data", None, None, None

    feats = {}
    
    def get_form(matches, team):
        gf = [m['home_score'] if m['home_team'] == team else m['away_score'] for _, m in matches.iterrows()]
        ga = [m['away_score'] if m['home_team'] == team else m['home_score'] for _, m in matches.iterrows()]
        xgf = [m['home_xg'] if m['home_team'] == team else m['away_xg'] for _, m in matches.iterrows()]
        xga = [m['away_xg'] if m['home_team'] == team else m['home_xg'] for _, m in matches.iterrows()]
        pts = sum([3 if f > a else 1 if f == a else 0 for f, a in zip(gf, ga)])
        return np.mean(gf), np.mean(ga), np.mean(xgf), np.mean(xga), pts

    feats['home_form_goals_for_5'], feats['home_form_goals_against_5'], feats['home_form_xg_for_5'], feats['home_form_xg_against_5'], feats['home_form_points_5'] = get_form(home_matches, home_team)
    feats['away_form_goals_for_5'], feats['away_form_goals_against_5'], feats['away_form_xg_for_5'], feats['away_form_xg_against_5'], feats['away_form_points_5'] = get_form(away_matches, away_team)
    
    if len(h2h_matches) > 0:
        hw = sum([1 for _, m in h2h_matches.iterrows() if (m['home_team'] == home_team and m['home_score'] > m['away_score']) or (m['away_team'] == home_team and m['away_score'] > m['home_score'])])
        aw = sum([1 for _, m in h2h_matches.iterrows() if (m['home_team'] == away_team and m['home_score'] > m['away_score']) or (m['away_team'] == away_team and m['away_score'] > m['home_score'])])
        goals = h2h_matches['home_score'].sum() + h2h_matches['away_score'].sum()
        feats['h2h_matches'] = len(h2h_matches)
        feats['h2h_home_win_pct'] = hw / len(h2h_matches)
        feats['h2h_away_win_pct'] = aw / len(h2h_matches)
        feats['h2h_avg_goals'] = goals / len(h2h_matches)
        feats['h2h_avg_home_goals'] = goals / 2 / len(h2h_matches) 
        feats['h2h_avg_away_goals'] = goals / 2 / len(h2h_matches) 
    else:
        feats.update({'h2h_matches':0, 'h2h_home_win_pct':0, 'h2h_away_win_pct':0, 'h2h_avg_goals':0, 'h2h_avg_home_goals':0, 'h2h_avg_away_goals':0})

    feats['home_xg'] = feats['home_form_xg_for_5']
    feats['away_xg'] = feats['away_form_xg_for_5']

    try:
        input_df = pd.DataFrame([feats])[model.feature_names_in_]
        return model.predict(input_df)[0], model.predict_proba(input_df)[0], feats, home_matches, away_matches, h2h_matches
    except: return None, None, "Model Error", None, None, None

# --- 4. UI IMPLEMENTATION ---
st.markdown("### ⚽ Advanced Football Predictor Pro")

if df.empty or model is None:
    st.error("System not ready.")
    st.stop()

# Selectors
all_teams = sorted(pd.concat([df['home_team'], df['away_team']]).unique())
c1, c2 = st.columns(2)
with c1: home_team = st.selectbox("Home Team", all_teams)
with c2: away_team = st.selectbox("Away Team", all_teams, index=1)

if st.button("🚀 Generate Prediction", type="primary"):
    if home_team == away_team: st.error("Select different teams.")
    else:
        with st.spinner("Analyzing Match..."):
            pred, prob, feats, h_match, a_match, h2h = make_prediction(home_team, away_team)
            
            if isinstance(feats, str): st.error(feats)
            else:
                # Strings
                def m_str(matches, team):
                    s = ""
                    for _, m in matches.iterrows():
                        res = "W" if (m['home_team']==team and m['home_score']>m['away_score']) or (m['away_team']==team and m['away_score']>m['home_score']) else "L" if (m['home_team']==team and m['home_score']<m['away_score']) or (m['away_team']==team and m['away_score']<m['home_score']) else "D"
                        opp = m['away_team'] if m['home_team']==team else m['home_team']
                        s += f"- {m['date'].strftime('%Y-%m-%d')} vs {opp} ({res}): {m['home_score']}-{m['away_score']} (xG: {m.get('home_xg',0)}-{m.get('away_xg',0)})\n"
                    return s

                h_s = m_str(h_match, home_team)
                a_s = m_str(a_match, away_team)
                h2h_s = "\n".join([f"- {m['date'].strftime('%Y-%m-%d')}: {m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']}" for _, m in h2h.tail(5).iterrows()])

                # --- HEADER CARD ---
                home_prob = prob[1] * 100
                away_prob = prob[0] * 100
                
                if pred == 1:
                    verdict = f"{home_team} to win"
                    icon = "🏠"
                    title = "HOME WIN PREDICTION"
                    main_conf = home_prob
                    opp_conf = away_prob
                    opp_label = "Draw / Away Win"
                else:
                    verdict = f"{away_team} +0.5 (Win/Draw)"
                    icon = "🚌"
                    title = "AWAY WIN / DRAW PREDICTION"
                    main_conf = away_prob
                    opp_conf = home_prob
                    opp_label = f"{home_team} Win"
                
                st.markdown(f"""
<div style="background-color: #4353cc; color: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px;">
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
<span style="font-size: 2.5rem;">{icon}</span>
<h2 style="margin: 0; color: white; font-size: 1.5rem; text-transform: uppercase; font-weight: 600;">{title}</h2>
</div>
<h1 style="margin: 10px 0; font-size: 2.5rem; font-weight: 700;">{verdict}</h1>
<div style="margin-top: 20px;">
<div style="display:flex; justify-content:space-between; margin-bottom:5px; font-size: 1.1rem;">
<span style="font-weight:bold;">Confidence</span>
<span style="font-weight:bold;">{main_conf:.1f}%</span>
</div>
<div style="width:100%; background: rgba(255,255,255,0.3); height:12px; border-radius:6px; overflow:hidden;">
<div style="width:{main_conf}%; background: #ffffff; height:100%;"></div>
</div>
<div style="display:flex; justify-content:space-between; margin-top:8px; font-size:0.9rem; opacity:0.8;">
<span>vs {opp_label}</span>
<span>{opp_conf:.1f}%</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

                # --- TABS ---
                st.markdown('<div class="sub-header">📊 Match Analysis</div>', unsafe_allow_html=True)
                t1, t2, t3 = st.tabs(["Form Analysis", "H2H History", "🤖 AI Insights"])
                
                with t1:
                    c1, c2 = st.columns(2)
                    with c1: 
                        st.write(f"**{home_team} Last 5**")
                        st.text(h_s)
                    with c2: 
                        st.write(f"**{away_team} Last 5**")
                        st.text(a_s)
                
                with t2:
                    if len(h2h) > 0:
                        st.write(f"**Last {len(h2h)} Head-to-Head**")
                        st.text(h2h_s)
                        hc1, hc2, hc3 = st.columns(3)
                        hc1.metric("H2H Home Wins", f"{feats['h2h_home_win_pct']*100:.0f}%")
                        hc2.metric("H2H Away Wins", f"{feats['h2h_away_win_pct']*100:.0f}%")
                        hc3.metric("Avg H2H Goals", f"{feats['h2h_avg_goals']:.2f}")
                    else:
                        st.info("No recent head-to-head matches.")

                with t3:
                    with st.spinner("Asking Gemini..."):
                        # HERE IS THE FIX: I am passing the stats explicitly to the AI
                        h2h_avg = f"{feats['h2h_avg_goals']:.2f}"
                        h2h_home_win_rt = f"{feats['h2h_home_win_pct']*100:.0f}%"
                        
                        ai = get_gemini_analysis(home_team, away_team, title, main_conf, h_s, a_s, h2h_s, h2h_avg, h2h_home_win_rt)
                        st.markdown(f'<div class="ai-text">{ai}</div>', unsafe_allow_html=True)

                # --- ADDITIONAL INSIGHTS ---
                st.markdown('<div class="sub-header">🔍 Additional Insights</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Key Statistics:**")
                    st.markdown(f"Home Form (Last 5)<br><span class='metric-value'>{feats['home_form_points_5']} pts</span>", unsafe_allow_html=True)
                    st.markdown(f"Away Form (Last 5)<br><span class='metric-value'>{feats['away_form_points_5']} pts</span>", unsafe_allow_html=True)
                    st.markdown(f"Home xG Advantage<br><span class='metric-value'>{(feats['home_xg'] - feats['away_xg']):.2f}</span>", unsafe_allow_html=True)

                with col2:
                    st.write("**Performance Metrics:**")
                    st.markdown(f"Home Goals/Game<br><span class='metric-value'>{feats['home_form_goals_for_5']:.2f}</span>", unsafe_allow_html=True)
                    st.markdown(f"Away Goals/Game<br><span class='metric-value'>{feats['away_form_goals_for_5']:.2f}</span>", unsafe_allow_html=True)
                    st.markdown(f"Defensive Ratio<br><span class='metric-value'>{(feats['home_form_goals_against_5'] - feats['away_form_goals_against_5']):.2f}</span>", unsafe_allow_html=True)