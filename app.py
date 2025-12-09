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

# --- 2. AI ANALYSIS (DETAILED & CRITICAL) ---
def get_gemini_analysis(home_team, away_team, prediction_text, confidence, home_recent_str, away_recent_str, h2h_str, h2h_avg_goals, h2h_home_win):
    model_name = "models/gemini-2.5-flash" 
    
    prompt = f"""
    You are a Senior Football Betting Analyst known for deep tactical breakdowns and finding value where others miss it.
    
    **MATCH:** {home_team} vs {away_team}
    
    **THE MACHINE MODEL SAYS:** {prediction_text} ({confidence}% Confidence)
    
    **THE REAL DATA:**
    - **H2H Avg Goals:** {h2h_avg_goals} per match
    - **{home_team} Recent Form:**
    {home_recent_str}
    - **{away_team} Recent Form:**
    {away_recent_str}
    - **Head-to-Head History:**
    {h2h_str}
    
    **YOUR INSTRUCTIONS:**
    1. **BE CRITICAL:** Do not blindly agree with the Machine Model. If the favorite has high xG but low goals, they are wasteful. If the underdog has a great H2H record, mention it!
    2. **BE DETAILED:** Do not write short sentences. Write a full analysis of the form. Explain *why* a team is winning or losing based on the xG data provided.
    3. **GOAL MARKETS:** Analyze the "Over 1.5" and "BTTS" potential deeply.
    
    **FORMAT EXACTLY AS FOLLOWS (Markdown):**
    
    ### 📝 The Analyst's Deep Dive
    [Write a detailed paragraph (4-5 sentences) analyzing the match. Discuss xG performance, defensive leaks, and whether the Machine Model is right or wrong. Be opinionated.]
    
    ### 🎯 Value Predictions
    *   **Best Bet:** [e.g., Over 1.5 Goals / Home Win / Draw]
    *   **The "Smart" Pick:** [A value pick like BTTS Yes or Under 2.5]
    *   **Reasoning:** [Detailed explanation of why these bets make sense statistically.]
    
    ### ⚠️ Risk & Tactical Analysis
    *   **Tactical Mismatch:** [Explain a specific tactical weakness shown in the form data]
    *   **H2H Factor:** [Analyze the historical psychological edge]

    ### 🏁 Final Verdict
    [One final, decisive sentence telling the user what to do. e.g., "Ignore the model's caution and back Newcastle to win based on their dominant H2H record."]
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
                # --- STRINGS FUNCTION (FIXED: Team-Centric & Explicit xG) ---
                def m_str(matches, team):
                    s = ""
                    for _, m in matches.iterrows():
                        # Determine if Team is Home or Away
                        if m['home_team'] == team:
                            venue = "(H)"
                            opp = m['away_team']
                            my_score = m['home_score']
                            opp_score = m['away_score']
                            my_xg = m.get('home_xg', 0)
                            opp_xg = m.get('away_xg', 0)
                        else:
                            venue = "(A)"
                            opp = m['home_team']
                            my_score = m['away_score']
                            opp_score = m['home_score']
                            my_xg = m.get('away_xg', 0)
                            opp_xg = m.get('home_xg', 0)
                        
                        # Determine Result (W/D/L)
                        if my_score > opp_score: res = "W"
                        elif my_score < opp_score: res = "L"
                        else: res = "D"
                            
                        # Format: Date (Venue) vs Opponent (Result): MyScore-OppScore (xG: MyXG vs OppXG)
                        # This removes ALL ambiguity.
                        s += f"- {m['date'].strftime('%Y-%m-%d')} {venue} vs {opp} ({res}): {my_score}-{opp_score} (xG: {my_xg:.2f} vs {opp_xg:.2f})\n"
                    return s

                h_s = m_str(h_match, home_team)
                a_s = m_str(a_match, away_team)
                h2h_s = "\n".join([f"- {m['date'].strftime('%Y-%m-%d')}: {m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']}" for _, m in h2h.tail(5).iterrows()])

                # --- 1. CALL AI ---
                h2h_avg = f"{feats['h2h_avg_goals']:.2f}"
                h2h_home_win_rt = f"{feats['h2h_home_win_pct']*100:.0f}%"
                prov_verdict = f"{home_team} Win" if pred == 1 else f"{away_team} Win/Draw"
                prov_conf = prob[1]*100 if pred == 1 else prob[0]*100
                
                ai_response = get_gemini_analysis(home_team, away_team, prov_verdict, int(prov_conf), h_s, a_s, h2h_s, h2h_avg, h2h_home_win_rt)
                
                # --- 2. HEADER CARD ---
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
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # --- 3. TABS ---
                st.markdown('<div class="sub-header">📊 Match Analysis</div>', unsafe_allow_html=True)
                t1, t2, t3 = st.tabs(["Form Analysis", "H2H History", "🤖 AI Insights (Detailed)"])
                
                with t1:
                    c1, c2 = st.columns(2)
                    with c1: 
                        st.markdown(f"**{home_team} Last 5 Matches:**")
                        st.text(h_s)
                    with c2: 
                        st.markdown(f"**{away_team} Last 5 Matches:**")
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
                    st.markdown(ai_response)

                # --- 4. ADDITIONAL INSIGHTS ---
                st.markdown('<div class="sub-header">🔍 Additional Insights</div>', unsafe_allow_html=True)
                
                def big_stat(label, value):
                    return f"""
                    <div style="margin-bottom: 20px;">
                        <p style="font-size: 14px; color: #555; margin-bottom: 2px;">{label}</p>
                        <p style="font-size: 28px; font-weight: 800; color: #111; margin-top: 0px;">{value}</p>
                    </div>
                    """
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Key Statistics:**")
                    st.markdown(big_stat("Home Form (Last 5)", f"{int(feats['home_form_points_5'])} pts"), unsafe_allow_html=True)
                    st.markdown(big_stat("Away Form (Last 5)", f"{int(feats['away_form_points_5'])} pts"), unsafe_allow_html=True)
                    st.markdown(big_stat("Home xG Advantage", f"{(feats['home_xg'] - feats['away_xg']):.2f}"), unsafe_allow_html=True)

                with col2:
                    st.markdown("**Performance Metrics:**")
                    st.markdown(big_stat("Home Goals/Game", f"{feats['home_form_goals_for_5']:.2f}"), unsafe_allow_html=True)
                    st.markdown(big_stat("Away Goals/Game", f"{feats['away_form_goals_for_5']:.2f}"), unsafe_allow_html=True)

                    st.markdown(big_stat("Defensive Ratio", f"{(feats['home_form_goals_against_5'] - feats['away_form_goals_against_5']):.2f}"), unsafe_allow_html=True)
