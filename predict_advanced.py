import pandas as pd
import numpy as np
import joblib
import os
import re
import difflib

def get_closest_team_name(user_input, all_teams):
    """Finds the closest matching team name."""
    matches = difflib.get_close_matches(user_input, all_teams, n=1, cutoff=0.5)
    return matches[0] if matches else None

def predict_real_data():
    print("\n" + "=" * 60)
    print("🎯 REAL DATA FOOTBALL PREDICTOR")
    print("=" * 60)
    
    # --- 1. LOAD MODEL ---
    try:
        model = joblib.load("real_data_football_model.pkl")
        trained_features = model.feature_names_in_
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please run train_system.py first!")
        return
    
    # --- 2. LOAD DATA ---
    print("...Loading Database...")
    if os.path.exists("real_live_data_filled.xlsx"):
        try:
            df = pd.read_excel("real_live_data_filled.xlsx", engine='openpyxl')
        except:
            df = pd.read_csv("real_live_data_filled.csv")
    elif os.path.exists("real_live_data_filled.csv"):
        df = pd.read_csv("real_live_data_filled.csv")
    else:
        print("❌ Data file not found!")
        return

    # Clean Columns
    columns_to_keep = []
    seen_columns = set()
    for col in reversed(df.columns):
        clean_col = re.sub(r'\.\d+$', '', col)
        if clean_col not in seen_columns:
            columns_to_keep.append(col)
            seen_columns.add(clean_col)
    columns_to_keep.reverse()
    df = df[columns_to_keep]
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    all_teams = list(pd.concat([df['home_team'], df['away_team']]).unique())
    print(f"✅ Database loaded: {len(df)} matches, {len(all_teams)} teams")
    
    # --- 3. USER INPUT ---
    print(f"\n🏟️  ENTER MATCH DETAILS")
    print("-" * 40)
    
    while True:
        raw_home = input("Enter HOME team: ").strip()
        home_team = get_closest_team_name(raw_home, all_teams)
        if home_team:
            print(f"   👉 Selected: {home_team}")
            break
        print("   ❌ Team not found.")

    while True:
        raw_away = input("Enter AWAY team: ").strip()
        away_team = get_closest_team_name(raw_away, all_teams)
        if away_team:
            print(f"   👉 Selected: {away_team}")
            break
        print("   ❌ Team not found.")
    
    if home_team == away_team:
        print("\n❌ Error: Teams must be different!")
        return

    # --- 4. GATHER DATA ---
    feature_dict = {}
    
    # Get recent matches
    home_matches = df[((df['home_team'] == home_team) | (df['away_team'] == home_team))].tail(5)
    away_matches = df[((df['home_team'] == away_team) | (df['away_team'] == away_team))].tail(5)
    
    # Get H2H
    h2h_matches = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team) |
         (df['home_team'] == away_team) & (df['away_team'] == home_team))
    ].tail(10)

    # --- 5. CALCULATE STATS (FORM & H2H) ---
    
    def calculate_stats(matches, team):
        if len(matches) == 0: return None
        gf, ga, xgf, xga, pts = [], [], [], [], []
        for _, m in matches.iterrows():
            if m['home_team'] == team:
                gf.append(m['home_score']); ga.append(m['away_score'])
                xgf.append(m['home_xg']); xga.append(m['away_xg'])
                pts.append(3 if m['home_score'] > m['away_score'] else 1 if m['home_score'] == m['away_score'] else 0)
            else:
                gf.append(m['away_score']); ga.append(m['home_score'])
                xgf.append(m['away_xg']); xga.append(m['home_xg'])
                pts.append(3 if m['away_score'] > m['home_score'] else 1 if m['away_score'] == m['home_score'] else 0)
        return {
            'goals_for': np.mean(gf), 'goals_against': np.mean(ga),
            'xg_for': np.mean(xgf), 'xg_against': np.mean(xga),
            'points': np.sum(pts)
        }

    h_stats = calculate_stats(home_matches, home_team)
    a_stats = calculate_stats(away_matches, away_team)

    if not h_stats or not a_stats:
        print("❌ Not enough data for one of the teams.")
        return

    # Store Form Features
    feature_dict['home_form_goals_for_5'] = h_stats['goals_for']
    feature_dict['home_form_goals_against_5'] = h_stats['goals_against']
    feature_dict['home_form_xg_for_5'] = h_stats['xg_for']
    feature_dict['home_form_xg_against_5'] = h_stats['xg_against']
    feature_dict['home_form_points_5'] = h_stats['points']
    
    feature_dict['away_form_goals_for_5'] = a_stats['goals_for']
    feature_dict['away_form_goals_against_5'] = a_stats['goals_against']
    feature_dict['away_form_xg_for_5'] = a_stats['xg_for']
    feature_dict['away_form_xg_against_5'] = a_stats['xg_against']
    feature_dict['away_form_points_5'] = a_stats['points']

    # Store H2H Features
    h_wins = 0; a_wins = 0; h_goals = 0; a_goals = 0
    if len(h2h_matches) > 0:
        for _, m in h2h_matches.iterrows():
            if m['home_team'] == home_team:
                h_goals += m['home_score']; a_goals += m['away_score']
                if m['home_score'] > m['away_score']: h_wins += 1
                elif m['home_score'] < m['away_score']: a_wins += 1
            else:
                h_goals += m['away_score']; a_goals += m['home_score']
                if m['away_score'] > m['home_score']: h_wins += 1
                elif m['away_score'] < m['home_score']: a_wins += 1
        
        feature_dict['h2h_matches'] = len(h2h_matches)
        feature_dict['h2h_home_win_pct'] = h_wins / len(h2h_matches)
        feature_dict['h2h_away_win_pct'] = a_wins / len(h2h_matches)
        feature_dict['h2h_avg_goals'] = (h_goals + a_goals) / len(h2h_matches)
        feature_dict['h2h_avg_home_goals'] = h_goals / len(h2h_matches)
        feature_dict['h2h_avg_away_goals'] = a_goals / len(h2h_matches)
    else:
        # Defaults
        feature_dict['h2h_matches'] = 0
        feature_dict['h2h_home_win_pct'] = 0; feature_dict['h2h_away_win_pct'] = 0
        feature_dict['h2h_avg_goals'] = 0; feature_dict['h2h_avg_home_goals'] = 0; feature_dict['h2h_avg_away_goals'] = 0

    # --- 6. DISPLAY STATS ---
    print(f"\n📊 FORM ANALYSIS (Last 5 Games)")
    print(f"   🏠 {home_team}: {h_stats['points']} pts, {h_stats['goals_for']:.1f} Scored/gm, {h_stats['goals_against']:.1f} Conceded/gm")
    print(f"   🚌 {away_team}: {a_stats['points']} pts, {a_stats['goals_for']:.1f} Scored/gm, {a_stats['goals_against']:.1f} Conceded/gm")
    
    print(f"\n⚔️  HEAD-TO-HEAD ({len(h2h_matches)} Matches)")
    if len(h2h_matches) > 0:
        print(f"   {home_team} Wins: {int(feature_dict['h2h_home_win_pct']*len(h2h_matches))}")
        print(f"   {away_team} Wins: {int(feature_dict['h2h_away_win_pct']*len(h2h_matches))}")
    else:
        print("   No history found.")

    # --- 7. xG INPUT ---
    print(f"\n⚖️  xG EXPECTATIONS")
    auto_h_xg = h_stats['xg_for']
    auto_a_xg = a_stats['xg_for']
    print(f"   Calculated Averages: {home_team} ({auto_h_xg:.2f}) vs {away_team} ({auto_a_xg:.2f})")
    
    use_custom = input("   Enter custom xG? (y/n): ").lower().strip()
    if use_custom == 'y':
        try:
            feature_dict['home_xg'] = float(input(f"   {home_team} xG: "))
            feature_dict['away_xg'] = float(input(f"   {away_team} xG: "))
        except:
            feature_dict['home_xg'] = auto_h_xg
            feature_dict['away_xg'] = auto_a_xg
    else:
        feature_dict['home_xg'] = auto_h_xg
        feature_dict['away_xg'] = auto_a_xg

    # --- 8. PREDICTION ---
    # Convert dictionary to DataFrame to fix the warning
    input_df = pd.DataFrame([feature_dict])
    
    # Ensure columns are in the exact same order as training
    input_df = input_df[trained_features]
    
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    print("\n" + "=" * 50)
    print(f"📢 FINAL VERDICT: {home_team} vs {away_team}")
    print("=" * 50)
    
    if prediction == 1:
        print(f"🏆 RESULT: HOME WIN ({home_team})")
        conf = probs[1]
    else:
        print(f"🛡️ RESULT: DRAW or AWAY WIN ({away_team})")
        conf = probs[0]
        
    print(f"\n📊 PROBABILITIES:")
    print(f"   🏠 {home_team} Win:     {probs[1]*100:.1f}%")
    print(f"   🚌 {away_team} Win/Draw: {probs[0]*100:.1f}%")
    
    print(f"\n💡 BETTING ADVICE:")
    if conf > 0.75: print("   ⭐⭐⭐ STRONG BET")
    elif conf > 0.60: print("   ⭐⭐ GOOD BET")
    else: print("   ⭐ RISKY / NO BET")

if __name__ == "__main__":
    predict_real_data()