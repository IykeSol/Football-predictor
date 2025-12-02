import soccerdata as sd
import pandas as pd
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
leagues = ["Big 5 European Leagues Combined"] 
seasons = ['2022', '2023', '2024', '2025'] 

print(f"--- INITIALIZING SOCCERDATA (Leagues: {leagues}) ---")
fbref = sd.FBref(leagues=leagues, seasons=seasons)

# --- 1. GET SCHEDULE AND SCORES ---
print("1/2 Downloading Schedule & Scores (Fast)...")
schedule = fbref.read_schedule().reset_index()

# --- 2. CLEAN AND FILTER DATA ---
print("2/2 Cleaning Data...")

# Fix 'score' column (Split "2-1" into Home=2, Away=1)
if 'score' in schedule.columns:
    scores = schedule['score'].str.split(r'[-–]', expand=True)
    schedule['home_score'] = pd.to_numeric(scores[0], errors='coerce')
    schedule['away_score'] = pd.to_numeric(scores[1], errors='coerce')

# Select ONLY the columns you need for the model
# We are removing all the messy columns that caused errors
cols_to_keep = [
    'league', 
    'season', 
    'date', 
    'home_team', 
    'away_team',
    'home_score', 
    'away_score', 
    'home_xg', 
    'away_xg'
]

# Create a clean dataframe with only these columns
clean_df = schedule[cols_to_keep].copy()

# Remove matches that haven't played yet (where score is NaN)
clean_df = clean_df.dropna(subset=['home_score', 'away_score'])

# Remove matches where xG is missing (to ensure 'Filled' data)
clean_df = clean_df.dropna(subset=['home_xg', 'away_xg'])

# Sort by date
clean_df['date'] = pd.to_datetime(clean_df['date'])
clean_df = clean_df.sort_values('date')

print(f"📊 Rows after cleaning: {len(clean_df)}")

# --- 3. SAVE TO EXCEL ---
file_name = "real_live_data_filled.xlsx"
print(f"...Saving to {file_name}...")

# We use index=False to keep the file clean
clean_df.to_excel(file_name, index=False)

print("\n" + "="*40)
print("✅ SUCCESS!")
print(f"📁 Data saved to: {file_name}")
print(f"🔢 Total Matches: {len(clean_df)}")
print("="*40)
print("This file contains ONLY:")
print("- Date, Teams, League")
print("- Final Scores (Home/Away)")
print("- Expected Goals (xG)")
print("(Empty columns and half-time scores were removed to prevent errors)")