import google.generativeai as genai
import toml

# Load secrets
try:
    secrets = toml.load(".streamlit/secrets.toml")
    api_key = secrets['google']['gemini_api_key']
    genai.configure(api_key=api_key)
except:
    print("❌ Could not load secrets.toml")
    exit()

print("🔎 Checking available Gemini models for your API key...")
print("-" * 50)

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ AVAILABLE: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")