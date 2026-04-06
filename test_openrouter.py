import os
import requests
from dotenv import load_dotenv

# Ensure the .env file is loaded
load_dotenv()
key = os.getenv("OPENROUTER_API_KEY")

if not key:
    print("❌ ERROR: OPENROUTER_API_KEY is missing from environment.")
else:
    print("✅ Key found in env!")

res = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Vichaar-Core"
    },
    json={
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}]
    }
)

print(f"Status: {res.status_code}")
print(f"Response: {res.text}")
