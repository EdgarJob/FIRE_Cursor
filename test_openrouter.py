import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.environ.get("OPENROUTER_API_KEY")
print(f"API key found: {'Yes' if api_key else 'No'}")
if api_key:
    # Remove any trailing characters if present
    api_key = api_key.strip()
    
    # Print the last 4 characters of the API key for verification
    print(f"Last 4 characters of API key: {api_key[-4:]}")
    
    # Try making a simple request to OpenRouter
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://fire.replit.app",
        "X-Title": "FIRE: Field Insight & Reporting Engine",
        "Content-Type": "application/json"
    }
    
    # Print the headers for debugging (hiding most of the API key)
    safe_headers = headers.copy()
    if len(api_key) > 8:
        safe_headers["Authorization"] = f"Bearer {api_key[:4]}...{api_key[-4:]}"
    print(f"Headers: {json.dumps(safe_headers, indent=2)}")
    
    data = {
        "model": "meta-llama/llama-4-scout:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, are you working properly?"}
        ]
    }
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    try:
        print("Sending test request to OpenRouter API...")
        response = requests.post(url, headers=headers, json=data)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        
        # Print all response headers for debugging
        print("\nResponse headers:")
        for header, value in response.headers.items():
            print(f"  {header}: {value}")
    except Exception as e:
        print(f"Error making request: {str(e)}")
else:
    print("No API key found in .env file. Please add your OpenRouter API key.") 