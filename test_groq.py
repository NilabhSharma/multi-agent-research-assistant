import os
from dotenv import load_dotenv
from groq import Groq

# Load variables from .env into the environment
load_dotenv()

# Read the key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

# Create a client
client = Groq(api_key=api_key)

# Make a simple chat completion call
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "In one sentence, what is a research assistant AI agent?"}
    ]
)

print("Response from Groq:")
print(response.choices[0].message.content)