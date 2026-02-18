from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Difficult Conversation Coach")

# Enable CORS (important for Flutter frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. Later restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# System prompt (controls AI behavior)
SYSTEM_PROMPT = """
You are a professional communication coach.

For every response, structure your answer exactly like this:

1️⃣ WHAT TO SAY (exact script)
2️⃣ TONE & DELIVERY TIPS
3️⃣ WHAT NOT TO SAY
4️⃣ POSSIBLE REACTIONS & HOW TO RESPOND

Be clear, practical, and emotionally intelligent.
Avoid generic advice.
Keep responses concise but powerful.
"""

# Health check route
@app.get("/")
def home():
    return {"message": "AI Conversation Coach Backend Running"}

# Main AI generation endpoint
@app.post("/generate")
async def generate(user_input: dict):
    try:
        scenario = user_input.get("scenario")
        details = user_input.get("details")

        if not scenario or not details:
            return {"error": "Both 'scenario' and 'details' are required."}

        user_prompt = f"""
        Scenario: {scenario}
        Details: {details}
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=800
        )

        return {
            "response": chat_completion.choices[0].message.content
        }

    except Exception as e:
        return {"error": str(e)}
