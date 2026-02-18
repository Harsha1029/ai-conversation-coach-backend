from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
import google.generativeai as genai

# Load env
load_dotenv()

app = FastAPI(title="TactAI - Multi AI Conversation Coach")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None

SYSTEM_PROMPT = """
You are TactAI, a professional AI Difficult Conversation Coach.

Behavior Rules:

1. If user greets (hi, hello, hey):
   - Respond politely and briefly.

2. If user gives simple question:
   - Respond clearly and professionally.

3. If user describes a real difficult situation:
   - Provide structured coaching:

   1️⃣ WHAT TO SAY
   2️⃣ TONE & DELIVERY TIPS
   3️⃣ WHAT NOT TO SAY
   4️⃣ POSSIBLE REACTIONS & HOW TO RESPOND

Be emotionally intelligent, realistic, and practical.
Avoid generic advice.
Adapt response length to situation complexity.
"""

@app.get("/")
def home():
    return {"message": "TactAI Multi-AI Backend Running 🚀"}

# -------- AI PROVIDERS --------

def generate_groq(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content


def generate_openai(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content


def generate_gemini(prompt):
    response = gemini_model.generate_content(
        SYSTEM_PROMPT + "\n\n" + prompt
    )
    return response.text


# -------- SMART ROUTER --------

def choose_best_model(message):
    message_length = len(message)

    # Complex professional scenario → OpenAI
    if message_length > 600 and openai_client:
        return "openai"

    # Emotional relationship scenario → Gemini
    emotional_keywords = ["breakup", "relationship", "partner", "feel hurt"]
    if any(word in message.lower() for word in emotional_keywords) and gemini_model:
        return "gemini"

    # Default → Groq (fast + cheap)
    if groq_client:
        return "groq"

    return None


@app.post("/generate")
async def generate(user_input: dict):
    try:
        message = user_input.get("message")

        if not message:
            return {"error": "Message is required."}

        # Simple greeting shortcut
        if message.lower().strip() in ["hi", "hello", "hey"]:
            return {
                "response": "Hello! How can I help you prepare for a difficult conversation today?"
            }

        provider = choose_best_model(message)

        # Try selected provider first
        try:
            if provider == "openai":
                return {"response": generate_openai(message)}

            elif provider == "gemini":
                return {"response": generate_gemini(message)}

            elif provider == "groq":
                return {"response": generate_groq(message)}

        except Exception:
            pass  # fallback below

        # Fallback chain
        if groq_client:
            try:
                return {"response": generate_groq(message)}
            except:
                pass

        if openai_client:
            try:
                return {"response": generate_openai(message)}
            except:
                pass

        if gemini_model:
            try:
                return {"response": generate_gemini(message)}
            except:
                pass

        return {"error": "All AI providers failed."}

    except Exception as e:
        return {"error": str(e)}
