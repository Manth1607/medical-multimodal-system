
import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


def get_gemini_api_key() -> str | None:
    """Retrieve and sanitize the Gemini API key from environment or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY")

    if not key:
        # Check Streamlit secrets if running in Streamlit Cloud / environment
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if not key:
        return None

    # Sanitize key: strip whitespace and accidental surrounding quotes
    key = str(key).strip().strip("'\"")

    # Reject placeholder values from example configurations
    placeholders = {
        "your-api-key-here",
        "your_api_key_here",
        "your-gemini-api-key-here",
        "your_gemini_api_key_here",
        "placeholder",
        "none",
        "null",
        "",
    }
    if key.lower() in placeholders:
        return None

    return key


def initialize_gemini():
    """Initialize Google GenAI client using the sanitized API key."""
    api_key = get_gemini_api_key()

    if not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception:
        return None


def _format_gemini_error(error: Exception) -> str:
    """Format Gemini API errors into clear, actionable, and secure messages."""
    error_msg = str(error)

    if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
        return (
            "⚠️ The configured Gemini API key is invalid or unauthorized. "
            "Please verify that 'GEMINI_API_KEY' in your Railway environment variables "
            "(Variables tab) is a valid, active key from Google AI Studio "
            "(https://aistudio.google.com/) without extra quotes or spaces."
        )
    elif "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
        return (
            "⚠️ Gemini API rate limit or quota exceeded. "
            "Please check your Google AI Studio quota or try again in a few moments."
        )
    elif "NOT_FOUND" in error_msg or "404" in error_msg:
        return (
            f"⚠️ Configured Gemini model '{MODEL_NAME}' was not found. "
            "Please check your GEMINI_MODEL setting in your environment variables."
        )
    elif "PERMISSION_DENIED" in error_msg or "403" in error_msg:
        return (
            "⚠️ Access denied by Google Gemini API. "
            "Please verify that the Generative Language API is enabled for your project/key."
        )

    # Generic clean error without leaking sensitive tokens
    first_line = error_msg.split("\n")[0]
    return f"⚠️ Error generating AI response: {first_line}"


def get_chat_response(messages, patient_context, language):
    client = initialize_gemini()

    if not client:
        return (
            "⚠️ Gemini API Key is missing or not configured. "
            "Please configure 'GEMINI_API_KEY' in your Railway Variables or local .env file."
        )

    try:
        # Build the system instructions
        system_prompt = f"""
You are an advanced AI Medical Assistant integrated into a highly professional hospital dashboard.
Your primary job is to assist the doctor by answering questions about the currently selected patient.

CURRENT PATIENT CONTEXT:
{patient_context}

RULES:
1. You must answer the doctor's questions based ONLY on the provided patient context.
2. If the required data is missing, clearly say that you don't have that information.
3. DO NOT hallucinate medical data.
4. Keep your answers concise, professional, and clear.
5. CRITICAL RULE: You MUST reply strictly and entirely in the language requested by the user.
6. Do not output English if Gujarati or Hindi is requested.

REQUESTED LANGUAGE:
{language}
"""

        # Convert Streamlit chat history to Gemini history
        formatted_history = []

        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"

            formatted_history.append(
                {
                    "role": role,
                    "parts": [
                        {
                            "text": msg["content"]
                        }
                    ]
                }
            )

        # Start Gemini chat
        chat = client.chats.create(
            model=MODEL_NAME,
            history=formatted_history
        )

        # Latest doctor question
        latest_query = messages[-1]["content"]

        full_query = f"""
{system_prompt}

DOCTOR'S QUESTION:
{latest_query}
"""

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = chat.send_message(full_query)
                return response.text
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return _format_gemini_error(e)

    except Exception as e:
        return _format_gemini_error(e)


def get_ai_recommendations(patient_context, language="Gujarati 🕉️"):
    client = initialize_gemini()

    if not client:
        return (
            "⚠️ Gemini API Key is missing or not configured. "
            "Please configure 'GEMINI_API_KEY' in your Railway Variables or local .env file."
        )

    try:
        prompt = f"""
You are an expert AI Medical Assistant.

The following patient has been predicted by our ML model to have a High Critical Emergency Risk (>60%).

Based strictly on the patient's available data, provide:

1. 💊 Recommended Immediate Medical Actions & Medicines
   - Phrase these as suggestions for the doctor to review.
   - Do not invent patient information.

2. 🧘‍♂️ Recommended Daily Routine / Lifestyle / Diet changes
   - Base suggestions only on the provided patient context.

PATIENT CONTEXT:
{patient_context}

Format the response nicely using markdown bullets.

CRITICAL RULE:
You MUST reply entirely and naturally in the requested language.
Do NOT output English if Gujarati or Hindi is requested.

REQUESTED LANGUAGE:
{language}
"""

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return _format_gemini_error(e)

    except Exception as e:
        return _format_gemini_error(e)


