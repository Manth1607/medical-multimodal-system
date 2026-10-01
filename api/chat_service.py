
import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


def initialize_gemini():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception:
        return None


def get_chat_response(messages, patient_context, language):
    client = initialize_gemini()

    if not client:
        return "⚠️ API Key is missing. Please add your GEMINI_API_KEY to the .env file or enter it in the sidebar."

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
                return f"⚠️ An error occurred while communicating with the AI: {error_msg}"

    except Exception as e:
        return f"⚠️ An error occurred while communicating with the AI: {str(e)}"


def get_ai_recommendations(patient_context, language="Gujarati 🕉️"):
    client = initialize_gemini()

    if not client:
        return "⚠️ API Key is missing. Cannot generate recommendations."

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
                return f"⚠️ Error generating recommendations: {error_msg}"

    except Exception as e:
        return f"⚠️ Error generating recommendations: {str(e)}"

