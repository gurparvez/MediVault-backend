from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import io
import json
from PIL import Image
from google.genai import Client
from google.genai import types
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ No Gemini API key found.")
app = FastAPI(title="MediVault Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class ChatRequest(BaseModel):
    query: str
    context: str  # The text from relevant local documents
    history: List[Dict[str, str]] = []  # Optional chat history


# --- Endpoints ---
@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    categories: List[str] = Form([]),
    allow_new_categories: bool = Form(False),
):
    """
    Full pipeline:
    1. Classify image
    2. Extract detailed context (OCR/Description)
    3. Extract structured EVENTS (for Calendar)
    4. Create summary
    5. Create embeddings
    """
    try:
        # Read file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="No file content received.")
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as img_err:
            raise HTTPException(
                status_code=400, detail=f"Invalid image file: {img_err}"
            )
        async with Client(api_key=api_key).aio as aclient:

            # --- 1. CLASSIFICATION ---
            classification_prompt = build_classification_prompt(
                categories, allow_new_categories
            )
            class_resp = await aclient.models.generate_content(
                model="gemini-2.0-flash",  # Use the latest fast model
                contents=[classification_prompt, image],
            )
            category = clean_text(class_resp.text) or "unknown"
            # --- 2. CONTEXT & EVENTS (Combined for efficiency) ---
            # We ask for a JSON response containing both the full text and any events found.
            analysis_prompt = """
            Analyze this medical document. Return a JSON object with two keys:
            1. "context_text": A detailed, plain-text description of EVERYTHING in the document. Include all visible text, dates, values, and doctor names.
            2. "events": A list of events found in the document (appointments, medication reminders, follow-ups). 
               Each event should have:
               - "title": Short title (e.g., "Cardiologist Appointment")
               - "date": The date/time in ISO format (YYYY-MM-DDTHH:MM:SS) if possible, or a clear string.
               - "type": One of ["appointment", "medication", "reminder", "other"]
               - "description": Brief details.
            
            If no events are found, "events" should be an empty list.
            """
            analysis_resp = await aclient.models.generate_content(
                model="gemini-2.0-flash",
                contents=[analysis_prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )

            raw_text = analysis_resp.text
            if not raw_text:
                raise HTTPException(
                    status_code=500, detail="Model returned empty analysis response."
                )

            analysis_data = json.loads(raw_text)

            context_text = analysis_data.get("context_text", "")
            extracted_events = analysis_data.get("events", [])
            # --- 3. SUMMARY (Short) ---
            summary_prompt = f"""
            Summarize this text into 2 lines for a quick preview card:
            {context_text}
            """
            summary_resp = await aclient.models.generate_content(
                model="gemini-2.0-flash", contents=[summary_prompt]
            )
            summary = clean_text(summary_resp.text)
            # --- 4. EMBEDDINGS ---
            # Create a rich string for embedding so search is accurate
            embedding_input = (
                f"Category: {category}\nSummary: {summary}\nContent: {context_text}"
            )

            emb_resp = await aclient.models.embed_content(
                model="text-embedding-004", contents=[embedding_input]
            )

            embedding = []
            if emb_resp.embeddings:
                embedding = emb_resp.embeddings[0].values
            return JSONResponse(
                {
                    "category": category,
                    "summary": summary,
                    "context_text": context_text,
                    "extracted_events": extracted_events,  # <--- NEW: Send this to your Calendar logic
                    "embedding": embedding,
                }
            )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_data(request: ChatRequest):
    """
    Chatbot endpoint.
    The App finds relevant local documents, extracts their text, and sends it here as 'context'.
    """
    try:
        system_instruction = """
        You are MediVault AI, a helpful medical assistant.
        Answer the user's question based ONLY on the provided context.
        If the answer is not in the context, say "I don't see that information in your records."
        Be concise, empathetic, and professional.
        """
        prompt = f"""
        CONTEXT FROM USER RECORDS:
        {request.context}
        USER QUESTION:
        {request.query}
        """
        async with Client(api_key=api_key).aio as aclient:
            response = await aclient.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
            )

            return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Helpers ----------
def build_classification_prompt(categories: List[str], allow_new: bool) -> str:
    base = "You are a medical document classifier. Categorize the image."
    if categories:
        list_str = ", ".join(f'"{c}"' for c in categories)
        if allow_new:
            return f"{base} Try to fit into: {list_str}. If not, generate a new concise name. Return ONLY the category name."
        else:
            return f"{base} Choose exactly one from: {list_str}. If none match, return 'other'."
    return "Analyze the image and return a single concise category name."


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.replace('"', "").replace("'", "").strip()
