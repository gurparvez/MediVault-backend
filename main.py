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
from pydantic import BaseModel, Field
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

class ExtractedEvent(BaseModel):
    title: str = Field(..., description="Short title, e.g., 'Cardiologist Appt'")
    date: str = Field(..., description="ISO format date string or clear text description")
    type: str = Field(..., description="One of: appointment, medication, reminder, other")
    description: str = Field(..., description="Brief details about the event")

class AnalysisResponse(BaseModel):
    category: str = Field(..., description="The best fitting category for the document")
    summary: str = Field(..., description="A concise 2-line summary of the content")
    context_text: str = Field(..., description="Detailed OCR/Text extraction of the document")
    events: List[ExtractedEvent] = Field(default_factory=list, description="List of calendar events found")

class QueryRequest(BaseModel):
    text: str

class ChatContextRequest(BaseModel):
    query: str
    context_text: str

# --- Endpoints ---
@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    categories: List[str] = Form([]),
    allow_new_categories: bool = Form(False),
):
    try:
        # 1. Read file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="No file content received.")
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as img_err:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {img_err}")

        # 2. Construct Dynamic Prompt
        # We inject the user's categories into the instructions.
        categories_str = ", ".join(categories) if categories else "Medical, Prescription, Lab Report, Bill, Other"
        
        prompt_instructions = f"""
        You are an expert AI medical assistant. Analyze the attached image and extracting structured data.
        
        Perform these 4 tasks simultaneously:
        1. **Classify**: Categorize the document. Choose strictly from this list: [{categories_str}]. 
           {'If the document does not fit, suggest a new concise category name.' if allow_new_categories else 'If it does not fit, use "Other".'}
        2. **OCR/Context**: Extract all visible text, dates, doctor names, and medical values into a detailed text block.
        3. **Events**: Identify any actionable events (appointments, refill reminders) for a calendar.
        4. **Summary**: Create a brief 2-line summary for a preview card.
        """

        async with Client(api_key=api_key).aio as aclient:
            
            # --- ONE GENERATION CALL (Replaces the previous 3 calls) ---
            response = await aclient.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt_instructions, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnalysisResponse,
                ),
            )

            # Gemini returns a strict JSON object matching our schema
            analysis_result: AnalysisResponse = response.parsed

            # --- ONE EMBEDDING CALL ---
            # We use the text we just generated to create the embedding.
            # This is fast and cheap.
            embedding_input = (
                f"Category: {analysis_result.category}\n"
                f"Summary: {analysis_result.summary}\n"
                f"Content: {analysis_result.context_text}"
            )

            emb_resp = await aclient.models.embed_content(
                model="text-embedding-004",
                contents=[embedding_input]
            )
            
            embedding_vector = []
            if emb_resp.embeddings:
                embedding_vector = emb_resp.embeddings[0].values

            # Return combined result
            return JSONResponse({
                "category": analysis_result.category,
                "summary": analysis_result.summary,
                "context_text": analysis_result.context_text,
                "extracted_events": [e.model_dump() for e in analysis_result.events],
                "embedding": embedding_vector
            })

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/get-embedding")
async def get_embedding_for_query(request: QueryRequest):
    """
    1. Mobile App sends user query here (e.g., "What is my vitamin D level?").
    2. We return the vector.
    3. Mobile App uses this vector to search SQLite.
    """
    try:
        async with Client(api_key=api_key).aio as aclient:
            emb_resp = await aclient.models.embed_content(
                model="text-embedding-004",
                contents=[request.text]
            )
            if not emb_resp.embeddings:
                raise HTTPException(status_code=500, detail="Failed to generate embedding")
            
            return JSONResponse({"embedding": emb_resp.embeddings[0].values})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-with-docs")
async def chat_with_docs(request: ChatContextRequest):
    """
    1. Mobile App finds the top 3 relevant documents from SQLite.
    2. Mobile App concatenates their text into 'context_text'.
    3. We send that + the query to Gemini to answer.
    """
    try:
        prompt = f"""
        You are a helpful medical assistant named MediVault.
        Answer the user's question strictly based on the provided context.
        
        CONTEXT FROM USER'S DOCUMENTS:
        {request.context_text}
        
        USER QUESTION:
        {request.query}
        
        If the answer is not in the context, say "I cannot find that information in your saved documents."
        """

        async with Client(api_key=api_key).aio as aclient:
            response = await aclient.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt]
            )
            
            return JSONResponse({"answer": response.text})

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
