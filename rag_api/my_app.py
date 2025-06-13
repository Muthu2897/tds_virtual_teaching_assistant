import os
import uuid
import base64
import imghdr
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.genai as genai
import google.genai.types as types
import traceback
from typing import Optional


# --- Load environment ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("🛑 GEMINI_API_KEY missing from .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# --- Load unified NPZ vectorstore ---

data = np.load("database.npz", allow_pickle=True)

embeddings = data["embeddings"]
metadata = data["metadata"]

# --- Embedder using Gemini SDK ---
def get_embedding(text, model="models/text-embedding-004"):
    try:
        response = client.models.embed_content(
            model=model,
            contents=[text],
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return None

# --- Search from NPZ vectorstore ---
def search_chunks(question, top_k=5):
    embedding = get_embedding(question)
    if embedding is None:
        return []

    embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
    similarities = np.dot(embeddings, embedding.T).squeeze()
    top_indices = similarities.argsort()[::-1][:top_k]
    return [metadata[i] for i in top_indices]

# --- FastAPI app setup ---
my_app = FastAPI()

# --- CORS Middleware ---
my_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Schema ---
class AskRequest(BaseModel):
    question: str
    image: Optional[str] = None

@my_app.post("/api")
async def ask(data: AskRequest):
    question = data.question
    base64_image = data.image

    if not question:
        return JSONResponse(status_code=400, content={"error": "question is required."})

    # --- Optional image handling ---
    if base64_image:
        try:
            image_data = base64.b64decode(base64_image)
            image_type = imghdr.what(None, h=image_data)
            if image_type not in {"jpeg", "png", "webp"}:
                return JSONResponse(status_code=400, content={"error": f"Unsupported image type: {image_type}"})
            filename = f"uploaded_image_{uuid.uuid4().hex}.{image_type}"
            with open(filename, "wb") as f:
                f.write(image_data)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Image processing failed: {str(e)}"})

    # --- Chunk search ---
    chunks = search_chunks(question, top_k=8)
    context = "\n---\n".join(chunk["content"] for chunk in chunks)

    prompt_text = f"""Use only the context below to answer the question. Be to the point. If the context does not contain the answer, respond with: "Not found in context."

{context}

Question: {question}"""

    # --- Gemini text generation ---
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt_text]
        )
        answer = response.text.strip()
    except Exception as e:
        print("❌ Gemini Chat Error:", e)
        traceback.print_exc()
        answer = "Sorry, couldn't fetch an answer due to a technical issue."

    # --- Collect top reference links ---
    def collect_links(chunks, max_links=4):
        links = []
        seen = set()
        for chunk in chunks:
            url = chunk.get("url")
            if url and url not in seen:
                links.append({
                    "url": url,
                    "text": chunk.get("content", "")[:250]
                })
                seen.add(url)
            if len(links) >= max_links:
                break
        return links

    return {
        "answer": answer,
        "links": collect_links(chunks)
    }
