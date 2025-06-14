import os
import base64
import imghdr
from PIL import Image
from io import BytesIO
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.genai as genai
import google.genai.types as types

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

# --- Setup FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in prod
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class QARequest(BaseModel):
    question: str
    image: str | None = None

# --- Embedding with Gemini ---
def get_embedding(text, model="models/text-embedding-004"):
    try:
        response = client.models.embed_content(
            model=model,
            contents=[text],
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

# --- Chunk search ---
def search_chunks(question, top_k=3):
    embedding = get_embedding(question)
    if embedding is None:
        return []
    embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
    similarities = np.dot(embeddings, embedding.T).squeeze()
    top_indices = similarities.argsort()[::-1][:top_k]
    return [metadata[i] for i in top_indices]

# --- Collect helpful links ---
def collect_links(chunks, max_links=5):
    links = []
    seen = set()
    for chunk in chunks:
        url = chunk.get("url")
        if url and url not in seen:
            links.append({"url": url, "text": chunk.get("content", "")[:250]})
            seen.add(url)
        if len(links) >= max_links:
            break
    return links

# --- Detect image MIME type ---
def get_image_mimetype(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        img_type = imghdr.what(None, h=image_data)
        mime_type = f'image/{img_type}' if img_type else 'application/octet-stream'
        return mime_type, img_type, image_data
    except Exception as e:
        print(f"[MIME Detection Error] {e}")
        return None, None, None

# --- Get image description from bytes using Gemini ---
def get_image_description_from_bytes(image_data, mime_type):
    try:
        image = Image.open(BytesIO(image_data))
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[image, "Describe the content of this image in detail, especially any text or UI elements."],
        )
        return response.text
    except Exception as e:
        print(f"[Gemini Image Description Error] {e}")
        return "Image could not be processed or described."

@app.post("/api")
async def ask(request: QARequest):
    question = request.question
    base64_image = request.image
    image_description = ""

    if base64_image:
        mime_type, img_type, image_data = get_image_mimetype(base64_image)
        if img_type not in {"jpeg", "png", "webp"}:
            return JSONResponse(content={"error": f"Unsupported image type: {img_type}"}, status_code=400)

        image_description = get_image_description_from_bytes(image_data, mime_type)

    if not question:
        return JSONResponse(content={"error": "question is required."}, status_code=400)

    # --- Search for relevant chunks ---
    chunks = search_chunks(question, top_k=5)
    context = "\n---\n".join(chunk["content"] for chunk in chunks) if chunks else ""

    # --- Build prompt for Gemini ---
    prompt = f"""You are an expert assistant.Answer the question using the following context and image description.
Be concise and accurate. Use the image information if it helps to answer the question.

Context:
{context}

Image Description:
{image_description if image_description else 'No image provided.'}

Question: {question}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        answer = response.text if hasattr(response, "text") else "Sorry, no valid answer returned."
    except Exception as e:
        print(f"[Gemini Generation Error] {e}")
        answer = "Sorry, couldn't fetch an answer due to a technical issue."

    return {"answer": answer, "links": collect_links(chunks)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5045)
