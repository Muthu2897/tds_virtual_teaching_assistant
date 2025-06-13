import json
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
import google.genai as genai
import google.genai.types as types
import pickle

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("🛑 GEMINI_API_KEY missing from .env")

# Initialize client and embedding model
client = genai.Client(api_key=GEMINI_API_KEY)

# Configuration
BATCH_SIZE = 30
MAX_RETRIES = 10

def embed_texts_gemini(texts, model="models/text-embedding-004", task_type="retrieval_document"):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🔗 Embedding"):
        batch = texts[i:i + BATCH_SIZE]
        retry_count = 0

        while retry_count <= MAX_RETRIES:
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                )
                # Ensure we get embeddings as plain list of floats
                for embedding in response.embeddings:
                    values = embedding.values if hasattr(embedding, "values") else embedding
                    all_embeddings.append(values.tolist() if hasattr(values, "tolist") else values)
                break  # Success
            except Exception as e:
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    print(f"❌ Failed batch {i}-{i+len(batch)-1}: {e}")
                    all_embeddings.extend([None] * len(batch))  # Fill with None for failures
                    break
                else:
                    wait = 2 ** retry_count
                    print(f"⚠️ Error on batch {i}-{i+len(batch)-1}, retrying {retry_count}/{MAX_RETRIES} in {wait}s: {e}")
                    time.sleep(wait)
    return all_embeddings

if __name__ == "__main__":
    with open("scripts/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["content"].replace("\n", " ") for chunk in chunks]
    embeddings = embed_texts_gemini(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    os.makedirs("scripts", exist_ok=True)

    try:
        with open("scripts/embedded_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        print("✅ Embeddings saved to scripts/embedded_chunks.json")
    except TypeError as e:
        print(f"⚠️ JSON serialization failed: {e}")
        with open("scripts/embedded_chunks_backup.pkl", "wb") as f:
            pickle.dump(chunks, f)
        print("✅ Fallback: Saved embeddings as Pickle in scripts/embedded_chunks_backup.pkl")
