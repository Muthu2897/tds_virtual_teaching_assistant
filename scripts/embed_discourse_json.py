import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import google.genai as genai
import google.genai.types as types

# Load .env variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("🛑 GEMINI_API_KEY missing from .env")

client = genai.Client(api_key=GEMINI_API_KEY)

def load_discourse_chunks(folder="discourse_json"):
    all_chunks = []
    for file in Path(folder).rglob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        posts = data.get("post_stream", {}).get("posts", [])
        topic_id = data.get("id")
        slug = data.get("slug")
        json_path = str(file.resolve())

        for post in posts:
            content = post.get("cooked") or post.get("raw") or ""
            full_url = f"https://discourse.onlinedegree.iitm.ac.in{post['post_url']}"
            all_chunks.append({
                "id": str(post.get("id")),
                "post_number": post.get("post_number"),
                "content": content,
                "url": full_url,
                "reply_to_post": str(post.get("reply_to_post_number")) if post.get("reply_to_post_number") is not None else "None",
                "topic_id": topic_id,
                "slug": slug,
            })

    return all_chunks

def embed_chunks(chunks, model="models/text-embedding-004", batch_size=32, max_retries=5):
    embedded_chunks = []
    texts = [chunk["content"] for chunk in chunks]
    total_chunks = len(texts)
    print(f"🔢 Total chunks to embed: {total_chunks}")

    success_count = 0
    fail_count = 0

    for i in tqdm(range(0, total_chunks, batch_size), desc="🔁 Embedding chunks"):
        batch = texts[i:i + batch_size]
        retry_count = 0

        while retry_count <= max_retries:
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                )
                embeddings = response["embedding"] if isinstance(response, dict) else response.embeddings

                for j, embedding in enumerate(embeddings):
                    idx = i + j
                    embedding_values=embedding.values if hasattr(embedding, "values") else embedding
                    chunks[idx]["embedding"] = embedding_values.tolist() if hasattr(embedding_values, "tolist") else embedding_values
                    embedded_chunks.append(chunks[idx])
                    print(f"✅ Chunk {idx} embedded")

                success_count += len(embeddings)
                break  # Success
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"❌ Failed to embed batch {i}-{i+len(batch)-1} after {max_retries} retries: {e}")
                    fail_count += len(batch)
                    break
                else:
                    backoff = 2 ** retry_count
                    print(f"⚠️ Error on batch {i}-{i+len(batch)-1}, retry {retry_count}/{max_retries} in {backoff}s: {e}")
                    time.sleep(backoff)

    print(f"\n📊 Embedding complete!")
    print(f"Total chunks: {total_chunks}")
    print(f"Successfully embedded: {success_count}")
    print(f"Failed embeddings: {fail_count}")

    return embedded_chunks

if __name__ == "__main__":
    chunks = load_discourse_chunks()
    embedded = embed_chunks(chunks)

    os.makedirs("scripts", exist_ok=True)
    try:
        with open("scripts/embedded_discourse.json", "w", encoding="utf-8") as f:
            json.dump(list(embedded), f, indent=2)

    except TypeError as e:
        print(f"⚠️ Failed to serialize JSON: {e}")
        import pickle
        with open("scripts/embedded_discourse_backup.pkl", "wb") as f:
            pickle.dump(embedded, f)
        print("✅ Backup saved as Pickle for recovery.")
    print(f"✅ Embedded and saved {len(embedded)} discourse chunks.")

