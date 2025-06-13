import json
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    # Load both types of embedded chunks
    with open("scripts/embedded_chunks.json", encoding="utf-8") as f:
        markdown_chunks = json.load(f)

    with open("scripts/embedded_discourse.json", encoding="utf-8") as f:
        discourse_chunks = json.load(f)

    all_chunks = markdown_chunks + discourse_chunks

    embeddings = np.array([chunk["embedding"] for chunk in all_chunks], dtype=np.float32)
    metadata = np.array(all_chunks, dtype=object)

    Path("output").mkdir(exist_ok=True)
    np.savez("output/database.npz", embeddings=embeddings, metadata=metadata)

    print(f"✅ NPZ store saved with {len(all_chunks)} chunks from both markdown and discourse.")
