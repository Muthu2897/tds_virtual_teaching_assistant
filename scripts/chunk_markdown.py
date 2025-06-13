from pathlib import Path
import markdown
import re
import json
import yaml

def load_and_chunk_markdown(file_path, chunk_size=300, overlap=50):
    text = Path(file_path).read_text(encoding='utf-8')
    html = markdown.markdown(text)
    clean_text = re.sub(r'<[^>]+>', '', html)  # strip HTML tags
    words = clean_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_all_files(folder_path):
    data = []
    for path in Path(folder_path).glob("*.md"):
        text = Path(path).read_text(encoding="utf-8")

        # Extract YAML frontmatter (between --- and ---)
        if text.startswith('---'):
            parts = text.split('---', 2)
            if len(parts) > 2:
                try:
                    metadata = yaml.safe_load(parts[1])
                    original_url = metadata.get("original_url", None)
                except Exception as e:
                    print(f"YAML parse error in {path.name}: {e}")
                    original_url = None
            else:
                original_url = None
        else:
            original_url = None

        chunks = load_and_chunk_markdown(path)
        for i, chunk in enumerate(chunks):
            data.append({
                "source": str(path.name),
                "chunk_id": i,
                "content": chunk,
                "url": original_url  # <- finally adding the correct URL here
            })
    return data

if __name__ == "__main__":
    folder = "tds_pages_md"
    output_file = "scripts/chunks.json"
    all_chunks = process_all_files(folder)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"✅ Saved {len(all_chunks)} chunks to {output_file}")
