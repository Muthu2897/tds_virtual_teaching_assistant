# 🎓 Virtual Teaching Assistant (TDS Assistant)

An AI-powered assistant designed to help teachers and learners interact with educational material more efficiently. It answers both **textual and visual queries** using Google Gemini and a custom semantic search system. Perfect for simplifying slide content, diagrams, and complex concepts.

---

## 🚀 Features

- 📚 **Contextual Answers** — Uses a pre-embedded knowledge base to answer with relevant, accurate context.
- 🖼️ **Image-Based Understanding** — Accepts base64-encoded images (e.g., slides or textbook pages) and describes them using Gemini Vision.
- 🌐 **FastAPI Backend** — Lightweight, high-performance API server.
- 🔍 **Semantic Chunk Search** — Embeds and retrieves top relevant passages using Gemini Embeddings.
- 🌎 **Serverless Deployment** — Deployed seamlessly on Vercel for scale and accessibility.
- 🔗 **Helpful Resource Links** — Provides clickable links tied to context snippets.

---

## 🛠️ Tech Stack

| Component              | Technology/Service                      | Purpose                                    |
|------------------------|------------------------------------------|--------------------------------------------|
| 🧠 LLM Backend         | Google Gemini 2.0 Pro / Flash            | Answer generation and image understanding |
| 🔎 Embedding Model     | Gemini `text-embedding-004`              | Semantic vector creation                   |
| 🧮 Vector DB           | NumPy-based in-memory array              | Fast cosine similarity search              |
| 🖼️ Image Processing    | Gemini Vision                            | Text and UI extraction from images         |
| 🧪 API Server          | FastAPI                                  | Handles incoming questions                 |
| ☁️ Deployment          | Vercel                                   | Serverless cloud deployment                |
| 🔐 Env Management      | dotenv                                    | API key handling                           |

---

## 📁 Folder Structure for Vercel
├── rag_api/
│ └── my_app.py # Main FastAPI server script
├── database.npz # Stored embeddings and metadata
├── .env # Environment file (not committed)
├── requirements.txt # Python dependencies
├── README.md # Project documentation


---

## 💻 Setup (Local Development)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/virtual-teaching-assistant.git
cd virtual-teaching-assistant

The .env file has to be setup with the key and API end point

Run command : python -m uvicorn rag_api.my_app:app --reload --host 0.0.0.0 --port 5045
