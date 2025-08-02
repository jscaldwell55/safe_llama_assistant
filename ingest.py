import fitz  # PyMuPDF
if not hasattr(fitz, "open"):
    raise ImportError("fitz module is broken — try pip uninstall fitz; pip install pymupdf again.")
from sentence_transformers import SentenceTransformer
import faiss
import os, pickle

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def extract_chunks(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text()
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append((chunk, {"page": i}))
    return chunks

def build_faiss_index(pdf_dir="data", index_path="faiss_index"):
    texts, metadata = [], []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            chunks = extract_chunks(path)
            for text, meta in chunks:
                texts.append(text)
                metadata.append({**meta, "source": file})
    embeddings = EMBEDDING_MODEL.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, f"{index_path}/faiss.index")
    with open(f"{index_path}/metadata.pkl", "wb") as f:
        pickle.dump((texts, metadata), f)

if __name__ == "__main__":
    build_faiss_index()
