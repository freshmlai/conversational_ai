import re
import os
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

# --- Configuration ---
MODEL_NAME_EMBEDDING = "all-MiniLM-L6-v2"
MODEL_NAME_GENERATION = "gpt2"
GPT2_MAX_LENGTH = 1024
CHUNK_SIZES = [100, 400]

# --- 2.1 Data Processing ---
def load_and_chunk_text(file_path, chunk_sizes):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_GENERATION)
    chunks = {}
    tokens = tokenizer.encode(text, add_special_tokens=False)

    for size in chunk_sizes:
        chunks[size] = []
        effective_size = min(size, GPT2_MAX_LENGTH - 50)  # buffer

        for i in range(0, len(tokens), effective_size):
            chunk_tokens = tokens[i:i + effective_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks[size].append({
                "id": f"chunk_{size}_{len(chunks[size])}",
                "text": chunk_text,
                "size": effective_size,
                "source_file": file_path
            })
    return chunks

# --- 2.2 Embedding & Indexing ---
class RAGSystem:
    def __init__(self, embedding_model=MODEL_NAME_EMBEDDING, cache_path="rag_index.pkl"):
        self.embedder = SentenceTransformer(embedding_model)
        self.generator_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_GENERATION)
        self.generator_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_GENERATION)
        self.index = None
        self.sparse_vectorizer = None
        self.sparse_matrix = None
        self.documents = []
        self.cache_path = cache_path

    def add_documents(self, chunks_dict):
        all_chunks = []
        for size in CHUNK_SIZES:
            all_chunks.extend(chunks_dict[size])

        if os.path.exists(self.cache_path):
            print(f"Loading cached RAG index from {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
                self.index = data["index"]
                self.documents = data["documents"]
                self.sparse_vectorizer = data["sparse_vectorizer"]
                self.sparse_matrix = data["sparse_matrix"]
            return

        print("Building new RAG index...")
        self.documents = all_chunks

        # Dense index with progress
        texts = [doc["text"] for doc in all_chunks]
        print(f"Encoding {len(texts)} chunks for dense embeddings...")
        # If too many, show progress every 100
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self.embedder.encode(batch, convert_to_numpy=True)
            embeddings.append(emb)
            print(f"  Encoded {min(i+batch_size, len(texts))}/{len(texts)} chunks...")
        embeddings = np.vstack(embeddings)
        print("Dense embedding complete.")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        # Sparse index (TF-IDF)
        print("Building sparse (TF-IDF) index...")
        self.sparse_vectorizer = TfidfVectorizer()
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(texts)
        print("Sparse index complete.")

        # Save cache
        print("Saving RAG index cache...")
        with open(self.cache_path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "documents": self.documents,
                "sparse_vectorizer": self.sparse_vectorizer,
                "sparse_matrix": self.sparse_matrix
            }, f)
        print("RAG index cache saved.")

    # --- 2.3 Hybrid Retrieval ---
    def preprocess_query(self, query):
        return re.sub(r'[^a-z0-9\s]', '', query.lower())

    def dense_retrieval(self, query, k=5):
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, k)
        return [(self.documents[i], 1 - distances[0][j]) for j, i in enumerate(indices[0])]

    def sparse_retrieval(self, query, k=5):
        query_vec = self.sparse_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.sparse_matrix).flatten()
        top_idx = sims.argsort()[-k:][::-1]
        return [(self.documents[i], sims[i]) for i in top_idx]

    def hybrid_retrieval(self, query, k=3):
        dense_results = self.dense_retrieval(query, k)
        sparse_results = self.sparse_retrieval(query, k)
        combined = {doc["id"]: (doc, score) for doc, score in dense_results}
        for doc, score in sparse_results:
            if doc["id"] in combined:
                combined[doc["id"]] = (doc, combined[doc["id"]][1] + score)
            else:
                combined[doc["id"]] = (doc, score)
        ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    # --- 2.5 Response Generation ---
    def generate_response(self, query, retrieved_passages, max_new_tokens=100):
        context = " ".join([p["text"] for p in retrieved_passages])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        inputs = self.generator_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=GPT2_MAX_LENGTH)
        output = self.generator_model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self.generator_tokenizer.eos_token_id
        )
        response = self.generator_tokenizer.decode(output[0], skip_special_tokens=True)
        return response.split("Answer:")[-1].strip()

    # --- 2.6 Guardrail ---
    def input_validation(self, query):
        if len(query) < 5:
            return False, "Query too short."
        keywords = ["revenue", "profit", "income", "balance sheet", "cash flow", "market share", "dividend", "capex"]
        if not any(k in query.lower() for k in keywords):
            return False, "Query not financial."
        return True, "Valid query."

    def get_confidence_score(self, query, retrieved_passages):
        if not retrieved_passages:
            return 0.0
        query_emb = self.embedder.encode([query])
        passage_emb = self.embedder.encode([retrieved_passages[0]["text"]])
        sim = cosine_similarity(query_emb, passage_emb)[0][0]
        return float(max(0, min(100, sim * 100)))


if __name__ == "__main__":
    # Load and chunk
    chunks_22 = load_and_chunk_text("../data/MM-Annual-Report-2022-23_cleaned.txt", CHUNK_SIZES)
    chunks_23 = load_and_chunk_text("../data/MM-Annual-Report-2023-24_cleaned.txt", CHUNK_SIZES)
    all_chunks = {size: chunks_22[size] + chunks_23[size] for size in CHUNK_SIZES}

    rag = RAGSystem()
    rag.add_documents(all_chunks)

    query = "What was the company's revenue in 2023-24?"
    ok, msg = rag.input_validation(query)
    if ok:
        docs = rag.hybrid_retrieval(query, k=3)
        ans = rag.generate_response(query, docs)
        conf = rag.get_confidence_score(query, docs)
        print("Q:", query)
        print("A:", ans)
        print("Confidence:", conf)
    else:
        print("Invalid:", msg)
