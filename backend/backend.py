import os
os.environ["HF_HOME"] = os.path.expanduser("~/sgoinfre/huggingface")

from flask import Flask, render_template, request, jsonify # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import fitz  # for PDF
from sentence_transformers import SentenceTransformer # type: ignore

# ✅ ChromaDB imports
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

app = Flask(__name__)

# === Model Setup ===
module_name = "google/flan-t5-base"
generating_tokenizer = AutoTokenizer.from_pretrained(module_name)
generate_response_model = AutoModelForSeq2SeqLM.from_pretrained(module_name)

# === Vector Setup ===
embedding_model_name = "all-MiniLM-L6-v2"
vectoring_model = SentenceTransformer(embedding_model_name)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)


client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="pdf_chunks", embedding_function=embedding_fn)

# === PDF Setup ===
pdf_path = "pdf/Large Dairy Herd Management, 3rd Edition (VetBooks.ir).pdf"

def get_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):

    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def populate_chroma_if_empty():
    # Only encode and insert if Chroma is empty
    if collection.count() == 0:
        print("[INFO] ChromaDB is empty. Populating from PDF...")
        text = get_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
        print(f"[INFO] Added {len(chunks)} chunks to ChromaDB.")
    else:
        print("[INFO] ChromaDB already populated.")

def get_relevant_context_chroma(question):
    results = collection.query(
        query_texts=[question],
        n_results=1,  # You can adjust this number
        include=["distances", "documents"]
    )
    print(f"[INFO] Query results: {results}")
    # Filter results by distance < 1
    filtered = [
        doc for doc, dist in zip(results["documents"][0], results["distances"][0])
        if dist < 0.5
    ]
    if not filtered:
        return None
    return filtered[0]  # Return the most relevant filtered chunk


def generate_response(question, context):
    if not context:
        return "I don't know the answer to that question."

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = generating_tokenizer(prompt, return_tensors="pt")
    outputs = generate_response_model.generate(
        **inputs,
        max_length=900,
        pad_token_id=generating_tokenizer.pad_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    full_response = generating_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated full response: {full_response}")
    return full_response

# === Flask Routes ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data['question']
    context = get_relevant_context_chroma(user_question)
    if context is None:
        return jsonify({'answer': "I don't know the answer to that question."})
    answer = generate_response(user_question, context)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    populate_chroma_if_empty()  # Only run once per server start
    app.run(debug=True)
