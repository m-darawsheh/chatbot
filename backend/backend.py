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

def chunk_text(text, max_chunk_size=350):
    # Split text into paragraphs using double newlines
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        words = para.split()
        word_count = len(words)

        # If paragraph is too large, split it
        if word_count > max_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            # Split large paragraph into sub-chunks
            for i in range(0, word_count, max_chunk_size):
                chunk = " ".join(words[i:i + max_chunk_size])
                chunks.append(chunk)
            continue

        # Add paragraph to current chunk if it fits
        if current_size + word_count <= max_chunk_size:
            current_chunk.append(para)
            current_size += word_count
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_size = word_count

    # Add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    new_chunks = []
    overlap_size = 50
    for i in range(len(chunks)):
        if i == 0:
            new_chunks.append(chunks[i])
        else:
            # Add overlap with the previous chunk
            overlap = " ".join(chunks[i-1].split()[-overlap_size:])
            new_chunk = overlap + " " + chunks[i]
            new_chunks.append(new_chunk)
    if len(new_chunks) > 1:
        # Ensure the last chunk has no overlap
        new_chunks[-1] = new_chunks[-1].split()[:max_chunk_size]

    return new_chunks


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
        n_results=5,  # Increased from 1 to get more results for debugging
        include=["distances", "documents"]
    )
    # print(f"documents: {results['documents'][0]} \n")
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"\nDistance: {dist}")
    # Filter results by distance
    filtered = [
        doc for doc, dist in zip(results["documents"][0], results["distances"][0])
        # if dist < 1.7
    ]

    if not filtered:
        filtered = documents[:2]

    print(f"Context found: {len(filtered)} relevant chunks.")

    def score_chunk(chunk):
        q_words = set(question.lower().split())
        c_words = set(chunk.lower().split())
        return len(q_words & c_words)

    filtered.sort(key=score_chunk, reverse=True)

    # Return merged context
    combined_context = " ".join(filtered[:2])  # or more
    return combined_context



def generate_response(question, context):
    if not context:
        return "I don't know the answer to that question."

    prompt = f"""Based on the following context, provide a detailed and accurate answer to the question.
    If the context doesn't contain enough information to fully answer the question, say so.

    Context: {context}

    Question: {question}

    Detailed Answer:"""

    inputs = generating_tokenizer(prompt, return_tensors="pt")
    outputs = generate_response_model.generate(
        **inputs,
        max_length=900,
        pad_token_id=generating_tokenizer.pad_token_id,
        do_sample=True,
        top_p=0.92,
        temperature=0.7,  # Lowered for more focused responses
        repetition_penalty=1.2  # Prevent repetitive text
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

# @app.route('/ask_web', methods=['POST'])
# def ask_web():
# 	data = request.get_json()
# 	user_question = data['question']
# 	context = get_relevant_context_chroma(user_question)
# 	if context is None:
# 		return jsonify({'answer': "I don't know the answer to that question."})
# 	answer = generate_response(user_question, context)
# 	return jsonify({'answer': answer})

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
