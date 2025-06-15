import os
os.environ["HF_HOME"] = os.path.expanduser("~/sgoinfre/huggingface")

from flask import Flask, render_template, request, jsonify # type: ignore

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from sklearn.metrics.pairwise import cosine_similarity # type: ignore

import fitz #type: ignore 

app = Flask(__name__)

module_name = "meta-llama/Llama-3-8B-Instruct"
generating_tokenizer = AutoTokenizer.from_pretrained(module_name)
generate_response_model = AutoModelForSeq2SeqLM.from_pretrained(module_name)

from sentence_transformers import SentenceTransformer # type: ignore
vectoring_model = SentenceTransformer('all-MiniLM-L6-v2')
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

def get_relevant_context(question, knowledge):
	"""
	Find the most relevant context from the knowledge base for a given question.
	Returns the best matching knowledge value, or None if no good match is found.
	"""
	if not question or not knowledge:
		return None

	question_vec = vectoring_model.encode(question)
	best_key = None
	best_score = -1

	for key, value in knowledge.items():
		value_vec = vectoring_model.encode(str(value))
		score = cosine_similarity([question_vec], [value_vec])[0][0]
		if score > best_score:
			best_score = score
			best_key = key

	print(f"Best key: {best_key}, Score: {best_score}")
	if best_score < 0.3:
		return None
	return knowledge[best_key]

def get_relevant_context_pdf(question ,pdf_chunks_vectors):
	"""
	Find the most relevant context from the PDF chunks for a given question.
	Returns the best matching PDF chunk, or None if no good match is found.
	"""
	if not question or not pdf_chunks_vectors:
		return None

	question_vec = vectoring_model.encode(question)
	best_index = -1
	best_score = -1

	for i, chunk_vec in enumerate(pdf_chunks_vectors):
		score = cosine_similarity([question_vec], [chunk_vec])[0][0]
		if score > best_score:
			best_score = score
			best_index = i

	print(f"Best index: {best_index}, Score: {best_score}")
	if best_score < 0.3:
		return None
	return best_index

def generate_response(question, context):
	"""
	Generate a conversational response using the provided context.
	Returns a string response.
	"""
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
	text = get_text_from_pdf(pdf_path)
	chunks = chunk_text(text)
	pdf_chunks_vectors = [vectoring_model.encode(chunk) for chunk in chunks]
	chunk_index = get_relevant_context_pdf(user_question, pdf_chunks_vectors)
	if chunk_index is None:
		return jsonify({'answer': "I don't know the answer to that question."})
	answer = generate_response(user_question, chunks[chunk_index])
	return jsonify({'answer': answer})

if __name__ == '__main__':
	app.run(debug=True)
