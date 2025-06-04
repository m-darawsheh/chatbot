import os
os.environ["HF_HOME"] = os.path.expanduser("~/sgoinfre/huggingface")

from flask import Flask, render_template, request, jsonify # type: ignore

from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

app = Flask(__name__)

generating_tokenizer = AutoTokenizer.from_pretrained("gpt2")
generate_response_model = AutoModelForCausalLM.from_pretrained("gpt2")

from sentence_transformers import SentenceTransformer # type: ignore
vectoring_model = SentenceTransformer('all-MiniLM-L6-v2')

COMPANY_KNOWLEDGE = {
    "about": "AI Solutions Inc. provides AI-powered business solutions.",
    "contact": "You can contact us at contact@aisolutions.com.",
    "services": "We offer NLP, computer vision, and data analytics services."
}

def get_relevant_context(question, knowledge):
    question_vector = vectoring_model.encode(question)
    knowledge_vectors = {k: vectoring_model.encode(str(v)) for k,v in knowledge.items()}
    similarities = {}
    for key, vector in knowledge_vectors.items():
        similarities[key] = cosine_similarity([question_vector], [vector])[0][0]
    best_match = max(similarities, key=similarities.get)
    return knowledge[best_match]

def generate_response(question, context):
    """Generate conversational response using context"""
    prompt = f"""
    [System] You're a helpful support assistant for AI Solutions Inc.
    Use this context to answer questions:
    {context}
    [User] {question}
    [Assistant]"""
    inputs = generating_tokenizer(prompt, return_tensors="pt")
    outputs = generate_response_model.generate(
        inputs.input_ids,
        max_length=200,
        pad_token_id=generating_tokenizer.eos_token_id,
        temperature=0.7
    )
    full_response = generating_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_response.split("[Assistant]")[-1].strip()

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
    context = get_relevant_context(user_question, COMPANY_KNOWLEDGE)
    answer = generate_response(user_question, context)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
