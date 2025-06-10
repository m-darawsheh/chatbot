import os
os.environ["HF_HOME"] = os.path.expanduser("~/sgoinfre/huggingface")

from flask import Flask, render_template, request, jsonify # type: ignore

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from sklearn.metrics.pairwise import cosine_similarity # type: ignore

app = Flask(__name__)

module_name = "google/flan-t5-base"
generating_tokenizer = AutoTokenizer.from_pretrained(module_name)
generate_response_model = AutoModelForSeq2SeqLM.from_pretrained(module_name)

from sentence_transformers import SentenceTransformer # type: ignore
vectoring_model = SentenceTransformer('all-MiniLM-L6-v2')

ABUYAHYA_KNOWLEDGE = {
    "name": "my name is mohammed abuyahya",
    "age": "I am 20 years old.",
    "location": "I live in amman, Jordan.",
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
    context: {context}
    question: {question}
    answer:"""
    # print("------------------")
    # print(prompt)
    inputs = generating_tokenizer(prompt, return_tensors="pt")
    outputs = generate_response_model.generate(
         **inputs,
        max_length=200,
        pad_token_id=generating_tokenizer.pad_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    full_response = generating_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(full_response)
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
    context = get_relevant_context(user_question, ABUYAHYA_KNOWLEDGE)
    print(context)
    answer = generate_response(user_question, context)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
