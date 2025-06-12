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
		max_length=200,
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
	context = get_relevant_context(user_question, ABUYAHYA_KNOWLEDGE)
	if context and cosine_similarity([vectoring_model.encode(user_question)], [vectoring_model.encode(context)])[0][0] > 0.8:
		answer = context
	else:
		answer = generate_response(user_question, context)
	return jsonify({'answer': answer})

if __name__ == '__main__':
	app.run(debug=True)
