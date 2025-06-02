import os
os.environ["HF_HOME"] = "/sgoinfre/mabuyahy/huggingface"
import flask
from flask import render_template
from transformers import AutoTokenizer, AutoModelForCausalLM

app = flask.Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/chat')
def chat():
	return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = flask.request.get_json()
    question = data['question']
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return flask.jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
