# import flask

# app = flask.Flask(__name__)
# @app.route('/ask' )
# def index():

# 	return '<h1>Hello, World!</h1>'
# # print(__name__)

# if __name__ == '__main__':
#     app.run(debug=True)
import flask
from flask import render_template

app = flask.Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = flask.request.get_json()
    question = data['question']
    return flask.jsonify({'answer': f'You asked: {question}'})

if __name__ == '__main__':
    app.run(debug=True)
