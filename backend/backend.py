# import flask

# app = flask.Flask(__name__)
# @app.route('/ask' )
# def index():

# 	return '<h1>Hello, World!</h1>'
# # print(__name__)

# if __name__ == '__main__':
#     app.run(debug=True)

import flask

app = flask.Flask(__name__)

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    return '<h1>Hello, World!</h1>'

if __name__ == '__main__':
    app.run(debug=True)
