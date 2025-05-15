from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').lower()

    if message == 'hi':
        response = 'hello, Jocel Maria D. Mandigma'
    else:
        response = "I didn't understand that."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
