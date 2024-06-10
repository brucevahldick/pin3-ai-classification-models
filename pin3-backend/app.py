from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/api/data', methods=['POST'])
def handle_form():
    data = request.json  # Assuming the data is sent as JSON
    ai_model = data.get('aiModel')
    image = data.get('image')

    # Process the form data
    # Example: Store the data in the database

    return jsonify({'message': 'Form data received successfully'})


if __name__ == '__main__':
    app.run(debug=True)
