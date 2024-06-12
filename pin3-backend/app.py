# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from pytorch_model import get_pytorch_prediction
from fastai_model import get_fastai_prediction, evaluate_and_retrain_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/api/data', methods=['POST'])
def handle_form():
    data = request.json
    ai_model = data.get('aiModel')
    image = data.get('image')

    if image:
        # Assuming the image is base64 encoded
        image_data = image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        if ai_model == 'pytorch':
            prediction = get_pytorch_prediction(image_bytes)
        elif ai_model == 'fastai':
            prediction = get_fastai_prediction(image_bytes)
        else:
            return jsonify({'message': 'Invalid AI model selected'}), 400

        return jsonify({'prediction': prediction})

    return jsonify({'message': 'Invalid request'}), 400


@app.route('/retrain', methods=['POST'])
def retrain():
    accuracy, precision, recall, params = evaluate_and_retrain_model(5, 0.9)
    return jsonify({
        'status': 'Model evaluated and retrained if necessary',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'best_params': params
    }), 200


if __name__ == '__main__':
    app.run(debug=True)
