from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def hello_world():
    return 'Hello, Docker!'

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data sent by the client
    data = request.get_json(force=True)

    # Ensure 'input' is in the data
    if 'input' not in data:
        return jsonify({'error': 'No input provided'}), 400

    # Predict using the model
    try:
        prediction = model.predict(np.array(data['input']).reshape(1, -1))
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
