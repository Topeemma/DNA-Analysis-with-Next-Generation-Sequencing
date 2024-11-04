from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load('dna_classifier_model.pkl')

# Define nucleotide encoding
nucleotide_map = {'a': 1, 'c': 2, 'g': 3, 't': 4}

# Preprocessing function for new sequences
def preprocess_sequence(sequence):
    sequence_encoded = [nucleotide_map[nt] for nt in list(sequence.lower())]
    return np.array(sequence_encoded).reshape(1, -1)

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sequence = data.get('sequence', '')

    if len(sequence) == 57:
        processed_seq = preprocess_sequence(sequence)
        prediction = model.predict(processed_seq)
        result = "Promoter" if prediction[0] == 1 else "Non-promoter"
        return jsonify({'prediction': result})
    else:
        return jsonify({'error': 'Invalid sequence length; must be 57 nucleotides.'})

if __name__ == '__main__':
    app.run(debug=True)
