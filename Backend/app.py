from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return "Spam Detection Service is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']

    # Preprocess the input message
    transformed_message = vectorizer.transform([message])

    # Make the prediction
    prediction = model.predict(transformed_message)[0]

    # Return the prediction as JSON
    result = "Spam" if prediction == 1 else "Not Spam"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
