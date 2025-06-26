from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

 
model = load_model('crop_recommendation_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        input_scaled = scaler.transform([features])
        prediction = model.predict(input_scaled)
        crop_index = np.argmax(prediction)
        crop_label = encoder.categories_[0][crop_index]
        return render_template('index.html', prediction_text=f"Recommended Crop: {crop_label}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

