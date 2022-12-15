from flask import Flask, request, jsonify
import numpy as np
import pickle

prediction_model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return "This is the Home page"

@app.route('/predict',methods=['GET'])
def predict():
    soil_nitrogen = request.args.get('soilN')
    soil_nitrogen1 = float(soil_nitrogen)
    soil_phosphorus = request.args.get('soilP')
    soil_phosphorus1 = float(soil_phosphorus)
    soil_potassium = request.args.get('soilK')
    soil_potassium1 = float(soil_potassium)
    atmospheric_temp = request.args.get('atmosphericTemp')
    atmospheric_temp1 = float(atmospheric_temp)
    atmospheric_humidity = request.args.get('atmosphericHumidity')
    atmospheric_humidity1 = float(atmospheric_humidity)
    soil_ph = request.args.get('soilPH')
    soil_ph1 = float(soil_ph)

    input_query = np.array([[soil_nitrogen1, soil_phosphorus1, soil_potassium1, atmospheric_temp1, atmospheric_humidity1, soil_ph1]])
    result = prediction_model.predict_proba(input_query)[0]
    return jsonify({'predicted_crops': result.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# List of all the crops
# ['apple' 'banana' 'blackgram' 'chickpea' 'coconut' 'coffee' 'cotton'
#  'grapes' 'jute' 'kidneybeans' 'lentil' 'maize' 'mango' 'mothbeans'
#  'mungbean' 'muskmelon' 'orange' 'papaya' 'pigeonpeas' 'pomegranate'
#  'rice' 'watermelon']
