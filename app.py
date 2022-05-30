import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('./model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    print('#####: request.form.values()', request.form.keys())

    int_features = []

    for x in request.form.values():
        print(x)
        int_features.append(int(x))

    int_features[0] = float(int_features[0])
    int_features[3] = float(int_features[3])
    final_features = [np.array(int_features)]
    heigth = int_features[2]
    weight = int_features[3]
    int_features.append(weight/((heigth/100)**2))

    print('#####: int_features', int_features)
    print('#####: final_features', final_features)
    prediction = model.predict([int_features])

    output = round(float(prediction[0]), 2)
    print('####: prediction', prediction)
    prediction_text = ''

    if output < 0.5:
        prediction_text = 'Вы не имеете ССЗ с вероятностью: {}%'.format(int((1 - output)*100))
    else:
        prediction_text = 'Вы имеете ССЗ с вероятностью: {}%'.format(int(output*100))

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
