import numpy as np
import pickle
from flask import Flask, request, render_template

# Create a Flask application

app = Flask(__name__, template_folder='templates')

# Load pickle model

model1 = pickle.load(open('Final_MultiOutputRF_model.pkl', 'rb'))
model2 = pickle.load(open('Final_MultiOutputGB_Model.pkl', 'rb'))


# Create a home page

@app.route('/')
def home():
    return render_template('index.html')

# Create a POST method

@app.route('/predict1', methods = ['POST'])
def predict1():
    '''
    For rendering results on HTML GUI

    '''
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction1 = model1.predict(final_features)
    
    return render_template('index.html', prediction_text1 = 'EC & pH with Random Forest Model should be {}'.format(prediction1))

@app.route('/predict2', methods = ['POST'])
def predict2():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction2 = model2.predict(final_features)
    
    return render_template('index.html', prediction_text2 = 'EC & pH with Gradient Boosting Model should be {}'.format(prediction2))

if __name__ == "__main__":
    app.run(debug = True)

