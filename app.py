from flask import Flask, request, render_template
import joblib
import pandas as pd

pipeline = joblib.load("pipeline.joblib")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    pclass = int(request.form['Pclass'])
    age = float(request.form['Age'])
    sibsp = int(request.form['SibSp'])
    parch = int(request.form['Parch'])
    fare = float(request.form['Fare'])

    sex = request.form['Sex']
    embarked = request.form['Embarked']

    sex_male = 1 if sex == 'male' else 0
    embarked_q = 1 if embarked == 'Q' else 0
    embarked_s = 1 if embarked == 'S' else 0

    data = pd.DataFrame([{
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Sex_male': sex_male,
        'Embarked_Q': embarked_q,
        'Embarked_S': embarked_s
    }])

    prediction = pipeline.predict(data)

    if prediction[0] == 1:
        result = "Passenger would likely survive"
    else:
        result = "Passenger would likely not survive"

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)