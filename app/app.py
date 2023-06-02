from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)


def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


model = load_model()


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        year = int(request.form['year'])
        status = int(request.form['status'])
        alcohol = float(request.form['alcohol'])
        adult_mortality = float(request.form['adult_mortality'])
        hepatitis_b = float(request.form['hepatitis_b'])
        measles = int(request.form['measles'])
        bmi = float(request.form['bmi'])
        under_five_deaths = int(request.form['under_five_deaths'])
        polio = float(request.form['polio'])
        total_expenditure = float(request.form['total_expenditure'])
        diphtheria = float(request.form['diphtheria'])
        hiv_aids = float(request.form['hiv_aids'])
        gdp = float(request.form['gdp'])
        population = float(request.form['population'])
        thinness_1_19_years = float(request.form['thinness_1_19_years'])
        income_composition = float(request.form['income_composition'])
        schooling = float(request.form['schooling'])

        # Prepare the input data for prediction
        input_data = np.array([[year, status, alcohol, adult_mortality, hepatitis_b, measles, bmi, under_five_deaths,
                                polio, total_expenditure, diphtheria, hiv_aids, gdp, population, thinness_1_19_years,
                                income_composition, schooling]])

        # Use the loaded model to make the prediction
        prediction = model.predict(input_data)

        # Redirect to the result page with the prediction
        return redirect(url_for('result', prediction=float(prediction[0])))

    # If it's a GET request, render the index page
    return render_template('index.html')


@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
