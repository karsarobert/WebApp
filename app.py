import flask
import pandas as pd
from joblib import dump, load


with open(f'model/carpriceprediction.joblib', 'rb') as f:
    model = load(f)


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        Year = flask.request.form['year']
        Kilometers_Driven = flask.request.form['kilometer']
        Seats = flask.request.form['seats']
        
        input_variables = pd.DataFrame([[Year, Kilometers_Driven, Seats]],
                                       columns=['Year', 'Kilometers_Driven', 'Seats'],
                                       dtype='float',
                                       index=['input'])

        predictions = model.predict(input_variables)[0]
        print(predictions)

        return flask.render_template('main.html', original_input={'Évjárat': Year, 'Futott kilométerek': Kilometers_Driven, 'Ülések száma': Seats},
                                     result=predictions)


if __name__ == '__main__':
    app.run(debug=True)