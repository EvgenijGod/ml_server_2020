import json
import os
import pickle
import plotly
import plotly.graph_objs as go
from collections import namedtuple
from flask_wtf import FlaskForm, Form
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect
import numpy as np
import pandas as pd
from forms import RF_Form, GB_Form
from ensembles import GradientBoostingMSE, RandomForestMSE
from sklearn.model_selection import train_test_split

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'

data_path = './../data'
app.config['UPLOAD_FOLDER'] = data_path
Bootstrap(app)


class AlgoStorage():
    def __init__(self):
        self.model_dict = {} #name - model

    def get_names(self):
        return self.model_dict.keys()

    def get_by_name(selfself, name):
        if name in self.model_dict[name]:
            return self.model_dict[name]
        else:
            return 'Wrong name!'

    def insert_model(self, model, name):
        self.model_dict[name] = model


storage = AlgoStorage()


@app.route('/')
@app.route('/main_page')
def func():
    return render_template('main_page.html')


@app.route('/gb_train_page', methods=['GET', 'POST'])
def goto_GB_page():
    # return redirect(url_for('GradientBoostingPage.html'))
    app.logger.info('goto --> gb_train_page.html')
    form = GB_Form(request.form)

    if request.method == 'POST' and form.validate_on_submit():
        return 'Goooood'
    # form =
    return render_template('GradientBoostingPage.html', form=form)


def create_plot(x, y):
    df = pd.DataFrame({'x': x, 'y': y})  # creating a sample dataframe
    data = [
        go.Bar(
            x=df['x'],  # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


@app.route('/rf_train_page', methods=['GET', 'POST'])
def goto_RF_page():
    app.logger.info('goto --> rf_train_page.html')
    form = RF_Form(request.form)

    if request.method == 'POST' and form.validate_on_submit():
        uploaded_file = request.files['train_dataset']

        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'train_data.csv'))

        data = pd.read_csv(data_path + '/train_data.csv')
        x_data = data[data.columns[1:]]
        y_data = data[data.columns[0]]
        X_train, X_test, y_train, y_test = train_test_split(x_data.values, y_data.values.reshape(-1),
                                                            train_size=0.8,
                                                            random_state=241)

        tmp_model = RandomForestMSE(n_estimators=form.n_estimators.data,
                                    max_depth=form.max_depth.data,
                                    feature_subsample_size=form.feature_subsample_size.data,
                                    )
        x, y = tmp_model.fit(X_train, y_train, True, X_test, y_test)
        bar = create_plot(x, y)

        storage.insert_model(tmp_model, 'RandomForest_' + uploaded_file.filename)
        return render_template('train_plot.html', plot=bar)

    return render_template('RandomForestPage.html', form=form)


# @app.route('/plots')
# def show_plot(form):


@app.route('/models_page')
def goto_models_page():
    # return redirect(url_for('models_page.html'))
    app.logger.info('goto --> models_page.htm')
    return render_template('models_page.html')
