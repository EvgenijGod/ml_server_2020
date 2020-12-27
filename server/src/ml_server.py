import json
import os
import pickle
import plotly
import plotly.graph_objs as go
from collections import namedtuple
from flask_wtf import FlaskForm, Form
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, send_file
from flask import render_template, redirect
import numpy as np
import pandas as pd
from forms import RF_Form, GB_Form, PredictForm
from ensembles import GradientBoostingMSE, RandomForestMSE
from sklearn.model_selection import train_test_split

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'

data_path = './../data/'
app.config['UPLOAD_FOLDER'] = data_path
app.config['TMP_FOLDER'] = data_path + 'tmp/'
app.config['PREDICT_FOLDER'] = data_path + 'predict/'

Bootstrap(app)


class ModelStorage():
    def __init__(self):
        self.model_dict = {}  # name - model
        self.model_train_set_names = {}
        self.model_plot = {}

    def get_models_list(self):
        return self.model_dict.keys()

    def get_model_by_name(self, name):
        if name in self.model_dict.keys():
            return self.model_dict[name]
        else:
            return 'Wrong name!'

    def get_model_info(self, name):
        if name in self.model_dict.keys():
            ans = self.model_dict[name].params_dict()
            ans['train_dataset'] = self.model_train_set_names[name]
            return ans
        return {}

    def insert_model(self, model, name, path):
        self.model_dict[name] = model
        self.model_train_set_names[name] = path

    def add_model_plot(self, name, bar):
        self.model_plot[name] = bar

    def get_model_plot(self, name):
        return self.model_plot[name]


storage = ModelStorage()


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


@app.route('/')
@app.route('/main_page')
def func():
    return render_template('main_page.html')


@app.route('/gb_train_page', methods=['GET', 'POST'])
def goto_GB_page():
    try:
        # return redirect(url_for('GradientBoostingPage.html'))
        app.logger.info('goto --> gb_train_page.html')
        form = GB_Form()

        if request.method == 'POST' and form.validate_on_submit():
            uploaded_file = request.files['train_dataset']
            if uploaded_file.filename != '':
                uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))
                os.chmod(app.config['UPLOAD_FOLDER'] + uploaded_file.filename, 0o666)
                data = pd.read_csv(app.config['UPLOAD_FOLDER'] + uploaded_file.filename)
                x_data = data[data.columns[1:]]
                y_data = data[data.columns[0]]
                X_train, X_test, y_train, y_test = train_test_split(x_data.values, y_data.values.reshape(-1),
                                                                    train_size=0.8,
                                                                    random_state=241)

                tmp_model = GradientBoostingMSE(n_estimators=form.n_estimators.data,
                                                max_depth=form.max_depth.data,
                                                feature_subsample_size=form.feature_subsample_size.data,
                                                learning_rate=form.learning_rate.data,
                                                )
                x, y = tmp_model.fit(X_train, y_train, True, X_test, y_test)
                bar = create_plot(x, y)
                storage.add_model_plot('GradientBoosting_' + uploaded_file.filename[:-4], bar)
                storage.insert_model(tmp_model, 'GradientBoosting_' + uploaded_file.filename[:-4],
                                     uploaded_file.filename)
                return render_template('train_plot.html', plot=bar)
            else:
                return redirect(url_for('goto_GB_page'))
        return render_template('GradientBoostingPage.html', form=form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/rf_train_page', methods=['GET', 'POST'])
def goto_RF_page():
    try:
        app.logger.info('goto --> rf_train_page.html')
        form = RF_Form()

        if request.method == 'POST' and form.validate_on_submit():
            uploaded_file = request.files['train_dataset']
            if uploaded_file.filename != '':
                uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))
                os.chmod(app.config['UPLOAD_FOLDER'] + uploaded_file.filename, 0o666)
                data = pd.read_csv(app.config['UPLOAD_FOLDER'] + uploaded_file.filename)
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
                storage.add_model_plot('RandomForest_' + uploaded_file.filename[:-4], bar)
                storage.insert_model(tmp_model, 'RandomForest_' + uploaded_file.filename[:-4], uploaded_file.filename)
                return render_template('train_plot.html', plot=bar)
            else:
                return redirect(url_for('goto_RF_page'))

        return render_template('RandomForestPage.html', form=form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

chosen_model_name = ''
bar = None
name = '!'


@app.route('/models_page', methods=["GET", "POST"])
def goto_models_page():
    try:
        global chosen_model_name, bar
        # chosen_model_name = ''
        # return redirect(url_for('models_page.html'))
        app.logger.info('goto --> models_page.htm')
        select = request.form.get('comp_select')
        if select is not None:
            chosen_model_name = str(select)
            bar = storage.get_model_plot(chosen_model_name)

        form = PredictForm()
        # x = np.random.randint(0, 10, 10)
        # y = np.random.randint(0, 10, 10)
        # bar = create_plot(x, y)
        if request.method == 'POST' and form.validate_on_submit():
            model = storage.get_model_by_name(chosen_model_name)
            chosen_model_name = 'DONE'
            uploaded_file = request.files['predict_dataset']
            uploaded_file.save(os.path.join(app.config['TMP_FOLDER'], uploaded_file.filename))
            os.chmod(app.config['TMP_FOLDER'] + uploaded_file.filename, 0o666)
            data = pd.read_csv(app.config['TMP_FOLDER'] + uploaded_file.filename)
            data = np.array(data)
            pred = {'pred': model.predict(data)}
            name = app.config['PREDICT_FOLDER'] + uploaded_file.filename[:-4] + '_predict.csv'
            pd.DataFrame(pred).to_csv(name, index=False)
            os.chmod(name, 0o666)
            send_file(name, as_attachment=True)
            os.remove(app.config['TMP_FOLDER'] + uploaded_file.filename)

        return render_template('models_page.html',
                               params=storage.get_model_info(chosen_model_name),
                               option_list=storage.get_models_list(),
                               chosen=chosen_model_name,
                               form=form,
                               plot=bar)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
