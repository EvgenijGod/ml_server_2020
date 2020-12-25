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

from forms import RF_Form, GB_Form


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'

data_path = './../data'
app.config['UPLOAD_FOLDER'] = data_path
Bootstrap(app)



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
def create_plot():
    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
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
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))

        bar = create_plot()
        return render_template('train.plot.html', plot=bar)

    return render_template('RandomForestPage.html', form=form)



# @app.route('/plots')
# def show_plot(form):


@app.route('/models_page')
def goto_models_page():
    # return redirect(url_for('models_page.html'))
    app.logger.info('goto --> models_page.htm')
    return render_template('models_page.html')
