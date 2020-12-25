import os
import pickle

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
Bootstrap(app)
messages = []



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


@app.route('/rf_train_page', methods=['GET', 'POST'])
def goto_RF_page():
    # return redirect(url_for('GradientBoostingPage.html'))
    app.logger.info('goto --> rf_train_page.html')
    form = RF_Form(request.form)

    if request.method == 'POST' and form.validate_on_submit():
        return 'Goooood'
    return render_template('RandomForestPage.html', form=form)


@app.route('/models_page')
def goto_models_page():
    # return redirect(url_for('models_page.html'))
    app.logger.info('goto --> models_page.htm')
    return render_template('models_page.html')
