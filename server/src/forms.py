from flask_wtf import FlaskForm, Form
from wtforms.validators import DataRequired, ValidationError
from wtforms import StringField, SubmitField, FileField, SubmitField, BooleanField, StringField, \
    PasswordField, validators, IntegerField


def num_validator(form, field):
    num = field.data
    if num <= 0 or num > 1e6:
        raise ValidationError('Num must be in range 1 ... 1e6')

class GB_Form(Form):
    n_estimators = IntegerField('n_estimators',
        [validators.DataRequired(), num_validator])

    learning_rate = IntegerField('learning_rate',
        [validators.DataRequired(), num_validator])

    max_depth = IntegerField('max_depth',
        [validators.DataRequired(), num_validator])

    feature_subsample_size = IntegerField('feature_subsample_size',
        [validators.DataRequired(), num_validator])

    trees_parameters = StringField('trees_parameters',
        [validators.DataRequired()])

    submit = SubmitField('Create and train')

class RF_Form(Form):
    n_estimators = IntegerField('n_estimators',
        [validators.DataRequired(), num_validator])

    max_depth = IntegerField('max_depth',
        [validators.DataRequired(), num_validator])

    feature_subsample_size = IntegerField('feature_subsample_size',
        [validators.DataRequired(), num_validator])

    trees_parameters = StringField('trees_parameters',
        [validators.DataRequired()])

    submit = SubmitField('Create and train')