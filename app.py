import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import joblib  

np.random.seed(42)
tf.random.set_seed(42)

# Flask 앱 생성
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap5(app)

# 폼 정의
class LabForm(FlaskForm):
    longitude = StringField('longitude(1~7)', validators=[DataRequired()])
    latitude = StringField('latitude(1~7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

model = keras.models.load_model("fires_model.keras")
full_pipeline = joblib.load("full_pipeline.pkl")  

# 라우트: 홈(index)
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# 라우트: 예측(prediction)
@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        input_data = {
            'longitude': float(form.longitude.data),
            'latitude': float(form.latitude.data),
            'month': form.month.data,
            'day': form.day.data,
            'avg_temp': float(form.avg_temp.data),
            'max_temp': float(form.max_temp.data),
            'max_wind_speed': float(form.max_wind_speed.data),
            'avg_wind': float(form.avg_wind.data)
        }
        df = pd.DataFrame([input_data])

        df_prepared = full_pipeline.transform(df)

        prediction = model.predict(df_prepared)
        result = np.round(prediction[0][0], 2)

        return render_template('result.html', prediction=result)

    return render_template('prediction.html', form=form)

# 실행
if __name__ == '__main__':
    app.run(debug=True)
