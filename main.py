from flask import Flask
from utils import *

app = Flask(_name_)


@app.route('/')
def root():
    return 'Bienvenu sur cette API qui prédit les tags de vos questions Stackoverflow'


@app.route('/predict_tags/<string:question>')
def api_predict_tags(question):
    return predict_tags(question)


if _name_ == '_main_':
    app.run(host="0.0.0.0")