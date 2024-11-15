from flask import Flask

app = Flask(__name__)


@app.route('/')
def root():
    return 'Bienvenu sur cette API qui prédit les tags de vos questions Stackoverflow'


@app.route('/predict_tags/<string:question>')
def predict_tags(quesiton):
    return result


if __name__ == '__main__':
    app.run(host="0.0.0.0")