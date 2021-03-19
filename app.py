from logging import log
from flask import Flask, jsonify, render_template
from flask_assets import Environment, Bundle
from core.CosinMeasure import CosineMeasure
app = Flask(__name__, template_folder='./views')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<query>')
def predict(query):
    data = measure.make_prediction(query)
    prediction = {"status": 200, "data": data, "totalData": measure.nos_of_documents}
    return jsonify(prediction)

def main():
    global measure
    measure = CosineMeasure()
    measure.prepare_dataset()
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
