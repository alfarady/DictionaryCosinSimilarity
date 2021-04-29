from logging import log
from flask import Flask, jsonify, render_template, request
from flask_assets import Environment, Bundle
from numpy.testing._private.utils import measure
from core.SemanticMeasure import SemanticMeasure
app = Flask(__name__, template_folder='./views')
measure = SemanticMeasure(verbose=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search/<query>', methods=['POST'])
def predict(query):
    if(query == 'favicon.ico'):
        return jsonify({})

    data = measure.similarity_query(query)
    prediction = {"status": 200, "data": data, "totalData": len(measure.documents)}
    return jsonify(prediction)

@app.route('/scoring', methods=['POST'])
def predict():
    dataForm = request.get_json()

    data = measure.walid_similarity_query(dataForm.answer, dataForm.key)
    prediction = {"status": 200, "data": data}
    return jsonify(prediction)

def main():
    app.run(host="0.0.0.0")

if __name__ == '__main__':
    main()
