from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
from infer import Inference

app = Flask(__name__)
CORS(app)
inf = Inference()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        base64_string = request.form['img_base64']
        base64_string = base64_string.replace("data:image/jpeg;base64,","")
        pred, conf = inf.predict(base64_string)
        conf = str(round((conf*100),2)) + "%"
        result = {"predict": pred,"confidence": conf}
        print(result)
        return jsonify(isError = False,
                    message= "Success",
                    statusCode= 200,
                    data = result), 200

if __name__ == '__main__':
    app.run(threaded=False, debug=False, port=3006)