from flask import Flask, request, jsonify
import pickle

app = Flask('Credit')

with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)
    
with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5
    
    result = {
        'probability': float(prediction),
        'classification': bool(churn),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run()