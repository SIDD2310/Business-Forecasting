from flask import Flask, jsonify, render_template, request, redirect
import joblib
import json
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route("/")
def pred_page():
    with open('templates\\prediction.txt', 'r') as f:
        text = f.read()
    return render_template("prediction.html",text = text)

@app.route('/predict',methods=['POST','GET'])
def result():

    item_weight= float(request.form['item_weight'])
    item_fat_content=float(request.form['item_fat_content'])
    item_visibility= float(request.form['item_visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year= float(request.form['outlet_establishment_year'])
    outlet_size= float(request.form['outlet_size'])
    outlet_location_type= float(request.form['outlet_location_type'])
    outlet_type= float(request.form['outlet_type'])

    X= np.array([[ item_weight,item_fat_content,item_visibility,item_type,item_mrp,
                  outlet_establishment_year,outlet_size,outlet_location_type,outlet_type ]])

    scaler_path = 'C:\\Users\\siddh\\Design Project\\models\\sc.sav'

    sc=joblib.load(scaler_path)

    X_std= sc.transform(X)

    model_path = 'C:\\Users\\siddh\\Design Project\\models\\rf.sav'

    model= joblib.load(model_path)

    Y_pred=model.predict(X_std)

    Y_pred_print = str(Y_pred[0])

    #Y_pred_print = np.array_str(Y_pred)

    with open('templates\\prediction.txt', 'w') as f:
        f.write(Y_pred_print)

    return pred_page()


if __name__ == "__main__":
    app.run(debug=True, port=9457)