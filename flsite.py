import pickle
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify, Response, json

app = Flask(__name__)

gender_dict = {
    0: "Мужской",
    1: "Женский"
}

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"}]

loaded_model_knn = pickle.load(open('model/Iris_pickle_fileKNN', 'rb'))
loaded_model_Log = pickle.load(open('model2/Iris_pickle_file', 'rb'))
loaded_model_Tree = pickle.load(open('model3/Iris_pickle_fileTREE', 'rb'))


metrics_classification_data = [
{"name": "Confusion matrix", "type": "logistic", "value": {"Predicted": [2, 1, 1, 37.4,19.4,178,3900,2007], "Actual": [2, 1, 1, 37.8,19.2,179,3910,2007]}},
{"name": "Confusion matrix", "type": "knn", "value": {"Predicted": [3, 2, 2, 39.8,21.7,182,4100,2007], "Actual": [3, 2, 2, 38.4,18.4,182,4000,2007]}},
{"name": "Confusion matrix", "type": "Tree", "value": {"Predicted": [6, 1, 1, 41.4,23.3,176,4200,2007], "Actual": [5, 1, 1, 41.6,23.4,180,4150,2007]}},
{"name": "Accuracy", "type": "logistic", "value": 0.8611111111111112},
{"name": "Accuracy", "type": "KNN", "value": 0.8333333333333334},
{"name": "Accuracy", "type": "Tree", "value": 0.7833333333333333},
{"name": "Precision", "type": "logistic", "value": 0.845360824742268},
{"name": "Precision", "type": "KNN", "value": 0.8297872340425532},
{"name": "Precision", "type": "Tree", "value": 0.797752808988764},
{"name": "Recall", "type": "logistic", "value": 0.8913043478260869},
{"name": "Recall", "type": "KNN", "value": 0.8478260869565217},
{"name": "Recall", "type": "Tree", "value": 0.7717391304347826},
]

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Игнатьев Артем Андреевич", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='', parametr=metrics_classification_data)
    if request.method == 'POST':
        X_new = np.array([[int(request.form['list1']),
                           int(request.form['list2']),
                           int(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           int(request.form['list6']),
                           int(request.form['list7']),
                           int(request.form['list8'])]])
        pred = loaded_model_knn.predict(X_new)
        gender = gender_dict[pred[0]]
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + gender, parametr = metrics_classification_data )

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,parametr = metrics_classification_data )
    if request.method == 'POST':
        X_new = np.array([[int(request.form['list1']),
                           int(request.form['list2']),
                           int(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           int(request.form['list6']),
                           int(request.form['list7']),
                           int(request.form['list8'])]])
        pred = loaded_model_Log.predict(X_new)
        gender = gender_dict[pred[0]]
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + gender, parametr = metrics_classification_data  )

@app.route("/p_lab3")
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Дерево решений", menu=menu,parametr = metrics_classification_data )
    if request.method == 'POST':
        X_new = np.array([[int(request.form['list1']),
                           int(request.form['list2']),
                           int(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           int(request.form['list6']),
                           int(request.form['list7']),
                           int(request.form['list8'])]])
        pred = loaded_model_Tree.predict(X_new)
        gender = gender_dict[pred[0]]
        return render_template('lab3.html', title="Дерево решений", menu=menu,
                               class_model="Это: " + gender, parametr = metrics_classification_data )

@app.route('/api_v2',methods=['get'])
def get_sort_v2():
    X_new = np.array([[int(request.args.get('list1')),
                       int(request.args.get('list2')),
                       int(request.args.get('list3')),
                       float(request.args.get('list4')),
                       float(request.args.get('list5')),
                       int(request.args.get('list6')),
                       int(request.args.get('list7')),
                       int(request.args.get('list8'))]])
    pred = loaded_model_knn.predict(X_new)
    gender = gender_dict[pred[0]]
    response = Response(response=json.dumps({'gender': gender}, ensure_ascii=False).encode('utf8'),
                        status=200,
                        mimetype='application/json; charset=utf-8')

    return response

if __name__ == "__main__":
    app.run(debug=True)
