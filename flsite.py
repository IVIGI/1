import pickle
import numpy as np
from flask import Flask, render_template, url_for, request

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


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Игнатьев Артем Андреевич", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
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
                               class_model="Это: " + gender)

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu)
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
                               class_model="Это: " + gender)

@app.route("/p_lab3")
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Дерево решений", menu=menu)
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
                               class_model="Это: " + gender)

if __name__ == "__main__":
    app.run(debug=True)
