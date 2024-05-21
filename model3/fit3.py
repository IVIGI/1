import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

iris_df = pd.read_csv("penguins3.csv")
label_encoder = LabelEncoder()
iris_df["island"] = label_encoder.fit_transform(iris_df["island"])
iris_df["species"] = label_encoder.fit_transform(iris_df["species"])
X = iris_df.drop(["sex"], axis=1)
Y = iris_df["sex"]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.2, random_state=3)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train1, Y_train1)
predictions = model.predict(X_test1)
accuracy = accuracy_score(Y_test1, predictions)
class_names = iris_df['sex']
with open('Iris_pickle_fileTREE.pkl', 'wb') as pkl:
    pickle.dump(model, pkl)
