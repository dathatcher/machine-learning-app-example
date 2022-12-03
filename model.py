import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

X = df[["Height", "Weight", "Eye"]]
X = X.replace(["Red", "Brown", "Blue"], [2, 1, 0])

y = df["Species"]

clf = LogisticRegression() 
clf.fit(X, y)

joblib.dump(clf, "clf.pkl")
