import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("system_data.csv")

X = df[["Tech:Continuous  Deployment - Our organizations deployment process is fully automated", "Tech:Continuous  Deployment - Our organizations releases are tightly coupled to sprint cycles"]]
#X = X.replace(["Tech:Continuous  Deployment - Our organizations deployment process is fully automated", "Tech:Continuous Integration - Code is checked into a version control system"], [2, 1, 0])

y = df["Deployment Frequency"]

clf = LogisticRegression() 
clf.fit(X, y)

joblib.dump(clf, "survey_clf.pkl")
