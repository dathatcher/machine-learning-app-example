import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("system_data.csv")

X = df[["Tech:Continuous Integration - Single code repository", "Tech:Continuous Integration - Code is checked into a version control system", "Tech:Continuous Integration - My team deploys off a single master trunk"]]
X = X.replace(["Tech:Continuous Integration - My team deploys off a single master trunk", "Tech:Continuous Integration - Code is checked into a version control system", "Tech:Continuous Integration - Single code repository"], [2, 1, 0])

y = df["Deployment Frequency"]

clf = LogisticRegression() 
clf.fit(X, y)

joblib.dump(clf, "survey_clf.pkl")
