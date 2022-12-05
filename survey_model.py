import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("survey_data.csv")

X = df[["Tech:Continuous  Deployment - Our organization’s deployment process is fully automated", "Tech:Continuous  Deployment - Our organization’s releases are tightly coupled to sprint cycles"]]
X = X.replace(["STRONGLY AGREE", "SOMEWHAT AGREE","SOMEWHAT DISAGREE","STRONGLY DISAGREE","NEITHER AGREE NOR DISAGREE"], [4, 3, 2, 1, 0])
y = df["On average, across all applications in your business unit, how frequently do you deploy code to your production environment?"]

clf = LogisticRegression() 
clf.fit(X, y)

joblib.dump(clf, "deployment_frequency_clf.pkl")
