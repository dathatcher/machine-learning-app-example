import streamlit as st
import pandas as pd
import joblib

# Title
st.header("DevOps Survey Machine Learning App")

# Input bar 1
#ci1 = st.number_input("Tech:Continuous  Deployment - Our organizations deployment process is fully automated")
ci1 = st.selectbox("Tech:Continuous  Deployment - Our organizations deployment process is fully automated", ("STRONGLY AGREE", "STRONGLY AGREE", "SOMEWHAT DISAGREE","STRONGLY DISAGREE","NEITHER AGREE NOR DISAGREE"))

ci2 = st.selectbox("Tech:Continuous  Deployment - Our organizations releases are tightly coupled to sprint cycles", ("STRONGLY AGREE", "STRONGLY AGREE", "SOMEWHAT DISAGREE","STRONGLY DISAGREE","NEITHER AGREE NOR DISAGREE") )

# Input bar 3
#ci3 = st.number_input("Tech:Continuous Integration - My team deploys off a single master trunk")

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("deployment_frequency_clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[ci1, ci2]], 
                     columns = ["Tech:Continuous  Deployment - Our organization’s deployment process is fully automated", "Tech:Continuous  Deployment - Our organization’s releases are tightly coupled to sprint cycles"])
    X = X.replace(["STRONGLY AGREE", "SOMEWHAT AGREE","SOMEWHAT DISAGREE","STRONGLY DISAGREE","NEITHER AGREE NOR DISAGREE"], [4, 3, 2, 1, 0])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"Deployment Frequency?  {prediction}")
