import streamlit as st
import pandas as pd
import joblib

# Title
st.header("DevOps Survey Machine Learning App")

# Input bar 1
ci1 = st.number_input("Tech:Continuous  Deployment - Our organizations deployment process is fully automated")

# Input bar 2
ci2 = st.number_input("Tech:Continuous  Deployment - Our organizations releases are tightly coupled to sprint cycles")

# Input bar 3
#ci3 = st.number_input("Tech:Continuous Integration - My team deploys off a single master trunk")

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("survey_clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[ci1, ci2]], 
                     columns = ["Tech:Continuous  Deployment - Our organizations deployment process is fully automated", "Tech:Continuous  Deployment - Our organizations releases are tightly coupled to sprint cycles"])
   # X = X.replace(["Red","Brown", "Blue"], [2, 1, 0])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"Deployment Frequency  {prediction}")
