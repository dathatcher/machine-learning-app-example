import streamlit as st
import pandas as pd
import joblib

# Title
st.header("IT Service Machine Learning App")

# Input bar 1
cpu = st.number_input("Enter CPU%")

# Input bar 2
trans = st.number_input("Enter Transaction Volume")

# Input bar 3
trans_speed = st.number_input("Enter avg Transaction speed")

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("service_clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[cpu, trans, trans_speed]], 
                     columns = ["CPU", "Trans", "Trans Speed"])
   # X = X.replace(["Red","Brown", "Blue"], [2, 1, 0])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"This instance is a {prediction}")
