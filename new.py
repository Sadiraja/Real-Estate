import streamlit as st
import numpy as np
import joblib as jb

scaler=jb.load("scaler.pkl")

model=jb.load("model.pkl")

st.title("Real Estate Price Prediction")

st.divider()


bed =st.number_input("Enter Number of bedrooms",value=2,step=1)
bath =st.number_input("Enter Number of Bathrooms",value=2,step=1)
size =st.number_input("Enter the size",value=1000,step=50)

X=[bed,bath,size]

predicts =st.button("Predict")

if predicts:
   X1 =np.array(X)
   X_array=scaler.transform([X1])
   prdediction=model.predict(X_array)[0]

   st.write(f"Predicted value={prdediction:.2f}")
else:
   st.write("Press the button")