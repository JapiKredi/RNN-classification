import streamlit as st
import numpy as pd
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model(
    "/Users/jasper/Desktop/RNN-classification/churn_model.h5"
)

# Load the encoders and the scaler

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("one_hot_encoder_Geography.pkl", "rb") as file:
    one_hot_encoder_Geography = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

## Streamlit app

st.title("Customer Churn Preduction")

# User_Input

geography = st.selectbox("Geography", one_hot_encoder_Geography.categories_[0])
gender = st.selectbox("Gemder", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credot Score")
estimated_salary = st.number("Estimated Salary")
