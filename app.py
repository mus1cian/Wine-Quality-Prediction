from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

def prediction(data):
    clf = joblib.load("my_model.pkl")
    full_pipeline = get_pipieline()
    data_prepared = full_pipeline.transform(data)
    return clf.predict(data_prepared)

def num_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

def get_pipieline():
    red_wine = pd.read_csv("datasets/wine_quality.csv")
    red_wine = red_wine.drop('quality', axis = 1)

    full_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
    ])

    full_pipeline.fit(red_wine)

    return full_pipeline

st.title("Predicting Wine Quality")
st.markdown("Model to classify wine quality based on wine properties")

col1, col2 = st.columns(2)
with col1:
    fixed_acidity = st.number_input("Fixed acidity", value = 7.5, min_value = 0.0)
    volatile_acidity = st.number_input("Volatile acidity", value = 0.52, min_value = 0.0)
    citric_acid = st.number_input("Citric acid", value = 0.16, min_value = 0.0, max_value = 1.0)
    residual_sugar = st.number_input("Residual sugar ", value = 1.9, min_value = 0.0)
    chlorides = st.number_input("Chlorides", value = 85.0, min_value = 0.0)
    free_sulfur_dioxide = st.number_input("Free sulfur dioxide", value = 12.0, min_value = 0.0)
    
with col2:
    
    total_sulfur_dioxide = st.number_input("Total sulfur dioxide", value = 35.0, min_value = 0.0)
    density = st.number_input("Density", value = 0.9968, min_value = 0.0)
    pH = st.number_input("PH", value = 3.38, min_value = 0.0, max_value = 14.0)
    sulphates = st.number_input("Sulphates", value = 0.62, min_value = 0.0)
    alcohol = st.number_input("Alcohol", value = 9.5, min_value = 0.0)

if st.button("Predict red wine quality"):  
    data = pd.DataFrame({
            'fixed acidity': [fixed_acidity],
            'volatile acidity': [volatile_acidity],
            'citric acid': [citric_acid],
            'residual sugar': [residual_sugar],
            'chlorides': [chlorides],
            'free sulfur dioxide': [free_sulfur_dioxide],
            'total sulfur dioxide': [total_sulfur_dioxide],
            'density': [density],
            'pH': [pH],
            'sulphates': [sulphates],
            'alcohol': [alcohol]
        }) 
    result = prediction(data)
    quality = ""
    if result == 0:
        quality = "Bad quality"
    elif result == 1:
        quality = "Regular quality"
    elif result == 2:
        quality = "Good quality"
    st.text(quality)
    st.balloons()