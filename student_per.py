import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open("student_lr_final_model.pkl", 'rb') as file:
        model,std,lable = pickle.load(file)
    return model,std,lable

def preprocessing_input_data(data, std, lable):
    data["Extracurricular Activities"] = lable.transform([data["Extracurricular Activities"]])[0]
    df = pd.DataFrame([data])
    df_transformed = std.transform(df)
    return df_transformed

def predict_data(data):
    model,std,lable = load_model()
    processed_data = preprocessing_input_data(data,std,lable)
    prediction = model.predict(processed_data)
    return prediction

def main():
    
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")

    hour_studied = st.number_input('Hours studied', min_value=1, max_value=10, value=5)
    previous_scor = st.number_input('Previous score', min_value=40, max_value=100, value=70)
    extra = st.selectbox("Extra curriculam activity", ['Yes','No'])
    sleeping_hours = st.number_input('Sleeping huors', min_value=4, max_value=10, value=7)
    num_ques = st.number_input('Number of question paper solved', min_value=0, max_value=8, value=5)

    if st.button("predict_your_score"):
        user_data = {
            "Hours Studied":hour_studied,
            "Previous Scores":previous_scor,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleeping_hours,
            "Sample Question Papers Practiced":num_ques
        }

        predict = predict_data(user_data)
        st.success(f"your prediction result is {predict}")


if __name__ == "__main__":
    main()