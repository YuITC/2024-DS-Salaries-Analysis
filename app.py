import streamlit as st
import pandas as pd
import joblib
    
st.set_page_config(layout="wide", page_title="Data Science Job Salary Predictor", page_icon="💵")

if __name__ == '__main__':
    model = joblib.load('model/xgb_salary_predictor.pkl')
    df    = pd.read_csv("data/data_model.csv")

    st.title("Data Science Job Salary Predictor 💵")
    st.write("This application allows you to enter job-related features and provides a prediction of the average salary 🤑.")
    st.divider()


    st.subheader("Please fill in what you know about this job 👇")
    col1, col2 = st.columns([4,1])
    with col1:
        st.write(f"**Note: You can leave these sections blank or select 'Unknown' if available.**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            name     = st.selectbox("Company Name", options=[""] + df['Company Name'].unique().tolist())
            location = st.selectbox("Location"    , options=[""] + df['Location'].unique().tolist())
            head     = st.selectbox("Headquarter" , options=[""] + df['Headquarters'].unique().tolist())
        with c2:
            size      = st.selectbox("Company Size"     , options=[""] + df['Size'].unique().tolist())
            ownership = st.selectbox("Type of ownership", options=[""] + df['Type of ownership'].unique().tolist())
            revenue   = st.selectbox("Revenue"          , options=[""] + df['Revenue'].unique().tolist())
        with c3:
            industry  = st.selectbox("Industry" , options=[""] + df['Industry'].unique().tolist())
            sector    = st.selectbox("Sector"   , options=[""] + df['Sector'].unique().tolist())
            job_state = st.selectbox("Job State", options=[""] + df['job_state'].unique().tolist())
        with c4:
            rating    = st.selectbox("Company Rating", options=[""] + df['Rating Category'].unique().tolist())
            job       = st.selectbox("Job Simplified", options=[""] + df['job_simplified'].unique().tolist())
            seniority = st.selectbox("Seniority"     , options=[""] + df['seniority'].unique().tolist())
    with col2:
        st.write(f"**Note: You must select an option.**")
        python = st.selectbox("Python required?", options=["Yes", "No"], index=1)
        spark  = st.selectbox("Spark required?" , options=["Yes", "No"], index=1)
        aws    = st.selectbox("AWS required?"   , options=["Yes", "No"], index=1)


    python = 1 if python == "Yes" else 0
    spark  = 1 if spark  == "Yes" else 0
    aws    = 1 if aws    == "Yes" else 0

    input_df = pd.DataFrame({
        'Company Name'     : [name], 
        'Location'         : [location], 
        'Headquarters'     : [head],
        'Size'             : [size], 
        'Type of ownership': [ownership], 
        'Industry'         : [industry],
        'Sector'           : [sector], 
        'Revenue'          : [revenue], 
        'job_simplified'   : [job],
        'seniority'        : [seniority], 
        'Rating Category'  : [rating], 
        'job_state'        : [job_state],
        'Python_yn'        : [python], 
        'Spark'            : [spark], 
        'AWS_yn'           : [aws]
    })


    st.divider()
    if st.button("**Predict My Salary!**", type='primary'):        
        predicted_salary = model.predict(input_df)[0]
        formatted_salary = f"{predicted_salary:,.0f}".replace(",", " ")
        st.success(f"Your predicted average salary is **${formatted_salary} per year** 💰.")