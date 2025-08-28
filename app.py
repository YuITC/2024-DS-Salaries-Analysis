"""
Data Science Job Salary Prediction Streamlit Application

This application allows users to predict data science job salaries using either:
1. Machine Learning approach (XGBoost model)
2. Fine-tuned LLM approach (Llama 3.1 8B Instruct LoRA SFT)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from typing import Optional, Dict, Any

# Page configuration
st.set_page_config(
    page_title="Data Science Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_FILE = "data/data_model.csv"
MODEL_FILE = "outputs/optuna_xgboost/xgb_optuna_tuning.pkl"
TUNED_MODEL = "YuITC/llama31-8b-ins-qlora-sft"

@st.cache_data
def load_data():
    """Load the reference data for getting default values and unique options."""
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_ml_model():
    """Load the trained XGBoost pipeline."""
    return joblib.load(MODEL_FILE)

@st.cache_resource
def load_llm_model():
    """Load the fine-tuned LLM model."""
    try:
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from peft import AutoPeftModelForCausalLM
        
        # Load tokenizer
        tuned_tokenizer = AutoTokenizer.from_pretrained(TUNED_MODEL, use_fast=True)
        if tuned_tokenizer.pad_token is None:
            tuned_tokenizer.pad_token = tuned_tokenizer.eos_token
        tuned_tokenizer.padding_side = 'right'
        
        # Load fine-tuned model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        tuned_model = AutoPeftModelForCausalLM.from_pretrained(
            TUNED_MODEL, 
            device_map='auto', 
            torch_dtype=torch.float16, 
            quantization_config=quantization_config
        )
        tuned_model.eval()
        tuned_model.config.use_cache = True
        
        return tuned_model, tuned_tokenizer
    except Exception as e:
        st.error(f"Error loading LLM model: {str(e)}")
        return None, None

def get_default_values(data: pd.DataFrame) -> Dict[str, Any]:
    """Get default values (mode for categorical, mean for numerical) for missing inputs."""
    defaults = {}
    
    # Categorical columns - use mode
    categorical_cols = ['Company Name', 'Location', 'Headquarters', 'Size', 'Type of ownership', 
                       'Industry', 'Sector', 'Revenue', 'job_simplified', 'seniority', 
                       'Rating Category', 'job_state']
     
    for col in categorical_cols:
        defaults[col] = data[col].mode().iloc[0] if not data[col].mode().empty else "Unknown"
    
    # Numerical columns - use mean
    defaults['Python_yn'] = int(data['Python_yn'].mean().round())
    defaults['Spark'] = int(data['Spark'].mean().round())
    defaults['AWS_yn'] = int(data['AWS_yn'].mean().round())
    
    return defaults

def create_input_form(data: pd.DataFrame) -> Dict[str, Any]:
    """Create the input form for user to enter job details."""
    st.header("📝 Enter Job Details")
    
    # Get unique values for dropdowns
    size_options = sorted(data['Size'].unique())
    ownership_options = sorted(data['Type of ownership'].unique())
    industry_options = sorted(data['Industry'].unique())
    sector_options = sorted(data['Sector'].unique())
    revenue_options = sorted(data['Revenue'].unique())
    job_options = sorted(data['job_simplified'].unique())
    seniority_options = sorted(data['seniority'].unique())
    rating_options = sorted(data['Rating Category'].unique())
    state_options = sorted(data['job_state'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Company Information")
        
        c1, c2 = st.columns(2)
        with c1:
            company_name = st.text_input("Company Name", placeholder="Enter company name (optional)")
            location = st.text_input("Location", placeholder="Enter location (optional)")
            headquarters = st.text_input("Headquarters", placeholder="Enter headquarters (optional)")
            size = st.selectbox("Company Size", [""] + size_options)
        with c2:
            ownership = st.selectbox("Type of Ownership", [""] + ownership_options)
            industry = st.selectbox("Industry", [""] + industry_options) 
            sector = st.selectbox("Sector", [""] + sector_options)
            revenue = st.selectbox("Revenue", [""] + revenue_options)
    
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Job Information")
            job_title = st.selectbox("Job Title", [""] + job_options)
            seniority = st.selectbox("Seniority Level", [""] + seniority_options)
            rating = st.selectbox("Company Rating Category", [""] + rating_options)
            job_state = st.selectbox("Job State", [""] + state_options)
        with c2:
            st.subheader("Technical Skills")
            python_yn = st.checkbox("Python Required", value=False)
            spark = st.checkbox("Spark Required", value=False)
            aws_yn = st.checkbox("AWS Required", value=False)
    
    return {
        'Company Name': company_name if company_name else None,
        'Location': location if location else None,
        'Headquarters': headquarters if headquarters else None,
        'Size': size if size else None,
        'Type of ownership': ownership if ownership else None,
        'Industry': industry if industry else None,
        'Sector': sector if sector else None,
        'Revenue': revenue if revenue else None,
        'job_simplified': job_title if job_title else None,
        'seniority': seniority if seniority else None,
        'Rating Category': rating if rating else None,
        'job_state': job_state if job_state else None,
        'Python_yn': 1 if python_yn else 0,
        'Spark': 1 if spark else 0,
        'AWS_yn': 1 if aws_yn else 0,
    }

def prepare_input_data(user_input: Dict[str, Any], defaults: Dict[str, Any]) -> pd.DataFrame:
    """Prepare input data by filling missing values with defaults."""
    # Create a copy and fill missing values
    filled_input = {}
    for key in defaults.keys():
        if key in user_input and user_input[key] is not None and user_input[key] != "":
            filled_input[key] = user_input[key]
        else:
            filled_input[key] = defaults[key]
    
    # Add a dummy Average Salary column (will be removed in prediction)
    filled_input['Average Salary'] = 0.0
    
    return pd.DataFrame([filled_input])

def predict_with_ml(model, input_df: pd.DataFrame) -> float:
    """Predict salary using the ML model."""
    # Remove the Average Salary column as it's the target
    features = input_df.drop('Average Salary', axis=1)
    prediction = model.predict(features)
    return prediction[0]

def build_prompt_for_inference(messages, tokenizer):
    """Build prompt for LLM inference."""
    msgs = [m for m in messages if m['role'] != 'assistant']
    msgs.append({'role': 'assistant', 'content': ''})
    return tokenizer.apply_chat_template(msgs, tokenize=False)

def predict_with_llm(model, tokenizer, input_data: Dict[str, Any]) -> Optional[float]:
    """Predict salary using the fine-tuned LLM."""
    try:
        import torch
        
        # Create a descriptive prompt from the input data
        prompt_text = f"""Based on the following job details, predict the salary:

Company: {input_data.get('Company Name', 'Unknown')}
Location: {input_data.get('Location', 'Unknown')}
Headquarters: {input_data.get('Headquarters', 'Unknown')}
Company Size: {input_data.get('Size', 'Unknown')}
Ownership Type: {input_data.get('Type of ownership', 'Unknown')}
Industry: {input_data.get('Industry', 'Unknown')}
Sector: {input_data.get('Sector', 'Unknown')}
Revenue: {input_data.get('Revenue', 'Unknown')}
Job Title: {input_data.get('job_simplified', 'Unknown')}
Seniority: {input_data.get('seniority', 'Unknown')}
Company Rating: {input_data.get('Rating Category', 'Unknown')}
State: {input_data.get('job_state', 'Unknown')}
Python Required: {'Yes' if input_data.get('Python_yn') else 'No'}
Spark Required: {'Yes' if input_data.get('Spark') else 'No'}
AWS Required: {'Yes' if input_data.get('AWS_yn') else 'No'}

What is the predicted salary?"""

        messages = [{'role': 'user', 'content': prompt_text}]
        prompt = build_prompt_for_inference(messages, tokenizer)
        
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=9,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,
            )
        
        gen = tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        gen_clean = gen.replace(',', '')
        
        int_pattern = re.compile(r"-?\d+")
        m = int_pattern.search(gen_clean)
        
        return int(m.group(0)) if m else None
        
    except Exception as e:
        st.error(f"Error in LLM prediction: {str(e)}")
        return None

def main():
    """Main application function."""
    st.title("💰 Data Science Job Salary Predictor")
    st.markdown("---")
    
    # Check if required files exist
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ Data file not found: {DATA_FILE}")
        st.stop()
    
    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ Model file not found: {MODEL_FILE}")
        st.stop()
    
    # Load data and models
    try:
        data = load_data()
        defaults = get_default_values(data)
        st.success(f"✅ Loaded dataset with {len(data):,} records")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    # Create input form
    user_input = create_input_form(data)
    
    st.markdown("---")
    
    # Model selection
    st.header("🤖 Choose Prediction Method")
    prediction_method = st.radio(
        "Select prediction approach:",
        ["Machine Learning (XGBoost)", "Fine-tuned LLM (Llama 3.1 8B)"],
        help="Choose between traditional ML approach or AI language model approach"
    )
    
    # Prediction button
    if st.button("🚀 Predict Salary", type="primary"):
        with st.spinner("Making prediction..."):
            # Prepare input data
            input_df = prepare_input_data(user_input, defaults)
            
            # Display filled input data
            st.subheader("📊 Input Data Summary")
            display_df = input_df.drop('Average Salary', axis=1).copy()
            
            # Show which values were filled with defaults
            filled_info = []
            for key, value in user_input.items():
                if value is None or value == "":
                    filled_info.append(f"**{key}**: {defaults[key]} *(default)*")
                else:
                    filled_info.append(f"**{key}**: {value}")
            
            col1, col2 = st.columns(2)
            with col1:
                for item in filled_info[:len(filled_info)//2]:
                    st.markdown(item)
            with col2:
                for item in filled_info[len(filled_info)//2:]:
                    st.markdown(item)
            
            st.markdown("---")
            
            # Make prediction based on selected method
            if prediction_method == "Machine Learning (XGBoost)":
                try:
                    model = load_ml_model()
                    prediction = predict_with_ml(model, input_df)
                    
                    st.success(f"🎯 **Predicted Salary: ${prediction:,.2f}**")
                    st.info("💡 Prediction made using XGBoost model with Optuna hyperparameter tuning")
                    
                except Exception as e:
                    st.error(f"Error with ML prediction: {str(e)}")
            
            else:  # LLM prediction
                try:
                    model, tokenizer = load_llm_model()
                    if model is not None and tokenizer is not None:
                        prediction = predict_with_llm(model, tokenizer, user_input)
                        
                        if prediction is not None:
                            st.success(f"🎯 **Predicted Salary: ${prediction:,.2f}**")
                            st.info("💡 Prediction made using fine-tuned Llama 3.1 8B Instruct model with QLoRA")
                        else:
                            st.error("Could not extract a valid prediction from the LLM response")
                    else:
                        st.error("LLM model could not be loaded")
                        
                except Exception as e:
                    st.error(f"Error with LLM prediction: {str(e)}")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts data science job salaries using two different approaches:
        
        **🔬 Machine Learning Approach:**
        - Uses XGBoost regressor
        - Trained on glassdoor job data
        - Hyperparameters optimized with Optuna
        
        **🧠 LLM Approach:**
        - Uses fine-tuned Llama 3.1 8B Instruct
        - Trained with QLoRA (Quantized LoRA)
        - Supervised fine-tuning on salary data
        
        **📝 Instructions:**
        1. Fill in the job details (optional fields will use dataset defaults)
        2. Choose prediction method
        3. Click "Predict Salary"
        """)
        
        st.header("📊 Dataset Info")
        if 'data' in locals():
            avg_salary = data['Average Salary'].mean()
            min_salary = data['Average Salary'].min()
            max_salary = data['Average Salary'].max()
            
            st.markdown(f"""
            - **Total records:** {len(data):,}
            - **Features:** {len(data.columns)-1}
            - **Target:** Average Salary
            - **Source:** Glassdoor job postings
            
            **Salary Statistics:**
            - **Average:** ${avg_salary:,.0f}
            - **Minimum:** ${min_salary:,.0f}
            - **Maximum:** ${max_salary:,.0f}
            """)
            
            # Show top job titles
            top_jobs = data['job_simplified'].value_counts().head(5)
            st.subheader("🏆 Top Job Types")
            for job, count in top_jobs.items():
                st.write(f"• **{job}**: {count} positions")
        else:
            st.markdown(f"""
            - **Total records:** Loading...
            - **Features:** 15
            - **Target:** Average Salary
            - **Source:** Glassdoor job postings
            """)

if __name__ == "__main__":
    main()
