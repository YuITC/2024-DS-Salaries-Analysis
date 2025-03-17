# 📊 Data Science Job Salary Prediction

**Dataset Source**: [Kaggle](https://www.kaggle.com/datasets/fahadrehman07/data-science-jobs-and-salary-glassdoor) - [Fahad Rehman](https://www.kaggle.com/fahadrehman07) (Kaggle Datasets Master).

**Web Application**: [Streamlit](https://data-science-salary-predict.streamlit.app/).

![Screenshot 2025-03-17 101217](https://github.com/user-attachments/assets/3570506a-346a-48b0-b61b-e05445be14a2)

## 📌 Project Overview

This project aims to analyze salary trends in the **Data Science job market** and build a **machine learning model** to predict salaries based on job characteristics. Using a dataset collected from **Glassdoor**, this project follows a structured pipeline from **data preprocessing, exploratory data analysis (EDA), feature selection, model training, and evaluation** to deploying a **Streamlit application** for salary prediction.

Whether you are a **job seeker** looking to estimate your expected salary or a **recruiter** aiming to understand market salary trends, this tool provides valuable insights into the compensation landscape of **Data Science roles**.


## 📂 Project Structure

```bash
2024-DataScience-Salaries-Analysis/
│── data/
│   ├── glassdoor_jobs.csv # Raw dataset
│   ├── data_EDA.csv       # Cleaned dataset after preprocessing
│   ├── data_model.csv     # Final dataset for model training
│
│── model/
│   ├── tuning/                  # Model tuning results
│   ├── xgb_salary_predictor.pkl # Best performing model (XGBoost)
│
│── notebooks/
│   ├── phase1_preprocessing.ipynb # Data preprocessing notebook
│   ├── phase2_EDA.ipynb           # Exploratory Data Analysis (EDA)
│   ├── phase3_modeling.ipynb      # Model training & tuning
│
│── app.py           # Streamlit app for salary prediction
│── requirements.txt # Dependencies for running the project
│── LICENSE          # Project license
│── README.md        # Project documentation
```

## 🚀 Key Features

**1. Data Preprocessing & Cleaning**

- Removed inconsistencies, missing values, and irrelevant features.
- Extracted job attributes such as location, job type, required skills, and seniority level.

**2. Exploratory Data Analysis (EDA)**

- Visualized salary distribution across different factors like job title, experience, company size, and location.
- Identified key salary-driving factors to enhance model performance.

**3. Machine Learning Model**

- Tested multiple models, some of them are Linear Regression, Decision Tree, Random Forest, and XGBoost.
- **XGBoost was selected as the best-performing model** based on RMSE, R² score, and cross-validation performance.
- Hyperparameter tuning was conducted to further optimize the model.

**4. Salary Prediction Web App (Streamlit)**

- A user-friendly interface allowing users to input job-related details and receive an estimated salary prediction.
- Deployed on Streamlit for easy accessibility.


## 📈 Model Performance

The table below compares the **Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score** of various machine learning models tested during salary prediction:

| Model                           | MSE         | MAE         | R² Score  |
|--------------------------------|------------|------------|----------|
| **XGBoost (Best)**              | 2.896e+08  | 1.056e+04  | **0.815** |
| Random Forest                   | 2.936e+08  | 1.127e+04  | 0.813    |
| Histogram Gradient Boosting      | 4.009e+08  | 1.456e+04  | 0.744    |
| Decision Tree                    | 4.605e+08  | 1.024e+04  | 0.706    |
| Linear Regression                | 4.624e+08  | 1.212e+04  | 0.705    |
| K-Nearest Neighbors (KNN)        | 8.030e+08  | 2.114e+04  | 0.488    |
| Support Vector Regression (SVR)  | 1.616e+09  | 3.117e+04  | **-0.031** |

**Conclusion**: Given these results, **XGBoost was selected as the best model** for salary prediction due to its superior accuracy and overall performance. The model was fine-tuned further to optimize its hyperparameters for deployment.


## 🛠️ Installation & Usage

```bash
# Clone the Repository
git clone https://github.com/YuITC/2024-DataScience-Salaries-Analysis.git
cd 2024-DataScience-Salaries-Analysis

# Install Dependencies
pip install -r requirements.txt

# Run the Streamlit App
streamlit run app.py
```


## 📜 License
This project is licensed under the MIT License – feel free to modify and distribute it as needed.

## 🤝 Acknowledgments
Special thanks to [Fahad Rehman](https://www.kaggle.com/fahadrehman07) for the dataset.

If you find this project useful, consider ⭐️ starring the repository or contributing to further improvements!

## 📬 Contact
For any questions or collaboration opportunities, feel free to reach out:

📧 Email: tainguyenphu2502@gmail.com
