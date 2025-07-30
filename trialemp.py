#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


st.set_page_config(layout="wide", page_title="Employee Attrition Analysis & Prediction")


@st.cache_resource # Cache the model loading
def load_data():
    """Loads the employee attrition dataset."""
   
    df = pd.read_csv(r'C:\Users\91904\Downloads\Employee-Attrition - Employee-Attrition.csv')
    return df

def clean_data(df):
    """Performs basic data cleaning steps."""
    dfnew = df.copy()
    columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    dfnew = dfnew.drop(columns=columns_to_drop, errors='ignore')
    return dfnew

def preprocess_data(df, target_column='Attrition', test_size=0.2, random_state=42):
   
    df_processed = df.copy()

   
    le = LabelEncoder()
    df_processed[target_column] = le.fit_transform(df_processed[target_column])

    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]

    # Identify categorical and numerical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features

@st.cache_resource # Cache the model loading
def get_trained_model():
    """Loads the pre-trained attrition model."""
    try:
       
        model = joblib.load('c:/users/91904/trained_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: Trained model 'trained_model.pkl' not found at 'c:/users/91904/'. Please ensure the path is correct.")
        st.stop() # Stop the app if model is not found

@st.cache_data # Cache the data loading and initial cleaning
def get_cleaned_data_for_eda():
    """Loads and cleans the initial dataset for EDA."""
    df = load_data()
    df_cleaned = clean_data(df.copy())
    
    if 'Attrition' not in df_cleaned.columns or not df_cleaned['Attrition'].dtype == 'object':
        original_df = load_data() # Reload original to get 'Attrition' string
        df_cleaned['Attrition'] = original_df['Attrition']
    return df_cleaned

# --- Global Data and Model Loading ---
model = get_trained_model()
df_eda = get_cleaned_data_for_eda()


_, _, _, _, _, numerical_features_list, categorical_features_list = preprocess_data(df_eda.copy())


# --- Prediction Functions ---
def predict_attrition_bulk(model, new_data_df):
    
   
    probabilities = model.predict_proba(new_data_df)[:, 1]
    predictions = model.predict(new_data_df)

    results_df = new_data_df.copy()
    results_df['Predicted_Attrition_Probability'] = probabilities
    results_df['Predicted_Attrition'] = predictions # 0 for Stayed, 1 for Left

    # Convert numeric predictions back to 'Yes'/'No' for clarity
   
    results_df['Predicted_Attrition'] = results_df['Predicted_Attrition'].map({1: 'Yes', 0: 'No'})
   
    return results_df

def get_at_risk_employees(prediction_df, threshold=0.5):
    """Filters employees based on predicted attrition probability."""
    at_risk_df = prediction_df[prediction_df['Predicted_Attrition_Probability'] >= threshold].sort_values(
        by='Predicted_Attrition_Probability', ascending=False
    )
    return at_risk_df

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Predict Attrition for an Employee", "Bulk Attrition Prediction"])

# --- Main Content ---

if page == "Dashboard Overview":
    st.title("📊 Employee Attrition Dashboard")
    st.write("Explore key factors influencing employee attrition.")

    # Attrition Rate
    
    attrition_count = df_eda[df_eda['Attrition'] == 'Yes'].shape[0]
    total_employees = df_eda.shape[0]
    attrition_rate = (attrition_count / total_employees) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", total_employees)
    with col2:
        st.metric("Employees Attrited", attrition_count)
    with col3:
        st.metric("Overall Attrition Rate", f"{attrition_rate:.2f}%")

    st.markdown("---")

    # EDA Visualizations
    st.header("Key Attrition Insights")

    # Attrition by Department
    st.subheader("Attrition by Department")
    fig_dept = plt.figure(figsize=(10, 6))
    sns.countplot(data=df_eda, x='Department', hue='Attrition', palette='viridis')
    plt.title('Attrition Count by Department')
    plt.xlabel('Department')
    plt.ylabel('Number of Employees')
    plt.xticks(rotation=45)
    st.pyplot(fig_dept)

    # Attrition by Job Role
    st.subheader("Attrition by Job Role")
    fig_role = plt.figure(figsize=(12, 7))
    sns.countplot(data=df_eda, y='JobRole', hue='Attrition', palette='cividis', order=df_eda['JobRole'].value_counts().index)
    plt.title('Attrition Count by Job Role')
    plt.xlabel('Number of Employees')
    plt.ylabel('Job Role')
    st.pyplot(fig_role)

    # Attrition by Monthly Income
    st.subheader("Attrition by Monthly Income")
    fig_income = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_eda, x='Attrition', y='MonthlyIncome', palette='coolwarm')
    plt.title('Monthly Income Distribution for Attrited vs. Non-Attrited Employees')
    plt.xlabel('Attrition')
    plt.ylabel('Monthly Income')
    st.pyplot(fig_income)

    # Attrition by OverTime
    st.subheader("Impact of Overtime on Attrition")
    fig_ot = plt.figure(figsize=(8, 5))
    sns.countplot(data=df_eda, x='OverTime', hue='Attrition', palette='plasma')
    plt.title('Attrition by OverTime Status')
    plt.xlabel('OverTime')
    plt.ylabel('Number of Employees')
    st.pyplot(fig_ot)

    # Correlation Heatmap (for numerical features)
    st.subheader("Correlation Heatmap of Numerical Features with Attrition")
   
    df_eda_corr = df_eda.copy()
    df_eda_corr['Attrition_Numeric'] = df_eda_corr['Attrition'].map({'Yes': 1, 'No': 0})
    numerical_cols_for_corr = df_eda_corr.select_dtypes(include=np.number).columns.tolist()
   
    numerical_cols_for_corr = [col for col in numerical_cols_for_corr if col not in ['EmployeeNumber', 'EmployeeCount', 'StandardHours']]

    corr_matrix = df_eda_corr[numerical_cols_for_corr].corr()
    fig_corr = plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features')
    st.pyplot(fig_corr)


elif page == "Predict Attrition for an Employee":
    st.title("🔮 Predict Individual Employee Attrition")
    st.write("Input employee details to get an attrition prediction.")

   
    unique_education_fields = sorted(df_eda['EducationField'].unique())
   
    unique_education_levels = sorted(df_eda['Education'].unique()) 

    with st.form("employee_prediction_form"):
        st.subheader("Employee Demographics")
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance_from_home = st.slider("Distance From Home (miles)", 1, 29, 10)
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

        st.subheader("Job Information")
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", sorted(df_eda['JobRole'].unique()))
        job_level = st.slider("Job Level", 1, 5, 2)
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], format_func=lambda x: f"{x} - {['Low', 'Medium', 'High', 'Very High'][x-1]}")
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], format_func=lambda x: f"{x} - {['Low', 'Medium', 'High', 'Very High'][x-1]}")
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], format_func=lambda x: f"{x} - {['Low', 'Medium', 'High', 'Very High'][x-1]}")
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], format_func=lambda x: f"{x} - {['Low', 'Medium', 'High', 'Very High'][x-1]}")
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], format_func=lambda x: f"{x} - {['Bad', 'Good', 'Better', 'Best'][x-1]}")
        overtime = st.selectbox("OverTime", ["No", "Yes"])

        st.subheader("Compensation & Tenure")
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=5000)
        monthly_rate = st.slider("Monthly Rate ($)", 1000, 150000, 4000)
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=1000, value=500)
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=15000, value=800)
        percent_salary_hike = st.slider("Percent Salary Hike", 11, 25, 14)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
        education = st.selectbox("Education Level", unique_education_levels) # Use actual unique values
        education_field = st.selectbox("Education Field", unique_education_fields) # Added missing input

        total_working_years = st.slider("Total Working Years", 0, 40, 10)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_current_role = st.slider("Years in Current Role", 0, 18, 3)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_curr_manager = st.slider("Years With Current Manager", 0, 17, 2)
        num_companies_worked = st.slider("Number of Companies Worked", 0, 9, 1)
        training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)
        performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])

        submitted = st.form_submit_button("Predict Attrition")

        if submitted:
            # Create a dictionary from user inputs
            user_input = {
                'Age': age,
                'BusinessTravel': business_travel,
                'DailyRate': daily_rate,
                'Department': department,
                'DistanceFromHome': distance_from_home,
                'Education': education,
                'EducationField': education_field, # Now correctly captured
                'EnvironmentSatisfaction': environment_satisfaction,
                'Gender': gender,
                'HourlyRate': hourly_rate,
                'JobInvolvement': job_involvement,
                'JobLevel': job_level,
                'JobRole': job_role,
                'JobSatisfaction': job_satisfaction,
                'MaritalStatus': marital_status,
                'MonthlyIncome': monthly_income,
                'MonthlyRate': monthly_rate,
                'NumCompaniesWorked': num_companies_worked,
                'OverTime': overtime,
                'PercentSalaryHike': percent_salary_hike,
                'PerformanceRating': performance_rating,
                'RelationshipSatisfaction': relationship_satisfaction,
                'StockOptionLevel': stock_option_level,
                'TotalWorkingYears': total_working_years,
                'TrainingTimesLastYear': training_times_last_year,
                'WorkLifeBalance': work_life_balance,
                'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': years_in_current_role,
                'YearsSinceLastPromotion': years_since_last_promotion,
                'YearsWithCurrManager': years_with_curr_manager
            }

          
            input_df_for_prediction = pd.DataFrame([user_input])

            try:
              
                prob = model.predict_proba(input_df_for_prediction)[0][1] # Probability of attrition (class 1)
                pred_numeric = model.predict(input_df_for_prediction)[0] # Numeric prediction (0 or 1)
                pred_label = 'Yes' if pred_numeric == 1 else 'No' # Convert to 'Yes'/'No'

                st.subheader("Prediction Result:")
                if pred_label == 'Yes':
                    st.error(f"This employee is **likely to attrit** with a probability of **{prob:.2f}**.")
                    st.write("Consider proactive retention strategies for this employee.")
                else:
                    st.success(f"This employee is **likely to stay** with a probability of **{(1-prob):.2f}**.")
                    st.write("This employee is likely to be retained.")

                st.markdown("---")
                #st.subheader("What are the potential drivers for this prediction?")
              #  st.info("To truly understand individual drivers, you would implement **feature importance analysis** (e.g., for Random Forest) or **Shapley values (SHAP)**. This dashboard can only highlight general insights from the EDA.")
                #st.write("Based on general trends, high probability of attrition might be linked to factors like low job satisfaction, high overtime, long commute, or low monthly income compared to peers.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please ensure all input fields are correctly filled and match the expected format and the model's training data columns.")


elif page == "Bulk Attrition Prediction":
    st.title("📦 Bulk Employee Attrition Prediction")
    st.write("Upload a CSV file with new employee data for bulk attrition prediction.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            bulk_data_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(bulk_data_df.head())

            st.write("Performing predictions...")
            predictions_bulk_df = predict_attrition_bulk(model, bulk_data_df) # Use the bulk prediction function
            

            st.subheader("Prediction Results (First 10 Rows):")
            st.dataframe(predictions_bulk_df[['Age', 'Department', 'MonthlyIncome', 'Predicted_Attrition_Probability', 'Predicted_Attrition']].head(10))

            st.subheader("Employees Identified as At-Risk:")
            attrition_threshold = st.slider("Set Attrition Probability Threshold", 0.0, 1.0, 0.5, 0.05)
            at_risk_employees_bulk = get_at_risk_employees(predictions_bulk_df, threshold=attrition_threshold)

            if not at_risk_employees_bulk.empty:
                st.write(f"Found {len(at_risk_employees_bulk)} employees with predicted attrition probability >= {attrition_threshold:.2f}.")
                st.dataframe(at_risk_employees_bulk[['Age', 'JobRole', 'Department', 'MonthlyIncome', 'Predicted_Attrition_Probability']])

                csv = at_risk_employees_bulk.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download At-Risk Employees CSV",
                    data=csv,
                    file_name="at_risk_employees.csv",
                    mime="text/csv",
                )
            else:
                st.info("No employees identified as at-risk with the current threshold.")

        except Exception as e:
            st.error(f"An error occurred: {e}. Please ensure your CSV file is correctly formatted and contains all necessary columns as expected by the model.")


