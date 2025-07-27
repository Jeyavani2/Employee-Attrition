Employee Attrition Analysis and Prediction
This project provides a comprehensive solution for analyzing employee attrition and predicting the likelihood of an employee leaving the company. It includes a machine learning model for prediction and an interactive Streamlit web application for exploration and individual/bulk predictions.

Table of Contents
Project Overview

Features

Technologies Used

Setup Instructions

Data

Model Training & Saving

Usage

Project Structure

Future Enhancements

1. Project Overview
Employee attrition is a critical concern for organizations, impacting productivity, morale, and financial stability. This project aims to address this challenge by:

Providing an interactive dashboard to visualize key factors influencing attrition.

Offering a tool to predict the attrition probability for individual employees based on their characteristics.

Enabling bulk predictions by uploading a dataset of new employees.

Training a robust machine learning model (Random Forest Classifier) to power these predictions.

2. Features
The Streamlit application provides the following functionalities:

Dashboard Overview:

Displays overall attrition rate and counts.

Visualizes attrition patterns by Department, Job Role, Monthly Income, and Overtime.

Shows a correlation heatmap of numerical features.

Predict Attrition for an Employee:

Interactive input form to enter an employee's details.

Predicts the probability of attrition and a "Yes"/"No" outcome.

Provides general insights into potential drivers (though not individual SHAP values in this version).

Bulk Attrition Prediction:

Allows users to upload a CSV file containing multiple employee records.

Performs predictions for all records in the uploaded file.

Displays a preview of predictions and lists "at-risk" employees based on a customizable probability threshold.

Enables downloading the list of at-risk employees.

3. Technologies Used
Python 3.x

Streamlit: For building the interactive web application.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Scikit-learn: For machine learning (preprocessing, model training, and evaluation).

LabelEncoder, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline

LogisticRegression, DecisionTreeClassifier, RandomForestClassifier

accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

Matplotlib: For creating static plots.

Seaborn: For enhanced statistical data visualizations.

Joblib: For saving and loading the trained machine learning model.

4. Setup Instructions
To get this project up and running on your local machine, follow these steps:

Clone the Repository (if applicable):
If this project is part of a Git repository, clone it:

git clone <repository-url>
cd <project-directory>

If it's just a single file, ensure app.py (or emp.py) and requirements.txt are in the same folder.

Create a Virtual Environment (Recommended):
It's good practice to use a virtual environment to manage dependencies.

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install Dependencies:
Install all required Python packages using the requirements.txt file:

pip install -r requirements.txt

Download the Dataset:
The project expects the dataset to be located at C:\Users\91904\Downloads\Employee-Attrition - Employee-Attrition.csv. Please ensure you download the "IBM HR Analytics Employee Attrition & Performance" dataset (commonly found on Kaggle) and place it at this exact path, or update the load_data function in app.py with the correct path to your dataset.

Train the Model (First Run):
The script is designed to train and save the model if it's not found. The first time you run the Streamlit app, or if you run the script directly, it will train the Random Forest model and save it as trained_model.pkl in c:/users/91904/.

5. Data
The project uses the "IBM HR Analytics Employee Attrition & Performance" dataset. This dataset contains various employee attributes, including demographic information, job-related features, and compensation details, along with an 'Attrition' column indicating whether an employee left the company.

Expected Dataset Path: C:\Users\91904\Downloads\Employee-Attrition - Employee-Attrition.csv

6. Model Training & Saving
The app.py script includes functions for:

load_data(): Reads the raw CSV file.

clean_data(): Drops irrelevant columns (EmployeeCount, StandardHours, Over18, EmployeeNumber).

preprocess_data(): Handles feature engineering, scaling numerical features (StandardScaler), and encoding categorical features (OneHotEncoder) using a ColumnTransformer, and splits the data into training and testing sets.

train_model(): Trains a specified machine learning model (Logistic Regression, Decision Tree, or Random Forest) within a Pipeline that includes the preprocessor.

evaluate_model(): Assesses model performance using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

save_model(): Saves the trained Pipeline object to a .pkl file using joblib.

get_trained_model(): Loads the saved model.

The if __name__ == '__main__': block at the end of the script handles the training and saving of the RandomForestClassifier model (named trained_model.pkl) upon direct execution of the script. This ensures the model is ready for the Streamlit app.

7. Usage
To run the Streamlit application:

Open your terminal or command prompt.

Navigate to the directory where Employee-Stream.py (or Employee-stream.py) is located.

Execute the command:

streamlit run Employee-Stream.py

(Replace Employee-Stream.py with your file name if different).

This will open the Streamlit application in your default web browser. You can then navigate through the dashboard, predict attrition for individual employees, or upload a CSV for bulk predictions.

8. Project Structure
This project is currently structured as a single Python file (Employee-Stream.py or Employee-Stream.py) that contains all the data loading, preprocessing, model training, evaluation, saving, and Streamlit application logic.

.
├── Employee-Stream.py           # Main Python script containing all project code
├── requirements.txt # List of Python dependencies
└── <your_dataset_path> # e.g., C:/Users/91904/Downloads/Employee-Attrition - Employee-Attrition.csv
└── <trained_model_path> # e.g., c:/users/91904/trained_model.pkl (generated after first run)

9. Future Enhancements
Advanced Model Interpretability: Implement SHAP (Shapley Additive Explanations) values to provide individual feature contributions for each prediction, offering more actionable insights to HR.

Model Retraining Interface: Add a Streamlit interface to trigger model retraining with new data or different hyperparameter settings.

Database Integration: Connect to a real database (e.g., PostgreSQL, MySQL) to store employee data and predictions, rather than relying on local CSV files.

User Authentication: Implement user login for secure access to the application.

More Sophisticated EDA: Add more interactive plots and filters to the dashboard.

Deployment: Deploy the Streamlit app to a cloud platform (e.g., Streamlit Community Cloud, AWS, GCP, Azure) for wider accessibility.

Error Handling and Input Validation: Enhance robustness for various user inputs and file uploads.
