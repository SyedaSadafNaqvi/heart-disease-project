# ❤️ Heart Disease Prediction Using Machine Learning

## Project Overview

This project implements a machine learning–based system to predict whether a patient is likely to have heart disease based on clinical and demographic attributes. The system covers the complete workflow, including dataset analysis, model training and evaluation, and deployment through a web-based dashboard for real-time prediction.

---

## Dataset Overview

The project uses the Heart Disease dataset (`heart.csv`) sourced from Kaggle. The dataset includes medical attributes such as age, cholesterol level, resting blood pressure, chest pain type, and other related clinical features.

### Target Variable
- `0` → No heart disease  
- `1` → Presence of heart disease  

The dataset contains no missing values; therefore, no records were removed during preprocessing.

---

## Data Preprocessing

- All features except the target column were used as input variables.
- The dataset was split into **80% training and 20% testing** using stratification to preserve class balance.
- **StandardScaler** was applied to:
  - Logistic Regression
  - Support Vector Machine (SVM)
- Random Forest was trained on unscaled data.
- All categorical features were already numerically encoded, so no additional encoding was required.

---

## Models Used

The following classification models were trained and evaluated using the same dataset for fair comparison:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM with RBF Kernel)**

---

## Model Evaluation

Models were evaluated using:
- **Accuracy**
- **Confusion Matrix Analysis**

### Observations
- Logistic Regression and SVM achieved strong performance.
- Random Forest initially showed very high training accuracy, indicating potential overfitting.
- After tuning, Random Forest generalization improved.
- **SVM achieved the highest test accuracy and best generalization performance.**

---

## Overfitting Analysis

Random Forest initially exhibited signs of overfitting. To address this:
- Maximum tree depth was limited
- Minimum samples per split were increased

This reduced the gap between training and testing accuracy and improved generalization.

---

## Final Model Selection

**Selected Model:** Support Vector Machine (SVM – RBF Kernel)

**Reason:**  
SVM demonstrated the highest accuracy, stability, and ability to model non-linear decision boundaries, making it well suited for medical datasets.

---

## Results Summary

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | 80.98% |
| Random Forest       | 91.71% |
| **SVM (RBF)**       | **92.68% ⭐** |

---

## Web Application (Dashboard)

A **Streamlit-based web dashboard** was developed to deploy the trained SVM model. The application allows users to:

- Enter patient medical information
- Apply the same preprocessing pipeline used during training
- Receive real-time predictions indicating the likelihood of heart disease

The trained model (`svm_model.joblib`) and scaler (`scaler.joblib`) are loaded directly into the application to ensure consistency with the training pipeline.

---

## Project Structure

heart-disease-project/
│
├── heart_diseases_prediction.ipynb
├── heart.csv
├── README.md
│
├── app/
│ ├── app.py
│ ├── svm_model.joblib
│ ├── scaler.joblib
│ └── requirements.txt


---

## Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib


---

## Installation & Usage

### Run the Jupyter Notebook
```bash
jupyter notebook heart_diseases_prediction.ipynb
cd app
pip install -r requirements.txt
streamlit run app.py

Team Members
@mirza1272 
@AdanAli951 
@khizardev123 
@moiz-mansoori
@MuhammadMusabYaqoob

---
##Conclusion
This project demonstrates a complete end-to-end machine learning workflow for healthcare prediction, from data preprocessing and model evaluation to deployment via a user-friendly web application. The system highlights the importance of model comparison, overfitting control, and practical deployment in medical decision-support systems.
