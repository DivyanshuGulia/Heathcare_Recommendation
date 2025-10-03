# Health Risk Prediction and Diet Recommendation System

## 📖 Project Description
This project is a **machine learning-based healthcare recommendation system** that predicts an individual’s **health risk category** and provides **personalized diet & lifestyle recommendations**.  
The system uses patient health data (demographics, BMI, blood pressure, cholesterol, blood sugar, habits, etc.) to classify risk levels (**Low, Moderate, High, Very High**) and suggests actionable steps like **diet plans, exercise, and monitoring**.

The project demonstrates an **end-to-end pipeline**: data preprocessing → model training → evaluation → interactive recommendation system (Flask app).

---

## ✨ Features
- **Data Preprocessing**: Cleans data, handles missing values, encodes categorical variables, and scales features.
- **Exploratory Data Analysis (EDA)**: Visualizations (scatter plots, bar charts, etc.) to uncover patterns in health data.
- **Machine Learning Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and SVM.
- **Model Evaluation**: Accuracy, precision, recall, F1-score, and classification reports.
- **High Accuracy**: Gradient Boosting model achieves ~95% accuracy in predicting risk categories.
- **Personalized Output**:
  - Predicted **risk level**
  - Comparison of patient health metrics vs healthy ranges (charts)
  - **Custom diet & lifestyle recommendations**
- **Interactive Flask App**: Simple UI for users to input their data and get real-time results with visualizations.

---

## 🛠 Technologies Used
- **Language**: Python (Jupyter Notebook)
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Machine Learning: `scikit-learn`, `imbalanced-learn`
  - Web App: `Flask`
  - Model Persistence: `joblib`
- **IDE**: Jupyter Notebook

---

## 📊 Dataset
1. **Personalized Diet Recommendations Dataset**  
   - ~5000 records with demographic, health, lifestyle, and diet recommendation details.  
   - Columns include Age, Gender, BMI, Blood Pressure, Cholesterol, Blood Sugar, Chronic Disease, Exercise Frequency, Dietary Habits, Allergies, etc.  
   - Target label: **Recommended Meal Plan** and derived **Risk Category**.

2. **Blood Count Dataset**  
   - ~417 samples with blood test metrics like Hemoglobin, Platelet Count, WBC, RBC, etc.  
   - Used for exploratory analysis and health insights.

---

