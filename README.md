# alzheimers-diagnosis-prediction-ml
This project explores the predictive modeling of Alzheimer's disease diagnosis using various machine learning techniques. 

# Alzheimer's Disease Diagnosis Prediction using Machine Learning

This project explores the predictive modeling of Alzheimer's disease diagnosis using various machine learning techniques. The dataset includes lifestyle ("Habitual") and long-term medical ("Chronic") features. Models including Logistic Regression, Multinomial Naive Bayes, and Random Forest were evaluated to assess their accuracy in predicting Alzheimer’s diagnosis.

---

## 📂 Dataset

- Source: [Kaggle - Rabie El Khairoua Alzheimer’s Dataset](https://www.kaggle.com/datasets/rabielkhairoua/alzheimer-disease-dataset)
- Total Records: 2149
- Target Variable: `Diagnosis`
- Feature Subsets:
  - **Habitual**: BMI, Diet, Sleep, Physical Activity, Blood Pressure, etc.
  - **Chronic**: Family History, Smoking, Diabetes, Cardiovascular Disease, etc.

---

## 🛠️ Technologies Used

- Python 3.x
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn (Logistic Regression, Naive Bayes, Random Forest)

---

## 📈 Machine Learning Models

| Model               | Feature Type    | Accuracy |
|--------------------|------------------|----------|
| Logistic Regression | Habitual         | 79.3%    |
| Logistic Regression | Chronic          | 75.59%   |
| Naive Bayes         | Habitual         | 70.93%   |
| Naive Bayes         | Chronic          | 74.88%   |
| Random Forest       | Habitual         | 82.56%   |
| Random Forest       | Chronic          | 72.56%   |
| Random Forest       | Habitual + Chronic | **94.65%** |

---

## 📊 Visualizations

- Boxplots and KDE plots to compare distributions.
- Bar plots for categorical chronic features.
- Correlation heatmaps.
- Pair plots for visualizing feature relationships.

---

## 📁 Project Structure

