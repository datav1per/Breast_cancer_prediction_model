# Breast_cancer_prediction_model
Breast Cancer Diagnosis Prediction (ML Project)

This project tackles the classification of breast tumors as benign or malignant using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to explore the dataset, understand the features, select the most relevant ones, and build an effective machine learning model RFECV and Random Forest.

This is my first full machine learning project, so the focus is on learning through hands-on practice.

##  Dataset Overview

- **Source**: [Kaggle - Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Samples**: 569
- **Target Variable**: `diagnosis` — Malignant (`M`) or Benign (`B`)
- **Features**: 30 numeric features related to tumor size, texture, symmetry, etc. (mean, standard error, and worst values)

---

##  What This Project Covers

### 1.  Data Preprocessing
- Converted the target labels to numeric (0 = Benign, 1 = Malignant)
- Checked for missing values and duplicates
- Standardized features for better visualization

### 2.  Exploratory Data Analysis (EDA)
- Used violin plots to visualize feature distributions
- Plotted a correlation heatmap to detect multicollinearity
- Identified visually important features using graphs

### 3. Feature Selection

I tried three different approaches to find the most important features:
- Manual selection based on plots
- RFECV (Recursive Feature Elimination with Cross-Validation) using Logistic Regression
- Feature importance from a Random Forest Classifier

### 4.  Model Building

I trained models on the selected features using both:
- Random Forest Classifier with top 10 fetaures
- Logistic Regression (mainly for RFECV)

Here’s a quick performance comparison:
RFECV-selected features( Logistic Regression) : accuracy ~93.9%    |
Random Forest (top 10 features): accuracy~94.7%  


**Random Forest with selected features performed the best**.

### 5.  Model Evaluation

- Evaluated models using accuracy, precision, recall, F1-score
- Compared classification reports
- Focused on reducing false negatives (important in cancer detection)


## Tools & Libraries

- Python
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `scikit-learn` for ML models, feature selection, evaluation

---

##  Conclusion

- Feature selection made a big difference in model performance and interpretability.
- Random Forest was a strong performer, even with fewer features.
- Visual tools helped identify the most influential features in distinguishing between malignant and benign cases.
- The project gave me hands-on experience in the end-to-end machine learning workflow—from EDA to model evaluation.

---

##  Future Improvements

This is my first ML project, so I’ve kept it simple on purpose. In the future, I’d like to:
- Try hyperparameter tuning using GridSearchCV
- Add more models (e.g. XGBoost or SVM)
- Deploy the model using Streamlit or Flask

---

## 🙋‍♂️ About Me

I'm learning machine learning and data analysis step by step. This project helped me understand the complete ML workflow, especially feature selection and evaluation techniques.

Feel free to check out the notebook or suggest feedback!

