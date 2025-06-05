# Obesity Prediction Application using ML and Streamlit

This is a simple Machine Learning-powered web app that can be used in the medical space, designed to predict a person's obesity level based on basic lifestyle and health indicators. It's built using Python, trained with real-world data, and deployed using Streamlit. The goal is to provide quick, accessible insights into lifestyle-related obesity risks.

---

### 📊 Dataset

- Provided during a data science bootcamp; also publicly available on Kaggle.
- Includes features like gender, age, weight, physical activity, dietary habits, and more.

---

### 🧹 Data Preprocessing

- No missing values, so the following steps were done:
  - Data description and EDA
  - Label encoding for categorical features
  - Train-test split
  - Feature scaling
  - Target label rounding (for easier classification)
  - Feature selection using `SelectKBest` with chi-squared scoring

---

### 🤖 Model Building

Multiple ML models were tested to find the highest-performing one:

| Model                   | Accuracy |
|------------------------|----------|
| Logistic Regression    | 69%      |
| Neural Network (MLP)   | 85%      |
| SVM                    | 73%      |
| K-Nearest Neighbors    | 86%      |
| Random Forest (default)| 90%      |
| **Random Forest (Tuned)** | **91%** ✅ |

```python
from sklearn.ensemble import RandomForestClassifier
mod = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
mod.fit(x_train, y_train)
y_pred = mod.predict(x_test)
accuracy_score(y_test, y_pred)  # → 91%
