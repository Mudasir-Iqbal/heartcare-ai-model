# 🫀 Heart Disease Prediction using Machine Learning

A complete end-to-end machine learning project that uses patient data to predict the likelihood of heart disease. The goal is to assist medical professionals and the general public in early detection using a trained ML model.

---

## 📌 Project Objective

The purpose of this project is to:

* Predict the presence or absence of heart disease using key medical attributes.
* Utilize machine learning to support early diagnosis and improve patient outcomes.
* Build a deployable interface (Streamlit) where users can input their health details and get a prediction instantly.

---

## 🧠 How It Works

The machine learning model is trained on structured health-related data which includes:

* Age
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Maximum Heart Rate
* ST Depression (Oldpeak)
* Gender, Chest Pain Type, Exercise Angina, ECG Results, ST Slope

These features are fed into the model to classify a person as:

* `0` → No heart disease
* `1` → Likely has heart disease

---

## 🧾 Medical Insights

| Feature             | Normal Range         | Risk Indicator                           |
| ------------------- | -------------------- | ---------------------------------------- |
| Age                 | < 45                 | > 50 increases risk                      |
| Cholesterol         | 125–200 mg/dL        | > 240 is considered high risk            |
| Resting BP          | 90–120 mm Hg         | > 140 indicates hypertension             |
| Fasting Blood Sugar | 0 = Normal, 1 = High | 1 indicates possible diabetes risk       |
| Max Heart Rate      | \~100–190            | Lower value may indicate poor heart func |
| Oldpeak (ST dep.)   | < 1.0                | > 2.0 indicates possible ischemia        |
| Chest Pain Type     | ATA, NAP, TA, ASY    | ASY = Asymptomatic = High risk           |

---

## 📁 Dataset Info

* Total rows: **918**
* Target column: `HeartDisease`
* No missing values or duplicates (cleaned)
* Source: Cleaned CSV file `heart.csv`

---

## 🔧 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit (Deployment)
* Joblib (Model Saving)

---

## 🚀 Project Pipeline

### 1️⃣ Data Understanding & Cleaning

```python
df.isnull().sum()
df.drop_duplicates()
```

### 2️⃣ Categorical to Numeric (One-Hot Encoding)

```python
df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', ...], drop_first=True)
```

### 3️⃣ Feature Scaling (StandardScaler)

Used to normalize numerical values so all features contribute equally.

**Formula:**

$$
Z = \frac{X - \mu}{\sigma}
$$

```python
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])
```

---

### 4️⃣ Exploratory Data Analysis (EDA)

* Heatmaps
* Correlation analysis
* Count plots for categorical vs target

```python
sns.heatmap(df.corr(), annot=True)
```

---

### 5️⃣ Model Training

Trained using two models:

* Logistic Regression
* Decision Tree Classifier

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
```

---

### 6️⃣ Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

```python
classification_report(y_test, y_pred)
```

---

### 7️⃣ Hyperparameter Tuning (GridSearchCV)

```python
GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
```

---

### 8️⃣ Deployment via Streamlit

Simple web interface to predict heart disease using trained model.

```python
joblib.load("best_model.pkl")
st.number_input("Age", ...)
```

---

## 📦 Folder Structure

```
├── heart.csv
├── best_model.pkl
├── scaler.pkl
├── streamlit_app.py
├── model_training.ipynb
├── README.md
```

---

## 💡 How to Run

1. Clone the repo

```bash
https://github.com/Mudasir-Iqbal/heartcare-ai-model.git
cd heartcare-ai-model.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run Streamlit app

```bash
streamlit run .\streamlit_app.py
```

---

## 📈 Sample Prediction Output

✅ **No heart disease detected**
or
⚠️ **Risk of heart disease found**

---

## 💻 Live Demo

👉 **Check out the live app here:**  
[🔗 Heart Disease Prediction App (Streamlit)](https://heartcare-ai-model-by-mudasir.streamlit.app/)


## 🤝 Contributions

Pull requests are welcome. Please open an issue first to discuss what you would like to change.

---

## 📃 License

This project is open-source and free to use under the MIT License.

---

Would you like me to generate:

* `requirements.txt`
* `streamlit_app.py` full version?
* GitHub repo structure auto-upload?

Let me know and I’ll help you launch it 🚀
