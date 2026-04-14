# Salary_Prediction_using_ML_Streamlit
# 💼 Software Developer Salary Prediction App

A machine learning-powered web application that predicts software developer salaries based on country, education level, and years of professional experience. Built with Streamlit and trained on the Stack Overflow Developer Survey 2020 dataset.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Live Demo / Screenshots](#live-demo--screenshots)
- [Tech Stack](#tech-stack)
- [Libraries & Dependencies](#libraries--dependencies)
- [Dataset](#dataset)
- [Machine Learning Model](#machine-learning-model)
- [Data Preprocessing](#data-preprocessing)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [App Pages](#app-pages)
- [Model Persistence](#model-persistence)
- [Known Limitations](#known-limitations)

---

## 📌 Project Overview

This project is an end-to-end machine learning application that allows users to:

1. **Predict** their expected salary as a software developer based on three inputs: country, education level, and years of coding experience.
2. **Explore** salary trends across countries and experience levels through interactive visualizations.

The entire pipeline — from data cleaning and model training (in a Jupyter Notebook) to deployment as a web app (via Streamlit) — is included in this repository.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Streamlit |
| Language | Python 3.12 |
| ML & Modeling | scikit-learn |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib |
| Model Serialization | Pickle |
| Development Environment | Jupyter Notebook |

---

## 📦 Libraries & Dependencies

```txt
streamlit
pandas
numpy
scikit-learn
matplotlib
pickle (built-in Python standard library)
```

Install all dependencies with:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

Or if a `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

**Source:** [Stack Overflow Developer Survey 2020](https://insights.stackoverflow.com/survey)

**File:** `survey_results_public.csv`

**Size:** 64,461 respondents × 61 columns

The survey covers a wide range of developer demographics and professional data. For this project, only the following columns are used:

| Column | Description |
|---|---|
| `Country` | Country of residence |
| `EdLevel` | Highest level of formal education |
| `YearsCodePro` | Years of professional coding experience |
| `ConvertedComp` | Annual salary converted to USD |
| `Employment` | Employment status |

### Filtering Criteria Applied

- Only **full-time employed** developers are included.
- Salaries are filtered between **$10,000 and $250,000** USD to remove extreme outliers.
- Countries with fewer than **400 survey responses** are grouped into an `"Other"` category and then excluded from modeling.

---

## 🤖 Machine Learning Model

### Model Selection Process

Three regression models were trained and evaluated using **Root Mean Squared Error (RMSE)** as the metric:

| Model | Notes |
|---|---|
| `LinearRegression` | Baseline model; high RMSE due to non-linear relationships |
| `DecisionTreeRegressor` | Better fit; used as the final model after tuning |
| `RandomForestRegressor` | Also tested; not selected as final model |

### Final Model

**`DecisionTreeRegressor`** (from `sklearn.tree`) — tuned using **GridSearchCV**.

**Hyperparameter Tuning:**

```python
from sklearn.model_selection import GridSearchCV

max_depth = [None, 2, 4, 6, 8, 10, 12]
parameters = {"max_depth": max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)

regressor = gs.best_estimator_
```

The best `max_depth` is automatically selected by cross-validation to minimize RMSE.

### Features Used for Training

| Feature | Type | Encoding |
|---|---|---|
| `Country` | Categorical | `LabelEncoder` |
| `EdLevel` | Categorical | `LabelEncoder` |
| `YearsCodePro` | Numerical (float) | No encoding needed |

**Target Variable:** `Salary` (Annual salary in USD)

---

## 🧹 Data Preprocessing

The following preprocessing steps are applied both during training (in the notebook) and at inference time (in the app):

### 1. Experience Cleaning (`clean_experience`)
Converts text-based experience values to floats:
- `"More than 50 years"` → `50`
- `"Less than 1 year"` → `0.5`
- All other values → `float(x)`

### 2. Education Level Cleaning (`clean_education`)
Simplifies education descriptions into four categories:
- Any mention of `"Bachelor"` → `"Bachelor's degree"`
- Any mention of `"Master"` → `"Master's degree"`
- `"Professional degree"` or `"doctoral"` → `"Post grad"`
- Everything else → `"Less than a Bachelor"`

### 3. Country Grouping (`shorten_categories`)
Countries with fewer than 400 responses are relabeled as `"Other"` and then excluded to reduce noise in the model.

### 4. Label Encoding
Both `Country` and `EdLevel` are encoded using `sklearn.preprocessing.LabelEncoder`. The fitted encoders are saved alongside the model so that the app can transform user inputs consistently at prediction time.

---

## 📁 Project Structure

```
Salary_prediction_streamlit_example_project/
│
├── app.py                    # Main Streamlit entry point; handles page routing
├── predict_page.py           # Prediction page UI and inference logic
├── explore_page.py           # Data exploration page with charts and visualizations
├── SalesPrediction.ipynb     # Jupyter Notebook for EDA, model training & evaluation
├── saved_steps.pkl           # Serialized model + LabelEncoders (ready for inference)
└── survey_results_public.csv # Stack Overflow Developer Survey 2020 raw dataset
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone or download the project
cd Salary_prediction_streamlit_example_project

# 2. Install dependencies
pip install streamlit pandas numpy scikit-learn matplotlib

# 3. Run the Streamlit app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> **Note:** Make sure `survey_results_public.csv` and `saved_steps.pkl` are in the same directory as `app.py` before running.

---

## 📱 App Pages

### 1. 🔮 Predict Page
Accessible via the sidebar dropdown (`"Predict"`).

**Inputs:**
- **Country** — dropdown, populated from the LabelEncoder classes
- **Education Level** — dropdown, populated from the LabelEncoder classes
- **Years of Experience** — slider from 0 to 50

**Output:**
- Displays the predicted annual salary in USD upon clicking `"Calculate Salary"`.

**How it works:**
1. User selections are assembled into a NumPy array.
2. Country and Education values are transformed using the pre-fitted `LabelEncoder` objects.
3. The array is cast to float and passed to the loaded `DecisionTreeRegressor`.
4. The predicted salary is displayed on screen.

---

### 2. 📊 Explore Page
Accessible via the sidebar dropdown (`"Explore"`).

Displays three visualizations derived from the cleaned survey data:

| Chart | Description |
|---|---|
| **Pie Chart** | Distribution of survey respondents across countries |
| **Bar Chart** | Mean salary by country (sorted ascending) |
| **Line Chart** | Mean salary by years of professional experience (sorted ascending) |

Data is loaded and cached using `@st.cache_data` to avoid reloading on every interaction.

---

## 💾 Model Persistence

The trained model and label encoders are serialized together into a single Pickle file (`saved_steps.pkl`) using Python's built-in `pickle` module:

```python
import pickle

data = {
    "model": regressor,
    "le_country": le_country,
    "le_education": le_education
}

with open("saved_steps.pkl", "wb") as file:
    pickle.dump(data, file)
```

At runtime, the app deserializes this file once on startup and reuses the objects for all predictions.

---

## ⚠️ Known Limitations

- **Dataset year:** The model is trained on 2020 survey data. Salary trends have changed since then, so predictions may not reflect current market rates.
- **Country coverage:** Only countries with 400+ responses are included. Developers from smaller countries will not find their country listed.
- **Salary range:** The model only considers salaries between $10,000 and $250,000 USD. Predictions outside this range may be unreliable.
- **Features:** Only three features are used for prediction. Other important factors (tech stack, company size, remote work, etc.) are not considered.
- **Overfitting risk:** The Decision Tree is evaluated on training data only. Cross-validated RMSE would give a more accurate picture of generalization performance.

---

## 📄 License

This project is intended for educational purposes. The dataset is sourced from Stack Overflow's annual Developer Survey and is subject to their [usage terms](https://insights.stackoverflow.com/survey).
