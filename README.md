<h1 align="center">🚀 Project Viability Classifier</h1>

<p align="center">
  A machine learning model that predicts the viability of projects based on key financial and impact metrics.
</p>

---

## 📌 About

This project uses logistic regression to classify whether a project is viable, based on:

- 💰 Investment
- 📈 Expected Return
- 🌍 Impact Score

It automatically trains and saves the model, scaler, and evaluation metrics. If already trained, it just loads and predicts viability for new inputs.

---

## 🧠 Technologies Used

- Python 🐍  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- joblib  

---

## 🧪 How It Works

- If the model doesn’t exist yet:
  - It reads a dataset (`projects_data.csv`)
  - Scales the data
  - Trains a Logistic Regression model
  - Evaluates it
  - Saves model, scaler, and metrics using `joblib`

- If the model already exists:
  - It loads the model and scaler
  - Takes new data as input
  - Returns predictions and success probabilities

---

## 📁 File Structure

```
project-folder/
│
├── logistic_model.joblib           # Saved ML model
├── logistic_model_scaler.joblib    # Saved Scaler
├── logistic_model_metrics.joblib   # Saved Evaluation Metrics
├── projects_data.csv               # Dataset
├── your_script.py                  # Python logic
└── README.md
```

---

## 🧪 Example Input Format

```python
new_projects = [
  {"investments": 15000, "expected_return": 20000, "impact_score": 80},
  {"investments": 5000, "expected_return": 3000, "impact_score": 50}
]
```

---

## 📊 Output Example

| Investment | Expected Return | Impact Score | Probability | Viability |
|------------|------------------|---------------|-------------|-----------|
| 15000      | 20000            | 80            | 0.87        | 1         |
| 5000       | 3000             | 50            | 0.32        | 0         |

---

## 🧠 Author

Made with 💻 by [@matheusIACreator](https://github.com/matheusIACreator)

---