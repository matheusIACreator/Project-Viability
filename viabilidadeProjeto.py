import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_or_predict(new_projects):
    # Load the dataset
    model_path = 'logistic_model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load(model_path.replace(".joblib","_scaler.joblib"))
    else:
        df_projects = pd.read_csv("projects_data.csv")

        #Separar as variáveis
        X = df_projects[["investments","expected_return","impact_score"]]
        y = df_projects[["viability"]]

        #Normaliza
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        #Divide em treino e teste 
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42, test_size=3)

        #Treino
        model = LogisticRegression()
        model.fit(X_train, y_train)

        #Avaliação
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        #Salvar o modelo, Scaler e Métricas em disco
        joblib.dump(model, model_path)
        joblib.dump(scaler, model_path.replace(".joblib","_scaler.joblib"))
        joblib.dump(report, model_path.replace("joblib","_metrics.joblib"))

    #Previsão
    if new_projects:
        df_new_project = pd.DataFrame(new_projects)

        X_new_scaler = scaler.transform(df_new_project)

        predictions = model.predict(X_new_scaler)

        #Probabilidade de 1
        probabilities = model.predict_proba(X_new_scaler)[:,1]
        df_new_project["probability"] = probabilities
        df_new_project['viability'] = predictions
        return df_new_project,joblib.load(model_path.replace(".joblib", "_metrics.joblib"))
    
    return None, joblib.load(model_path.replace(".joblib", "_metrics.joblib"))
