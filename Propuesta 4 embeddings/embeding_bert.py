from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import pandas as pd
import numpy as np
import os
import re

from sklearn.svm import SVC
from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ruta de los archivos
folder_path = "../Entrenamiento/datos entrenamiento traducidos"

# Leer etiquetas del Excel
def Leer_excel():
    File = pd.read_excel('../Entrenamiento/tfg.xlsx', header=None)
    return File.iloc[:, 1].tolist()

# Ordenar alfanuméricamente
def alphanumeric_key(file_name):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', file_name)]

# Leer los textos
Lista_textos = []
files = sorted(os.listdir(folder_path), key=alphanumeric_key)
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            Lista_textos.append(content)

# Leer etiquetas
etiquetas = Leer_excel()
y = np.array(etiquetas)

# Cargar modelo BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

def obtener_avg_embeddings(texts, batch_size=8):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            avg_pool = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(avg_pool.cpu())
    return torch.cat(all_embeddings).numpy()


X = obtener_avg_embeddings(Lista_textos)

model = LogisticRegression()
# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=5)

scores_acc = []
scores_recall = []
scores_precision = []
scores_f1 = []
conf_matrices = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    scores_acc.append(acc)
    scores_recall.append(recall)
    scores_precision.append(precision)
    scores_f1.append(f1)
    conf_matrices.append(cm)
    mean_conf_matrix = np.mean(conf_matrices, axis=0)

print("Logistic regresion  con embedding:\n")
print(f"Accuracy por fold:     {scores_acc}")
print(f"Precision por fold:    {scores_precision}")
print(f"Recall por fold:       {scores_recall}")
print(f"F1-score por fold:     {scores_f1}\n")

print(f"Accuracy promedio:     {np.mean(scores_acc):.4f}")
print(f"Precision promedio:    {np.mean(scores_precision):.4f}")
print(f"Recall promedio:       {np.mean(scores_recall):.4f}")
print(f"F1-score promedio:     {np.mean(scores_f1):.4f}\n")
print("Matriz de confusión promedio:")
print(mean_conf_matrix)

model = RandomForestClassifier(n_estimators=100, random_state=5)

scores_acc = []
scores_recall = []
scores_f1 = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    scores_acc.append(acc)
    scores_recall.append(recall)
    scores_precision.append(precision)
    scores_f1.append(f1)
    conf_matrices.append(cm)
    mean_conf_matrix = np.mean(conf_matrices, axis=0)

print("RandomForestClassifier con Embeding:\n")
print(f"Accuracy por fold:     {scores_acc}")
print(f"Precision por fold:    {scores_precision}")
print(f"Recall por fold:       {scores_recall}")
print(f"F1-score por fold:     {scores_f1}\n")

print(f"Accuracy promedio:     {np.mean(scores_acc):.4f}")
print(f"Precision promedio:    {np.mean(scores_precision):.4f}")
print(f"Recall promedio:       {np.mean(scores_recall):.4f}")
print(f"F1-score promedio:     {np.mean(scores_f1):.4f}\n")
print("Matriz de confusión promedio:")
print(mean_conf_matrix)

model = SVC(kernel="linear")

scores_acc = []
scores_recall = []
scores_f1 = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    scores_acc.append(acc)
    scores_recall.append(recall)
    scores_precision.append(precision)
    scores_f1.append(f1)
    conf_matrices.append(cm)
    mean_conf_matrix = np.mean(conf_matrices, axis=0)

print("SVC con Embeding:\n")
print(f"Accuracy por fold:     {scores_acc}")
print(f"Precision por fold:    {scores_precision}")
print(f"Recall por fold:       {scores_recall}")
print(f"F1-score por fold:     {scores_f1}\n")

print(f"Accuracy promedio:     {np.mean(scores_acc):.4f}")
print(f"Precision promedio:    {np.mean(scores_precision):.4f}")
print(f"Recall promedio:       {np.mean(scores_recall):.4f}")
print(f"F1-score promedio:     {np.mean(scores_f1):.4f}\n")
print("Matriz de confusión promedio:")
print(mean_conf_matrix)