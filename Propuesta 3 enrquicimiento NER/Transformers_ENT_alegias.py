from scipy.sparse import hstack, csr_matrix
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Ruta de datos
folder_path = "../Entrenamiento/datos entrenamiento traducidos"

# Cargar etiquetas
def Leer_excel():
    File = pd.read_excel('../Entrenamiento/tfg.xlsx', header=None)
    return File.iloc[:, 1].tolist()

etiquetas = Leer_excel()

# Orden natural de archivos
def alphanumeric_key(file_name):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', file_name)]

# Leer archivos de texto
def cargar_textos(folder_path):
    textos = []
    files = sorted(os.listdir(folder_path), key=alphanumeric_key)
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                textos.append(file.read())
    return textos

# Extraer vector binario de entidades NER
def extraer_vector_entidades(entidades):
    vector = [0, 0, 0]  # [CHEM, ANAT, DISO]
    for ent in entidades:
        if ent["entity_group"] == "CHEM":
            vector[0] = 1
        elif ent["entity_group"] == "ANAT":
            vector[1] = 1
        elif ent["entity_group"] == "DISO":
            vector[2] = 1
    return vector


Lista_textos_raw = cargar_textos(folder_path)
ner = pipeline("token-classification",
               model="lcampillos/roberta-es-clinical-trials-ner",
               aggregation_strategy="simple")


docs_entidades =  [ner(texto) for texto in Lista_textos_raw]

# Extraer vectores de presencia de entidades
vectores_ner = [extraer_vector_entidades(entidades) for entidades in docs_entidades]

# Vectorizar texto original (sin enriquecer)
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(Lista_textos_raw)
X_ner = np.array(vectores_ner)

# Combinar TF-IDF + NER
X = hstack([X_text, X_ner])
X = csr_matrix(X)
y = np.array(etiquetas)

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

print("Logistic regresion  con TfidfVectorizer:\n")
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

model = MultinomialNB()

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

print("MultinomialNB con TfidfVectorizer:\n")
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

print("RandomForestClassifier con TfidfVectorizer:\n")
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

print("SVC con TfidfVectorizer:\n")
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


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(Lista_textos_raw)
model = LogisticRegression()
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

print("LogisticRegression con CountVectorizer:\n")
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

model = MultinomialNB()

# Validación cruzada K-Fold

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

print("MultinomialNB con CountVectorizer:\n")
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

print("RandomForestClassifier con CountVectorizer:\n")
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

print("SVC con TfidfVectorizer CountVectorizer:\n")
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

vectorizer = TfidfVectorizer(ngram_range=(1,2))

X = vectorizer.fit_transform(Lista_textos_raw)


model = LogisticRegression()

# K-Fold

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

print("LogisticRegression con TfidfVectorizer prarametrizado:\n")
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

model = MultinomialNB()

# Validación cruzada K-Fold
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
print("MultinomialNB con TfidfVectorizer prarametrizado:\n")
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

print("RandomForestClassifier con TfidfVectorizer prarametrizado:\n")
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

print("SVC con TfidfVectorizer prarametrizado:\n")
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
