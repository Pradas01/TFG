from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import re
import pandas as pd
from sklearn.svm import SVC

Lista_textos = []

folder_path = "../Entrenamiento/datos entrenamiento traducidos"

def Leer_excel():
    File = pd.read_excel('../Entrenamiento/tfg.xlsx', header=None)
    primera_columna = File.iloc[:, 0].tolist()
    segunda_columna = File.iloc[:, 1].tolist()
    return primera_columna


def alphanumeric_key(file_name):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', file_name)]

etiquetas = Leer_excel()

files = sorted(os.listdir(folder_path), key=alphanumeric_key)
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            Lista_textos .append(content)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(Lista_textos)
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
X = vectorizer.fit_transform(Lista_textos)

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

X = vectorizer.fit_transform(Lista_textos)


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
