import torch
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import os
import pandas as pd
from datasets import Dataset, DatasetDict
import re
import csv

# Rutas
folder_path = "../Entrenamiento/datos entrenamiento traducidos"
etiquetas_path = '../Entrenamiento/tfg.xlsx'
Lista_textos = []

# Tokenizador
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

# Leer etiquetas desde Excel
def Leer_excel():
    File = pd.read_excel(etiquetas_path, header=None)
    return File.iloc[:, 0].tolist()

# Ordenar archivos alfanuméricamente
def alphanumeric_key(file_name):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', file_name)]

# Cargar etiquetas y duplicarlas
etiquetas = Leer_excel()
etiquetas = etiquetas * 2

# Leer textos
files = sorted(os.listdir(folder_path), key=alphanumeric_key)
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            Lista_textos.append(file.read())

# Añadir traducciones
traducciones = []
with open('traducciones.csv', mode='r', encoding='utf-8') as archivo:
    lector = csv.reader(archivo)
    next(lector)
    for fila in lector:
        traducciones.append(fila[0])

Lista_textos += traducciones

# Dataset original
dataset = Dataset.from_dict({'text': Lista_textos, 'label': etiquetas})

# Dividir 80/10/10
# Primero 80% train y 20% temp
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split['train']
temp_dataset = split['test']

# Dividir el 20% restante en val y test (50/50 → 10% + 10%)
temp_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
val_dataset = temp_split['train']
test_dataset = temp_split['test']

# Tokenizar todos los splits
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Cargar modelo
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=2
)

# Configuración entrenamiento
training_args = TrainingArguments(
    output_dir="../multilingual-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Entrenar modelo
trainer.train()

test_texts = test_dataset['text']
true_labels = test_dataset['label']


inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Desactivar gradientes
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, axis=1).cpu().numpy()

# Métricas
acc = accuracy_score(true_labels, preds)
recall = recall_score(true_labels, preds)
f1 = f1_score(true_labels, preds)
cm = confusion_matrix(true_labels, preds)

print(f"Accuracy test: {acc:.4f}")
print(f"Recall test: {recall:.4f}")
print(f"F1-score test: {f1:.4f}")
print("Matriz de confusión:")
print(cm)

