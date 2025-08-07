import csv
import random

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import os


folder_path = "Entrenamiento/datos entrenamiento traducidos"

# Cargar modelo y tokenizador desde la carpeta guardada
model = AutoModelForSequenceClassification.from_pretrained("./modelo_multilenguaje_final")
tokenizer = AutoTokenizer.from_pretrained("./modelo_multilenguaje_final")
Lista_textos = []

def alphanumeric_key(file_name):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', file_name)]

# Leer textos desde la carpeta
files = sorted(os.listdir(folder_path), key=alphanumeric_key)

for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

            Lista_textos.append(content)
traducciones = []

# Abrir el archivo CSV para lectura
with open('Propuesta 5 Fine tuning/traducciones.csv', mode='r', encoding='utf-8') as archivo:
    lector = csv.reader(archivo)
    next(lector)
    for fila in lector:
        traducciones.append(fila[0])

def Leer_excel():
    File = pd.read_excel('Entrenamiento/tfg.xlsx', header=None)
    primera_columna = File.iloc[:, 0].tolist()
    segunda_columna = File.iloc[:, 1].tolist()
    return primera_columna

etiquetas = Leer_excel()

data = list(zip(Lista_textos, etiquetas))

output_dir = 'Entrenamiento/fine-tuning/'
os.makedirs(output_dir, exist_ok=True)

random.seed(42)
random.shuffle(data)

total = len(data)
train_end = int(0.8 * total)
val_end = int(0.9 * total)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]


train_notas, train_etiquetas = zip(*train_data)
val_notas, val_etiquetas = zip(*val_data)
test_notas, test_etiquetas = zip(*test_data)

# Crear DataFrames
df_train = pd.DataFrame({'text': train_notas, 'label': train_etiquetas})
df_val = pd.DataFrame({'text': val_notas, 'label': val_etiquetas})
df_test = pd.DataFrame({'text': test_notas, 'label': test_etiquetas})

# Guardar en CSV (sin índice)
df_train.to_csv('train.csv', index=False)
df_val.to_csv('val.csv', index=False)
df_test.to_csv('test.csv', index=False)