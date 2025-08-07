import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Modelo base RoBERTa
model_name = "lcampillos/roberta-es-clinical-trials-ner"

# Cargar tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Función para tokenizar
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

# Leer CSVs
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

assert 'text' in train_df.columns and 'label' in train_df.columns, "Faltan columnas necesarias"

# Convertir a HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizar
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

#  modelo RoBERTa para clasificación
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Configurar entrenamiento
training_args = TrainingArguments(
    output_dir="./roberta-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Definir métricas personalizadas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        "accuracy": acc,
        "recall": rec,
        "f1": f1
    }

# Instanciar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Entrenar modelo
trainer.train()

# Evaluar con test
test_texts = list(test_df['text'])
test_labels = list(test_df['label'])

inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, axis=1).numpy()


acc = accuracy_score(test_df['label'], preds)
recall = recall_score(test_df['label'], preds)
f1 = f1_score(test_df['label'], preds)
cm = confusion_matrix(test_df['label'], preds)

print(f"Accuracy test: {acc:.4f}")
print(f"Recall test: {recall:.4f}")
print(f"F1-score test: {f1:.4f}")
print("Matriz de confusión:")
print(cm)

