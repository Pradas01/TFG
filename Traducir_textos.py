import asyncio
from googletrans import Translator
import re
import os
import csv
# Función asincrónica para traducir textos
async def traducir_textos(lista_textos):
    translator = Translator()

    # Lista para almacenar las traducciones
    traducciones = []

    # Traducir cada texto en la lista
    for texto in lista_textos:
        traduccion = await translator.translate(texto, src='es', dest='en')
        traducciones.append(traduccion.text)

    # Devolver las traducciones
    return traducciones

Lista_textos = []

# Lista de textos en español
folder_path = "Entrenamiento/datos entrenamiento traducidos"

def alphanumeric_key(file_name):
    # Esta expresión regular divide el nombre en partes alfabéticas y numéricas
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', file_name)]

files = sorted(os.listdir(folder_path), key=alphanumeric_key)
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            Lista_textos .append(content)


def guardar_traducciones_en_csv(traducciones, nombre_archivo='traducciones.csv'):
    with open(nombre_archivo, mode='w', newline='', encoding='utf-8') as archivo:
        writer = csv.writer(archivo)
        writer.writerow(['Traducción'])  # Solo la columna de "Traducción"
        for traducido in traducciones:
            writer.writerow([traducido])  # Escribe solo la traducción

    print(f" Traducciones guardadas en {nombre_archivo}")
# Ejecutar la traducción asincrónica
async def main():
    traducciones = await traducir_textos(Lista_textos)
    guardar_traducciones_en_csv(traducciones)
    return traducciones


# Ejecutar el proceso asincrónico
asyncio.run(main())