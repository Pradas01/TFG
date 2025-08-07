import stanza
import re
import os
import Negacion
from Negacion import ejecutar_jar

# Cargar modelo de Stanza en español
nlp = stanza.Pipeline(lang='es', processors='tokenize,pos,lemma')

# Función para extraer términos relacionados con alergias
def extraer_termino_alergia(sent, alergia_keywords):
    for word in sent.words:
        for kw in alergia_keywords:
            if kw in word.lemma:
                return kw
    return None

# Función para normalizar texto
def normal_text(texto):
    doc = nlp(texto)
    text_res = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.text.isalpha():
                text_res.append(word.lemma.lower())
    return text_res

# Palabras clave
alergia_keywords = [
    # Síntomas digestivos
    "dolor", "abdominal", "diarrea", "náusea", "vómito", "malestar", "estomacal",

    # Síntomas cutáneos
    "urticaria", "roncha", "picazón", "eccema", "enrojecimiento", "hinchazón", "piel",

    # Problemas respiratorios
    "congestión", "nasal", "estornudo","estornudar", "garganta", "goteo", "respirar",
    "dificultad", "opresión", "pecho",

    # Anafilaxia
    "lengua", "labio", "caída", "presión", "mareo",
    "desmayo", "latido", "cardíaco", "acelerado", "consciencia",
]

causa_alergia_keywords = [
    # Acciones
    "consumo", "comer", "ingerir", "beber", "exposición", "contacto",
    "tocar", "inhalar", "respirar", "absorber", "manipular", "usar", "aplicar",
]

alergias_comunes_keywords = [
    # Medicamentos comunes que provocan alergias
    "penicilina", "antibiótico", "amoxicilina", "sulfa",
    "aspirina", "ibuprofeno", "naproxeno", "paracetamol",
    "anestésico", "lidocaína", "opiáceo", "morfina",
    "vacuna", "contraste", "radiológico", "yodo",

    # Alimentos comunes que provocan alergias
    "leche", "huevo", "pescado", "marisco", "crustáceo",
    "fruto seco", "almendra", "nuez", "cacahuate",
    "soja", "trigo", "gluten", "fresa", "kiwi",
    "plátano", "melón", "apio", "sésamo", "maíz",
    "chocolate", "café", "mostaza", "manzana",
    # Otros
    "látex"
]
#  detectar alergias en texto
def detectar_alergias(texto):

    lemas = normal_text(texto.lower())
    categorias = {
        'alergias': alergia_keywords,
        'causas': causa_alergia_keywords,
        'alergias_comunes': alergias_comunes_keywords
    }
    conteos = {key: 0 for key in categorias}
    frase_limpia = texto.replace("\n", " ").strip()

    for key, keywords in categorias.items():
        for kw in keywords:
            if kw in lemas and not ejecutar_jar(kw, frase_limpia):
                conteos[key] += lemas.count(kw)

    return 1 if sum(conteos.values()) >= 2 else 0

#  guardar textos orden deseado
def alphanumeric_key(file_name):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', file_name)]

folder_path = "../Entrenamiento/datos entrenamiento traducidos"
files = sorted(os.listdir(folder_path), key=alphanumeric_key)
Lista_resultado=[]
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            Lista_resultado.append(detectar_alergias(content))

print(Lista_resultado)