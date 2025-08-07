import subprocess
import os

# Definir rutas relativas
ruta_jar = "../NegEx-MES/smn/main/smn.jar"
ruta_config_files = "../NegEx-MES/smn/config_files/"
ruta_in_text_file = "../NegEx-MES/smn/in/in.txt"
ruta_out_text_file = "../NegEx-MES/smn/out/out.txt"

def leer_resultado_out():
    ruta_fichero = ruta_out_text_file
    try:
        with open(ruta_fichero, "r", encoding="utf-8") as archivo:
            contenido = archivo.readlines()
        return contenido
    except FileNotFoundError:
        print(f"El archivo {ruta_fichero} no existe. Verifica la ejecución del JAR.")
        return []

def sobrescribir_fichero_in(identificador, frase):
    try:
        ruta_fichero = ruta_in_text_file
        linea = f"1\t{identificador}\t\"{frase}\"\n"
        with open(ruta_fichero, "w", encoding="utf-8") as archivo:
            archivo.write(linea)
    except Exception as e:
        print(f"Error al sobrescribir el archivo {ruta_fichero}: {e}")

def ejecutar_jar(termino, frase):
    opciones = [
        "-displayon", "true",
        "-language", "SPANISH",
        "-answerOptionYes", "true",
        "-isOuputFileGenerated", "true",
        "-lemmaConfigFiles", "false",
        "-routeConfigFiles", ruta_config_files,
        "-routeInTextFile", ruta_in_text_file,
        "-routeOutTextFile", ruta_out_text_file
    ]
    comando = ["java", "-jar", ruta_jar] + opciones
    sobrescribir_fichero_in(termino,frase)
    resultado = subprocess.run(comando, capture_output=True, text=True, check=True)
    resultado_out = leer_resultado_out()

    for linea in resultado_out:
        if "Negated" in linea:
            return True
    return False



