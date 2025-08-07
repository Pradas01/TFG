import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
from sklearn.metrics import ConfusionMatrixDisplay

# Datos de especialidades
especialidades = ['Alergología', 'Atención primaria', 'Endocrinología', 'Medicina interna']
cantidad = [216, 34, 13, 56]

# Crear gráfico de pastel
plt.figure(figsize=(6, 6))
plt.pie(cantidad, labels=especialidades, autopct='%1.1f%%', startangle=140)
plt.title('Distribución de informes clínicos por especialidad')

# Guardar imagen
plt.savefig('grafico_informes_especialidad.png', dpi=300, bbox_inches='tight')
plt.show()
# Datos para alergia
categorias = ['Alérgico', 'No Alérgico']

# Tu cadena de bits
data = "1111100110000111110000010110111111000101100010001000101110111001111100101100100011000000111110110000110011111111110110111111101001000001111001110011111110101001111110001111011111101000000111111111111011110000001011110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"

# Convertir la cadena en lista de enteros
bits = list(map(int, data))

# Contar alérgicos (1) y no alérgicos (0)
cantidad = [bits.count(1), bits.count(0)]

# Crear gráfico de pastel
plt.figure(figsize=(6, 6))
plt.pie(cantidad, labels=categorias, autopct='%1.1f%%', startangle=140, colors=['#ff6666', '#66b266'])
plt.title('Distribución de Alérgicos y No Alérgicos')

# Guardar imagen
plt.savefig('grafico_alergicos.png', dpi=300, bbox_inches='tight')
plt.show()


# Datos de la matriz
conf_matrix = np.array([[24 , 10],[ 5, 25]])


# Visualización
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Alergia', 'Alergia'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de confusión modelo multilingüe')
plt.savefig('matriz_SVC5.png', dpi=300, bbox_inches='tight')
plt.show()

import networkx as nx
import matplotlib.pyplot as plt

