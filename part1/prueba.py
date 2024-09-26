import pandas as pd
import nltk
from collections import Counter
import numpy as np

# Descargar los paquetes necesarios de NLTK para tokenización y stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Leer el archivo CSV que contiene las reseñas
data = pd.read_csv(r'C:/Users/ORLANDO/Documents/Git/BigData/ProyectoFinalPart1-DanielEstebanTusarmaG-OrlandoNarvaezB/part2/sdata.scv')

# Seleccionar únicamente la columna que contiene el texto de las reseñas
reviews = data['text']

# Convertir todas las reseñas en palabras individuales, ignorando las stopwords
stop_words = set(stopwords.words('english'))

# Unir todo el texto en una sola cadena para facilitar la tokenización
combined_text = ' '.join(reviews.astype(str))

# Dividir el texto en palabras (tokens), eliminando puntuación y stopwords
tokens = nltk.word_tokenize(combined_text.lower())
filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

# Contar cuántas veces aparece cada palabra en las reseñas
word_frequencies = Counter(filtered_tokens)

# Sumar el total de palabras para referencia
total_word_count = sum(word_frequencies.values())

# Identificar las palabras más y menos comunes en el 1% superior e inferior
sorted_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

# Determinar la cantidad de palabras que forman el 1% superior e inferior
top_1_percent_index = int(len(sorted_frequencies) * 0.01)
low_1_percent_index = int(len(sorted_frequencies) * 0.99)

# Extraer las palabras más frecuentes y menos frecuentes
top_1_percent_words = sorted_frequencies[:top_1_percent_index]
low_1_percent_words = sorted_frequencies[low_1_percent_index:]

# Mostrar las palabras más repetidas
print("High-frequency words (Top 1%):")
for word, count in top_1_percent_words:
    print(f"{word}: {count}")

# Mostrar las palabras menos repetidas
print("\nLow-frequency words (Bottom 1%):")
for word, count in low_1_percent_words:
    print(f"{word}: {count}")