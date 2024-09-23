import pandas as pd
import matplotlib.pyplot as plt

# Carga el archivo csv
data = pd.read_csv('sdata.csv')
data.head()

# Extrae solo la columna texto
AllReviews = data['text']
AllReviews.head()

# Calcular el tamaño de las reseñas en términos de número de palabras
# utilizando la función `str.split()` para dividir las palabras, y luego usamos `len()` para contar las palabras
tamanios_resenas = AllReviews.apply(lambda x: len(x.split()))

# Encontrar la reseña más corta y más larga
resena_larga = AllReviews[tamanios_resenas.idxmax()]
resena_corta = AllReviews[tamanios_resenas.idxmin()]

# Diagrama de barras con la información calculada anteriormente 
plt.figure(figsize=(10,6))
plt.hist(tamanios_resenas, bins=30, edgecolor='black')
plt.title('Tamaño de las reseñas en funión del número de palabras')
plt.xlabel('Número de palabras')
plt.ylabel('Cantidad')

# Mostrar la gráfica
plt.show()