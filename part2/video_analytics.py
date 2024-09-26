import sys
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

nltk.download('stopwords')
nltk.download('punkt')

# Configurar salida en UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Cargar el archivo CSV
video_data = pd.read_csv(r'C:/Users/ORLANDO/Documents/Git/BigData/ProyectoFinalPart1-DanielEstebanTusarmaG-OrlandoNarvaezB/part2/CAVideos.csv', nrows=5000)

# Filtrar títulos no válidos y duplicados
video_data = video_data[video_data['title'].str.len() > 5].drop_duplicates(subset=['title'])

# Extraer los títulos de los videos y los coloca en minuscula
titles_list = video_data['title'].astype(str).str.lower()

# Función para manejar la selección de un título en el combobox
def on_title_select(event):
    global selected_title
    selected_title = title_combobox.get()  # Guardar el título elegido
    window.quit()  # Cerrar la ventana al confirmar

# Crear interfaz gráfica con Tkinter
window = tk.Tk()
window.title("Select a Title")
window.geometry("600x400")  # Ajustar el tamaño de la ventana según la imagen

# Cargar imagen de fondo
try:
    image = Image.open(r'C:/Users/ORLANDO/Documents/Git/BigData/ProyectoFinalPart1-DanielEstebanTusarmaG-OrlandoNarvaezB/part2/background.jpg')  # Ruta absoluta corregida
    image = image.resize((600, 400), Image.LANCZOS)  # Redimensionar la imagen usando LANCZOS
    bg_image = ImageTk.PhotoImage(image)
    bg_label = tk.Label(window, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Colocar la imagen como fondo
except Exception as e:
    print("Error al cargar la imagen de fondo:", e)

# Etiqueta descriptiva en negrilla
label = ttk.Label(window, text="Seleccionar el título que desea comparar:", background="#FFFFFF", font=("Arial", 10, "bold"))
label.pack(pady=10)

# Crear un combobox con los primeros 50 títulos
title_combobox = ttk.Combobox(window, values=[f"{i + 1}: {title}" for i, title in enumerate(titles_list[:50])], width=80)
title_combobox.pack(pady=10)
title_combobox.bind("<<ComboboxSelected>>", on_title_select)  # Detectar selección

# Botón para confirmar la selección en negrilla
select_button = ttk.Button(window, text="Confirm Selection", command=on_title_select, style='Bold.TButton')
select_button.pack(pady=10)

# Estilo del botón en negrilla
style = ttk.Style()
style.configure('Bold.TButton', font=('Arial', 10, 'bold'))

# Ejecutar la ventana emergente
window.mainloop()

# Verificar si el título fue seleccionado
if 'selected_title' not in globals():
    print("No title was selected.")
    exit()

# Mostrar el título seleccionado
print(f"Selected title: {selected_title}")

# Limpiar el título seleccionado (remover índice)
selected_title = selected_title.split(": ", 1)[1]

# Definir stopwords en inglés
stop_words = set(stopwords.words('english'))

# Convertir el conjunto de stopwords en una lista
stop_words = list(stop_words)

# Inicializar el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)

# Calcular la matriz TF-IDF para todos los títulos
tfidf_matrix = tfidf_vectorizer.fit_transform(titles_list)

# Transformar el título seleccionado en su representación TF-IDF
title_vector = tfidf_vectorizer.transform([selected_title])

# Calcular la similitud de coseno entre el título seleccionado y los demás
cosine_sim = cosine_similarity(title_vector, tfidf_matrix).flatten()

# Ordenar los índices de los títulos más similares (excluyendo el título base)
similar_indices = cosine_sim.argsort()[::-1]
similar_indices = [idx for idx in similar_indices if titles_list.iloc[idx] != selected_title]

# Obtener los 10 títulos más similares
top_10_indices = similar_indices[:10]

# Extraer los títulos y sus similitudes correspondientes
recommended_titles = titles_list.iloc[top_10_indices]
recommended_similarities = cosine_sim[top_10_indices]

# Mostrar los 10 títulos más similares
print("\nLos 10 títulos más similares son:")
print(recommended_titles)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.barh(recommended_titles[::-1], recommended_similarities[::-1], color='lightcoral')
plt.xlabel('Cosine Similarity')
plt.title('Top 10 Similar Titles')
plt.tight_layout()
plt.show()