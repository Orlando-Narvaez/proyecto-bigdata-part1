import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Función que genera una nube de palabras basada en el número de estrellas de las reseñas
def generate_word_cloud(data, star_rating):
    # Filtrar las reseñas según el número de estrellas
    filtered_reviews = data[data['stars'] == star_rating]
    
    # Unir todas las reseñas en un solo texto
    reviews_text = ' '.join(filtered_reviews['text'].astype(str))

    # Definir las stopwords en inglés
    english_stopwords = set(stopwords.words('english'))
    
    # Generar la nube de palabras
    word_cloud = WordCloud(stopwords=english_stopwords, max_font_size=100, max_words=100, background_color="white", scale=10, width=800, height=400).generate(reviews_text)
    
    # Mostrar la nube de palabras en pantalla
    plt.figure(figsize=(10, 5))
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Cargar el archivo CSV que contiene las reseñas
data = pd.read_csv('sdata.csv')

# Llamar a la función para generar la nube de palabras de las reseñas de 1 estrella
generate_word_cloud(data, 1)