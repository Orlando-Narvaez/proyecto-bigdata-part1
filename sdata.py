import nltk # imports the natural language toolkit
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
import pandas as pd
import numpy as np
import plotly
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# LOADING THE DATASET AND SEEING THE DETAILS
# Carga el archivo csv y sus primeras 5000 líneas
data = pd.read_csv('sdata.csv', nrows=5000)
data.head()

# Extrae solo la columna texto
AllReviews = data['text']
AllReviews.head()

# Imprime el primer comentario de la data extraida
AllReviews[0]

# Tokeniza los textos en oraciones
sentences = nltk.sent_tokenize(AllReviews[0])
for sentence in sentences:
    print(sentence)
    print()

# Las oraciones las vuelve en palabras
sentences = nltk.sent_tokenize(data['text'][1])
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    print(sentence)
    print(words)
    print()

# Importa y construye una nube de palabras
word_cloud_text = ''.join(data.text)
wordcloud = WordCloud(max_font_size=100, max_words=100, background_color="white",\
scale = 10,width=800, height=400).generate(word_cloud_text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Vectoriza
vec = CountVectorizer()
X = vec.fit_transform(AllReviews)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
df.head()

print(stopwords.words("english"))
print(stopwords.words("spanish"))

# Following code grabbed from:
# https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
# we will use it in our context to create some visualizations.
def get_top_n_words(corpus, n=1,k=1):
    vec = CountVectorizer(ngram_range=(k,k),stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# We start by getting a list of the most common words.
common_words = get_top_n_words(data['text'], 20,1)
for word, freq in common_words:
    print(word, freq)
    df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
    df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 words from all reviews')

plt.show()