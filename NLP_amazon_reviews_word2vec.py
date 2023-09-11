
"""NLP_Final_Hackathon_Amazon_reviews_Word2Vec_2nd_Oct_2022.ipynb

"""

### Load all the required libraries/modules
import os
import pandas as pd
import numpy as np
import json

import warnings
warnings.filterwarnings('ignore')

## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

##tensorflow modules
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

## text preprocessing modules
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Word2Vec
import re
import nltk
nltk.download('punkt')

reviews = []
for line in open('/content/drive/MyDrive/Cell_Phones_and_Accessories_5.json', 'r'):
    reviews.append(json.loads(line))

reviews[0:5]

reviews=pd.DataFrame(reviews)

reviews.head()

reviews=reviews[['reviewText','overall']]

reviews.tail()

# Removing punctuations
reviews.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Convertng headlines to lower case
reviews['reviewText']=reviews['reviewText'].str.lower()
reviews.head()

# pip install nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words('english')
# word_tokens = word_tokenize(reviews['reviewText'])
  
reviews['reviewText'] = reviews['reviewText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
print(reviews)

reviews

# pip install nltk
from nltk.tokenize import word_tokenize

reviews['reviewText'] = reviews.apply(lambda row: nltk.word_tokenize(row['reviewText']), axis=1)
print(reviews['reviewText'])

X_val=reviews['reviewText'][:19450]
y_val=reviews['overall'][:19450]

X=reviews['reviewText'][19450:]
y=reviews['overall'][19450:]

import gensim
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# Create CBOW model
model1 = Word2Vec(X_train, min_count = 2,size = 100, window = 3,sg=1)
# Create CBOW model
model2 = Word2Vec(X_train, min_count = 2,size = 100, window = 7,sg=1)
# Create CBOW model
model3 = Word2Vec(X_train, min_count = 5,size = 100, window = 3,sg=1)
# Create CBOW model
model4 = Word2Vec(X_train, min_count = 5,size = 100, window = 7,sg=1)
Create CBOW model
model5 = Word2Vec(X_train, min_count = 2,size = 200, window = 3,sg=1)
# Create CBOW model
model6 = Word2Vec(X_train, min_count = 2,size = 200, window = 7,sg=1)
# Create CBOW model
model7 = Word2Vec(X_train min_count = 5,size = 200, window = 3,sg=1)
# Create CBOW model
model8 = Word2Vec(X_train min_count = 5,size = 200, window = 7,sg=1)
# Create CBOW model
model9 = Word2Vec(X_train min_count = 2,size = 300, window = 3,sg=1)
# Create CBOW model
model10 = Word2Vec(X_train min_count = 2,size = 300, window = 7,sg=1)
# Create CBOW model
model11 = Word2Vec(X_train min_count = 5,size = 300, window = 3,sg=1)
# Create CBOW model
model12 = Word2Vec(X_train min_count = 5,size = 300, window = 7,sg=1)

vocabulary = model1.wv.vocab
print(vocabulary)

model1.wv.most_similar('received')

model1.vector_size

len(model1.wv.vocab)

