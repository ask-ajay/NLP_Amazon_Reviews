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

from google.colab import drive
drive.mount('/content/drive')

reviews = []
for line in open('/content/drive/MyDrive/Cell_Phones_and_Accessories_5.json', 'r'):
    reviews.append(json.loads(line))

reviews[0:1]

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

stop_words = stopwords.words('english')

reviews['reviewText'] = reviews['reviewText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
print(reviews)

reviews.shape

X_val=reviews['reviewText'][:19450]
y_val=reviews['overall'][:19450]

X=reviews['reviewText'][19450:]
y=reviews['overall'][19450:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

X_train

len(X_train)

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(ngram_range=(1,1),min_df=0.25)
# traindataset = vectorizer.fit_transform(X_train)
# testdataset = vectorizer.transform(X_test)
# valdataset = vectorizer.transform(X_val)
# vectorizer.get_feature_names_out()

# # pip install nltk
# from nltk.tokenize import word_tokenize

# reviews['reviewText'] = reviews.apply(lambda row: nltk.word_tokenize(row['reviewText']), axis=1)
# print(reviews['reviewText'])

### Now the reviews are fairly clean, we will tokenize the reviews and get word embeddings for these
tokenizer=Tokenizer(num_words=10000,lower=True,oov_token='UNK')# num_words= top 'n' most frequent words from the corpus
### Now lets tokenize the reviews
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_test)
tokenizer.fit_on_texts(X_val)

tokenizer.word_index # the index of the top 'n' most frequent words from the corpus is returned

len(tokenizer.word_index)

len(X_train)

#### Now we need embeddings for these words  
## Get the glove vectors
embeddings_index= dict()
glove= open('/content/drive/MyDrive/glove.6B.100d.txt','r',encoding='utf-8')
for line in glove:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs

glove.close()

len(embeddings_index)

### Create an embedding matrix for the vocabulary created for the reviews 
# vocab= len(tokenizer.word_index)+1
# embedding_matrix = np.zeros((vocab, 100)) 
# for word, i in tokenizer.word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

# len(embedding_matrix)

# embedding_matrix.shape


train_indices=tokenizer.texts_to_sequences(X_train)

from tensorflow.keras.preprocessing.text import one_hot
vocab= len(tokenizer.word_index)+1
one_hot_rep=[one_hot(words,vocab) for words in X_train]
one_hot_rep[0:2]

from tensorflow.keras.preprocessing.sequence import pad_sequences
embedded_doc=pad_sequences(one_hot_rep,padding='pre',maxlen=100)
embedded_doc[0]

#####   RNN with 1 layer, 20 neurons
from keras.layers import RNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM, Dropout, BatchNormalization
import numpy as np

embedding_vector_features=100 
model1= Sequential()
model1.add(Embedding(input_dim=vocab,output_dim=embedding_vector_features,input_length=100))
model1.add(SimpleRNN(20))
model1.add(Dense(10,activation='softmax'))
model1.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())

test_indices=tokenizer.texts_to_sequences(X_test)
one_hot_rep_test=[one_hot(words,vocab) for words in X_test]
embedded_doc_test=pad_sequences(one_hot_rep_test,padding='pre',maxlen=100)

import numpy as np
X_train=np.array(embedded_doc)
X_test=np.array(embedded_doc_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

X_train.shape,y_train.shape,X_test.shape,y_test.shape

model1.fit(X_train,y_train,batch_size=32,epochs=5)
# Final evaluation of the model
scores = model1.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy: %.2f%%" % (scores[1]*100))

#####   RNN with 2 layers, 50,20 neurons each
from keras.layers import RNN
model2=Sequential()
model2.add(Embedding(input_dim=vocab,output_dim=embedding_vector_features,input_length=100))
model2.add(SimpleRNN(50, return_sequences = True))
model2.add(SimpleRNN(20, return_sequences = False))
model2.add(Dense(10,activation='softmax'))
model2.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model2.summary())
model2.fit(X_train,y_train,batch_size=32,epochs=5)

scores2 = model2.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy: %.2f%%" % (scores2[1]*100))

#####   RNN with 3 layers, 100,50,20 neurons each
from keras.layers import RNN
model3=Sequential()
model3.add(Embedding(input_dim=vocab,output_dim=embedding_vector_features,input_length=100))
model3.add(SimpleRNN(100, return_sequences = True))
model3.add(SimpleRNN(50, return_sequences = True))
model3.add(SimpleRNN(20, return_sequences = False))
model3.add(Dense(10,activation='softmax'))
model3.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model3.summary())
model3.fit(X_train,y_train,batch_size=32,epochs=5)

scores3 = model3.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy: %.2f%%" % (scores3[1]*100))

"""# LSTM """

from keras.layers import LSTM
from keras.layers.embeddings import Embedding
#########MODEL 1 (LSTM with 1 hidden layer and 100 neurons)
model4= Sequential()
model4.add(Embedding(input_dim=vocab,output_dim=embedding_vector_features,input_length=100))
#adding a LSTM layer of dim 1--
model4.add(LSTM(100))
#adding the final output activation with activation function of softmax
model4.add(Dense(10, activation='softmax'));
model4.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model4.summary())
model4.fit(X_train,y_train,batch_size=32,epochs=5)

scores4 = model4.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy: %.2f%%" % (scores4[1]*100))

#####        MODEL 2 : LSTM with 2 layers, 200 neurons
model5= Sequential()
model5.add(Embedding(input_dim=vocab,output_dim=embedding_vector_features,input_length=100))
#adding a LSTM layer of dim 1--
model5.add(LSTM(200, return_sequences=True));
model5.add(LSTM(100, return_sequences=False));
#adding a dense layer with activation function of relu
model5.add(Dense(100, activation='relu'));#best 50,relu
#adding the final output activation with activation function of softmax
model5.add(Dense(50, activation='softmax'));
model5.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model5.summary())

model5.fit(X_train,y_train,batch_size=32,epochs=5)

scores5 = model5.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy: %.2f%%" % (scores5[1]*100))

"""# Bidirectional LSTM"""

#######        MODEL -3 Bidirectional LSTM
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

model6= Sequential()
model6.add(Embedding(input_dim=vocab,output_dim=embedding_vector_features,input_length=100))
#adding a LSTM layer of dim 1--
model6.add(Bidirectional(LSTM(50, return_sequences=True)))
model6.add(Bidirectional(LSTM(50, return_sequences=False)))
#adding a dense layer with activation function of relu
model6.add(Dense(50, activation='relu'))
model6.add(BatchNormalization())
#adding the final output activation with activation function of softmax
model6.add(Dense(10, activation='softmax'))
model6.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model6.summary())

model6.fit(X_train,y_train,batch_size=32,epochs=5)

model6 = model6.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy: %.2f%%" % (scores6[1]*100))