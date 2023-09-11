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

# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

reviews.shape

X_val=reviews['reviewText'][:19450]
y_val=reviews['overall'][:19450]

X=reviews['reviewText'][19450:]
y=reviews['overall'][19450:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# ## implement BAG OF WORDS
# countvector=CountVectorizer(ngram_range=(2,2))
# traindataset=countvector.fit_transform(X_train)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,1),min_df=0.25)
traindataset = vectorizer.fit_transform(X_train)
testdataset = vectorizer.transform(X_test)
valdataset = vectorizer.transform(X_val)
vectorizer.get_feature_names_out()

traindataset

# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=5,criterion='entropy')
randomclassifier.fit(traindataset,y_train)

def print_report(model):
  print('Accuracy Score for Train data')
  score=accuracy_score(y_train,model.predict(traindataset))
  print(score)
  print('Accuracy Score for Test data')
  score=accuracy_score(y_test,model.predict(testdataset))
  print(score)
  print('Accuracy Score for Validation data')
  score=accuracy_score(y_val,model.predict(valdataset))
  print(score)
  print('Classification Report for Train data')
  report=classification_report(y_train,model.predict(traindataset))
  print(report)
  print('Classification Report for Test data')
  report=classification_report(y_test,model.predict(testdataset))
  print(report)
  print('Classification Report for Validation data')
  report=classification_report(y_val,model.predict(valdataset))
  print(report)

print_report(randomclassifier)

# implement Logistic Regression 
from sklearn.linear_model import LogisticRegression  
lgr_classifier= LogisticRegression(random_state=0)  
lgr_classifier.fit(traindataset, y_train)

print_report(lgr_classifier)

#Fitting Decision Tree classifier to the training set  
from sklearn.tree import DecisionTreeClassifier  
ds_classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
ds_classifier.fit(traindataset, y_train)

print_report(ds_classifier)

# from sklearn.svm import SVC # "Support vector classifier"  
# svm_classifier = SVC(kernel='rbf', random_state=0)  
# svm_classifier.fit(traindataset, y_train)

# print_report(svm_classifier)

import xgboost as xgb
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(traindataset,y_train)

print_report(xgb_classifier)

