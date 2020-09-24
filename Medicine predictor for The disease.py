# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 12:26:24 2020

@author: shaur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


os.chdir("D:\data sets")

dataset = pd.read_csv('web_sacrapped.csv', delimiter = ',', quoting = 3)
a=dataset.iloc[:,:2]
x=a.iloc[:,:1]
y=a.iloc[:,1:2]



import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 213):
  review = re.sub('[^a-zA-Z]', ' ', x['a'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  review = ' '.join(review)
  corpus.append(review)

corpus1 = []
for i in range(0, 213):
  review1 = re.sub('[^a-zA-Z]', ' ', y['b'][i])
  review1 = review1.lower()
  review1 = review1.split()
  ps = PorterStemmer()
 
  review1 = ' '.join(review1)
  corpus1.append(review1)
  
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


