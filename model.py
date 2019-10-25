# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:08:32 2019

@author: mabasha
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle

dataset=pd.read_csv("spam.csv",encoding='latin-1')
dataset.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

dataset['label']=dataset['v1'].map({'ham':0,'spam':1})
X=dataset['v2']
y=dataset['label']

cv=CountVectorizer()
X=cv.fit_transform(X)

pickle.dump(cv,open('transform.pkl','wb'))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

filename='nlp_model.pkl'
pickle.dump(clf,open(filename,'wb'))