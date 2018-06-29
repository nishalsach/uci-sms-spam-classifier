import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

df.columns = ['Labels', 'Text']

from sklearn.preprocessing import LabelEncoder
result_enc = LabelEncoder()
df["Labels"] = result_enc.fit_transform(df["Labels"])
# 1 means SPAM


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

Y = df["Labels"].values
X = df['Text']

#TFIDF Vectorizer is used
#tfidf = TfidfVectorizer(decode_error='ignore', stop_words = ['in', 'a', '.'])
#X = tfidf.fit_transform(df['Text'])

#Count Vectorizer is used
count_vectorizer = CountVectorizer(decode_error='ignore', stop_words = ['in', 'a', '.'])
X = count_vectorizer.fit_transform(df['Text'])

#METHOD 1: NORMAL CLASSIFICATION 

# split up the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

clf = MultinomialNB(fit_prior = False)
clf.fit(Xtrain, Ytrain)
print("train score:", clf.score(Xtrain, Ytrain))
print("test score:", clf.score(Xtest, Ytest))

from sklearn import metrics
predicted = clf.predict(Xtest)
print(metrics.classification_report(Ytest, predicted, target_names=['Not Spam', 'Spam']))

