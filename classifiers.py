import pickle
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold,cross_val_score,learning_curve


X_train, y_train, X_test, y_test = train_test_split(features, labels, test_size=0.15)

# Vectoriser les données avec un TFIDF.
vectorizer = TfidfVectorizer(
)
vectorizer = vectorizer.fit(X_train)
vectors = vectorizer.transform(X_train)

# Définir plusieurs classificateurs différents.
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier(n_neighbors=49)
mnb = MultinomialNB(alpha=0.2)
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=31, random_state=111)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

clf = GridSearchCV(svc, parameters)

# Définir un dictionnaire de classificateurs pour le grid search.
classifiers = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc}

def train(classifier, features, targets):
    classifier.fit(features, targets)

def predict(classifier, features):
    return (classifier.predict(features))

# Entrainer les différents classificateurs.
pred_scores_word_vectors = []
for k,v in classifiers.items():
    train(v, X_train, y_train)
    pred = predict(v, X_test)
    pred_scores_word_vectors.append((k, [accuracy_score(y_test , pred)]))

print(pred_scores_word_vectors)
