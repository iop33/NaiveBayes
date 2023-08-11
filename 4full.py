!pip install nltk
import nltk
nltk.download()
# TODO popuniti kodom za problem 4
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize, regexp_tokenize
from nltk import FreqDist
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

data = pd.read_csv('/content/disaster-tweets.csv')

X = data.iloc[:, 3].values  
y = data.iloc[:, 4].values  
X_clean= []

stop_punc = set(stopwords.words('english')).union(set(punctuation))

for doc in X:
  words = wordpunct_tokenize(doc)
  words_lower = [w.lower() for w in words]
  words_filtered = [w for w in words_lower if w not in stop_punc]

  porter= PorterStemmer()
  words_stemmed = [porter.stem(w) for w in words_filtered]

  X_clean.append(words_stemmed)
  

# Kreiramo vokabular
vocab_set = set()
for doc in X_clean:
  for word in doc:
    vocab_set.add(word)
vocab = list(vocab_set)

class MultinomialNaiveBayes:
  def __init__(self, nb_classes, nb_words, pseudocount):
    self.nb_classes = nb_classes
    self.nb_words = nb_words
    self.pseudocount = pseudocount
  
  def metrika(self,occs):
          metrika_max =[(-1,0),(-1,0),(-1,0),(-1,0),(-1,0)]
          metrika_min =[(sys.maxsize, 0),(sys.maxsize, 0),(sys.maxsize, 0),(sys.maxsize, 0),(sys.maxsize, 0)]
          for i in range(len(occs[0])):
                if occs[1][i] >= 10 and occs[0][i] >= 10:
                    val = occs[0][i] / occs[1][i]
                    if val < metrika_min[0][0]:
                      metrika_min[0] = (val, i)
                      metrika_min = sorted(metrika_min, reverse= True)
                    if val > metrika_max[0][0]:
                      metrika_max[0] = (val, i)
                      metrika_max = sorted(metrika_max)
          print("metrika_max:", metrika_max)
          print("metrika_min:", metrika_min)
          for w in metrika_max:
            print("WORD MAX:", vocab[w[1]])
          for w in metrika_min:
            print("WORD MIN:", vocab[w[1]])

  def fit(self, X, Y):
    nb_examples = len(X)

    # Racunamo P(Klasa) - priors
    # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
    # broja u intervalu [0, maksimalni broj u listi]
    self.priors = np.bincount(Y) / nb_examples
   
    occs = np.zeros((self.nb_classes, self.nb_words)) 
    for i in range(nb_examples):
      c = Y[i]                                        
      for w in range(self.nb_words):
        cnt = X[i][w]                                 
        occs[c][w] += cnt
    
    top_poz_5_common = sorted(range(len(occs[1])), key=lambda k: occs[1][k], reverse=True)[:5]
    top_neg_5_common = sorted(range(len(occs[0])), key=lambda k: occs[0][k], reverse=True)[:5]

    self.metrika(occs)

    for i in range(len(top_neg_5_common)):
      print(i+1, " rec  pozitivni :", vocab[top_poz_5_common[i]])
      print(i+1, " rec negativni :", vocab[top_neg_5_common[i]])

   

    # Racunamo P(Rec_i|Klasa) - likelihoods
    self.like = np.zeros((self.nb_classes, self.nb_words))
    for c in range(self.nb_classes):
      for w in range(self.nb_words):
        up = occs[c][w] + self.pseudocount
        down = np.sum(occs[c]) + self.nb_words*self.pseudocount
        self.like[c][w] = up / down
  
          
  def predict(self, bow):
    # Racunamo P(Klasa|bow) za svaku klasu
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = np.log(self.priors[c])
      for w in range(self.nb_words):
        cnt = bow[w]
        prob += cnt * np.log(self.like[c][w])
      probs[c] = prob
    prediction = np.argmax(probs)
    return prediction

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=0)

X_train_bow_matrix = np.zeros((len(X_train), len(vocab)))

for i, doc in enumerate(X_train):
    X_train_bow = np.zeros(len(vocab))
    for word in doc:
        if word in vocab:
            X_train_bow[vocab.index(word)] += 1
    X_train_bow_matrix[i] = X_train_bow



model = MultinomialNaiveBayes(nb_classes=2, nb_words= len(vocab), pseudocount=1)
model.fit(X_train_bow_matrix, y_train)


y_pred = []
for doc in X_test:
    bow = np.zeros(len(vocab))
    for word in doc:
        if word in vocab:
            bow[vocab.index(word)] += 1
    prediction = model.predict(bow)
    y_pred.append(prediction)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print('Accuracy is:', accuracy)