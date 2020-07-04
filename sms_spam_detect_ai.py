#!/usr/bin/python3

import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier


# load the dataset of SMS messages
df = pd.read_table('dataset/SMSSpamCollection', header=None, encoding='utf-8')


# check class distribution
classes = df[0]
# convert class labels to binary values, 0 = ham and 1 = spam
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
# store the SMS message data
text_messages = df[1]

# use regular expressions to replace email addresses, URLs, phone numbers, other numbers,
#  punctuation, whitespace, and lower cases all the words.

processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
processed = processed.str.replace(r'£|\$', 'moneysymb')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumbr')
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')
processed = processed.str.lower()

# remove stop words from text messages
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

# Remove word stems using a Porter stemmer
ps = PorterStemmer()
processed = processed.apply(lambda x: ' '.join(
    ps.stem(term) for term in x.split()))

# create bag-of-words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = FreqDist(all_words)
# use the 1500 most common words as features [:1500]
word_features = list(all_words.keys())

# The find_features function will determine which of the 1500 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


# Now lets do it for all the messages
messages = zip(processed, Y)
# define a seed for reproducibility
seed = 1
np.random.seed = seed

# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]

# we can split the featuresets into training and testing datasets using sklearn
# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("[+] {} Accuracy: {}%".format(name, accuracy))

# make class label prediction for testing set
txt_features, labels = zip(*testing)

prediction = nltk_model.classify_many(txt_features)

# print a confusion matrix and a classification report
pd.DataFrame(
	confusion_matrix(labels, prediction),
    	index = [['actual', 'actual'], ['ham', 'spam']],
    	columns = [['predicted', 'predicted'], ['ham', 'spam']])

print(classification_report(labels, prediction))


