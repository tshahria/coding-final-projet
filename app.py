import json
import pandas as pd
import os
import pickle
from flask import Flask,render_template, request,json


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


import numpy as np


ask = pd.read_csv("./final_ask.csv", header=None)
ask = ask.iloc[1:].drop_duplicates(subset=[0])
college = pd.read_csv("./college_csv.csv", header=None).drop_duplicates(subset=[0])
combined = pd.read_csv("./combined_csv.csv", header=None).drop_duplicates(subset=[0])
support = pd.read_csv("./support_csv.csv", header=None).drop_duplicates(subset=[0])


#concats all the data
df = pd.concat([ask, college, support[:200000], combined])


#takes a sample from the data. using all of it
df = df.sample(frac=1).reset_index(drop=True)


##it converts every word to a vector
count_vect = CountVectorizer(stop_words='english')


#it fits and trasnforms the data to the requirements
X_train_counts = count_vect.fit_transform(df.iloc[0:300000, 0])


#helps us weight the words by the frequency
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


#classification model/ naive bayes
clf = MultinomialNB().fit(X_train_tfidf, df.iloc[0:300000, 1])

#helps us predict
predicted = clf.predict(count_vect.transform(df.iloc[400001:, 0]))



np.mean(predicted == df.iloc[400001:, 1])


pickle.dump (clf, open ('trained_nlp.p', 'wb'))


app = Flask(__name__)

@app.route('/')
def signUp():
    return render_template('signUp.html', rank= "type in text!")

@app.route('/signUpUser', methods=['POST'])
def signUpUser():
    user =  request.form['text'];
    result = int(clf.predict(count_vect.transform([str(user)]))[0])
    if (result == 0):
        return json.dumps({'status':'This doesn\'t count as bullying!'});
    else:
        return json.dumps({'status': 'This  counts as bullying!'});


if __name__=="__main__":
    app.run()
