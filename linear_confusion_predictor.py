# Take a code embedding and use it to predict confusion

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from gensim.models import Doc2Vec
from gensim_doc2vec import tokenize_source

import pandas as pd

pv_dm = Doc2Vec.load('tmp/linux_2018-02-20T18:32:02.694306_model_best.pkl')

snippet_results = pd.read_csv('tmp/results_normalized.csv')
snippet_questions = pd.read_csv('tmp/questions.csv')

vectors = [pv_dm.infer_vector(tokenize_source(source), steps=3, alpha=0.1) for source in snippet_questions['Code']]
#vectors[0][0]

tfs = snippet_results.groupby('CodeID')['Correct'].value_counts().unstack().fillna(0)
scores = tfs['T'] / (tfs['T'] + tfs['F'])

regr = linear_model.LinearRegression()

train_vectors = vectors[::2]
test_vectors = vectors[1::2]
#train_vectors[0][0]

train_scores = scores[::2]
test_scores = scores[1::2]

# Train the model using the training sets
regr.fit(train_vectors, train_scores)

# Make predictions using the testing set
mean_squared_error(train_scores, regr.predict(train_vectors))
mean_squared_error(test_scores, regr.predict(test_vectors))
