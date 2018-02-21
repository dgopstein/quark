# Take a code embedding and use it to predict confusion

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import pandas

pv_dm = Doc2Vec.load('tmp/linux_2018-02-20T18:32:02.694306_model_best.pkl')

sources = ["int main() {\n  int x = 2; x += 3; }", "int main() {\n  int x = 4; x += 5; }", "int main() {\n  int y = 2; y += 3; }"]

vectors = [pv_dm.infer_vector(tokenize_source(source), steps=3, alpha=0.1) for source in sources]

snippet_results = pandas.read_csv('results_normalized.csv')

snippet_results.columns

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(vectors, [1, 2, 3])

# Make predictions using the testing set
regr.predict([vectors[2]])
