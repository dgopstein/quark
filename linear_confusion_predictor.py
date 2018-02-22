# Take a code embedding and use it to predict confusion

import statsmodels as sm
from ggplot import *
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from gensim.models import Doc2Vec
from gensim_doc2vec import tokenize_source

import pandas as pd

pv_dm = Doc2Vec.load('tmp/linux_2018-02-21T15:57:56.584064_model_best.pkl')

snippet_results = pd.read_csv('tmp/results_normalized.csv')
snippet_questions = pd.read_csv('tmp/questions.csv')

vectors = [pv_dm.infer_vector(tokenize_source(source), steps=3, alpha=0.1) for source in snippet_questions['Code']]
#vectors[0][0]

tfs = snippet_results.groupby('CodeID')['Correct'].value_counts().unstack().fillna(0)
scores = tfs['T'] / (tfs['T'] + tfs['F'])

tfs = snippet_results.groupby('CodeID')['Correct'].value_counts().unstack().fillna(0)
tfs

train_vectors, test_vectors, train_scores, test_scores = train_test_split(vectors, scores, test_size=0.15, random_state=2 )

#alphas = []
#alpha_error = []
#for a in np.arange(0, .2, 0.001):

# sm_regr = sm.discrete.discrete_model.Logit(train_scores-0.0001, train_vectors)
# sm_regr.fit(cov_type='HC0')

#alphas.append(a)
#alpha_error.append(float(mean_squared_error(test_scores, regr.predict(test_vectors))))

#plt.plot(alphas, alpha_error)
#plt.show()

# GLM - Binomial/Logit
# sm_regr = sm.genmod.generalized_linear_model.GLM(train_scores, train_vectors, family=sm.genmod.families.family.Binomial(link=sm.genmod.families.links.logit))
# sm_res = sm_regr.fit_regularized(alpha=0.0000001)

# Lasso + Regularization
# sm_res = regr = skl.linear_model.Lasso(alpha=0)
# regr.fit(train_vectors, train_scores)

# Ridge + Regularization
sm_res = regr = skl.linear_model.Ridge(alpha=20000)
regr.fit(train_vectors, train_scores)

train_predicted = np.minimum(10, sm_res.predict(train_vectors))
test_predicted  = np.minimum(10, sm_res.predict(test_vectors))

print("train: ", np.corrcoef(train_scores, sm_res.predict(train_vectors))[0][1])
print("test: ", np.corrcoef(test_scores, sm_res.predict(test_vectors))[0][1])

predicted_confusion = pd.concat([
    pd.DataFrame.from_records({"observed": train_scores, "predicted": train_predicted, "inferred": False}),
    pd.DataFrame.from_records({"observed": test_scores, "predicted": test_predicted, "inferred": True})
    ])

ggplot(aes(x='observed', y='predicted', color='inferred'), data=predicted_confusion) +\
    geom_point() +\
    scale_color_brewer(type='qual', palette=2)

