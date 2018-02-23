# Take a code embedding and use it to predict confusion

import statsmodels as sm
from ggplot import *
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from gensim.models import Doc2Vec
from gensim_doc2vec import tokenize_source

import pandas as pd

#pv_dm = Doc2Vec.load('tmp/linux_2018-02-21T23:25:02.016544_model_final.pkl') # 50dim
pv_dm = Doc2Vec.load('tmp/linux_2018-02-21T23:34:44.493096_model_final.pkl') # 30dim

snippet_results = pd.read_csv('tmp/results_normalized.csv')
snippet_questions = pd.read_csv('tmp/questions.csv')

snippet_questions['ID'] = 's' + snippet_questions['ID'].astype('str')
snippet_results['CodeID'] = 's' + snippet_results['CodeID'].astype('str')

context_results = pd.read_csv('tmp/context_study_usercode.csv')
context_questions = pd.read_csv('tmp/context_study_code.csv')

context_questions['ID'] = 'c' + context_questions['ID'].astype('str')
context_results['CodeID'] = 'c' + context_results['CodeID'].astype('str')

combined_questions = pd.concat([snippet_questions[['ID', 'Code']],
                                context_questions[['ID', 'Code']]])
combined_results = pd.concat([snippet_results[['CodeID', 'Correct']],
                              context_results[['CodeID', 'Correct']]])

combined_tfs = combined_results.groupby('CodeID')['Correct'].value_counts().unstack().fillna(0)
combined_scores = combined_tfs['T'] / (combined_tfs['T'] + combined_tfs['F'])

code_scores = pd.merge(combined_questions, combined_scores.to_frame('Score'), left_on="ID", right_index=True)

vectors = [pv_dm.infer_vector(tokenize_source(source), steps=3, alpha=0.1) for source in code_scores['Code']]
#vectors[0][0]

train_vectors, test_vectors, train_scores, test_scores = train_test_split(vectors, code_scores['Score'], test_size=0.15, random_state=2 )

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
# sm_res = regr = linear_model.Lasso(alpha=0.0001)
# regr.fit(train_vectors, train_scores)

# Ridge + Regularization
#sm_res = regr = linear_model.Ridge(alpha=1) # 50dim
sm_res = regr = linear_model.Ridge(alpha=0.1) # 30dim
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
    geom_point(size=40) +\
    xlim(0,1) + ylim(0,1) +\
    scale_color_manual(values=['blue', 'red'])
