# Take a code embedding and use it to predict confusion

import statsmodels as sm
#from ggplot import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from gensim.models import Doc2Vec
from gensim_doc2vec import tokenize_code, tokenize_edn

import pandas as pd

#pv_dm = Doc2Vec.load('tmp/linux_2018-02-21T23:34:44.493096_model_final.pkl') # 30dim
#tokenize = tokenize_code

pv_dm = Doc2Vec.load('tmp/linux-edn_2018-02-28T14:45:06.776662_model_final.pkl')
tokenize = tokenize_edn

snippet_results = pd.read_csv('tmp/results_normalized.csv')
snippet_questions = pd.read_csv('tmp/questions.csv.edn')

snippet_questions['ID'] = 's' + snippet_questions['ID'].astype('str')
snippet_questions['Type'] = snippet_questions['Type'].map({'Confusing': 'C', 'Non-confusing': 'NC'})
snippet_results['CodeID'] = 's' + snippet_results['CodeID'].astype('str')

context_results = pd.read_csv('tmp/context_study_usercode.csv')
context_questions = pd.read_csv('tmp/context_study_code.csv.edn')

context_questions['ID'] = 'c' + context_questions['ID'].astype('str')
context_results['CodeID'] = 'c' + context_results['CodeID'].astype('str')

combined_questions = pd.concat([snippet_questions[['ID', 'Code', 'Type']],
                                context_questions[['ID', 'Code', 'Type']]])
combined_results = pd.concat([snippet_results[['CodeID', 'Correct']],
                              context_results[['CodeID', 'Correct']]])


combined_tfs = combined_results.groupby('CodeID')['Correct'].value_counts().unstack().fillna(0)
combined_scores = combined_tfs['T'] / (combined_tfs['T'] + combined_tfs['F'])

code_scores = pd.merge(combined_questions, combined_scores.to_frame('Score'), left_on="ID", right_index=True)

vectors = [pv_dm.infer_vector(tokenize(source), steps=3, alpha=0.1) for source in code_scores['Code']]

# Tell the regression whether the code is C/NC - TODO REMOVE
#vectors = [np.append(v, t) for (v, t) in zip(vectors, code_scores['Score'])]
#vectors = [np.append(v, t) for (v, t) in zip(vectors, code_scores['Type'].map({'NC':0,'C':1}))]

X = vectors
y = code_scores['Score']

#regr = linear_model.LinearRegression()
#regr = linear_model.Lasso()
regr = linear_model.Ridge()
#print(regr.intercept_)
#print(regr.coef_)

scorer = 'neg_mean_absolute_error'
#scorer = 'r2'
gscv = GridSearchCV(estimator=regr, param_grid=dict(alpha=np.logspace(-6, 3, 10)), n_jobs=-1, scoring=scorer)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)#, random_state=2 )
gscv.fit(X_train, y_train)
gscv.best_estimator_
print("gscv.best_score: ", gscv.best_score_)
regr = gscv.best_estimator_

print("score:  ", np.mean(cross_val_score(regr, X, y, cv=10, scoring=scorer)))

predictor = regr.fit(X_train, y_train)
y_train_predicted = predictor.predict(X_train)
y_test_predicted  = predictor.predict(X_test)

predicted_confusion = pd.concat([
    pd.DataFrame.from_records({"observed": y_train, "predicted": y_train_predicted, "inferred": False}),
    pd.DataFrame.from_records({"observed": y_test, "predicted": y_test_predicted, "inferred": True})
    ])

#ggplot(aes(x='observed', y='predicted', color='inferred', size='inferred'), data=predicted_confusion)+\
#    geom_point() +\
#    xlim(0,1) + ylim(0,1) +\
#    scale_color_manual(values=['blue', 'red'])
predicted_confusion
predicted_confusion.plot.scatter(x='observed', y='predicted',
                                 c=predicted_confusion.inferred.map({False:'b', True:'r'}),
                                 s=predicted_confusion.inferred.map({False:10, True:40}))
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

