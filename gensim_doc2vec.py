# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

###################################################
# Read the corpus
###################################################

from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing
from collections import namedtuple
from find_files import filetypes_in_dirs

CodeDocument = namedtuple('CodeDocument', 'words split tags')

alldocs = []
doc_idx = 0
for filename in filetypes_in_dirs(["java"], ["~/opt/src/tomcat85"]):
    with open(filename, 'r') as file_obj:
        file_content = file_obj.read()
        split = ['train', 'test', 'validate', 'extra'][doc_idx % 4]
        tags = [filename]
        doc_idx += 1
        alldocs.append(CodeDocument(file_content, split, tags))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # For reshuffling per pass

###################################################
# Create the models
###################################################

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

# PV-DM w/ average - alternatives include using dm_concat and using PV-DBOW
pv_dm = Doc2Vec(dm=1, dm_mean=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, workers=cores)

pv_dm.build_vocab(alldocs)

models_by_name = {"PV-DM": pv_dm}

###################################################
# Evaluate the models
###################################################

import numpy as np
import statsmodels.api as sm
from random import sample

# For timing
from contextlib import contextmanager
from timeit import default_timer
import time

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def logistic_predictor_from_data(train_targets, train_regressors):
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    # print(predictor.summary())
    return predictor

def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    test_regressors = sm.add_constant(test_regressors)

    # Predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)

###################################################
# Training
###################################################

from collections import defaultdict
best_error = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved

from random import shuffle
import datetime

alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes

print("START %s" % datetime.datetime.now())

for epoch in range(passes):
    shuffle(doc_list)  # Shuffling gets best results

    # Train
    duration = 'na'
    pv_dm.alpha, pv_dm.min_alpha = alpha, alpha
    with elapsed_timer() as elapsed:
        pv_dm.train(doc_list, total_examples=len(doc_list), epochs=1)
        duration = '%.1f' % elapsed()

    ## Evaluate
    #eval_duration = ''
    #with elapsed_timer() as eval_elapsed:
    #    err, err_count, test_count, predictor = error_rate_for_model(pv_dm, train_docs, test_docs)
    #eval_duration = '%.1f' % eval_elapsed()
    #best_indicator = ' '
    #if err <= best_error[name]:
    #    best_error[name] = err
    #    best_indicator = '*'
    #print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1, name, duration, eval_duration))

    #if ((epoch + 1) % 5) == 0 or epoch == 0:
    #    eval_duration = ''
    #    with elapsed_timer() as eval_elapsed:
    #        infer_err, err_count, test_count, predictor = error_rate_for_model(pv_dm, train_docs, test_docs, infer=True)
    #    eval_duration = '%.1f' % eval_elapsed()
    #    best_indicator = ' '
    #    if infer_err < best_error[name + '_inferred']:
    #        best_error[name + '_inferred'] = infer_err
    #        best_indicator = '*'
    #    print("%s%f : %i passes : %s %ss %ss" % (best_indicator, infer_err, epoch + 1, name + '_inferred', duration, eval_duration))

    print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta

print("END %s" % str(datetime.datetime.now()))

pv_dm.most_similar("f")
pv_dm.wv.vocab

[d.tags for d in alldocs[:10]]
