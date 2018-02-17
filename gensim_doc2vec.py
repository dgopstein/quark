# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

###################################################
# Read the corpus
###################################################

from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing
from collections import namedtuple, Counter
from find_files import filetypes_in_dirs
import re

def tokenize_source(s):
    token_pat = '(?:[][.,;{}()]|(?: +))'
    return re.findall(re.compile(token_pat + '|(?:(?!' + token_pat + ').)+'), s)

CodeDocument = namedtuple('CodeDocument', 'words split tags directory')

# file_content = "alpha.bravo(charlie,delta); echo + foxtrot;\n\ngolf;"
# print('|'.join(tokenize_source(file_content)[-50:]))
# >>> alpha|.|bravo|(|charlie|,|delta|)|;| |echo| |+| |foxtrot|;|golf|;

# with open('/home/dgopstein/opt/src/tomcat85/webapps/examples/jsp/plugin/applet/Clock2.java', 'r') as file_obj:
#     print(tokenize_source(file_obj.read())[-50:])

def load_docs():
    java_files = filetypes_in_dirs(["java"], ["~/opt/src/tomcat85"])
    doc_idx = 0
    alldocs = []
    for filename in java_files:
        with open(filename, 'r') as file_obj:
            file_content = tokenize_source(file_obj.read())
            split = ['train', 'test', 'validate', 'extra'][doc_idx % 4]
            tags = [filename]
            #directory = re.match(r'.*(tomcat85/.*/)[^/]*', filename).group(1)
            #directory = re.match(r'.*tomcat85/([^/]*(?:/[^/]*)?).*', filename).group(1)
            directory = re.match(r'.*tomcat85/([^/]*).*', filename).group(1)
            doc_idx += 1
            alldocs.append(CodeDocument(file_content, split, tags, directory))

    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']

    return (alldocs, train_docs, test_docs)

alldocs, train_docs, test_docs = load_docs()

classes = list(Counter([x.directory for x in train_docs]).keys())
doc_list = alldocs[:]  # For reshuffling per pass

###################################################
# Create the models
###################################################

assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

# PV-DM w/ average - alternatives include using dm_concat and using PV-DBOW
pv_dm = Doc2Vec(dm=1, dm_mean=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, workers=multiprocessing.cpu_count())

pv_dm.build_vocab(alldocs)

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
    logit = sm.MNLogit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    # print(predictor.summary())
    return predictor

# test_model = pv_dm
# train_set = train_docs
# test_set = test_docs
def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets, train_regressors = zip(*[(doc.directory, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    #train_regressors = sm.add_constant(train_regressors)
    train_targets[0]

    with elapsed_timer() as elapsed:
        predictor = logistic_predictor_from_data(train_targets, train_regressors)
        #print("elapsed time", elapsed())

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    #test_regressors = sm.add_constant(test_regressors)

    # Predict & evaluate
    test_regressors[0]
    with elapsed_timer() as elapsed:
        test_predictions = predictor.predict(test_regressors)
        #print("elapsed time", elapsed())
    predictor
    [doc.directory for doc in test_data]
    predicted_classes = [classes[x] for x in np.argmax(test_predictions, axis=1)]
    correct_classes = [x == y for (x, y) in zip(predicted_classes, [doc.directory for doc in test_data])]
    corrects = Counter(correct_classes)[True]
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)

###################################################
# Training
###################################################

from collections import defaultdict

from random import shuffle
import datetime

def train_pvdm():
    alpha, min_alpha, passes = (0.025, 0.001, 200)
    alpha_delta = (alpha - min_alpha) / passes
    best_error = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved
    name = "pvdm"

    start_time = datetime.datetime.now()
    for epoch in range(passes):
        shuffle(doc_list)  # Shuffling gets best results

        # Train
        duration = 'na'
        pv_dm.alpha, pv_dm.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            pv_dm.train(doc_list, total_examples=len(doc_list), epochs=1)
            duration = '%.1f' % elapsed()

        ## Evaluate
        eval_duration = ''
        with elapsed_timer() as eval_elapsed:
            err, err_count, test_count, predictor = error_rate_for_model(pv_dm, train_docs, test_docs)
        eval_duration = '%.1f' % eval_elapsed()
        best_indicator = ' '
        if err <= best_error[name]:
            best_error[name] = err
            best_indicator = '*'
        print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1, name, duration, eval_duration))

        if ((epoch + 1) % 5) == 0 or epoch == 0:
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                infer_err, err_count, test_count, predictor = error_rate_for_model(pv_dm, train_docs, test_docs, infer=True)
            eval_duration = '%.1f' % eval_elapsed()
            best_indicator = ' '
            if infer_err < best_error[name + '_inferred']:
                best_error[name + '_inferred'] = infer_err
                best_indicator = '*'
            print("%s%f : %i passes : %s %ss %ss" % (best_indicator, infer_err, epoch + 1, name + '_inferred', duration, eval_duration))

        #print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta

    end_time = datetime.datetime.now()
    print("DURATION %s" % (end_time-start_time))

train_pvdm()

pv_dm.most_similar("char")
pv_dm.wv.most_similar(positive=['boolean', 'int'], negative=['Integer'])
list(pv_dm.wv.vocab.keys())[:100]
