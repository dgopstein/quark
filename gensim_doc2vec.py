# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

best_linux_hyper_params = {
    'dataset': 'linux',
    'test_split': 4,
    'vector_size': 100,
    'window': 10,
    'negative_samples': 5,
    'min_count': 1000,
    'alpha_start': 0.1,
    'alpha_stop': 0.001,
    'passes': 5}

best_nginx_hyper_params = {
    'dataset': 'nginx',
    'test_split': 4,
    'vector_size': 100,
    'window': 10,
    'negative_samples': 5,
    'min_count': 5,
    'alpha_start': 0.01,
    'alpha_stop': 0.001,
    'passes': 20}

best_mongo_hyper_params = {
    'dataset': 'nginx',
    'test_split': 4,
    'vector_size': 100,
    'window': 10,
    'negative_samples': 5,
    'min_count': 5,
    'alpha_start': 0.01,
    'alpha_stop': 0.001,
    'passes': 5}

hyper_params = best_linux_hyper_params

datasets = {
    'linux': {'file_finder': lambda: c_files_in_dirs(["~/opt/src/linux"]),
              'module_matcher': r'.*?/linux/([^/]*).*'},
    'mongo': {'file_finder': lambda: c_files_in_dirs(["~/opt/src/mongo/src/mongo"]),
              'module_matcher': r'.*?/mongo/src/mongo/([^/]*).*'},
    'nginx': {'file_finder': lambda: c_files_in_dirs(["~/opt/src/nginx"]),
              'module_matcher': r'.*?/nginx/src/([^/]*).*'},
    'tomcat':{'file_finder': lambda: filetypes_in_dirs(["java"], ["~/opt/src/tomcat85"]),
              'module_matcher': r'.*tomcat85/([^/]*(?:/[^/]*)?).*'}
}

###################################################
# Read the corpus
###################################################

import json
from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing
from collections import namedtuple, Counter
from find_files import filetypes_in_dirs, c_files_in_dirs
from datetime_json import *
import re

# https://github.com/statsmodels/statsmodels/issues/3931
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

def tokenize_source(s):
    token_pat = '(?:[][.,;{}()|+=*&^%!~-]|(?:\s+))'
    return re.findall(re.compile(token_pat + '|(?:(?!' + token_pat + ').)+'), s)

CodeDocument = namedtuple('CodeDocument', 'words split tags directory')

def load_docs():
    files = datasets[hyper_params['dataset']]['file_finder']()
    doc_idx = 0
    alldocs = []
    #[x for x in files if re.match(r'.*/linux/.*/linux/', x)][:3]
    for filename in files: #[:100]:
        with open(filename, 'r') as file_obj:
            try:
                file_content = tokenize_source(file_obj.read())
                #split = ['train', 'test', 'validate', 'extra'][doc_idx % 4]
                split = ['test', 'train'][doc_idx % hyper_params['test_split'] == 0]
                tags = [filename]
                directory = re.match(datasets[hyper_params['dataset']]['module_matcher'], filename).group(1)
                doc_idx += 1
                alldocs.append(CodeDocument(file_content, split, tags, directory))
            except:
                pass

    # Linux filter out uncommon modules
    # valid_dirs = set({k:v for k,v in Counter([x.directory for x in alldocs]).items() if v > 1000}.keys())
    # alldocs = [x for x in alldocs if x.directory in valid_dirs]
    #TODO remove

    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']

    return (alldocs, train_docs, test_docs)

alldocs, train_docs, test_docs = load_docs()
doc_list = alldocs[:]  # For reshuffling per pass

id2class = sorted(list(Counter([x.directory for x in train_docs]).keys()))
len(id2class)
###################################################
# Create the models
###################################################

assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

# PV-DM w/ average - alternatives include using dm_concat and using PV-DBOW
pv_dm = Doc2Vec(dm=1, dm_mean=1, vector_size=hyper_params['vector_size'], window=hyper_params['window'], negative=hyper_params['negative_samples'], hs=0, min_count=hyper_params['vector_size'], workers=multiprocessing.cpu_count())

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

# test_model = pv_dm
# train_set = train_docs
# test_set = test_docs
def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets, train_regressors = zip(*[(doc.directory, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_regressors = sm.add_constant(train_regressors)

    # logit2 = sm.MNLogit(["a", "b", "c", "a", "b", "c"], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    # predictor2 = logit2.fit(method='bfgs', disp=1)
    # np.rint(predictor2.predict([[.8, .1, .1, 0], [.2, .6, .2, 0], [.1, .2, .7, 0]]))
    # predictor2.summary()

    #Counter(train_target_names)
    logit = sm.MNLogit(train_targets, train_regressors)
    id2class = logit._ynames_map
    predictor = logit.fit(method='bfgs', disp=0)
    # predictor.summary()
    #predictor.mle_retvals

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    test_regressors = sm.add_constant(test_regressors)

    # Predict & evaluate
    #test_regressors[0]
    with elapsed_timer() as elapsed:
        test_predictions = predictor.predict(test_regressors)
        #print("elapsed time", elapsed())
    #predictor
    #[doc.directory for doc in test_data]
    predicted_classes = [id2class[x] for x in np.argmax(test_predictions, axis=1)]
    correct_classes = [doc.directory for doc in test_data]
    matched_classes = list(zip(predicted_classes, correct_classes))
    matched_classes[1000:1010]
    correct_matches  = [x == y for (x, y) in matched_classes]
    correct_counts = Counter(correct_matches)
    corrects = correct_counts[True]
    errors = correct_counts[False]
    error_rate = float(errors) / (corrects + errors)
    return (error_rate, errors, len(test_predictions), predictor)

###################################################
# Training
###################################################


from collections import defaultdict

from random import shuffle
import datetime

infer = False
def train_pvdm():
    print("training pvdm with: ", hyper_params)

    def eval_error(name, infer):
        eval_duration = ''
        with elapsed_timer() as eval_elapsed:
            err, err_count, test_count, predictor = error_rate_for_model(pv_dm, train_docs, test_docs, infer=infer)
            eval_duration = '%.1f' % eval_elapsed()
        best_indicator = ' '
        if err < best_error[name]:
            pv_dm.save(save_filename+'_model_best.pkl')
            best_error[name] = err
            best_indicator = '*'
        all_errors[name].append(err)
        print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1, name, duration, eval_duration))

    alpha, min_alpha, passes = (hyper_params['alpha_start'], hyper_params['alpha_stop'], hyper_params['passes'])
    alpha_delta = (alpha - min_alpha) / passes
    best_error = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved
    all_errors = defaultdict(lambda: [])  # To selectively print only best errors achieved
    name = "pvdm"

    start_time = datetime.datetime.now()
    save_filename = 'tmp/%s_%s'%(hyper_params['dataset'], start_time.isoformat())

    for epoch in range(passes):
        shuffle(doc_list)  # Shuffling gets best results

        # Train
        duration = 'na'
        pv_dm.alpha, pv_dm.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            pv_dm.train(doc_list, total_examples=len(doc_list), epochs=1)
            duration = '%.1f' % elapsed()

        ## Evaluate
        if ((epoch + 1) % 1) == 0 or epoch == 0:
            eval_error(name, infer=False)

        #if ((epoch + 1) % 50) == 0 or epoch == 0:
        #    eval_error(name + '_inferred', infer=True)

        #print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta

    end_time = datetime.datetime.now()


    # Write state

    with open(save_filename+'.json', 'a') as f:
        print(json.dumps({
            'hyper_params': hyper_params,
            'start_time': start_time,
            'end_time': end_time,
            'duration': (end_time-start_time),
            'best_error': dict(best_error),
            'all_errors': dict(all_errors),
            }, cls=DateTimeAwareJSONEncoder), file=f)

    pv_dm.save(save_filename+'_model_final.pkl')

    print("Best error: ", dict(best_error))
    print("Duration %s" % (end_time-start_time))

train_pvdm()

pv_dm.most_similar("int")
pv_dm.wv.most_similar(positive=['int', 'int'], negative=['0'])
len(pv_dm.wv.vocab.keys())

#import gnuplotlib as gp
#gp.plot( [1, 2, 3, 2, 1] )
