# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

hyper_params = {'dataset': 'linux',
                'vector_size': 100,
                'min_count': 1000,
                'alpha_start': 0.1,
                'alpha_stop': 0.001,
                'passes': 5}

datasets = {
    'linux': lambda: c_files_in_dirs(["~/opt/src/linux"]),
    'tomcat': lambda: filetypes_in_dirs(["java"], ["~/opt/src/tomcat85"])
}

###################################################
# Read the corpus
###################################################

from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing
from collections import namedtuple, Counter
from find_files import filetypes_in_dirs, c_files_in_dirs
import re

# https://github.com/statsmodels/statsmodels/issues/3931
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

def tokenize_source(s):
    token_pat = '(?:[][.,;{}()|+=*&^%!~-]|(?:\s+))'
    return re.findall(re.compile(token_pat + '|(?:(?!' + token_pat + ').)+'), s)

CodeDocument = namedtuple('CodeDocument', 'words split tags directory')

# file_content = "alpha.bravo(charlie,delta); echo + foxtrot;\n\ngolf;"
# print('|'.join(tokenize_source(file_content)[-50:]))
# >>> alpha|.|bravo|(|charlie|,|delta|)|;| |echo| |+| |foxtrot|;|golf|;

# with open('/home/dgopstein/opt/src/tomcat85/webapps/examples/jsp/plugin/applet/Clock2.java', 'r') as file_obj:
#     print(tokenize_source(file_obj.read())[-50:])

def load_docs():
    files = datasets[hyper_params['dataset']]()
    doc_idx = 0
    alldocs = []
    #[x for x in files if re.match(r'.*/linux/.*/linux/', x)][:3]
    for filename in files: #[:100]:
        with open(filename, 'r') as file_obj:
            try:
                file_content = tokenize_source(file_obj.read())
                #split = ['train', 'test', 'validate', 'extra'][doc_idx % 4]
                split = ['train', 'train', 'train', 'test'][doc_idx % 4]
                tags = [filename]
                #directory = re.match(r'.*(tomcat85/.*/)[^/]*', filename).group(1)
                #directory = re.match(r'.*tomcat85/([^/]*(?:/[^/]*)?).*', filename).group(1)
                directory = re.match(r'.*?/linux/([^/]*).*', filename).group(1)
                doc_idx += 1
                alldocs.append(CodeDocument(file_content, split, tags, directory))
            except:
                pass

    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']

    return (alldocs, train_docs, test_docs)

alldocs, train_docs, test_docs = load_docs()
doc_list = alldocs[:]  # For reshuffling per pass

id2class = sorted(list(Counter([x.directory for x in train_docs]).keys()))
###################################################
# Create the models
###################################################

assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

# PV-DM w/ average - alternatives include using dm_concat and using PV-DBOW
pv_dm = Doc2Vec(dm=1, dm_mean=1, vector_size=hyper_params['vector_size'], window=10, negative=5, hs=0, min_count=hyper_params['vector_size'], workers=multiprocessing.cpu_count())

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


    with open('tmp/%s_%s'%(hyper_params['dataset'], start_time), 'a') as f:
        print({'hyper_params': hyper_params,
               'start_time': start_time,
               'end_time': end_time,
               'duration': end_time-start_time,
               'best_error': dict(best_error),
               'all_errors': dict(all_errors)}, file=f)

    print("DURATION %s" % (end_time-start_time))

train_pvdm()

pv_dm.most_similar("<")
pv_dm.wv.most_similar(positive=['true', 'false'], negative=['bool'])
len(pv_dm.wv.vocab.keys())

#import gnuplotlib as gp
#gp.plot( [1, 2, 3, 2, 1] )
