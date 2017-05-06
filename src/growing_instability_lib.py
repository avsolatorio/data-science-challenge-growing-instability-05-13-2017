import warnings; warnings.simplefilter('ignore')
import sys
sys.path.insert(0, "/home/avsolatorio/WORK/generating-reviews-discovering-sentiment")

import fasttext
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import re
from datetime import datetime, timedelta
import glob
import json
import tensorflow
import keras
import multiprocessing as mp
from joblib import Parallel, delayed
import os
from wordsegment import segment
from sklearn.metrics.pairwise import cosine_similarity
import ijson
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# http://thinknook.com/10-ways-to-improve-your-classification-algorithm-performance-2013-01-21/

fsmodel = None
wvmodel = None

sample_sub = pd.read_csv('../data/sampleSubmission.csv')

topics = sorted(set(sample_sub.columns.difference(['id'])))
target_columns = sorted(topics)

topic2actual = {}
for i in sample_sub.columns:
    if 'id' == i:
        continue
    topic2actual[i] = segment(i)


def transform_text(df, retain_special_chars=False):
    body = df.bodyText\
    .str.replace("I'm ", "I am ")\
    .str.replace("It's ", 'It is ')\
    .str.replace("'ve ", " have ")\
    .str.replace("'re ", ' are ')\
    .str.replace("n't ", " not ")\
    .str.replace(" ([a-z]+)('s) ", r' \1 is ')\
    .str.lower()

    if retain_special_chars:
        body = body\
        .str.replace('([a-zA-Z0-9]+)(\W)', r'\1 \2')\
        .str.replace('(\W)([a-zA-Z0-9])', r'\1 \2')
    else:
        body = body\
        .str.replace('\W', ' ')

    body = body\
    .str.replace('\s\s+', ' ')
    
    return body
    

# Methods for preparing data for word2vec and fasttext training    
# -------------------------------------------------------------

def normalize_for_fasttext(df, topics, fname='fasttext_normalized_train_body_data.csv', retain_special_chars=False, with_labels=True):
    body = transform_text(df, retain_special_chars=retain_special_chars)
    labels = df.topics.map(topics.intersection)

    if with_labels:
        if not topics:
            raise ValueError('Topics must not be empty if with_labels is set to True!')
        
        labels = labels.map(
            lambda x: u' , '.join([u'__label__{}'.format(i) for i in x if i in topics]) if x else u'__label__null'
        )

        body = labels + ' , ' + body

    if fname:
        body.to_csv(fname, mode='a', index=False, encoding='utf-8')
        return labels.reset_index()
    else:
        return body
    
    
num_proc = mp.cpu_count() - 1

def parallel_corpus_processor(samp, normalize_for_fasttext, batch, num_proc, processing_kwargs):

    with Parallel(n_jobs=num_proc) as parallel:
        dataset = []
        is_break = False
        i = 0
        
        base_dir = './corpus'
        base_name = 'train_body_data-with_labels_{with_labels}-retain_special_chars_{retain_special_chars}'.format(**processing_kwargs)
        base_name = os.path.join(base_dir, base_name)

        while not is_break:
            payload = []

            for j in xrange(num_proc):
                t_df = samp[(i + j) * batch: (i + 1 + j) * batch]

                if t_df.empty:
                    is_break = True
                    continue

                payload.append(
                    delayed(normalize_for_fasttext)(
                        t_df, fname=base_name + '_{}.csv'.format(j), **processing_kwargs
                    )
                )

            print('Current batch in main thread: {}'.format((i + j) * batch))

            if payload:
                results = parallel(payload)
                dataset.extend(results)
                i += num_proc

    return pd.concat(dataset)


def generate_training_corpus(compile_df=True, store_fasttext=True, batch=5000):
    train_df = pd.DataFrame()
    valid_labels_df = pd.DataFrame()

    processing_kwargs = dict(
        with_labels=False,
        retain_special_chars=False,
        topics=topics
    )

    for fname in sorted(glob.glob('../data/TrainingData/*.json')):
        print('Processing {}...'.format(fname))
        with open(fname) as fl:
            data = json.load(fl)
            df = pd.DataFrame(data['TrainingData'])
            df = df.T

            if store_fasttext:
                # for i in range(0, df.shape[0] + batch, batch):
                #     ddf = df[i: i + batch]

                #     if not ddf.empty:
                #         normalize_for_fasttext(ddf, topics, fname='fasttext_normalized_train_body_data.csv', with_labels=True)
                lf = parallel_corpus_processor(
                    df, normalize_for_fasttext=normalize_for_fasttext,
                    batch=batch, num_proc=num_proc, processing_kwargs=processing_kwargs,
                )

                if lf.empty:
                    valid_labels_df = lf
                else:
                    valid_labels_df = valid_labels_df.append(lf)

            if compile_df:
                cols = ['topics', 'webPublicationDate']
                if train_df.empty:
                    train_df = df[cols]
                else:
                    train_df = train_df.append(df[cols])

    with open('../data/TestData.json') as fl:
        data = json.load(fl)
        test_df = pd.DataFrame(data['TestData']).T
        if store_fasttext:
            parallel_corpus_processor(
                test_df, normalize_for_fasttext=normalize_for_fasttext,
                batch=batch, num_proc=num_proc, processing_kwargs=processing_kwargs,
            )

    if train_df.shape[0] == 1600462:
        if (train_df.columns == [u'topics', u'webPublicationDate']).sum() == 2:
            train_df.to_json('../data/training_id_topics.json')
        
    return train_df, valid_labels_df, test_df


def parse_training_data_with_valid_topics(path, keys=[], topics=[], limit=None):
    with open(path) as fl:
        parser = ijson.parse(fl)
        dataset = {}
        curr_index = None
        skip_index = None

        for prefix, event, value in parser:
            if 'map_key' == event and '_TrainingData_' in value:
                if limit:
                    if len(dataset) > limit:
                        break
            
            template = prefix.replace('TrainingData.', '').replace('.topics.item', '.topics')
            template = template.split('.')

            if event == 'string':
                for i in template:
                    if i == skip_index:
                        break

                    if '_TrainingData_' in i:
                        if keys and i not in keys:
                            skip_index = i
                            break

                        if curr_index != i:
                            if curr_index and len(set(dataset[curr_index].keys()).difference(['bodyText', 'webPublicationDate'])) == 0:
                                dataset.pop(curr_index)

                            curr_index = i
                            dataset[curr_index] = {}
                        
                    elif 'bodyText' == i:
                        dataset[curr_index]['bodyText'] = value
                        
                    elif 'topics' == i:
                        if topics and value not in topics:
                            continue

                        dataset[curr_index][value] = 1
                    elif 'webPublicationDate' == i:
                        dataset[curr_index]['webPublicationDate'] = value
        
        # Remove end data that doesn't have relevant topics.
        if curr_index and len(set(dataset[curr_index].keys()).difference(['bodyText', 'webPublicationDate'])) == 0:
            dataset.pop(curr_index)

    return dataset


def transform_fasttext(tokens, stopwords=[]):
    global fsmodel
    # This requires fsmodel to be present in the namespace.
    fs_feature_vec = tokens.map(
        lambda x: [w for w in x.split() if (w not in stopwords)]
    ).map(lambda x: np.array([fsmodel[w] for w in x]).mean(axis=0) if len(x) > 0 else np.nan)

    return fs_feature_vec


def transform_unsupervised_sentiment_neuron(tokens, stopwords=[]):
    # This requires fsmodel to be present in the namespace.
    
    usn_feature_vec = usnmodel.transform(tokens)

    # usn_feature_vec = tokens.map(
    #     lambda x: [w for w in x.split() if (w not in stopwords)]
    # ).map(lambda x: np.array([usnmodel[w] for w in x]).mean(axis=0) if len(x) > 0 else np.nan)

    return usn_feature_vec


def transform_word2vec(tokens, stopwords=[]):
    global wvmodel
    # This requires wvmodel to be present in the namespace.
    wv_feature_vec = tokens.map(
        lambda x: [w for w in x.split() if (w not in stopwords and w in wvmodel.wv.vocab)]
    ).map(lambda x: np.array([wvmodel[w] for w in x]).mean(axis=0) if len(x) > 0 else np.nan)

    return wv_feature_vec


def parallel_generate_word_vectors(samp, transformer, stopwords, batch, num_proc):
    with Parallel(n_jobs=num_proc) as parallel:
        dataset = []
        is_break = False
        i = 0

        while not is_break:
            payload = []

            for j in xrange(num_proc):
                t_df = samp[(i + j) * batch: (i + 1 + j) * batch]

                if t_df.empty:
                    is_break = True
                    continue

                payload.append(
                    delayed(transformer)(
                        t_df, stopwords
                    )
                )

            print('Current batch in main thread: {}'.format((i + j) * batch))

            if payload:
                results = parallel(payload)
                dataset.extend(results)
                i += num_proc

    return pd.concat(dataset)


def resample_data(df, upsample=100, maxsample=400):
    indices = []
    unique_indices = set()
    target_topics = df.columns.intersection(topics)
    freq = df[target_topics].sum(axis=0)
    freq = freq.sort_values()
    for i in freq.index:
        sub = df[df[i].notnull()]

        if freq[i] < upsample:
            indices.extend(sub.index)
            n = upsample - sub.shape[0]
            # Prioritize pure observations, i.e., class is unique.
            w = 1. / sub[target_topics].sum(axis=1)
            indices.extend(sub.sample(n=n, replace=True, weights=w).index)

        elif freq[i] > maxsample:
            w = 1. / sub[target_topics].sum(axis=1)
            indices.extend(sub.sample(n=maxsample, weights=w).index)

        else:
            indices.extend(sub.index)
        
        unique_indices.update(indices)
            
    return indices
