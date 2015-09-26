#!/usr/bin/env python2
from __future__ import division

import sys
import argparse

import numpy
from sklearn import linear_model
from sklearn import metrics

import okc


TRAIN_PROPORTION = 0.8

REGRESSORS = {
    'linear'      : linear_model.LinearRegression(),
    'ridge'       : linear_model.Ridge(alpha=.5),
}


def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--regressor", default='linear')
    argparser.add_argument("path")
    return argparser


class Regressor(object):

    def __init__(self, instances, labels, regressor_type, tfidf=True):
        regressor = self.get_regressor(regressor_type, tfidf, n_features)
        self.regressor = regressor.fit(instances, labels)
        
        
    def get_regressor(self, regressor_type, tfidf):
        count_vect = CountVectorizer(
            analyzer='word',
            ngram_range=(1,1), 
            strip_accents='unicode',
        )
        pipeline = [('vect', count_vect)]

        if tfidf:
            pipeline.append(('tfidf', TfidfTransformer()))

        pipeline.append(('regressor', REGRESSORS[regressor_type]))
        return Pipeline(pipeline)

    def test(self, instances, labels):
        predicted = self.regressor.predict(instances)
        print metrics.mean_absolute_error(labels, predicted)

        
text_func = lambda user : user.text
match_func = lambda user : int(user.match >= MATCH_THRESHOLD)


def train_test(users, regressor_type):
    train_users, test_users = train_test_split(
        users, train_size=TRAIN_PROPORTION, random_state=42
    )
    train_instances = (text_func(u) for u in train_users)
    train_labels = [match_func(u) for u in train_users]
    test_instances = (text_func(u) for u in test_users)
    test_labels = [match_func(u) for u in test_users]
    regressor = Regressor(train_instances, train_labels, regressor_type)
    regressor.test(test_instances, test_labels)


def main():
    args = argparser().parse_args()
    users = okc.load_users(args.path, keep_punct=True)
    train_test(users, args.regressor)

    
if __name__ == "__main__":
    sys.exit(main())
