#!/usr/bin/env python2
from __future__ import division

import sys
import os
import argparse

import numpy
from sklearn import linear_model
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

import okc


TRAIN_PROPORTION = 0.8

text_func = lambda user : user.text
match_func = lambda user : float(user.match)


REGRESSORS = {
    'linear'      : linear_model.LinearRegression(),
    'ridge'       : linear_model.Ridge(alpha=.5),
    'ridgecv'     : linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0]),
    
}


def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--regressor", default='linear')
    argparser.add_argument("path")
    return argparser


class Regressor(object):

    def __init__(self, instances, labels, regressor_type, tfidf=True):
        regressor = self.get_regressor(regressor_type, tfidf)
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
        print "Mean absoulte error: {}".format(metrics.mean_absolute_error(labels, predicted))
        print "Mean squared error: {}".format(metrics.mean_squared_error(labels, predicted))

        
def train_test(user_paths, regressor_type):
    train_user_paths, test_user_paths = train_test_split(
        user_paths, train_size=TRAIN_PROPORTION, random_state=42
    )

    train_instances = okc.load_users(train_user_paths, func=text_func)
    train_labels = okc.load_users(train_user_paths, func=match_func)

    test_instances = okc.load_users(test_user_paths, func=text_func)
    test_labels = okc.load_users(test_user_paths, func=match_func)

    print "Training model"
    regressor = Regressor(train_instances, list(train_labels), regressor_type)

    print "Testing model"
    regressor.test(test_instances, list(test_labels))
    return regressor


def main():
    args = argparser().parse_args()
    paths = [os.path.join(args.path, p) for p in os.listdir(args.path)]
    regressor = train_test(paths, args.regressor)

    
if __name__ == "__main__":
    sys.exit(main())
