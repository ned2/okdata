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


def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--regressor", default='linear')
    argparser.add_argument("--alpha", type=float, default=0.1)
    argparser.add_argument("path")
    argparser.add_argument("--topfeatures", default=0, type=int)
    return argparser


def get_regressor(regressor_type):
    return {
        'linear'   : linear_model.LinearRegression(),
        'ridge'    : linear_model.Ridge(alpha=ALPHA),
        'ridgecv'  : linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0]),
        'lasso'    : linear_model.Lasso(alpha=ALPHA),    
        'sgd'      : linear_model.SGDRegressor(alpha=ALPHA),
        'pa'       : linear_model.PassiveAggressiveRegressor(),
        'bridge'   : linear_model.BayesianRidge(),
    }[regressor_type]

    
class Regressor(object):

    def __init__(self, instances, labels, regressor_type, tfidf=True):
        regressor = self.get_regressor(regressor_type, tfidf)
        self.regressor = regressor.fit(instances, labels)
        
        
    def get_regressor(self, regressor_type, tfidf):
        count_vect = CountVectorizer(
            analyzer='char_wb',
            ngram_range=(1,4), 
            strip_accents='unicode',
        )
        pipeline = [('vect', count_vect)]

        if tfidf:
            pipeline.append(('tfidf', TfidfTransformer()))

        pipeline.append(('regressor', get_regressor(regressor_type)))
        return Pipeline(pipeline)

    def test(self, instances, labels):
        predicted = self.regressor.predict(instances)
        print "Mean absoulte error: {}".format(metrics.mean_absolute_error(labels, predicted))
        print "Mean squared error: {}".format(metrics.mean_squared_error(labels, predicted))

    def show_most_informative_features(self, instances, topn):
        coefs_with_fns = sorted(zip(self.coefficients, self.features, self.feature_values(instances)))
        top = zip(coefs_with_fns[:topn], coefs_with_fns[:-(topn + 1):-1])
        for (coef_1, fn_1, freq_1), (coef_2, fn_2, freq_2) in top:
            fn_1 = okc.decode_punct(fn_1)
            fn_2 = okc.decode_punct(fn_2)
            print "{:10.4f}{:>20}{:10}   |{:10.4f}{:>20}{:10}".format(coef_1, fn_1, freq_1, coef_2, fn_2, freq_2)

    def feature_values(self, instances):
        """So we can get the raw counts of the features as used by the regressor."""
        matrix = self.regressor.steps[0][1].fit_transform(instances)
        return matrix.sum(axis=0).tolist()[0]

    @property
    def coefficients(self):
        return self.regressor.steps[-1][1].coef_[0]

    @property
    def features(self):
        return self.regressor.steps[0][1].get_feature_names()

        
def train_test(user_paths, regressor_type, topfeatures=0):
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

    if topfeatures:
        regressor.show_most_informative_features(train_instances, topfeatures)

    return regressor


def main():
    global ALPHA
    args = argparser().parse_args()
    ALPHA = args.alpha
    paths = [os.path.join(args.path, p) for p in os.listdir(args.path)]
    regressor = train_test(paths, args.regressor, topfeatures=args.topfeatures)

    
if __name__ == "__main__":
    sys.exit(main())
