#!/usr/bin/env python2

from __future__ import division

import sys
import argparse
import numpy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn import metrics

import okc


MATCH_THRESHOLD = 70
TRAIN_PROPORTION = 0.8


def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path")
    argparser.add_argument("--classifier", default='pa')
    argparser.add_argument("--punct", action='store_true')
    argparser.add_argument("--match", default=70, type=int)
    argparser.add_argument("--topfeatures", default=0, type=int)
    return argparser


text_func = lambda user : user.text
match_func = lambda user : int(user.match >= MATCH_THRESHOLD)
gender_func = lambda user : user.gender



class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)


CLASSIFIERS = {
    'l1svc'      : L1LinearSVC(),
    'ridge'      : RidgeClassifier(tol=1e-2, solver="lsqr"),
    'perceptron' : Perceptron(n_iter=50, alpha=0.1, penalty=None),
    'pa'         : PassiveAggressiveClassifier(n_iter=50),
    'svc'        : LinearSVC(loss='l2', penalty='l1',
                             dual=False, tol=1e-3),
    #'nb'         : MultinomialNB(),
    #'bnb'        : BernoulliNB(),
    #'nc'         : NearestCentroid(),
    #'sgd'        : SGDClassifier(loss='log', alpha=0.1, penalty='l2', n_iter=5),
    #'knn'        : KNeighborsClassifier(n_neighbors=10), 
}            


class Classifier(object):
    """Why does using the TfidfTransformer make the results for successful
    identification go to zero for match_threshold > 60 or so?

    Scikit learn documentation suggests that tf-idf values for shorter 
    documents can be noiser and in this case using binary feature 
    can be more stable

    To do next:
      * look at most informative features when analyzer='char'
      * investigate how to add other features in, such as size of essay, age, etc.<
    """
    
    def __init__(self, instances, labels, class_type='pa', tfidf=True,
                 grid_search=False, n_features=None):
        classifier = self.get_cls(class_type, tfidf, n_features)

        if grid_search:
            classifier = self.grid_search(classifier, tfidf, n_features)
        
        self.classifier = classifier.fit(instances, labels)

        if grid_search:
            print self.classifier.best_params_
            
    def get_cls(self, classifier_type, tfidf, n_features):
        if classifier_type == 'nb':
            count_vect = CountVectorizer(
                analyzer='char',
                ngram_range=(1,2), 
                strip_accents='unicode',
            )
        elif classifier_type == 'bnb':
            count_vect = CountVectorizer(
                analyzer='char_wb',
                ngram_range=(1,3), 
                strip_accents='unicode',
            )
        else: 
            count_vect = CountVectorizer(
                analyzer='word',
                ngram_range=(1,3), 
                strip_accents='unicode',
            )

        pipeline = [('vect', count_vect)]

        if n_features is not None:
            pipeline.append(('feat_sel', SelectKBest(chi2, k=n_features)))

        if tfidf:
            pipeline.append(('tfidf', TfidfTransformer()))

        pipeline.append(('clf', CLASSIFIERS[classifier_type]))
        return Pipeline(pipeline)

    def grid_search(self, classifier, grid_score, tfidf, n_features):
        param_grid = {
            'vect__analyzer'      : ('word', 'char', 'char_wb'),
            'vect__ngram_range'   : ((1, 1), (1, 2), (1,3)),
            'vect__stop_words'    : ('english', None),
            'vect__strip_accents' : ('unicode', None),
            #'vect__binary'        : (True, False),
            #'clf__loss'           : ('hinge','perceptron', 'huber', 'log', 'modified_huber'),
            #'clf__alpha'          : (1e-1, 1e-2, 1e-3, 1e-4),
            #'clf__penalty'        : ('l1', 'l2', None),
        }

        if n_features is not None:
            param_grid['feat_sel__k'] = (100, 200)
            #param_grid['feat_sel__score_func'] = (chi2, f_classif)
        
        if tfidf:
            param_grid['tfidf__use_idf'] = (True, False),
            param_grid['tfidf__sublinear_tf'] = (True, False),

        classifier = GridSearchCV(classifier, param_grid, scoring=grid_score)
        return classifier
        
    def test(self, instances, labels):
        predicted = self.classifier.predict(instances)
        print metrics.classification_report(labels, predicted)
        print metrics.confusion_matrix(labels, predicted)

    def show_most_informative_features(self, instances, topn):
        coefs_with_fns = sorted(zip(self.coefficients, self.features, self.feature_values(instances)))
        top = zip(coefs_with_fns[:topn], coefs_with_fns[:-(topn + 1):-1])
        for (coef_1, fn_1, freq_1), (coef_2, fn_2, freq_2) in top:
            fn_1 = okc.decode_punct(fn_1)
            fn_2 = okc.decode_punct(fn_2)
            print "{:10.4f}{:>20}{:10}   |{:10.4f}{:>20}{:10}".format(coef_1, fn_1, freq_1, coef_2, fn_2, freq_2)

    def feature_values(self, instances):
        """So we can get the raw counts of the features as used by the classifier."""
        matrix = self.classifier.steps[0][1].fit_transform(instances)
        return matrix.sum(axis=0).tolist()[0]

    @property
    def coefficients(self):
        return self.classifier.steps[-1][1].coef_[0]

    @property
    def features(self):
        return self.classifier.steps[0][1].get_feature_names()
    

    
#TODO update this
def test_all(users, grid_search=False):
    for classifier_type in CLASSIFIERS:
        print classifier_type
        RunTrainTest(users, tfidf=tfidf, classifier_type=classifier_type,
                     grid_search=grid_search)
        print '\n'


def train_test(users, class_type, topfeatures):
    train_users, test_users = train_test_split(
        users, train_size=TRAIN_PROPORTION, random_state=42
    )
    train_instances = (text_func(u) for u in train_users)
    train_labels = [match_func(u) for u in train_users]
    test_instances = (text_func(u) for u in test_users)
    test_labels = [match_func(u) for u in test_users]
    cls = Classifier(train_instances, train_labels, class_type)
    cls.test(test_instances, test_labels)

    if topfeatures:
        train_instances = (text_func(u) for u in train_users)
        cls.show_most_informative_features(train_instances, topfeatures)

    return cls


def main():
    global MATCH_THRESHOLD
    args = argparser().parse_args()
    MATCH_THRESHOLD = args.match
    users = okc.load_users(args.path, keep_punct=args.punct)
    classifier = train_test(users, args.classifier, args.topfeatures)

        
if __name__ == "__main__":
    sys.exit(main())
