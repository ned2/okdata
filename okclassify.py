#!/usr/bin/env python2

from __future__ import division

import sys
import argparse

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


def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path")
    return argparser

     
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
    'nb'         : MultinomialNB(),
    'bnb'        : BernoulliNB(),
    'ridge'      : RidgeClassifier(tol=1e-2, solver="lsqr"),
    'perceptron' : Perceptron(n_iter=50, alpha=0.1, penalty=None),
    'pa'         : PassiveAggressiveClassifier(n_iter=50),
    'sgd'        : SGDClassifier(loss='log', alpha=0.1, penalty='l2', n_iter=5),
    'knn'        : KNeighborsClassifier(n_neighbors=10), 
    'nc'         : NearestCentroid(),
    'svc'        : LinearSVC(loss='l2', penalty='l1',
                             dual=False, tol=1e-3),
    'l1svc'      : L1LinearSVC()
}            


def test_all(users):
    for classifier_type in CLASSIFIERS:
        print classifier_type
        RunTrainTest(users, classifier_type=classifier_type, grid_search=True)
        print '\n'


class RunTrainTest(object):
    """Why does using the TfidfTransformer make the results for successful
    identification go to zero for match_threshold > 60 or so?

    Scikit learn documentation suggests that tf-idf values for shorter 
    documents can be noiser and in this case using binary feature 
    can be more stable

    To do next:
      * inspect most informative features
        http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
      * look at most informative features when analyzer='char'
      * investigate how to add other features in, such as size of essay, age, etc.<
    """
    
    train_proportion = 0.8
    test_proportion = 0.2

    def __init__(self, users, label_type='match', match_threshold=70, 
                 classifier_type='nb', tfidf=False, grid_search=False,
                 grid_score='f1', n_features=None):
        self.label_type = label_type
        self.match_threshold = match_threshold
        self.set_data(users)
        classifier = self.get_cls(classifier_type, tfidf, n_features)

        if grid_search:
            classifier = self.grid_search(classifier, grid_score, tfidf,
                                          n_features)
        
        self.classifier = classifier.fit(self.train_data, self.train_labels)
        self.test()

        if grid_search:
            print self.classifier.best_params_

            
    def set_data(self, users):
        self.train_users, self.test_users = train_test_split(
            users, train_size=self.train_proportion, random_state=42)

        self.train_data = [user.text for user in self.train_users]
        self.test_data = [user.text for user in self.test_users]
        self.train_labels = self.labels(self.train_users)
        self.test_labels = self.labels(self.test_users)
        
    def labels(self, users):
        labels = []
        for user in users:
            if self.label_type == 'gender':
                labels.append(user.data['gender'])
            else:
                labels.append(int(user.match >= self.match_threshold))
        return labels
                
    def get_cls(self, classifier_type, tfidf, n_features):
        # strip_accents doesn't seem to affect things for either
        # binary=True pulls down the recall of matches for nb but
        # increases average f-score does not affect bnb (obviously)
        # and destroys precision and recall for SGD.
        #
        # stopping has no effect on nb and bnb but kills matches
        # recall for SGD.
        #
        # strip_accents seems to have no noticable effect.

        # TODO: work out if removing short profiles improves things

        if classifier_type == 'nb':
            count_vect = CountVectorizer(
                analyzer='word',
                ngram_range=(1,3), 
                stop_words=None, 
                binary=False, 
                strip_accents='unicode',
            )
        elif classifier_type == 'bnb':
            count_vect = CountVectorizer(
                analyzer='char',
                ngram_range=(1,3), 
                stop_words=None, 
                binary=False, 
                strip_accents='unicode',
            )
        elif classifier_type == 'nc':
            count_vect = CountVectorizer(
                analyzer='word',   # also works slightly better with 'char'
                ngram_range=(1,1), 
                stop_words=None, 
                binary=False, 
                strip_accents='unicode',
            )
        elif classifier_type == 'perceptron':
            count_vect = CountVectorizer(
                analyzer='word',
                ngram_range=(1,3), 
                stop_words=None, 
                binary=False,
                #token_pattern='\w+',
                strip_accents='unicode',
            )
        else: #'sgd' and defaults for others 
            count_vect = CountVectorizer(
                analyzer='word',
                ngram_range=(1,2), 
                stop_words=None, 
                binary=False,
                #token_pattern='\w+',
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
            #'vect__analyzer'      : ('word', 'char', 'char_wb'),
            #'vect__ngram_range'   : ((1, 1), (1, 2), (1,3)),
            #'vect__stop_words'    : ('english', None),
            #'vect__strip_accents' : ('unicode', None),
            #'vect__binary'        : (True, False),
            #'clf__loss'           : ('hinge','perceptron', 'huber', 'log', 'modified_huber'),
            'clf__alpha'          : (1e-1, 1e-2, 1e-3, 1e-4),
            'clf__penalty'        : ('l1', 'l2', None),
        }

        if n_features is not None:
            param_grid['feat_sel__k'] = (100, 200)
            #param_grid['feat_sel__score_func'] = (chi2, f_classif)
        
        if tfidf:
            param_grid['tfidf__use_idf'] = (True, False),
            param_grid['tfidf__sublinear_tf'] = (True, False),

        classifier = GridSearchCV(classifier, param_grid, scoring=grid_score)
        return classifier
        
    def test(self):
        predicted = self.classifier.predict(self.test_data)
        print metrics.classification_report(self.test_labels, predicted)
        print metrics.confusion_matrix(self.test_labels, predicted)

    def show_most_informative_features(self, n=50):
        coefs_with_fns = sorted(zip(self.coefficients, self.features))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

    @property
    def coefficients(self):
        return self.classifier.steps[-1][1].coef_[0]

    @property
    def features(self):
        return self.classifier.steps[0][1].get_feature_names()

    
def main():
    args = argparser().parse_args()
    users = okc.load_users(args.path)
    run = RunTrainTest(users, label_type='match')

    
if __name__ == "__main__":
    sys.exit(main())

    
