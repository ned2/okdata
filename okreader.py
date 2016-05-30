#!/USSR/bin/env python3

import sys
import argparse
import glob
import json
import os

from itertools import chain
from collections import namedtuple


from nltk.corpus import words
from nltk.stem import PorterStemmer
import okc


# from collections import Counter
# from itertools import chain
# users = list(okreader.get_users_data('/home/nejl/data/okc_collections/ned_melb_new/'))
# counts = Counter(w.lower() for w in chain.from_iterable(user.tokens for user in users)
#                  if is_interesting_word(w))

def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", metavar="PATH")
    argparser.add_argument("-n", default=10)
    argparser.add_argument("--debug", action='store_true')
    return argparser


def get_english_vocab(lemmatize=False):
    vocab = (w.lower() for w in words.words())

    if lemmatize:
        stemmer = PorterStemmer()
        vocab = (stemmer.stem(w) for w in vocab)
    return set(vocab)


def is_interesting_word(token, lemmas=get_english_vocab(lemmatize=True),
                        stemmer=PorterStemmer()):
    if not token.isalpha():
        return False
    if stemmer.stem(token.lower()) in lemmas:
        return False
    return True


def get_corpus_tokens(path):
    user_tokens = (user.get_tokens() for user in okc.load_users(path))
    return chain.from_iterable(user_tokens)


def get_users_data(path):
    english_vocab = get_english_vocab()
    UserFeats = namedtuple('UserFeats', ['match', 'age', 'gender', 'enemy',
                                         'tokens', 'lexdiv', 'len', 'num_essays',
                                         'out_vocab'])
    for user in okc.load_users(path):
        yield UserFeats(
            match = user.match,
            age = user.age,
            gender = user.gender,
            enemy = user.enemy,
            tokens = user.tokens,
            lexdiv = user.lexical_diversity,
            len = len(user.tokens),
            num_essays = sum(1 for essay in user.essays if essay),
            out_vocab = len(user.vocabulary.difference(english_vocab))
        )
    

    
def main(argv=None):
    if argv is None:
        arg = argparser().parse_args()
    else:
        arg = argparser().parse_args(args=argv)


    
if __name__ == "__main__":
    sys.exit(main())
