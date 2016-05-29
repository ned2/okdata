#!/usr/bin/env python3

import sys
import argparse
import glob
import json
import os

from itertools import chain

import okc


# things to investigate:

# 1. words not occurring in known word lists, sorted by frequency


def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", metavar="PATH")
    argparser.add_argument("-n", default=10)
    argparser.add_argument("--debug", action='store_true')
    return argparser


# def get_user_essays(path):
#     json_paths = glob.glob(os.path.join(path, '*.json'))
            
#     for json_path in json_paths:
#         with open(json_path, encoding='utf-8') as json_file:
#             user = json.loads(json_file.read())
#             user_essays = [] 
#             for essay in user['essays']:
#                 this_essay = essay['essay']
#                 if this_essay == []:
#                     user_essays.append([])
#                 else:
#                     user_essays.append(this_essay[0]['rawtext'])
#             yield user_essays


# def get_corpus(path):
#     user_essays = get_user_essays(path)
#     essays = chain.from_iterable(user_essays)
#     essay_tokens = (word_tokenize(essay) for essay in essays if essay)
#     return chain.from_iterable(essay_tokens)


def get_corpus_tokens(path):
    return chain.from_iterable(user.get_tokens() for user in okc.load_users(path))


def get_users_data(path):
    return ({
        'match': user.match,
        'age': user.age,
        'gender': user.gender,
        'enemy': user.enemy,
        'tokens': list(user.get_tokens()),
    } for user in okc.load_users(path))
    

    
def main(argv=None):
    if argv is None:
        arg = argparser().parse_args()
    else:
        arg = argparser().parse_args(args=argv)


    
if __name__ == "__main__":
    sys.exit(main())
