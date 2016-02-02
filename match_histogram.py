#!/usr/bin/env python2

import sys
import argparse
import okc
import matplotlib.pyplot as plt


FEATURE_CHOICES = (
    'match',
    'enemy',
    'questions',
    'age'
)


def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("feature", choices=FEATURE_CHOICES)
    argparser.add_argument("path")
    argparser.add_argument("--bins", type=int, default=100)
    return argparser


def main():
    args = argparser().parse_args()
    user_paths = okc.get_user_paths(args.path)
    users = okc.load_users(user_paths)

    if args.feature == 'match':
        # ignore zero match scores because they overwhelm the other values
        values = [user.match for user in users if user.match > 0]
    elif args.feature == 'enemy':
        values = [user.enemy for user in users]
    elif args.feature == 'age':
        values = [user.age for user in users]
    elif args.feature == 'questions':
        user_data = okc.get_user_data(args.path)
        values = [user['num_questions'] for user in user_data.values() if
                  user['num_questions'] > 10]
        
    plt.hist(values, bins=args.bins)
    plt.show()
        
    
if __name__ == "__main__":
    sys.exit(main())
