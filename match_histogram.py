import sys
import argparse
import okc
import matplotlib.pyplot as plt

def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path")
    return argparser

def main():
    args = argparser().parse_args()
    users = okc.load_users(args.path)
    match_score_counts = [0]*100

    for user in users:
        if user.match == 0:
            print user.username

    print len(users)
    match_scores = [user.match for user in users if user.match > 0]
    #gender_to_score = [(user.match, user.gender) for user in users if user.match > 0]

    plt.hist(match_scores, bins=100)
    #plt.hist(gender_to_score, bins=100)
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
