#!/usr/bin/env python3

import os
import sys
import argparse

import okc
import settings


"""Yet another OKCupid scraper.

If you want to use this software to interact with www.okcupid.com,
please first read and make sure you are complying with their terms of
service.

https://www.okcupid.com/legal/terms

Note that this code was developed for personal educational purposes
and is not intended as a reusable API.

Instructions for use:

1. Copy the settings_example.py file to settings.py and modify its
   contents appropriately.

2. Find users:
   $ python okscraper.py find usernames.txt

3. Scrape profiles from usernames found in usernames.txt into path 'profiles':
   $ python okscraper.py scrape usernames.txt profiles

   If the scraper aborts prematurely, you can resume like so:
   $ python okscraper.py scrape --resume usernames.txt profiles

Note that the default settings in settings.py will retrieve hopefully
almost all users in a 25 m/km radius from you. If you provide your login
credentials here you can omit them from the command line.

If you run the scraper when not in private browsing mode, you'll also
happen to show up in their visited list. Turns out this is a useful
way to be seen by a lot of people. If you run it over an exhaustive
search for users in an area, be prepared for some interesting and
potentially offensive responses, you'll be visiting people with whom
you have drastically low match ratings (and likely radically different
life outlooks), and to them it will look like you intentionally
clicked on them.

Note that if you want to exhaustively find all users in an area using
find mode, you need to have an A-List subscription in order to access
the random search ranking. With this we can get around the 1000
profile limit on each search by repeating the random search until no
new users are returned.

I'm not sure what happens if you try and use the RANDOM ranking (which
the find mode assumes you have without an A-List subscription. I
haven't checked if they've been thorough and applied the A-List
restrictions to the API as well as the front end.

"""


def argparser():
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers(dest='command')

    ap1 = subparsers.add_parser('find')
    ap1.add_argument("--location", default=None)

    ap2 = subparsers.add_parser('scrape')
    ap2.add_argument("inpath", metavar='USERNAME_PATH')
    ap2.add_argument("--resume", action='store_true')
    
    # arguments common to both
    for subparser in (ap1, ap2):
        subparser.add_argument("--username", default=settings.USERNAME)
        subparser.add_argument("--password", default=settings.PASSWORD)
        subparser.add_argument("outpath", default=os.getcwd(), metavar='OUTPUTPATH')

    return argparser


def main():
    args = argparser().parse_args()
    session = okc.Session(username=args.username, password=args.password)

    if args.command == 'find':
        usernames = session.find_all_users(location=args.location)
        with open(args.outpath, 'w', encoding='utf8') as file:
            file.write('\n'.join(usernames))
    elif args.command == 'scrape':
        with open(args.inpath, encoding='utf8') as file:
            usernames = set(file.read().split())
        session.dump_profiles(usernames, args.outpath, resume=args.resume)

    
if __name__ == "__main__":
    sys.exit(main())
