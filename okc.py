#!/usr/bin/env python2

from __future__ import division

import os
import json
import time
import sys
import argparse
import json
import re
from collections import Counter

import requests

from settings import *


"""Yet another OKCupid data mining thing.

If you want to use this software to interact with www.okcupid.com,
please first read and make sure you are complying with their terms of
service.

https://www.okcupid.com/legal/terms

Dependencies:
* Requests -- http://docs.python-requests.org

This library was developed for personal use rather than as a reusable
API for others, so no complaining now.

Instructions for use:

1. Copy the settings_example.py file to settings.py and modify its
   contents appropriately.

2. Find users:
   $ python okc.py --outpath=usernames.txt find

3. Scrape profiles from usernames found in usernames.txt into path 'profiles':
   $ python okc.py --outpath=profiles scrape usernames.txt

   If the scraper aborts prematurely, you can resume like so:
   $ python okc.py --outpath=profiles scrape usernames.json

Note that the default settings in settings.py will retrieve hopefully
almost all users in a 25 m/km radius from you. If you provide your login
credentials here you can omit them from the commandline.

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

PROFILE_URL = u'https://www.okcupid.com/profile/{username}'
LOGIN_URL = 'http://www.okcupid.com/login'
MATCH_URL = 'https://www.okcupid.com/match'
QUICKMATCH_URL = 'https://www.okcupid.com/quickmatch/{username}' 
VISITORS_URL = 'https://www.okcupid.com/visitors/{username}' 

HEADERS = {
    'User-agent' : USER_AGENT,
    'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
}

MATCH_ORDERS = (
    'MATCH',
    'SPECIAL_BLEND',
    'RANDOM',
    'ENEMY',
    'JOIN',
    'LOGIN',
    'MATCH_AND_NEW',
    'MATCH_AND_LOGIN',
    'MATCH_AND_DISTANCE',
)


def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--username", default=USERNAME)
    argparser.add_argument("--password", default=PASSWORD)
    argparser.add_argument("--outpath", default=os.getcwd())
    subparsers = argparser.add_subparsers(dest='command')

    ap1 = subparsers.add_parser('find')
    ap2 = subparsers.add_parser('scrape')
    ap2.add_argument("--resume", action='store_true')
    ap2.add_argument("inpath", metavar='USERNAME_PATH')
    return argparser


class OkcError(Exception):
    pass


class OkcNoSuchUserError(OkcError):
    pass


class OkcIncompleteProfileError(OkcError):
    def __str__(self):
        return 'Incomplete profile: {}'.format(self.message)

    
class User(object):
    """Models an OKC user profile.

    The JSON data representing a profile is stored with some
    attributes being pulled out explicitly to be used.

    Class attributes:

    json_data       copy of profile converted from JSON (dict)
    username        username (string)
    age             age (int)
    gender          gender (string)
    match           match score
    enemy           enemy score
    essay_titles    list of essay titles (strings)
    essays          list of essay contents (strings)
    text            combined text from all essays (string)
    num_tokens      number of tokens used in all essays (int)
    token_counts    frequencies  token occurrences in all text (Counter object)
    
    The nth item in essay_titles is the title of the essay in the nth
    position in essays list. A value of None in the essays list
    indicates the user did not fillout that essay.

    """
    
    def __init__(self, data):
        """data param = dict"""
        if 'matchpercentage' not in data:
            raise OkcIncompleteProfileError(data['username'])

        self.json_data = data
        self.username = data['username']
        self.age = int(data['age'])
        self.gender = int(data['gender'])
        self.match = data['matchpercentage']
        self.enemy = data['enemypercentage']
        self.process_essays()
        
    def process_essays(self):
        """Move essays from data object into two lists: a list of essay titles
        and a list of corresponding essay contents. Also massage the
        essay contents to have sensible newlines.

        """
        self.essays = []
        self.essay_titles = []
        for essay in self.json_data['essays']:
            self.essay_titles.append(essay['title'])
            this_essay = essay['essay']
            if this_essay == []:
                # User did not fill this essay out
                self.essays.append(None)
            else:
                # Single newlines are are removed and double newlines
                # are converted into single newlines.
                text = this_essay[0]['rawtext']
                text = re.sub(r'\n(?=[^\n])', ' ', text)
                self.essays.append(text)

        # do post processing on essay text
        tokens = self.tokens
        self.num_tokens = len(tokens)
        self.token_counts = Counter(tokens)
        
    @property
    def text(self):
        """Returns the complete text from all essays in a user's profile"""
        return '\n'.join(essay for essay in self.essays if essay is not None)

    @property
    def tokens(self):
        """Returns a list of tokens from all essays."""
        return re.split('\W+', self.text.lower())
    
    @property    
    def data(self):
        """Returns the user's profile data possibly with extra goodies."""
        extra = {'num_tokens' : self.num_tokens}
        self.json_data.update(extra)
        return self.json_data
    
    def __unicode__(self):
        return self.username

    def __str__(self):
        return self.username.encode('utf8')


def load_user(json_path):
    """Returns a User based on JSON profile file path; None if errors."""
    with open(json_path) as file:
        json_contents = file.read()
        data = json.loads(json_contents)
    try:
        return User(data)
    except OkcIncompleteProfileError as e:
        print e
        return None


def load_users(path):
    """Return a list of User objects from a directory of JSON profiles.""" 
    users = []
    for item in os.listdir(path):
        user = load_user(os.path.join(path, item))
        if user is not None:
            users.append(user)
    return users


class Session(object):
    """Class for interacting with okcupid.com."""
    
    def __init__(self, username=USERNAME, password=PASSWORD):
        """Logs into okcupid.com. Uses parameters for logging in if provided,
        otherwise defaults to credentials specified in settings.py.

        """
        self.login(username, password)

    def login(self, username, password):
        """Logs into okcupid.com."""
        params = {
            'username': username, 
            'password': password, 
            'okc_api' : '1',
        }

        r = requests.post(LOGIN_URL, params=params, headers=HEADERS)
        self.cookies = r.cookies
        
    def search(self, count=1000, matchorder='MATCH', locid=0, distance=DISTANCE, 
               min_age=MIN_AGE, max_age=MAX_AGE, gender='all', orientation='all',
               time='year'):
        """Make a search GET request to OKC API. Note: POST does not work.
        Returns a dictionary with the following keys:
        
        'username'         : user logged in as
        'foundany'         : 0 or 1 
        'maxmatch'         : highest match percentage found
        'lquery',          : ??
        'amateur_results'  : list of user dictionary results
        'total_matches'    : the number of results
        'cache_timekey'    : ??
        'alist_results'    : ??
        'last_online'      : unknown format
        'filters'          : dictionary of filters applied
        'numanswered'      : number of match questions answered

        """
        
        # Note that gender numbers depend on what has been specified
        # for orientation. This first set is for selecting
        # fine-grained every orientation possible -- ie 4095
        genders = {
            'male'   : 21,
            'female' : 42,
            'all'    : 63,
        }

        # for orientation = 'everybody'; ie no orientation param
        genders_orientation_everybody = {
            'male'   : 16,
            'female' : 32,
            'all'    : 48,
        }

        #last online
        times = {
            'now'   : 3600,
            'day'   : 86400,
            'week'  : 604800,
            'month' : 2678400,
            'year'  : 31536000,
        }

        # use most promiscuous values possible for rest...

        # This orientation number of 4095 is for selecting every tick
        # box possible in orientation. If you just want 'everyone',
        # omit this param.
        orientation = 4095
        status = 0
        
        # the filter names (eg 'filter1, 'filter2') are not relevant,
        # just indicate the nth filter applied. The value of the
        # filter parameters are themselves <key,value> pairs separated
        # by commas, indicating the filter type and value to be
        # filtered on.
        params = {
            'okc_api'       : 1,
            'timekey'       : 1,
            'discard_prefs' : 1,
            'count'         : count,
            'matchOrderBy'  : matchorder,
            'locid'         : locid,
            'filter1'       : '0,{}'.format(genders[gender]),
            'filter2'       : '76,{}'.format(orientation),
            'filter3'       : '2,{},{}'.format(min_age, max_age),
            'filter4'       : '3,{}'.format(distance),
            'filter5'       : '5,{}'.format(times[time]),
            'filter6'       : '35,{}'.format(status),
        }
        result = requests.get(MATCH_URL, cookies=self.cookies, params=params)
        return result.json()['amateur_results']

    def get_profile(self, username):
        """Given a username, return their profile as a JSON string."""
        params = {'okc_api' : 1}
        url = PROFILE_URL.format(username=username)
        result = requests.get(url, cookies=self.cookies, params=params)
        if result.status_code != requests.codes.ok:
            message = u"Error getting profile: {}\n{}".format(username, result.text)
            raise OKCNoSuchUserError(message)
        return result.json()

    def dump_profiles(self, usernames, path, resume=False):
        """Retrieves user profiles and write them all to disk."""
        for count, username in enumerate(usernames):
            outpath = os.path.join(path, u"{}.json".format(username))
            if resume and os.path.exists(outpath):
                continue
            try:
                user = self.get_profile(username)
                with open(outpath, 'w') as file:
                    json_string = json.dumps(user).encode('utf8')
                    file.write(json_string)
                print u"{}: Wrote {}".format(count+1, username)
            except OKCNoSuchUserError as error:
                print "NO SUCH USER: {}".format(username)
            except requests.ConnectionError as error:
                print "CONNECTION ERROR: {}".format(username)
            time.sleep(SLEEP_TIME)

    def visit_profiles(self, usernames):
        """Retrieves the profiles of a sequence of usernames. Nothing is done
        with the profiles, but this has the effect of making you appear in 
        their visitors list if you are not in invisible browsing mode.

        """
        for count, username in enumerate(usernames):
            try:
                user = self.get_profile(username)
                print u"{}: Visited {}".format(count+1, username)
            except OKCNoSuchUserError as error:
                print "NO SUCH USER: {}".format(username)
            except requests.ConnectionError as error:
                print "CONNECTION ERROR: {}".format(username)
            time.sleep(SLEEP_TIME)

    def find_and_visit_profiles(self, cutoff=None, threshold=None, **kwargs):
        """Perform a search for users and then visit their profiles.

        threshold argument:  minimum match percentage score to stop at.
        cuttoff argument:    number of profiles to stop at.
        
        Also accepts all keyword arguments accepted by the search()
        method.

        """
        profiles = self.search(**kwargs)

        if threshold is not None:
            usernames = [profile['username'] for profile in profiles if
                         int(profile['matchpercentage']) > threshold]

        if cutoff is not None:
            usernames = usernames[:cutoff]

        self.visit_profiles(usernames)

    def find_users(self, **kwargs):
        """Given search parameters as keyword arguments, returns a set of
        usernames.  Avoids the 1000 user response limit by repeating
        search for randomly sorted results until no new users are
        found.

        """
        usernames = set()
        num_found = True
        while num_found:
            num_before = len(usernames)
            profiles = self.search(matchorder='RANDOM', **kwargs)
            usernames.update([profile['username'] for profile in profiles])
            num_after = len(usernames)
            num_found = num_after - num_before 
            print "Found {} new users".format(num_found)
            time.sleep(SLEEP_TIME)
        return usernames

    def find_all_users(self, binsize=5, **kwargs):
        """In addition to the 1000 user result limit, OKC seems to also
        silently filter out some users from the search results if
        there are too many *potential* matches that could be returned.
        ie there are just some users that won't be returned no matter
        how many times you hit the RANDOM buttom. This can be avoided
        by searching over a smaller age range.

        This function gets around the limitation by invoking
        self.find_users() across a series of ranges of length
        specified by binsize.

        """
        usernames = set()
        pairs = []

        curr = MIN_AGE
        while curr <= MAX_AGE:
            if curr + binsize > MAX_AGE:
                this_max = MAX_AGE
            else:
                this_max = curr + binsize
            pairs.append((curr, this_max))
            curr += binsize + 1

        kwargs.pop('min_age', None)
        kwargs.pop('max_age', None)

        for min_age, max_age in pairs:
            found = self.find_users(min_age=min_age, max_age=max_age, **kwargs)
            usernames.update(found)
            print "===================================="
            print "Found {} users in age bracket {},{}\n".format(len(found), min_age, max_age)
        return usernames


def main():
    args = argparser().parse_args()
    session = Session(username=args.username, password=args.password)

    if args.command == 'find':
        usernames = session.find_all_users()
        with open(args.outpath, 'w') as file:
            file.write('\n'.join(usernames).encode('utf8'))
    elif args.command == 'scrape':
        with open(args.inpath) as file:
            usernames = set(file.read().decode('utf').split())
        session.dump_profiles(usernames, args.outpath, resume=args.resume)

    
if __name__ == "__main__":
    sys.exit(main())
    
