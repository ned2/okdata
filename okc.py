import json
import glob
import time
import re
import os

from collections import Counter
from itertools import chain

from nltk.tokenize import word_tokenize


import requests
import settings


PROFILE_URL = u'https://www.okcupid.com/profile/{username}'
QUESTIONS_URL = u'https://www.okcupid.com/profile/{username}/questions'
LOGIN_URL = 'http://www.okcupid.com/login'
MATCH_URL = 'https://www.okcupid.com/match'
QUICKMATCH_URL = 'https://www.okcupid.com/quickmatch/{username}' 
VISITORS_URL = 'https://www.okcupid.com/visitors/{username}' 

HEADERS = {
    'User-agent' : settings.USER_AGENT,
    'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
}

USER_DATA_FILE = 'USER_DATA'

LOCATIONS = {
    'melbourne'     : 976925,
    'sydney'        : 974455,
    'san_francisco' : 4265540,
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


def get_user_paths(path):
    """Returns an iterator with all sub-paths paths corresponding to *.json"""
    return glob.glob(os.path.join(path, '*.json'))


def load_user(json_path):
    """Returns a User based on JSON profile file path"""
    with open(json_path, encoding='utf-8') as file:
        json_contents = file.read()
        data = json.loads(json_contents)
        return User(data)


def load_users(collection_path):
    """Return an iterator of User objects from a directory of JSON profiles.""" 
    for path in get_user_paths(collection_path):
        try:
            yield load_user(path)
        except (ValueError, OkcIncompleteProfileError) as e:
            print(e)
            pass

            
def filter_users(paths, question_min):
    """Takes a list of user paths and filters out ones with fewer than question_min answered."""
    filtered_paths = []

    # load the additional user data stored alongside profiles
    user_data_path = os.path.join(os.path.split(paths[0])[0], USER_DATA_FILE)

    with open(user_data_path, encoding='utf-8') as f:
        user_data = json.loads(f.read())

    for user_path in paths:
        username = os.path.splitext(os.path.basename(user_path))[0]

        if username not in user_data:
            print("user {} missing from user data dictionary".format(username))
        else:
            data = user_data[username] 
            if data['num_questions'] >= question_min:
                yield user_path

    
def write_user_data(data, path):
    user_data_path = os.path.join(path, USER_DATA_FILE)

    with open(user_data_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data))

        
class OkcError(Exception):
    pass


class OkcNoSuchUserError(OkcError):
    def __str__(self):
        return 'No such user: {}'.format(self.message)


class OkcIncompleteProfileError(OkcError):
    def __str__(self):
        return 'Incomplete profile: {}'.format(self.message)

    
class User(object):
    """Models an OKC user profile.

    Instance attributes:

    username        username (string)
    age             age (int)
    gender          gender (string)
    match           match score
    enemy           enemy score
    essay_titles    list of essay titles (strings)
    essays          list of essay contents (strings)
    text            combined text from all essays (string)
    tokens
    
    The nth item in essay_titles is the title of the essay in the nth
    position in essays list. A value of None in the essays list
    indicates the user did not fillout that essay.

    """
    
    def __init__(self, data):

        if 'matchpercentage' not in data:
            raise OkcIncompleteProfileError(data['username'])

        self.username = data['username']
        self.age = int(data['age'])
        self.gender = int(data['gender'])
        self.match = int(data['matchpercentage'])
        self.enemy = int(data['enemypercentage'])
        self.essays = self.process_essays(data)
        
    def process_essays(self, data):
        found_essays = []
        for essay in data['essays']:
            this_essay = essay['essay']
            if this_essay == []:
                # User did not fill this essay out
                found_essays.append([])
            else:
                text = this_essay[0]['rawtext']
                found_essays.append(text)
        return found_essays
    
    def get_tokens(self, tokenize=word_tokenize):
        """Returns an iterator that yields tokens from all essays"""
        essay_tokens = (word_tokenize(essay) for essay in self.essays if essay)
        return chain.from_iterable(essay_tokens)
    
    @property
    def text(self):
        """Returns the complete text from all essays in a user's profile"""
        text = '\n'.join(essay for essay in self.essays if essay)
        return text

    def __str__(self):
        return self.username


class Session(object):
    """Class for interacting with okcupid.com."""
    
    def __init__(self, username=settings.USERNAME, password=settings.PASSWORD):
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
        
    def search(self, count=1000, matchorder='MATCH', location=None, distance=settings.DISTANCE, 
               min_age=settings.MIN_AGE, max_age=settings.MAX_AGE, gender='all', orientation='all',
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
        
        # location
        if location is None:
            locid = 0
        else:
            locid = LOCATIONS[location]
        
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
            message = "Error getting profile: {}\n{}".format(username, result.text)
            raise OkcNoSuchUserError(message)
        return result.json()

    def get_num_questions(self, username):
        params = {'okc_api' : 1}
        url = QUESTIONS_URL.format(username=username)
        result = requests.get(url, cookies=self.cookies, params=params)
        if result.status_code != requests.codes.ok:
            message = "Error getting profile: {}\n{}".format(username, result.text)
            raise OkcNoSuchUserError(message)
        json_data = result.json()
        num_questions = int(json_data['pagination']['raw']['total_num_results'])
        return num_questions
    
    def dump_profiles(self, usernames, path, resume=False):
        """Retrieves user profiles and write them all to disk."""
        user_data = get_user_data(path)
        try:
            for count, username in enumerate(usernames):
                outpath = os.path.join(path, "{}.json".format(username))
                if resume and os.path.exists(outpath):
                    continue
                try:
                    user = self.get_profile(username)
                    num_questions = self.get_num_questions(username)
                    if username not in user_data:
                        user_data[username] = {}
                    user_data[username]['num_questions'] = num_questions
                    with open(outpath, 'w', encoding='utf-8') as file:
                        json_string = json.dumps(user)
                        file.write(json_string)    
                    print("{}: Wrote {}".format(count+1, username))
                except OkcNoSuchUserError as error:
                    print("NO SUCH USER: {}".format(username))
                except requests.ConnectionError as error:
                    print("CONNECTION ERROR: {}".format(username))
                time.sleep(settings.SLEEP_TIME)
        except:
            write_user_data(user_data, path)
            raise
        self.write_user_data(user_data, path)

    def visit_profiles(self, usernames):
        """Retrieves the profiles of a sequence of usernames. Nothing is done
        with the profiles, but this has the effect of making you appear in 
        their visitors list if you are not in invisible browsing mode.

        """
        for count, username in enumerate(usernames):
            try:
                self.get_profile(username)
                print("{}: Visited {}".format(count+1, username))
            except OkcNoSuchUserError as error:
                print("NO SUCH USER: {}".format(username))
            except requests.ConnectionError as error:
                print("CONNECTION ERROR: {}".format(username))
            time.sleep(settings.SLEEP_TIME)

    def find_and_visit_profiles(self, cutoff=None, threshold=None, **kwargs):
        """Perform a custom search for users and then visit their profiles.

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
            usernames.update(profile['username'] for profile in profiles)
            num_after = len(usernames)
            num_found = num_after - num_before 
            print("Found {} new users".format(num_found))
            time.sleep(settings.SLEEP_TIME)
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

        curr = settings.MIN_AGE
        while curr <= settings.MAX_AGE:
            if curr + binsize > settings.MAX_AGE:
                this_max = settings.MAX_AGE
            else:
                this_max = curr + binsize
            pairs.append((curr, this_max))
            curr += binsize + 1

        kwargs.pop('min_age', None)
        kwargs.pop('max_age', None)

        for min_age, max_age in pairs:
            found = self.find_users(min_age=min_age, max_age=max_age, **kwargs)
            usernames.update(found)
            print("====================================")
            print("Found {} users in age bracket {},{}\n".format(len(found), min_age, max_age))
        return usernames
