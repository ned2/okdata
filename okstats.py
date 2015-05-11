import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import okc


"""
This code may be currently broken.

Male token count histogram:
df[(df['num_tokens'] < 800) & (df['gender'] == 1)]['num_tokens'].plot(kind='hist', bins=10)

Female Token counte histogram:
df[(df['num_tokens'] < 800) & (df['gender'] == 2)]['num_tokens'].plot(kind='hist', bins=10)
"""


def get_dataframe(path):    
    users = okc.load_users(path)
    return pd.DataFrame(u.data for u in users)

    
def ratio_gender_at_age(users, age_min, age_max):
    users = [u for u in users if age_min <= u.age <= age_max]
    tot = len(users)
    len_m = len([u for u in users if u.gender == 1])
    len_f = len([u for u in users if u.gender == 2])
    return len_m/tot, len_f/tot


def gender_proportion(df, path=None):
    hist = munge(df)
    hist[['male%', 'female%']].plot(kind='bar', stacked=True)

    if path is None:
        plt.show()

        
def age_score(df, score='matchpercentage', gridsize=15, path=None):
    """Plot age against percentage scores. Use a hexgrid as data is too
    dense for a scatterplot. Looks like any signal here is swamped by
    age bias in this population and/or age bias in who is more likely 
    to answer questions

    Score options are: matchpercentage, enemypercentage, friendpercentage

    """
    # filter out zero matches
    filtered = df[df['matchpercentage'] > 0]
    filtered.plot(kind='hexbin', x='age', y=score, gridsize=gridsize)

    if path is None:
        plt.show()

    
def histogram(df, path=None):   
    """Creates a histogram plot of an already age-binned dataframe"""
    hist = munge(df)
    hist[['male', 'female']].plot(kind='bar', stacked=True)
    if path is None:
        plt.show()


def munge(df):
    # get the counts of users across age/gender
    df1 = df.groupby(['age', 'gender'])['gender'].apply(np.sum).unstack('gender')
    df1.rename(columns={1:'male', 2:'female'}, inplace=True)

    # insert total (male + female) column
    total_col = df1[['male', 'female']].apply(np.sum, axis=1)
    df1.insert(0, 'total', total_col)

    # add colums with relative percentages
    func = lambda x:100*x.astype(float)/x.sum()
    df1[['male%', 'female%']] = df1[['male', 'female']].apply(func, axis=1)

    # get the average match score across age/gender
    df2 = df.groupby(['age', 'gender'])['matchpercentage'].apply(np.mean).unstack('gender')    
    df2.rename(columns={1:'male_match', 2:'female_match'}, inplace=True)

    # get the average match score across age/gender
    df3 = df.groupby(['age', 'gender'])['enemypercentage'].apply(np.mean).unstack('gender')    
    df3.rename(columns={1:'male_enemy', 2:'female_enemy'}, inplace=True)

    return pd.concat((df1, df2, df3), axis=1)


