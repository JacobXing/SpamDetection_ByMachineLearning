'''
Preprocessing High Level Overview
1. import raw data sets
2. preprocess static features and dynamic features (tweet corpus) dataframes
3. square dataframes; ensure all static features user_id's exist within tweet corpus and vice versa
(ensures a '1:many' relationship exists for user:tweets)
4. export finalized static features, intermediate dynamic features dataframes

Preprocessing Todo
Declare column datatypes upon import => optimize storage (eg. use int instead of floats)
Consolidate raw, static features as csv file instead of arff format 
'''

from scipy.io import arff
import pandas as pd
import numpy as np
import time
import sys

sys.path.append('../util/.')  # expose utility functions in parallel dir
import util

# define import/export directories
dirs = {'static_import': '../../../data_sets/honey_pot/preprocessed/FinalDataFull.csv.arff',
        'static_export': '../../../data_sets/honey_pot/final_features/static_features.csv',
        'dynamic_imports': ['../../../data_sets/honey_pot/raw/cp_tweets.csv',
                            '../../../data_sets/honey_pot/raw/lu_tweets.csv'],
        'dynamic_export': '../../../data_sets/honey_pot/preprocessed/dynamic_features_intermediate.csv',
        'param_import': '../configs/hp_preprocessing_config.json', }

# rename dataframe columns
static_rename_dict = {'UserID': 'user_id', 'UserType': 'user_type', 'NumberOfTweets': 'num_tweets', 'numOfFollowers': 'num_followers',
                      'numOfFollowings': 'num_followings', 'lengthAboutMe': 'about_me_length', 'lengthUsername': 'user_name_length',
                      'NumOfAnnotation': 'num_annotations', 'NumOfHttp': 'num_http', 'avgLengthOfTweets': 'tweet_avg_length',
                      'totalNumOfUniqWords': 'num_unique_words'}
dynamic_rename_dict = {'UserID': 'user_id', 'TweetID': 'tweet_id',
                       'Tweet': 'tweet', 'CreatedAt': 'creation_date'}


def preprocess_static_features(import_path):
    data = arff.loadarff(import_path)
    df = pd.DataFrame(data[0])

    # rename columns headers
    df.rename(columns=static_rename_dict, inplace=True)

    # correct for usertype boolean type
    df['user_type'] = df['user_type'].astype(int)

    # drop any duplicate entries
    return df.drop_duplicates(['user_id'])


def preprocess_dynamic_features(import_paths):
    cp_tweets = preprocess_tweet_set(import_paths[0])
    lu_tweets = preprocess_tweet_set(import_paths[1])

    # ensure cp and lu tweet sets are complimentary
    cp_tweets_set = cp_tweets.loc[~cp_tweets['user_id'].isin(
        lu_tweets['user_id'])]  # negated match
    lu_tweets_set = lu_tweets.loc[~lu_tweets[
        'user_id'].isin(cp_tweets['user_id'])]

    # flag as illegitimate/legitimate users
    cp_tweets_set['user_type'] = 1
    lu_tweets_set['user_type'] = 0

    # merge data frames
    return pd.concat([cp_tweets_set, lu_tweets_set])


def preprocess_tweet_set(tweets_path):
    tweets = util.import_frame(tweets_path)

    # rename columns headers
    tweets.rename(columns=dynamic_rename_dict, inplace=True)

    # variable removal
    tweets = tweets.drop(['tweet_id', 'creation_date'],
                         axis=1)

    # flag NAN entries in UserID column
    tweets['user_id'] = pd.to_numeric(tweets['user_id'], errors='coerce')

    # remove row entries with malformed UserID
    tweets = tweets.dropna(subset=['user_id'])

    # tweet collating, group by user_id
    tweets = tweets.groupby(['user_id'])['tweet'].apply(
        list)

    # join tweets together into single document
    tweetJoin = lambda x: ' '.join('%s' %id for id in x)
    tweets = tweets.apply(tweetJoin)

    # dataframe reformatting
    tweets = tweets.to_frame()  # cast back to frame
    tweets = tweets.reset_index()  # reset/adjust index

    return tweets


def square_frames(df_a, df_b, params):
    # join two dataframes based upon user_id
    df_a_set = df_a.loc[df_a['user_id'].isin(
        df_b['user_id'])]  # ensure for match
    df_b_set = df_b.loc[df_b['user_id'].isin(df_a['user_id'])]

    # return small sample for prototyping speed, if specified
    if (params['random_sub_sample']):
        df_a_set = df_a_set.sample(
            n=params['sample_size'], axis=0, random_state=42)
        df_b_set = df_b_set.loc[df_b['user_id'].isin(
            df_a_set['user_id'])]  # match user type

    return df_a_set, df_b_set  # retain original frame housing


def main():
    st = time.time()
    print('\nPreprocessing static and dynamic features..\n')
    params = util.parse_params(dirs['param_import'], 'Preprocessing')

    # 1. preprocess static and dynamic dataframes
    static_df = preprocess_static_features(dirs['static_import'])
    dynamic_df = preprocess_dynamic_features(dirs['dynamic_imports'])

    # 2. square dataframes; ensure 1:many user:tweets relationship
    static_df, dynamic_df = square_frames(static_df, dynamic_df, params)
    util.export_frames([static_df, dynamic_df], [dirs['static_export'],
                                                 dirs['dynamic_export']])  # 3. export dataframes

    et = time.time() - st
    print('\nPreprocessing completed in {0} seconds. Preprocessed files saved to:\n\n {1}\n{2}'.format(
        et, dirs['static_export'], dirs['dynamic_export']))

if __name__ == '__main__':
    main()
