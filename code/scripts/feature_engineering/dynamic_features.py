'''
Dynamic Feature Generation Overview
1. create LDA model object based upon a supplied term frequency corpus (vectorized corpus)
2. generate Document/Topic and Topic/Word (unused currently) distributions based upon LDA model object
3. calculate DT distribution entropy for all entries within term frequency matrix
4. calculate LOSS and GOSS scores for all entries within term frequency matrix, utilizing DT distribution

Dynamic Feature ToDo
Optimize GOSS/LOSS functions
'''

from sklearn.decomposition import LatentDirichletAllocation
import scipy as scipy
import numpy as np
import pandas as pd
from math import sqrt
import sys

sys.path.append('../util/.')  # expose utility functions in parallel dir
from util import export_frame


def create_lda(tf_matrix, params):
    return LatentDirichletAllocation(n_components=params['lda_topics'], max_iter=params['iterations'],
                                     learning_method='online', learning_offset=10,
                                     random_state=0).fit(tf_matrix)


def create_tw_dist(model):
    # return normalized topic-word distribution
    normTWDist = model.components_ / \
        model.components_.sum(axis=1)[:, np.newaxis]

    return normTWDist


def create_dt_dist(model, tf_matrix):
    # return normalized document-topic distribution
    normDTDist = model.transform(tf_matrix)

    return normDTDist


def entropy_single(x):
    # calculate entropy for a given sequence of values
    return scipy.stats.entropy(x)


def entropy_all(dt_dist):
    # calculate entropy for an entire document-topic distribution
    np_entropy = np.apply_along_axis(entropy_single, axis=1, arr=dt_dist)

    return pd.DataFrame(np_entropy, columns=['dt_entropy'])


def single_goss(topic_dist, i, k):
    # calculate GOSS score for a single particular user/topic (i/k) combination

    # 1.0 return mu(xk) for specific topic, sum topic probabilities for all
    # users, average across all users
    mu_xk = np.sum(topic_dist[:, k]) / topic_dist.shape[0]

    # 2.0 GOSS equation numerator
    goss_numerator = topic_dist[i, k] - mu_xk

    # 3.0 for all users specific topic probability:
    # - sum the squared difference of their relevant topic probability
    # - find the square of this sum
    goss_denominator = 0
    for user_prob in topic_dist[:, k]:
        goss_denominator += (user_prob - mu_xk) ** 2

    # 3.1 find sqrt of goss_denominator
    goss_denominator = sqrt(goss_denominator)

    # 4.0 divide numerator/denominator to find final GOSS score for user/topic
    # combination
    return goss_numerator / goss_denominator


def all_goss(topic_dist):
    # calculate GOSS scores for a particular topic distribution
    goss = []
    topics = range(topic_dist.shape[1])
    topic_labels = list('goss_' + str(each) for each in topics)

    for user in range(topic_dist.shape[0]):  # each user
        temp_goss = list(single_goss(topic_dist, user, topic)
                         for topic in topics)  # calculate all GOSS scores per topic
        goss.append(temp_goss)  # store all GOSS via nested lists

    np_goss = np.array(goss)  # recast as np array..
    return pd.DataFrame(goss, columns=topic_labels)  # and then to pandas df..


def single_loss(topic_dist, i, k):
    # calculate loss score for a particular user/topic (i/k) combination
    # 1.0 return mu(xi) for specific user, sum topic probabilities, return
    # average
    mu_xi = np.sum(topic_dist[i, :]) / topic_dist.shape[1]

    # 2.0 calculate muXI diff - GOSS equation numerator
    loss_numerator = topic_dist[i, k] - mu_xi

    # 3.0 for all topics (k) and a specific user (i):
    # - sum the squared difference of all associated topic probabilities and mu(xi)
    # - find the square of this sum
    loss_denominator = 0
    for user_prob in topic_dist[i, :]:
        loss_denominator += (user_prob - mu_xi) ** 2

    # 3.1 find sqrt of loss denominator
    loss_denominator = sqrt(loss_denominator)

    # 4.0 divide loss numerator by loss denominator to find loss score for
    # specific user
    return loss_numerator / loss_denominator


def all_loss(topic_dist):
    # calculate LOSS scores for a particular topic distribution
    loss = []
    topics = range(topic_dist.shape[1])
    topic_labels = list('loss_' + str(each) for each in topics)

    for user in range(topic_dist.shape[0]):  # each user
        temp_loss = list(single_loss(topic_dist, user, topic)
                         for topic in topics)  # calculate all loss scores per topic
        # store all loss scores for each user via nested lists
        loss.append(temp_loss)

    np_loss = np.array(loss)  # cast to np array..
    # and finally to pandas df..
    return pd.DataFrame(np_loss, columns=topic_labels)


def generate_dynamic_features(tf_matrix, params):
    # 1. fit LDA model using term frequency matrix
    lda = create_lda(tf_matrix, params)

    # 2. generate document-topic distribution
    dt_dist = create_dt_dist(lda, tf_matrix)

    # 3. retrieve entropy for document-topic distribution
    dt_entropy = entropy_all(dt_dist)

    # 4. retrieve GOSS and LOSS scores
    goss_df = all_goss(dt_dist)
    loss_df = all_loss(dt_dist)

    # 5. glue new features together into single df
    dynamic_features = pd.concat([dt_entropy, goss_df, loss_df], axis=1)

    # sanitize for possible NAN entries, possible bug withing GOSS/LOSS
    # generator functions
    dynamic_features = dynamic_features.dropna()

    return dynamic_features
