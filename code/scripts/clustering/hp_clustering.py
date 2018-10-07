'''
Preliminary Clustering Overview
1. create master feature dataframe (merge static/dynamic), selectively create cluster features dataframe
2. perform kmeans clustering using cluster features dataframe, append cluster allocation as additional feature within master dataframe
3. segment master dataframe based upon newly created cluster allocation, generate n cluster dataframes
4. evaluate the composition of the newly created, segmented dataframes
5. export cluster dataframes and cluster analysis

Cluster ToDo
Implement cluster analysis, export as seperate data object
Implement a variety of cluster algorithms => implement mechanism that is capable of selecting the "best" clusters
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
import time
import json


sys.path.append('../util/.')  # expose utility functions in parallel dir
import util

# define IO directories and files
dirs = {'static_features': '../../../data_sets/honey_pot/final_features/static_features.csv',
        'dynamic_features': '../../../data_sets/honey_pot/final_features/dynamic_features.csv',
        'cluster_frames': '../../../data_sets/honey_pot/final_features/cluster_frames',
        'param_import': '../configs/hp_cluster_config.json',
        'report_output': '../../output/clustering_results.json'}


def consolidate_features(static_features, dynamic_features):
    # join static and dynamic features
    static = util.import_frame(static_features)
    dynamic = util.import_frame(dynamic_features)

    # join features along axis
    master_df = pd.concat([static, dynamic], axis=1)

    # replace infinite entries with nan
    master_df.replace([np.inf, -np.inf], np.nan)
    master_df = master_df.dropna()  # drop all nan entries

    return master_df


def scale_features(df):
    # scale features to ensure clustering is not skewed by large values
    scaler = StandardScaler()
    df = df.as_matrix()

    return scaler.fit_transform(df)


def configure_df(static, dynamic, params):
    # consolidate static/dynamic features, select features, scale values
    master_df = consolidate_features(static, dynamic)

    # configure clustering dataframe
    features = util.extract_features(params['clustering_features'])
    cluster_df = util.choose_features(
        master_df, ['dt_entropy', 'num_unique_words'])

    # return master feature df and scaled cluster df
    return master_df, scale_features(cluster_df)


def create_kmeans(df, df_matrix, params):
    km = KMeans(n_clusters=params['cluster_config']['clusters'], random_state=42,
                max_iter=params['cluster_config']['max_iterations']).fit(df_matrix)  # create/fit kmeans to matrix

    df['cluster'] = km.labels_  # augment cluster result as attribute

    return df, km


def evaluate_frames(master_df, seg_df, results):
    for idx, each in enumerate(seg_df):  # membership composition
        composition = each['user_type'].value_counts().tolist()
        inter = {'size': each.shape[0], 'composition': {
            '0': composition[0], '1': composition[1]}}
        # append intermediate results to main
        results['cluster' + str(idx)] = inter

    results['total_entries'] = master_df.shape[0]
    print(json.dumps(results, sort_keys=True, indent=4))
    return results


def segment_df(df, params):
    # filter and split main df based upon cluster allocation
    segmented_frames = []

    # segment df based upon cluster
    for cluster in range(params['cluster_config']['clusters']):
        segmented_frames.append(df.loc[df['cluster'] == cluster])

    return segmented_frames


def main():
    st = time.time()
    print('\nPerforming cluster filtering for honeypot dataset..')
    params = util.parse_params(dirs['param_import'], 'Clustering')
    results = dict()

    # 1.create master dataframe
    df, cluster_matrix = configure_df(dirs['static_features'], dirs[
                                      'dynamic_features'], params)

    # 2.perform kmeans clustering, append cluster allocation as feature
    df, km = create_kmeans(df, cluster_matrix, params)

    # 3.segment master dataframe based upon cluster allocation
    seg_df = segment_df(df, params)

    # 3.describe cluster segment composition
    results = evaluate_frames(df, seg_df, results)

    # 5.export segmented dataframes and clustering results
    util.export_frames_destructive(seg_df, dirs['cluster_frames'])
    util.export_results(dirs['report_output'], results)

    et = time.time() - st
    print('\nCluster filtering completed in {0} seconds. Individual frames saved to:\n\n {1}'.format(
        et, dirs['cluster_frames']))

if __name__ == '__main__':
    main()
