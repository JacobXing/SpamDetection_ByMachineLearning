'''
Feature Engineering Overview
1. import intermediate, dynamic features dataframe (tweet corpus)
2. configure and generate count vector
3. create term frequency matrix by vectorizing tweet corpus, applying lemmatization, stemming etc.
4. generate dynamic features using term frequency matrix (GOSS, LOSS, document-topic distribution entropy)
5. export dynamic features dataframe
'''

from nlp_vector_config import generate_vector, vectorize
from dynamic_features import generate_dynamic_features
import time
import sys

sys.path.append('../util/.')  # expose utility functions in parallel dir
import util

# define import/export directories
dirs = {'dynamic_import': '../../../data_sets/honey_pot/preprocessed/dynamic_features_intermediate.csv',
        'dynamic_export': '../../../data_sets/honey_pot/final_features/dynamic_features.csv',
        'param_import': '../configs/hp_fe_config.json'}


def main():
    st = time.time()
    print('\nGenerating dynamic features for honeypot dataset..\n')
    params = util.parse_params(dirs['param_import'], 'Feature Engineering')

    # 1.import dataframe
    df = util.import_frame(dirs['dynamic_import'])

    # 2.configure count vector
    cv = generate_vector(params['count_vector'])

    # 3. vectorize corpus, create term frequency matrix
    tf_matrix, tf_feature_names = vectorize(cv, df)

    # 4.generate dynamic features
    df = generate_dynamic_features(tf_matrix, params['lda_modelling'])

    # 5.export dataframe
    util.export_frame(df, dirs['dynamic_export'])

    et = time.time() - st
    print('\nDynamic feature generation completed in {0} seconds. Features saved to:\n {1}'.format(
        et, dirs['dynamic_export']))

if __name__ == '__main__':
    main()
