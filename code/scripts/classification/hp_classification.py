'''
Classification Overview
1. import all cluster-segregated dataframes
2. for each dataframe, generate specified models using specified features
3. for each group of models, evaluate the accuracy of the models
4. concatenate results into single dictionary object, export object

Model Evaluation Notes
Each model is evaluated using kfold cross validation and other
performance metrics (confusion matrix, precision, recall, f1 and support
scores).

To keep the output of this analysis manageable, similar yet distinct
splitting operations are performed upon the data. For KFold CV, the
data is split into x/y components. Cross fold validation is performed
via an sklearn shuffle-split CV iterator, upon which an averaged
accuracy score is obtained over the sequence of CV outcomes.

The latter evaluations (conf. matrix etc.) are derived from a model
fitted with the data once (as opposed to multiple fits, as is the
case with CV). To facilitate this, the data is split once into
x/y, train/test portions and fitted/analysed. There is therefore
a difference in the way the accuracy derived and the way the latter
analysis is derived. A stratified and shuffled mechanism is used to
split the data for the lattter evaluations, in an attempt to replicate
the mechanism utilized by the CV iterator.

Classification ToDo
Visualize model performance => export visualizations as additional outputs
'''

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC  # or linearSVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from collections import defaultdict
import numpy as np
import time
import sys
import os

sys.path.append('../util/.')  # expose utility functions in parallel dir
import util

# define IO directories and files
dirs = {'cluster_frames': '../../../data_sets/honey_pot/final_features/cluster_frames',
        'param_import': '../configs/hp_classification_config.json',
        'report_output': '../../output/classification_results.json'}


def extract_models(model_dict):
    # return dictionary of models based upon user specification
    selected_models = {'decision_tree': DecisionTreeClassifier(),  # prepopulate model options
                       'random_forest': RandomForestClassifier(),
                       'adaboosted_dt': AdaBoostClassifier(),
                       'linear_svc': LinearSVC()}

    for key, val in model_dict.items():  # filter based upon config input
        if not (val):
            del selected_models[key]

    return selected_models


def course_split(df):
    # entire x/y partition for cross fold validation process
    return df.drop(['user_type'], axis=1), df['user_type']


def fine_split(df):
    # x/y train/test splits, required for one time fitting
    y = df['user_type']
    X = df.drop(['user_type'], axis=1)
    X_mat = X.as_matrix().astype(np.float)
    return train_test_split(X_mat, y, test_size=0.4, random_state=42, stratify=y, shuffle=True)


def kfold(model, model_name, X, y):
    # define cv iterator parameters
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.4, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)

    kfold_results = {'averaged_accuracy_score': '{0:.2f}'.format(scores.mean()),
                     'averaged_accuracy_sd': '{0:.2f}'.format(scores.std() * 2)}

    # retrieve trained model accuracy using cross fold validation score -
    # using all data
    print("{0} Accuracy: {1:.2f} (+/- {2:.2f})".format(model_name,
                                                       scores.mean(), scores.std() * 2))
    return kfold_results

# evaluate models performance using classification report and confusion matrix


def metrics(model, X_train, X_test, y_train, y_test):
    # classification report and confusion matrix - using train/test partitions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metric_results = defaultdict(dict)
    class_report = util.format_class_report(
        precision_recall_fscore_support(y_test, y_pred))
    metric_results = {'classification_report': class_report,
                      'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()}

    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return metric_results


def generate_models_single(df, df_name, models, results_dict):
    print('Generating models for {}'.format(df_name))
    # intermediate results object for single frame
    temp_dict = defaultdict(dict)

    # partion dataframe
    X, y = course_split(df)
    X_train, X_test, y_train, y_test = fine_split(df)

    # evaluate all models, append results
    for model_name, model in models.items():
        st = time.time()

        # accuracy score based upon multiple model accuracy scores => kfold
        # validation
        temp_dict[model_name]['kfold_scores'] = kfold(model, model_name, X, y)
        # scores based upon a single model fitting
        temp_dict[model_name]['accuracy_metrics'] = metrics(
            model, X_train, X_test, y_train, y_test)

        # record time taken to fit/evaluate each model
        et = time.time() - st
        temp_dict[model_name]['time_elapsed'] = '{0:.2f}'.format(et)

    # append to main results object
    results_dict[df_name] = temp_dict


def generate_models_all(df_list, df_names, params, results_object):
    # generate models for a list of dataframes

    # configure model listing
    models = extract_models(params['classification_models'])

    for df, df_name, feature_dict in zip(df_list, df_names, params['classification_features']):
        # configure features for each dataframe
        features = util.extract_features(feature_dict)
        df = util.choose_features(df, features)
        generate_models_single(df, df_name, models, results_object)


def main():
    st = time.time()
    print('\nPerforming classification upon final dataframe/s for honeypot dataset..\n')
    params = util.parse_params(dirs['param_import'], 'Classification')
    results = defaultdict(dict)

    # 1. import all dataframes
    file_names = util.retrieve_files(
        dirs['cluster_frames'])
    all_frames = util.import_frames(file_names)

    # 2. generate all models
    generate_models_all(all_frames, util.trim_file_paths(
        file_names), params, results)

    et = time.time() - st
    results['total_time_elapsed'] = '{0:.2f}'.format(et)

    # 3. export all results
    util.export_results(dirs['report_output'], results)
    print('\nClassifications completed in {0} seconds.\n Output saved to {1}'.format(
        et, dirs['report_output']))

if __name__ == '__main__':
    main()
