import json
import glob
import os
import pandas as pd

# parse user params contained within a json file


def parse_params(filename, operation):
    with open(filename, 'r') as f:
        params = json.load(f)
        print('Executing {0} with the following params:\n {1}'.format(
            operation, json.dumps(params, sort_keys=True, indent=4)))
        return params

# retrieve list of file names within specified dir


def retrieve_files(folder_location):
    return glob.glob(folder_location + '/*')

# trim path names of files, used for labelling


def trim_file_paths(files):
    return list(os.path.basename(each) for each in files)

# save program execution output


def save_output(output_dict, filename):
    with open(filename, 'w') as f:
        print(output_dict)
        json.dump(datastore, f)


def export_results(file_name, results):
    # Writing JSON data
    with open(file_name, 'w') as f:
        json.dump(results, f)

# single frame import


def import_frame(filename):
    return pd.read_csv(filename, error_bad_lines=False)

# multi frame import


def import_frames(frames):
    all_frames = []

    for df in frames:
        all_frames.append(import_frame(df))
    return all_frames

# single frame export


def export_frame(df, file_path):
    df.to_csv(file_path, index=False)
    return df

# multi frame export


def export_frames(frames, locations):
    for frame, location in zip(frames, locations):
        export_frame(frame, location)

# clear parent directory of all files, export clustered frames


def export_frames_destructive(frames, folder_location):
    files = glob.glob(folder_location + '/*')  # clear holding dir
    for f in files:
        os.remove(f)

    for idx, df in enumerate(frames):  # enumerate/write clustered frames
        f_name = folder_location + '/cluster' + str(idx) + '.csv'
        export_frame(df, f_name)

# format list of values to 2 decimal places, casts values to strings


def decimal_format(values):
    return list('{0:.2f}'.format(each) for each in values)

# format sklearn classification report output


def format_class_report(class_report):
    # extract, reformat and dictorize np arrays
    class_report = {'precision': decimal_format(class_report[0].tolist()),
                    'recall': decimal_format(class_report[1].tolist()),
                    'f1': decimal_format(class_report[2].tolist()),
                    'support': decimal_format(class_report[3].tolist())}
    return class_report

# extract list of features used to construct custom dataframe


def extract_features(feature_list):
    features = []
    for key, value in feature_list.items():
        if (value):
            features.append(key)
    return features

# create custom dataframe based upon feature list


def choose_features(df, feature_list):
    return df.loc[:, feature_list]
