'''
Wrapper Overview
A wrapper script capable of calling various combinations of process subscripts for either/both the honey pot or 14 million tweets datasets.
'''

import sys
sys.path.append('../util/')
sys.path.append('../preprocessing/')
sys.path.append('../feature_engineering/')
sys.path.append('../clustering/')
sys.path.append('../classification/')

# honeypot submodules
import util
import hp_preprocessing
import hp_dynamic_feature_generation
import hp_clustering
import hp_classification


def run_hp_sub_processes(process_dict):
    if(process_dict['hp_preprocessing']):
        hp_preprocessing.main()

    if(process_dict['hp_dynamic_feature_generation']):
        hp_dynamic_feature_generation.main()

    if(process_dict['hp_clustering']):
        hp_clustering.main()

    if(process_dict['hp_classification']):
        hp_classification.main()


def main():
    params = util.parse_params('./wrapper_config.json', 'Pseudo-Main')

    # execute honey pot analysis
    if (params['hp']['in_use']):
        run_hp_sub_processes(params['hp'])

if __name__ == '__main__':
    main()
