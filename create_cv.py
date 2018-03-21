import os
import numpy as np
import glob
import csv
import argparse
import re
import yaml
from sklearn.model_selection import KFold

import prepare_data as pp_data
import config as cfg


def create_cv_csv(out_path):
    """Create cross validation csv file. 
    """
    dataset_dir = cfg.dataset_dir
    workspace = cfg.workspace
    events = cfg.events
    n_folds = cfg.n_folds
    
    pp_data.create_folder(os.path.dirname(out_path))
    f = open(out_path, 'w')
    f.write("name\tfold\n")
    
    names = os.listdir(dataset_dir)
    
    for event in events:
        event_names = [e for e in names if event in e]
        kf = KFold(n_splits=n_folds, shuffle=False, random_state=None)
        fold = 0
        for (tr_idxes, te_idxes) in kf.split(event_names):
            for idx in te_idxes:
                event_name = event_names[idx]
                f.write("%s\t%d\n" % (event_name, fold))
            fold += 1
    f.close()
    
    print("Write out to %s" % n_folds)
    
    
def _get_n_largest_events(dict, n_largest, rs):
    """Return n events with the most files. 
    """
    pairs = [(e, len(dict[e])) for e in dict.keys()]    
    idxes = np.arange(len(pairs))
    rs.shuffle(idxes)
    pairs = [pairs[e] for e in idxes]   # Random shuffle
    pairs = sorted(pairs, key=lambda e: e[1], reverse=True)   # Sort 
    n_largest_events = [pair[0] for pair in pairs][0 : n_largest]
    return n_largest_events
        
        
def _get_n_elements_in_dict(dict):
    return np.sum([len(dict[e]) for e in dict.keys()])
    
    
def create_mix_yaml(cv_path, n_events, out_path):
    """Create yaml file containing the mixture information. 
    """
    workspace = cfg.workspace
    events = cfg.events
    n_folds = cfg.n_folds
    onset_list = cfg.onset_list
    
    rs = np.random.RandomState(0)
    
    # Read cross validation csv
    cv_path = os.path.join(workspace, "cross_validation.csv")
    with open(cv_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
    
    yaml_data = []
    cnt = 0
    for tar_fold in xrange(n_folds):
        for loop in xrange(n_events):
            
            # Initialize dict
            dict = {}
            for e in events:
                dict[e] = []
                
            # Read all rows in cross validation csv
            for i1 in xrange(1, len(lis)):
                [name, fold] = lis[i1]
                fold = int(fold)
                if fold == tar_fold:
                    for e in events:
                        if e in name:
                            dict[e].append(name)
    
            while _get_n_elements_in_dict(dict) >= n_events:
                # Randomly select event files. 
                selected_names = []
                events_pool = _get_n_largest_events(dict, n_events, rs)
    
                selected_events = rs.choice(events_pool, size=n_events, replace=False)
                for e in selected_events:
                    sel_na = rs.choice(dict[e], replace=False)
                    sel_na = str(sel_na)
                    selected_names.append(sel_na)
                    dict[e].remove(sel_na)
                    if len(dict[e]) == 0:
                        dict.pop(e)
                    
                # Combine yaml info. 
                mixture_data = {'name': "%05d.wav" % cnt, 
                            'fold': tar_fold, 
                            'events': []}
                cnt += 1
                for (j1, na) in enumerate(selected_names):
                    event_data = {'file_name': na, 
                                'event': re.split('(\d+)', na)[0], 
                                'onset': onset_list[j1], 
                                'fold': 0}
                    mixture_data['events'].append(event_data)
                    
                yaml_data.append(mixture_data)
                
    # Write out yaml file. 
    pp_data.create_folder(os.path.dirname(out_path))
    with open(out_path, 'w') as f:
        f.write(yaml.dump(yaml_data, default_flow_style=False))
    print("len(yaml_file): %d" % len(yaml_data))
    print("Write out to %s" % out_path)


if __name__ == '__main__':
    workspace = cfg.workspace
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_a = subparsers.add_parser('create_cv_csv')
    parser_b = subparsers.add_parser('create_mix_yaml')
    
    args = parser.parse_args()
    
    # Create files for cross validation
    if args.mode == "create_cv_csv":
        create_cv_csv(out_path=os.path.join(workspace, "cross_validation.csv"))
        
    # Create mixture yaml file
    elif args.mode == "create_mix_yaml":
        for n_events in [1, 2, 3, 4]:
            create_mix_yaml(cv_path=os.path.join(workspace, "cross_validation.csv"), 
                            n_events=n_events, 
                            out_path=os.path.join(workspace, "mixture_yamls", "n_events=%d.yaml" % n_events))
