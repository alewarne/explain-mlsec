import json
from sklearn.feature_extraction.text import CountVectorizer
import os


# converts list of filepaths into sklearn count vectorizer
def get_count_vectorizer(documents_path, label_dict):
    all_paths = [os.path.join(documents_path, name) for name in os.listdir(documents_path) if name in label_dict]
    vec = CountVectorizer(input='filename', token_pattern='.+', lowercase=False)
    vec.fit_transform(all_paths)
    return vec


# returns dict with key=sha256 hash of malware, value = 0/1 where 1 indicates malware and 0 indicates no malware
# malware label is set if at least 'min_no_positive_scans' scanners return label "detected: true"
def virustotal_json_to_labels(path_to_virustotal_json, min_no_positive_scans):
    label_dict = {}
    with open(path_to_virustotal_json, 'r') as f:
        # json modules needs json dicts in a list seperated by comma
        lines = f.readlines()
        json_text = '[' + ','.join(lines) + ']'
        data = json.loads(json_text)
        for d in data:
            # for successful scans...
            if d['response_code'] == 1:
                sha256 = d['sha256']
                no_scanners = len(d['scans'])
                positive_results, negative_results = 0,0
                for result in d['scans'].values():
                    if result['detected']:
                        positive_results += 1
                    else:
                        negative_results += 1
                # if all scanners return benign, label is 0, if at least min_no_positive_scans return true, label is 1
                # else, sample is discarded
                if negative_results == no_scanners:
                    label_dict[sha256] = 0
                elif positive_results >= min_no_positive_scans:
                    label_dict[sha256] = 1
    return label_dict


# returns a list with tuples of (train_names, test_names, valid_names) for each split in the drebin dataset
def get_train_test_valid_names(path_to_split):
    split_names = []
    for root, dir, files in os.walk(path_to_split):
        if 'test_cs' in files and 'validate_cs' in files and 'train_cs' in files:
            with open(os.path.join(root, 'test_cs'), 'r') as test_f, open(os.path.join(root,'validate_cs'), 'r') as val_f, open(os.path.join(root, 'train_cs'), 'r') as train_f:
                train_names = test_f.read().splitlines()
                test_names = train_f.read().splitlines()
                val_names = val_f.read().splitlines()
                split_names.append((train_names, test_names, val_names))
    return split_names
