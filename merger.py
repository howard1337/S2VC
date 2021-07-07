import os
import sys
import json

"""Merge the metadata.json of different features"""

dataset_dir = sys.argv[1]
sub_dirs = [i for i in os.listdir(sys.argv[1]) if 'json' not in i]



metas = []
merged = {}

for sub_dir in sub_dirs:
    metas.append(json.load(open(os.path.join(dataset_dir, sub_dir, 'metadata.json'))))

for key in metas[0].keys():
    if key == 'feature_name':
        continue
    merged[key] = [{} for i in range(len(metas[0][key]))]
    for subdir, meta in zip(sub_dirs, metas):
        for idx, value in enumerate(meta[key]):
            merged[key][idx]['audio_path'] = value['audio_path']
            merged[key][idx][meta['feature_name']] = os.path.join(subdir, value['feature_path'])

json.dump(merged, open(os.path.join(dataset_dir, 'metadata.json'), 'w'), indent=2)
