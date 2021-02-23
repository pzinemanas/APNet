import sys
import os
import numpy as np
import json
import argparse

sys.path.append('../../DCASE-models')
from dcase_models.data.feature_extractor import FeatureExtractor
from dcase_models.util.files import load_json

parser = argparse.ArgumentParser(description='Extract features')
parser.add_argument('dataset', type=str, help='dataset to extract the features')
parser.add_argument('-a','--augmentation', type=str, help='data augmentation type')
parser.add_argument('-p','--parameter', type=float, help='parameter of data augmentation')
args = parser.parse_args()

params = load_json('parameters.json')
params_dataset = params["datasets"][args.dataset]

if args.augmentation is not None:
    params['features']['augmentation'] = {args.augmentation: args.parameter}

# extract features and save files
feature_extractor = FeatureExtractor(**params['features'])

n_folds = len(params_dataset["folds"])
if args.dataset == 'ESC50':
    feature_extractor.extract(audio_folder, feature_folder)
else:
    for fold in params_dataset["folds"]:
        print(fold)
        audio_folder_fold = os.path.join(audio_folder, fold)
        feature_folder_fold = os.path.join(feature_folder, fold)
        feature_extractor.extract(audio_folder_fold, feature_folder_fold)

feature_extractor.save_mel_basis(os.path.join(feature_folder,'mel_basis.npy'))
feature_extractor.save_parameters_json(os.path.join(feature_folder,'parameters.json'))