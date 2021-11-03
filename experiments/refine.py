import sys
import os
import numpy as np
import json
import glob
import argparse

from dcase_models.data.datasets import UrbanSound8k
from dcase_models.data.features import MelSpectrogram

from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.scaler import Scaler
from dcase_models.util.files import load_json, mkdir_if_not_exists, save_pickle, load_pickle
from dcase_models.util.data import evaluation_setup
from dcase_models.util.ui import progressbar 

sys.path.append('../')
from apnet.model import APNet
from apnet.layers import PrototypeLayer, WeightedSum
from apnet.datasets import MedleySolosDb, GoogleSpeechCommands

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

available_models = {
    'APNet' :  APNet,
}

available_features = {
    'MelSpectrogram' :  MelSpectrogram,
}

available_datasets = {
    'UrbanSound8k' :  UrbanSound8k,
    'MedleySolosDb' : MedleySolosDb,
    'GoogleSpeechCommands' : GoogleSpeechCommands
}

#def main():
parser = argparse.ArgumentParser(description='Refine APNet model')

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    '-d', '--dataset', type=str,
    help='dataset name (e.g. UrbanSound8k, ESC50, URBAN_SED, SONYC_UST)',
    default='UrbanSound8k'
)
parser.add_argument(
    '-f', '--features', type=str,
    help='features name (e.g. Spectrogram, MelSpectrogram, Openl3)',
    default='MelSpectrogram'
)
parser.add_argument(
    '-r', '--refinement_type', type=str,
    help='Refinement mode (prototypes or channels)',
)

parser.add_argument(
    '-mp', '--models_path', type=str,
    help='path to load the trained model',
    default='./'
)
parser.add_argument(
    '-dp', '--dataset_path', type=str,
    help='path to load the dataset',
    default='./'
)

parser.add_argument(
    '-m', '--model', type=str,
    help='model name (e.g. MLP, SB_CNN, SB_CNN_SED, A_CRNN, VGGish)')

parser.add_argument('-fold', '--fold_name', type=str, help='fold name',
                    default='fold1')


args = parser.parse_args()

if args.dataset not in available_datasets:
    raise AttributeError('Dataset not available')

if args.features not in available_features:
    raise AttributeError('Features not available')

model_name = args.model
if model_name not in available_models:
    model_name = args.model.split('/')[0]
    if model_name not in available_models:
        raise AttributeError('Model not available')

# Model paths
model_input_folder = os.path.join(args.models_path, args.dataset, args.model)
model_output_folder = os.path.join(model_input_folder, 'refine_' + args.refinement_type)

# Get parameters
parameters_file = os.path.join(model_input_folder, 'config.json')
params = load_json(parameters_file)

params_features = params['features'][args.features]
params_dataset = params['datasets'][args.dataset]
params_model = params['models'][model_name]

# Get and init dataset class
dataset_class = available_datasets[args.dataset]
dataset_path = os.path.join(args.dataset_path, params_dataset['dataset_path'])
dataset = dataset_class(dataset_path)

# Get and init feature class
features_class = available_features[args.features]
features = features_class(**params_features)

print('Features shape: ', features.get_shape())

# Check if features were extracted
if not features.check_if_extracted(dataset):
    print('Extracting features ...')
    features.extract(dataset)
    print('Done!')

n_folds = len(dataset.fold_list)

if args.fold_name is not None:
    fold_list = [args.fold_name]
else:
    if params_dataset['evaluation_mode'] == 'cross-validation':
        fold_list = dataset.fold_list
    else:
        fold_list = ['test'] 

model_containers = {}

for fold, fold_name in enumerate(fold_list):#enumerate(fold_list):
    print(fold_name)
    exp_folder_fold = os.path.join(model_input_folder, fold_name)
    exp_folder_output_fold = os.path.join(model_output_folder, fold_name)
    mkdir_if_not_exists(exp_folder_output_fold, parents=True)

    kwargs = {'custom_objects': {'PrototypeLayer': PrototypeLayer, 'WeightedSum': WeightedSum}}
    model_containers[fold_name] = APNet(
        model=None, model_path=exp_folder_fold, metrics=['classification'],
        **kwargs, **params['models']['APNet']['model_arguments']
    )

    model_containers[fold_name].load_model_weights(exp_folder_fold)

    model_containers[fold_name].model.summary()

    scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle') 
    scaler = load_pickle(scaler_path)

    data_instances_path = os.path.join(exp_folder_fold, 'data_instances.pickle')
    prototypes_path = os.path.join(exp_folder_fold, 'prototypes.pickle')

    folds_train, folds_val, _ = evaluation_setup(
        args.fold_name, dataset.fold_list,
        params_dataset['evaluation_mode'],
        use_validate_set=True
    )

    data_gen_train = DataGenerator(
        dataset, features, folds=folds_train,
        batch_size=params['train']['batch_size'],
        shuffle=True, train=True, scaler=scaler
    )

    if args.dataset == 'GoogleSpeechCommands':
        N_train = len(data_gen_train.audio_file_list)
        print('len_data_train', N_train)
        data_gen_train.audio_file_list = data_gen_train.audio_file_list[:int(N_train/10)]

    X_train, _ = data_gen_train.get_data()
    print(X_train.shape)

    if args.refinement_type == "prototypes":
        model_containers[fold_name].refine_prototypes(X_train)
        # save_pickle(model_containers[fold_name].prototypes, prototypes_path)
    elif args.refinement_type == "channels":
        distance = params['models']['APNet']['model_arguments']['distance']
        model_containers[fold_name].refine_channels(
            X_train, distance=distance)
        # save_pickle(model_containers[fold_name].prototypes, prototypes_path)
    else:
        raise AttributeError('Incorrect refinement mode : %s' % args.refinement_type)

    model_containers[fold_name].model.summary()
    model_containers[fold_name].save_model_json(exp_folder_output_fold)
    model_containers[fold_name].save_model_weights(exp_folder_output_fold)

    scaler_path = os.path.join(exp_folder_output_fold, 'scaler.pickle') 
    save_pickle(scaler, scaler_path)


# Save new params
params_path = os.path.join(model_output_folder, 'config.json') 

params['models']['APNet']['train_arguments']['init_last_layer'] = 0
params['models']['APNet']['model_arguments']['n_prototypes'] = -1

with open(params_path, 'a') as outfile:
    json.dump(params, outfile, indent=2)
