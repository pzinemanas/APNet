import os
import argparse
import sys
import numpy as np

sys.path.append('../')

# Datasets
from apnet.datasets import MedleySolosDb, GoogleSpeechCommands
from dcase_models.data.datasets import UrbanSound8k

# Models
from dcase_models.model.models import SB_CNN, MLP
from apnet.model import APNet, AttRNNSpeechModel

# Features
from dcase_models.data.features import MelSpectrogram, Openl3

from apnet.layers import PrototypeLayer, WeightedSum
from dcase_models.data.data_generator import DataGenerator
from dcase_models.util.files import load_json, load_pickle, save_pickle


available_models = {
    'APNet' :  APNet,
    'SB_CNN' : SB_CNN,
    'MLP' : MLP,
    'AttRNNSpeechModel' : AttRNNSpeechModel
}

available_features = {
    'MelSpectrogram' :  MelSpectrogram,
    'Openl3' : Openl3
}

available_datasets = {
    'UrbanSound8k' :  UrbanSound8k,
    'MedleySolosDb' : MedleySolosDb,
    'GoogleSpeechCommands' : GoogleSpeechCommands
}

def main():
    # Parse arguments
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
        '-m', '--model', type=str,
        help='model name (e.g. APNet, MLP, SB_CNN ...)')
    
    parser.add_argument('-fold', '--fold_name', type=str, help='fold name')

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
    parser.add_argument('-gpu', '--gpu_visible', type=str, help='gpu_visible',
                        default='0')

    args = parser.parse_args()

    # only use one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible
    print(__doc__)

    if args.dataset not in available_datasets:
        raise AttributeError('Dataset not available')

    if args.features not in available_features:
        raise AttributeError('Features not available')

    model_name = args.model
    if args.model not in available_models:
        base_model = args.model.split('/')[0]
        if base_model not in available_models:
            raise AttributeError('Model not available')
        else:
            model_name = base_model


    # Model paths
    model_folder = os.path.join(args.models_path, args.dataset, args.model)

    # Get parameters
    parameters_file = os.path.join(model_folder, 'config.json')
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


    if args.fold_name is not None:
        fold_list = [args.fold_name]
    else:
        if params_dataset['evaluation_mode'] == 'cross-validation':
            fold_list = dataset.fold_list
        else:
            fold_list = ['test']        

    accuracy = np.zeros(len(fold_list))

    for fold_ix, fold in enumerate(fold_list):
    
        exp_folder = os.path.join(model_folder, fold)

        # Load scaler
        scaler_file = os.path.join(exp_folder, 'scaler.pickle')
        scaler = load_pickle(scaler_file)

        # Init data generator
        data_gen_test = DataGenerator(
            dataset, features, folds=[fold],
            batch_size=params['train']['batch_size'],
            shuffle=False, train=False, scaler=scaler
        )

        # Load model and best weights
        model_class = available_models[model_name]
        metrics = ['classification']

        kwargs = {}
        if (args.model == 'APNet') | (args.model.split('/')[0] == 'APNet'):
            kwargs = {
                'custom_objects': {
                    'PrototypeLayer': PrototypeLayer,
                    'WeightedSum': WeightedSum
                }
            }
        model_container = model_class(
            model=None, model_path=exp_folder, metrics=metrics,
            **kwargs
        )

        model_container.load_model_weights(exp_folder)

        results = model_container.evaluate(
            data_gen_test, label_list=dataset.label_list
        )

        if len(fold_list) == 1:
            print(results[metrics[0]])

        results_path = os.path.join(exp_folder, 'results.pickle')
        save_pickle(results, results_path)

        results = results[metrics[0]].results()
        accuracy[fold_ix] = results['overall']['accuracy']

        print(accuracy[fold_ix])

    print('mean', np.mean(accuracy))

if __name__ == "__main__":
    main()
