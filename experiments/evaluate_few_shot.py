import os
import argparse
import sys
import numpy as np

sys.path.append('../')

# Datasets
from apnet.datasets import MedleySolosDb, GoogleSpeechCommands, DCASE2021Task5
from dcase_models.data.datasets import UrbanSound8k

# Models
from dcase_models.model.models import SB_CNN, MLP
from apnet.model import APNet, AttRNNSpeechModel, APNetFewShot

# Features
from dcase_models.data.features import MelSpectrogram, Openl3

from apnet.layers import PrototypeLayer, WeightedSum
from dcase_models.data.data_generator import DataGenerator
from dcase_models.util.files import load_json, load_pickle, save_pickle

from dcase_models.util.events import event_roll_to_event_list

import csv

available_models = {
    'APNetFewShot' : APNetFewShot,
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
    'GoogleSpeechCommands' : GoogleSpeechCommands,
    'DCASE2021Task5': DCASE2021Task5
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
        fold_list = ['validate']   
                 
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
        metrics = ['sed']

        kwargs = {}
        if (args.model == 'APNet') | (args.model.split('/')[0] == 'APNet') | (args.model.split('/')[0] == 'APNetFewShot'):
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

        X_test, Y_test = data_gen_test.get_data()

        for j in range(len(X_test)):
            print(X_test[j].shape, Y_test[j].shape)
            X_test[j] = X_test[j][Y_test[j][:,0]==1]
            Y_test[j] = Y_test[j][Y_test[j][:,0]==1]
            print(X_test[j].shape, Y_test[j].shape)

        model_distances = model_container.model_input_to_distances()

        X_test_all, _ = data_gen_test.get_data()

        with open('results.csv', 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow(['Audiofilename','Starttime','Endtime'])
            for j in range(len(X_test)):

                bs = len(X_test[j])
                predictions = np.zeros((len(X_test_all[j]),1))
                n_batches = int(np.ceil(len(X_test_all[j])/bs))
                for batch in range(n_batches):
                    Xq = X_test_all[j][batch*bs:(batch+1)*bs]
                    Xfs = X_test[j][:len(Xq)]
                    Xfs_neg = X_test_all[j][:len(Xq)]
                   # print(Xq.shape, Xfs.shape, X_test[j].shape)
                    pred = model_container.model.predict([Xfs, Xfs, Xfs_neg, Xq])[-1]
                    predictions[batch*bs:(batch+1)*bs, 0] = np.argmax(pred, axis=1)
                    #print(pred)

                #predictions = predictions > 0 #np.mean(predictions)
                #predictions = predictions.astype(int)
                print(np.sum(predictions))
                # preds = model_container.model.predict(X_test[j])[0]
                # preds = np.mean(preds, axis=0)
                # class_pred = np.argmax(preds)
                # print(class_pred)

                # predictions_all = model_container.model.predict(X_test_all[j])[0]
                # print(predictions_all.shape)
                # predictions_all = np.argmax(predictions_all, axis=1)
                # print(predictions_all.shape)

                # print(predictions_all)
                # predictions_all = predictions_all == class_pred
                # predictions = predictions_all.astype(int)
                # predictions = np.expand_dims(predictions, axis=1)

                # distances = model_distances.predict(X_test[j])
                # distance_mean = np.mean(distances, axis=0)
                # argsort = np.argsort(distance_mean, axis=0)
                # print(argsort[-3:])
                # print(distances.shape)
                # #print(np.argmax(distance_mean))
                # #print('argsort', argsort.shape)

                # distances_all = model_distances.predict(X_test_all[j])
                # distances_all_sort = np.argsort(distances_all, axis=1)
                # #print(distances_all_sort.shape)
                # predictions = np.zeros((len(distances_all_sort), 1))
                # for k in range(len(predictions)):
                #     #closest_prototypes = distances_all_sort[j, -3:]
                #     #if (closest_prototypes[0] in argsort[-3:]) | (closest_prototypes[1] in argsort[-3:]) | (closest_prototypes[2] in argsort[-3:]):
                #     closest_prototype = distances_all_sort[j, -1]
                #     if closest_prototype == argsort[-1]:
                #         predictions[k] = 1
                #         print('ahora')
                #     #print(closest_prototypes, argsort[-3:])
                # predictions = np.argmax(distances_all, axis=1) == np.argmax(distance_mean)

                # predictions = np.expand_dims(predictions, axis=1).astype(int)
                # print(predictions.shape)

                event_list = event_roll_to_event_list(predictions, ['Q'], features.sequence_hop_time)
                #print(event_list)
                filename = os.path.basename(data_gen_test.audio_file_list[j]['file_original'])
                print(filename)
                for event in event_list:
                    spamwriter.writerow( [filename, str(event['event_onset']), str(event['event_offset'])] )


        results = model_container.evaluate(
            data_gen_test, label_list=['Q']
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
