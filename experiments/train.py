import os
import argparse
import sys
import numpy as np
from imblearn.over_sampling import RandomOverSampler 
from numpy.random import default_rng

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
from dcase_models.data.scaler import Scaler
from dcase_models.util.files import load_json
from dcase_models.util.files import mkdir_if_not_exists, save_pickle
from dcase_models.util.data import evaluation_setup


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
    'DCASE2021Task5' : DCASE2021Task5
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
        help='model name (e.g. MLP, SB_CNN, SB_CNN_SED, A_CRNN, VGGish)')

    parser.add_argument('-fold', '--fold_name', type=str, help='fold name',
                        default='fold1')

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

    parser.add_argument('--c', dest='continue_training', action='store_true')
    parser.set_defaults(continue_training=False)

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
    if model_name not in available_models:
        model_name = args.model.split('/')[0]
        if model_name not in available_models:
            raise AttributeError('Model not available')

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

    if args.fold_name not in dataset.fold_list:
        raise AttributeError('Fold not available')

    # Get and init feature class
    features_class = available_features[args.features]
    features = features_class(**params_features)

    print('Features shape: ', features.get_shape(4.0))

    if not features.check_if_extracted(dataset):
        print('Extracting features ...')
        features.extract(dataset)
        print('Done!')


    folds_train, folds_val, folds_test = evaluation_setup(
        args.fold_name, dataset.fold_list,
        params_dataset['evaluation_mode'],
        use_validate_set=True
    )

    if (model_name == 'APNet') | (model_name == 'APNetFewShot'):
        outputs = ['annotations', features, 'zeros']
    else:
        outputs = 'annotations'
        
    data_gen_train = DataGenerator(
        dataset, features, folds=folds_train,#+['validate'],
        batch_size=params['train']['batch_size'],
        buffer_size=params['train']['buffer_size'],
        shuffle=True, train=True, scaler=None,
        outputs=outputs
    )

    # if args.dataset == 'GoogleSpeechCommands':
    #     N_train = len(data_gen_train.audio_file_list)
    #     print('len_data_train', N_train)
    #     data_gen_train.audio_file_list = data_gen_train.audio_file_list[:int(N_train/10)]

    # # Upsample the train set
    # n_instances = np.zeros(len(dataset.label_list), dtype=int)
    # files_by_class = {x: [] for x in range(len(dataset.label_list))}
    # for j in range(len(data_gen_train.audio_file_list)):
    #     file_name = data_gen_train.audio_file_list[j]['file_original']
    #     annotations = dataset.get_annotations(file_name, np.zeros(1,), 0)
    #     #print(annotations)
    #     class_ix = np.argmax(annotations)
    #     n_instances[class_ix] += 1
    #     files_by_class[class_ix].append(data_gen_train.audio_file_list[j])
    # print(n_instances)
    # max_instances = np.amax(n_instances)

    # print(len(data_gen_train.audio_file_list))
    # for class_ix in range(len(dataset.label_list)):
    #     if n_instances[class_ix] < max_instances:
    #         new_instances = max_instances - n_instances[class_ix]
    #         repetitions = int(new_instances/len(files_by_class[class_ix]))
    #         print(new_instances, repetitions)
    #         if repetitions > 1:
    #             for j in range(repetitions):
    #                 data_gen_train.audio_file_list.extend(files_by_class[class_ix])
    #         else:
    #             data_gen_train.audio_file_list.extend(files_by_class[class_ix][:new_instances])
    # print(len(data_gen_train.audio_file_list))


    scaler = Scaler(normalizer=params_model['normalizer'])

    print('Fitting scaler ...')
    if args.model in ['MLP', 'SB_CNN', 'AttRNNSpeechModel']:
        scaler_outputs = None
    else:
        scaler_outputs = Scaler(normalizer=[None, params_model['normalizer'], None])
        scaler_outputs.fit(data_gen_train, inputs=False)
        data_gen_train.set_scaler_outputs(scaler_outputs)

    scaler.fit(data_gen_train)
    data_gen_train.set_scaler(scaler)
    print('Done!')

    # Pass scaler to data_gen_train to be used when data
    # loading

    if args.dataset == 'DCASE2021Task5':
        folds_val = ["train"]

    data_gen_val = DataGenerator(
        dataset, features, folds=folds_val,
        batch_size=params['train']['batch_size'],
        buffer_size=params['train']['buffer_size'],
        shuffle=False, train=False, scaler=scaler
    )

    # Define model
    features_shape = features.get_shape()
    if len(features_shape) > 2:
        n_frames_cnn = features_shape[1]
        n_freq_cnn = features_shape[2]
    else:
        n_freq_cnn = features_shape[1]
    n_classes = len(dataset.label_list)

    model_class = available_models[model_name]

    metrics = ['classification']

    # Set paths
    exp_folder = os.path.join(model_folder, args.fold_name)
    mkdir_if_not_exists(exp_folder, parents=True)

    # Save model json and scaler
    save_pickle(scaler, os.path.join(exp_folder, 'scaler.pickle'))

    if (args.dataset == 'UrbanSound8k') | (args.dataset == 'DCASE2021Task5'):
        # Upload all data to RAM only for this dataset
        print('Getting data...')
        data_gen_train = data_gen_train.get_data()
        print('Done!')

    if args.dataset == 'DCASE2021Task5':
        X_train = data_gen_train[0]
        Y_train = data_gen_train[1][0]
        zeros = data_gen_train[1][2]
        del data_gen_train
        
        print(X_train.shape, zeros.shape, Y_train.shape)
        X_train = X_train[np.sum(Y_train, axis=1) > 0]
        Y_train = Y_train[np.sum(Y_train, axis=1) > 0]
        print(X_train.shape, zeros.shape, Y_train.shape)

        number_of_instances = np.zeros(19)
        for j in range(19):
            number_of_instances[j] = np.sum(Y_train[:, j] == 1)

        print(number_of_instances)
        selected_classes = np.argwhere(number_of_instances > 60)
        n_classes = len(selected_classes)
        print(n_classes)

        Y_train = Y_train[:, selected_classes]
        print(Y_train.shape)

        ros = RandomOverSampler(random_state=42)
        Y_train = np.argmax(Y_train, axis=1)
        X_train_shape = X_train.shape
        X_train = np.reshape(X_train, (len(X_train), -1))
        print(X_train.shape, Y_train.shape)
        X_train, Y_train2 = ros.fit_resample(X_train, Y_train)
        X_train = np.reshape(X_train, (len(X_train), X_train_shape[1], X_train_shape[2]))
        
        #Y_train2 = np.argmax(Y_train, axis=1)

        Nc = 10
        Nway = 5
        bs = params['train']['batch_size']
        n_batches = int(len(X_train)/bs)
        length = int(n_batches*bs)
        X = np.zeros((length, X_train.shape[1], X_train.shape[2]))
        Xfs = np.zeros((length, X_train.shape[1], X_train.shape[2]))
        Xfs_neg = np.zeros((length, X_train.shape[1], X_train.shape[2]))
        XQ = np.zeros((length, X_train.shape[1], X_train.shape[2]))
        outQ = np.zeros((length, 2))
        zeros = np.zeros(length)
        Y_train = np.zeros((length, n_classes))
        samples_per_class = int(bs / Nc)
        rng = default_rng()
        classes = rng.choice(n_classes, size=n_classes, replace=False)
        support_classes = classes[:Nc]
        query_classses = classes[Nc:]
        for j in range(n_batches):
            for ix_c, c in enumerate(support_classes):
                indexes = np.argwhere(Y_train2 == c)[:,0]
                batch_indexes = indexes[rng.choice(len(indexes), size=samples_per_class, replace=False)]
                X[j*bs+ix_c*samples_per_class:j*bs+ix_c*samples_per_class+samples_per_class] = X_train[batch_indexes]
                Y_train[j*bs+ix_c*samples_per_class:j*bs+ix_c*samples_per_class+samples_per_class, c] = 1
            
            #Query dataset
            query_class = query_classses[np.random.randint(len(query_classses))]
            indexes = np.argwhere(Y_train2 == query_class)[:,0]
            if len(indexes) < bs+int(bs/2):
                print(query_class)
            batch_indexes = indexes[rng.choice(len(indexes), size=bs+int(bs/2), replace=False)]
            Xfs[j*bs:(j+1)*bs] = X_train[batch_indexes[:bs]]
            XQ[j*bs:j*bs+int(bs/2)] = X_train[batch_indexes[bs:]]
            outQ[j*bs:j*bs+int(bs/2), 1] = 1
            indexes = np.argwhere(Y_train2 != query_class)[:,0]
            batch_neg_indexes = indexes[rng.choice(len(indexes), size=int(bs/2), replace=False)]
            XQ[j*bs+int(bs/2):(j+1)*bs] = X_train[batch_neg_indexes]
            outQ[j*bs+int(bs/2):(j+1)*bs, 0] = 1

            batch_neg_indexes = indexes[rng.choice(len(indexes), size=bs, replace=False)]
            Xfs_neg[j*bs:(j+1)*bs] = X_train[batch_neg_indexes]

        data_gen_train = ([X, Xfs, Xfs_neg, XQ], [Y_train, X, zeros, outQ])

        # X_val, Y_val = data_gen_val.get_data()
        # X_val_list = []
        # Y_val_list = []
        # for j in range(len(X_val)):
        #     for k in range(len(X_val[j])):
        #         if np.sum(Y_val[j][k]) > 0:
        #             X_val_list.append(np.expand_dims(X_val[j][k], 0))
        #             Y_val_list.append(np.expand_dims(Y_val[j][k], 0))
            #print(X_val[j].shape, Y_val[j].shape)
    
    label_list = [dataset.label_list[int(j)] for j in selected_classes]

    if args.continue_training:
        model_container = model_class(
            model=None, model_path=exp_folder, 
            custom_objects={
                'PrototypeLayer': PrototypeLayer,
                'WeightedSum': WeightedSum
            }
        )
        model_container.load_model_weights(exp_folder)
        params_model['train_arguments']['init_last_layer'] = 0
    else:
        if args.model == 'MLP':
            model_container = model_class(
                model=None, model_path=None, n_classes=n_classes,
                n_frames=None, n_freqs=n_freq_cnn,
                metrics=metrics,
                **params_model['model_arguments']
            )
        else:         
            model_container = model_class(
                model=None, model_path=None, n_classes=n_classes,
                n_frames_cnn=n_frames_cnn, n_freq_cnn=n_freq_cnn,
                metrics=metrics,
                **params_model['model_arguments']
            )
        model_container.save_model_json(exp_folder)

    model_container.model.summary()


    # Train model
    model_container.train(
        data_gen_train, (None, None),# (X_val_list, Y_val_list),
        label_list=label_list,
        weights_path=exp_folder, **params['train'],
        sequence_time_sec=params_features['sequence_hop_time'],
        **params_model['train_arguments']
    )


if __name__ == "__main__":
    main()
