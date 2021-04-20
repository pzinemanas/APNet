import os
import numpy as np
import csv
from dcase_models.util.files import list_wav_files
from dcase_models.data.dataset_base import Dataset
import mirdata.medley_solos_db
from sed_eval.util.event_roll import event_list_to_event_roll
from librosa.util import fix_length


class GoogleSpeechCommands(Dataset):
    """ Google Speech Commands dataset.

    This class inherits all functionality from Dataset and
    defines specific attributs and methods for Google Speech Commands.

    Url: https://www.tensorflow.org/datasets/catalog/speech_commands

    Pete Warden
    “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition,”
    http://arxiv.org/abs/1804.03209
    August 2018

    Parameters
    ----------
    dataset_path : str
        Path to the dataset fold. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/UrbanSound8k).

    Examples
    --------
    To work with GoogleSpeechCommands dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import GoogleSpeechCommands
    >>> dataset = GoogleSpeechCommands('../datasets/GoogleSpeechCommands')


    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = self.dataset_path
        
        self.fold_list = ["train", "validate", "test"]
        self.label_list = ["backward", "bed", "bird",
                           "cat", "dog", "down", "eight",
                           "five", "follow", "forward",
                           "four", "go", "happy",
                           "house", "learn", "left", "marvin",
                           "nine", "no", "off",
                           "on", "one", "right",
                           "seven", "sheila", "six", "stop",
                           "three", "tree", "two",
                           "up", "visual", "wow",
                           "yes", "zero"]

        self.validation_file = os.path.join(self.dataset_path, 'validation_list.txt')
        self.test_file = os.path.join(self.dataset_path, 'testing_list.txt')

    def generate_file_lists(self):
        validation_list = []
        with open(self.validation_file ) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')        
            for row in csv_reader:
                validation_list.append(row[0])
                
        test_list = []           
        with open(self.test_file ) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')        
            for row in csv_reader:
                test_list.append(row[0])
                
        self.file_lists = {'train': [], 'validate': [], 'test': []}
        all_files = list_wav_files(self.audio_path) 

        for wav_file in all_files:
            path_split = wav_file.split('/')
            base_name = os.path.join(path_split[-2], path_split[-1])
            if base_name in validation_list:
                self.file_lists['validate'].append(wav_file)
                continue
            elif base_name in test_list:
                self.file_lists['test'].append(wav_file)
                continue
            else:
                self.file_lists['train'].append(wav_file)
        #print(len(validation_list), len(test_list), len(all_files))
        #print(all_files[0].endswith('forward/ea0cf37f_nohash_1.wav'))
        #for fold in self.fold_list:
        #    audio_folder = os.path.join(self.audio_path, fold)
        #    self.file_lists[fold] = list_wav_files(audio_folder)

    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        word = file_name.split('/')[-2]
        class_ix = int(self.label_list.index(word))
        y[:, class_ix] = 1
        return y

    def download(self, force_download=False):
        pass



class MedleySolosDb(Dataset):
    """ MedleySolosDb dataset.

    This class inherits all functionality from Dataset and
    defines specific attributs and methods for MedleySolosDb.

    Url: https://zenodo.org/record/1344103

    Vincent Lostanlen and Carmine-Emanuele Cella
    “Deep convolutional networks on the pitch spiral for musical instrument recognition,”
    17th International Society for Music Information Retrieval Conference (ISMIR)
    New York, USA, 2016

    Parameters
    ----------
    dataset_path : str
        Path to the dataset fold. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/UrbanSound8k).

    Examples
    --------
    To work with UrbanSound8k dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import MedleySolosDb
    >>> dataset = UrbanSound8k('../datasets/MedleySolosDb')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        
        self.fold_list = ["train", "validate", "test"] 

        self.label_list = ["clarinet", "distorted electric guitar", "female singer",
                           "flute", "piano", "tenor saxophone", "trumpet",
                           "violin"]
        self.data = mirdata.medley_solos_db.load(data_home=self.dataset_path)

        self.file_lists = {}

    def generate_file_lists(self):
        if len(self.file_lists) > 0:
            return True

        fold_tran = {
            "training": "train", "validation": "validate", "test": "test"}
        self.file_lists = {'train': [], 'validate': [], 'test': []}
        self.file_to_class = {}
        
        for track_id, track_data in self.data.items():  
            self.file_lists[fold_tran[track_data.subset]].append(track_data.audio_path)
            #print(track_data)
            self.file_to_class[track_data.audio_path] = track_data.instrument_id

    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        #word = file_name.split('/')[-2]
        #class_ix = int(self.label_list.index(word))
        class_ix = self.file_to_class[file_name]
        #print(class_ix)
        y[:, class_ix] = 1
        return y

    def download(self, force_download=False):
        mirdata.medley_solos_db.download(self.dataset_path)
        if mirdata.medley_solos_db.validate():
            self.set_as_downloaded()

    def upsample_train_set(self):
        n_instances = np.zeros(len(self.label_list), dtype=int)
        files_by_class = {x: [] for x in range(len(self.label_list))}
        for track_id, track_data in self.data.items():
          #  print(track_data.subset)
            if track_data.subset != 'training':
                continue
            n_instances[track_data.instrument_id] += 1
            files_by_class[track_data.instrument_id].append(track_data.audio_path)
        print(n_instances)

        max_instances = np.amax(n_instances)

        print(len(self.file_lists['train']))
        for class_ix in range(len(self.label_list)):
            if n_instances[class_ix] < max_instances:
                new_instances = max_instances - n_instances[class_ix]
                repetitions = int(len(files_by_class[class_ix])/new_instances)
                for j in range(repetitions):
                    self.file_lists['train'].extend(files_by_class[class_ix])
        print(len(self.file_lists['train']))


class DCASE2021Task5(Dataset):
    """ DCASE2021Task5 dataset.

    This class inherits all functionality from Dataset and
    defines specific attributs and methods for DCASE2021Task5.

    Url: https://zenodo.org/record/4543504#.YG8ZqBKxVH4

    Parameters
    ----------
    dataset_path : str
        Path to the dataset fold. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/DCASE2021Task5).

    Examples
    --------
    To work with DCASE2021Task5 dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import DCASE2021Task5
    >>> dataset = DCASE2021Task5('../datasets/DCASE2021Task5')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'Development_Set_Audio')

        self.fold_list = ["train", "validate"]
        self.label_list = [
            "AMRE","BBWA","BTBW","COYE","OVEN","RBGR","SWTH",
            "GCTH","CHSP","SAVS","WTSP","GRN",
            "GIG","SQT","CALL","SNMK","CCMK","AGGM","SOCM"
        ]

    def generate_file_lists(self):
        for fold in self.fold_list:
            if fold == 'train':
                folder = "Training_Set"
            else:
                folder = "Validation_Set"
            audio_folder = os.path.join(self.audio_path, folder)
            self.file_lists[fold] = list_wav_files(audio_folder)

    def get_annotations(self, file_name, features, time_resolution):
        events = []
        #./../datasets/DCASE2021Task5/Development_Set_Audio/Training_Set/BV/2015-10-14_23-59-59_unit05.wav
        csv_filename = file_name.replace("Development_Set_Audio", "Development_Set_Annotations").replace('wav', 'csv')
        with open(csv_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    header = row
                    line_count += 1
                    continue
                event_onset = float(row[1])
                event_offset = float(row[2])
                for j in range(3, len(row)):
                    class_name = header[j]
                    if row[j] == 'POS':
                        events.append({'event_label': class_name, 'event_onset': event_onset, 'event_offset': event_offset})
                if (len(events) == 5) & (csv_filename.split('/')[-3]=='Validation_Set'):
                    print("Load only first 5 events")
                    break
        label_list = self.label_list if csv_filename.split('/')[-3]=='Training_Set' else ['Q']
        event_roll = event_list_to_event_roll(
            events,
            label_list,
            time_resolution
        )
        if event_roll.shape[0] > features.shape[0]:
            event_roll = event_roll[:len(features)]
        else:
            event_roll = fix_length(event_roll, features.shape[0], axis=0)
        assert event_roll.shape[0] == features.shape[0] 
        return event_roll

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/4543504/files"
        zenodo_files = [
            "Development_Set_Annotations.zip",
            "Development_Set_Audio.zip"]
        downloaded = super().download(
            zenodo_url, zenodo_files, force_download
        )
        if downloaded:
            self.set_as_downloaded()