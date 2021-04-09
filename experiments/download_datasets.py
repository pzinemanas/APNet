import os
import argparse
import sys
import numpy as np

sys.path.append('../')

# Datasets
from apnet.datasets import MedleySolosDb, GoogleSpeechCommands
from dcase_models.data.datasets import UrbanSound8k


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
        '-dp', '--dataset_path', type=str,
        help='path to load the dataset',
        default='../datasets'
    )

    args = parser.parse_args()

    print(__doc__)

    if args.dataset not in available_datasets:
        raise AttributeError('Dataset not available')

    # Get and init dataset class
    dataset_class = available_datasets[args.dataset]
    dataset_path = os.path.join(args.dataset_path, args.dataset)
    dataset = dataset_class(dataset_path)

    # Download dataset
    dataset.download()
    
    print('Done!')

if __name__ == "__main__":
    main()
