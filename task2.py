
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from prettytable import PrettyTable
from rich import print as rprint
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import track
from object_held_classification import DatasetLoader, Predict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)


    args = parser.parse_args()

    data_loader = DatasetLoader()
    data_loader.data_set.train_path = Path(args.input_data)
    data_loader.dataset.files = [data_loader.data_set.train_path] #natsorted(list(data_loader.data_set.train_path.rglob("*.csv")))


    num_feature  = 3
    feature_set = ['force','position','mass','forceXposition'][:num_feature] #['force','position','mass','forceXposition']


    data_dict = data_loader.read_raw_data( feature_set=feature_set,loader='test')

    test_X = data_dict["test"]


    rprint(Panel(f'test_X : (n_samples,n_feature) \n {test_X.shape}'))
  

    pred = Predict()
    pred.model_path = Path(args.model_path)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)

    pred.pred_output_path = Path(args.output) 
    pred.model_param.one_hot_classes = data_loader.one_hot_classes
    pred.model_param.feature_set = feature_set

    f_set = '_'.join(feature_set)
    f_path = pred.model_path / f_set 

    pred.run(test_X,f_path=f_path)

    print()

if __name__=="__main__":
    main()