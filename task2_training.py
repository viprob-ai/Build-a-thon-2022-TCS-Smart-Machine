
import argparse
from copy import deepcopy
from operator import itemgetter
from pathlib import Path

import numpy as np
from natsort import natsorted
from prettytable import PrettyTable
from rich import print as rprint
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import track
from object_held_classification import DatasetLoader, Train


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_data', type=str, required=True)
    # parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    data_loader = DatasetLoader()
    data_loader.data_set.train_path = Path(args.input_data)
    data_loader.dataset.files = [data_loader.data_set.train_path] #natsorted(list(data_loader.data_set.train_path.rglob("*.csv")))

    SAMPLE_X_PERC_DATA_BASED_ON = 'category'
    

    feature_set_available =  ['force','position','mass','forceXposition'] # force_dwt_freq_tf


    Festures_Set = PrettyTable()
    
    Festures_Set.add_column('Features',feature_set_available)
    Festures_Set.add_column('Index',list(range(len(feature_set_available))))
    
    print(Festures_Set.get_string())

    try:
        num_feature  = int(input('Feature to run  : ' ))
    except:
        num_feature  = [0,1,2]

    try:
        pretrain  = int(input('PreTrain  : ' ))
    except:
        pretrain  = False

    

    if type(num_feature)==list:
        feature_set =  list(itemgetter(*num_feature)(feature_set_available))
    else:
        feature_set =  [itemgetter(num_feature)(feature_set_available)]
    
    

    data_dict,object_held_data = data_loader.read_raw_data(SAMPLE_X_PERC_DATA=80, feature_set=feature_set,loader='train')

    if SAMPLE_X_PERC_DATA_BASED_ON == 'class':

        train_X,train_y_class,train_y_category = data_dict["class"][:3]
        test_X,test_y_class,test_y_category = data_dict["class"][3:]

        train_X,train_y = np.vstack(train_X),np.vstack(train_y_class)
        test_X,test_y = np.vstack(test_X),np.vstack(test_y_class)

        n_label = object_held_data.n_classes

    if SAMPLE_X_PERC_DATA_BASED_ON == 'category':
        train_X,train_y_class,train_y_category = data_dict["category"][:3]
        test_X,test_y_class,test_y_category = data_dict["category"][3:]

        train_X,train_y = np.vstack(train_X),np.vstack(train_y_class)
        test_X,test_y = np.vstack(test_X),np.vstack(test_y_class)

        n_label = object_held_data.n_classes

    

    

    rprint(Panel(f'train_X : (n_samples,n_feature) \n {train_X.shape}'+'\n'+f'train_y : (n_samples,1) \n {train_y.shape}' + "\n\n"+
                f'test_X : (n_samples,n_feature) \n {test_X.shape}'+'\n'+f'test_y : (n_samples,1) \n {test_y.shape}'
                ,title="Min Object Class Balanced Data")
                )


    
   
    train = Train()
    train.model_path = Path(args.model_path)
    (train.model_path).mkdir(parents=True,exist_ok=True)
    # train.model_param.one_hot_classes = n_label
    train.model_param.one_hot_classes = data_loader.one_hot_classes
    train.model_param.feature_set = feature_set
    train.pretrain=pretrain

    if pretrain:
        f_set = '_'.join(train.model_param.feature_set)
        train.f_path = train.model_path / f_set 
        train.restore_model(train.f_path)


    train.run(train_X,train_y,test_X,test_y)


if __name__=="__main__":
    main()