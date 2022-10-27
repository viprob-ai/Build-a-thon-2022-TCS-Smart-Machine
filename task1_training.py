import argparse
from operator import itemgetter
from pathlib import Path
import warnings

import numpy as np
from natsort import natsorted
from prettytable import PrettyTable
from rich import print as rprint
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import track
from slip_crumple_classification import DatasetLoader, Train,Param


warnings.filterwarnings('ignore')


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_data', type=str, required=True)
    # parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    data_loader = DatasetLoader()

    data_loader.data_set.train_path = Path(args.input_data)
    data_loader.dataset.files = np.random.choice(natsorted(list(data_loader.data_set.train_path.rglob("*.csv"))),1870,replace=False)
    # data_loader.dataset.files = natsorted(list(data_loader.data_set.train_path.rglob("*.csv")))



    ############################
    # num_feature = 4
    feature_set_available = ['force','position','mass','joint_wise_fp','force_derivative','joint_derivative'] # force_dwt_freq_tf

    Festures_Set = PrettyTable()
    
    Festures_Set.add_column('Features',feature_set_available)
    Festures_Set.add_column('Index',list(range(len(feature_set_available))))
    
    print(Festures_Set.get_string())

    try:
        num_feature  = int(input('Feature to run  : ' ))
    except:
        num_feature  = 3
    
    if type(num_feature)==list:
        feature_set =  list(itemgetter(*num_feature)(feature_set_available))
    else:
        feature_set =  [itemgetter(num_feature)(feature_set_available)]
    
    try:
        observe_window  = int(input('n_timestep to observe to predict class at nth step  : ' ))
    except:
        observe_window  = 10
    
    try:
        batch_size  = int(input('Batch Size  : ' ))
    except:
        batch_size  = 32

    
    try:
        epochs  = int(input('Epochs  : ' ))
    except:
        epochs  = 10

    
    try:
        pretrain  = int(input('PreTrain  : ' ))
    except:
        pretrain  = False

    
    


    param = Param(observe_window=observe_window,batch_size=batch_size,epochs=epochs,feature_set = feature_set)
    ############################

    train_X,train_y = data_loader.read_raw_data(load_first_n_files=len(data_loader.dataset.files),
                                                feature_set=param.feature_set,
                                                n_timestep=param.observe_window,
                                                loader='train')

    rprint(Panel(f'train_X : (n_event_files,n_samples,n_observe_steps,n_feature) \n {train_X.shape}'
                +"\n"+ f'train_y : (n_event_files,n_samples,1) \n {train_y.shape}'
                ,title="Time Event Batch")
                )

    train_X,train_y = np.vstack(train_X),np.vstack(train_y)

    rprint(Panel(f'train_X : (n_event_files,n_samples,n_observe_steps,n_feature) \n {train_X.shape}'
                +"\n"+ f'train_y : (n_event_files,n_samples,1) \n {train_y.shape}'
                ,title="Time Event Batch Stacked")
                )

    
    # from collections import Counter
    # class_counts = Counter(train_y.flatten())

    # class_0 = np.random.choice(np.arange(train_X.shape[0])[(train_y==0).flatten()],int(class_counts.get(0)*(0.35)),replace=False)
    # class_1 = np.random.choice(np.arange(train_X.shape[0])[(train_y==1).flatten()],int(class_counts.get(1)*(0.65)),replace=False)
    # class_2 = np.random.choice(np.arange(train_X.shape[0])[(train_y==2).flatten()],int(class_counts.get(2)*(4.1)),)
    # class_3 = np.random.choice(np.arange(train_X.shape[0])[(train_y==3).flatten()],int(class_counts.get(3)*(2.1)),)
    # balanced_class_index = np.hstack([class_0,class_1,class_2,class_3])

    # train_X = train_X[balanced_class_index]
    # train_y = train_y[balanced_class_index]
    # class_counts = Counter(train_y.flatten())

    
    try:
        split_dataset  = int(input('Train/Test Split 0-1  : ' ))
    except:
        split_dataset  = 0.8

    split_at = int(train_X.shape[0]*split_dataset) ##

    
    train_X,train_y,test_X,test_y = train_X[:split_at],train_y[:split_at],train_X[split_at:],train_y[split_at:]
    
    rprint(Panel(f'train_X : (n_event_files,n_samples,n_observe_steps,n_feature) \n {train_X.shape}'
    +"\n"+ f'train_y : (n_event_files,n_samples,1) \n {train_y.shape}'
    +"\n\n"+f'test_X : (n_event_files,n_samples,n_observe_steps,n_feature) \n {test_X.shape}'
    +"\n"+f'test_y : (n_event_files,n_samples,1) \n {test_y.shape}'
    ,title="Train / Test Split")
    )

    # np.random.shuffle(train_X)
    # np.random.shuffle(train_y)


    train = Train()
    train.model_path = Path(args.model_path)
    train.model_param.n_event_rows = data_loader.n_event_rows

    train.model_param.one_hot_classes = data_loader.one_hot_classes
    train.model_param.n_batch_size = param.batch_size
    train.model_param.feature_set = param.feature_set
    train.model_param.epochs = param.epochs
    train.pretrain = pretrain


    if pretrain:
        f_set = '_'.join(param.feature_set)
        train.f_path = train.model_path / f_set / str(test_X.shape[1])
        train.restore_model(train.f_path)

    train.run(train_X,train_y,test_X,test_y)


    print()


if __name__=="__main__":
    main()