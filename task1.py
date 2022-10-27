import argparse
from operator import itemgetter
from pathlib import Path

import numpy as np
from natsort import natsorted
from rich import console, print as rprint
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import track
from slip_crumple_classification import DatasetLoader, Predict,Param


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)


    args = parser.parse_args()

    data_loader = DatasetLoader()

    data_loader.data_set.train_path = Path(args.input_data)
    data_loader.dataset.files = natsorted(list(data_loader.data_set.train_path.rglob("*.csv")))


    ############################
    num_feature = 3
    feature_set_available = ['force','position','mass','joint_wise_fp','force_derivative','joint_derivative'] # force_dwt_freq_tf
    
    if type(num_feature)==list:
        feature_set =  list(itemgetter(*num_feature)(feature_set_available))
    else:
        feature_set =  [itemgetter(num_feature)(feature_set_available)]
        
    param = Param(observe_window=10,batch_size=32,epochs=10,feature_set = feature_set)
    ############################

    test_X = data_loader.read_raw_data(load_first_n_files=len(data_loader.dataset.files),
                                        feature_set=param.feature_set,
                                        n_timestep=param.observe_window,
                                        loader='test')
    
    rprint(Panel(f'test_X : (n_event_files,n_samples,n_observe_steps,n_feature) \n {test_X.shape}',title="Time Event Batch"))

    test_X = np.vstack(test_X)

    rprint(Panel(f'test_X : (n_event_files,n_samples,n_observe_steps,n_feature) \n {test_X.shape}',title="Time Event Batch Stacked"))


    
    pred = Predict()
    
    pred.model_path = Path(args.model_path)
    Path(args.output).mkdir(parents=True,exist_ok=True)

    pred.model_param.n_batch_size = param.batch_size
    pred.model_param.one_hot_classes = data_loader.one_hot_classes
    pred.model_param.n_event_rows = data_loader.n_event_rows
    pred.model_param.n_event_file = len(data_loader.dataset.files)

    f_set = '_'.join(param.feature_set)
    f_path = pred.model_path / f_set / str(test_X.shape[1])


    outputs_df = pred.run(test_X,f_path=f_path)
    console = Console()
    with console.status("[blue] Saving Predictions ") as status: 
        for _,(file,output) in enumerate(zip(data_loader.dataset.files,outputs_df)):#,total=len(data_loader.dataset.files),description="Saving Predictions....",):
            pred.pred_output_path = Path(args.output) / f'{file.stem}.csv'
            
            output.to_csv(pred.pred_output_path.as_posix(),index=False)

            status.update(f"[green]{pred.pred_output_path}")


    
    print()

if __name__=="__main__":
    main()