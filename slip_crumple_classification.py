

from datetime import datetime
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, NamedTuple

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorboard
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import Sequential
from natsort import natsorted
from prettytable import PrettyTable
from rich import print as rprint
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import track
# from keras.utils import to_categorical
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from config import Dataset

# import tensorflow as tf
# # import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import pyrtools as prt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

import seaborn as sns




class Param(NamedTuple):
    observe_window:int=20 #
    batch_size:int=250 
    epochs:int=10
    feature_set:List=['force','position','mass']



class DatasetLoader(object):

    def __init__(self) -> None:

        self.path = Path(__file__).parents[2] / "data" / "raw" / "PS_1" 


        self.data_set = Dataset(name='slip_detection',
                        data_path = self.path, 
                        train_path = (self.path/ "FORCE_ESTIMATION_TRAIN_SET"),
                        test_path = (self.path/  "FORCE_ESTIMATION_VAL_SET"))

        self.dataset = SimpleNamespace()
        self.dataset.files = natsorted(list(self.data_set.train_path.rglob("*.csv")))
        

        self.data_set_processed = Dataset(name='slip_detection',
                        data_path = (self.path.parents[1] / "processed" / "PS1" ), 
                        train_path = (self.path.parents[1] / "processed" / "PS1" / "train" ),
                        test_path = (self.path.parents[1] / "processed" / "PS1" / "test" ))

       

        self.one_hot_classes = np.array([ [0,0],
                                                [1,0],
                                                [0,1],
                                                [1,1] ])
        


    def pre_process(self,file,feature_set=['force'],n_timestep=None,loader='train') :
        
        read_event = lambda fpath : pd.read_csv(fpath.as_posix())
        pad_event = lambda df_event : pd.concat([pd.concat([df_event.iloc[0].to_frame().T]*(n_timestep)),df_event],ignore_index=True)
        read_tf_events = lambda fpath : pad_event(read_event(fpath))

        from rich.progress import Progress

       
        df = pd.concat(track(map(read_event , file),total=len(file),description='Loading raw files'),ignore_index=True,axis=0)

        
        df_position = df.loc[:,'J0_position':'J15_position':2].to_numpy(dtype=np.float32)
        df_force = df.loc[:,'J0_Force':'J15_Force':2].to_numpy(dtype=np.float32)
        df_mass = df.loc[:,'Mass'].to_numpy(dtype=np.float32)[:,np.newaxis]
        df_size = df.loc[:,'Size']

        if loader=='train':

            df_label = df.loc[:,['Slip','Crumple']].to_numpy(dtype=np.float32)

            #### Construct Event class out of slip and crumple label
            
            df_event = np.full((df_label.shape[0],1),10)
            df_event[np.logical_and(df_label[:,0]==0 , df_label[:,1]==0)] = 0
            df_event[np.logical_and(df_label[:,0]==1 , df_label[:,1]==0)] = 1
            df_event[np.logical_and(df_label[:,0]==0 , df_label[:,1]==1)] = 2
            df_event[np.logical_and(df_label[:,0]==1 , df_label[:,1]==1)] = 3

            event_class_counts = pd.DataFrame([['no_slip_crumple',np.count_nonzero(df_event==0)],
                            ['slip_nocrumple',np.count_nonzero(df_event==1)],
                            ['no_slip_crumple',np.count_nonzero(df_event==2)],
                            ['slip_crumple',np.count_nonzero(df_event==3)]],columns=["class","counts"])


            self.one_hot_classes = np.array([   [0,0],
                                                [1,0],
                                                [0,1],
                                                [1,1] ])

            # df_event = df_event[1:]

        
        

        df_data = []
        if 'force' in feature_set:
            df_data.append(df_force)
        if 'position' in feature_set:
            df_data.append(df_position)
        if 'mass' in feature_set:
            df_data.append(df_mass)
        if 'size' in feature_set:
            df_data.append(df_size)

        if 'joint_wise_fp' in feature_set:
            df_data.append(df.loc[:,'J0_position':'J15_Force'].to_numpy(dtype=np.float32))
            print('Joint_wise_fp')

        if 'force_derivative' in feature_set:
            df_df = np.subtract(df_force[1:],df_force[:-1])
            df_data.append(np.vstack([df_df[0][np.newaxis,:],df_df]))
            print('derivative_force')

        if 'joint_derivative' in feature_set:
            df_df = np.subtract(df_position[1:],df_position[:-1])
            df_data.append(np.vstack([df_df[0][np.newaxis,:],df_df]))
            print('derivative_joint')

        if 'force_dwt_freq_tf' in feature_set:
            pyr = prt.pyramids.WaveletPyramid(df_force[:,0], 1, 'haar', 'reflect1')
            D = pyr.pyr_coeffs[0,0]
            A = pyr.pyr_coeffs["residual_lowpass"]

            slip_onset = np.argwhere(D>20)[:,0].tolist() *2 


        df_data = np.concatenate(df_data,axis=1)

        df_data_split = np.array(np.array_split(df_data,len(file)))

        n_feature = df_data.shape[1]
        self.n_event_rows = df_data_split.shape[1]
        

        
        n_batch_n_series_feature_timestep_stack = np.lib.stride_tricks.sliding_window_view(df_data_split, (n_timestep),axis=1)
        n_batch_n_series_feature_timestep_stack = np.transpose(n_batch_n_series_feature_timestep_stack,(0,1,3,2))

        if loader=='train':
            df_event_split = np.array(np.array_split(df_event,len(file)))
            n_outputs = np.lib.stride_tricks.sliding_window_view(df_event_split, (n_timestep,),axis=1)[:,:,:,-1]
            return n_batch_n_series_feature_timestep_stack,n_outputs
            
        if loader=='test':
            return n_batch_n_series_feature_timestep_stack

        

    def read_raw_data(self,load_first_n_files=100,feature_set=['force'],n_timestep=None,loader='train'):

        if loader=='train':
            ### Raw
            train_X,train_y = self.pre_process(self.dataset.files[:load_first_n_files],feature_set=feature_set,n_timestep=n_timestep,loader='train')
            
            return train_X,train_y

        if loader=='test':
            test_X = self.pre_process(self.dataset.files,feature_set=feature_set,n_timestep=n_timestep,loader='test')
            return test_X

    def save_processed_data(self,train_X,train_y,test_X,test_y):
        #### Processed
        self.data_set_processed.train_path.mkdir(parents=True,exist_ok=True)
        np.savez((self.data_set_processed.train_path / "train").as_posix(),train_X=train_X,train_y=train_y)
        self.data_set_processed.test_path.mkdir(parents=True,exist_ok=True)
        np.savez((self.data_set_processed.test_path / "test").as_posix(),test_X=test_X,test_y=test_y)

    def load_processed_data(self):
        
        #### Load Processed
        with np.load((self.data_set_processed.train_path / "train.npz").as_posix(),allow_pickle=True) as _data:
            train_X = _data["train_X"]
            train_y = _data["train_y"]

        with np.load((self.data_set_processed.test_path / "test.npz").as_posix(),allow_pickle=True) as _data:
            test_X = _data["test_X"]
            test_y = _data["test_y"]


        return train_X,train_y,test_X,test_y




class Model(object):

    def __init__(self) -> None:
        # pass
        # Output classes to learn how to classify

        self.model_param = SimpleNamespace(class_name='')
        self.model_param.one_hot_classes = None

        self.model_path = Path(__file__).parents[2] / "models" / "PS1" 
        self.fpath = None


    def one_hot(self,y_):
        # Function to encode neural one-hot output labels from number indexes 
        # e.g.: 
        # one_hot(y_=[[5], [0], [3]], n_classes=6):
        #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        
        y_ = y_.reshape(len(y_))
        return np.eye(len(self.model_param.one_hot_classes))[np.array(y_, dtype=np.int32)]  # Returns FLOATS



    def extract_batch_size(self,_train, step, batch_size):
        # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
        
        shape = list(_train.shape)
        shape[0] = batch_size
        batch_s = np.empty(shape)

        for i in range(batch_size):
            # Loop index
            index = ((step-1)*batch_size + i) % len(_train)
            batch_s[i] = _train[index] 

        return batch_s
    
    def zero_offset_class_values(self,train_y,test_y):
        # zero-offset class values 
        return train_y - 1,test_y - 1

    def model_step_inc_drop(self,n_timesteps, n_features, n_outputs):
        self.model.add(LSTM(units = 16, return_sequences = True, input_shape = (n_timesteps,n_features,)))
        self.model.add(Dropout(0.2))#0.5
        self.model.add(LSTM(units = 8,return_sequences = True))
        self.model.add(Dropout(0.2))#0.5
        self.model.add(LSTM(units = 8,))


        # self.model.add(LSTM(units = 32 ))
        # self.model.add(Dropout(0.2))#0.5

    def model_drop_const(self,n_timesteps, n_features, n_outputs):
        self.model.add(LSTM(units = 8, return_sequences = True, input_shape = (n_timesteps,n_features,)))
        self.model.add(Dropout(0.2))#0.5
        self.model.add(LSTM(units = 8 ))
        self.model.add(Dropout(0.2))#0.5

    def model_smooth_step_inc_drop(self,n_timesteps, n_features, n_outputs):
        self.model.add(LSTM(units = 32, return_sequences = True, input_shape = (n_timesteps,n_features,)))
        self.model.add(Dropout(0.2))#0.5
        self.model.add(LSTM(units = 64 ,return_sequences=True))
        self.model.add(Dropout(0.2))#0.5
        self.model.add(LSTM(units = 64 ,return_sequences=True))
        self.model.add(Dropout(0.2))#0.5
        self.model.add(LSTM(units = 32 ))
        self.model.add(Dropout(0.2))#0.5


    def model_sharp_inc_drop(self,n_timesteps, n_features, n_outputs):
        self.model.add(LSTM(units = 50,  input_shape = (n_timesteps,n_features,)))
        self.model.add(Dropout(0.2))#0.5
    
    def model_org(self,n_timesteps, n_features, n_outputs):
        self.model.add(LSTM(units = 128, return_sequences = True, input_shape = (n_timesteps,n_features,)))
        self.model.add(Dropout(0.2))#0.5
        self.model.add(LSTM(units = 50 ))
        self.model.add(Dropout(0.2))#0.5

    def run_model(self,train_X,train_y,test_X,test_y):
        
        train_y,test_y = self.one_hot(train_y),self.one_hot(test_y)

        n_batch_size,iter_epochs = self.model_param.n_batch_size,self.model_param.epochs
        n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
        
        """
        Model : Build
        """
        self.model = Sequential()
        # self.model_step_inc_drop(n_timesteps, n_features, n_outputs)
        # self.model_smooth_step_inc_drop(n_timesteps, n_features, n_outputs)
        # self.model_sharp_inc_drop(n_timesteps, n_features, n_outputs)
        # self.model_drop_const(n_timesteps, n_features, n_outputs)
        self.model_step_inc_drop(n_timesteps, n_features, n_outputs)
        self.model.add(Dense(n_outputs,activation='softmax'))

        
        """
        Model : Compile
        """
        loss_lstm = ['categorical_crossentropy','mean_squared_error']
        self.model.compile(loss=loss_lstm[0], optimizer='adam', metrics=['accuracy'])

        """
        Run Session : Fit/Train
        """
        f_set = '_'.join(self.model_param.feature_set)
        self.f_path = self.model_path / f_set / str(n_timesteps)
        from keras.callbacks import ModelCheckpoint,TensorBoard
        self.f_path.mkdir(parents=True,exist_ok=True) 
        fname = self.f_path.as_posix() + "/model-{epoch:03d}-{val_accuracy:.4f}.hdf5"
        model_checkpoint_callback = [
                        ModelCheckpoint(filepath=fname,monitor='val_accuracy',verbose=1, save_best_only=True, mode='max'),
                        TensorBoard(log_dir=self.f_path.as_posix()+"/"+datetime.now().strftime("%Y%m%d-%H%M%S"))
                    ]

        
        self.history = self.model.fit(train_X, train_y, epochs = iter_epochs, batch_size = n_batch_size,callbacks=model_checkpoint_callback,validation_split=0.3)
        

        """
        Evaluate
        """
        _, accuracy = self.model.evaluate(test_X, test_y, batch_size=n_batch_size, verbose=0)
        rprint(Panel(f'Accuracy : {accuracy}'))

    def viz(self):
        self.model.summary()

        self.plot_curves(self.history)
        # import visualkeras
        # visualkeras.layered_view(self.model,legend=True,draw_volume=True,to_file='./img.png')

        print()

    def predict(self,test_X):

        one_hot_predictions = self.model.predict(test_X,batch_size=self.model_param.n_batch_size)

        return one_hot_predictions
        

    def get_pred_class_from_one_hot(self,one_hot_predictions,one_hot_classes=None):
        """
        Transform Predictions
        """        
        # Results
        # one_hot_predictions = one_hot_predictions.argmax(1)
        # sc.inverse_transform(predictions)

        one_hot_predictions = one_hot_predictions.argmax(1)[:,np.newaxis]

        one_hot_to_class_func = lambda one_hot_ix ,arr = one_hot_classes : arr[one_hot_ix] 
        # one_hot_to_class = np.vectorize(one_hot_to_class_func)
        one_hot_to_class = np.vstack(list(map(one_hot_to_class_func,one_hot_predictions)))

        class_predictions = one_hot_to_class # np.reshape(one_hot_to_class,(-1,2))

        

        return class_predictions

    def save_model(self,delete_model=True):
        self.model.save((self.model_path).as_posix() + "/model.h5")  # creates a HDF5 file 'my_model.h5'
        if delete_model:
            del self.model  # deletes the existing model

       
        
    def restore_model(self,f_path=None):
        from keras.models import load_model

        if f_path is None:

            # returns a compiled model
            # identical to the previous one
            best_model_path = natsorted(list(self.model_path.glob("*.hdf5")))[-1]
        else:
            best_model_path = natsorted(list(f_path.glob("*.hdf5")))[-1]

        rprint(Panel("\n\n Model restored from : "+ best_model_path.as_posix()))
        # self.model = load_model((self.model_path).as_posix() + "/model.h5")
        self.model = load_model((best_model_path).as_posix() )

        
        return self.model

    def plot_curves(self,history):

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)


        font = {
            'family' : 'Bitstream Vera Sans',
            'weight' : 'bold',
            'size'   : 18
        }
        matplotlib.rc('font', **font)

        width = 12
        height = 12
        plt.figure(figsize=(width, height))

        # indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
        plt.plot(epochs, loss,     "b--", label="Train losses")
        plt.plot(epochs, acc, "g--", label="Train accuracies")

        # indep_test_axis = np.append(
        #     np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
        #     [training_iters]
        # )
        plt.plot(epochs, val_loss,     "b-", label="Test losses")
        plt.plot(epochs, val_acc, "g-", label="Test accuracies")

        plt.title("Training session's progress over iterations")
        plt.legend(loc='upper right', shadow=True)
        plt.ylabel('Training Progress (Loss or Accuracy values)')
        plt.xlabel('Training iteration')
        plt.show()

        plt.savefig(self.f_path.as_posix()+'/history.png',bbox_inches='tight')




    def plot_cf(self,y_test,one_hot_predictions,LABELS):

        predictions = one_hot_predictions.argmax(1)

        cf_matrix = confusion_matrix(y_test,predictions)

        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]

        group_percentages = ["{0:.2%}".format(value) for value in
                            cf_matrix.flatten()/np.sum(cf_matrix)]

        labels = [f"{v1}\n{v2}" for v1, v2 in
                zip(group_counts,group_percentages)]

        labels = np.asarray(labels).reshape(4,4)

    
        ax = sns.heatmap(cf_matrix, annot=labels,fmt='', annot_kws={'fontsize':8},cmap='Blues') 
        accuracy = accuracy_score(y_test,predictions)
        precision = metrics.precision_score(y_test, predictions, average="weighted")
        recall  = metrics.recall_score(y_test, predictions, average="weighted")
        f1_score  = metrics.f1_score(y_test, predictions, average="weighted")

        stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
            accuracy,precision,recall,f1_score)
        
        ax.set_xlabel('Predicted Values'+stats_text,size='small',labelpad=-0.1)
        ax.set_ylabel('Actual Values ',size='small',labelpad=-0.1)

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(LABELS,size='small')
        ax.yaxis.set_ticklabels(LABELS,size='small',rotation=0)
        # plt.tight_layout()

        # plt.title('hdhd')
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(5, 5) 

        # plt.savefig(self.f_path.as_posix()+'/result.png')
        plt.savefig(self.f_path.as_posix()+'/result.png',bbox_inches='tight')



        print()

  

class Train(Model):

    def __init__(self) -> None:
        super().__init__()
        self.pretrain=False
        

    def run(self,train_X,train_y,test_X,test_y):

        if not self.pretrain:
            self.run_model(train_X,train_y,test_X,test_y)
            self.viz()
            self.save_model(delete_model=False)

        one_hot_predictions = self.predict(test_X)
       
        LABELS = ['no\nslip\nand\ncrumple','slip\nbut\nno-crumple','no-slip\nbut\ncrumple','slip\nand\ncrumple']
        self.plot_cf(test_y,one_hot_predictions,LABELS)

        
        # predictions = self.get_pred_class_from_one_hot(one_hot_predictions,one_hot_classes = self.model_param.one_hot_classes)
        

    

        print()


class Predict(Model):

    def __init__(self) -> None:
        super().__init__()
        self.pred_output_path = None


    def run(self,test_X,f_path=None):    

        

        self.restore_model(f_path)
        one_hot_predictions = self.predict(test_X)
        predictions = self.get_pred_class_from_one_hot(one_hot_predictions,one_hot_classes = self.model_param.one_hot_classes)
        
        partitions = int(predictions.shape[0]/(self.model_param.n_event_rows-9))
        file_wise_predictions = np.array(np.array_split(predictions,partitions))

        split_pred = lambda predict_data: pd.DataFrame(np.vstack([np.zeros((9,2)),predict_data]),columns=["Slip","Crumple"])
        outputs = list(map(split_pred,file_wise_predictions))

        print()

        return outputs


if __name__=="__main__":
    
    data_loader = DatasetLoader()
    sequence= Param(observe_window=20)
    train_X,train_y,test_X,test_y = data_loader.read_raw_data(split_train_test=80 ,load_first_n_files=100,
                                                                feature_set=['force','position'],n_timestep=sequence.observe_window,n_batch_size = 250)

    rprint(Panel(f'train_X : (n_event_files,n_samples,n_observe_steps,n_feature) \n {train_X.shape}'),
    Panel(f'train_y : (n_event_files,n_samples,1) \n {train_y.shape}'),
    Panel(f'test_X : (n_event_files,n_samples,n_observe_steps,n_feature) \n {test_X.shape}'),
    Panel(f'test_y : (n_event_files,n_samples,1) \n {test_y.shape}'))

    