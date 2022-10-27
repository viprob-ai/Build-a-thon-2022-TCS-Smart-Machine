import os
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from natsort import natsorted
from prettytable import PrettyTable
from rich import print as rprint
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import track

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from config import Dataset

import seaborn as sns

class DatasetLoader(object):

    def __init__(self) -> None:

        self.path = Path(__file__).parents[2] / "data" / "raw" / "PS_2" 


        self.data_set = Dataset(name='object_detection',
                        data_path = self.path, 
                        train_path = (self.path/ "OBJECT_DETECTION_TRAIN_SET"),
                        test_path = (self.path/  "OBJECT_DETECTION_VAL_SET"))

        self.dataset = SimpleNamespace()
        self.dataset.files = natsorted(list(self.data_set.train_path.rglob("*.csv")))
        


        
        self.data_set_processed = Dataset(name='object_detection',
                        data_path = (self.path.parents[1] / "processed" / "PS2" ), 
                        train_path = (self.path.parents[1] / "processed" / "PS2" / "train" ),
                        test_path = (self.path.parents[1] / "processed" / "PS2" / "test" ))


        self.data_set_feature = Dataset(name='object_detection',
                        data_path = (self.path.parents[1] / "feature" / "PS2" ), 
                        train_path = (self.path.parents[1] / "feature" / "PS2" / "train" ),
                        test_path = (self.path.parents[1] / "feature" / "PS2" / "test" ))


        self.one_hot_classes = np.array(['BALL_A', 'BALL_B', 'BALL_C', 'CUBE_A', 'CUBE_B', 'CUBE_C', 'CUBOID_A', 'CUBOID_B', 'RUGBY_A', 'RUGBY_B', 'SPHERE_A', 'SPHERE_B','SPHERE_C'])[:,np.newaxis]
        


        


    def pre_process(self,file,loader='train',feature_set=['force'],SAMPLE_X_PERC_DATA=None) :

        read_event = lambda fpath : pd.read_csv(fpath.as_posix())
        # pad_event = lambda df_event : pd.concat([pd.concat([df_event.iloc[0].to_frame().T]*(n_timestep)),df_event],ignore_index=True)
        read_tf_events = lambda fpath : read_event(fpath)

        df = pd.concat(map(read_tf_events , file),ignore_index=True,axis=0)
        
        # if split_data is not None:
        #     df = df.iloc[split_data[0]:split_data[-1],:]

        object_held_data = SimpleNamespace(n_classes='',n_categories='')

        df_position = df.loc[:,'J0_position':'J15_position':2].to_numpy(dtype=np.float32)
        df_force = df.loc[:,'J0_Force':'J15_Force':2].to_numpy(dtype=np.float32)
        df_mass = df.loc[:,'MASS_kg'].to_numpy(dtype=np.float32)[:,np.newaxis]
        # df_size = df.loc[:,'Size']

        if loader=='train':
            n_classes = natsorted(df["Object_Held"].unique())
            n_categories = np.unique([(n_class.split('_')[0]) for n_class in n_classes])

         
            object_held_data.n_classes = n_classes
            object_held_data.n_categories = n_categories

            df_label = df.loc[:,"Object_Held"]

            n_class_dict = []
            for n_class in n_classes:
                n_class_dict.append([n_class,np.count_nonzero(df_label.str.contains(n_class, case=False, na=False))])

            n_class_dict = pd.DataFrame(n_class_dict,columns=["class","counts"])
            rprint(Panel(n_class_dict.to_string()))

            
            n_category_dict = []
            for n_category in n_categories:
                n_category_dict.append([n_category,np.count_nonzero(df_label.str.contains(n_category, case=False, na=False))])

            n_category_dict = pd.DataFrame(n_category_dict,columns=["category","counts"])
            
            rprint(Panel(n_category_dict.to_string()))

            df_class = np.zeros((df_label.shape[0],1))#df.loc[:,"Object_Held"].copy()
            df_category = np.zeros((df_label.shape[0],1))
        

        df_data = []
        if 'force' in feature_set:
            df_data.append(df_force)
            
        if 'position' in feature_set:
            df_data.append(df_position)
        
        if 'mass' in feature_set:
            df_data.append(df_mass)

        if 'forceXposition' in feature_set:
            df_data.append(np.multiply(df_force,df_position))

        # if 'size' in feature_set:
        #     df_data.append(df_size)

        df_data = np.concatenate(df_data,axis=1)

        if loader=='train':

            n_sample_class = int(min(n_class_dict["counts"])*(SAMPLE_X_PERC_DATA/100))
            n_sample_category = int(min(n_category_dict["counts"])*(SAMPLE_X_PERC_DATA/100))

        

        sample_class = []
        sample_category = []

      


        if loader=='train':
            sample_train_X = []
            sample_train_y_class = []
            sample_train_y_category = []

            sample_test_X = []
            sample_test_y_class = []
            sample_test_y_category = []

            for ix,n_class in track(enumerate(n_classes),description='class....',total=len(n_classes)):
            
                mask = df_label.str.contains(n_class, case=False, na=False)
                df_class[mask] = ix

                # print(ix)
                
                train_index = np.random.choice(np.arange(df_label.shape[0])[mask], n_sample_class, replace=False)
                
                test_index = []
                [test_index.append(_ix) if _ix not in train_index else None for _ix in np.arange(df_label.shape[0])[mask] ]


                sample_train_X.append(df_data[train_index])
                sample_train_y_class.append(df_class[train_index])
                # sample_train_y_category.append(df_category[train_index])

                sample_test_X.append(df_data[test_index])
                sample_test_y_class.append(df_class[test_index])
                # sample_test_y_category.append(df_category[test_index])

                
            sample_class = [sample_train_X,sample_train_y_class,sample_train_y_category,sample_test_X,sample_test_y_class,sample_test_y_category]


            sample_train_X = []
            sample_train_y_class = []
            sample_train_y_category = []

            sample_test_X = []
            sample_test_y_class = []
            sample_test_y_category = []
            for ix,n_category in track(enumerate(n_categories),description='category....',total=len(n_categories)):
                mask = df_label.str.contains(n_category, case=False, na=False)
                df_category[mask] = ix
                # print(ix)


                train_index = np.random.choice(np.arange(df_label.shape[0])[mask], n_sample_category, replace=False)
                test_index = []
                [test_index.append(_ix) if _ix not in train_index else None for _ix in np.arange(df_label.shape[0])[mask] ]

                sample_train_X.append(df_data[train_index])
                sample_train_y_class.append(df_class[train_index])
                # sample_train_y_category.append(df_category[train_index])

                sample_test_X.append(df_data[test_index])
                sample_test_y_class.append(df_class[test_index])
                # sample_test_y_category.append(df_category[test_index])

            sample_category = [sample_train_X,sample_train_y_class,sample_train_y_category,sample_test_X,sample_test_y_class,sample_test_y_category]
            
                

            # train_X_class = np.vstack(sample_train_X_class)
            # train_y_class = np.vstack(sample_train_y_class)
            # test_X_class = np.vstack(sample_test_X_class)
            # test_y_class = np.vstack(sample_test_y_class)


            # train_X_category = np.vstack(sample_train_X_category)
            # train_y_category = np.vstack(sample_train_y_category)
            # test_X_category = np.vstack(sample_test_X_category)
            # test_y_category = np.vstack(sample_test_y_category)




            

            return {"class":sample_class,"category" : sample_category},object_held_data

        if loader=='test':
            return df_data

 

    def read_raw_data(self,SAMPLE_X_PERC_DATA=80,loader='train',feature_set=['force']):

        if loader=='train':
            ### Raw
            data,object_held_data = self.pre_process(self.dataset.files,feature_set=feature_set,SAMPLE_X_PERC_DATA=SAMPLE_X_PERC_DATA)
         
            
         
            return data,object_held_data 

        if loader=='test':
            test_X = self.pre_process(self.dataset.files,loader='test',feature_set=feature_set)
            return {"test": test_X}

    def save_processed_data(self,dict_to_save):
        #### Processed
        self.data_set_processed.train_path.mkdir(parents=True,exist_ok=True)
        # np.savez((data_loader.data_set_processed.train_path / "train").as_posix(),train_X=train_X,train_y_class=train_y_class,train_y_category=train_y_category,n_classes=train_data.n_classes,n_categories=train_data.n_categories)
        np.savez((self.data_set_processed.train_path / "train").as_posix(),**dict_to_save)

        # self.data_set_processed.test_path.mkdir(parents=True,exist_ok=True)
        # np.savez((self.data_set_processed.test_path / "test").as_posix(),test_X=test_X,test_y=test_y)

    def load_processed_data(self):
        
        #### Load Processed
        with np.load((self.data_set_processed.train_path / "train.npz").as_posix(),allow_pickle=True) as _data:
            train_X = _data["train_X"]
            train_y_class = _data["train_y_class"]
            train_y_category = _data["train_y_category"]
            n_classes = _data["n_classes"]
            n_categories = _data["n_categories"]

        # with np.load((self.data_set_processed.test_path / "test.npz").as_posix(),allow_pickle=True) as _data:
        #     test_X = _data["test_X"]
        #     test_y = _data["test_y"]


        return  train_X,train_y_class,train_y_category,n_classes,n_categories


   





   

class Model(object):
    
    def __init__(self) -> None:

        
        self.model_param = SimpleNamespace(class_name='')
        self.model_param.one_hot_classes = np.array(['BALL_A', 'BALL_B', 'BALL_C', 'CUBE_A', 'CUBE_B', 'CUBE_C', 'CUBOID_A', 'CUBOID_B', 'RUGBY_A', 'RUGBY_B', 'SPHERE_A', 'SPHERE_B','SPHERE_C'])[:,np.newaxis]

        self.model_path = Path(__file__).parents[2] / "models" / "PS2" 


    def one_hot(self,y_):
        # Function to encode neural one-hot output labels from number indexes 
        # e.g.: 
        # one_hot(y_=[[5], [0], [3]], n_classes=6):
        #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        
        y_ = y_.reshape(len(y_))
        return np.eye(len(self.model_param.one_hot_classes ))[np.array(y_, dtype=np.int32)]  # Returns FLOATS


    def classifier_model(self,train_X,train_y,test_X,test_y):

        train_y,test_y = self.one_hot(train_y),self.one_hot(test_y)
       

        self.classifier = RandomForestClassifier()
        self.classifier.fit( train_X,train_y)

       
    
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
        one_hot_to_class = np.array(list(map(one_hot_to_class_func,one_hot_predictions)))

        class_predictions = np.reshape(one_hot_to_class,(-1,1))

        return class_predictions



    def predict(self,_X):
        y_ = self.classifier.predict(_X)
        # self.classifier.eval %% percenen

        return y_


    def save_model(self,delete_model=True):

        f_set = '_'.join(self.model_param.feature_set)
        self.f_path = self.model_path / f_set 
        
        self.f_path.mkdir(parents=True,exist_ok=True) 
        
        # save
        joblib.dump(self.classifier, (self.f_path).as_posix() + "/model.joblib")

        rprint("Saving the model  "+(self.f_path).as_posix() + "/model.joblib")
        if delete_model:
            del self.classifier  # deletes the existing model

       
        
    def restore_model(self,f_path=None):

        if f_path is None:

            # load
            self.classifier= joblib.load((self.model_path).as_posix() + "/model.joblib")
            rprint("Loading the model from   "+(self.model_path).as_posix() + "/model.joblib")
            
        else:
            # load
            self.classifier= joblib.load((f_path).as_posix() + "/model.joblib")
            rprint("Loading the model from   "+(f_path).as_posix() + "/model.joblib")



        return self.classifier


    def plot_cf(self,y_test,one_hot_predictions,LABELS):

        predictions = one_hot_predictions.argmax(1)

        cf_matrix = confusion_matrix(y_test,predictions)


        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]

        group_percentages = ["{0:.2%}".format(value) for value in
                            cf_matrix.flatten()/np.sum(cf_matrix)]

        labels = [f"{v1}\n{v2}" for v1, v2 in
                zip(group_counts,group_percentages)]

        labels = np.asarray(labels).reshape(13,13)

        
        ax = sns.heatmap(cf_matrix, annot=labels,fmt='', annot_kws={'fontsize':16},cmap='Blues')

        accuracy = accuracy_score(y_test,predictions)
        precision = metrics.precision_score(y_test, predictions, average="weighted")
        recall  = metrics.recall_score(y_test, predictions, average="weighted")
        f1_score  = metrics.f1_score(y_test, predictions, average="weighted")

        stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
            accuracy,precision,recall,f1_score)
        
        ax.set_xlabel('Predicted Values'+stats_text,size='large',labelpad=-0.1)
        ax.set_ylabel('Actual Values ',size='large',labelpad=-0.1)

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(LABELS,size='large')
        ax.yaxis.set_ticklabels(LABELS,size='large',rotation=0)
        # plt.tight_layout()
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        # plt.show()
        # plt.pause(0.001)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(32, 18) 

        plt.savefig(self.f_path.as_posix()+'/result.png')



        print()


class Train(Model):

    def __init__(self) -> None:
        super().__init__()
        self.pretrain=False

    def run(self,train_X,train_y,test_X,test_y):

        # rprint(Panel(f'train_X : (n_samples,n_feature) \n {train_X.shape}'+'\n'+f'train_y : (n_samples,1) \n {train_y.shape}' + "\n\n"+
        #         f'test_X : (n_samples,n_feature) \n {test_X.shape}'+'\n'+f'test_y : (n_samples,1) \n {test_y.shape}'
        #         ,title="TrainTest_Split")
        #         )

        if not self.pretrain:
            self.classifier_model(train_X,train_y,test_X,test_y)
            self.save_model(delete_model=False)


        one_hot_predictions = self.predict(test_X)

        LABELS = self.model_param.one_hot_classes

        self.plot_cf(test_y,one_hot_predictions,LABELS)

        # predictions = self.get_pred_class_from_one_hot(one_hot_predictions,one_hot_classes = self.model_param.one_hot_classes)
        # truth = self.get_pred_class_from_one_hot(test_y,one_hot_classes = self.model_param.one_hot_classes)
        # rprint(Panel(f"Accuracy : {accuracy_score(truth,predictions)}"))


        # result = PrettyTable(title='Result')
        # result.field_names = ["Object_Held",'truth']
        # result.add_rows(np.array([predictions,truth]).T)

        # print(result.get_string(start=1,end=10))

    

        print()



class Predict(Model):

    def __init__(self) -> None:
        super().__init__()
        self.pred_output_path = None


    def run(self,test_X,f_path=None):

        self.restore_model(f_path)
        one_hot_predictions = self.predict(test_X)
        predictions = self.get_pred_class_from_one_hot(one_hot_predictions,one_hot_classes = self.model_param.one_hot_classes)
        
 
        # result = PrettyTable(title='Result')
        # result.field_names = ["Object_Held"]
        # result.add_rows(predictions)

        # print(result.get_string(start=1,end=10))

        output = pd.DataFrame(predictions)
        output.columns = ["Object_Held"]
        output.to_csv(self.pred_output_path.as_posix(),index=False)

        print('Saving the predictions')

    


if __name__=="__main__":

    data_loader = DatasetLoader()
    
    data_dict,object_held_data = data_loader.read_raw_data(SAMPLE_X_PERC_DATA=80, feature_set=['force','position'],loader='train')

    train_X,train_y = data_dict["train"]["class"]
    test_X,test_y = data_dict["test"]["class"]

    rprint(Panel(f'train_X : (n_samples,n_feature) \n {train_X.shape}'),
    Panel(f'train_y : (n_samples,1) \n {train_y.shape}'),
    Panel(f'test_X : (n_samples,n_feature) \n {test_X.shape}'),
    Panel(f'test_y : (n_samples,1) \n {test_y.shape}'))

  

    print()