# PS-IITD-USE-CASES : TCS-SMART-MACHINE

`This repo provides a code and report on activities by the TCS SMART Machine team towards the Build-a-thon 2022.`

## Requirements

- create virtual python environment using anaconda with **`python version 3.7`** and above.
```
pip install -r requirements
```

## Training and Evaluation [Reproducing the Results]

### <span style='background-color:black;color:blue;'>**`Raw Dataset`**</span>

### Problem statement I - Slip Dtetection and Force Estimation
- [Training Dataset Download](https://drive.google.com/file/d/1GMl2EDRemXZrdwgYTCV4mk10lXxQdHb-/view)

### Problem statement II - Object Detection
- [Training Dataset download](https://drive.google.com/file/d/19uaJjJY0-ItKJ35f_OI6Nb_A_3uAuKTc/view)

### <span style='background-color:black;color:blue;'>**`Training`**</span>

- Inside the root of the repo, there would be the following files (exactly as named below):
    
    
    - <span style='color:blue;'>task1.py</span>, this script contains the code that reads the test set  and predicts the specified output for Task Problem Statement I: Slip Detection and Force Estimation in CSV format (one per line). The format of test set will be same as pre-evaluation dataset (i.e., the path will be the root folder containing all CSV files in the testset)
    - <span style='color:blue;'>task2.py</span>, this script contains the code that reads the test set and predicts the specified output for Task Problem Statement II: Object Detection in CSV format (one per line). A sample output is here. The format of test set will be same as pre-evaluation dataset (i.e., the path will be the complete path to ONE AND ONLY CSV file in the testset)
    - <span style='color:blue;'>requirements.txt</span>, this contains the requirements to load the dependent modules for your scripts.
    - <span style='color:blue;'>task1_training.py</span>, this script contains the code to read a train set and train a model with the data that must be saved/dumped as ./model_checkpoint_task1
    - <span style='color:blue;'>task2_training.py</span>, this script contains the code to read a train set and train a model with the data that must be saved/dumped as ./model_checkpoint_task2
    - model_checkpoint1
    - model_checkpoint2
    

### **For task1 (Problem Statement I)**

```
python task1_training.py --model_path ./model_checkpoint1 --input_data ./path/to/trainset/directory
```
- --input_data argument contains the path to the directory of the training set. `--input_data expects the path of a root directory containing multiple CSV files belonging to the training set.`

### **For task2 (Problem Statement II)**
```
python task2_training.py --model_path ./model_checkpoint2 --input_data ./path/to/trainset.csv
```
- --input_data argument contains the path to the training set.

### **`Testing`**

### **For task1 (Problem Statement I)**
```
python task1.py --model_path ./model_checkpoint1 --input_data /path/to/testset/directory --output ./path/to/predictions/directory
```
- --input_data argument contains the path to the directory of the hidden test set. --input_data expects the path of a root directory where multiple CSV files exist belonging to the test-set

### **For task1 (Problem Statement II)**
```
python task2.py --model_path ./model_checkpoint2 --input_data ./path/to/testset.csv --output ./predictions_task2.csv
```
- --input_data argument contains the end path to the hidden test set. --input_data expects the argument to be the complete path to the single CSV file representing the testset. 