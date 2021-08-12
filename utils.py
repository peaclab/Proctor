#!/usr/bin/env python3
### Burak Adaptability Project Utils @BurHack

### GENERIC
import copy
import datetime
import io
import os
from os import listdir
from os.path import isfile, join, isdir
import sys
from functools import partial
from pathlib import Path




### DATA PROCESS
import pandas as pd
import numpy as np
import ast 
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
import re
from tqdm import tqdm

### PLOTTING & LOGS
import matplotlib.pyplot as plt
import logging
from pylab import rcParams
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import seaborn as sns
sns.set(rc={'figure.figsize':(12,10)})
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("talk")


### DATA STORING
import h5py
import pickle
import json

### RANDOM
import random
import time
#from numpy.random import seed

### TENSORFLOW
import tensorflow as tf

### MULTIPROCESSING
import multiprocessing
from multiprocessing import Pool
#print("CPU COUNT:", multiprocessing.cpu_count())
from fast_features import generate_features
from scipy.stats import ks_2samp


### PLOTTING RELATED
def changeBarWidth(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
        
def collectUnseenExperimentResults(conf, cv_index, selected_anom, DEBUG=False):
    
    score_df = pd.DataFrame()    
    alarm_df = pd.DataFrame()        
    report_path = conf['results_dir'] 
    
    #Read model's results for different finetuning data sizes
    for json_file in sorted(report_path.glob("*.json")):
                
        method_name = str(json_file).split('/')[-1].split('_')[0]        
        anom_type = str(json_file).split('/')[-1].split('_')[5]        
        
        if anom_type == str(selected_anom):        
#             print(anom_type)
#             print(percentage)
#             print(json_file)
            
            if method_name == 'tuncer':
                method_name = 'RF-Tuncer'
            elif method_name == 'aksar':
                method_name = 'Proctor'
                               
            try:
                with open(json_file) as file:
                    report_dict = json.load(file)
            except:
                print("No such file or directory for split:",json_file)                 

                        
            model_name = str(json_file).split('/')[-1].split('_')[1]
            dataset = str(json_file).split('/')[-1].split('_')[2]
            percentage = float(str(json_file).split('/')[-1].split('_')[3])
            threshold = float((str(json_file).split('/')[-1].split('_')[4])) 

            score_df = score_df.append({'threshold':threshold,
                                        'model': method_name + '_' + model_name,
                                        'percentage':percentage,
                                        'anomaly':anom_type,                                        
                                        'f1-score':report_dict['macro_fscore'],
                                         'dataset': dataset
                                       },ignore_index=True)

            alarm_df = alarm_df.append({'threshold':threshold,
                                        'model': method_name + '_' + model_name,
                                        'percentage':percentage,                                    
                                        'anomaly':anom_type,                                                                     
                                        'false_alarm_rate':report_dict['false_alarm_rate'],
                                        'anom_miss_rate':report_dict['anom_miss_rate'],
                                        'dataset': dataset
                                       },ignore_index=True)            
            
        
    score_df.sort_values(by='threshold',ascending=False,inplace=True)
    alarm_df.sort_values(by='threshold',ascending=False,inplace=True)
    
    return score_df,alarm_df
    

def collectExperimentResults(conf, cv_index, DEBUG=False):
        
    score_df = pd.DataFrame(columns=['size','f1-score'])    
    alarm_df = pd.DataFrame(columns=['size','anom_miss_rate','false_alarm_rate'])        
    
    report_path = conf['results_dir'] 
    
    #Read model's results for different finetuning data sizes
    for json_file in sorted(report_path.glob("*.json")):
        
        if DEBUG:
            print(json_file)
        method_name = str(json_file).split('/')[-1].split('_')[0]
        logging.info(method_name)        
        if method_name == 'tuncer':
            method_name = 'RF-Tuncer'
        elif method_name == 'aksar':
            method_name = 'Proctor'
        
        logging.info(method_name)
                    
        try:
            with open(json_file) as file:
                report_dict = json.load(file)
                
        except:
            print("No such file or directory for split:",json_file)                 
        
            
        #report_type = str(json_file).split('/')[-1].split('_')[4]
        #size = (int(float((str(json_file).split('/')[-1].split('_')[3])) * 100)) 
        size = float((str(json_file).split('/')[-1].split('_')[3])) 
        model_name = str(json_file).split('/')[-1].split('_')[1]
        dataset = str(json_file).split('/')[-1].split('_')[2]

                              

        score_df = score_df.append({'size':size,
                                    'model': method_name + '_' + model_name,
                                    'f1-score':report_dict['macro_fscore']},ignore_index=True)

        alarm_df = alarm_df.append({'size':size,
                                    'model': method_name + '_' + model_name,
                                    'false_alarm_rate':report_dict['false_alarm_rate'],
                                    'anom_miss_rate':report_dict['anom_miss_rate'],},ignore_index=True)            
                            
    score_df.sort_values(by='size',ascending=False,inplace=True)
    alarm_df.sort_values(by='size',ascending=False,inplace=True)
    
    return score_df,alarm_df




def collectBorghesiResults(conf, cv_index, DEBUG=False):
        
    score_df = pd.DataFrame(columns=['size','f1-score'])    
    alarm_df = pd.DataFrame(columns=['size','anom_miss_rate','false_alarm_rate'])        
    
    report_path = conf['results_dir'] 
    
    #Read model's results for different finetuning data sizes
    for json_file in sorted(report_path.glob("*.json")):
        
        if DEBUG:
            print(json_file)
        method_name = str(json_file).split('/')[-1].split('_')[0]
        logging.info(method_name)        
        if method_name == 'tuncer':
            method_name = 'RF-Tuncer'
        elif method_name == 'aksar':
            method_name = 'Proctor'
        
        logging.info(method_name)
                    
        try:
            with open(json_file) as file:
                report_dict = json.load(file)
                
        except:
            print("No such file or directory for split:",json_file)                 
                    
        #size = (int(float((str(json_file).split('/')[-1].split('_')[3])) * 100)) 

        model_name = str(json_file).split('/')[-1].split('_')[1]
        dataset = str(json_file).split('/')[-1].split('_')[2]
        size = float((str(json_file).split('/')[-1].split('_')[3])) 
        report_type = str(json_file).split('/')[-1].split('_')[4]        
        
        if report_type == 'report':                                    

                score_df = score_df.append({'size':size,
                                            'model': method_name + '_' + model_name,
                                            'dataset': dataset,
                                      'f1-score':report_dict['macro avg']['f1-score']},ignore_index=True)

        elif report_type == 'alert':          

                alarm_df = alarm_df.append({'size':size,
                                            'model': method_name + '_' + model_name,
                                            'dataset': dataset,
                                      'false_alarm_rate':report_dict['false_alarm_rate'],
                                      'anom_miss_rate':report_dict['anom_miss_rate'],},ignore_index=True)            
                            
    score_df.sort_values(by='size',ascending=False,inplace=True)
    alarm_df.sort_values(by='size',ascending=False,inplace=True)
    
    return score_df,alarm_df


def readModelConfig(exp_name,cv_index,model_name,system):
    """Reads saved config file and returns as a dictionary"""
    
    import math    
    config_path = Path('/projectnb/peaclab-mon/aksar/adaptability_experiments/{system}/{exp_name}/CV_{cv_index}/{model_name}/model_config.csv'.format(system=system,exp_name=exp_name,cv_index=cv_index,model_name=model_name))
    conf = {}
    try:
        conf_csv = pd.read_csv(config_path)
    except:
        logging.info("Config.csv doesn't exist")
    

    for column in conf_csv.columns:
        if isinstance(conf_csv[column][0],str):
            if 'dir' in column:
                conf[column] = Path(conf_csv[column][0])
            else:
                conf[column] = conf_csv[column][0]
                
        #FIXME: Find a generic comparison for integers
        elif isinstance(conf_csv[column][0],np.int64):
                conf[column] = conf_csv[column][0]  
                
        elif isinstance(conf_csv[column][0],np.bool_):
                conf[column] = conf_csv[column][0]                  
        else:
            if math.isnan(conf_csv[column][0]):
                conf[column] = None
        
    return conf


def readExperimentConfig(exp_name,system):
    """Reads saved config file and returns as a dictionary"""
    
    import math    
    config_path = Path('/projectnb/peaclab-mon/aksar/adaptability_experiments/{system}/{exp_name}/exp_config.csv'.format(system=system,exp_name=exp_name))
    
    try:
        conf_csv = pd.read_csv(config_path)
    except:
        logging.info("Config.csv doesn't exist")
    
    conf = {}
    for column in conf_csv.columns:
        if isinstance(conf_csv[column][0],str):
            if 'dir' in column:
                conf[column] = Path(conf_csv[column][0])
            else:
                conf[column] = conf_csv[column][0]
                
        #FIXME: Find a generic comparison for integers
        elif isinstance(conf_csv[column][0],np.int64):
                conf[column] = conf_csv[column][0]  
                
        elif isinstance(conf_csv[column][0],np.bool_):
                conf[column] = conf_csv[column][0]                  
        else:
            if math.isnan(conf_csv[column][0]):
                conf[column] = None
        
    return conf





class WindowShopper: 

    def __init__(self, data, labels, window_size = 64, trim=30, silent=False):
        '''Init'''
        self.data = data
        self.labels = labels
        if self.labels is not None:
            self.label_count = len(labels['anom'].unique()) #Automatically assuming anomaly classification
        self.trim = trim
        self.silent = silent

        #Windowed data and labels
        self.windowed_data = []
        self.windowed_label = []
        
        #Output shape
        self.window_size = window_size
        self.metric_count = len(data.columns)
        self.output_shape = (self.window_size, self.metric_count)    

        #Prepare windows
        self._get_windowed_dataset()
    
    #Not calling this but it is good to have
    def _process_sample_count(self):
        self.per_label_count = {x: 0 for x in self.labels[self.labels.columns[0]].unique()}
        self.sample_count = 0
        for node_id in self.data.index.get_level_values('node_id').unique():
            counter = 0
            cur_array = self.data.loc[node_id, :, :]
            for i in range(self.trim, len(cur_array) - self.window_size - self.trim):
                counter += 1
            self.sample_count += counter
            self.per_label_count[self.labels.loc[node_id, self.labels.columns[0]]] += counter

    def _get_windowed_dataset(self):

        if self.labels is not None:
            #Iterate unique node_ids
            for node_id in self.labels.index.unique():
              # print(node_id)
                cur_array = self.data.loc[node_id,:,:]

                temp_data = []
                temp_label = []
                #Iterate over application runtime
                for i in range(self.trim, len(cur_array) - self.window_size - self.trim):

                    self.windowed_data.append(cur_array.iloc[i:i+self.window_size].to_numpy(
                      dtype=np.float32).reshape(self.output_shape))
                    self.windowed_label.append(self.labels.loc[node_id])

            self.windowed_data = np.dstack(self.windowed_data)
            self.windowed_data = np.rollaxis(self.windowed_data,2)
            if not self.silent:
                logging.info("Windowed data shape: %s",self.windowed_data.shape)
            #FIXME: column names might be in reverse order for HPAS data, Used app, anom for Cori data but it was anom,app

            self.windowed_label = pd.DataFrame(np.asarray(self.windowed_label).reshape(len(self.windowed_label),2),columns=['app','anom'])

            if not self.silent:
                logging.info("Windowed label shape: %s",self.windowed_label.shape)
        else:
            logging.info("Deployment selection - no label provided")
            
            cur_array = self.data

            temp_data = []
            temp_label = []
            #Iterate over application runtime
            for i in range(self.trim, len(cur_array) - self.window_size - self.trim):

                self.windowed_data.append(cur_array.iloc[i:i+self.window_size].to_numpy(
                  dtype=np.float32).reshape(self.output_shape))

            self.windowed_data = np.dstack(self.windowed_data)
            self.windowed_data = np.rollaxis(self.windowed_data,2)                
                
            self.windowed_label = None

    def return_windowed_dataset(self):

        return self.windowed_data, self.windowed_label
    
def granularityAdjust(data,granularity=60):
    
    result = pd.DataFrame()
    for nid in data.index.get_level_values('node_id').unique():
        temp_data = data[data.index.get_level_values('node_id') == nid]
        temp_data = temp_data.iloc[ \
            (temp_data.index.get_level_values('timestamp').astype(int) -
             int(temp_data.index.get_level_values('timestamp')[0])) \
            % granularity == 0]
        result = pd.concat([result,temp_data])
                
    return result    

class MyEncoder:
    def fit_transform(self, labels,dataset):
        self.dataset = dataset
        self.fit_anom(labels)
        self.fit_appname(labels)
        return self.transform(labels)

    def fit_anom(self, labels):
        self.anoms = labels['anom'].unique()
        self.anom_dict = {}
        for idx, i in enumerate(self.anoms):
            self.anom_dict[i] = idx
            
    def fit_appname(self,labels):
        self.apps = labels['app'].unique()
        self.app_dict = {}
        for idx, i in enumerate(self.apps):
            self.app_dict[i] = idx
            
    def transform(self, labels):
        if self.dataset == 'tpds':
            labels['anom'] = labels['anom'].apply(self.anom_dict.get)
            labels['app'] = labels['app'].apply(self.app_dict.get)
        elif self.dataset == 'hpas':
            labels['anom'] = labels['anom'].apply(self.anom_dict.get)
            labels['app'] = labels['app'].apply(self.app_dict.get)            

        elif self.dataset == 'cori':
            raise NotImplemented
        #labels.rename(columns={'anomaly':"anom",'appname':"app"},inplace=True)
        
        return labels    
    
    
#TODO: Make the second reader parallel
_TIMESERIES = None

def _get_features(node_id, features=None, **kwargs):
    global _TIMESERIES
    assert (
        features == ['max', 'min', 'mean', 'std', 'skew', 'kurt',
                     'perc05', 'perc25', 'perc50', 'perc75', 'perc95']
    )
#    print("Kwargs Trial",kwargs['trim']);

    if isinstance(_TIMESERIES, pd.DataFrame):
        df = pd.DataFrame(
            generate_features(
                np.asarray(_TIMESERIES.loc[node_id, :, :].values.astype('float'), order='C'),
                trim=kwargs['trim']
            ).reshape((1, len(_TIMESERIES.columns) * 11)),
            index=[node_id],
            columns=[feature + '_' + metric
                     for metric in _TIMESERIES.columns
                     for feature in features])
        return df
    else:
        # numpy array format compatible with Burak's notebooks
        return generate_features(
                np.asarray(_TIMESERIES[node_id].astype(float), order='C'),
                trim=kwargs['trim']

            ).reshape((1, _TIMESERIES.shape[2] * 11))

class _FeatureExtractor:
    def __init__(self, features=None, window_size=None, trim=None):
        self.features = features
        self.window_size = window_size
        self.trim = trim

    def __call__(self, node_id):
        return _get_features(
            node_id, features=self.features,
            window_size=self.window_size, trim=self.trim)

class TSFeatureGenerator:
    """Wrapper class for time series feature generation"""

    def __init__(self, trim=60, threads=multiprocessing.cpu_count(),
                 features=['max', 'min', 'mean', 'std', 'skew', 'kurt',
                           'perc05', 'perc25', 'perc50', 'perc75', 'perc95']):
        self.features = features
        self.trim = trim
        self.threads = threads

    def fit(self, x, y=None):
        """Extracts features
            x = training data represented as a Pandas DataFrame
            y = training labels (not used in this class)
        """
        return self

    def transform(self, x, y=None):
        """Extracts features
            x = testing data/data to compare with training data
            y = training labels (not used in this class)
        """
        global _TIMESERIES
        _TIMESERIES = x
        if isinstance(x, pd.DataFrame):
            with Pool(processes=self.threads) as pool:
                result = pool.map(
                    _FeatureExtractor(features=self.features,
                                      window_size=0, trim=self.trim),
                    x.index.get_level_values('node_id').unique())
                pool.close()
                pool.join()
                return pd.concat(result)
        else:
            # numpy array format compatible with Burak's notebooks
            result = [
                      _FeatureExtractor(features=self.features,
                                  window_size=0, trim=self.trim)(i) for i in range(len(x))]
            return np.concatenate(result, axis=0)

def generate_rolling_features(time_series, features=None, window_size=0, trim=60):
    
    assert(features is not None)
    if trim != 0:
        time_series = time_series[trim:- trim]
    if window_size > len(time_series) or window_size < 1:
        window_size = len(time_series)
    df_rolling = time_series.rolling(window_size)
    columns = time_series.columns
    df_features = []
    col_map = {}

    def add_feature(f, name):
        nonlocal df_features
        nonlocal df_rolling
        col_map = {}
        for c in columns:
            col_map[c] = feature + '_' + c
        df_features.append(f(df_rolling)[window_size - 1:].rename(index=str, columns=col_map))

    percentile_regex = re.compile(r'perc([0-9]+)')
    for feature in features:
        percentile_match = percentile_regex.fullmatch(feature)
        if feature == 'max':
            add_feature(lambda x: x.max(), feature)
        elif feature == 'min':
            add_feature(lambda x: x.min(), feature)
        elif feature == 'mean':
            add_feature(lambda x: x.mean(), feature)
        elif feature == 'std':
            add_feature(lambda x: x.var(), feature)
        elif feature == 'skew':
            add_feature(lambda x: x.skew().fillna(0), feature)
        elif feature == 'kurt':
            add_feature(lambda x: x.kurt().fillna(-3), feature)
        elif percentile_match is not None:
            quantile = float(percentile_match.group(1)) / 100
            add_feature(lambda x: x.quantile(quantile), feature)
        else:
            raise ValueError("Feature '{}' could not be parsed".format(feature))

    df = pd.concat(df_features, axis=1)
    return df        
        
        
def get_nids_apps(metadata,appname):

    nids = metadata[metadata['app'] == appname]['node_ids']
    nids = nids.apply(ast.literal_eval)
    nids_list = []
    for temp_list in nids:
        nids_list = nids_list + temp_list

    return nids_list

def smart_select(label_df, case, anom_type=None, app_type=None):

    anom_dict = dict(label_df['anom'].value_counts())
    logging.info("Anomaly distribution %s", anom_dict)
    app_dict = dict(label_df['app'].value_counts())
    logging.info("App distribution %s",app_dict)


    #Select only one anomaly
    if case == 1:
        logging.info("Selected ANOMALY type: %s",anom_type)
        return pd.DataFrame(label_df[label_df['anom'] == anom_type])
    #Select only one app
    elif case == 2:
        logging.info("Selected APP type: %s",app_type)
        return pd.DataFrame(label_df[label_df['app'] == app_type])
    #Select multiple anoms
    elif case == 3:
        logging.info("Selected ANOMALY types: %s",anom_type)
        return pd.DataFrame(label_df[label_df['anom'].isin(anom_type)])
    #Select multiple apps
    elif case == 4:
        logging.info("Selected APP types: %s",app_type)
        return pd.DataFrame(label_df[label_df['app'].isin(app_type)])
    #Select multiple apps and anoms
    elif case ==5:
        logging.info("Selected APP type, %s", app_type)
        logging.info("Selected ANOM type, %s",anom_type)

        try:
            if(len(label_df[label_df['anom'].isin(anom_type) & label_df['app'].isin(app_type)]) == 0):
                raise Exception
            else:
                return label_df[label_df['anom'].isin(anom_type) & label_df['app'].isin(app_type)]        
        except:
            logging.info("Provided combination does NOT exist!")
            return

        
    else:
        logging.info("Invalid case selection")        
        return

def read_h5file(READ_PATH, filename):

    logging.info("Reading h5file!")

    if isdir(READ_PATH):
        tempFilename = str(filename) + ".h5"
        tempPath = join(READ_PATH,str(tempFilename))
        hf_read = h5py.File(tempPath, 'r')
        tempData = np.array(hf_read.get(filename))

        return tempData

    else:
        logging.info("Error in PATH!")

#Reads the h5 file and csv file names windowed_test_data and windowed_test_label
def read_windowed_test_data(READ_PATH):

    windowed_test_data = read_h5file(READ_PATH,'windowed_test_data')
    windowed_test_label = pd.read_csv(join(READ_PATH,"windowed_test_label.csv"))
    
    logging.info("Windowed test data shape: %s", windowed_test_data.shape)
    logging.info("Windowed test label shape: %s", windowed_test_label.shape)

    return windowed_test_data, windowed_test_label

#Reads the h5 file and csv file names windowed_train_data and windowed_train_label
def read_windowed_train_data(READ_PATH):

    windowed_train_data = read_h5file(READ_PATH,'windowed_train_data')
    windowed_train_label = pd.read_csv(join(READ_PATH,"windowed_train_label.csv"))
    
    logging.info("Windowed train data shape: %s", windowed_train_data.shape)
    logging.info("Windowed train label shape: %s", windowed_train_label.shape)

    return windowed_train_data, windowed_train_label


### FEATURE SELECTION
def get_p_values_per_data(target_anomalous_features,target_healthy_features):
    #target_anomalous_features, _ = data_object.train_data(anomalous_features)
    #target_healthy_features, _ = data_object.train_data(healthy_features)
    if len(target_anomalous_features) == 0 or \
            len(target_healthy_features) == 0:
        logging.warn('Make sure that the excluded item is an application')
        return pd.Series([1] * len(healthy_features.columns),
                         healthy_features.columns, name='feature')
    
    p_values = [None] * len(target_healthy_features.columns)
    for f_idx, feature in enumerate(target_healthy_features.columns):
        p_values[f_idx] = ks_2samp(target_anomalous_features[feature],
                                   target_healthy_features[feature])[1]
    p_values_series = pd.Series(p_values, target_healthy_features.columns,
                                name='feature')
    return p_values_series

def benjamini_hochberg(p_values_df, apps, anomalies, fdr_level):
    n_features = len(p_values_df)
    selected_features = set()
    for app in apps:
        for anomaly in anomalies:
            col_name = '{}_{}'.format(app, anomaly)
            target_col = p_values_df[col_name].sort_values()
            K = list(range(1, n_features + 1))
            # Calculate the weight vector C
            weights = [sum([1 / i for i in range(1, k + 1)]) for k in K]
            # Calculate the vector T to compare to the p_value
            T = [fdr_level * k / n_features * 1 / w
                 for k, w in zip(K, weights)]
            # select
            selected_features |= set(target_col[target_col <= T].index)
    return selected_features
        