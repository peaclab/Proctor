#!/usr/bin/env python
# coding: utf-8


from abc import ABC, abstractmethod
from pathlib import Path
import os

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np

import json 
import logging, sys
logging.basicConfig(format='%(asctime)s %(levelname)-7s %(message)s',
                    stream=sys.stderr, level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

#Python files
from config import Configuration
from utils import *



class BaseDataset(ABC):
    """Anomaly detection dataset base class."""
    
    def __init__(self):
        logging.info("BaseDataset Class Initialization")        
        
        super().__init__()

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.anom_classes = None  # tuple with original class labels that define the outlier class

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    @abstractmethod
    def load_dataset(self):
        pass
        
    def __repr__(self):
        return self.__class__.__name__        

class ODDSDataset(BaseDataset):
    """
        ODDSDataset class for datasets from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/
        Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
        to also return the semi-supervised target as well as the index of a data sample.
        
        dataset names: 
            arrhythmia.mat  cardio.mat  satellite.mat  satimage-2.mat  shuttle.mat  thyroid.mat        
    """
    
    def __init__(self, root: str, dataset_name: str, random_state=None):
        super(BaseDataset, self).__init__()
        
        self.root = Path(root)        
        self.classes = [0,1]
        
        self.dataset_name = dataset_name
        self.file_name = self.dataset_name + '.mat'
        self.data_file = self.root / self.file_name
        self.random_state = random_state
        
    def load_dataset(self):
        
        mat = loadmat(self.data_file)
        X = mat['X']
        y = mat['y'].ravel()
        idx_norm = y == 0
        idx_out = y == 1

        # 60% data for training and 40% for testing; keep outlier ratio
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X[idx_norm], y[idx_norm],
                                                                                test_size=0.4,
                                                                                random_state=self.random_state)
        X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X[idx_out], y[idx_out],
                                                                            test_size=0.4,
                                                                            random_state=self.random_state)
        X_train = np.concatenate((X_train_norm, X_train_out))
        X_test = np.concatenate((X_test_norm, X_test_out))
        
        self.y_train = np.concatenate((y_train_norm, y_train_out))
        self.y_test = np.concatenate((y_test_norm, y_test_out))
        
        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        self.X_train = minmax_scaler.transform(X_train_stand)        
        self.X_test = minmax_scaler.transform(X_test_stand)
        
        return self.X_train, self.y_train, self.X_test, self.y_test    

class HPCDataset(BaseDataset):
        
    def __init__(self, conf):
        super().__init__()      
        logging.info("HPCDataset Class Initialization")
        self.conf = conf
        
    def read_label(self,TRAIN_DATA=True):
        """Read train or test label"""

        if TRAIN_DATA:
            raw_labels = pd.read_hdf(self.conf['hdf_data_path'] / 'train_label.hdf')
        else:
            raw_labels = pd.read_hdf(self.conf['hdf_data_path'] / 'test_label.hdf')

        if self.conf['system'] == 'volta':
            raw_labels = raw_labels[raw_labels['anom'] != 'linkclog']

        elif self.conf['system'] == 'eclipse':
            raw_labels = raw_labels.rename(columns={'appname':'app','anomaly':'anom'})
            raw_labels = raw_labels[raw_labels['anom'] != 'iometadata']    

        return raw_labels   
    
    def prepare_labels(self):
        
        """Prepares labels for each CV set"""        
        #Common encoder for train and test labels 
        encoder = MyEncoder()
        train_labels = self.read_label(TRAIN_DATA=True)    
        train_labels = encoder.fit_transform(train_labels,dataset='hpas')

        test_labels = self.read_label(TRAIN_DATA=False)    
        test_labels = encoder.transform(test_labels) 

        anom_dict = encoder.anom_dict
        app_dict = encoder.app_dict        
                
        if not (self.conf['experiment_dir'] / ('anom_dict.json')).exists():    

            json_dump = json.dumps(anom_dict)
            f_json = open(self.conf['experiment_dir'] / "anom_dict.json","w")
            f_json.write(json_dump)
            f_json.close()         

            json_dump = json.dumps(app_dict)
            f_json = open(self.conf['experiment_dir'] / "app_dict.json","w")
            f_json.write(json_dump)
            f_json.close()
        else:
            logging.info("Anom and app dict already exists")         
            
            
        all_labels = pd.concat([train_labels, test_labels])

        if self.conf['system'] == 'eclipse':
            normal_label = all_labels[all_labels['anom'] == anom_dict['None']]
            anom_label = all_labels[all_labels['anom'] != anom_dict['None']]
        elif self.conf['system'] == 'volta':
            normal_label = all_labels[all_labels['anom'] == anom_dict['none']]
            anom_label = all_labels[all_labels['anom'] != anom_dict['none']]               
               
        
        skf = StratifiedKFold(n_splits=self.conf['num_split'],random_state=30,shuffle=True)

        n_nodeids = all_labels.shape[0]
        cv_index = 0
        logging.info("Generating labels for CV folders")
        for train_index, test_index in skf.split(np.zeros(n_nodeids),all_labels['anom']):

            logging.info("Cross Validation Number :%d",cv_index)
            cv_path = self.conf['experiment_dir'] / ("CV_" + str(cv_index))
            if not cv_path.exists():
                cv_path.mkdir(parents=True)
                
            train_label = all_labels.iloc[train_index]
            test_label = all_labels.iloc[test_index]
            finetune_label = train_label
        
            logging.info("\nTrain label dist \n%s\n", train_label['anom'].value_counts())
            logging.info("\nTest label dist \n%s\n", test_label['anom'].value_counts())
            logging.info("\nFinetune label dist \n%s\n", finetune_label['anom'].value_counts())     
            
            train_label.to_csv(cv_path / 'train_label.csv')
            test_label.to_csv(cv_path / 'test_label.csv')   
            finetune_label.to_csv(cv_path / 'finetune_label.csv')               
                                    
            cv_index += 1            
    
    @abstractmethod
    def prepare_data(self):
        pass
        
        
        
    
class HPCSampledDataset(BaseDataset):
        
    def __init__(self, conf):
        super().__init__()      
        logging.info("HPCDataset Class Initialization")
        self.conf = conf
        
    def read_label(self,TRAIN_DATA=True):
        """Read train or test label"""

        if TRAIN_DATA:
            raw_labels = pd.read_csv(self.conf['hdf_data_path'] / 'normal_labels.csv',index_col = ['node_id'])
        else:
            raw_labels = pd.read_csv(self.conf['hdf_data_path'] / 'anomaly_labels.csv',index_col = ['node_id'])

        if self.conf['system'] == 'volta':
            raw_labels = raw_labels[raw_labels['anom'] != 'linkclog']

        elif self.conf['system'] == 'eclipse':
            raw_labels = raw_labels.rename(columns={'appname':'app','anomaly':'anom'})
            raw_labels = raw_labels[raw_labels['anom'] != 'iometadata']    

        return raw_labels   
    
    def granularity_adjust(self,data,granularity=60):

        result = pd.DataFrame()
        for nid in data.index.get_level_values('node_id').unique():
            temp_data = data[data.index.get_level_values('node_id') == nid]
            temp_data = temp_data.iloc[ \
                (temp_data.index.get_level_values('timestamp').astype(int) -
                 int(temp_data.index.get_level_values('timestamp')[0])) \
                % granularity == 0]
            result = pd.concat([result,temp_data])

        return result    
    
    def prepare_borghesi(self,data,label):
        
            new_data = self.granularity_adjust(data,300)
            new_data.reset_index(level=1,inplace=True)
            new_data.drop(columns=['timestamp'],inplace=True)   
            
            #Multiply label names according to the new duplicated indices
            new_label = pd.DataFrame(columns=['node_id','app','anom'])
            for node_id in new_data.index:
                new_label = new_label.append({'node_id': node_id,
                                              'app': label.loc[node_id]['app'],
                                              'anom':label.loc[node_id]['anom']},ignore_index=True)

            new_label.set_index('node_id',inplace=True)  
            
            return new_data, new_label
    
    def prepare_labels(self):
        
        """Prepares labels for each CV set"""        
        #Common encoder for train and test labels 
        encoder = MyEncoder()

        normal_labels = self.read_label(TRAIN_DATA=True)   
        anomaly_labels = self.read_label(TRAIN_DATA=False)
        
        if self.conf['system'] == 'eclipse':
            normal_labels.drop(normal_labels[normal_labels['app'] == 'miniAMR'].index,inplace=True)        
            anomaly_labels.drop(anomaly_labels[anomaly_labels['app'] == 'miniAMR'].index,inplace=True)
        
        all_labels = pd.concat([normal_labels, anomaly_labels])
        all_labels = encoder.fit_transform(all_labels,dataset='hpas')
        
        normal_labels = encoder.transform(normal_labels)         
        anomaly_labels = encoder.transform(anomaly_labels) 
        
        anom_dict = encoder.anom_dict
        app_dict = encoder.app_dict        
                
        if not (self.conf['experiment_dir'] / ('anom_dict.json')).exists():    

            json_dump = json.dumps(anom_dict)
            f_json = open(self.conf['experiment_dir'] / "anom_dict.json","w")
            f_json.write(json_dump)
            f_json.close()         

            json_dump = json.dumps(app_dict)
            f_json = open(self.conf['experiment_dir'] / "app_dict.json","w")
            f_json.write(json_dump)
            f_json.close()
        else:
            logging.info("Anom and app dict already exists")         
                                  
                
        #return normal_labels, anomaly_labels
    
        for cv_index in range(self.conf['num_split']):
            
            logging.info("CV fold %s",cv_index)
            cv_path = self.conf['experiment_dir'] / ("CV_" + str(cv_index))
            
            if not cv_path.exists():
                cv_path.mkdir(parents=True)                                                
                  
            ANOM_RATIO = 0.1    
            
            if self.conf['system'] == 'eclipse':
                
#                 test_normal_label, train_normal_label = train_test_split(normal_labels,test_size = 0.4,stratify=normal_labels[['app','anom']])
                test_normal_label, train_normal_label = train_test_split(normal_labels,test_size = 0.4,stratify=normal_labels[['app','anom']],random_state=1234)                
                train_total_anom  = int(ANOM_RATIO*len(train_normal_label) / (1 - ANOM_RATIO)) 
                print(train_total_anom)
                train_ratio_anom = train_total_anom / len(anomaly_labels)  
                
                test_anom_label, train_anom_label = train_test_split(anomaly_labels,test_size = train_ratio_anom,stratify=anomaly_labels[['app','anom']],random_state=1234)                 


            elif self.conf['system'] == 'volta':
                #Determine normal label division    
                test_normal_label, train_normal_label = train_test_split(normal_labels,test_size = 0.3,stratify=normal_labels[['app','anom']])                
                train_total_anom  = int(ANOM_RATIO*len(train_normal_label) / (1 - ANOM_RATIO)) 
                train_ratio_anom = train_total_anom / len(anomaly_labels)

                test_anom_label, train_anom_label = train_test_split(anomaly_labels,test_size = train_ratio_anom,stratify=anomaly_labels[['app','anom']])                
             
                                                                                               
            train_label = pd.concat([train_normal_label,train_anom_label])
            test_label = pd.concat([test_normal_label,test_anom_label])
                         
            logging.info("Train data class dist \n%s\n",train_label['anom'].value_counts())    
            logging.info("Train data app dist \n%s\n",train_label['app'].value_counts())                
            logging.info("Test data class dist \n%s\n",test_label['anom'].value_counts())                 
            logging.info("Test data app dist \n%s\n",test_label['app'].value_counts())                 

            train_label.to_csv(cv_path / 'train_label.csv')
            test_label.to_csv(cv_path / 'test_label.csv') 
            
               
    @abstractmethod
    def prepare_data(self):
        pass    

class EclipseSampledDataset(HPCSampledDataset):
    """
        EclipseSampledDataset class for datasets for Eclipse HPC monitoring data
        This class is designed to experiment with equal number of instances in each anomaly and separate normal labels
        Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
        to also return the semi-supervised target as well as the index of a data sample.
    """
    
    def __init__(self, conf, borghesi=False):
        super().__init__(conf)
        logging.info("EclipseSampledDataset Class Initialization")        
        
        self.normal_class = [0]
        self.anom_classes = [1,2,3,4]
        self.classes = [0,1,2,3,4]
        self.borghesi = borghesi
                                                     
    def prepare_data(self):
        
        """Prepares data according to the labels"""
                
        #These two code blocks read previously saved train and test data - do NOT confuse
        anomaly_data = pd.read_hdf(self.conf['hdf_data_path'] / 'anomaly_data.hdf','anomaly_data')
        anomaly_data = anomaly_data[[x for x in anomaly_data.columns if 'per_core' not in x]]
        logging.info("Anomaly data shape: %s",anomaly_data.shape)

        normal_data = pd.read_hdf(self.conf['hdf_data_path'] / 'normal_data.hdf','normal_data')
        normal_data = normal_data[[x for x in normal_data.columns if 'per_core' not in x]]
        logging.info("Normal data shape: %s",normal_data.shape)

        all_data = pd.concat([normal_data,anomaly_data])
        logging.info("Full data shape: %s",all_data.shape)

        all_data = all_data.dropna()
        logging.info("Is NaN: %s",np.any(np.isnan(all_data)))
        logging.info("Data shape: %s",all_data.shape)

        CV_NUM_STR = ("CV_" + str(self.conf['cv_fold']))
        
        train_label = pd.read_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'train_label.csv',index_col=['node_id'])
        train_data = all_data[all_data.index.get_level_values('node_id').isin(train_label.index)]
        logging.info("Train data shape %s",train_data.shape)  
        logging.info("Train label shape %s",train_label.shape)          

        
        test_label = pd.read_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'test_label.csv',index_col=['node_id'])
        test_data = all_data[all_data.index.get_level_values('node_id').isin(test_label.index)]
        logging.info("Test data shape %s",test_data.shape)    
        logging.info("Test label shape %s",test_label.shape)                  
        
        logging.info("Train data label dist: \n%s",train_label['anom'].value_counts())
        logging.info("Test data label dist: \n%s",test_label['anom'].value_counts())            

                
        if self.conf['feature_select']:
            cache_path = self.conf['experiment_dir'] / '{}_feature_p_values.hdf'.format(self.conf['system'])
            all_labels = pd.concat([train_label,test_label])            
            apps = set(all_labels['app'].unique())
            anomalies = self.anom_classes
            
            if cache_path.exists():
                logging.info('Retrieving feature p-values')
                p_values_df = pd.read_hdf(cache_path)
            else:    
                
                logging.info('Calculating feature p-values')
                all_columns = train_data.columns
                all_labels = pd.concat([train_label,test_label])
                                
                p_values_df = pd.DataFrame()
                pbar = tqdm(total=len(apps)*len(anomalies))

                for app in apps:
                    n_anomalous_runs = len(all_labels[all_labels['app'] == app][all_labels['anom'] != self.normal_class[0]])

                    healthy_node_ids = set(list(all_labels[all_labels['app'] == app][all_labels['anom'] == self.normal_class[0]].index))
                    temp_node_data = all_data[all_data.index.get_level_values('node_id').isin(healthy_node_ids)]

                    
                    feature_generator = TSFeatureGenerator(trim=30)
                    healthy_features = feature_generator.transform(temp_node_data)

                    for anomaly in anomalies:
                        col_name = '{}_{}'.format(app, anomaly)
                        anomalous_node_ids = set(list(all_labels[all_labels['app'] == app][all_labels['anom'] == anomaly].index))
                        temp_node_data = all_data[all_data.index.get_level_values('node_id').isin(anomalous_node_ids)]

                        anomalous_features = feature_generator.transform(temp_node_data)

                        p_values_df[col_name] = get_p_values_per_data(anomalous_features,healthy_features)

                        pbar.update(1)   

                p_values_df.to_hdf(cache_path,key='key')
            fdr_level = 0.01
            selected_features = benjamini_hochberg(p_values_df, apps, anomalies, fdr_level)
            pd.DataFrame(selected_features).to_csv(self.conf['experiment_dir'] / 'selected_features.csv')
            logging.info('Selected %d features', len(selected_features))
        else:
            logging.info("No feature selection")

        if self.borghesi:       
            borghesi_data, borghesi_label = self.prepare_borghesi(train_data,train_label)
            borghesi_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR /'train_data_borghesi.hdf',key='train_data_borghesi',complevel=9)            
            borghesi_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR /'train_label_borghesi.csv')
            
            borghesi_data, borghesi_label = self.prepare_borghesi(test_data,test_label)
            borghesi_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR /'test_data_borghesi.hdf',key='test_data_borghesi',complevel=9)            
            borghesi_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR /'test_label_borghesi.csv')

            del borghesi_data, borghesi_label
              
           
        
        if self.conf['feature_extract']:
            #FIXME: It might need an update for TPDS data 
            logging.info("Generating features")    
            feature_generator = TSFeatureGenerator(trim=0) #Don't change the trim
            
            train_data = feature_generator.transform(train_data)
            test_data = feature_generator.transform(test_data)
            
                                     
        ### Save data as hdf
        logging.info("Saving training data")
        train_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'train_data.hdf',key='train_data',complevel=9)
        
        train_label = train_label.loc[train_data.index]
        train_label.index.name = 'node_id'        
        train_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'train_label.csv')


        logging.info("Saving test data")
        test_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'test_data.hdf',key='test_data',complevel=9)
        
        test_label = test_label.loc[test_data.index]
        test_label.index.name = 'node_id'      
        test_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'test_label.csv')                 
        
        logging.info("Train data shape %s",train_data.shape)
        logging.info("Train label shape %s",train_label.shape)        
        logging.info("Test data shape %s",test_data.shape)    
        logging.info("Test label shape %s",test_label.shape)          
           
        logging.info("Saved data and labels\n")
        logging.info("Train data label dist: \n%s",train_label['anom'].value_counts())
        logging.info("Test data label dist: \n%s",test_label['anom'].value_counts())              
        
        
         
        
    def load_dataset(self,scaler='MinMax',time=False,borghesi=False): 
           
        if not time:
            
            CV_NUM_STR = ("CV_" + str(self.conf['cv_fold'])) 
            DATA_PATH = self.conf['experiment_dir'] / CV_NUM_STR
            
            ###Read training data
            if borghesi:
                X_train = pd.read_hdf(DATA_PATH / 'train_data_borghesi.hdf',key='train_data_borghesi')
                self.y_train = pd.read_csv(DATA_PATH / 'train_label_borghesi.csv',index_col=['node_id'])
                
            else:
                X_train = pd.read_hdf(DATA_PATH / 'train_data.hdf',key='train_data')
                self.y_train = pd.read_csv(DATA_PATH / 'train_label.csv',index_col=['node_id'])
                self.y_train = self.y_train.loc[X_train.index]
                self.y_train.index.name = 'node_id'        

            ###Read test data     
            if borghesi:
                X_test = pd.read_hdf(DATA_PATH / 'test_data_borghesi.hdf',key='test_data_borghesi')
                self.y_test = pd.read_csv(DATA_PATH / 'test_label_borghesi.csv',index_col=['node_id'])
                        
            else:
                X_test = pd.read_hdf(DATA_PATH / 'test_data.hdf',key='test_data')
                self.y_test = pd.read_csv(DATA_PATH / 'test_label.csv',index_col=['node_id'])
                self.y_test = self.y_test.loc[X_test.index]
                self.y_test.index.name = 'node_id'         


            logging.info("Train data shape %s",X_train.shape)
            logging.info("Train label shape %s",self.y_train.shape)

            logging.info("Test data shape %s",X_test.shape)
            logging.info("Test label shape %s",self.y_test.shape)        

            with open(self.conf['experiment_dir'] / ('anom_dict.json')) as f:
                ANOM_DICT = json.load(f)        

            if scaler == 'Standard':

                # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)        
                scaler = StandardScaler().fit(X_train)
                self.X_train = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
                self.X_test = pd.DataFrame(scaler.transform(X_test),columns=X_train.columns,index=X_test.index)

            elif scaler == 'MinMax':
                # Scale to range [0,1]
                minmax_scaler = MinMaxScaler().fit(X_train)
                self.X_train = pd.DataFrame(minmax_scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
                self.X_test = pd.DataFrame(minmax_scaler.transform(X_test),columns=X_test.columns,index=X_test.index)


            return self.X_train, self.y_train, self.X_test, self.y_test        

        else:
            
            CV_NUM_STR = ("CV_" + str(self.conf['cv_fold'])) 
            DATA_PATH = self.conf['experiment_dir'] / CV_NUM_STR

            ###Read training data
            self.X_train = pd.read_hdf(DATA_PATH / 'train_data.hdf',key='train_data')

            self.y_train = pd.read_csv(DATA_PATH / 'train_label.csv',index_col=['node_id'])            
            
            
            ###Read test data            
            self.X_test = pd.read_hdf(DATA_PATH / 'test_data.hdf',key='test_data')

            self.y_test = pd.read_csv(DATA_PATH / 'test_label.csv',index_col=['node_id'])     
            
            
            logging.info("Train data shape %s",self.X_train.shape)
            logging.info("Train label shape %s",self.y_train.shape)

            logging.info("Test data shape %s",self.X_test.shape)
            logging.info("Test label shape %s",self.y_test.shape)               
            
            
            return self.X_train, self.y_train, self.X_test, self.y_test 
        
        
class VoltaSampledDataset(HPCSampledDataset):
    """
        VoltaSampledDataset class for datasets for Volta HPC monitoring data
        This class is designed to experiment with equal number of instances in each anomaly and separate normal labels
        Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
        to also return the semi-supervised target as well as the index of a data sample.
    """
    
    def __init__(self, conf):
        super().__init__(conf)
        logging.info("VoltaSampledDataset Class Initialization")        
        
        self.normal_class = [0]
        self.anom_classes = [1,2,3,4]
        self.classes = [0,1,2,3,4]
                                                     
    def prepare_data(self):
        
        """Prepares data according to the labels"""
                
        #These two code blocks read previously saved train and test data - do NOT confuse
        anomaly_data = pd.read_hdf(self.conf['hdf_data_path'] / 'anomaly_data.hdf','anomaly_data')
        anomaly_data = anomaly_data[[x for x in anomaly_data.columns if 'per_core' not in x]]
        logging.info("Anomaly data shape: %s",anomaly_data.shape)

        normal_data = pd.read_hdf(self.conf['hdf_data_path'] / 'normal_data.hdf','normal_data')
        normal_data = normal_data[[x for x in normal_data.columns if 'per_core' not in x]]
        logging.info("Normal data shape: %s",normal_data.shape)

        all_data = pd.concat([normal_data,anomaly_data])
        logging.info("Full data shape: %s",all_data.shape)

        all_data = all_data.dropna()
        logging.info("Is NaN: %s",np.any(np.isnan(all_data)))
        logging.info("Data shape: %s",all_data.shape)

        CV_NUM_STR = ("CV_" + str(self.conf['cv_fold']))
        
        train_label = pd.read_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'train_label.csv',index_col=['node_id'])
        train_data = all_data[all_data.index.get_level_values('node_id').isin(train_label.index)]
        logging.info("Train data shape %s",train_data.shape)  
        logging.info("Train label shape %s",train_label.shape)   
        
        test_label = pd.read_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'test_label.csv',index_col=['node_id'])
        test_data = all_data[all_data.index.get_level_values('node_id').isin(test_label.index)]
        logging.info("Test data shape %s",test_data.shape)    
        logging.info("Test label shape %s",test_label.shape)  
        
        logging.info("Train data label dist: \n%s",train_label['anom'].value_counts())
        logging.info("Test data label dist: \n%s",test_label['anom'].value_counts())            
                        
        if self.conf['feature_select']:
            cache_path = self.conf['experiment_dir'] / '{}_feature_p_values.hdf'.format(self.conf['system'])
            all_labels = pd.concat([train_label,test_label])            
            apps = set(all_labels['app'].unique())
            anomalies = self.anom_classes
            
            if cache_path.exists():
                logging.info('Retrieving feature p-values')
                p_values_df = pd.read_hdf(cache_path)
            else:    
                
                logging.info('Calculating feature p-values')
                all_columns = train_data.columns
                all_labels = pd.concat([train_label,test_label])
                                
                p_values_df = pd.DataFrame()
                pbar = tqdm(total=len(apps)*len(anomalies))

                for app in apps:
                    n_anomalous_runs = len(all_labels[all_labels['app'] == app][all_labels['anom'] != self.normal_class[0]])

                    healthy_node_ids = set(list(all_labels[all_labels['app'] == app][all_labels['anom'] == self.normal_class[0]].index))
                    temp_node_data = all_data[all_data.index.get_level_values('node_id').isin(healthy_node_ids)]

                    
                    feature_generator = TSFeatureGenerator(trim=30)
                    healthy_features = feature_generator.transform(temp_node_data)

                    for anomaly in anomalies:
                        col_name = '{}_{}'.format(app, anomaly)
                        anomalous_node_ids = set(list(all_labels[all_labels['app'] == app][all_labels['anom'] == anomaly].index))
                        temp_node_data = all_data[all_data.index.get_level_values('node_id').isin(anomalous_node_ids)]

                        anomalous_features = feature_generator.transform(temp_node_data)

                        p_values_df[col_name] = get_p_values_per_data(anomalous_features,healthy_features)

                        pbar.update(1)   

                p_values_df.to_hdf(cache_path,key='key')
            fdr_level = 0.01
            selected_features = benjamini_hochberg(p_values_df, apps, anomalies, fdr_level)
            pd.DataFrame(selected_features).to_csv(self.conf['experiment_dir'] / 'selected_features.csv')
            logging.info('Selected %d features', len(selected_features))
            
            logging.info('Selected %d features', len(selected_features))
        else:
            logging.info("No feature selection")
                                
        
        
        if self.conf['feature_extract']:
            #FIXME: It might need an update for TPDS data 
            logging.info("Generating features")    
            feature_generator = TSFeatureGenerator(trim=0) #Don't change the trim
            
            train_data = feature_generator.transform(train_data)
            test_data = feature_generator.transform(test_data)
            
        ### Save data as hdf
        logging.info("Saving training data")
        train_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'train_data.hdf',key='train_data',complevel=9)
        
        train_label = train_label.loc[train_data.index]
        train_label.index.name = 'node_id'        
        train_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'train_label.csv')


        logging.info("Saving test data")
        test_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'test_data.hdf',key='test_data',complevel=9)
        
        test_label = test_label.loc[test_data.index]
        test_label.index.name = 'node_id'      
        test_label.to_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'test_label.csv')                 
        
        logging.info("Train data shape %s",train_data.shape)
        logging.info("Train label shape %s",train_label.shape)        
        logging.info("Test data shape %s",test_data.shape)    
        logging.info("Test label shape %s",test_label.shape)          
           
        logging.info("Saved data and labels\n")
        logging.info("Train data label dist: \n%s",train_label['anom'].value_counts())
        logging.info("Test data label dist: \n%s",test_label['anom'].value_counts())              
            
        
        
    def load_dataset(self,scaler='MinMax',time=False,borghesi=False): 
           
        if not time:
            
            CV_NUM_STR = ("CV_" + str(self.conf['cv_fold'])) 
            DATA_PATH = self.conf['experiment_dir'] / CV_NUM_STR

            ###Read training data
            if borghesi:
                X_train = pd.read_hdf(DATA_PATH / 'train_data_borghesi.hdf',key='train_data_borghesi')
                self.y_train = pd.read_csv(DATA_PATH / 'train_label_borghesi.csv',index_col=['node_id'])
                self.y_train = self.y_train.loc[X_train.index]
                self.y_train.index.name = 'node_id'        
                
            else:
                X_train = pd.read_hdf(DATA_PATH / 'train_data.hdf',key='train_data')
                self.y_train = pd.read_csv(DATA_PATH / 'train_label.csv',index_col=['node_id'])
                self.y_train = self.y_train.loc[X_train.index]
                self.y_train.index.name = 'node_id'        

            ###Read test data     
            if borghesi:
                X_train = pd.read_hdf(DATA_PATH / 'test_data_borghesi.hdf',key='test_data_borghesi')
                self.y_train = pd.read_csv(DATA_PATH / 'test_label_borghesi.csv',index_col=['node_id'])
                self.y_train = self.y_train.loc[X_train.index]
                self.y_train.index.name = 'node_id'        
                        
            else:
                X_test = pd.read_hdf(DATA_PATH / 'test_data.hdf',key='test_data')
                self.y_test = pd.read_csv(DATA_PATH / 'test_label.csv',index_col=['node_id'])
                self.y_test = self.y_test.loc[X_test.index]
                self.y_test.index.name = 'node_id'        


            logging.info("Train data shape %s",X_train.shape)
            logging.info("Train label shape %s",self.y_train.shape)

            logging.info("Test data shape %s",X_test.shape)
            logging.info("Test label shape %s",self.y_test.shape)        

            with open(self.conf['experiment_dir'] / ('anom_dict.json')) as f:
                ANOM_DICT = json.load(f)        

            if scaler == 'Standard':

                # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)        
                scaler = StandardScaler().fit(X_train)
                self.X_train = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
                self.X_test = pd.DataFrame(scaler.transform(X_test),columns=X_train.columns,index=X_test.index)

            elif scaler == 'MinMax':
                # Scale to range [0,1]
                minmax_scaler = MinMaxScaler().fit(X_train)
                self.X_train = pd.DataFrame(minmax_scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
                self.X_test = pd.DataFrame(minmax_scaler.transform(X_test),columns=X_test.columns,index=X_test.index)


            return self.X_train, self.y_train, self.X_test, self.y_test        

        else:
            
            CV_NUM_STR = ("CV_" + str(self.conf['cv_fold'])) 
            DATA_PATH = self.conf['experiment_dir'] / CV_NUM_STR

            ###Read training data
            self.X_train = pd.read_hdf(DATA_PATH / 'train_data.hdf',key='train_data')

            self.y_train = pd.read_csv(DATA_PATH / 'train_label.csv',index_col=['node_id'])            
            
            
            ###Read test data            
            self.X_test = pd.read_hdf(DATA_PATH / 'test_data.hdf',key='test_data')

            self.y_test = pd.read_csv(DATA_PATH / 'test_label.csv',index_col=['node_id'])     
            
            
            logging.info("Train data shape %s",self.X_train.shape)
            logging.info("Train label shape %s",self.y_train.shape)

            logging.info("Test data shape %s",self.X_test.shape)
            logging.info("Test label shape %s",self.y_test.shape)               
            
            
            return self.X_train, self.y_train, self.X_test, self.y_test             
        
        
        
        
# class EclipseSmallDataset(HPCDataset):
#     """
#         EclipseSmallDataset class for datasets for Eclipse HPC monitoring data
#         Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
#         to also return the semi-supervised target as well as the index of a data sample.
#     """
    
#     def __init__(self, conf):
#         super().__init__(conf)
#         logging.info("EclipseSmallDataset Class Initialization")        
        
#         self.normal_class = [0]
#         self.anom_classes = [1,2,3,4]
#         self.classes = [0,1,2,3,4]
                                                     
#     def prepare_data(self):
        
#         """Prepares data according to the labels"""
                
#         #These two code blocks read previously saved train and test data - do NOT confuse
#         train_data = pd.read_hdf(self.conf['hdf_data_path'] / 'train_data.hdf','train_data')
#         train_data = train_data[[x for x in train_data.columns if 'per_core' not in x]]
#         logging.info("Train data shape: %s",train_data.shape)

#         test_data = pd.read_hdf(self.conf['hdf_data_path'] / 'test_data.hdf','test_data')
#         test_data = test_data[[x for x in test_data.columns if 'per_core' not in x]]
#         logging.info("Test data shape: %s",test_data.shape)

#         train_data_full = pd.concat([train_data,test_data])
#         logging.info("Full data shape: %s",train_data_full.shape)    

#         train_data_full = train_data_full.dropna()
#         logging.info("Is NaN: %s",np.any(np.isnan(train_data_full)))
#         logging.info("Data shape: %s",train_data_full.shape)

#         CV_NUM_STR = ("CV_" + str(self.conf['cv_fold']))
        
#         train_label = pd.read_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'train_label.csv',index_col=['node_id'])
#         test_label = pd.read_csv(self.conf['experiment_dir'] / CV_NUM_STR / 'test_label.csv',index_col=['node_id'])
#         finetune_label = train_label
                
#         train_data = train_data_full[train_data_full.index.get_level_values('node_id').isin(train_label.index)]
#         logging.info("Train data shape %s",train_data.shape)        
        
#         test_data = train_data_full[train_data_full.index.get_level_values('node_id').isin(test_label.index)]
#         logging.info("test data shape %s",test_data.shape)    

#         finetune_data = train_data
#         logging.info("Finetune data shape %s",finetune_data.shape)        
                
#         if self.conf['feature_select']:
#             cache_path = self.conf['experiment_dir'] / '{}_feature_p_values.hdf'.format(self.conf['system'])
#             all_labels = pd.concat([train_label,test_label])            
#             apps = set(all_labels['app'].unique())
#             anomalies = self.anom_classes
            
#             if cache_path.exists():
#                 logging.info('Retrieving feature p-values')
#                 p_values_df = pd.read_hdf(cache_path)
#             else:    
                
#                 logging.info('Calculating feature p-values')
#                 all_columns = train_data.columns
#                 all_labels = pd.concat([train_label,test_label])
                                
#                 p_values_df = pd.DataFrame()
#                 pbar = tqdm(total=len(apps)*len(anomalies))

#                 for app in apps:
#                     n_anomalous_runs = len(all_labels[all_labels['app'] == app][all_labels['anom'] != self.normal_class[0]])

#                     healthy_node_ids = set(list(all_labels[all_labels['app'] == app][all_labels['anom'] == self.normal_class[0]].index))
#                     temp_node_data = train_data_full[train_data_full.index.get_level_values('node_id').isin(healthy_node_ids)]

                    
#                     feature_generator = TSFeatureGenerator(trim=30)
#                     healthy_features = feature_generator.transform(temp_node_data)

#                     for anomaly in anomalies:
#                         col_name = '{}_{}'.format(app, anomaly)
#                         anomalous_node_ids = set(list(all_labels[all_labels['app'] == app][all_labels['anom'] == anomaly].index))
#                         temp_node_data = train_data_full[train_data_full.index.get_level_values('node_id').isin(anomalous_node_ids)]

#                         anomalous_features = feature_generator.transform(temp_node_data)

#                         p_values_df[col_name] = get_p_values_per_data(anomalous_features,healthy_features)

#                         pbar.update(1)   

#                 p_values_df.to_hdf(cache_path,key='key')
#             fdr_level = 0.01
#             selected_features = benjamini_hochberg(p_values_df, apps, anomalies, fdr_level)
#             logging.info('Selected %d features', len(selected_features))
#         else:
#             logging.info("No feature selection")
                                
        
        
#         if self.conf['feature_extract']:
#             #FIXME: It might need an update for TPDS data 
#             logging.info("Generating features")    
#             feature_generator = TSFeatureGenerator(trim=0) #Don't change the trim
            
#             train_data = feature_generator.transform(train_data)
#             test_data = feature_generator.transform(test_data)
#             finetune_data = train_data   
            
#         if self.conf['feature_select']:
#             train_data = train_data[selected_features] 
#             test_data = test_data[selected_features]       
#             finetune_data = train_data
            
#         logging.info("Train data shape %s",train_data.shape)
#         logging.info("test data shape %s",test_data.shape)    
#         logging.info("Finetune data shape %s",finetune_data.shape)            
                
        
#         ### Save data as hdf
#         logging.info("Saving training data")
#         train_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'train_data.hdf',key='train_data',complevel=9)

#         logging.info("Saving test data")
#         test_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'test_data.hdf',key='test_data',complevel=9)

#         logging.info("Saving finetune data")
#         finetune_data.to_hdf(self.conf['experiment_dir'] / CV_NUM_STR / 'finetune_data.hdf',key='finetune_data',complevel=9)      
        
        
#     def load_dataset(self): 
           
#         CV_NUM_STR = ("CV_" + str(self.conf['cv_fold'])) 
#         DATA_PATH = self.conf['experiment_dir'] / CV_NUM_STR
        
#         ###Read training data
#         X_train = pd.read_hdf(DATA_PATH / 'train_data.hdf',key='train_data')
            
#         self.y_train = pd.read_csv(DATA_PATH / 'train_label.csv',index_col=['node_id'])
#         self.y_train = self.y_train.loc[X_train.index]
        
#         ###Read test data            
#         X_test = pd.read_hdf(DATA_PATH / 'test_data.hdf',key='test_data')
            
#         self.y_test = pd.read_csv(DATA_PATH / 'test_label.csv',index_col=['node_id'])
#         self.y_test = self.y_test.loc[X_test.index]
        
#         logging.info("Train data shape %s",X_train.shape)
#         logging.info("Test data shape %s",X_test.shape)
        
#         with open(self.conf['experiment_dir'] / ('anom_dict.json')) as f:
#             ANOM_DICT = json.load(f)        
            
#         #TODO:when loading dataset call the create_semisupervised_data
        
#         # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)        
#         scaler = StandardScaler().fit(X_train)
#         X_train_stand = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
#         X_test_stand = pd.DataFrame(scaler.transform(X_test),columns=X_train.columns,index=X_test.index)
        
#         # Scale to range [0,1]
#         minmax_scaler = MinMaxScaler().fit(X_train_stand)
#         self.X_train = pd.DataFrame(minmax_scaler.transform(X_train_stand),columns=X_train_stand.columns,index=X_train_stand.index)
#         self.X_test = pd.DataFrame(minmax_scaler.transform(X_test_stand),columns=X_test_stand.columns,index=X_test_stand.index)


#         return self.X_train, self.y_train, self.X_test, self.y_test           