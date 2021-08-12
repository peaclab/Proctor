#!/usr/bin/env python

###GENERIC
import pandas as pd
import numpy as np
import os,sys
from pathlib import Path
import logging
import json
logging.basicConfig(format='%(asctime)s %(levelname)-7s %(message)s',
                    stream=sys.stderr, level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


###SKLEARN
from sklearn.model_selection import train_test_split   


###TENSORFLOW
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import layers,losses

from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Flatten
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Input, ZeroPadding2D, ZeroPadding1D
from tensorflow.keras.layers import GlobalAveragePooling1D, BatchNormalization, UpSampling1D

from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l1,l2,l1_l2
from tensorflow.keras.losses import CategoricalCrossentropy



### Generic Helpers

def trainSupervisedAE(params,model_config,topology,X_train,y_train,verbose=0,freeze=False,stacked=False):
    
    #DL MODELS
    reconstructedModel = LoadModel(model_config,topology)

    if params['optimizer'] == 'adam':
        opt = optimizers.Adam(params['learning_rate'])
    elif params['optimizer'] == 'adadelta':
        opt = optimizers.Adadelta(params['learning_rate'])            
    elif params['optimizer'] == 'sgd':
        opt = optimizers.SGD(params['learning_rate'], momentum=0.9)           

    METRICS = [
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.CategoricalAccuracy(name='acc'),
      tf.keras.metrics.AUC(name='auc'),
    ]        
    
    if freeze:
        logging.info("Freeze active")
        if not stacked:
            for layer in reconstructedModel.layers:
                if layer.name != 'code':
                    layer.trainable = False
                elif layer.name == 'code':
                    layer.trainable = False
                    break
        else:
            for layer in reconstructedModel.layers:
                print(layer.name)
                if layer.name != 'supervised_output':
                    layer.trainable = False
                else:
                    break                        
        
    print(reconstructedModel.summary())                

    reconstructedModel.compile(loss="categorical_crossentropy", 
                       #loss = tf.keras.losses.CosineSimilarity(axis=1),
                       optimizer=opt,
                       metrics=METRICS)

    y_train = to_categorical(y_train)
    model_history = reconstructedModel.fit(                      
                                            X_train,
                                            y_train,
                                            epochs=params['epochs'], 
                                            shuffle=True,
                                            callbacks=[
                                                        #es,
                                            ],            
                                            #validation_split = 0.1,
                                            verbose=verbose)

    #plot_supervised_metrics(model_history)
    
    reconstructedModel.save(model_config['model_dir'] / 'supervised-dae')
    
    return reconstructedModel


def buildBinaryAE(params,model_config,topology,model_name,stacked=False):
    """Classify anomalous vs. normal"""
    
    #It means loading a full autoencoder
    if not stacked:
        reconstructedModel = LoadModel(model_config,topology)
        pretrainedEncoder = reconstructedModel.encoder
        
    else:
        pretrainedEncoder = LoadModel(model_config,topology)
    
    if not params['dense_neuron'] is None:
        pretrainedEncoder.add(Dense(params['dense_neuron'],name="supervised_dense")) 
    pretrainedEncoder.add(Dense(2,activation='softmax',name="supervised_output"))      

    pretrainedEncoder._name = model_name
    print(pretrainedEncoder.summary())
    
    logging.info("Created binary AE")
    
    pretrainedEncoder.save(model_config['model_dir'] / model_name)
    
    return pretrainedEncoder    

def buildMulticlassAE(params,model_config,topology,num_class,model_name,stacked=False):
    """Classify anomalous vs. normal"""
    
    #It means loading a full autoencoder
    if not stacked:
        reconstructedModel = LoadModel(model_config,topology)
        pretrainedEncoder = reconstructedModel.encoder        
    else:
        pretrainedEncoder = LoadModel(model_config,topology)    
        
    if not params['dense_neuron'] is None:
        pretrainedEncoder.add(Dense(params['dense_neuron'],name="supervised_dense")) 
        
    pretrainedEncoder.add(Dense(num_class,activation='softmax',name="supervised_output"))      
    pretrainedEncoder._name = model_name
    print(pretrainedEncoder.summary())
    
    logging.info("Created multiclass AE")
    
    pretrainedEncoder.save(model_config['model_dir'] / model_name)
    
    return pretrainedEncoder    

def prepareFinetuningDataset(percentage,train_data,train_label,system,rs=42):
        
    #Decide normal and anomalous counts
    train_anom_label = train_label[train_label['anom'] != 0]
    num_unique_anoms = len(train_anom_label['anom'].unique())
    train_normal_label = train_label[train_label['anom'] == 0]

    total_anom_count = len(train_anom_label)
    average_class_anom_count = int(total_anom_count / num_unique_anoms)
    total_normal_count = len(train_normal_label)

    total_count = total_anom_count + total_normal_count

    #Find how many instance needed for each class
    num_instance_class = int((total_count * percentage) / (num_unique_anoms + 1))
    print(num_instance_class)
    print(average_class_anom_count)

    _, labeled_normal_labels = train_test_split(train_normal_label, 
                                                test_size = num_instance_class/total_normal_count, 
                                                random_state=42)        
    
    if num_instance_class/average_class_anom_count >= 1:
        labeled_anom_labels = train_anom_label
    else:
        _, labeled_anom_labels = train_test_split(train_anom_label, 
                                                  test_size = num_instance_class/average_class_anom_count, 
                                                  stratify = train_anom_label[['anom']], 
                                                  random_state=rs)
    

    #Prepare semi-supervised data and labels
    train_semisup_label = pd.concat([labeled_normal_labels,labeled_anom_labels])

    #Select semisup_training data
    train_semisup_data = train_data[train_data.index.get_level_values('node_id').isin(train_semisup_label.index)]
    train_semisup_label = train_semisup_label.reindex(train_semisup_data.index.get_level_values('node_id'))
    assert list(train_semisup_label.index) == list(train_semisup_data.index.get_level_values('node_id')) 

    #logging.info("######PERCENTAGE: %s ########",percentage)
    logging.info("\nSemi-supervised labeled data class distributions\n%s\n",train_semisup_label['anom'].value_counts())    

    #If the order is protected, convert data to array format
    train_semisup_data = train_semisup_data.values
    train_semisup_label = train_semisup_label['anom'].values
    
    return train_semisup_data, train_semisup_label



def plot_supervised_metrics(history):

    metrics =  ['loss', 'accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        #plt.plot(history.epoch, history.history['val_'+metric],color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0,1])
        else:
            plt.ylim([0,1])

        plt.legend()  

    plt.show()
    
    
    
def filteredTestingProctor(binary_model,multiclass_model,X_test):
    
    logging.info("Double layer testing started")
    
    test_pred_label = []

    test_pred_label = np.argmax(binary_model.predict(X_test),1)

    pred_as_anom_index = np.where(test_pred_label != 0)
    pred_as_anom_data = X_test[pred_as_anom_index]
    
    pred_as_anom_label = np.argmax(multiclass_model.predict(pred_as_anom_data),1)
        
    test_pred_label[pred_as_anom_index] = pred_as_anom_label        
    
    
    return test_pred_label

def filteredTestingProctorScikit(binary_model,multiclass_model,X_test):
    
    logging.info("Double layer testing started")
    
    test_pred_label = []

    test_pred_label = binary_model.predict(X_test)

    pred_as_anom_index = np.where(test_pred_label != 0)
    pred_as_anom_data = X_test[pred_as_anom_index]
    
    pred_as_anom_label = multiclass_model.predict(pred_as_anom_data)
        
    test_pred_label[pred_as_anom_index] = pred_as_anom_label        
    
    
    return test_pred_label    


###Model Loaders


def LoadModel(config,model_name):
    """Loads model with all the weights and necessary architecture"""
    logging.info("Loading model!")

    
    loaded_model = tf.keras.models.load_model(str(config['model_dir'] / (model_name)))

    return loaded_model 

def LoadEncoder(config,model_name):
    """Loads model with all the weights and necessary architecture"""
    logging.info("Loading encoder model!")
    
    loaded_model = tf.keras.models.load_model(str(config['model_dir'] / (model_name + '_encoder')))

    return loaded_model 

def LoadModelAndWeights(config,model_name):
    """Loads model and set model weights saved by Checkpoint callback"""
    logging.info("Loading model with checkpoint weights!")

    loaded_model = tf.keras.models.load_model(str(config['model_dir'] / (model_name)))
    loaded_model.load_weights(config['model_dir'] / (model_name + "_weights.h5"))

    return loaded_model 

###Evaluation

def supervisedTPDSEvaluation(train_data, train_label, test_data, test_label, conf, anom_ratio, cv_index, name, plot_cm=False):
    ######################## RAW DATA ########################
        
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier    

    clf = Pipeline([
        ('clf', RandomForestClassifier(n_estimators=100))
    ])

    clf.fit(train_data, train_label)
    pipelineAnalysis(clf,
                     train_data,train_label,
                     test_data, test_label,
                     conf=conf,
                     cv_index=cv_index,
                     size=anom_ratio,
                     save_name=name,
                     name_cm = name,
                     plot_cm = plot_cm)     

def supervisedEvaluation(encoder, train_data, train_label, test_data, test_label, conf, label_ratio, cv_index, plot_cm=False,name_suffix=''):
    
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier     
    from sklearn import svm
            
    #stacked_encoder.summary() 
    hidden_train = encoder.predict(train_data)
    hidden_test = encoder.predict(test_data)            
            
    ######################## ENCODER OUTPUT ########################    
    from sklearn.multiclass import OneVsRestClassifier
    
    #Aksar LR 
#     from sklearn.linear_model import LogisticRegression

#     clf = LogisticRegression(random_state=0,max_iter=10000,multi_class='ovr')

#     clf.fit(hidden_train, train_label)
#     pipelineAnalysis(clf,
#                      hidden_train,train_label,
#                      hidden_test, test_label,
#                      conf=conf,
#                      cv_index=cv_index,
#                      size=label_ratio,
#                      save_name='aksar_lr' + name_suffix,
#                      name_cm = 'aksar_lr' + name_suffix,
#                      plot_cm = plot_cm)           
         
            
#     #Aksar SVM - OVR    

#     clf = svm.SVC(kernel='rbf')

#     clf.fit(hidden_train, train_label)
#     pipelineAnalysis(clf,
#                      hidden_train,train_label,
#                      hidden_test, test_label,
#                      conf=conf,
#                      cv_index=cv_index,
#                      size=label_ratio,
#                      save_name='aksar_svm-OVR'+ name_suffix,
#                      name_cm = 'aksar_svm-OVR'+ name_suffix,
#                      plot_cm = plot_cm)        
    
    
    #Aksar LINEAR SVM - OVR    
    clf = svm.LinearSVC(max_iter=5000)
    
    clf.fit(hidden_train, train_label)
    pipelineAnalysis(clf,
                     hidden_train,train_label,
                     hidden_test, test_label,
                     conf=conf,
                     cv_index=cv_index,
                     size=label_ratio,
                     save_name='aksar_l-svm-OVR'+ name_suffix,
                     name_cm = 'aksar_l-svm-OVR'+ name_suffix,
                     plot_cm = plot_cm) 
    
#     #Aksar LINEAR SVM - OVR    
#     clf = svm.LinearSVC(max_iter=5000,
#                        loss = 'hinge')
    
#     clf.fit(hidden_train, train_label)
#     pipelineAnalysis(clf,
#                      hidden_train,train_label,
#                      hidden_test, test_label,
#                      conf=conf,
#                      cv_index=cv_index,
#                      size=label_ratio,
#                      save_name='aksar_l-svm-OVR-2',
#                      name_cm = 'aksar_l-svm-OVR-2',
#                      plot_cm = plot_cm)  
    
#     clf = svm.LinearSVC(max_iter=5000,
#                        #loss = 'hinge'
#                        penalty='l1')
    
#     clf.fit(hidden_train, train_label)
#     pipelineAnalysis(clf,
#                      hidden_train,train_label,
#                      hidden_test, test_label,
#                      conf=conf,
#                      cv_index=cv_index,
#                      size=label_ratio,
#                      save_name='aksar_l-svm-OVR-3',
#                      name_cm = 'aksar_l-svm-OVR-3',
#                      plot_cm = plot_cm)     
    
#     clf = svm.LinearSVC(max_iter=5000,
#                        #loss = 'hinge'
#                        penalty='l2',
#                        C=0.5)
    
#     clf.fit(hidden_train, train_label)
#     pipelineAnalysis(clf,
#                      hidden_train,train_label,
#                      hidden_test, test_label,
#                      conf=conf,
#                      cv_index=cv_index,
#                      size=label_ratio,
#                      save_name='aksar_l-svm-OVR-3',
#                      name_cm = 'aksar_l-svm-OVR-3',
#                      plot_cm = plot_cm)     

    
    #Aksar XGboost

#     from xgboost import XGBClassifier
        
#     clf = OneVsRestClassifier(XGBClassifier())
#     clf.fit(hidden_train, train_label)
#     pipelineAnalysis(clf,
#                      hidden_train,train_label,
#                      hidden_test, test_label,
#                      conf=conf,
#                      cv_index=cv_index,
#                      size=label_ratio,
#                      save_name='aksar_xgb',
#                      name_cm = 'aksar_xgb',
#                      plot_cm = plot_cm)     
    
#     #Aksar RF OVR
#     from sklearn.ensemble import RandomForestClassifier
        
#     clf = OneVsRestClassifier(RandomForestClassifier())
#     clf.fit(hidden_train, train_label)
#     pipelineAnalysis(clf,
#                      hidden_train,train_label,
#                      hidden_test, test_label,
#                      conf=conf,
#                      cv_index=cv_index,
#                      size=label_ratio,
#                      save_name='aksar_rf-OVR'+ name_suffix,
#                      name_cm = 'aksar_rf-OVR'+ name_suffix,
#                      plot_cm = plot_cm)        
    

#     #Aksar DNN                                        

#     from sklearn.neural_network import MLPClassifier

#     clf = OneVsRestClassifier(MLPClassifier(
#                             hidden_layer_sizes = [32],
#                             solver = 'lbfgs',                            
#                             max_iter = 600))
    
#     clf.fit(hidden_train, train_label)
#     pipelineAnalysis(clf,
#                      hidden_train,train_label,
#                      hidden_test, test_label,
#                      conf=conf,
#                      cv_index=cv_index,
#                      size=label_ratio,
#                      save_name='aksar_mlp-lbfgs'+ name_suffix,
#                      name_cm = 'aksar_mlp-lbfgs'+ name_suffix,
#                      plot_cm = plot_cm)      

    
#     #Aksar DNN-Adam                                  

#     from sklearn.neural_network import MLPClassifier

#     clf = OneVsRestClassifier(MLPClassifier(
#                             hidden_layer_sizes = [32],
#                             solver = 'adam',                            
#                             max_iter = 600))
    
#     clf.fit(hidden_train, train_label)
#     pipelineAnalysis(clf,
#                      hidden_train,train_label,
#                      hidden_test, test_label,
#                      conf=conf,
#                      cv_index=cv_index,
#                      size=label_ratio,
#                      save_name='aksar_mlp-adam'+ name_suffix,
#                      name_cm = 'aksar_mlp-adam'+ name_suffix,
#                      plot_cm = plot_cm)    


def FAR_AMR_Calculate(true_label,pred_label):
    """
        Calculates false alarm rate and anomaly miss rate
        Assumes 0 is normal label and other labels are anomalies
        
        Args:
            true_label: Array composed of integer labels, e.g., [0,0,4,2]
            pred_label: Array composed of integer labels, e.g., [0,0,4,2]
    """    
    # • False alarm rate: The percentage of the healthy windows that are identified as anomalous (any anomaly type).
    # • Anomaly miss rate: The percentage of the anomalous windows that are identified as healthy
    alarm_dict = {}
        
    normal_true_idx = np.where(true_label==0)[0]
    anom_true_idx = np.where(true_label!=0)[0]
        
    #Find number of normal samples labeled as anomalous
    fp_deploy = pred_label[normal_true_idx][pred_label[normal_true_idx] != 0]

    false_alarm_rate = len(fp_deploy) / len(normal_true_idx)
    logging.info("Total misclassified normal runs: %s, Total normal runs %s ",str(len(fp_deploy)),str(len(normal_true_idx)))
    logging.info("FAR: %s",false_alarm_rate)
    
    #Find number of anomalous samples labeled as normal
    fn_deploy = pred_label[anom_true_idx][pred_label[anom_true_idx] == 0]

    anom_miss_rate = len(fn_deploy) / len(anom_true_idx)
    logging.info("Total misclassified anom runs: %s, Total anom runs %s ",str(len(fn_deploy)),str(len(anom_true_idx)))
    logging.info("AMR: %s",anom_miss_rate)    
    
    
    return false_alarm_rate, anom_miss_rate
    


def analysis_wrapper_multiclass(true_labels, pred_labels,conf,cv_index,name,name_cm='Deployment Data',save=True,plot=True):
    """
        true_labels: it should be in the format of an array [0,2,1,3,...]
        pred_labels: it should be in the format of an array [0,1,1,4,...]        
    """
    from sklearn.metrics import classification_report
    logging.info("####################################")

    logging.info("%s\n%s",name_cm,classification_report(y_true=true_labels, y_pred =pred_labels))
    logging.info("#############")
    deploy_report = classification_report(y_true=true_labels, y_pred =pred_labels,output_dict=True)

    if save:
        cv_path = conf['results_dir']


        json_dump = json.dumps(deploy_report)
        f_json = open(cv_path / ("{}_report_dict.json".format(name)),"w")
        f_json.write(json_dump)
        f_json.close() 
        
    if plot:
        plot_cm(true_labels, pred_labels,name=name_cm)    
        
    false_anom_rate_calc(true_labels,pred_labels,conf,cv_index,name,save)        

    
def analysis_wrapper_binary(true_labels, pred_labels,conf,cv_index,name,name_cm='Deployment Data',save=True,plot=True):
    """
        true_labels: it should be in the format of an array [0,0,1,0,...]
        pred_labels: it should be in the format of an array [0,0,1,0,...]        
    """
    from sklearn.metrics import classification_report
    logging.info("####################################")

    logging.info("%s\n%s",name_cm,classification_report(y_true=true_labels, y_pred =pred_labels))
    logging.info("#############")
    deploy_report = classification_report(y_true=true_labels, y_pred =pred_labels,output_dict=True)
    
    
#     target_names = ['normal','anomaly']
#     print(classification_report(y_true=true_labels, y_pred =pred_labels,target_names=target_names))
#     deploy_report = classification_report(y_true=true_labels, y_pred =pred_labels,target_names=target_names,output_dict=True)

    if save:
        cv_path = conf['results_dir']
#         cv_path = conf['plots_dir'] / ("CV_" + str(cv_index))
#         if not cv_path.exists():
#             cv_path.mkdir(parents=True)

        json_dump = json.dumps(deploy_report)
        f_json = open(cv_path / ("{}_report_dict.json".format(name)),"w")
        f_json.write(json_dump)
        f_json.close() 
        
    if plot:    
        plot_cm(true_labels, pred_labels,name=name_cm)    
        
    false_anom_rate_calc(true_labels,pred_labels,conf,cv_index,name,save)     
    
            
def pipelineAnalysis(clf,train_data,train_true_label,test_data, test_true_label, conf, cv_index, size, save_name="", name_cm="",plot_cm=True,save=True):
    """
        Send the classifier and generate necessary results for train and test data
    """    
    train_pred_label = clf.predict(train_data)
    analysis_wrapper_multiclass(true_labels=train_true_label,
                                pred_labels=train_pred_label,
                                conf=conf,
                                cv_index=cv_index,
                                name = (save_name + "_train_{}").format(size),
                                name_cm = name_cm + " Train",
                                save=save,
                                plot=plot_cm                                
                               )

    test_pred_label = clf.predict(test_data)
    analysis_wrapper_multiclass(true_labels=test_true_label,
                                pred_labels=test_pred_label,
                                conf=conf,
                                cv_index=cv_index,
                                name = (save_name + "_test_{}").format(size),
                                name_cm = name_cm + " Test",
                                save=save,
                                plot=plot_cm
                               )    
    
def pipelineUnknownAnalysis(test_true_label,test_pred_label, conf, cv_index, size, save_name="", name_cm="",plot_cm=True,save=True):
    """
        Send the classifier and generate necessary results for train and test data
    """    
#     train_pred_label = clf.predict(train_data)
#     analysis_wrapper_multiclass(true_labels=train_true_label,
#                                 pred_labels=train_pred_label,
#                                 conf=conf,
#                                 cv_index=cv_index,
#                                 name = (save_name + "_train_{}").format(size),
#                                 name_cm = name_cm + " Train",
#                                 save=save,
#                                 plot=plot_cm                                
#                                )

#     test_pred_label = clf.predict(test_data)

    analysis_wrapper_multiclass(true_labels=test_true_label,
                                pred_labels=test_pred_label,
                                conf=conf,
                                cv_index=cv_index,
                                name = (save_name + "_test_{}").format(size),
                                name_cm = name_cm + " Test",
                                save=save,
                                plot=plot_cm
                               )      
    
    
def pipelineAnalysisBorghesi(clf, threshold, train_data,train_true_label,test_data, test_true_label, conf, cv_index, size, save_name="", name_cm="",plot_cm=True,save=True):
    """
        Send the classifier and generate necessary results for train and test data
    """    
    
#     _, mae_loss = get_MAE_loss(clf,train_data)
    
#     train_pred_label = np.zeros(len(mae_loss)).astype(int)
#     train_pred_label[mae_loss > threshold] = 1

#     analysis_wrapper_multiclass(true_labels=train_true_label,
#                                 pred_labels=train_pred_label,
#                                 conf=conf,
#                                 cv_index=cv_index,
#                                 name = (save_name + "_train_{}").format(size),
#                                 name_cm = name_cm + " Train",
#                                 save=save,
#                                 plot=plot_cm                                
#                                )

    _, mae_loss = get_MAE_loss(clf,test_data)
    
    test_pred_label = np.zeros(len(mae_loss)).astype(int)
    test_pred_label[mae_loss > threshold] = 1    

    analysis_wrapper_multiclass(true_labels=test_true_label,
                                pred_labels=test_pred_label,
                                conf=conf,
                                cv_index=cv_index,
                                name = (save_name + "_test_{}").format(size),
                                name_cm = name_cm + " Test",
                                save=save,
                                plot=plot_cm
                               )         

        
def pipelineAnalysisKeras(clf,train_data,train_true_label,test_data, test_true_label, conf, cv_index, size, save_name="", name_cm="",plot_cm=True,save=True):
    """
        Send the classifier and generate necessary results for train and test data
    """    
    
    train_pred_label = np.argmax(clf.predict(train_data),1)
    analysis_wrapper_multiclass(true_labels=train_true_label,
                                pred_labels=train_pred_label,
                                conf=conf,
                                cv_index=cv_index,
                                name = (save_name + "_train_{}").format(size),
                                name_cm = name_cm + " Train",
                                save=save,
                                plot=plot_cm                                
                               )

    test_pred_label = np.argmax(clf.predict(test_data),1)
    analysis_wrapper_multiclass(true_labels=test_true_label,
                                pred_labels=test_pred_label,
                                conf=conf,
                                cv_index=cv_index,
                                name = (save_name + "_test_{}").format(size),
                                name_cm = name_cm + " Test",
                                save=save,
                                plot=plot_cm
                               )    
    
    
def generate_results(clf, X, y, result_name):
    """
        Prints classification report, plots confusion matrix and returns predictions
    """
    target_names = ['normal','membw','memleak','cachecopy','cpuoccupy']

    X_pred = clf.predict(X)
    print(classification_report(y_true=y, y_pred=X_pred, target_names=target_names))
    
    plot_cm(y, X_pred, name=result_name + '') 
    
    return X_pred



def plot_cm(labels, predictions, name):

    cm = tf.math.confusion_matrix(labels, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('{}'.format(name))    
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def false_anom_rate_calc(true_label,pred_label,conf,cv_index,name,save):
    """
        Calculates false alarm rate and anomaly miss rate
        Assumes 0 is normal label and other labels are anomalies
        
        Args:
            true_label: Array composed of integer labels, e.g., [0,0,4,2]
            pred_label: Array composed of integer labels, e.g., [0,0,4,2]
    """    
    # • False alarm rate: The percentage of the healthy windows that are identified as anomalous (any anomaly type).
    # • Anomaly miss rate: The percentage of the anomalous windows that are identified as healthy
    alarm_dict = {}
        
    normal_true_idx = np.where(true_label==0)[0]
    anom_true_idx = np.where(true_label!=0)[0]
        
    #Find number of normal samples labeled as anomalous
    fp_deploy = pred_label[normal_true_idx][pred_label[normal_true_idx] != 0]

    false_alarm_rate = len(fp_deploy) / len(normal_true_idx)
    logging.info("Total misclassified normal runs: %s, Total normal runs %s ",str(len(fp_deploy)),str(len(normal_true_idx)))
    logging.info(false_alarm_rate)
    
    #Find number of anomalous samples labeled as normal
    fn_deploy = pred_label[anom_true_idx][pred_label[anom_true_idx] == 0]

    anom_miss_rate = len(fn_deploy) / len(anom_true_idx)
    logging.info("Total misclassified anom runs: %s, Total anom runs %s ",str(len(fn_deploy)),str(len(anom_true_idx)))
    logging.info(anom_miss_rate)    
    
    alarm_dict['false_alarm_rate'] = false_alarm_rate
    alarm_dict['anom_miss_rate'] = anom_miss_rate
    
    if save:
        json_dump = json.dumps(alarm_dict)
        f_json = open(conf['results_dir'] / ("{}_alert_dict.json".format(name)),"w")
        f_json.write(json_dump)
        f_json.close()
        

def falseAnomRateCalc(true_label,pred_label):
    """
        Calculates false alarm rate and anomaly miss rate
        Assumes 0 is normal label and other labels are anomalies
        
        Args:
            true_label: Array composed of integer labels, e.g., [0,0,4,2]
            pred_label: Array composed of integer labels, e.g., [0,0,4,2]
    """    
    # • False alarm rate: The percentage of the healthy windows that are identified as anomalous (any anomaly type).
    # • Anomaly miss rate: The percentage of the anomalous windows that are identified as healthy
    alarm_dict = {}
        
    normal_true_idx = np.where(true_label==0)[0]
    anom_true_idx = np.where(true_label!=0)[0]
        
    #Find number of normal samples labeled as anomalous
    fp_deploy = pred_label[normal_true_idx][pred_label[normal_true_idx] != 0]

    false_alarm_rate = len(fp_deploy) / len(normal_true_idx)
    logging.info("Total misclassified normal runs: %s, Total normal runs %s ",str(len(fp_deploy)),str(len(normal_true_idx)))
    logging.info(false_alarm_rate)
    
    #Find number of anomalous samples labeled as normal
    fn_deploy = pred_label[anom_true_idx][pred_label[anom_true_idx] == 0]

    anom_miss_rate = len(fn_deploy) / len(anom_true_idx)
    logging.info("Total misclassified anom runs: %s, Total anom runs %s ",str(len(fn_deploy)),str(len(anom_true_idx)))
    logging.info(anom_miss_rate)    
    
    alarm_dict['false_alarm_rate'] = false_alarm_rate
    alarm_dict['anom_miss_rate'] = anom_miss_rate
    
    return alarm_dict
        
    

### TRAINING RELATED PLOTS/RESULTS

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import seaborn as sns
sns.set(rc={'figure.figsize':(12,10)})
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("talk")

def calc_MSE_class(model, data,labels,ANOM_DICT):

    
    mse_df = pd.DataFrame(columns=['Average_MSE','Class'])
    for label in np.unique(labels):
        temp_data = data[np.where(labels==label)]
        _, mse_loss = get_MSE_loss(model,inputs=temp_data)
        
        for key, value in ANOM_DICT.items():
            if value == label:
                anom_name = key        
        mse_df = mse_df.append(
                      {'Average_MSE':np.mean(mse_loss),
                       'Class': anom_name
                      }, ignore_index=True)        
    return mse_df

def calc_MAE_class(model, data,labels,ANOM_DICT):
    
    mse_df = pd.DataFrame(columns=['Average_MAE','Class'])
    for label in np.unique(labels):
        temp_data = data[np.where(labels==label)]
        _, mse_loss = get_MAE_loss(model,inputs=temp_data)
        
        for key, value in ANOM_DICT.items():
            if value == label:
                anom_name = key        
        mse_df = mse_df.append(
                      {'Average_MAE':np.mean(mse_loss),
                       'Class': anom_name
                      }, ignore_index=True)        
    return mse_df

def get_MAE_loss(trained_model, inputs):
    
    # Get test MAE loss    
    recs = trained_model.predict(inputs)
    mae_loss = np.mean(np.abs(inputs - recs), axis=-1)    
    
    return recs, mae_loss

def get_MSE_loss(trained_model, inputs):
    
    # Get test MSE loss.
    recs = trained_model.predict(inputs)
    
    mse_loss = np.mean(np.square(inputs - recs), axis=-1)
        
    return recs, mse_loss

def plot_calc_train_loss(data,thresh_dict,loss_type):
    """Plot the loss data and only plot threshold lines"""            
    sns.distplot(data,kde=False,color='b')
    plt.axvline(x=thresh_dict['thresh_90'], color='g',linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_95'], color='m', linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_max'], color='y', linewidth=4, ls='--')
    plt.legend(labels=["90th Percentile ","95th Percentile","Maximum"])

    plt.xlabel("Train Data {} loss".format(loss_type))
    plt.ylabel("No. of samples")
    
    plt.show()
    
def plot_normal_loss(data, thresh_dict, loss_type):
    """This will plot MAE/MSE for normal data points and show the accuracies by looking at thresholds"""
    sns.distplot(data,kde=False,color='b')
    plt.axvline(x=thresh_dict['thresh_90'], color='g',linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_95'], color='m', linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_max'], color='y', linewidth=4, ls='--')
    plt.legend(labels=["90th Percentile ","95th Percentile","Maximum"])
    
    
    acc_dict = {}
    for key,thresh in thresh_dict.items():
        acc_dict['acc_{}'.format(key)] = round((data[thresh > data].shape[0] / data.shape[0]) * 100,2)

    textstr = '\n'.join((
        r'$Acc_{Threshold = 90}=%.2f$' % (acc_dict['acc_thresh_90'], ),
        r'$Acc_{Threshold = 95}=%.2f$' % (acc_dict['acc_thresh_95'], ),
        r'$Acc_{Threshold = Max}=%.2f$' % (acc_dict['acc_thresh_max'], )))      
    
    ax = plt.axes()      
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)    
    
    ax.text(0.65, 0.7, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)    

    plt.xlabel("Train Data {} loss".format(loss_type))
    plt.ylabel("No. of samples")
    x_limiter = 0.3
    plt.xlim(0,x_limiter)
    
    plt.show()    
    
def plot_anomalous_loss(data,thresh_dict,loss_type,anom_name='all'):
    """This will plot MAE/MSE for anomalous data points and show the accuracies by looking at thresholds"""
    sns.distplot(data,kde=False,color='b')
    plt.axvline(x=thresh_dict['thresh_90'], color='g',linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_95'], color='m', linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_max'], color='y', linewidth=4, ls='--')
    plt.legend(labels=["90th Percentile ","95th Percentile","Maximum"])
    
    
    acc_dict = {}
    for key,thresh in thresh_dict.items():
        acc_dict['acc_{}'.format(key)] = round((data[thresh < data].shape[0] / data.shape[0]) * 100,2)

    textstr = '\n'.join((
        r'$Acc_{Threshold = 90}=%.2f$' % (acc_dict['acc_thresh_90'], ),
        r'$Acc_{Threshold = 95}=%.2f$' % (acc_dict['acc_thresh_95'], ),
        r'$Acc_{Threshold = Max}=%.2f$' % (acc_dict['acc_thresh_max'], )))        
        
    ax = plt.axes()      
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper right in axes coords
    ax.text(0.65, 0.7, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)    

    plt.xlabel("Test Data({}) {} loss".format(anom_name,loss_type))
    plt.ylabel("No. of samples")
    x_limiter = 0.3
    plt.xlim(0,x_limiter)
    
    plt.show()
    
def calculate_thresholds(data_mae):
    '''Calculate max,95th and 90th percentile of the MAE data'''
    
    thresh_dict = {}
    thresh_dict['thresh_max'] = np.max(data_mae)
    thresh_dict['thresh_95'] = np.percentile(data_mae, 95)
    thresh_dict['thresh_90'] = np.percentile(data_mae, 90)

    print("Reconstruction error threshold max: ", thresh_dict['thresh_max'])
    print("Reconstruction error threshold 95th: ", thresh_dict['thresh_95'])
    print("Reconstruction error threshold 95th: ", thresh_dict['thresh_90'])
    
    return thresh_dict    

def plot_loss_per_anom(thresh_dict,loss_type):

    if conf['experiment_num'] == 1:
        unique_anoms = test_label['anom'].unique()
        for anom in unique_anoms:
            logging.info('Chosen anomaly %s',anom)
            anom_nids = test_label[test_label['anom'] == anom].index

            anom_data = scaled_test_data[scaled_test_data.index.isin(anom_nids)]
            logging.info("Anomaly data shape: %s",anom_data.shape)

            _, anom_mae_loss = _get_MAE_loss(BurhackIsBackInBusiness.trained_model,anom_data.values)  

            _plot_test_MAE_loss(anom_mae_loss,thresh_dict,anom_name=str(anom))


    elif conf['experiment_num'] == 2:
        unique_anoms = deployment_label['anom'].unique()

        for anom in unique_anoms:

            #Name conversion
            for key, value in ANOM_DICT.items():
                if value == anom:
                    anom_name = key
                    
            if anom_name != 'None':
                logging.info('Chosen anomaly %s',anom_name)
                anom_nids = deployment_label[deployment_label['anom'] == anom].index
                anom_data = scaled_deployment_data[scaled_deployment_data.index.isin(anom_nids)]

                logging.info("Anomaly data shape: %s",anom_data.shape)

                #FIXME: Is threshold_dict correct?
                if loss_type == "MAE":
                    _, anom_mae_loss = _get_MAE_loss(BurhackIsBackInBusiness.trained_model,anom_data.values)          
                    _plot_anomalous_loss(anom_mae_loss,thresh_dict,anom_name=str(anom_name),loss_type='MAE')                

                elif loss_type == "MSE":
                    _, anom_mse_loss = _get_MSE_loss(BurhackIsBackInBusiness.trained_model,anom_data.values)          
                    _plot_anomalous_loss(anom_mse_loss,thresh_dict,anom_name=str(anom_name),loss_type='MSE')
                    

### BORGHESI SPECIFIC FUNCTIONS
def plot_calc_train_loss_borghesi(data,thresh_dict,loss_type):
    """Plot the loss data and only plot threshold lines"""            
    sns.distplot(data,kde=False,color='b')
    plt.axvline(x=thresh_dict['thresh_94'], color='g',linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_97'], color='m', linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_99'], color='y', linewidth=4, ls='--')
    plt.legend(labels=["94th Percentile ","97th Percentile","99th Percentile"])

    plt.xlabel("Train Data {} loss".format(loss_type))
    plt.ylabel("No. of samples")
    
    plt.show()
    
def plot_normal_loss_borghesi(data, thresh_dict, loss_type):
    """This will plot MAE/MSE for normal data points and show the accuracies by looking at thresholds"""
    sns.distplot(data,kde=False,color='b')
    plt.axvline(x=thresh_dict['thresh_95'], color='g',linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_97'], color='m', linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_99'], color='y', linewidth=4, ls='--')
    plt.legend(labels=["95th Percentile ","97th Percentile","99th Percentile"])
    
    
    acc_dict = {}
    for key,thresh in thresh_dict.items():
        acc_dict['acc_{}'.format(key)] = round((data[thresh > data].shape[0] / data.shape[0]) * 100,2)

    textstr = '\n'.join((
        r'$Acc_{Threshold = 95}=%.2f$' % (acc_dict['acc_thresh_95'], ),
        r'$Acc_{Threshold = 97}=%.2f$' % (acc_dict['acc_thresh_97'], ),
        r'$Acc_{Threshold = 99}=%.2f$' % (acc_dict['acc_thresh_99'], )))  
    
    ax = plt.axes()      
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)    
    
    ax.text(0.65, 0.7, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)    

    plt.xlabel("Train Data {} loss".format(loss_type))
    plt.ylabel("No. of samples")
    x_limiter = 0.1
    plt.xlim(0,x_limiter)
    
    plt.show()   
    
def plot_normal_MAE_borghesi(data, loss_type):
    """This will plot MAE/MSE for normal data points and show the accuracies by looking at thresholds"""
    sns.distplot(data,kde=False,color='b')
    ax = plt.axes()      
    
    plt.xlabel("Train Data {} loss".format(loss_type))
    plt.ylabel("No. of samples")
    x_limiter = 0.1
    plt.xlim(0,x_limiter)
    
    plt.show()      
    
def plot_anomalous_loss_borghesi(data,thresh_dict,loss_type,anom_name='all'):
    """This will plot MAE/MSE for anomalous data points and show the accuracies by looking at thresholds"""
    sns.distplot(data,kde=False,color='b')
    plt.axvline(x=thresh_dict['thresh_95'], color='g',linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_97'], color='m', linewidth=4, ls='--')
    plt.axvline(x=thresh_dict['thresh_99'], color='y', linewidth=4, ls='--')
    plt.legend(labels=["95th Percentile ","97th Percentile","99th Percentile"])
    
    
    acc_dict = {}
    for key,thresh in thresh_dict.items():
        acc_dict['acc_{}'.format(key)] = round((data[thresh < data].shape[0] / data.shape[0]) * 100,2)

    textstr = '\n'.join((
        r'$Acc_{Threshold = 95}=%.2f$' % (acc_dict['acc_thresh_95'], ),
        r'$Acc_{Threshold = 97}=%.2f$' % (acc_dict['acc_thresh_97'], ),
        r'$Acc_{Threshold = 99}=%.2f$' % (acc_dict['acc_thresh_99'], )))        
        
    ax = plt.axes()      
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper right in axes coords
    ax.text(0.65, 0.7, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)    

    plt.xlabel("Test Data({}) {} loss".format(anom_name,loss_type))
    plt.ylabel("No. of samples")
    x_limiter = 0.12
    plt.xlim(0,x_limiter)
    
    plt.show()
    
def calculate_thresholds_borghesi(data_mae):
    '''Calculate max,95th and 90th percentile of the MAE data'''
    
    thresh_dict = {}
    thresh_dict['thresh_99'] = np.percentile(data_mae, 99)
    thresh_dict['thresh_97'] = np.percentile(data_mae, 97)
    thresh_dict['thresh_94'] = np.percentile(data_mae, 94)

    print("Reconstruction error threshold 99th: ", thresh_dict['thresh_99'])
    print("Reconstruction error threshold 97th: ", thresh_dict['thresh_97'])
    print("Reconstruction error threshold 94th: ", thresh_dict['thresh_94'])
    
    return thresh_dict

### BORGHESI SPECIFIC FUNCTIONS
