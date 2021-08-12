#!/usr/bin/env python

###GENERIC
import pandas as pd
import os,sys
from pathlib import Path
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-7s %(message)s',
                    stream=sys.stderr, level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
from io import StringIO
import pprint


###PLOTS
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import seaborn as sns
sns.set(rc={'figure.figsize':(12,10)})
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("talk")

###TENSORFLOW
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten, MaxPooling1D, Conv2D, MaxPooling2D, Activation, UpSampling2D
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Flatten, Input, ZeroPadding2D
from tensorflow.keras.layers import GlobalAvgPool1D
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1,l2,l1_l2
from tensorflow.keras.models import Model, load_model


class BaseModel:

    def __init__(self,config,params,
                 x_dim,verbose=0):        
        """Initializing the base model class"""
        """Args:
            config: dict
                Contains directories and metadata
            
            params: dict
                Contains model related parameters        
        """        
        self.x_dim = x_dim        
        self._params = params
        self._config = config
        self.verbose = verbose
        #Set after the training
        self.model_history = None
        self.validation = False
        
    def set_params(self,params,**kwargs):
        self._params = params
        
    def set_config(self,config,**kwargs):
        self._config = config       
        
    def _buildModel(self,**kwargs):
        raise NotImplementedError("Please don't use the base class, Master!")
    
    def trainModel(self,**kwargs):
        raise NotImplementedError("Please don't use the base class, Master!")

    def printParams(self,comment_out=True,**kwargs):
        
        string_stream = StringIO()
        pprint.pprint(self._params, stream=string_stream, compact=True)
        
        writer = print
        if comment_out:
            writer('# The configuration used for this run:')
            writer('# ' + '\n# '.join(string_stream.getvalue().splitlines()))
        else:
            writer('The configuration used for this run:')
            writer('\n'.join(string_stream.getvalue().splitlines()))            

    def _saveModel(self,model):
        """Saves model"""        
        logging.info("Saving model!")

        model.save(str(self._config['model_dir'] / (self._params['model_name'])))                
        logging.info("Model saved!")

    def loadModel(self,model_name):
        """Loads model with all the weights and necessary architecture"""
        logging.info("Loading model!")

        loaded_model = load_model(str(self._config['model_dir'] / (model_name)))
            
        return loaded_model 
    
    def loadModelAndWeights(self,model_name):
        """Loads model and set model weights saved by Checkpoint callback"""
        logging.info("Loading model with Checkpoint weights!")

        loaded_model = load_model(str(self._config['model_dir'] / (model_name)))
        loaded_model.load_weights(conf['model_dir'] / model_name + "_weights.h5")
            
        return loaded_model 
    

    ### Save and plot the train loss ###
    def _plotTrainLoss(self,model_name):        
        plt.plot(self.model_history['loss'], linewidth=2, label='Train')
        if self.validation:
            plt.plot(self.model_history['val_loss'], linewidth=2, label='Validation')
        plt.legend(loc='upper right')
        plt.title('Model Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(self._config['plots_dir'] / (model_name + "_train_loss.png"))
        plt.show()

    ### Save and plot the validation loss ###
    def _plotValLoss(self,model_name):
        plt.plot(self.model_history['val_loss'], linewidth=2, label='Validation')
        plt.legend(loc='upper right')
        plt.title('Model Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(self._config['plots_dir'] / (model_name + "_val_loss.png"))
        plt.show()
                
        
class BaseAutoencoder(BaseModel):
    """This model use tensor slices approach"""

    def __init__(self, config, params, x_dim,**kwargs):
        super().__init__(config, params, x_dim,**kwargs)
        logging.info("Inside BaseAutoencoder")
                    
    def compileModel(self,**kwargs):
          
        self.lr_history = []
        
        ### Optimizer ###        
        if self._params['optimizer'] == 'adam':
            opt = optimizers.Adam(learning_rate=self._params['learning_rate'])
        elif self._params['optimizer'] == 'adadelta':
            opt = optimizers.Adadelta(learning_rate=self._params['learning_rate'])
        elif self._params['optimizer'] == 'sgd':
            opt = optimizers.SGD(learning_rate=self._params['learning_rate'])
                            
        ### Compile ###
        self.trained_model.compile(loss=self._params['loss'], optimizer=opt, metrics=[self._params['loss']])        
        
                       
    def trainModel(self, x_train, validation=False, **kwargs):
        
        self.validation = validation
        
        tf.keras.backend.clear_session()
        
        self.compileModel()
        
        ### Callbacks ###
        cp = ModelCheckpoint(filepath=str(self._config['model_dir'] / (self._params['model_name'] + "_weights.h5")),
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss', 
                             mode ='min',
                             verbose=0)        
                
        es = EarlyStopping(monitor='val_loss', 
                                            verbose=1,
                                            patience=10,
                                            mode='min',
                                            restore_best_weights=True)        
        def scheduler(epoch, lr):
            
            if epoch < 10:
                new_lr = lr
            else:
                new_lr = lr * tf.math.exp(-0.1)      

            self.lr_history.append(new_lr)
            
            return new_lr
        
        def decayed_learning_rate(epoch, lr):
            
            decay_rate = lr / self._params['epochs']
            
            new_lr = lr * (1.0 / (1.0 + decay_rate*(epoch*40)))
            self.lr_history.append(new_lr)
            
            return new_lr
        
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
                                                                decayed_learning_rate, 
                                                                verbose=1
                                                                )                
        if self.validation:
            logging.info("Validation is active!")            

            ### Fit the model ###
            model_history = self.trained_model.fit(                      
                                        x_train,
                                        x_train,
                                        epochs=self._params['epochs'], 
                                        callbacks=[
                                                cp,
                                                es,
                                                #lr_schedule,
                                                #tensorboard_callback
                                                ],

                                        validation_split = 0.2,
                                        shuffle=True,                                                
                                        verbose=self.verbose).history

            
        self.model_history = model_history

        self._saveModel(self.trained_model)
        self._saveEncoder()
        self._plotTrainLoss(self._params['model_name'])

    def _saveEncoder(self):
        """Saves encoder model"""        
        logging.info("Saving encoder!")

        self.getEncoder().save(str(self._config['model_dir'] / (self._params['model_name'] + '_encoder')))                
        logging.info("Encoder saved!")          
        
    def getEncoder(self):
        raise NotImplementedError("Please don't use the base class directly")    

    def getDecoder(self):
        raise NotImplementedError("Please don't use the base class directly")
        
    def getCodeLayer(self):        
        raise NotImplementedError("Please don't use the base class directly")        
        
# class DNN_AE(BaseAutoencoder):
    
#     def __init__(self, config, params, x_train, hidden_layer, x_val=None,**kwargs):        
#         super().__init__(config, params, x_train,x_val,**kwargs)
#         logging.info("Inside Dense Deep AE")        
        
#         self._buildBaseModel(hidden_layer)
                
#     def _buildBaseModel(self,hidden_layer):
#         """Builds model using Keras sequential API"""
        
#         self.hidden_layer = hidden_layer
        
#         if self._params['regularizer'] == 'l1':
#             layerRegularizer = regularizers.l1(l = self._params['regularization_rate'])

#         elif self._params['regularizer'] == 'l2':
#             layerRegularizer = regularizers.l2(l = self._params['regularization_rate'])                        
            
#         elif self._params['regularizer'] == "l1l2":        
#             layerRegularizer = regularizers.l1_l2(l1=self._params['regularization_rate'] , l2 = self._params['regularization_rate'])
        
        
#         self.base_model = Sequential(name=self._params['model_name'] + '_base_model')
        
#         input_layer = Input(shape =(self.train_shape[1],))
        
#         self.base_model.add(input_layer)
        
# #         self.base_model.add(Dense(256, activation=self._params['activation'],
# #                                   kernel_regularizer= layerRegularizer,name='EncoderStart')) 
        
#         self.base_model.add(Dropout(0.1))
                
#         self.base_model.add(Dense(
#                                     hidden_layer, activation=self._params['activation'],                                  
#                                     kernel_regularizer=layerRegularizer,
#                                     kernel_initializer=tf.keras.initializers.GlorotNormal(), 
#                                     name="hidden_layer_" + str(hidden_layer)))    
        
#         self.base_model.add(Dropout(0.1))
        
# #         self.base_model.add(Dense(256, activation=self._params['activation'],
# #                                   kernel_regularizer= layerRegularizer))
                    
#         decoded = Dense(self.train_shape[1], activation ='sigmoid',name="DecoderOut")

#         self.base_model.add(decoded)
                
#         logging.info(self.base_model.summary())
        
#         self.trained_model = self.base_model
                
#     def get_encoder(self):

#         hidden_encoder = Sequential()
#         for layer in self.trained_model.layers:
            
#             if layer.name != ('hidden_layer_' + str(self.hidden_layer)):
#                 hidden_encoder.add(layer)
            
#             elif layer.name == ('hidden_layer_' + str(self.hidden_layer)):
#                 hidden_encoder.add(layer)
#                 break                
                
#         return hidden_encoder
    
#     def get_hidden_layer(self):
        
#         for layer in self.trained_model.layers:
            
#             if layer.name == ('hidden_layer_' + str(self.hidden_layer)):
#                 return layer
            
#         return None       
    
    
    
# class Supervised_SAE(BaseAutoencoder):
    
#     def __init__(self, config, params, x_train, y_train, x_val=None, y_val=None, **kwargs):        
#         logging.info("Inside Supervised Stacked Deep AE") 
#         self._config = config
#         self._params = params
#         self.x_train_shape = x_train.shape
#         self.y_train_shape = y_train.shape
                 
#     def buildSupervisedModel(self,hidden_layer_1,hidden_layer_2,output_bias=None):
#         """Load the previously trained autoencoder and prepare for supervised learning"""
                                                
#         if self._params['finetune_regularizer'] == 'l1':
#             layerRegularizer = regularizers.l1(l = self._params['finetune_regularization_rate'])

#         elif self._params['finetune_regularizer'] == 'l2':
#             layerRegularizer = regularizers.l2(l = self._params['finetune_regularization_rate'])                        
            
#         elif self._params['finetune_regularizer'] == "l1l2":        
#             layerRegularizer = regularizers.l1_l2(l1=self._params['finetune_regularization_rate'] , l2 = self._params['finetune_regularization_rate'])
        
#         if output_bias is not None:
#             output_bias = tf.keras.initializers.Constant(output_bias)                
                    
#         if self._params['finetune_optimizer'] == 'adam':
#             opt = optimizers.Adam(self._params['finetune_learning_rate'])
#         elif self._params['finetune_optimizer'] == 'adadelta':
#             opt = optimizers.Adadelta(self._params['finetune_learning_rate'])            
#         elif self._params['finetune_optimizer'] == 'sgd':
#             opt = optimizers.SGD(self._params['finetune_learning_rate'], momentum=0.9)        

#         METRICS = [
#               tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
#         ]        
        
                    
#         self.supervised_model = Sequential(name=self._params['finetune_model_name'])
#         input_layer = Input(shape =(self.x_train_shape[1],))

#         self.supervised_model.add(input_layer)
#         self.supervised_model.add(hidden_layer_1)
#         self.supervised_model.add(hidden_layer_2)
        
#         ###mark all remaining layers as non-trainable
#         for layer in self.supervised_model.layers:
#             layer.trainable = False            

# #         self.supervised_model.add(Dense(128))        
# #         self.supervised_model.add(Dropout(0.4))
# #         self.supervised_model.add(Dense(64))                
#         self.supervised_model.add(Dense(self.y_train_shape[1],activation='sigmoid',name="supervised_output"))        
                
#         # compile model
#         self.supervised_model.compile(loss=self._params['finetune_loss'], optimizer=opt, metrics=METRICS)            
                            
#         print(self.supervised_model.summary())        
                                                                        
#     def trainSupervisedModel(self,train_data,train_label,verbose=0):
                
#         ### Callbacks ###
#         cp = ModelCheckpoint(filepath=str(self._config['model_dir'] / (self._params['finetune_model_name'] + "_checkpoint_classifier.h5")),
#                              save_best_only=True,
#                              monitor='loss', 
#                              mode ='min',
#                              verbose=0)
        
#         reduce_lr = ReduceLROnPlateau(monitor='loss', 
#                                           factor=0.1, 
#                                           patience=20, 
#                                           min_lr=0.000000001,
#                                           verbose=0)
        


#         ### Fit the model ###
#         self.finetuning_model_history = self.supervised_model.fit(                      
#                                     train_data,
#                                     train_label,
#                                     epochs=self._params['finetune_epochs'], 
#                                     batch_size=self._params['finetune_batch_size'], 
#                                     callbacks=[
#                                             cp,
#                                             #es,
#                                             reduce_lr,
#                                             #tensorboard_callback
#                                             ],
#                                     shuffle=True,
#                                     validation_split=0.1,
#                                     verbose=verbose)

#         self._saveModel(self.supervised_model,self._params['finetune_model_name'])
    

#     def plot_supervised_metrics(self):
        
#         history = self.finetuning_model_history
#         metrics =  ['loss', 'accuracy']
#         for n, metric in enumerate(metrics):
#             name = metric.replace("_"," ").capitalize()
#             plt.subplot(2,2,n+1)
#             plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
#             plt.plot(history.epoch, history.history['val_'+metric],color=colors[1], linestyle="--", label='Val')
#             plt.xlabel('Epoch')
#             plt.ylabel(name)
#             if metric == 'loss':
#                 plt.ylim([0, plt.ylim()[1]])
#             elif metric == 'auc':
#                 plt.ylim([0,1])
#             else:
#                 plt.ylim([0,1])

#             plt.legend()
            
            
class Borghesi_AE(BaseAutoencoder):
    
    def __init__(self, config, params, x_train, x_val=None,**kwargs):        
        super().__init__(config, params, x_train, x_val,**kwargs)
        logging.info("Inside Dense Deep AE")        
        
        self._buildBaseModel()
        
    def _buildModel(self):
        """Builds model using Keras functional API"""        
        if self._params['regularizer'] == 'l1':
            layerRegularizer = regularizers.l1(l = self._params['regularization_rate'])

        elif self._params['regularizer'] == 'l2':
            layerRegularizer = regularizers.l2(l = self._params['regularization_rate'])                        
            
        elif self._params['regularizer'] == "l1l2":        
            layerRegularizer = regularizers.l1_l2(l1=self._params['regularization_rate'] , l2 = self._params['regularization_rate'])
        
        input_layer = Input(shape =(self.train_shape[1],))
                         
        encoded = Dense(self.train_shape[1] * 10, activation=self._params['activation'],
                        kernel_regularizer= layerRegularizer,name="CodeLayer")(x)
        
        decoded = Dense(self.train_shape[1], activation ='linear',name="DecoderOut")(x)
        
        self.trained_model = Model(input_layer,decoded,name=self._params['model_name'])
        
        logging.info(self.trained_model.summary())
        
    def _buildBaseModel(self):
        """Builds model using Keras sequential API"""
        if self._params['regularizer'] == 'l1':
            layerRegularizer = regularizers.l1(l = self._params['regularization_rate'])

        elif self._params['regularizer'] == 'l2':
            layerRegularizer = regularizers.l2(l = self._params['regularization_rate'])                        
            
        elif self._params['regularizer'] == "l1l2":        
            layerRegularizer = regularizers.l1_l2(l1=self._params['regularization_rate'] , l2 = self._params['regularization_rate'])
        
        
        self.base_model = Sequential(name=self._params['model_name'] + '_base_model')
        
        input_layer = Input(shape =(self.train_shape[1],))
        
        self.base_model.add(input_layer)
        
        self.base_model.add(Dense(self.train_shape[1]*10, activation=self._params['activation'],
                        kernel_regularizer=layerRegularizer,name="EncoderOut"))    
                 
        decoded = Dense(self.train_shape[1], activation ='sigmoid',name="DecoderOut")

        self.base_model.add(decoded)
                
        logging.info(self.base_model.summary())
        
        self.trained_model = self.base_model
                
    def get_encoder(self):

        hidden_encoder = Sequential()
        for layer in self.trained_model.layers:
            if layer.name != 'DecoderStart':
                hidden_encoder.add(layer)
            else:
                break
        return hidden_encoder                
                        
            