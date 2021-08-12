# Proctor
Proctor: A Semi-Supervised Performance Anomaly Diagnosis Framework for Production HPC Systems

Maintainer:

Burak Aksar - baksar@bu.edu


## Usage 

```python
from config import Configuration
from utils import *
from datasets import EclipseSampledDataset, VoltaSampledDataset, ODDSDataset

conf = Configuration(ipython=True,
                     overrides={
                         'system' : 'eclipse', #Update 
                         'operation':'label_generate', 
                         'exp_name':'config_trial', #Update
                         'output_dir': Path('/some/path/to/generate/output'), #Update
                         #CV related
                         'num_split': 5,
                         'cv_fold':0, 
                         #Windowing
                         'window_size' : 0,
                         'windowing': False,
                         #Feature selection
                         'feature_select': True,
                         'feature_extract': True,                         
                     })
                     
eclipseDataset = EclipseSampledDataset(conf)

eclipseDataset.prepare_labels()     

for cv_fold in [0,1,2,3,4]:    

    conf['cv_fold'] = cv_fold 
    eclipseDataset = EclipseSampledDataset(conf)
    eclipseDataset.prepare_data()
    
After data and label creation are completed, run autoencoder.ipynb with the same configuration above. You can set the parameters for the model inside. 

After AE training is completed, you can use dnn_multimodal.ipynb to train different supervised classifiers on top of the AE model. 

```


## Authors

[Proctor: A Semi-Supervised Performance Anomaly Diagnosis Framework for Production HPC Systems](https://link.springer.com/chapter/10.1007/978-3-030-78713-4_11)


Authors:
    Burak Aksar (1), Yijia Zhang (1), Emre Ates (1), Benjamin Schwaller (2), Omar Aaziz (2), Vitus J. Leung (2), Jim Brandt (2), Manuel Egele (1), Ayse K. Coskun (1)

Affiliations:
    (1) Department of Electrical and Computer Engineering, Boston University
    (2) Sandia National Laboratories

This work has been partially funded by Sandia National Laboratories. Sandia
National Laboratories is a multimission laboratory managed and operated by
National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy’s National Nuclear Security Administration under Contract DENA0003525.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details
