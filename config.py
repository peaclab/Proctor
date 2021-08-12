#!/usr/bin/env python

import argparse
from pathlib import Path
from io import StringIO
import pprint
import logging



class Configuration:
    """Keeps configurations for everything"""
    """It will create a folder under training_out_dir for the provided configuration data"""
    def __init__(self, silent=False, ipython=False, overrides=None):
        self._conf = {                        
            ### General config
            'system': 'eclipse',
            'windowing' : False,
            'window_size': 3, #In terms of minutes granularity
            'feature_select': False,
            'feature_extract': False, 
            'operation' : 'read',
            'exp_name': None,
            ### To generate HDF files for training and testing
            'raw_ldms_data_path' : None,
            'metadata_path' : None,
            'processed_ldms_data_path' : None,
            'hdf_data_path': None,
            'num_split': 5,
            ### Training 
            'output_dir' : None,
            'model_config' : None,
            'cv_fold': 0,
            ### Testing
            'runtime_testing_dir': None, #Only necessary if you are reading runtime data from a CSV
            ### Operation select

        }
        ### Calling from terminal
        if not ipython:
            self._parse_arguments()
                        
          
        ### Override default values
        if overrides is not None: 
            for key, item in overrides.items():
                self._conf[key] = item    
                
        self._set_directories()                

        self._prepare_directories()

        self._save_conf()

        if not silent:
            self._print_conf()
            
    def _set_directories(self):
        logging.info("Setting directory names")
        
        if self._conf['system'] == 'eclipse':
            
            self._conf['raw_ldms_data_path'] = None
            self._conf['metadata_path'] = None
            self._conf['processed_ldms_data_path'] = None
            self._conf['hdf_data_path'] = Path('/projectnb/peaclab-mon/aksar/datasets/eclipse_sampled_hdfs')
            #self._conf['hdf_data_path'] = Path('/projectnb/peaclab-mon/aksar/datasets/eclipse_paper_hdfs')
            #self._conf['output_dir'] = Path('/projectnb/peaclab-mon/aksar/adaptability_experiments') / self._conf['system']   
            self._conf['output_dir'] = self._conf['output_dir'] / self._conf['system']   
            
        elif self._conf['system'] == 'volta':
            
            self._conf['raw_ldms_data_path'] = None
            self._conf['metadata_path'] = None
            self._conf['processed_ldms_data_path'] = None
            self._conf['hdf_data_path'] =  Path('/projectnb/peaclab-mon/aksar/datasets/tpds_data_hdfs')
            #self._conf['output_dir'] = Path('/projectnb/peaclab-mon/aksar/adaptability_experiments') / self._conf['system']     
            self._conf['output_dir'] = self._conf['output_dir'] / self._conf['system']   
            
        elif self._conf['system'] == 'borghesi':
            
            self._conf['raw_ldms_data_path'] = None
            self._conf['metadata_path'] = None
            self._conf['processed_ldms_data_path'] = None
            self._conf['hdf_data_path'] =  None
            #self._conf['output_dir'] = Path('/projectnb/peaclab-mon/aksar/adaptability_experiments') / self._conf['system']     
            self._conf['output_dir'] = self._conf['output_dir'] / self._conf['system']   
                
        else:
            logging.info("Invalid system name")
            
    def __getitem__(self, key):
        return self._conf[key]
    
    def __setitem__(self, key, value):
        self._conf[key] = value
        
    def _parse_arguments(self):
        
        parser = argparse.ArgumentParser()
        #parser.add_argument('--test', action='store_false', dest='train')
        for key, value in self._conf.items():
            if isinstance(value, bool):
                parser.add_argument('--%s' % key, action='store_true',
                                    default=value)
                parser.add_argument('--no-%s' % key, dest=str(key),
                                    action='store_false')
            elif isinstance(value, (str, int, float)):
                parser.add_argument('--%s' % key, default=value,
                                    type=type(value))
            elif isinstance(value, Path):
                parser.add_argument(
                    '--%s' % key, default=value,
                    type=lambda x: Path(x).expanduser().resolve())
            elif isinstance(value, (list, dict)):
                # specifying dict on command line doesn't work
                parser.add_argument('--%s' % key, default=value, nargs='+')
            else:
                raise Exception('Unknown config argument type! ' + str(key) +
                                ' : ' + str(value))

        self._conf = vars(parser.parse_args())        
        
        
    def _prepare_directories(self):
        """Create the necessary directories under 'output_dir' """
        
        if not self._conf['output_dir'].exists():
            self._conf['output_dir'].mkdir(parents=True)
                                    
        self._conf['experiment_dir'] = self._conf['output_dir'] / self._conf['exp_name']
        if not self._conf['experiment_dir'].exists():
            self._conf['experiment_dir'].mkdir(parents=True)
            
        
        if not self._conf['model_config'] is None:
            
            self._conf['model_config_dir'] = self._conf['experiment_dir'] / ('CV_' + str(self._conf['cv_fold'])) / self._conf['model_config']
            
            if not self._conf['model_config_dir'].exists():
                self._conf['model_config_dir'].mkdir(parents=True)
            else:
                logging.info("Model config folder already exists, be careful, otherwise it will overwrite!")            


            self._conf['model_dir'] = self._conf['model_config_dir'] / 'model'
            if not self._conf['model_dir'].exists():
                self._conf['model_dir'].mkdir(parents=True)

            self._conf['plots_dir'] = self._conf['model_dir'] / 'plots'
            if not self._conf['plots_dir'].exists():
                self._conf['plots_dir'].mkdir(parents=True)                

            self._conf['results_dir'] = self._conf['model_config_dir'] / 'results'
            if not self._conf['results_dir'].exists():
                self._conf['results_dir'].mkdir(parents=True)


                                            
    def _save_conf(self):
        logging.info("Saving configuration")
        import csv
        
        if not self._conf['model_config'] is None:
            csv_file = self._conf['model_config_dir'] / 'model_config.csv'
        else:
            csv_file = self._conf['experiment_dir'] / 'exp_config.csv'

        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(self._conf.keys()))
            writer.writeheader()
            writer.writerow(self._conf)   
            
        csvfile.close()
        
    def _print_conf(self,comment_out=True):
        string_stream = StringIO()
        pprint.pprint(self._conf, stream=string_stream, compact=True)
        
        writer = print
        if comment_out:
            writer('# The configuration used for this run:')
            writer('# ' + '\n# '.join(string_stream.getvalue().splitlines()))
        else:
            writer('The configuration used for this run:')
            writer('\n'.join(string_stream.getvalue().splitlines()))
      