import logging
import multiprocessing
import os
import random
import glob
import h5py
import shutil
import datetime
from omegaconf import OmegaConf
import pandas as pd
import torch
from deepethogram.sequence import train as trainseq
from deepethogram import configuration, postprocessing, projects, utils
from deepethogram.debug import print_dataset_info
from deepethogram.flow_generator.train import flow_generator_train
from deepethogram.feature_extractor.train import feature_extractor_train
from deepethogram.feature_extractor.inference import feature_extractor_inference
from deepethogram.sequence.train import sequence_train
from deepethogram.sequence.inference import sequence_inference
from deepethogram.postprocessing import postprocess_and_save
from deepethogram.configuration import make_postprocessing_cfg
from deepethogram.flow_generator.inference import make_feature_extractor_inference_cfg, flow_generator_inference

# import urllib
# import mmcv
# from mmcv.runner import load_checkpoint
from encoder.inference import encoder_inference
import argparse

class Train():
    def __init__(self, args, class_list, save_proj_path,
                 backboneWeightspath):
        self.vids_list = glob.glob(args.labeled_data_path + '/**/*.mp4',
                                   recursive=True)
        self.csv_list = glob.glob(args.labeled_data_path + '/**/*.csv',
                                   recursive=True)        
        self.class_list = class_list
        self.save_projects_path = save_proj_path
        self.backboneWeightspath = backboneWeightspath
        self.exp_name = args.exp_name
        self.args = args
    
    def CopyModelsBackbone(self):
        try:
            src = '/home/ofer.nissim/DeepEthogram_SchillersLab_v1-1/MODELS_BACKBONE/pretrained_models'
            dst = self.project_config['project']['path'] + '/models/pretrained_models'
            shutil.copytree(src, dst)
            print("Directory backbone pretrained models copied successfully!")
        except shutil.Error as e:
            print(f"Backbone pretrained model copy Error: {e}")
        except Exception as e:
            print(f"Unexpected error in copy backbone pretrained models: {e}")
    
    def CreatProjectName(self):
        
        # time getter 
        current_datetime = datetime.datetime.now()
        date_time_string = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
        return 'Exp_'+ self.exp_name + '_' + date_time_string
    
    def creatnewproject(self,project_name):
    # this will create a folder called /mnt/DATA/open_field_deepethogram
    # there will be subdirectories called DATA and models
    # there will also be a project_config.yaml
        project_config = projects.initialize_project(self.save_projects_path,
                                                     project_name,
                                                     self.class_list)
        self.project_config = project_config
        self.CopyModelsBackbone()
        self.project_config = project_config
        return project_config
    
    def addDataTrain(self):
        list_of_movies =self.vids_list
        csvs = self.csv_list
        mode = 'copy' # or 'copy' 'symlink' or 'move'
        # depending on the mode, it will copy, symlink, or move each video file
        # it will also compute the mean and standard deviation of each RGB channel
        for movie_path in list_of_movies:
            projects.add_video_to_project(self.project_config, movie_path, mode=mode)

        # change path to vids to be the new one 
        vids_project = self.project_config['project']['path'] + '/DATA/*/*.mp4'
        list_of_movies_updated = glob.glob(vids_project)
        
        # sort by name 
        vids_sorted = sorted(list_of_movies_updated, key=lambda x: os.path.basename(x).split('.')[0])
        cvs_sorted = sorted(csvs, key=lambda x: os.path.basename(x).split('.')[0])  
        
        # adds csv files to the dir of the vid, and adds '_labels' if not exist
        for movie_path, label_path in zip(vids_sorted, cvs_sorted):
            projects.add_label_to_project(label_path, movie_path)


    # NEW PROJECT CREATION, import this 
    def initialization(self):
        projectname = self.CreatProjectName()
        self.creatnewproject(projectname)
        self.addDataTrain()
        
        # stage 0: check dirs for project and initializations
        self.project_path = self.project_config['project']['path'] # updated 
        # projects.convert_all_videos(os.path.join(self.project_path, 'project_config.yaml') , movie_format='hdf5', codec='.jpg') # additional function to compress data
        files = os.listdir(self.project_path)
        assert 'DATA' in files, 'DATA directory not found! {}'.format(files)
        assert 'models' in files, 'models directory not found! {}'.format(files)
        assert 'project_config.yaml' in files, 'project config not found! {}'.format(files)
        
        log = self.reset_logger()
        print_dataset_info(os.path.join(self.project_path, 'DATA'))
            
        # check the gpus used 
        print(torch.__version__)
        print('gpu available: {}'.format(torch.cuda.is_available()))
        # print('gpu name: {}'.format(torch.cuda.get_device_name(0)))
        # assert torch.cuda.is_available(), 'Please select a GPU runtime and then restart!'
        
    def reset_logger(self):
        # First, overwrite any logger so that we can actually see log statements
        # https://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a
        log = logging.getLogger()  # root logger
        log.setLevel(logging.INFO)
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        log.addHandler(logging.StreamHandler())
        self.log = log
        return log
    
    
    # TRAIN FEATURE EXTRACTOR 
    def tain_feature_extractor(self):
        preset = 'deg_f' # type of model -> deg_f, deg_m, deg_s
        cfg = configuration.make_feature_extractor_train_cfg(self.project_path,
                                                             preset=preset)
        
        cfg.feature_extractor.arch = self.args.arch_FE
        cfg.feature_extractor.n_flows = self.args.flow_max_FG # should be as flow generation 
        cfg.feature_extractor.curriculum = self.args.curriculum_FE
        cfg.train.num_epochs = self.args.num_epochs_FE
        cfg.compute.batch_size = 8
        
        # the latest string will find the most recent model by date
        # you can also pass a specific .pt or .ckpt file here
        cfg.flow_generator.weights = 'latest'
        n_cpus = multiprocessing.cpu_count()
        cfg.compute.num_workers = n_cpus
        log = self.reset_logger()
        
        print(OmegaConf.to_yaml(cfg))
        
        feature_extractor = feature_extractor_train(cfg)


    # INFERENCE FEATURE EXTRACTOR 
    def inference_feature_extractor(self):
        preset = 'deg_f'
        cfg = configuration.make_feature_extractor_inference_cfg(project_path=self.project_path, preset=preset)
        
        cfg.feature_extractor.weights = 'latest' 
        cfg.flow_generator.weights = 'latest' # do not change since it is adopted to the specific training 
        cfg.inference.overwrite = True
        # make sure errors are thrown
        cfg.inference.ignore_error = False
        cfg.compute.num_workers = 2
        cfg.compute.batch_size = 8
        
        print(OmegaConf.to_yaml(cfg))

        feature_extractor_inference(cfg)

    # INFERENCE ENCODER
    def inference_encoder(self):
        # Note that this configuration refers to the original feature extractor model.
        # A separate encoder model configuration can be set according to future need, but it requires some similar
        # attributes like the original one.
        cfg = configuration.make_feature_extractor_inference_cfg(project_path=self.project_path)
        encoder_inference(cfg)
        
    # TRAIN SEQUENCE 
    def train_sequence(self):
        cfg = trainseq.make_sequence_train_cfg(self.project_path,
                                               use_command_line=True)
        
        cfg.train.num_epochs = self.args.num_epochs_S
        cfg.sequence.rnn_style = self.args.rnn_style_S
        cfg.compute.batch_size = 8
        cfg.sequence.latent_name = 'encoder'
        
        trainseq.sequence_train(cfg)
 
    # INFERENCE SEQUENCE MODEL
    def inference_sequence(self):
        cfg = configuration.make_sequence_inference_cfg(self.project_path)
        cfg.sequence.weights = 'latest'
        n_cpus = multiprocessing.cpu_count()
        cfg.compute.num_workers = n_cpus
        cfg.inference.overwrite = True
        cfg.inference.ignore_error = False
        sequence_inference(cfg)

    # POST PROCESS PLOTS
    def postprocessing(self):
        cfg = make_postprocessing_cfg(self.project_path)
        postprocess_and_save(cfg)
   

def main():
    
    parser = argparse.ArgumentParser(description='get inputs from the user for hyperparm etc.')
    
    # start from other location
    parser.add_argument('--Exist_path', type = str, default= None, help = 'If the project already exist then take the path to it')
    
    # arguments from the user
    parser.add_argument('--labeled_data_path', type=str, default= '/home/ofer.nissim/DeepEthogram_SchillersLab_v1-1/Dataset_deepethogram_28_12',
                        help='the path to the data')
    parser.add_argument('--exp_name', type=str, default= '', help='the name to add to the experiment')
    
    # Flow generator parameters 
    parser.add_argument('--num_epochs_FG', type=int, default= 10, help='the num of epochs')
    parser.add_argument('--flow_max_FG', type=int, default= 10, help='the num of frames') # the input_images should be this +1 and also n_rgb
    parser.add_argument('--arch_FG', type=str, default= 'TinyMotionNet', 
                        help = 'Could also be TinyMotionNet, MotionNet, TinyMotionNet3d')
    
    # Feature extractor parameters
    parser.add_argument('--num_epochs_FE', type=int, default= 20, help='the num of epochs')
    parser.add_argument('--arch_FE', type=str, default= 'resnet18', help='the arch of FE, could be resnet18, resnet50, resnet3d_34')
    parser.add_argument('--curriculum_FE', type=bool, default= False, 
                        help='if true, first trains the spatial CNN, then the flow CNN, and finally the two jointly end-to-end')
    
    # sequence parameters 
    parser.add_argument('--num_epochs_S', type=int, default= 100, help='the num of epochs')
    parser.add_argument('--rnn_style_S', type=str, default= 'lstm', help='can be from these options: rnn, gru, lstm')
    
    args = parser.parse_args()

    # inputs from user 
    class_list = ['background', 
            'Perch', 
            'Lift', 
            'Reach', 
            'Grab_nonPellet',
            'Grab',
            'Sup',
            'AtMouth',
            'AtMouth_nonPellet',
            'BackPerch',
            'Table']
    
    save_proj_path = '/home/ofer.nissim/DeepEthogram_SchillersLab_v1-1/Models_results_deepEto/'
    backboneWeightspath = '/home/ofer.nissim/DeepEthogram_SchillersLab_v1-1/MODELS_BACKBONE/pretrained_models'
    
    # create object 
    trainer = Train(args, 
                    class_list, 
                    save_proj_path,
                    backboneWeightspath)
    
    trainer.initialization()
    trainer.inference_encoder()
    trainer.train_sequence()
    trainer.inference_sequence()
    trainer.postprocessing()

if __name__ == '__main__':
    main()