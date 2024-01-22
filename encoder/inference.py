from collections import defaultdict
import logging
import os
import sys
import time
from typing import Type, Union

import h5py
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch import nn
from tqdm import tqdm

from deepethogram import utils, projects
from deepethogram.configuration import make_feature_extractor_inference_cfg
from deepethogram.data.augs import get_cpu_transforms, get_gpu_transforms_inference, get_gpu_transforms
from deepethogram.data.datasets import VideoIterable
from deepethogram.feature_extractor.train import build_model_from_cfg as build_feature_extractor
from deepethogram.file_io import read_labels
from deepethogram.postprocessing import get_postprocessor_from_cfg

log = logging.getLogger(__name__)

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


def get_linear_layers(model: nn.Module) -> list:
    """unpacks the linear layers from a nn.Module, including in all the sequentials

    Parameters
    ----------
    model : nn.Module
        CNN

    Returns
    -------
    linear_layers: list
        ordered list of all the linear layers
    """
    linear_layers = []
    children = model.children()
    for child in children:
        if isinstance(child, nn.Sequential):
            linear_layers.append(get_linear_layers(child))
        elif isinstance(child, nn.Linear):
            linear_layers.append(child)
    return linear_layers

def predict_single_video(videofile: Union[str, os.PathLike],
                         model: nn.Module,
                         num_rgb: int,
                         mean_by_channels: np.ndarray,
                         device: str = 'cuda:0',
                         cpu_transform=None,
                         gpu_transform=None,
                         should_print: bool = False,
                         num_workers: int = 1,
                         batch_size: int = 16):
    """Runs inference on one input video, caching the output probabilities and image and flow feature vectors

    Parameters
    ----------
    videofile : Union[str, os.PathLike]
        Path to input video
    model : nn.Module
        Hidden two-stream model
    activation_function : nn.Module
        Either sigmoid or softmax
    fusion : str
        How features are fused. Needed for extracting them from the model architecture
    num_rgb : int
        How many images are input to the model
    mean_by_channels : np.ndarray
        Image channel mean for z-scoring
    device : str, optional
        Device on which to run inference, by default 'cuda:0'. Options: ['cuda:N', 'cpu']
    cpu_transform : callable, optional
        CPU transforms to perform, e.g. center cropping / resizing, by default None
    gpu_transform : callable, optional
        GPU augmentations. For inference, should just be conversion to float and z-scoring, by default None
    should_print : bool, optional
        If true, print more debug statements, by default False
    num_workers : int, optional
        Number of workers to read the video in parallel, by default 1
    batch_size : int, optional
        Batch size for inference. Values above 1 will be much faster. by default 16

    Returns
    -------
    dict
        keys: values
        probabilities: torch.Tensor, T x K probabilities of each behavior
        logits: torch.Tensor, T x K outputs for each behavior, before activation function
        spatial_features: T x 512 feature vectors from images
        flow_features: T x 512 feature vectors from optic flow
        debug: T x 1 tensor storing the number of times each frame was read. Should be full of ones and only ones

    Raises
    ------
    ValueError
        If input from dataloader is not a dict or a Tensor, raises
    """

    model.eval()
    # model.set_mode('inference')

    if type(device) != torch.device:
        device = torch.device(device)

    dataset = VideoIterable(videofile,
                            transform=cpu_transform,
                            num_workers=num_workers,
                            sequence_length=num_rgb,
                            mean_by_channels=mean_by_channels)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    video_frame_num = len(dataset)


    buffer = {}

    # log.debug('model training mode: {}'.format(model.training))
    for i, batch in enumerate(tqdm(dataloader, leave=False)):
        if isinstance(batch, dict):
            images = batch['images']
        elif isinstance(batch, torch.Tensor):
            images = batch
        else:
            raise ValueError('unknown input type: {}'.format(type(batch)))

        if images.device != device:
            images = images.to(device)
        # images = batch['images']
        with torch.no_grad():
            images = gpu_transform(images)

            features = model(images)
            logits = features
        # because we are using iterable datasets, each batch will be a consecutive chunk of frames from one worker
        # but they might be from totally different chunks of the video. therefore, we return the frame numbers,
        # and use this to store into our buffer in the right location
        frame_numbers = batch['framenum'].detach().cpu()

        features = features.detach().cpu()
        logits = logits.detach().cpu()

        if i == 0:
            # print(f'~~~ N: {N} ~~~')
            buffer['logits'] = torch.zeros((video_frame_num, logits.shape[1]), dtype=logits.dtype)
            buffer['features'] = torch.zeros((video_frame_num, features.shape[1]), dtype=features.dtype)
            buffer['debug'] = torch.zeros((video_frame_num, )).float()
        buffer['logits'][frame_numbers] = logits
        buffer['features'][frame_numbers] = features
        buffer['debug'][frame_numbers] += 1
    return buffer


def extract(rgbs: list,
            model,
            mean_by_channels,
            num_rgb: int,
            latent_name: str,
            class_names: list = ['background'],
            device: str = 'cuda:0',
            cpu_transform=None,
            gpu_transform=None,
            ignore_error=True,
            overwrite=False,
            num_workers: int = 1,
            batch_size: int = 1):
    """ Use the model to extract spatial and flow feature vectors, and predictions, and save them to disk.
    Assumes you have a pretrained model, and K classes. Will go through each video in rgbs, run inference, extracting
    the 512-d spatial features, 512-d flow features, K-d probabilities to disk for each video frame.
    Also stores thresholds for later reloading.

    Output file structure (outputs.h5):
        - latent_name
            - spatial_features: (T x 512) neural activations from before the last fully connected layer of the spatial
                model
            - flow_features: (T x 512) neural activations from before the last fully connected layer of the flow model
            - logits: (T x K) unnormalized logits from after # The above code is not valid Python
            # code. It contains syntax errors.
            the fusion layer
            - P: (T x K) values after the activation function (specified by final_activation)
            - thresholds: (K,) loaded thresholds that convert probabilities to binary predictions
            - class_names: (K,) loaded class names for your project

    Args:
        rgbs (list): list of input video files
        model (nn.Module): a hidden-two-stream deepethogram model
            see deepethogram/feature_extractor/models/hidden_two_stream.py
        final_activation (str): one of sigmoid or softmax
        thresholds (np.ndarray): array of shape (K,), thresholds between 0 and 1 that turns probabilities into
            binary predictions
        fusion (str): one of [average, concatenate]
        num_rgb (int): number of input images to your model
        latent_name (str): an HDF5 group with this name will be in your output HDF5 file.
        class_names (list): a list of class names. Will be saved so that this HDF5 file can be read without any project
            configuration files
        device (str): cuda device on which models will be run
        transform (transforms.Compose): data augmentation. Since this is inference, should only include resizing,
            cropping, and normalization
        ignore_error (bool): if True, an error on one video will not stop inference
        overwrite (bool): if an HDF5 group with the given latent_name is present in the HDF5 file:
            if True, overwrites data with current values. if False, skips that video
    """
    # make sure we're using CUDNN for speed
    torch.backends.cudnn.benchmark = True

    assert isinstance(model, torch.nn.Module)

    device = torch.device(device)
    if device.type != 'cpu':
        torch.cuda.set_device(device)
    model = model.to(device)
    # double checknig
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    # 16 is a decent trade off between CPU and GPU load on datasets I've tested
    if batch_size == 'auto':
        batch_size = 16
    batch_size = min(batch_size, 16)
    log.info('inference batch size: {}'.format(batch_size))

    # iterate over movie files
    for i in tqdm(range(len(rgbs))):
        rgb = rgbs[i]

        basename = os.path.splitext(rgb)[0]
        # make the outputfile have the same name as the video, with _outputs appended
        h5file = basename + '_outputs.h5'

        # iterate over each frame of the movie
        outputs = predict_single_video(rgb,
                                       model,
                                       num_rgb,
                                       mean_by_channels,
                                       device,
                                       cpu_transform,
                                       gpu_transform,
                                       should_print=i == 0,
                                       num_workers=num_workers,
                                       batch_size=batch_size)
        if i == 0:
            for k, v in outputs.items():
                log.info('{}: {}'.format(k, v.shape))
                if k == 'debug':
                    log.debug('All should be 1.0: min: {:.4f} mean {:.4f} max {:.4f}'.format(
                        v.min(), v.mean(), v.max()))

        # if running inference from multiple processes, this will wait until the resource is available
        has_worked = False
        while not has_worked:
            try:
                f = h5py.File(h5file, 'w')
            except OSError as e:
                log.warning('resource unavailable, waiting 30 seconds...')
                time.sleep(30)
            else:
                has_worked = True

        # these assignments are where it's actually saved to disk
        group = f.create_group('encoder')
        group.create_dataset('features', data=outputs['features'], dtype=np.float32)
        group.create_dataset('logits', data=outputs['logits'], dtype=np.float32)
        print(outputs['features'])
        print(outputs['features'].shape)
        del outputs
        f.close()


def encoder_inference(cfg: DictConfig):
    """Runs inference on the feature extractor from an OmegaConf configuration. 

    Parameters
    ----------
    cfg : DictConfig
        Configuration, e.g. that returned by deepethogram.configuration.make_feature_extractor_inference_cfg

    Raises
    ------
    ValueError
        cfg.inference.directory_list must contain a list of input directories, or 'all'
    ValueError
        Checks directory list types
    """
    BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    ## could be applied instead if weights are located in local memory
    # model = torch.hub.load('dinov2', backbone_name, source='local', pretrained=False) 
    # model.load_state_dict(torch.load('dinov2_vitl14.pth'))
    
    device = torch.device('cuda:{}'.format(cfg.compute.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    cfg = projects.setup_run(cfg)
    # turn "models" in your project configuration to "full/path/to/models"
    log.info('args: {}'.format(' '.join(sys.argv)))

    log.info('configuration used in inference: ')
    log.info(OmegaConf.to_yaml(cfg))
    directory_list = cfg.inference.directory_list
    
    if 'sequence' not in cfg.keys() or 'latent_name' not in cfg.sequence.keys() or cfg.sequence.latent_name is None:
        latent_name = cfg.feature_extractor.arch
    else:
        latent_name = cfg.sequence.latent_name
    log.info('Latent name used in HDF5 file: {}'.format(latent_name))
    directory_list = cfg.inference.directory_list

    if directory_list is None or len(directory_list) == 0:
        raise ValueError('must pass list of directories from commmand line. '
                         'Ex: directory_list=[path_to_dir1,path_to_dir2]')
    elif type(directory_list) == str and directory_list == 'all':
        basedir = cfg.project.data_path
        directory_list = utils.get_subfiles(basedir, 'directory')
    elif isinstance(directory_list, str):
        directory_list = [directory_list]
    elif isinstance(directory_list, list):
        pass
    elif isinstance(directory_list, ListConfig):
        directory_list = OmegaConf.to_container(directory_list)
    else:
        raise ValueError('unknown value for directory list: {}'.format(directory_list))

    # video files are found in your input list of directories using the records.yaml file that should be present
    # in each directory
    records = []
    for directory in directory_list:
        assert os.path.isdir(directory), 'Not a directory: {}'.format(directory)
        record = projects.get_record_from_subdir(directory)
        assert record['rgb'] is not None
        records.append(record)

    # get the validation transforms. should have resizing, etc
    cpu_transform = get_cpu_transforms(cfg.augs)['val']
    gpu_transform = get_gpu_transforms(cfg.augs)['val']
    log.info('gpu_transform: {}'.format(gpu_transform))

    rgb = []
    for record in records:
        rgb.append(record['rgb'])
        
    num_images_receptive_field = 1
    class_names = list(cfg.project.class_names)
    class_names = np.array(class_names)

    extract(rgb,
            model,
            mean_by_channels=cfg.augs.normalization.mean,
            num_rgb=num_images_receptive_field,
            latent_name=latent_name,
            class_names=class_names,
            device=device,
            cpu_transform=cpu_transform,
            gpu_transform=gpu_transform,
            ignore_error=cfg.inference.ignore_error,
            overwrite=cfg.inference.overwrite,
            num_workers=cfg.compute.num_workers,
            batch_size=cfg.compute.batch_size)


if __name__ == '__main__':
    project_path = projects.get_project_path_from_cl(sys.argv)
    encoder_inference(project_path)
