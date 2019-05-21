#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Praveen B
# DATE CREATED: 17-May-2019                                   
# REVISED DATE: 19-May-2019
# PURPOSE: Create a function that retrieves the command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. 
 
# Imports python modules
import argparse

#function for getting arguments for training module 
def get_input_args4training():
    """
    # Basic usage: python train.py data_directory
    # Options:
    # Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    # Choose architecture: python train.py data_dir --arch "vgg13"
    # Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    # Use GPU for training: python train.py data_dir --gpu
    # 1. data_dir : default value : 'flowers' 
    # 2. save_dir : default value : root dir
    # 3. arch: default value : vgg19
    # 4. learning_rate : default value : 0.003
    # 5. hidden_units : default value : 512
    # 6. epochs : default value : 20
    # 7. device : default value : gpu
    # 8. cat_file : default value : 'cat_to_name.json'
    """    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 7 command line arguments as mentioned above using add_argument() from ArguementParser method    
    #Argument #1
    parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'path to the folder of image data')
    #Argument #2
    parser.add_argument('--save_dir', type = str, default = './', help = 'path to the folder where model parameters have to be stored')
    #Argument #3
    parser.add_argument('--arch', type = str, default = 'vgg19', help = 'name of clasifier architecture to be used. Currently supports only vgg19')    
    #Argument #4
    parser.add_argument('--learning_rate', type = float, default = 0.003, help = 'value for learning rate')
    #Argument #5
    parser.add_argument('--hidden_units', type = int, default = 4096, help = 'number of hidden layers')
    #Argument #6
    parser.add_argument('--epochs', type = int, default = 20, help = 'number of time training has to be done')
    #Argument #7
    parser.add_argument('--device', type = str, default = 'gpu', help = 'device where network has to be trained, gpu/cpu') 
    #Argument #8
    parser.add_argument('--cat_file', type = str, default = 'cat_to_name.json', help = 'Categories to name file, for labels') 
    
    return parser.parse_args()

#function for getting arguments for prediction module 
def get_input_args4prediction():
    """
    #Basic usage: python predict.py /path/to/image checkpoint
    #Options:
    #Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
    #Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    #Use GPU for inference: python predict.py input checkpoint --gpu
    # 1. input : default value : 'flowers/train/100/image_07898.jpg' 
    # 2. chkpoint_file : default value : checkpoint.pth
    # 3. device : default value : gpu
    # 4. cat_file : default value : 'cat_to_name.json'
    # 5. top_k : default value : 5
    """    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method    
    #Argument #1
    parser.add_argument('--input', type = str, default = 'flowers/train/100/image_07898.jpg', help = 'image file to be predicted')
    #Argument #2
    parser.add_argument('--chkpoint_file', type = str, default = 'checkpoint.pth', help = 'name of the checkpoint file. Should be in same dir as predict.py')
    #Argument #3
    parser.add_argument('--device', type = str, default = 'cpu', help = 'device where network has to be trained, gpu/cpu')
     #Argument #4
    parser.add_argument('--cat_file', type = str, default = 'cat_to_name.json', help = 'Categories to name file, for labels')
    #Argument #5
    parser.add_argument('--top_k', type = str, default = 5, help = 'top x classes to be returned')
    
    return parser.parse_args()