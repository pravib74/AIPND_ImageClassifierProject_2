#!/usr/bin/env python3
# -*- coding: utf-8 -*-
                                                                     
# PROGRAMMER: Praveen B
# DATE CREATED: 17-May-2019                                 
# REVISED DATE: 19-May-2019
# PURPOSE: to predict class using stored trained data
# 
#Basic usage: python predict.py /path/to/image checkpoint
    #Options:
    #Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
    #Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    #Use GPU for inference: python predict.py input checkpoint --gpu

##
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# Imports python modules
import os
import torch
import torch.nn.functional as F
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)

import numpy as np
from PIL import Image
# Imports here
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Imports functions created for this program
from get_input_args import get_input_args4prediction

# Main program function defined below
def main():
        
    # This function retrieves y Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args4prediction()
    print(f'working with below inputs')
    print(in_arg.input)
    print(in_arg.chkpoint_file)
    print(in_arg.device)
    print(in_arg.cat_file) 
    print(in_arg.top_k)      
    
    #file for saving model parameters
    saved_file = os.path.abspath("./") + "/" + in_arg.chkpoint_file        
    try:
        fh = open(saved_file, 'r')
        fh.close()
        print(f"Model parameter will be loaded from the path and file : {saved_file}")
    except FileNotFoundError:
        print(f"file {saved_file} doesnot exist...exiting the program")
        exit(1)
    
    
    #categary file path
    cat_path = os.path.abspath("./") + "/" + in_arg.cat_file
            
    if not os.path.isfile(cat_path):
        print(f'Save directory "{cat_path}" does not exist')
        exit(1)    
    
    if not os.path.isfile(in_arg.input):
        print(f'Category directory "{in_arg.input}" does not exist')
        exit(1)
    
    #loadig original flower names
    with open(in_arg.cat_file, 'r') as f:
        cat_to_name = json.load(f)
        
    # Check which device needs to be used. 
    if(in_arg.device == "gpu"):
        if(torch.cuda.is_available()):
            device ="cuda"
        else:
            print(f'Sorry GPU is not available, exiting program...')
            exit(1)
    elif(in_arg.device == "cpu"):
        device = "cpu"
    else:
        print(f'{in_arg.device} not supported, exiting program...')
        exit(1)
        
    #prediction steps
    print("working to predcit, please wait !")
    
    #load model
    model = load_checkpoint(saved_file,device)
        
    #predict flower class
    top_probs, top_lables=predict(in_arg.input,model,cat_to_name,in_arg.top_k)

    #convert probabilities tensor to list
    t_ps=top_probs.to('cpu')
    ps = t_ps.detach().numpy()[0].tolist()

    #plot original flower
    fig, ax = plt.subplots()
    #flower class index from path
    flowerIdx=in_arg.input.split('/')[-2]
    #plot title
    plt_title=cat_to_name[flowerIdx]

    #process the image
    image = process_image(in_arg.input)
    #plot the image
    imshow(image,ax,title=plt_title)
    

    #plotting prediction
    index = np.arange(len(top_lables))
    fig, ax = plt.subplots()
    plt.barh(index, ps)
    plt.xlabel('Prediction', fontsize=10)
    plt.ylabel('Flowers', fontsize=10)
    plt.yticks(index, top_lables, fontsize=10)
    plt.show()
    
    print(type(ps))
    print(f'Input image class : {plt_title}')
    print(f'Predicted image class : {top_lables[0]}')
    print(f'Top {in_arg.top_k} predictions are : {top_lables}')
    print(f'Top {in_arg.top_k} probabilities are : {ps}%')
    
    print("All done...")
    return ps[0],top_lables

#--------------------------------------------------------------------------------
# function  loads a checkpoint and rebuilds the model
def load_checkpoint(filepath,device):
        
    #solution from pytorch discussion, for CUDA runtime error 31
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)    
    
    model = models.vgg19(pretrained=True)    
    model.to(device)
                    
    model.classifier = checkpoint['model_classifier']          
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']   
    
    #loading optimizer gives error vgg19 doesnit require optimer. to be solved
    #model.load_optimizer_state_dict(checkpoint['opt_state_dict'])
    
    print(f'Model load from {filepath}....is completed')
    return model
#--------------------------------------------------------------------------------
#function to preprocess the input image, preparing for prediction
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
       
    pil_image = Image.open(image)
    
    #reducing size
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((10000,256),Image.ANTIALIAS)
    else:
        pil_image.thumbnail((256,10000),Image.ANTIALIAS)
    #inspiration from Deeping Learning using pytorch article in medium by Josh Bernhard 
    #croping
    lmargine = (pil_image.width-224)/2
    rmargine = lmargine+224
    bmargine = (pil_image.height-224)/2
    tmargine = bmargine+224
    
    cropped_im=pil_image.crop((lmargine,bmargine,rmargine,tmargine))    
            
    #converting image to numpy array
    numpy_image = np.array(cropped_im)    
       
    #converting encoded values to float between 0 -1
    numpy_im= numpy_image/255
    
    #normalising 
    numpy_im = (numpy_im - mean)/std
    
    #transpose
    tp_image = numpy_im.transpose((2, 0, 1)) 
    
    return(tp_image) 
#--------------------------------------------------------------------------------
#function to display processed image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    plt.title(title, fontsize=10)
    ax.imshow(image)
    
    return ax
#--------------------------------------------------------------------------------
#function for predicition
def predict(image_path, model,cat_file, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    #list to hold flower classes
    lables = []
    
    model.eval()    
    model.cpu()
    
    image=process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    
    input =image_tensor.unsqueeze(0)
            
    with torch.no_grad():
        output = model.forward(input)
        ps = torch.exp(output)
        top_probs, top_lables = ps.topk(topk)
        
    #convert values to key, to match idx as in cat_to_name file
    for class_idx in np.array(top_lables).flatten():
        for key, value in model.class_to_idx.items():
            if class_idx == value:
                lables.append(cat_file[key])
                
    print(f'Top 5 probability and classes {top_probs[0]} {lables}')
    return top_probs, lables   

# Call to main function to run the program
if __name__ == "__main__":
    main()