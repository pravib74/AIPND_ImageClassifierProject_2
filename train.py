#!/usr/bin/env python3
# -*- coding: utf-8 -*-
                                                                     
# PROGRAMMER: Praveen B
# DATE CREATED: 17-May-2019                                 
# REVISED DATE: 20-May-2019
# PURPOSE: train a network on given set of data, save the training data
# 
# Basic usage: python train.py data_directory
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu

##

# Imports python modules
import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
# Imports here
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

# Imports functions created for this program
from get_input_args import get_input_args4training
from workspace_utils import active_session

# Main program function defined below
def main():
        
    # This function retrieves y Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args4training()
    print(f"Working below input parameters")
    print(in_arg.data_dir)
    print(in_arg.save_dir)
    print(in_arg.arch)
    print(in_arg.hidden_units)
    print(in_arg.epochs)
    print(in_arg.device)
    print(in_arg.learning_rate)
    print(in_arg.cat_file)
    
    if(in_arg.arch != "vgg19"):
        print(f'Sorry doesnot support {in_arg.arch}, exiting program...')
        exit(1)
    
    #training and validation data directories
    train_dir = in_arg.data_dir + '/train'
    valid_dir = in_arg.data_dir + '/valid'
    
    #file for saving model parameters
    filename = "checkpoint1.pth"
    if(in_arg.save_dir == "./"):
        save_file = os.path.abspath(in_arg.save_dir) + '/' + filename
    else:
        save_file = in_arg.save_dir + '/' + filename
    print(f"Model parameter save path and file is {save_file}")
    
    #categary file path
    cat_path = os.path.abspath("./") + "/" + in_arg.cat_file
    
    
    if not os.path.isdir(train_dir):
        print(f'Train directory "{train_dir}" does not exist')
        exit(1)
    
    if not os.path.isdir(valid_dir):
        print(f'Validation directory "{valid_dir}" does not exist')
        exit(1)
        
    if not os.path.isdir(in_arg.save_dir):
        print(f'Save directory "{in_arg.save_dir}" does not exist')
        exit(1)
        
    if not os.path.isfile(cat_path):
        print(f'Category directory "{in_arg.cat_file}" does not exist')
        exit(1)
        
    if(in_arg.epochs > 100):
        print(f'Epochs of {in_arg.epochs} is too large to use')
        exit(1)
    
    #data transformations 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    trainData_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,])
    
    #for test and validation transforms, no rotation, cropping and flips
    test_val_Data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize,]) 
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=trainData_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=test_val_Data_transforms)
                                              

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valloader =  torch.utils.data.DataLoader(val_data, batch_size=64)           
    
    #loadig original flower names
    with open(in_arg.cat_file, 'r') as f:
        cat_to_name = json.load(f)
    #find out size of output
    num_of_cat = len(cat_to_name)
    
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
    
    #load pretrained module
    model = models.vgg19(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    #define model for classifier training
    fc = nn.Sequential(nn.Linear(25088, in_arg.hidden_units),
                   nn.ReLU(),
                   nn.Dropout(0.15),
                   nn.Linear(in_arg.hidden_units, num_of_cat),
                   nn.LogSoftmax(dim=1))


    model.classifier = fc
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)

    model.to(device);
    
    print(f'Started training with {in_arg.arch} pretrained model...')
    
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    
    with active_session():
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                sys.stdout.write("*")
                sys.stdout.flush()
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                #do validation every 5 steps once...
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in valloader:
                            sys.stdout.write("#")
                            sys.stdout.flush()
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(valloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(valloader):.3f}")
                    running_loss = 0
                    model.train()
        print("Training complete...")
    
    #save all hperparameters and model state dictionary
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 25098,
              'hidden_layer': in_arg.hidden_units,
              'output_size': num_of_cat,              
              'class_to_idx': model.class_to_idx,
              'epoch':in_arg.epochs,
              'learning_rate':in_arg.learning_rate,
              'state_dict': model.state_dict(),
              'opt_state_dict' : optimizer.state_dict(),
              'model_classifier' : model.classifier
             }
    print(f'Saving model parameters to {save_file}...')
    torch.save(checkpoint, save_file)
    
    print("All done...")
    return None
    
    
    

# Call to main function to run the program
if __name__ == "__main__":
    main()