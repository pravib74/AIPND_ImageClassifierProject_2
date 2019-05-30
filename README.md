# AIPND_ImageClassifierProject_2
This is a project done for udacity AI WITH PYTHON PROGRAMMING course.
Project aim is to use a pretrained ined network like vgg19 to classify flowers.
Using pretrained build classifier, train the classifier, validate the accuracy, Fino utility trading loss.
Once trained use test data to test.
A predict function is developed to predict input image class and its probability.
1.imageclassier jupiter notebook
2.train.py - trains the model
3.predict.py - predicts the input image
4.get_input_args.py - for praising input
5.cat_to_name.json - original class lables
6.work_utilities.py - provided by udacity useful for keeping workspace awake when training for long
You will need image data set from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html for training.
Procedure for running:
first time you have to train the model (training set not uploaded due to huge size) using image data set. 
for example python train.py (with appropriate input arguements) this will generate checkpoint.pth which contains weights and other parameters 
Now you can use predict.py with a flower image as input (with appropriate input arguements) to predict the flower
Note: training has to be done on  CUDA=GPU otherwise it may take lot of time for training.
prediction can be run on normal CPU
