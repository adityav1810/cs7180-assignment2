# cs7180-assignment2
Homework 2 : Color Constancy 
Authors : Luv Verma and Aditya Varshney
----------------------

This repository holds the entire codebase of our submission for Assigment 2.
Please refer to the pdf file for a full project report summary



---------------------------
Instructions to run the code 
1. setup pytorch
2. Download dataset from https://www2.cs.sfu.ca/~colour/data/shi_gehler/ and unzip it in dataset/images
4. preprocess the images by running preprocess.py
5. unzip the trained_model folder to get trained weights for the model with and without attention
6. use the terminal to run the model in train, test or validate mode

 train the model
-------------------------

python train.py 

test
-----------------------
python test.py 

validate
---------------------------
python visualize.py 
