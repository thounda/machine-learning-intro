#!/usr/bin/python3

'''
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

INSTRUCTIONS
To create and train a Naive Bayes classifier in your naive_bayes/nb_author_id.py file, you'll typically follow these steps:

Import Necessary Libraries: Make sure to import the required libraries, such as numpy, sklearn.naive_bayes, and any other libraries you need for data handling.

Load the Data: Load your training and test datasets. This is usually done using a function that preprocesses the email data.

Create the Classifier: Instantiate the Naive Bayes classifier.

Train the Classifier: Fit the classifier on your training data.

Make Predictions: Use the trained classifier to make predictions on the test set.

Calculate Accuracy: Compare the predicted labels with the actual labels to calculate accuracy.

TIP
1 Ensure that you have enough memory available, as some students have encountered memory issues. If you do face memory errors, consider setting test_size = 0.5 in your email_preprocess.py file, as mentioned in the course material.

2 You can also explore different types of Naive Bayes classifiers, such as MultinomialNB or BernoulliNB, depending on your data characteristics.

'''

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels

##############################################################
# Enter Your Code Here
# Load modules

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 
import sys
import os
from time import time

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the tools directory
tools_path = os.path.join(script_dir, '../tools')

# Add the tools directory to the system path
sys.path.append(tools_path)
# sys.path.append("../tools/") - REMOVE THIS STATEMENT AFTER CONFIRMED WORKING
# Now you can import correctly
from email_preprocess import preprocess

### Load the data
features_train, features_test, labels_train, labels_test = preprocess()

# Create the classifier
clf = GaussianNB()

# Train the classifier
clf.fit(features_train, labels_train)

# Make predictions
predictions = clf.predict(features_test)

# Calculate accuracy
accuracy = accuracy_score(labels_test, predictions)
print("Accuracy:", accuracy)


##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################