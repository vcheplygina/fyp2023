
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Default packages for the minimum example
from skimage import morphology #for measuring things in the masks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit #for cross-validation
from sklearn.metrics import accuracy_score #for measuring performance


# Functions you will implement during your project work
import groupXY_functions as util


import pickle #for saving/loading trained classifiers



# Load the data you saved, then do some analysis
df_features = pd.read_csv(file_features)
image_id = list(df_features['id'])
features_area = np.array(df_features['area'])
features_perimeter = np.array(df_features['perimeter'])


# Load features and labels for the melanoma task
x = df_features.iloc[:,1:].to_numpy()
y = is_melanoma


#Prepare cross-validation
n_splits=5
kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.4, random_state=1)

acc_val = np.empty([n_splits,1])
acc_test = np.empty([n_splits,1])

index_fold = 0

#Parameter for nearest neighbor classifier
k = 5

# Predict labels for each fold using the KNN algortihm
for train_index, test_val_index in kf.split(x, y):
    
    
    # split dataset into a train, validation and test dataset
    test_index, val_index = np.split(test_val_index, 2)
    
    x_train, x_val, x_test = x[train_index,:], x[val_index,:], x[test_index,:]
    y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train,y_train)
    
    y_pred_val = classifier.predict(x_val)
    y_pred_test = classifier.predict(x_test)
    
    # Calculate some performance metric
    acc_val[index_fold] = accuracy_score(y_val, y_pred_val)
    index_fold += 1
    
print(acc_val)

#Let's say you now decided to use the 5-NN with al the features...
classifier = KNeighborsClassifier(n_neighbors = 5)

#It will be tested on external data, so we can try to maximize the use of our available data by training on 
#all of x and y
classifier = classifier.fit(x,y)

#This is the classifier you need to save using pickle, add this to your zip file submission
filename = 'groupXY_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))



