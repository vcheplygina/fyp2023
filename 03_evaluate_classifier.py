#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:18:43 2023

@author: vech
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 08:36:13 2022

@author: vech
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import morphology #for measuring things in the masks


import pickle #for saving/loading trained classifiers


#Import anything you used here 


#The function that should classify new images. The image and mask are the same size, and are already loaded using plt.imread
def classify(img, mask):
    
    
     #Extract features (the same ones that you used for training)
     x = util.extract_features(img, mask)
         
     
     #Load the trained classifier
     classifier = pickle.load(open('groupXY_classifier.sav', 'rb'))
    
    
     #Use it on this example to predict the label 
     pred_label = classifier.predict(x)
     pred_prob = classifier.predict_proba(x)
     
     
     #print('predicted label is ', pred_label)
     #print('predicted probability is ', pred_prob)
     return pred_label, pred_prob
 
    
