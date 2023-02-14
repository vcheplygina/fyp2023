"""
Main script for the FYP project imaging
"""

import os
import pandas as pd
import numpy as np

# Import packages for image processing
from skimage import morphology #for measuring things in the masks


#Define some functions    
def extract_features(image, mask):
    
    [a,p] = measure_area_perimeter(mask)
   return [[a,p]]

    #TODO here you need to call more of your custom-made functions for measuring features!


def measure_area_perimeter(mask):
    # Measure area: the sum of all white pixels in the mask image
    area = np.sum(mask)

    # Measure perimeter: first find which pixels belong to the perimeter.
    struct_el = morphology.disk(1)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    image_perimeter = mask - mask_eroded

    # Now we have the perimeter image, the sum of all white pixels in it
    perimeter = np.sum(image_perimeter)

    return area, perimeter




#Where is the raw data
file_data = 'data/example_ground_truth.csv'
path_image = 'data/example_image'
path_mask = 'data/example_segmentation'


#Where we will store the features
file_features = 'features/features.csv'



#Read meta-data into a Pandas dataframe
df = pd.read_csv(file_data)

# Extract image IDs and labels from the data
image_id = list(df['image_id'])
is_melanoma = np.array(df['melanoma'])
is_keratosis = np.array(df['seborrheic_keratosis'])

num_images = len(image_id)

#Make empty array to store features
features = np.empty([num_images,2])

#Loop through all images
for i in np.arange(num_images):
    
    # Define filenames related to this image
    file_image = path_image + os.sep + image_id[i] + '.jpg'
    file_mask = path_mask + os.sep + image_id[i] + '_segmentation.png'
    
    # Read the images with these filenames
    im = plt.imread(file_image)
    mask = plt.imread(file_mask)
    
    # Measure features - inside this function you should add more features! 
    x = util.measure_features(img,mask) 
       
    
    # Store in the variable we created before
    features[i,:] = x
    
#Save the features to a file     
df_features.to_csv(file_features, index=False)  
    

