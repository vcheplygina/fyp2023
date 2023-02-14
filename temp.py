#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:59:44 2023

@author: vech
"""



# Store these features so you can reuse them later
#feature_data = {"id": image_id, 
#                "area": features_area.flatten(),
#                "perimeter": features_perimeter.flatten()
#                }

#df_features = pd.DataFrame(feature_data)
#df_features.to_csv(file_features, index=False)    


# Display the features measured in a scatterplot
#axs = util.scatter_data(features_area, features_perimeter, is_melanoma)
#axs.set_xlabel('X1 = Area')
#axs.set_ylabel('X2 = Perimeter')
#axs.legend()


#  #-------This part is just for testing the function in class
# file_data = 'data/example_ground_truth.csv'
# path_image = 'data/example_image'
# path_mask = 'data/example_segmentation'
# file_features = 'features/features.csv'

# df = pd.read_csv(file_data)

# #We just test on one image
# image_id = list(df['image_id'])
# i = 0

# file_image = path_image + os.sep + image_id[i] + '.jpg'
# file_mask = path_mask + os.sep + image_id[i] + '_segmentation.png'
# img = plt.imread(file_image)
# mask = plt.imread(file_mask)


# classify(img,mask)

#-------End of part for testing 