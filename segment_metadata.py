import numpy as np
import pandas as pd

from skimage import io, filters, morphology, segmentation, transform, color
import matplotlib.pyplot as plt
from skimage.draw import polygon

import os
import pandas as pd


def do_segm(img):

  img_col = transform.resize(img, (200, 200), anti_aliasing=True)

  image = color.rgb2gray(img)
  image = transform.resize(image, (200, 200), anti_aliasing=True)
  image = filters.gaussian(image)
  
  # thresholding the image 
  thresholds = filters.threshold_multiotsu(image, classes=3)
  regions = np.digitize(image, thresholds)
  output = regions < 1


  # making a circle
  s = np.linspace(0, 2*np.pi, 100)   #Number of points on the circle
  y = 100 + 58*np.sin(s)            #Row 
  x = 100 + 70*np.cos(s)            #Column
  init = np.array([y, x]).T

  # and a snake
  snake = segmentation.active_contour(output, init, w_line=0)

  # Find coordinates inside the polygon defined by the snake
  rr, cc = polygon(snake[:, 0], snake[:, 1], output.shape)
  mask = np.zeros_like(output)

  # applying a mask on the polygon area
  mask[rr, cc] = 1
  cropped_img2 = output * mask[:, :]

  # doing dilation
  struct_el = morphology.disk(4)
  mask_dilated = morphology.binary_dilation(cropped_img2, struct_el)

  # applying the mask and extracting the lesion from the image
  im2 = img_col.copy()
  im2[mask_dilated == 0] = 0

  return im2

def make_df():
  df = pd.read_csv(os.path.join(os.getcwd(), "metadata.csv"))
  new_df = df[["patient_id", "img_id", "diagnostic"]]

  new_df["healthy"] = np.where(new_df["diagnostic"] == "NEV", 1, 0) 
  return new_df 


def select_data(new_df):
  final_data = new_df[new_df["healthy"] == 1]
  filtered_data = new_df[new_df["healthy"] == 0]

  random_rows = filtered_data.sample(n = 244, random_state=42)
  final_data = pd.concat([final_data, random_rows])
  final_data = final_data.set_index("patient_id")

  final_data = final_data.sample(frac=1)

  return final_data


def do_metadata_segm(data):
  path = os.path.join(os.getcwd(),"imgs_part_1")
  images = data["img_id"].tolist()
  segmented_imgs = []

  for i in images:
    img = io.imread(os.path.join(path, i))
    image = transform.resize(img, (200, 200), anti_aliasing=True)

    fig, ax = plt.subplots(1, 1)
    ax[0].imshow(do_segm(image))

    fig.savefig(os.path.join(os.getcwd(), "segmented_photos"))