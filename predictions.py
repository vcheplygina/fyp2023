import numpy as np
from skimage import io, filters, morphology, segmentation, transform, color
import matplotlib.pyplot as plt
from skimage.draw import polygon


import joblib


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

def calc_correlation(vertical, horizontal):
  # Normalize the histograms to ensure they represent probability distributions
    hist1_normalized = vertical / np.sum(vertical)
    hist2_normalized = horizontal / np.sum(horizontal)

    # Compute the Bhattacharyya coefficient
    bc = np.sum(np.sqrt(hist1_normalized * hist2_normalized))

    # Compute the Bhattacharyya distance
    bd = -np.log(bc)

    return bd

def make_projections(bin_img):
  binary_image = bin_img.copy()

  vertical = np.sum(binary_image, 0)
  horizontal = np.sum(binary_image, 1)

  return vertical, horizontal

def check_asymmetry(masked_img):
  vert, horiz = make_projections(masked_img.astype("double"))
  corr = calc_correlation(vert, horiz)

  return corr

def slic_samples(img):
  new_img = img.copy()
  new_img = new_img[:, :, :3]

  foreground_mask = np.all(new_img != [0, 0, 0], axis=-1)

  segments = segmentation.slic(new_img * foreground_mask[..., np.newaxis], n_segments=36, compactness=3)

  mean_colours = np.zeros((np.max(segments)+1, 3))

  for label in enumerate(np.unique(segments)):
    mask = segments == label[1]
    mean_colours[label[0], :] = new_img[mask].mean(axis=0)

  palette_height, palette_width = 50, 300
  colours = mean_colours[np.all(mean_colours, axis=1)]
  color_palette = np.zeros((palette_height, len(colours), 3))

  for i in range(len(colours)):
    color_palette[:, i, :] = colours[i]

  return color_palette




def predict_mask_col(img):
  rndF = joblib.load('C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\random_forest_model_col.pkl')
  image = transform.resize(img, (200, 200), anti_aliasing=True)

  col = slic_samples(image)
  common_shape = (50, 27, 3)

  col = np.pad(col, [(0, common_shape[0] - col.shape[0]),
                                  (0, common_shape[1] - col.shape[1]),
                                  (0, common_shape[2] - col.shape[2])], mode='constant')

  col = col.ravel()
  print(col)
  result = rndF.predict([col])

  return result


def predict_mask_cor(img):
  rndF = joblib.load('C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\random_forest_model_cor.pkl')
  image = transform.resize(img, (200, 200), anti_aliasing=True)
  asym = check_asymmetry(image)
  asym = [asym]

  result = rndF.predict(np.array([asym]))

  return result


def predict_img_col(img):
  rndF = joblib.load('C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\random_forest_model_col.pkl')
  img = img[:, :, :3]
  image = do_segm(img)
  plt.imshow(image)
  plt.show()
  col = slic_samples(image)
  common_shape = (50, 27, 3)

  col = np.pad(col, [(0, common_shape[0] - col.shape[0]),
                                  (0, common_shape[1] - col.shape[1]),
                                  (0, common_shape[2] - col.shape[2])], mode='constant')

  col = col.ravel()
  result = rndF.predict([col])

  return result


def predict_img_cor(img):
    rndF = joblib.load('C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\random_forest_model_cor.pkl')
    image = do_segm(img)
    asym = check_asymmetry(image)
    asym = [asym]

    result = rndF.predict(np.array([asym]))

    return result