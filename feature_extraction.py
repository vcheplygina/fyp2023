import numpy as np
import pandas as pd
from collections import Counter
from skimage import io, filters, morphology, segmentation, img_as_ubyte, transform, color
import matplotlib.pyplot as plt
from skimage.draw import polygon

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib

import os
import shutil
import pandas as pd

np.random.seed(1)



def make_df():
  path = os.path.join(os.getcwd(), "metadata.csv")
  df = pd.read_csv(path)
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


def make_datasample_symetry(img, name):
  asym = check_asymmetry(img)
  x = [asym]

  if (final_data[final_data["img_id"] == name]["healthy"] == 1).bool():
    y = 1
  else:
    y = 0
  
  return [x, y]

def make_datasample(img, name):
  # asym = check_asymmetry(img)
  col = slic_samples(img)
  common_shape = (50, 27, 3)

  col = np.pad(col, [(0, common_shape[0] - col.shape[0]),
                                  (0, common_shape[1] - col.shape[1]),
                                  (0, common_shape[2] - col.shape[2])], mode='constant')

  col = col.ravel()
  x = col

  if (final_data[final_data["img_id"] == name]["healthy"] == 1).bool():
    y = 1
  else:
    y = 0
  
  return [x, y]


def build_datasample_new():
  path = os.path.join(os.getcwd(), "segmented_photos")
  arr = []

  for i in os.listdir(path):
    image = io.imread(os.path.join(path, i))
    image = transform.resize(image, (200, 200), anti_aliasing=True)

    arr.append(make_datasample(image, i))

  np.random.shuffle(arr)
  return arr


def build_datasample_asym_new():
  path = os.path.join(os.getcwd(), "segmented_photos")

  arr = []

  for i in os.listdir(path):
    image = io.imread(os.path.join(path, i))
    image = transform.resize(image, (200, 200), anti_aliasing=True)

    arr.append(make_datasample_symetry(image, i))

  np.random.shuffle(arr)
  return arr












final_data = select_data(make_df())




arr_col = build_datasample_new()
arr_col = pd.DataFrame(arr_col)
arr_col.to_csv('color_feature_csv.csv')

arr_asym = build_datasample_asym_new()
arr_asym = pd.DataFrame(arr_col)
arr_asym.to_csv('assymetry_feature_csv.csv')


print(arr_col)