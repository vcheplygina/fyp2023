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


def img_segm2(path, num):

  # reading a colourful image and resizing it
  img_col = io.imread(path)
  img_col = transform.resize(img_col, (200, 200), anti_aliasing=True)


  # reading a grayscale image and resizing it
  image = io.imread(path, as_gray=True)
  image = transform.resize(image, (200, 200), anti_aliasing=True)
  image = filters.gaussian(image)
  

  # thresholding the image 
  thresholds = filters.threshold_multiotsu(image, classes=num)
  print(thresholds)
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
  

  # plotting results
  fig, ax = plt.subplots(1, 2, figsize=(15, 8))
  ax[0].imshow(img_col, cmap='gray')
  ax[1].imshow(im2, cmap='gray')          # resulting image

  return fig


def img_segm(path, num):
  # reading a colourful image and resizing it
  img_col = io.imread(path)
  img_col = transform.resize(img_col, (200, 200), anti_aliasing=True)


  # reading a grayscale image and resizing it
  image = io.imread(path, as_gray=True)
  image = transform.resize(image, (200, 200), anti_aliasing=True)
  image = filters.gaussian(image)
  

  # thresholding the image 
  thresholds = filters.threshold_multiotsu(image, classes=num)
  print(thresholds)
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
  

  # plotting results
  fig, ax = plt.subplots()

  ax.set_xticks([])
  ax.set_yticks([])

  fig.set_size_inches(2, 2)

  ax.imshow(im2)

  return fig



def make_df():
  df = pd.read_csv("C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\metadata.csv")
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


df = select_data(make_df())
print(df)


def do_img_segmentation(df):

  lst = df["img_id"].tolist()

  for i in lst:
    segmented_img = img_segm2(f"C:\\Users\\dubst\\Desktop\\DataScience\\First Year Project (Second)\\imgs_part_1\\{i}", 3)
    segmented_img.suptitle(i)

    healthy_check = df[df["img_id"] == i]["healthy"] == 0


    if healthy_check.bool():
      out_path = f"C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\healthy\\{i}"
      if not os.path.exists(out_path):
        segmented_img.savefig(out_path)
        plt.close()
    else:
      out_path = f"C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\unhealthy\\{i}"
      if not os.path.exists(out_path):
        segmented_img.savefig(out_path)
        plt.close()


def do_cleaning():
    good = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\good"
    cancel = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\cancel"

    entries = os.listdir(good)
    entries2 = os.listdir(cancel)
    entries3 = []

    entries3.extend(entries)
    entries3.extend(entries2)


    return entries3


def delete_rows(final_data, entries):
  data_clean = final_data[~final_data["img_id"].isin(entries)]

  return data_clean


def change_photos():
  photo_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\good"
  source_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\First Year Project (Second)\\imgs_part_1"

  old_photos = os.listdir(photo_dir)

  for file_name in old_photos:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(photo_dir, file_name)

        if os.path.exists(destination_path):
            os.remove(destination_path) 

        shutil.copy2(source_path, destination_path)


def do_folder_segm():
  photo_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\good"
  photos = os.listdir(photo_dir)

  for i in photos:
    path = os.path.join(photo_dir, i)
    segmented_img = img_segm(path, 3)
    os.remove(path) 
    segmented_img.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def sort_imgs():
  df = select_data(make_df())
  imgs_lst = df["img_id"].tolist()


  photo_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\photo2"
  healthy_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\healthy"
  unheatlhy_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\unhealthy"

  folder_imgs_lst = os.listdir(photo_dir)

  for i in folder_imgs_lst:
    if (df[df["img_id"] == i]["healthy"] == 1).bool():
      shutil.copy2(os.path.join(photo_dir, i), os.path.join(healthy_dir, i))
    else:
      shutil.copy2(os.path.join(photo_dir, i), os.path.join(unheatlhy_dir, i))


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

final_data = select_data(make_df())

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


def build_datasample():
  healthy_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\healthy"
  unheatlhy_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\unhealthy"

  arr = []

  for i in os.listdir(healthy_dir):
    image = io.imread(os.path.join(healthy_dir, i))
    image = transform.resize(image, (200, 200), anti_aliasing=True)

    arr.append(make_datasample(image, i))
  
  for i in os.listdir(unheatlhy_dir):
    image = io.imread(os.path.join(unheatlhy_dir, i))
    image = transform.resize(image, (200, 200), anti_aliasing=True)

    arr.append(make_datasample(image, i))

  np.random.shuffle(arr)
  return arr


def build_datasample_new():
  path = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\segmented_photos"

  arr = []

  for i in os.listdir(path):
    image = io.imread(os.path.join(path, i))
    image = transform.resize(image, (200, 200), anti_aliasing=True)

    arr.append(make_datasample(image, i))

  np.random.shuffle(arr)
  return arr


def build_datasample_asym_new():
  path = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\segmented_photos"

  arr = []

  for i in os.listdir(path):
    image = io.imread(os.path.join(path, i))
    image = transform.resize(image, (200, 200), anti_aliasing=True)

    arr.append(make_datasample_symetry(image, i))

  np.random.shuffle(arr)
  return arr


def build_datasample_asym():
  healthy_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\healthy"
  unheatlhy_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\unhealthy"

  arr = []

  for i in os.listdir(healthy_dir):
    image = io.imread(os.path.join(healthy_dir, i))
    image = transform.resize(image, (200, 200), anti_aliasing=True)

    arr.append(make_datasample_symetry(image, i))
  
  for i in os.listdir(unheatlhy_dir):
    image = io.imread(os.path.join(unheatlhy_dir, i))
    image = transform.resize(image, (200, 200), anti_aliasing=True)

    arr.append(make_datasample_symetry(image, i))

  np.random.shuffle(arr)
  return arr


def make_csv_features(path):
  healthy_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\healthy"
  unhealthy_dir = "C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\unhealthy"
  df = pd.DataFrame([])

  for i in os.listdir(healthy_dir):
    image = io.imread(os.path.join(healthy_dir, i))[:, :, :3]
    image = transform.resize(image, (200, 200), anti_aliasing=True)

    colour = make_datasample(image, i)
    asym = make_datasample_symetry(image, i)
    df["img_id"] = i 
    df["colour"] = colour[0]
    df["asymmetry coef"] = asym[0][0]
    df["healthy"] = asym[1]

  for i in os.listdir(unhealthy_dir):
    image = io.imread(os.path.join(unhealthy_dir, i))[:, :, :3]
    image = transform.resize(image, (200, 200), anti_aliasing=True)

    colour = make_datasample(image, i)
    asym = make_datasample_symetry(image, i)
    df["img_id"] = i 
    df["colour"] = colour[0]
    df["asymmetry coef"] = asym[0][0]
    df["healthy"] = asym[1]

  df.to_csv(os.path.join(path, "output.csv"))


def separate_data(arr):
  x = list()
  y = list()

  for i in arr:
    x.append(i[0])
    y.append(i[1])

  return (x, y)


def do_training(X, Y, n_folds):

  kf = KFold(n_splits=n_folds, shuffle=True)
  clf = NearestCentroid()
  neigh = KNeighborsClassifier(n_neighbors=3)
  rndF = RandomForestClassifier()
  classifier = LogisticRegression(max_iter=207,random_state = 0, penalty = None)

  score_neigh = []
  score_mn = []
  score_rndF = []
  score_logist = []

  fold_prediction_probs = []
  fold_labels = []

  for train_idx, test_idx in kf.split(X):
    
    x_np = np.array(X)
    y_np = np.array(Y)

    X_train, X_test, Y_train, Y_test = x_np[train_idx], x_np[test_idx], y_np[train_idx], y_np[test_idx]
    neigh.fit(X_train, Y_train)
    clf.fit(X_train, Y_train)
    rndF.fit(X_train, Y_train)
    classifier.fit(X_train, Y_train)
    
    score = roc_auc_score(Y_test, neigh.predict_proba(X_test)[:, 1])
    score_neigh.append(score)

    score = roc_auc_score(Y_test, clf.predict(X_test))
    score_mn.append(score)

    prob = rndF.predict_proba(X_test)[:, 1]
    fold_prediction_probs.append(prob)
    fold_labels.append(Y_test)

    score = roc_auc_score(Y_test, prob)
    score_rndF.append(score)

    all_prediction_probs = np.zeros(len(X))
    all_prediction_probs[test_idx] = prob

    score = roc_auc_score(Y_test, classifier.predict_proba(X_test)[:, 1])
    score_logist.append(score)


  print(f"MEAN: {np.mean(score_neigh), np.mean(score_mn), np.mean(score_rndF), np.mean(score_logist)}")


  fig, ax = plt.subplots(1, 4, figsize=(15, 8))
  ax[0].plot(range(1, n_folds+1), score_rndF, marker='o')
  ax[0].set_xlabel('Fold')
  ax[0].set_ylabel('ROC AUC Score')
  ax[0].set_title('Random forest')

  ax[1].plot(range(1, n_folds+1), score_neigh, marker='o')
  ax[1].set_xlabel('Fold')
  ax[1].set_title('KNN')

  ax[2].plot(range(1, n_folds+1), score_logist, marker='o')
  ax[2].set_xlabel('Fold')
  ax[2].set_title('Logistic regression')

  ax[3].plot(range(1, n_folds+1), score_mn, marker='o')
  ax[3].set_xlabel('Fold')
  ax[3].set_title('Closest mean')

  # fig.savefig(f"C:\\Users\\dubst\\Desktop\\lightshot\\fold_cor{n_folds}")

  return rndF


def final_training(arr, n_folds):

  x, y = separate_data(arr)

  x = np.array(x)
  y = np.array(y)
  
  (train_col, test_col, train_lab, test_lab) = train_test_split(
	x, y, test_size=0.25, random_state=42) 

  kf = KFold(n_splits=n_folds)

  clf = NearestCentroid()
  neigh = KNeighborsClassifier(n_neighbors=3)
  rndF = RandomForestClassifier()
  classifier = LogisticRegression(max_iter=207,random_state = 0, penalty = None)

  result_clf = cross_val_score(clf, train_col, train_lab, cv=n_folds)
  result_neigh = cross_val_score(neigh, train_col, train_lab, cv=n_folds)
  result_rndF = cross_val_score(rndF, train_col, train_lab, cv=n_folds)
  result_classifier = cross_val_score(classifier, train_col, train_lab, cv=n_folds)

  clf.fit(train_col, train_lab)
  neigh.fit(train_col, train_lab)
  rndF.fit(train_col, train_lab)
  classifier.fit(train_col, train_lab)
  
  eval_clf = evaluate(clf, test_col, test_lab); print(f"Cross-val: {np.mean(result_clf)}"); print("")
  eval_neigh = evaluate(neigh, test_col, test_lab); print(f"Cross-val: {np.mean(result_neigh)}"); print("")
  eval_rndF = evaluate(rndF, test_col, test_lab); print(f"Cross-val: {np.mean(result_rndF)}"); print("")
  eval_class = evaluate(classifier, test_col, test_lab); print(f"Cross-val: {np.mean(result_classifier)}"); print("")

  # Create a list of labels for the models
  models = ['NC', 'KNN', 'RFC', 'LR']

  # Create a list of metrics for each model
  average_errors = [eval_clf[0], eval_neigh[0], eval_rndF[0], eval_class[0]]
  accuracies = [eval_clf[1], eval_neigh[1], eval_rndF[1], eval_class[1]]
  precisions = [eval_clf[2], eval_neigh[2], eval_rndF[2], eval_class[2]]
  f1_scores = [eval_clf[3], eval_neigh[3], eval_rndF[3], eval_class[3]]
  recalls = [eval_clf[4], eval_neigh[4], eval_rndF[4], eval_class[4]]
  aucs = [eval_clf[5], eval_neigh[5], eval_rndF[5], eval_class[5]]
  cross_vals = [np.mean(result_clf), np.mean(result_neigh), np.mean(result_rndF), np.mean(result_classifier)]

  # Plotting the histograms
  fig, axs = plt.subplots(2, 4, figsize=(12, 5))
  axs = axs.flatten()

  # Histogram for Average Error
  axs[0].bar(models, average_errors)
  axs[0].set_title('Average Error')
  axs[0].set_ylabel('Degrees')

  # Histogram for Accuracy
  axs[1].bar(models, accuracies)
  axs[1].set_title('Accuracy')
  axs[1].set_ylabel('Percentage')

  # Histogram for Precision
  axs[2].bar(models, precisions)
  axs[2].set_title('Precision')
  axs[2].set_ylabel('Percentage')

  # Histogram for F1 Score
  axs[3].bar(models, f1_scores)
  axs[3].set_title('F1 Score')
  axs[3].set_ylabel('Percentage')

  # Histogram for Recall
  axs[4].bar(models, recalls)
  axs[4].set_title('Recall')
  axs[4].set_ylabel('Percentage')

  # Histogram for AUC
  axs[5].bar(models, aucs)
  axs[5].set_title('AUC')
  axs[5].set_ylabel('Percentage')

  axs[6].bar(models, cross_vals)
  axs[6].set_title('Cross-Validation')
  axs[6].set_ylabel('Value')

  return fig


def best_random_forest(arr, n_folds):
  x, y = separate_data(arr)

  x = np.array(x)
  y = np.array(y)
  
  (train_col, test_col, train_lab, test_lab) = train_test_split(
	x, y, test_size=0.25, random_state=42) 

  kf = KFold(n_splits=n_folds)
  
  rndF = RandomForestClassifier(n_estimators = 1200, min_samples_split=5, min_samples_leaf=3, max_features='sqrt', max_depth=30, bootstrap=True)
  result_rndF = cross_val_score(rndF, train_col, train_lab, cv=n_folds)
  
  rndF.fit(train_col, train_lab)

  joblib.dump(rndF, 'C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\random_forest_model_cor.pkl')

  score_rndF = rndF.score(test_col, test_lab)

  eval_rndF = evaluate(rndF, test_col, test_lab)
  
  print(f"Cross-val score for Random Forest {result_rndF.mean()}\t Accuracy for Random Forest {score_rndF}")


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


def evaluate(model, test_features, test_labels):
  predictions = model.predict(test_features)
  errors = abs(predictions - test_labels)
  
  accuracy = accuracy_score(test_labels, predictions)
  precision = precision_score(test_labels, predictions)
  f1_score_res = f1_score(test_labels, predictions)
  recall = recall_score(test_labels, predictions)
  auc = roc_auc_score(test_labels, predictions)

  print(f'Model Performance {type(model).__name__}')
  print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
  print('Accuracy = {:0.4f}%.'.format(accuracy*100))
  print('Precision = {:0.4f}%.'.format(precision*100))
  print('F1 Score = {:0.4f}%.'.format(f1_score_res*100))
  print('Recall = {:0.4f}%.'.format(recall*100))
  print('AUC = {:0.4f}%.'.format(auc*100))
  
    
  return (np.mean(errors), accuracy, precision, f1_score_res, recall, auc)


def try_grid_search(arr):
  x, y = separate_data(arr)

  x = np.array(x)
  y = np.array(y)
  
  (train_col, test_col, train_lab, test_lab) = train_test_split(
	x, y, test_size=0.25, random_state=42) 

  param_grid = {
    'bootstrap': [True],
    'max_depth': [15, 20, 25, 30],
    'max_features': ['sqrt'],
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [1100, 1150, 1200, 1250, 1300]
}
  # Create a based model
  rf = RandomForestRegressor()
  # Instantiate the grid search model
  grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 10, n_jobs = -1, verbose = 2)
  
  # Fit the grid search to the data
  grid_search.fit(train_col, train_lab)

  print(grid_search.best_params_)
  
  best_grid = grid_search.best_estimator_

  score_best = best_grid.score(test_col, test_lab)
  grid_accuracy = evaluate(best_grid, test_col, test_lab)


def RandomSearchCV(x_train,y_train,classifier, param_range, folds):
  # params = np.random.uniform(param_range[0],param_range[1],10)
  # params = np.array([int(i) for i in params])
  params = np.arange(1, param_range, 1)
  # params = np.sort(params)

  kf = KFold(n_splits=folds)

  x_train = pd.DataFrame(x_train)
  y_train = pd.DataFrame(y_train)

  TRAIN_SCORES = []
  TEST_SCORES  = [] 
  for p in params:

    training_scores = []
    crossval_scores = []
    classifier.n_neighbors = int(p)

    for i in range(folds):
      result = next(kf.split(x_train),None)
      x_training = x_train.iloc[result[0]]
      x_cv = x_train.iloc[result[1]]

      y_training = y_train.iloc[result[0]]
      y_cv = y_train.iloc[result[1]]
      
      model = classifier.fit(x_training,y_training)
      training_scores.append(model.score(x_training,y_training))
      crossval_scores.append(model.score(x_cv,y_cv))
    TRAIN_SCORES.append(np.mean(training_scores))
    TEST_SCORES.append(np.mean(crossval_scores))
  return(TRAIN_SCORES , TEST_SCORES)


def train_test_data(arr):
  x, y = separate_data(arr)

  x = np.array(x)
  y = np.array(y)
  
  (train_col, test_col, train_lab, test_lab) = train_test_split(
	x, y, test_size=0.25, random_state=42) 

  classifier = KNeighborsClassifier()
  train_score , cv_scores = RandomSearchCV(train_col, train_lab,classifier, 10, 8)

  # params = np.random.uniform(1,21,10)
  # params = np.array([int(i) for i in params])
  # params = np.sort(params)

  params = np.arange(1, 10, 1)

  plt.plot(params,train_score, label='train cruve')
  plt.plot(params,cv_scores, label='cv cruve')
  plt.xlabel("Hyperparameter k")
  plt.ylabel("Accuracy")
  plt.title('Hyper-parameter VS accuracy plot')
  plt.legend()
  plt.show()

# arr_col = build_datasample()
# arr_cor = build_datasample_asym()
# best_random_forest(arr_cor, 10)
# best_random_forest(arr_col, 10)

# best_random_forest(arr_col)

def make_figures_tables(path):
  final_training(arr_col, 10).savefig(os.path.join(path, "colour_barchart.png"))
  final_training(arr_cor, 10).savefig(os.path.join(path, "asymmetry_barchart.png"))

  make_csv_features(path)
  select_data(make_df()).to_csv(os.path.join(path, "train_test_data.csv"))

# make_figures_tables("C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023")

def random_search_training(arr):
  x, y = separate_data(arr)

  x = np.array(x)
  y = np.array(y)
  
  (train_col, test_col, train_lab, test_lab) = train_test_split(
	x, y, test_size=0.25, random_state=42) 

  # Number of trees in random forest
  n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt']
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  max_depth.append(None)
  # Minimum number of samples required to split a node
  min_samples_split = [2, 5, 10]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 2, 4]
  # Method of selecting samples for training each tree
  bootstrap = [True, False]
  # Create the random grid
  random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

  # Use the random grid to search for best hyperparameters
  # First create the base model to tune
  rf = RandomForestRegressor()
  # Random search of parameters, using 3 fold cross validation, 
  # search across 100 different combinations, and use all available cores
  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
  # Fit the random search model
  rf_random.fit(train_col, train_lab)

  print(rf_random.best_params_)


def try_training(arr):
  x, y = separate_data(arr)

  x = np.array(x)
  y = np.array(y)
  
  (train_col, test_col, train_lab, test_lab) = train_test_split(
	x, y, test_size=0.25, random_state=42) 

  model_knn = KNeighborsClassifier(n_neighbors=3)
  model_knn.fit(train_col, train_lab)
  acc = model_knn.score(test_col, test_lab)

  return acc

