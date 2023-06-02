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



def separate_data(arr):
  x = list()
  y = list()

  for i in arr:
    x.append(i[0])
    y.append(i[1])

  return (x, y)



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
  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
  # Fit the random search model
  rf_random.fit(train_col, train_lab)

  print(rf_random.best_params_)

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

  # joblib.dump(rndF, 'C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\random_forest_model_cor.pkl')

  score_rndF = rndF.score(test_col, test_lab)

  eval_rndF = evaluate(rndF, test_col, test_lab)

  fig, axs = plt.subplots(2, 4, figsize=(12, 5))
  axs = axs.flatten()

  # Histogram for Average Error
  axs[0].bar("Average Error", eval_rndF[0])
  axs[0].set_title(eval_rndF[0])

  axs[1].bar("Accuracy", eval_rndF[1])
  axs[1].set_title(eval_rndF[1])

  axs[2].bar("Precision", eval_rndF[2])
  axs[2].set_title(eval_rndF[2])

  axs[3].bar("F1 Score", eval_rndF[3])
  axs[3].set_title(eval_rndF[3])

  axs[4].bar("Recall", eval_rndF[4])
  axs[4].set_xlabel(eval_rndF[4])

  axs[5].bar("AUC", eval_rndF[5])
  axs[5].set_xlabel(eval_rndF[5])

  axs[6].bar("Cross-Validation", np.mean(result_rndF))
  axs[6].set_xlabel(np.mean(result_rndF))
  
  
  print(f"Cross-val score for Random Forest {result_rndF.mean()}\t Accuracy for Random Forest {score_rndF}")

  return fig, eval_rndF, np.mean(result_rndF)


def regular_random_forest(arr, n_folds):
  x, y = separate_data(arr)

  x = np.array(x)
  y = np.array(y)
  
  (train_col, test_col, train_lab, test_lab) = train_test_split(
	x, y, test_size=0.25, random_state=42) 

  kf = KFold(n_splits=n_folds)
  
  rndF = RandomForestClassifier()
  result_rndF = cross_val_score(rndF, train_col, train_lab, cv=n_folds)
  
  rndF.fit(train_col, train_lab)

  # joblib.dump(rndF, 'C:\\Users\\dubst\\Desktop\\DataScience\\Project 2\\fyp2023\\random_forest_model_cor.pkl')

  score_rndF = rndF.score(test_col, test_lab)

  eval_rndF = evaluate(rndF, test_col, test_lab)

  fig, axs = plt.subplots(2, 4, figsize=(12, 5))
  axs = axs.flatten()

  # Histogram for Average Error
  axs[0].bar("Average Error", eval_rndF[0])
  axs[0].set_title(eval_rndF[0])

  axs[1].bar("Accuracy", eval_rndF[1])
  axs[1].set_title(eval_rndF[1])

  axs[2].bar("Precision", eval_rndF[2])
  axs[2].set_title(eval_rndF[2])

  axs[3].bar("F1 Score", eval_rndF[3])
  axs[3].set_title(eval_rndF[3])

  axs[4].bar("Recall", eval_rndF[4])
  axs[4].set_xlabel(eval_rndF[4])

  axs[5].bar("AUC", eval_rndF[5])
  axs[5].set_xlabel(eval_rndF[5])

  axs[6].bar("Cross-Validation", np.mean(result_rndF))
  axs[6].set_xlabel(np.mean(result_rndF))
  
  
  print(f"Cross-val score for Random Forest {result_rndF.mean()}\t Accuracy for Random Forest {score_rndF}")

  return fig, eval_rndF, np.mean(result_rndF)






final_data = select_data(make_df())

arr_col = build_datasample_new()
arr_asym = build_datasample_asym_new()

#Training all the models on color features and outputting the eveluation of each model

#final_training(arr_col, 10)

#Training all the models on assymetry features and outputting the eveluation of each model


#final_training(arr_asym, 10)

#Doig random search for a region where the hyperparameters for the random forest model are the best fit. This takes a long time
# random_search_training(arr_col)

#Tuning the hyperparameters found in random search with grid search. 

# try_grid_search(arr_col)

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



arr_col = build_datasample()
arr_cor = build_datasample_asym()


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


def compare_models(evals1, evals2, cros1, cros2):
  fig, axs = plt.subplots(2, 4, figsize=(12, 5))
  axs = axs.flatten()

  # Histogram for Average Error
  axs[0].bar("New", evals1[0])
  axs[0].set_title("Average Error")
  axs[0].bar("Old", evals2[0])
  axs[0].set_title("Average Error")

  axs[1].bar("New", evals1[1])
  axs[1].set_title("Accuracy")
  axs[1].bar("Old", evals2[1])
  axs[1].set_title("Accuracy")

  axs[2].bar("New", evals1[2])
  axs[2].set_title("Precision")
  axs[2].bar("Old", evals2[2])
  axs[2].set_title("Precision")

  axs[3].bar("New", evals1[3])
  axs[3].set_title("F1 Score")
  axs[3].bar("Old", evals2[3])
  axs[3].set_title("F1 Score")

  axs[4].bar("New", evals1[4])
  axs[4].set_title("Recall")
  axs[4].bar("Old", evals2[4])
  axs[4].set_title("Recall")

  axs[5].bar("New", evals1[5])
  axs[5].set_title("AUC")
  axs[5].bar("Old", evals2[5])
  axs[5].set_title("AUC")

  axs[6].bar("New", cros1)
  axs[6].set_title("Cross-Validation")
  axs[6].bar("Old", cros2)
  axs[6].set_title("Cross-Validation")

  plt.show()


def make_figures_tables(path):
  final_training(arr_col, 10).savefig(os.path.join(path, "colour_barchart.png"))
  final_training(arr_cor, 10).savefig(os.path.join(path, "asymmetry_barchart.png"))

  make_csv_features(path)
  select_data(make_df()).to_csv(os.path.join(path, "train_test_data.csv"))
  fig1, evals1, cros1 = best_random_forest(arr_col, 10)
  fig1.savefig(os.path.join(path, "best_rndF_col.png"))
  fig11, evals11, cros11 = best_random_forest(arr_cor, 10)
  fig11.savefig(os.path.join(path, "best_rndF_asym.png"))

  fig2, evals2, cros2 = regular_random_forest(arr_col, 10)
  fig2.savefig(os.path.join(path, "best_rndF_col_reg.png"))
  fig22, evals22, cros22 = regular_random_forest(arr_cor, 10)
  fig22.savefig(os.path.join(path, "best_rndF_asym_reg.png"))
  
  compare_models(evals11, evals22, cros11, cros22)


make_figures_tables(os.getcwd())

#Running the best random forest for the tuned hyperparameters and evaluating it's scores
# best_random_forest(arr, 10)

