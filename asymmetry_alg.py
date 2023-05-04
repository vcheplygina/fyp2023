import numpy as np

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