import numpy as np
from skimage import io, filters, morphology, segmentation, transform
import matplotlib.pyplot as plt
from skimage.draw import polygon

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
  cropped_img = image * mask[:, :]

  # doing dilation
  struct_el = morphology.disk(6)
  mask_dilated = morphology.binary_dilation(cropped_img, struct_el)

  # applying the mask and extracting the lesion from the image
  im2 = img_col.copy()
  im2[mask_dilated == 0] = 0


  # plotting results
  fig, ax = plt.subplots(1, 6, figsize=(15, 8))
  ax[0].imshow(image, cmap='gray')
  ax[1].imshow(output, cmap='gray')
  ax[2].imshow(cropped_img, cmap='gray')
  ax[3].imshow(mask_dilated, cmap='gray')
  ax[4].imshow(regions, cmap='gray')
  ax[5].imshow(im2, cmap='gray')          # resulting image
  

  # a circle and a snake for the second plot
  ax[0].plot(init[:, 1], init[:, 0], '--r', lw=3)
  ax[0].plot(snake[:, 1], snake[:, 0], '-b', lw=3)