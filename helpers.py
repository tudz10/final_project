import numpy as np
from PIL import Image, ImageOps
from skimage import exposure
from skimage.filters import gabor
import cv2


# Create 5x5 pixel window for local histogram
def compute_rgb_histogram(x, i, j):
  region = np.zeros([3,5,5])
  
  a = 0
  for l in range (i-2, i+2):
    b = 0
    for m in range (j-2, j+2):
      if (l < 0) or (l > (x.shape[0]-1)) or (m < 0) or (m > (x.shape[1] -1)):
        pass
      else:
        region[0][a][b] = x[l][m][0]
        region[1][a][b] = x[l][m][1]
        region[2][a][b] = x[l][m][2]
      b += 1
    a += 1 
  red, bins = np.histogram(region[0,...].ravel(), bins=256, range=[0, 256])
  green, bins = np.histogram(region[1,...].ravel(), bins=256, range=[0, 256])
  blue, bins = np.histogram(region[2,...].ravel(), bins=256, range=[0, 256])
  hlist = []
  hlist.extend(red.ravel().astype('float32'))
  hlist.extend(green.ravel().astype('float32'))
  hlist.extend(blue.ravel().astype('float32'))
  return np.array(hlist)

def compute_gray_histogram(x, i, j):
  region = np.zeros([5,5])
  a = 0
  for l in range (i-2, i+2):
    b = 0
    for m in range (j-2, j+2):
      if (l < 0) or (l > (x.shape[0]-1)) or (m < 0) or (m > (x.shape[1] -1)):
        pass
      else:
        region[a][b] = x[l][m]
      b += 1
    a += 1 
  gray, bin_edges = np.histogram(region, bins=256, range=(0, 1))
  hlist = []

  hlist.extend(gray.ravel().astype('float32'))
  return np.array(hlist)

def chi_square_affinity(x_train, rgb = True):
  n = x_train.shape[0]
  W = np.zeros((n*n, n*n))
  local_histogram = []
  scale_param_i = 0
  scale_param_j = 0
  pixel_distance = 7
  scale = 0.25
  compare_funct = cv2.HISTCMP_CHISQR_ALT
  for i in range(n):
    for j in range(n):
      if rgb:
        local_histogram.append(compute_rgb_histogram(x_train, i, j))
      else:
        local_histogram.append(compute_gray_histogram(x_train, i, j))
  for i in range(n*n):
    for j in range(n*n):
      chi_square = scale*cv2.compareHist(local_histogram[i], local_histogram[j], compare_funct)
      try:
        scale_param_i = np.sqrt(scale*cv2.compareHist(local_histogram[i], local_histogram[i+pixel_distance], compare_funct))
      except:
        scale_param_i = np.sqrt(scale*cv2.compareHist(local_histogram[i], local_histogram[i-pixel_distance], compare_funct))
      try:
        scale_param_j = np.sqrt(scale*cv2.compareHist(local_histogram[j], local_histogram[j+pixel_distance], compare_funct))
      except:
        scale_param_j = np.sqrt(scale*cv2.compareHist(local_histogram[j], local_histogram[j-pixel_distance], compare_funct))
      constraint = scale_param_i*scale_param_j
      if constraint == 0:
        constraint = 1
      W[i][j] = np.exp(-chi_square/(constraint))
  return W

def rbf_affinity(x_train, gamma):
  n = x_train.shape[0] # num_data
  m = x_train.shape[-1] # num_features
  cross_ = x_train @ x_train.T
  cross_diag = np.diag(cross_)
  all_one_v = np.ones([n])
  square_mat = np.kron(all_one_v, cross_diag).reshape([n, n])
  square_mat += np.kron(cross_diag, all_one_v).reshape([n, n])
  square_mat -= 2 * cross_
  return np.exp(-gamma * square_mat)

def geometric(h, gamma):
  row_num = h.shape[0]
  col_num = h.shape[1]
  result = np.zeros((h.shape[0], h.shape[0]))
  for i in range(row_num):
    for j in range (row_num):
      ins = 0
      for k in range(col_num):
        top  = h[i,k] - h[j, k]
        bot = h[i,k] + h[j, k]
        if bot != 0:
          ins += (top*top) / bot
    result[i, j] = -ins 
  result = result*gamma   
  return np.exp(result, result)

def image_preprocess(img, img_size):
  img = img.resize((img_size,img_size))
  g_img = np.array(ImageOps.grayscale(img)) 
  g_img = exposure.equalize_adapthist(g_img, clip_limit=0.03)
  return g_img
