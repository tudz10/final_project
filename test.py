from spectral_clustering import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans
from sklearn import metrics

from skimage import io
from skimage.io import imread_collection
import skimage.io as skio
from skimage.color import rgb2gray
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans

import os

from skimage.transform import resize
from skimage import data
from skimage.data import coins
from skimage.transform import rescale
from skimage import exposure
from PIL import Image
import cv2
import helpers
import argparse
from PIL import Image, ImageOps


def histogram(pic):
  # import training images
  timages = []
  cv2_images = []
  tdir = './'
  #tdir = './images/'
  #for filename in os.listdir(tdir):
  #    if filename.endswith(".jpg"):
  #    #if filename.endswith(".tif"):
  #      #timages.append(skio.imread(tdir + filename, plugin= "tifffile"))
  #      timages.append(Image.open(tdir + filename))
  timages.append(Image.open(tdir + pic))
  n_clusters = 4
  img_size =  100
  #timages[0].thumbnail((img_size,img_size))
  img = timages[0].resize((img_size,img_size))
  #img = timages[0]
  sc = SpectralClustering(n_clusters, 0)
  img_arr = np.array(img)
  sc.affinity_matrix_ = sc._get_affinity_matrix(img_arr)
  #sc.affinity_matrix_ = sc._get_affinity_matrix(img)
  embedding_features = sc._get_embedding()
  kmeans = KMeans(n_clusters=n_clusters)
  kmeans.fit(embedding_features)
  labels_ = kmeans.labels_
  labels = labels_.reshape((img_size,img_size))
  spec_cmap = plt.cm.get_cmap("Spectral")
  #f, axarr = plt.subplots(1,2)
  #axarr[0].imshow(img)
  #axarr[1].matshow(labels, cmap=spec_cmap)
  plt.matshow(labels,cmap=spec_cmap)
  plt.show()


def texture(pic):
  # import training images
  timages = []
  tdir = './'
  #tdir = './images/'
  #for filename in os.listdir(tdir):
  #    if filename.endswith("flower_berkeley.jpg"):
  #    #if filename.endswith(".tif"):
  #      #timages.append(skio.imread(tdir + filename, plugin= "tifffile"))
  timages.append(Image.open(tdir + pic))
  n_clusters = 3
  img = timages[0]
  img_size =  100


  # Save grayscale version
  g_img = ImageOps.grayscale(img)
  g_img.save("grey_" + pic,"JPEG")
  # Preprocess
  p_img = helpers.image_preprocess(img, img_size)

  # Run Program
  sc = SpectralClustering(n_clusters, 1)
  sc.affinity_matrix_ = sc._get_affinity_matrix(p_img)
  embedding_features = sc._get_embedding()
  kmeans = KMeans(n_clusters=n_clusters)
  kmeans.fit(embedding_features)
  labels_ = kmeans.labels_
  labels = labels_.reshape((img_size,img_size))

  # Plot and show 
  #f, axarr = plt.subplots(1,3)
  spec_cmap = plt.cm.get_cmap("Spectral")
  #axarr[0].imshow(img)
  #axarr[1].imshow(p_img, cmap = "Greys")
  #axarr[2].matshow(labels, cmap=spec_cmap)
  plt.matshow(labels, cmap=spec_cmap)
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('kernel', type=str, help='kernel_type')
  parser.add_argument('pic', type=str, help='picture')
  args = parser.parse_args()
  if args.kernel == "histogram":
    histogram(args.pic)
  elif args.kernel == "texture":
    texture(args.pic)
